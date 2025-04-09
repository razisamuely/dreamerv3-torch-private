import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools

import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()

def verify_model_dimensions(agent, checkpoint, config):
    """
    Verifies that the dimensions of the loaded model match the current configuration.
    
    Args:
        agent: The agent object to load the checkpoint into
        checkpoint: The loaded checkpoint dictionary
        config: The current configuration object
    
    Returns:
        bool: True if dimensions match, False otherwise
    """
    try:
        # Get the action space dimension from the agent's dynamics model
        if hasattr(agent._wm.dynamics, '_num_actions'):
            checkpoint_action_dim = agent._wm.dynamics._num_actions
            current_action_dim = config.num_actions
            
            # Check if the dimensions match
            if checkpoint_action_dim != current_action_dim:
                print(f"WARNING: Action dimension mismatch!")
                print(f"Checkpoint action dimension: {checkpoint_action_dim}")
                print(f"Current config action dimension: {current_action_dim}")
                return False
        
        # Check the first layer of the img_in_layers which connects to the action space
        # Get the weight shape of the first linear layer
        if hasattr(agent._wm.dynamics, '_img_in_layers'):
            for module in agent._wm.dynamics._img_in_layers:
                if isinstance(module, torch.nn.Linear):
                    checkpoint_weight = checkpoint["agent_state_dict"]["_wm.dynamics._img_in_layers.0.weight"]
                    current_weight_shape = module.weight.shape
                    
                    if checkpoint_weight.shape != current_weight_shape:
                        print(f"WARNING: Linear layer dimension mismatch!")
                        print(f"Checkpoint weight shape: {checkpoint_weight.shape}")
                        print(f"Current model weight shape: {current_weight_shape}")
                        return False
                    break
        
        return True
    except Exception as e:
        print(f"Error during model dimension verification: {e}")
        return False
    

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        
        # Init the world model
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)

        # Init the task behavior
        self._task_behaviors = [models.ImagBehavior(config, self._wm) for _ in range(config.n_agents)]

        # Model compilation
        if (config.compile and os.name != "nt"):
            self._wm = torch.compile(self._wm)
            for i in range(config.n_agents):
                self._task_behaviors[i]._world_model = self._wm
                self._task_behaviors[i] = torch.compile(self._task_behaviors[i])

        # Reward head
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()

        # Task behavior, 1) greedy, 2) random, 3) plan2explore
        self._expl_behaviors = []
        for _ in range(config.n_agents):
            if config.expl_behavior == "greedy":
                expl_behavior = self._task_behaviors[_]
            elif config.expl_behavior == "random":
                expl_behavior = expl.Random(config, act_space)
            elif config.expl_behavior == "plan2explore":
                expl_behavior = expl.Plan2Explore(config, self._wm, reward)
            else:
                raise NotImplementedError(config.expl_behavior)
            
            self._expl_behaviors.append(expl_behavior.to(self._config.device))

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            # Pretrain or normal training steps
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
                
            # Log metrics if it's time
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        # Get policy outputs for all agents
        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
            
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        
        # Preprocess observations and encode them
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        
        # Update the latent state based on new observations
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        
        # Get features from latent state
        feat = self._wm.dynamics.get_feat(latent)
        
        # Initialize container for all agents' actions and logprobs
        actions = []
        logprobs = []
        
        # Get actions for each agent
        for agent_idx in range(self._config.n_agents):
            if not training:
                # Use task behavior (pure exploitation) during evaluation
                actor = self._task_behaviors[agent_idx].actor(feat)
                agent_action = actor.mode()
            elif self._should_expl(self._step):
                # Use exploration behavior during exploration phase
                actor = self._expl_behaviors[agent_idx].actor(feat)
                agent_action = actor.sample()
            else:
                # Use task behavior but with sampling during normal training
                actor = self._task_behaviors[agent_idx].actor(feat)
                agent_action = actor.sample()
            
            # Get log probability of the action
            agent_logprob = actor.log_prob(agent_action)
            
            # Store agent's action and logprob
            actions.append(agent_action.detach())
            logprobs.append(agent_logprob)
        
        # Combine actions from all agents
        if self._config.n_agents > 1:
            # Combine actions based on your multi-agent environment's needs
            # This could be concatenation, stacking, or a custom function
            combined_action = torch.cat(actions, dim=-1)  # Assuming actions are concatenated in action space
            combined_logprob = torch.stack(logprobs, dim=-1)
        else:
            combined_action = actions[0]
            combined_logprob = logprobs[0]
        
        # Handle one-hot conversion if needed
        if self._config.actor["dist"] == "onehot_gumble":
            # You might need to adapt this for multi-agent case
            combined_action = torch.one_hot(
                torch.argmax(combined_action, dim=-1), self._config.num_actions
            )
        
        # Detach latent for next step
        latent = {k: v.detach() for k, v in latent.items()}
        
        policy_output = {
            "action": combined_action, 
            "logprob": combined_logprob,
            "agent_actions": actions  # Keep individual actions for debugging or custom environment needs
        }
        
        state = (latent, combined_action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        
        # Train world model (shared among all agents)
        post, context, wm_metrics = self._wm._train(data)
        metrics.update(wm_metrics)
        
        # Start state for imagination
        start = post
        
        # Reward function lambda used by all agents
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        
        # Train each agent's actor-critic separately
        all_policy_actors = [behavior.actor for behavior in self._task_behaviors]
 
        for agent_idx, behavior in enumerate(self._task_behaviors):
            # Train current agent's policy using the shared world model
            _, _, _, _, agent_metrics = behavior._train(start, reward, all_policy_actors)
            
            # Add agent identifier to metrics
            agent_prefixed_metrics = {f"agent{agent_idx}_{key}": value 
                                    for key, value in agent_metrics.items()}
            metrics.update(agent_prefixed_metrics)
        
        # Train exploration behavior if using non-greedy exploration
        if self._config.expl_behavior != "greedy":
            for agent_idx, expl_behavior in enumerate(self._expl_behaviors):
                # Skip if the exploration behavior is the same as task behavior (greedy case)
                if expl_behavior is self._task_behaviors[agent_idx]:
                    continue
                    
                # Train exploration behavior
                mets = expl_behavior.train(start, context, data)[-1]
                metrics.update({f"agent{agent_idx}_expl_{key}": value 
                            for key, value in mets.items()})
        
        # Update metrics tracking
        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
                
        return metrics


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "vmas":
        if "spread" in task:
            from envs.vmas_simple_spread import VmasSpread
            n_agents = getattr(config, 'n_agents', 2)
            env = VmasSpread(
                task, config.action_repeat, config.size, seed=config.seed + id, device=config.device,
                n_agents=n_agents
            )
            print(env.action_space)
        elif "navigation" in task:
            from envs.vmas_navigation import VmasNavigationEnv
            n_agents = getattr(config, 'n_agents', 1)
            env = VmasNavigationEnv(
                task, config.action_repeat, config.size, seed=config.seed + id, device=config.device,
                n_agents=n_agents
            )
        else:
            import envs.vmas_simple as vmas
            # Use getattr instead of get for Namespace objects
            n_agents = getattr(config, 'n_agents', 1)
            env = vmas.Vmas(
                task, config.action_repeat, config.size, seed=config.seed + id, device=config.device,
                n_agents=n_agents
            )
        env = wrappers.NormalizeActions(env)

    else:
        raise NotImplementedError(suite)
    
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
        dimensions_match = verify_model_dimensions(agent, checkpoint, config)
        pass
    

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps, # This pointer is get updated
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))


# python dreamer.py --configs vmas --task vmas_simple --logdir ./logdir/vmas_simple
# python dreamer.py --configs vmas --task vmas_simple_spread --logdir ./logdir/vmas_simple_spread
# python dreamer.py --configs vmas --task vmas_simple_spread --logdir ./logdir/vmas_simple_spread
# python dreamer.py --configs vmas --task vmas_navigation --logdir ./logdir/vmas_navigation