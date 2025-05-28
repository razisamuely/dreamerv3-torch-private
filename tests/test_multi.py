import unittest
import gym.spaces
import torch
import numpy as np
from pathlib import Path
import sys
import pathlib

sys.path.append(str(Path(__file__).parent.parent))
from dreamer import Dreamer, count_steps, make_env
from tests.utils import load_config
from parallel import Damy
from tools import Logger, OneHotDist
from torch import distributions as torchd

NUM_AGENTS = 4
AGENT_OBS_SIZE = 18
IMAGE_SHAPE = (64, 64, 3)
ACTION_SPACE_SIZE = 8
BATCH_SIZE = 2
SINGLE_BATCH = 1

AGENT_OBS_KEYS = [f'agent{i}_obs' for i in range(NUM_AGENTS)]
BOOLEAN_KEYS = ['is_first', 'is_last', 'is_terminal']

ACTION_SPACE = gym.spaces.Box(low=-1, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.float32)
OBSERVATION_SPACE = gym.spaces.Dict({
    **{key: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(AGENT_OBS_SIZE,), dtype=np.float32) 
       for key in AGENT_OBS_KEYS},
    'image': gym.spaces.Box(low=0, high=255, shape=IMAGE_SHAPE, dtype=np.uint8),
    **{key: gym.spaces.Discrete(2) for key in BOOLEAN_KEYS}
})


class TestMultiAgentDreamerComplete(unittest.TestCase):

    def setUp(self):
        self.config = load_config()
        self.config.logdir = str(pathlib.Path(__file__).parent.parent / "logs")
        
        logdir = pathlib.Path(self.config.logdir).expanduser()
        self.config.traindir = self.config.traindir or logdir / "train_eps"
        
        step = count_steps(self.config.traindir)
        logger = Logger(logdir, self.config.action_repeat * step)
        
        self.train_dataset = self._create_mock_dataset()
        
        make_env_fn = lambda mode, id: make_env(self.config, mode, id)
        self.train_envs = [Damy(make_env_fn("train", i)) for i in range(self.config.envs)]
        
        acts = self.train_envs[0].action_space[0]
        self.config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        
        self.agent = Dreamer(
            obs_space=OBSERVATION_SPACE,
            act_space=ACTION_SPACE,
            config=self.config,
            logger=logger,
            dataset=self.train_dataset
        )

    def _create_mock_dataset(self):
        def generator():
            while True:
                batch = {
                    **{key: torch.zeros((self.config.batch_size, self.config.batch_length, AGENT_OBS_SIZE)) 
                       for key in AGENT_OBS_KEYS},
                    'image': torch.zeros((self.config.batch_size, self.config.batch_length, *IMAGE_SHAPE)),
                    'action': torch.zeros((self.config.batch_size, self.config.batch_length, 
                                         self.config.num_actions * self.config.n_agents)),
                    'reward': torch.zeros((self.config.batch_size, self.config.batch_length)),
                    'discount': torch.ones((self.config.batch_size, self.config.batch_length)),
                    'is_first': torch.zeros((self.config.batch_size, self.config.batch_length), dtype=torch.bool),
                    'is_terminal': torch.zeros((self.config.batch_size, self.config.batch_length), dtype=torch.bool)
                }
                yield batch
        return generator()

    def _create_test_observation(self):
        return {
            'image': torch.zeros((BATCH_SIZE, *IMAGE_SHAPE), dtype=torch.float32),
            **{key: torch.zeros((BATCH_SIZE, AGENT_OBS_SIZE), dtype=torch.float32) 
               for key in AGENT_OBS_KEYS},
            'is_first': torch.ones((BATCH_SIZE,), dtype=torch.bool),
            'is_last': torch.zeros((BATCH_SIZE,), dtype=torch.bool),
            'is_terminal': torch.zeros((BATCH_SIZE,), dtype=torch.bool)
        }

    def _create_single_observation(self):
        return {
            'image': torch.zeros((SINGLE_BATCH, *IMAGE_SHAPE), dtype=torch.float32),
            **{key: torch.zeros((SINGLE_BATCH, AGENT_OBS_SIZE), dtype=torch.float32) 
               for key in AGENT_OBS_KEYS},
            'is_first': torch.ones((SINGLE_BATCH,), dtype=torch.bool),
            'is_last': torch.zeros((SINGLE_BATCH,), dtype=torch.bool),
            'is_terminal': torch.zeros((SINGLE_BATCH,), dtype=torch.bool)
        }

    def _create_distinct_agent_observations(self):
        return {
            **{f'agent{i}_obs': torch.ones((SINGLE_BATCH, AGENT_OBS_SIZE), device=self.config.device) * (i + 1)
               for i in range(NUM_AGENTS)},
            'image': torch.ones((SINGLE_BATCH, *IMAGE_SHAPE), device=self.config.device) * 0.5,
            'is_first': torch.ones((SINGLE_BATCH,), dtype=torch.bool, device=self.config.device),
            'is_last': torch.zeros((SINGLE_BATCH,), dtype=torch.bool, device=self.config.device),
            'is_terminal': torch.zeros((SINGLE_BATCH,), dtype=torch.bool, device=self.config.device)
        }

    def _create_random_agent_distributions(self, acts):
        if hasattr(acts, "discrete"):
            return [OneHotDist(torch.zeros(BATCH_SIZE, acts.n)) for _ in range(NUM_AGENTS)]
        else:
            return [torchd.independent.Independent(
                       torchd.uniform.Uniform(
                           torch.tensor(acts.low).repeat(BATCH_SIZE, 1),
                           torch.tensor(acts.high).repeat(BATCH_SIZE, 1),
                       ), 1) for _ in range(NUM_AGENTS)]

    def _random_agent_policy(self, obs, reset, state):
        acts = self.train_envs[0].action_space[0]
        action_dim = acts.n if hasattr(acts, "n") else acts.shape[0]
        
        distributions = self._create_random_agent_distributions(acts)
        
        actions = []
        logprobs = []
        
        for dist in distributions:
            action = dist.sample()
            logprob = dist.log_prob(action)
            actions.append(action)
            logprobs.append(logprob)
        
        combined_action = torch.cat(actions, dim=1)
        combined_logprob = torch.stack(logprobs, dim=1)
        
        return {"action": combined_action, "logprob": combined_logprob}, None

    def test_agent_output_format_matches_random_agent(self):
        obs = self._create_test_observation()
        reset = torch.zeros(BATCH_SIZE, dtype=torch.bool)
        
        agent_output, _ = self.agent(obs, reset, None, training=False)
        random_output, _ = self._random_agent_policy(obs, reset, None)
        
        self.assertEqual(agent_output["logprob"].shape, random_output["logprob"].shape)
        self.assertEqual(agent_output["action"].shape, random_output["action"].shape)
        
        self.assertIn('action', agent_output)
        self.assertIn('logprob', agent_output)
        
        expected_action_shape = (BATCH_SIZE, self.config.num_actions * NUM_AGENTS)
        self.assertEqual(agent_output['action'].shape, expected_action_shape)

    def test_initialization_creates_correct_agent_count(self):
        self.assertEqual(len(self.agent._task_behaviors), NUM_AGENTS)
        self.assertEqual(len(self.agent._expl_behaviors), NUM_AGENTS)
        
        for behavior in self.agent._task_behaviors:
            self.assertIs(behavior._world_model, self.agent._wm)

    def test_training_updates_all_agent_parameters(self):
        initial_params = []
        for agent_idx in range(NUM_AGENTS):
            for param in self.agent._task_behaviors[agent_idx].actor.parameters():
                initial_params.append(param.clone().detach())
                break
        
        for _ in range(3):
            batch = next(self.train_dataset)
            self.agent._train(batch)
        
        for agent_idx in range(NUM_AGENTS):
            current_param = None
            for param in self.agent._task_behaviors[agent_idx].actor.parameters():
                current_param = param
                break
            
            param_changed = not torch.allclose(initial_params[agent_idx], current_param)
            self.assertTrue(param_changed, f"Agent {agent_idx} parameters should update during training")

    def test_state_persistence_between_calls(self):
        obs = self._create_test_observation()
        
        _, state1 = self.agent._policy(obs, None, training=False)
        _, state2 = self.agent._policy(obs, state1, training=False)
        
        self.assertIsNot(state1, state2)
        self.assertEqual(type(state1), type(state2))
        
        latent1, action1 = state1
        latent2, action2 = state2
        
        for key in latent1:
            self.assertIn(key, latent2)
            self.assertEqual(latent1[key].shape, latent2[key].shape)

    def test_policy_output_structure(self):
        obs = self._create_single_observation()
        policy_output, _ = self.agent._policy(obs, None, training=True)
        
        self.assertIn('action', policy_output)
        self.assertIn('logprob', policy_output)
        
        if self.config.n_agents > 1:
            expected_action_dim = self.config.num_actions * self.config.n_agents
            self.assertEqual(policy_output['action'].shape[-1], expected_action_dim)

    def test_multi_agent_observation_encoding_preserves_differences(self):
        obs = self._create_distinct_agent_observations()
        
        preprocessed = self.agent._wm.preprocess(obs)
        embed = self.agent._wm.encoder(preprocessed)
        
        self.assertFalse(torch.allclose(embed, torch.zeros_like(embed)))

    def test_device_placement_consistency(self):
        target_device = self.config.device
        
        wm_params = next(self.agent._wm.parameters())
        self.assertEqual(str(wm_params.device), target_device)
        
        for i, behavior in enumerate(self.agent._task_behaviors):
            behavior_params = next(behavior.parameters())
            self.assertEqual(str(behavior_params.device), target_device)
        
        for i, behavior in enumerate(self.agent._expl_behaviors):
            behavior_params = next(behavior.parameters())
            self.assertEqual(str(behavior_params.device), target_device)

    def test_training_updates_counters(self):
        initial_count = self.agent._update_count
        
        obs = self._create_single_observation()
        reset = torch.ones((SINGLE_BATCH,), dtype=torch.bool)
        
        self.agent._should_train = lambda x: True
        self.agent._should_pretrain = lambda: False
        
        self.agent(obs, reset, training=True)
        
        self.assertGreater(self.agent._update_count, initial_count)

    def test_training_produces_agent_metrics(self):
        batch = next(self.train_dataset)
        metrics = self.agent._train(batch)
        
        for i in range(NUM_AGENTS):
            agent_metrics = [k for k in metrics.keys() if k.startswith(f"agent{i}_")]
            self.assertGreater(len(agent_metrics), 0)

    def test_gradient_flow_to_all_agents(self):
        grad_collected = {i: False for i in range(NUM_AGENTS)}
        hooks = []
        
        for agent_idx, behavior in enumerate(self.agent._task_behaviors):
            def create_hook(idx):
                def hook(grad):
                    if grad is not None and torch.sum(torch.abs(grad)) > 0:
                        grad_collected[idx] = True
                return hook
            
            first_param = next(behavior.actor.parameters())
            hook = first_param.register_hook(create_hook(agent_idx))
            hooks.append(hook)
        
        batch = next(self.train_dataset)
        self.agent._train(batch)
        
        for hook in hooks:
            hook.remove()
        
        for agent_idx in range(NUM_AGENTS):
            self.assertTrue(grad_collected[agent_idx], f"Agent {agent_idx} should receive gradients")


if __name__ == "__main__":
    unittest.main()