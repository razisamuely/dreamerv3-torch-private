import unittest
import gym.spaces
import torch
import numpy as np
from pathlib import Path
import sys
import yaml
import pathlib
import gym 
from tools import Logger, load_episodes, args_type
sys.path.append(str(Path(__file__).parent.parent))
from dreamer import Dreamer, count_steps, make_dataset,make_env
import argparse


# train_envs[0].action_space Box(-1.0, 1.0, (8,), float32)
ACTION_SPACE = gym.spaces.Box(low = -1,high = 1, shape = (8,), dtype=np.float32)
OBSERVATION_SPACE = gym.spaces.Dict({
    'agent0_obs': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32), # Agent i
    'agent1_obs': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32), # Agent i
    'agent2_obs': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32), # Agent i
    'agent3_obs': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32), # Agent i
    'image': gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8), # Image observation
    'is_first': gym.spaces.Discrete(2), # Boolean indicating if it's the first step
    'is_last': gym.spaces.Discrete(2), # Boolean indicating if it's the last step
    'is_terminal': gym.spaces.Discrete(2) # Boolean indicating if it's a terminal state
})


import yaml
import pathlib
import types
import sys
import os

def load_config():
    # Load YAML file
    config_path = pathlib.Path(sys.argv[0]).parent.parent / "configs.yaml"
    config_data = yaml.safe_load(config_path.read_text())
    
    # Get defaults section
    defaults = config_data["defaults"]
    
    # Helper function to convert values and handle nested dictionaries
    def process_value(value):
        if isinstance(value, dict):
            # Keep as dictionary for compatibility with ** unpacking
            return {k: process_value(v) for k, v in value.items()}
        elif isinstance(value, str):
            # Try to convert scientific notation strings to float
            if 'e' in value.lower() or 'e+' in value.lower() or 'e-' in value.lower():
                try:
                    return float(value)
                except ValueError:
                    pass
        return value
    
    # Create config object with proper types
    config = types.SimpleNamespace()
    for key, value in defaults.items():
        setattr(config, key, process_value(value))
    
    # Add additional settings
    config.logdir = str(pathlib.Path(__file__).parent.parent / "logs")
    config.traindir = config.traindir or pathlib.Path(config.logdir) / "train_eps"
    
    # Set multi-agent configuration
    config.n_agents = 4  # Set number of agents for testing
    
    return config

class TestMultiAgentDreamer(unittest.TestCase):

    def setUp(self):
        self.configs = load_config()
        self.configs.logdir =  str(pathlib.Path(__file__).parent.parent / "logs")

        logdir = pathlib.Path(self.configs.logdir).expanduser()
        # Parse arguments (if any) and override defaults
        self.configs.traindir = self.configs.traindir or logdir / "train_eps"
                

        # Set up Logger
        step = count_steps(self.configs.traindir)
        self.logger =  Logger(logdir, self.configs.action_repeat * step)

        # Set up dataset
        directory = self.configs.traindir
        train_eps =  load_episodes(directory, limit=self.configs.dataset_size)
        self.train_dataset = make_dataset(train_eps, self.configs)

        make = lambda mode, id: make_env(self.configs, mode, id)
        train_envs = [make("train", i) for i in range(self.configs.envs)]
        acts = train_envs[0].action_space
        self.configs.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        
        


        self.agent = Dreamer(
            obs_space = OBSERVATION_SPACE, 
            act_space = ACTION_SPACE,
            config = self.configs,
            logger = self.logger, 
            dataset = self.train_dataset
        )

    def test_first(self):
        pass 
    # def test_init(self):
    #     """Test if initialization creates the correct number of actors"""
    #     self.assertEqual(len(self.agent._task_behaviors), self.config.n_agents, 
    #                     "Should create n_agents task behaviors")
    #     self.assertEqual(len(self.agent._expl_behaviors), self.config.n_agents, 
    #                     "Should create n_agents exploration behaviors")
        
    #     # Test if world model is shared
    #     for behavior in self.agent._task_behaviors:
    #         self.assertIs(behavior._world_model, self.agent._wm, 
    #                      "World model should be shared between all behaviors")

    # def test_policy(self):
    #     """Test if policy method returns actions for all agents"""
    #     batch_size = 2
    #     obs = {
    #         'image': torch.zeros((batch_size, *self.obs_shape)),
    #         'is_first': torch.zeros((batch_size,), dtype=torch.bool)
    #     }
        
    #     # Call policy method
    #     policy_output, state = self.agent._policy(obs, None, training=True)
        
    #     # Check output shape and content
    #     self.assertIn('action', policy_output, "Policy output should contain 'action'")
    #     self.assertIn('logprob', policy_output, "Policy output should contain 'logprob'")
    #     self.assertIn('agent_actions', policy_output, "Policy output should contain 'agent_actions'")
        
    #     # Check if agent_actions contains n_agents elements
    #     self.assertEqual(len(policy_output['agent_actions']), self.config.n_agents,
    #                     "agent_actions should contain actions for all agents")
        
    #     # Check if combined action has the right shape
    #     expected_action_shape = (batch_size, self.act_dim * self.config.n_agents)
    #     self.assertEqual(policy_output['action'].shape, expected_action_shape,
    #                     f"Combined action shape should be {expected_action_shape}")

    # def test_train_method(self):
    #     """Test if train method trains all agents and the world model"""
    #     batch = next(self.dataset)
    #     metrics = self.agent._train(batch)
        
    #     # Check if metrics contain entries for all agents
    #     for agent_idx in range(self.config.n_agents):
    #         agent_prefix = f"agent{agent_idx}_"
    #         # Find at least one metric for each agent
    #         agent_metrics = [k for k in metrics.keys() if k.startswith(agent_prefix)]
    #         self.assertGreater(len(agent_metrics), 0, 
    #                           f"No metrics found for agent {agent_idx}")

    # def test_call_method(self):
    #     """Test if __call__ method works correctly with multiple agents"""
    #     batch_size = 2
    #     obs = {
    #         'image': torch.zeros((batch_size, *self.obs_shape)),
    #         'is_first': torch.zeros((batch_size,), dtype=torch.bool)
    #     }
    #     reset = torch.zeros(batch_size, dtype=torch.bool)
        
    #     # Call the agent
    #     policy_output, state = self.agent(obs, reset, None, training=True)
        
    #     # Check that policy output contains actions for all agents
    #     self.assertIn('agent_actions', policy_output, 
    #                  "Policy output should contain individual agent actions")
    #     self.assertEqual(len(policy_output['agent_actions']), self.config.n_agents,
    #                     "Should return actions for all agents")

    # def test_integrated_workflow(self):
    #     """Test a full step of the agent to ensure components work together"""
    #     # Setup initial state
    #     batch_size = 2
    #     obs = {
    #         'image': torch.zeros((batch_size, *self.obs_shape)),
    #         'is_first': torch.zeros((batch_size,), dtype=torch.bool)
    #     }
    #     reset = torch.zeros(batch_size, dtype=torch.bool)
        
    #     # First call to get policy and state
    #     policy_output, state = self.agent(obs, reset, None, training=True)
        
    #     # Check policy output contains expected keys
    #     self.assertIn('action', policy_output)
    #     self.assertIn('logprob', policy_output)
    #     self.assertIn('agent_actions', policy_output)
        
    #     # Second call with the state from the first call
    #     policy_output2, state2 = self.agent(obs, reset, state, training=True)
        
    #     # State should be different from None
    #     self.assertIsNotNone(state2)


if __name__ == "__main__":
        unittest.main()