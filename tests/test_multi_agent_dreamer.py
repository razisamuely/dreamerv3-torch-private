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
import yaml
import pathlib
import types
import sys
import os
from tests.utils import load_config

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
        self.train_dataset = self._create_mock_dataset()

        # Set up environments - make env is a function that creates the environment
        make = lambda mode, id: make_env(self.configs, mode, id)
        train_envs = [make("train", i) for i in range(self.configs.envs)]

        # Set up action and observation spaces
        acts = train_envs[0].action_space
        self.configs.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        
        # Set up observation shape
        self.agent = Dreamer(
            obs_space = OBSERVATION_SPACE, 
            act_space = ACTION_SPACE,
            config = self.configs,
            logger = self.logger, 
            dataset = self.train_dataset
        )

    def _create_mock_dataset(self):
        """Create a mock dataset that yields fake batches indefinitely."""
        def generator():
            while True:
                # Create batch with proper dimensions
                batch = {
                    'agent0_obs': torch.zeros((self.configs.batch_size, self.configs.batch_length, 18)),
                    'agent1_obs': torch.zeros((self.configs.batch_size, self.configs.batch_length, 18)),
                    'agent2_obs': torch.zeros((self.configs.batch_size, self.configs.batch_length, 18)),
                    'agent3_obs': torch.zeros((self.configs.batch_size, self.configs.batch_length, 18)),
                    'image': torch.zeros((self.configs.batch_size, self.configs.batch_length, 64, 64, 3)),
                    'action': torch.zeros((self.configs.batch_size, self.configs.batch_length, self.configs.num_actions * self.configs.n_agents)),
                    'reward': torch.zeros((self.configs.batch_size, self.configs.batch_length)),
                    'discount': torch.ones((self.configs.batch_size, self.configs.batch_length)),
                    'is_first': torch.zeros((self.configs.batch_size, self.configs.batch_length), dtype=torch.bool),
                    'is_terminal': torch.zeros((self.configs.batch_size, self.configs.batch_length), dtype=torch.bool)
                }
                yield batch
        
        return generator()
    


    # def test_call(self):
    #     """Test if the __call__ method works correctly with multiple agents."""
        
    #     # Create a small batch of observations
    #     batch_size = 2
    #     obs = {
    #         'image': torch.zeros((batch_size, 64, 64, 3), dtype=torch.float32),
    #         'agent0_obs': torch.zeros((batch_size, 18), dtype=torch.float32),
    #         'agent1_obs': torch.zeros((batch_size, 18), dtype=torch.float32),
    #         'agent2_obs': torch.zeros((batch_size, 18), dtype=torch.float32),
    #         'agent3_obs': torch.zeros((batch_size, 18), dtype=torch.float32),
    #         'is_first': torch.ones((batch_size,), dtype=torch.bool),
    #         'is_last': torch.zeros((batch_size,), dtype=torch.bool),
    #         'is_terminal': torch.zeros((batch_size,), dtype=torch.bool)
    #     }
        
    #     # Create reset tensor
    #     reset = torch.zeros(batch_size, dtype=torch.bool)
        
    #     # Call the agent
    #     policy_output, state = self.agent(obs, reset, None, training=False)
        
    #     # Check that policy output contains the expected keys
    #     self.assertIn('action', policy_output, "Policy output should contain 'action'")
    #     self.assertIn('logprob', policy_output, "Policy output should contain 'logprob'")
    #     self.assertIn('agent_actions', policy_output, "Policy output should contain 'agent_actions'")
        
    #     # Check if agent_actions contains the right number of elements (one per agent)
    #     self.assertEqual(len(policy_output['agent_actions']), self.configs.n_agents,
    #                     f"agent_actions should contain {self.configs.n_agents} elements")
        
    #     # Check if combined action has the right shape
    #     expected_action_shape = (batch_size, self.configs.num_actions * self.configs.n_agents)
    #     self.assertEqual(policy_output['action'].shape, expected_action_shape,
    #                     f"Combined action shape should be {expected_action_shape}")
        
    #     # Make a second call with the state from the first call to check state handling
    #     policy_output2, state2 = self.agent(obs, reset, state, training=False)
        
    #     # State should not be None
    #     self.assertIsNotNone(state2, "State should not be None after second call")
        
    #     # Make sure the policy output from the second call has the same structure
    #     self.assertEqual(len(policy_output2['agent_actions']), self.configs.n_agents,
    #                     "Second call should return the same number of agent actions")

    def test_call_train_true(self):
        batch_size = 2
        obs = {
            'image': torch.zeros((batch_size, 64, 64, 3), dtype=torch.float32),
            'agent0_obs': torch.zeros((batch_size, 18), dtype=torch.float32),
            'agent1_obs': torch.zeros((batch_size, 18), dtype=torch.float32),
            'agent2_obs': torch.zeros((batch_size, 18), dtype=torch.float32),
            'agent3_obs': torch.zeros((batch_size, 18), dtype=torch.float32),
            'is_first': torch.ones((batch_size,), dtype=torch.bool),
            'is_last': torch.zeros((batch_size,), dtype=torch.bool),
            'is_terminal': torch.zeros((batch_size,), dtype=torch.bool)
        }
        
        # Create reset tensor
        reset = torch.zeros(batch_size, dtype=torch.bool)
        
        self.agent._should_train = lambda x: 1 
        
        # Call the agent
        policy_output, state = self.agent(obs, reset, None, training=True)

        # TODO : Fix training with multiple agents, looks like mismatch in dimension of world model 



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