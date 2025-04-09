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
from parallel import Parallel, Damy
import cv2
import time 

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
        self.train_envs = [Damy(make("train", i)) for i in range(self.configs.envs)]

        # Set up action and observation spaces
        acts = self.train_envs[0].action_space[0]
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

    # def test_call_train_true(self):
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
        
    #     self.agent._should_train = lambda x: 1 
        
    #     # Call the agent
    #     policy_output, state = self.agent(obs, reset, None, training=True)


    # def test_training_update_count(self):
    #     """Test if training updates counters correctly"""
    #     # Track initial update count
    #     initial_update_count = self.agent._update_count
        
    #     # Create a mock observation
    #     obs = {
    #         'agent0_obs': torch.zeros((1, 18)),
    #         'agent1_obs': torch.zeros((1, 18)),
    #         'agent2_obs': torch.zeros((1, 18)),
    #         'agent3_obs': torch.zeros((1, 18)),
    #         'image': torch.zeros((1, 64, 64, 3)),
    #         'is_first': torch.ones((1,), dtype=torch.bool),
    #         'is_last': torch.zeros((1,), dtype=torch.bool),
    #         'is_terminal': torch.zeros((1,), dtype=torch.bool)
    #     }
    #     reset = torch.ones((1,), dtype=torch.bool)
        
    #     # Ensure training will happen
    #     self.agent._should_train = lambda x: True
    #     self.agent._should_pretrain = lambda: False
        
    #     # Call the agent (this should trigger training)
    #     self.agent(obs, reset, training=True)
        
    #     # Check that update count increased
    #     self.assertGreater(self.agent._update_count, initial_update_count,
    #                     "Update count should increment after training")
        

    # def test_train_method(self):
    #     """Test if _train method processes data for all agents"""
    #     # Create a sample batch
    #     batch = next(self.train_dataset)
        
    #     # Run training
    #     metrics = self.agent._train(batch)
        
    #     # Check agent-specific metrics
    #     for i in range(self.configs.n_agents):
    #         # Check if metrics contain agent-specific entries
    #         agent_metrics = [k for k in metrics.keys() if k.startswith(f"agent{i}_")]
    #         self.assertGreater(len(agent_metrics), 0,
    #                         f"Should have metrics for agent{i}")
    # def test_rendering(self):
    #     self.agent.requires_grad_(requires_grad=False)
    #     self.agent._wm.requires_grad_(requires_grad=False)

    #     print("\nStarting visualization with trained model...")
    #     window_title = f"DreamerV3 - {self.configs.task}"
    #     episodes = 0
    #     max_episodes = 5
    #     agent_state = None

    #     while episodes < max_episodes:
    #         print(f"\nStarting episode {episodes+1}")
    #         env = self.train_envs[0]
    #         obs = env.reset()()
    #         done = False
    #         episode_reward = 0
    #         step = 0
            
    #         # Add a resize factor parameter to control window size
    #         resize_factor = 6.0  # Increase this to make the window larger

    #         # In the rendering loop, modify the display section:
    #         # Use fixed display size instead of proportional scaling
    #         display_width = 500  # Adjust to your preferred width
    #         display_height = 300  # Adjust to your preferred height

    #         # In the rendering loop:
    #         while not done:
    #             # Get action from the model
    #             with torch.no_grad():
    #                 # Convert observation to dictionary with batch dimension
    #                 obs_dict = {k: np.array([v]) for k, v in obs.items()}
    #                 # Add is_first and is_terminal if they don't exist (needed by the model)
    #                 if "is_first" not in obs_dict:
    #                     obs_dict["is_first"] = np.array([False])
    #                 if "is_terminal" not in obs_dict:
    #                     obs_dict["is_terminal"] = np.array([False])
                    
    #                 policy_output, agent_state = self.agent(
    #                     obs_dict,
    #                     np.array([done]),
    #                     agent_state,
    #                     training=False
    #                 )
                    
    #                 # Extract the action (remove batch dimension)
    #                 action = policy_output["action"][0].cpu().numpy()
    #                 action_dict = {"action": action}
                
    #             # Render the frame
    #             frame = env.render(mode='rgb_array',get_orignal_frame = True)  # Use the original frame
    #             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
    #             # save frame as png 
    #             cv2.imwrite(f"frame.png", frame)
    #             # Use fixed size for display
    #             display_frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_CUBIC)
                
    #             # Add info text with appropriate font size and position
    #             font_scale = 0.8
    #             text_thickness = 2
    #             text_color = (255, 255, 255)  # White
                
    #             # Add black outline around text for better readability
    #             cv2.putText(display_frame, f"Step: {step} | Reward: {episode_reward:.2f}", 
    #                         (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 
    #                         text_thickness + 2)
                
    #             # Add white text over the outline
    #             cv2.putText(display_frame, f"Step: {step} | Reward: {episode_reward:.2f}", 
    #                         (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 
    #                         text_thickness)
                
    #             # Display the frame
    #             cv2.imshow(window_title, display_frame)
    #             key = cv2.waitKey(20)  # Adjust for speed (higher = slower)
                
    #             if key == 27 or key == ord('q') or key == ord('Q'):
    #                 print("Visualization stopped by user")
    #                 cv2.destroyAllWindows()
    #                 return
                
    #             # Environment step with action in the correct format
    #             obs, reward, done, _ = env.step(action_dict)()
    #             episode_reward += reward
    #             step += 1
                
    #             # Slow down visualization slightly
    #             time.sleep(0.01)
            
    #         print(f"Episode {episodes+1} finished with reward {episode_reward:.2f} in {step} steps")
    #         episodes += 1
        
    #     cv2.destroyAllWindows()
    #     print("Visualization complete")
        

    
    # def test_init(self):
    #     """Test if initialization creates the correct number of actors"""
    #     self.assertEqual(len(self.agent._task_behaviors), self.configs.n_agents,
    #                     "Should create n_agents task behaviors")
    #     self.assertEqual(len(self.agent._expl_behaviors), self.configs.n_agents, 
    #                     "Should create n_agents exploration behaviors")
        
    #     # Test if world model is shared
    #     for behavior in self.agent._task_behaviors:
    #         self.assertIs(behavior._world_model, self.agent._wm, 
    #                      "World model should be shared between all behaviors")

    # def test_policy_output(self):
    #     """Test if policy output has the correct format for multi-agent setting"""
    #     obs = {
    #         'agent0_obs': torch.zeros((1, 18)),
    #         'agent1_obs': torch.zeros((1, 18)),
    #         'agent2_obs': torch.zeros((1, 18)),
    #         'agent3_obs': torch.zeros((1, 18)),
    #         'image': torch.zeros((1, 64, 64, 3)),
    #         'is_first': torch.ones((1,), dtype=torch.bool),
    #         'is_last': torch.zeros((1,), dtype=torch.bool),
    #         'is_terminal': torch.zeros((1,), dtype=torch.bool)
    #     }
    #     policy_output, state = self.agent._policy(obs, None, training=True)
        
    #     # Check structure
    #     self.assertIn('action', policy_output, "Policy output should contain 'action'")
    #     self.assertIn('logprob', policy_output, "Policy output should contain 'logprob'")
    #     self.assertIn('agent_actions', policy_output, "Policy output should contain 'agent_actions'")
        
    #     # Check dimensions
    #     self.assertEqual(len(policy_output['agent_actions']), self.configs.n_agents, 
    #                     "Should have actions for each agent")
        
    #     # Check combined action shape
    #     if self.configs.n_agents > 1:
    #         expected_action_dim = self.configs.num_actions * self.configs.n_agents
    #         self.assertEqual(policy_output['action'].shape[-1], expected_action_dim,
    #                         "Combined action should have correct dimension")

    # def test_multi_agent_observation_encoding(self):
    #     """Test that all agent observations are properly encoded and combined"""
    #     # Create a sample batch with distinct values per agent
    #     obs = {
    #         'agent0_obs': torch.ones((1, 18), device=self.configs.device),
    #         'agent1_obs': torch.ones((1, 18), device=self.configs.device) * 2,
    #         'agent2_obs': torch.ones((1, 18), device=self.configs.device) * 3,
    #         'agent3_obs': torch.ones((1, 18), device=self.configs.device) * 4,
    #         'image': torch.ones((1, 64, 64, 3), device=self.configs.device) * 0.5,
    #         'is_first': torch.ones((1,), dtype=torch.bool, device=self.configs.device),
    #         'is_last': torch.zeros((1,), dtype=torch.bool, device=self.configs.device),
    #         'is_terminal': torch.zeros((1,), dtype=torch.bool, device=self.configs.device)
    #     }
        
    #     # Get the embedded representation
    #     preprocessed = self.agent._wm.preprocess(obs)
    #     embed = self.agent._wm.encoder(preprocessed)
        
    #     # Check that embed is not all zeros or all the same value
    #     # This verifies that agent-specific information is being preserved
    #     self.assertFalse(torch.allclose(embed, torch.zeros_like(embed)), 
    #                     "Embedded representation should not be all zeros")
    
    # def test_state_persistence(self):
    #     """Test that state persists between policy calls"""
    #     obs = {
    #         'agent0_obs': torch.zeros((1, 18)),
    #         'agent1_obs': torch.zeros((1, 18)),
    #         'agent2_obs': torch.zeros((1, 18)),
    #         'agent3_obs': torch.zeros((1, 18)),
    #         'image': torch.zeros((1, 64, 64, 3)),
    #         'is_first': torch.zeros((1,), dtype=torch.bool),  # Not first step
    #         'is_last': torch.zeros((1,), dtype=torch.bool),
    #         'is_terminal': torch.zeros((1,), dtype=torch.bool)
    #     }
        
    #     # Initial call with no state
    #     _, state1 = self.agent._policy(obs, None, training=False)
        
    #     # Second call with previous state
    #     _, state2 = self.agent._policy(obs, state1, training=False)
        
    #     # Check that states are different objects but have the same structure
    #     self.assertIsNot(state1, state2, "States should be different objects")
    #     self.assertEqual(type(state1), type(state2), "States should have same type")
        
    #     # Check latent state structure
    #     latent1, action1 = state1
    #     latent2, action2 = state2
        
    #     for key in latent1:
    #         self.assertIn(key, latent2, f"Key {key} should exist in both states")
    #         self.assertEqual(latent1[key].shape, latent2[key].shape, 
    #                         f"Shape for {key} should be the same")
    


    # def test_device_placement(self):
    #     """Test that all models are on the correct device"""
    #     target_device = self.configs.device
        
    #     # Check world model
    #     params = next(self.agent._wm.parameters())
    #     self.assertEqual(str(params.device), target_device,
    #                     "World model should be on the configured device")
        
    #     # Check task behaviors
    #     for i, behavior in enumerate(self.agent._task_behaviors):
    #         params = next(behavior.parameters())
    #         self.assertEqual(str(params.device), target_device,
    #                         f"Task behavior {i} should be on the configured device")
        
    #     # Check exploration behaviors
    #     for i, behavior in enumerate(self.agent._expl_behaviors):
    #         params = next(behavior.parameters())
    #         self.assertEqual(str(params.device), target_device,
    #                         f"Exploration behavior {i} should be on the configured device")

    def test_policy_determinism_sources(self):
        """Test to identify sources of non-determinism in the policy"""
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        obs = {
            'agent0_obs': torch.zeros((1, 18)),
            'agent1_obs': torch.zeros((1, 18)),
            'agent2_obs': torch.zeros((1, 18)),
            'agent3_obs': torch.zeros((1, 18)),
            'image': torch.zeros((1, 64, 64, 3)),
            'is_first': torch.ones((1,), dtype=torch.bool),
            'is_last': torch.zeros((1,), dtype=torch.bool),
            'is_terminal': torch.zeros((1,), dtype=torch.bool)
        }
        
        # Test 1: Fresh state each time
        policy_output1, _ = self.agent._policy(obs, None, training=False)
        policy_output2, _ = self.agent._policy(obs, None, training=False)
        
        print("Test 1 - Fresh states:")
        print("Actions 1:", policy_output1["action"])
        print("Actions 2:", policy_output2["action"])
        print("Differ:", not torch.allclose(policy_output1["action"], policy_output2["action"], atol=1e-5))
        
        # Test 2: Reuse the same exact state
        _, state1 = self.agent._policy(obs, None, training=False)
        policy_output3, _ = self.agent._policy(obs, state1, training=False)
        policy_output4, _ = self.agent._policy(obs, state1, training=False)
        
        print("\nTest 2 - Same exact state:")
        print("Actions 3:", policy_output3["action"])
        print("Actions 4:", policy_output4["action"])
        print("Differ:", not torch.allclose(policy_output3["action"], policy_output4["action"], atol=1e-5))
        
        # Test 3: Actor determinism directly
        with torch.no_grad():
            # Get features
            obs_p = self.agent._wm.preprocess(obs)
            embed = self.agent._wm.encoder(obs_p)
            latent, _ = self.agent._wm.dynamics.obs_step(None, None, embed, obs["is_first"])
            if self.configs.eval_state_mean:
                latent["stoch"] = latent["mean"]
            feat = self.agent._wm.dynamics.get_feat(latent)
            
            # Test actor directly
            actor = self.agent._task_behaviors[0].actor(feat)
            action1 = actor.mode()
            action2 = actor.mode()
            
            print("\nTest 3 - Actor directly:")
            print("Action 1:", action1)
            print("Action 2:", action2)
            print("Differ:", not torch.allclose(action1, action2, atol=1e-5))
            
    # def test_exploration_exploitation_switch(self):
    #     """Test if agent switches between exploration and exploitation without mocks"""
    #     obs = {
    #         'agent0_obs': torch.zeros((1, 18)),
    #         'agent1_obs': torch.zeros((1, 18)),
    #         'agent2_obs': torch.zeros((1, 18)),
    #         'agent3_obs': torch.zeros((1, 18)),
    #         'image': torch.zeros((1, 64, 64, 3)),
    #         'is_first': torch.ones((1,), dtype=torch.bool),
    #         'is_last': torch.zeros((1,), dtype=torch.bool),
    #         'is_terminal': torch.zeros((1,), dtype=torch.bool)
    #     }
        
    #     # Save original step and should_expl function
    #     original_step = self.agent._step
    #     original_should_expl = self.agent._should_expl
        
    #     try:
    #         # Force exploration mode by making _should_expl always return True
    #         self.agent._should_expl = lambda x: True
            
    #         # Call policy in exploration mode
    #         policy_output_expl, _ = self.agent._policy(obs, None, training=True)
            
    #         # Force exploitation mode by making _should_expl always return False
    #         self.agent._should_expl = lambda x: False
            
    #         # Call policy in exploitation mode
    #         policy_output_expl2, _ = self.agent._policy(obs, None, training=True)
            
    #         # Now test evaluation mode (should be deterministic/exploitation)
    #         policy_output_eval, _ = self.agent._policy(obs, None, training=False)
    #         policy_output_eval2, _ = self.agent._policy(obs, None, training=False)
            
    #         # Verify evaluation is deterministic (should get same actions)
    #         self.assertTrue(torch.allclose(policy_output_eval["action"], policy_output_eval2["action"]),
    #                         "Evaluation actions should be deterministic")
            
    #         # Check that exploration and exploitation policies are different
    #         # Note: This may occasionally fail due to random chance, but is usually reliable
    #         actions_different = not torch.allclose(policy_output_expl["action"], policy_output_eval["action"])
    #         self.assertTrue(actions_different, 
    #                         "Exploration and evaluation actions should typically differ")
            
    #     finally:
    #         # Restore original values
    #         self.agent._step = original_step
    #         self.agent._should_expl = original_should_expl
    
    ### ------------------------------------------------




    ### ------------------------------------------------




    ### ------------------------------------------------


    # If your encoder combines agent observations differently, you may need
    # more specific tests based on your implementation

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