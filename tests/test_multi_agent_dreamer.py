import unittest
import gym.spaces
import torch
import numpy as np
from pathlib import Path
import envs.wrappers as wrappers
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

    # def test_policy_determinism_sources(self):
    #     """Test to identify sources of non-determinism in the policy"""
    #     # Set random seeds for reproducibility
    #     torch.manual_seed(42)
    #     np.random.seed(42)
        
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
        
    #     # Test 1: Fresh state each time
    #     policy_output1, _ = self.agent._policy(obs, None, training=False)
    #     policy_output2, _ = self.agent._policy(obs, None, training=False)
        
    #     print("Test 1 - Fresh states:")
    #     print("Actions 1:", policy_output1["action"])
    #     print("Actions 2:", policy_output2["action"])
    #     print("Differ:", not torch.allclose(policy_output1["action"], policy_output2["action"], atol=1e-5))
        
    #     # Test 2: Reuse the same exact state
    #     _, state1 = self.agent._policy(obs, None, training=False)
    #     policy_output3, _ = self.agent._policy(obs, state1, training=False)
    #     policy_output4, _ = self.agent._policy(obs, state1, training=False)
        
    #     print("\nTest 2 - Same exact state:")
    #     print("Actions 3:", policy_output3["action"])
    #     print("Actions 4:", policy_output4["action"])
    #     print("Differ:", not torch.allclose(policy_output3["action"], policy_output4["action"], atol=1e-5))
        
    #     # Test 3: Actor determinism directly
    #     with torch.no_grad():
    #         # Get features
    #         obs_p = self.agent._wm.preprocess(obs)
    #         embed = self.agent._wm.encoder(obs_p)
    #         latent, _ = self.agent._wm.dynamics.obs_step(None, None, embed, obs["is_first"])
    #         if self.configs.eval_state_mean:
    #             latent["stoch"] = latent["mean"]
    #         feat = self.agent._wm.dynamics.get_feat(latent)
            
    #         # Test actor directly
    #         actor = self.agent._task_behaviors[0].actor(feat)
    #         action1 = actor.mode()
    #         action2 = actor.mode()
            
    #         print("\nTest 3 - Actor directly:")
    #         print("Action 1:", action1)
    #         print("Action 2:", action2)
    #         print("Differ:", not torch.allclose(action1, action2, atol=1e-5))
            

    # def test_agent_training_effectiveness(self):
    #     """Test that each agent's policy parameters are actually updated during training"""
    #     # Get initial parameter values for each agent's policy
    #     initial_params = []
    #     for agent_idx in range(self.configs.n_agents):
    #         # Store one parameter tensor from each agent's policy
    #         for param in self.agent._task_behaviors[agent_idx].actor.parameters():
    #             initial_params.append(param.clone().detach())
    #             break
        
    #     # Ensure we got one parameter per agent
    #     self.assertEqual(len(initial_params), self.configs.n_agents,
    #                     "Should have one parameter tensor per agent")
        
    #     # Run several training steps to ensure parameters change
    #     for _ in range(5):  # Run multiple steps for more reliable changes
    #         batch = next(self.train_dataset)
    #         self.agent._train(batch)
        
    #     # Check if parameters changed for each agent
    #     for agent_idx in range(self.configs.n_agents):
    #         # Get the same parameter we stored initially
    #         current_param = None
    #         for param in self.agent._task_behaviors[agent_idx].actor.parameters():
    #             current_param = param
    #             break
            
    #         # Check if parameter changed
    #         param_changed = not torch.allclose(initial_params[agent_idx], current_param)
    #         self.assertTrue(param_changed, 
    #                     f"Agent {agent_idx}'s policy parameters should change during training")


    # def test_episode_termination(self):
    #     """Test that episodes correctly end when terminal states are reached"""
    #     # Create a batch with explicit episode boundaries
    #     batch_size = 2
    #     batch_length = 10
        
    #     # Create a custom batch with clear episode boundaries
    #     custom_batch = {
    #         'agent0_obs': torch.zeros((batch_size, batch_length, 18)),
    #         'agent1_obs': torch.zeros((batch_size, batch_length, 18)),
    #         'agent2_obs': torch.zeros((batch_size, batch_length, 18)),
    #         'agent3_obs': torch.zeros((batch_size, batch_length, 18)),
    #         'image': torch.zeros((batch_size, batch_length, 64, 64, 3)),
    #         'action': torch.zeros((batch_size, batch_length, self.configs.num_actions * self.configs.n_agents)),
    #         'reward': torch.ones((batch_size, batch_length)),  # Set all rewards to 1
    #         'discount': torch.ones((batch_size, batch_length)),
    #         'is_first': torch.zeros((batch_size, batch_length), dtype=torch.bool),
    #         'is_terminal': torch.zeros((batch_size, batch_length), dtype=torch.bool)
    #     }
        
    #     # Set episode boundaries - make middle timestep terminal
    #     terminal_step = 5
    #     custom_batch['is_terminal'][:, terminal_step] = True
    #     custom_batch['is_first'][:, 0] = True  # First step is start of episode
    #     custom_batch['is_first'][:, terminal_step + 1] = True  # Step after terminal is start of new episode
        
    #     # Train on this batch
    #     post, context, wm_metrics = self.agent._wm._train(custom_batch)
        
    #     # Check if the mask (discount) in the world model training
    #     # properly zeroes out transitions after terminal states
    #     if 'mask' in wm_metrics:
    #         # If mask is available in metrics, check it directly
    #         mask = wm_metrics['mask']
    #         self.assertEqual(torch.sum(mask[:, terminal_step+1:]).item(), 0,
    #                         "Mask should zero out steps after terminal state")
        
    #     # Alternative check: verify the GRU state resets after terminal state
    #     # Extract features before and after terminal state
    #     feat_post = self.agent._wm.dynamics.get_feat(post)
        
    #     # Check if there's a discontinuity in features at terminal state
    #     # This is a bit implementation-specific, but we can look for larger changes
    #     # at the terminal boundary compared to normal transitions
    #     diffs = []
    #     for t in range(1, batch_length):
    #         diff = torch.mean(torch.abs(feat_post[:, t] - feat_post[:, t-1])).item()
    #         diffs.append(diff)
        
    #     # The difference at the terminal boundary should be larger
    #     terminal_diff = diffs[terminal_step]
    #     avg_other_diff = sum(d for i, d in enumerate(diffs) if i != terminal_step) / (len(diffs) - 1)
        
    #     # Print for debugging
    #     print(f"Feature difference at terminal boundary: {terminal_diff}")
    #     print(f"Average difference at other steps: {avg_other_diff}")
        
    #     # The terminal difference should typically be larger as the state resets
    #     self.assertGreater(terminal_diff, avg_other_diff ,
    #                     "State change at terminal boundary should be larger than normal transitions")
    
    # def test_episode_truly_ends(self):
    #     """Test that episodes truly end at terminal states and agent resets properly"""
        
    #     # Get a reference to one of your environments
    #     env = self.train_envs[0]
        
    #     # Initialize variables
    #     agent_state = None
    #     total_steps = 0
    #     episode_lengths = []
    #     max_episodes = 3
    #     episodes_completed = 0
        
    #     # Run for multiple episodes
    #     while episodes_completed < max_episodes and total_steps < 1000:
    #         # Reset environment
    #         obs = env.reset()()
            
    #         # Add required flags if not present
    #         if "is_first" not in obs:
    #             obs["is_first"] = np.array([True])
    #         if "is_terminal" not in obs:
    #             obs["is_terminal"] = np.array([False])
                
    #         # Convert numpy arrays to tensors and ensure correct dimensions
    #         obs_tensor = {}
    #         for k, v in obs.items():
    #             # Handle image specially to ensure correct dimensions
    #             if k == 'image':
    #                 if len(v.shape) == 3:  # If missing batch dimension
    #                     v = np.expand_dims(v, 0)
    #             elif isinstance(v, np.ndarray) and len(v.shape) == 1 and v.shape[0] == 1:
    #                 # Keep arrays with shape [1] as is
    #                 pass
    #             elif isinstance(v, np.ndarray) and len(v.shape) == 1:
    #                 # Add batch dimension to 1D arrays
    #                 v = np.expand_dims(v, 0)
    #             elif not isinstance(v, np.ndarray):
    #                 # Convert scalars to arrays with batch dimension
    #                 v = np.array([v])
                    
    #             obs_tensor[k] = v
            
    #         done = False
    #         episode_step = 0
            
    #         # Run until episode ends
    #         while not done and episode_step < 200:  # Cap at 200 steps per episode
    #             # Get action from agent
    #             policy_output, agent_state = self.agent(
    #                 obs_tensor, 
    #                 np.array([done]),
    #                 agent_state,
    #                 training=False
    #             )
                
    #             # Extract action and step in environment
    #             action = policy_output["action"][0].cpu().numpy()
    #             action_dict = {"action": action}
                
    #             # Take step in environment
    #             obs, reward, done, _ = env.step(action_dict)()
                
    #             # Process observation again
    #             if "is_first" not in obs:
    #                 obs["is_first"] = np.array([False])
    #             if "is_terminal" not in obs:
    #                 obs["is_terminal"] = np.array([done])  # Mark as terminal if done
                    
    #             # Convert to tensors with proper dimensions
    #             obs_tensor = {}
    #             for k, v in obs.items():
    #                 # Handle image specially to ensure correct dimensions
    #                 if k == 'image':
    #                     if len(v.shape) == 3:  # If missing batch dimension
    #                         v = np.expand_dims(v, 0)
    #                 elif isinstance(v, np.ndarray) and len(v.shape) == 1 and v.shape[0] == 1:
    #                     # Keep arrays with shape [1] as is
    #                     pass
    #                 elif isinstance(v, np.ndarray) and len(v.shape) == 1:
    #                     # Add batch dimension to 1D arrays
    #                     v = np.expand_dims(v, 0)
    #                 elif not isinstance(v, np.ndarray):
    #                     # Convert scalars to arrays with batch dimension
    #                     v = np.array([v])
                        
    #                 obs_tensor[k] = v
                    
    #             episode_step += 1
    #             total_steps += 1
            
    #         # Record completed episode
    #         episode_lengths.append(episode_step)
    #         episodes_completed += 1
            
    #         # Verify agent state reset happened
    #         if agent_state is not None:
    #             # Check if agent's internal state has proper reset indicators
    #             self.assertTrue(
    #                 "stoch" in agent_state[0],
    #                 "Agent state should maintain stochastic component after episode termination"
    #             )
        
    #     # Verify multiple episodes completed
    #     self.assertEqual(
    #         episodes_completed, 
    #         max_episodes, 
    #         f"Should complete {max_episodes} episodes but only completed {episodes_completed}"
    #     )
        
    #     # Verify episodes had reasonable length
    #     self.assertTrue(
    #         all(length > 0 for length in episode_lengths),
    #         "All episodes should have positive length"
    #     )
        
    #     # Verify episodes had reasonable length
    #     self.assertTrue(
    #         all(length > 0 for length in episode_lengths),
    #         "All episodes should have positive length"
    #     )

    #     # Check if all episodes hit the time limit or not
    #     if all(length == 200 for length in episode_lengths):
    #         print("NOTE: All episodes reached the maximum step limit (200)")
    #         print("This suggests episodes are terminating due to the time limit, not natural termination")
    #         # This isn't necessarily a failure, just informational
    #     else:
    #         # If lengths vary, then some episodes ended naturally
    #         self.assertTrue(
    #             len(set(episode_lengths)) > 1,
    #             "Episode lengths should vary if natural termination occurs"
    #         )

    #     print(f"Episode lengths: {episode_lengths}")
    

    # def test_gradient_flow_to_all_agents(self):
    #     """Test that gradients flow to each agent's parameters independently."""
    #     # Setup for gradient checking
    #     grad_magnitudes = {i: [] for i in range(self.configs.n_agents)}
    #     hooks = []
        
    #     # Register hooks to capture gradients
    #     for agent_idx, behavior in enumerate(self.agent._task_behaviors):
    #         for name, param in behavior.actor.named_parameters():
    #             # Create a closure to keep agent_idx in context
    #             def get_hook(agent_i, param_name):
    #                 def hook(grad):
    #                     if grad is not None:
    #                         grad_magnitudes[agent_i].append(float(torch.sum(torch.abs(grad))))
    #                 return hook
                
    #             hook = param.register_hook(get_hook(agent_idx, name))
    #             hooks.append(hook)
        
    #     # Run one training step
    #     batch = next(self.train_dataset)
    #     metrics = self.agent._train(batch)
        
    #     # Remove hooks
    #     for h in hooks:
    #         h.remove()
        
    #     # Check that gradients flowed to each agent
    #     for agent_idx in range(self.configs.n_agents):
    #         self.assertTrue(len(grad_magnitudes[agent_idx]) > 0, 
    #                     f"Agent {agent_idx} should have gradients")
    #         self.assertTrue(sum(grad_magnitudes[agent_idx]) > 0, 
    #                     f"Agent {agent_idx}'s gradients should not all be zero")
        
    #     # Print the magnitudes for comparison
    #     for agent_idx in range(self.configs.n_agents):
    #         avg_grad = sum(grad_magnitudes[agent_idx]) / len(grad_magnitudes[agent_idx])
    #         print(f"Agent {agent_idx} average gradient magnitude: {avg_grad}")

    
    # def test_gradient_directions_differ_between_agents(self):
    #     """Test that different agents' gradients point in different directions."""
    #     # Store flattened gradients for each agent
    #     agent_gradients = []
        
    #     # Get a batch of data and run a training step
    #     batch = next(self.train_dataset)
        
    #     # Record gradients for each agent
    #     for agent_idx, behavior in enumerate(self.agent._task_behaviors):
    #         # Clear previous gradients
    #         for param in behavior.actor.parameters():
    #             if param.grad is not None:
    #                 param.grad.zero_()
            
    #         # Run training for this specific agent
    #         post, context, _ = self.agent._wm._train(batch)
    #         start = post
    #         reward = lambda f, s, a: self.agent._wm.heads["reward"](
    #             self.agent._wm.dynamics.get_feat(s)
    #         ).mode()
    #         all_policies = [b.actor for b in self.agent._task_behaviors]
    #         _, _, _, _, _ = behavior._train(start, reward, all_policies)
            
    #         # Collect and flatten all gradients
    #         flat_grad = []
    #         for param in behavior.actor.parameters():
    #             if param.grad is not None:
    #                 flat_grad.append(param.grad.view(-1))
            
    #         # Concatenate into one vector
    #         if flat_grad:
    #             agent_gradients.append(torch.cat(flat_grad))
        
    #     # Compute cosine similarities between agent gradients
    #     if len(agent_gradients) > 1:
    #         similarities = []
    #         for i in range(len(agent_gradients)):
    #             for j in range(i+1, len(agent_gradients)):
    #                 cos_sim = torch.nn.functional.cosine_similarity(
    #                     agent_gradients[i], agent_gradients[j], dim=0
    #                 )
    #                 similarities.append(cos_sim.item())
    #                 print(f"Cosine similarity between agent {i} and {j}: {cos_sim.item()}")
            
    #         # If agents are learning independently, gradients shouldn't be too aligned
    #         avg_similarity = sum(similarities) / len(similarities)
    #         self.assertLess(avg_similarity, 0.9, 
    #                     "Agent gradients should not be too similar (independence check)")



    # def test_parameters_are_trainable(self):
    #     """Test that world model and agent parameters can be directly updated."""
    #     print("\nTesting that parameters are directly trainable")
        
    #     # First check: Apply manual gradients to world model
    #     wm_param = next(self.agent._wm.parameters())
    #     original_wm_param = wm_param.clone().detach()
        
    #     # Create a dummy loss and backward directly
    #     dummy_loss = wm_param.sum()
    #     dummy_loss.backward()
        
    #     # Now manually step the optimizer
    #     original_optimizer = self.agent._wm._model_opt
        
    #     # Create a temporary optimizer to ensure updates
    #     temp_optimizer = torch.optim.Adam([wm_param], lr=0.1)
    #     temp_optimizer.step()
        
    #     # Check if the parameter was updated
    #     wm_changed = not torch.allclose(original_wm_param, wm_param)
    #     print(f"World model parameter changed after direct update: {wm_changed}")
    #     print(f"Change magnitude: {torch.sum(torch.abs(original_wm_param - wm_param)).item()}")
        
    #     # Reset for the test with agent
    #     temp_optimizer.zero_grad()
        
    #     # Second check: Apply manual gradients to agent
    #     agent_param = next(self.agent._task_behaviors[0].parameters())
    #     original_agent_param = agent_param.clone().detach()
        
    #     # Create a dummy loss and backward
    #     dummy_loss = agent_param.sum()
    #     dummy_loss.backward()
        
    #     # Create a temporary optimizer
    #     temp_optimizer = torch.optim.Adam([agent_param], lr=0.1)
    #     temp_optimizer.step()
        
    #     # Check if the parameter was updated
    #     agent_changed = not torch.allclose(original_agent_param, agent_param)
    #     print(f"Agent parameter changed after direct update: {agent_changed}")
    #     print(f"Change magnitude: {torch.sum(torch.abs(original_agent_param - agent_param)).item()}")
        
    #     # Verify that parameters can be updated directly
    #     self.assertTrue(wm_changed, "World model parameters should be directly updatable")
    #     self.assertTrue(agent_changed, "Agent parameters should be directly updatable")

    
    # def test_single_train_call_updates_all_components(self):
    #     """Test that a single train call updates the world model and all agents."""
    #     print("\nTesting that a single train call updates all components")
        
    #     # Get a batch of data with non-zero values
    #     batch = next(self.train_dataset)
    #     batch['image'] = torch.rand_like(batch['image'])
    #     batch['reward'] = torch.rand_like(batch['reward'])
    #     batch['agent0_obs'] = torch.rand_like(batch['agent0_obs'])
    #     batch['agent1_obs'] = torch.rand_like(batch['agent1_obs'])
    #     batch['agent2_obs'] = torch.rand_like(batch['agent2_obs'])
    #     batch['agent3_obs'] = torch.rand_like(batch['agent3_obs'])
        
    #     # Store initial parameters by keeping references to the actual parameter objects
    #     # World model parameters
    #     wm_params = list(self.agent._wm.parameters())
    #     wm_initial_values = [p.clone().detach() for p in wm_params]
        
    #     # Agent parameters
    #     agent_params = []
    #     agent_initial_values = []
    #     for agent_idx in range(self.configs.n_agents):
    #         params = list(self.agent._task_behaviors[agent_idx].parameters())
    #         agent_params.append(params)
    #         agent_initial_values.append([p.clone().detach() for p in params])
        
    #     # Call the train method once
    #     metrics = self.agent._train(batch)
        
    #     # Check if world model parameters changed
    #     wm_diffs = []
    #     for i, (param, initial) in enumerate(zip(wm_params, wm_initial_values)):
    #         diff = torch.sum(torch.abs(param - initial)).item()
    #         if diff > 1e-10:
    #             wm_diffs.append((f"param_{i}", diff))
        
    #     wm_changed = len(wm_diffs) > 0
    #     print(f"World model parameter changes: {wm_diffs[:5]}..." if len(wm_diffs) > 5 else wm_diffs)
        
    #     # Check if agent parameters changed
    #     agents_changed = [False] * self.configs.n_agents
    #     for agent_idx in range(self.configs.n_agents):
    #         agent_diffs = []
    #         for i, (param, initial) in enumerate(zip(agent_params[agent_idx], agent_initial_values[agent_idx])):
    #             diff = torch.sum(torch.abs(param - initial)).item()
    #             if diff > 1e-10:
    #                 agent_diffs.append((f"param_{i}", diff))
            
    #         agents_changed[agent_idx] = len(agent_diffs) > 0
    #         print(f"Agent {agent_idx} parameter changes: {agent_diffs[:5]}..." if len(agent_diffs) > 5 else agent_diffs)
        
    #     # Verify world model was updated
    #     self.assertTrue(wm_changed, "World model parameters should change after training")
        
    #     # Verify all agents were updated
    #     for agent_idx in range(self.configs.n_agents):
    #         self.assertTrue(agents_changed[agent_idx], 
    #                     f"Agent {agent_idx} parameters should change after training")
        
    #     # Print some training metrics
    #     loss_metrics = [(k, v) for k, v in metrics.items() if 'loss' in k]
    #     print(f"Training metrics: {loss_metrics[:5]}..." if len(loss_metrics) > 5 else loss_metrics)

    def test_agent_and_random_have_same_input_output(self):

        """Test if the __call__ method works correctly with multiple agents."""
        
        # Create a small batch of observations
        batch_size = 2
        num_agents = self.configs.n_agents
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
        
        # Call the agent
        policy_output, state = self.agent(obs, reset, None, training=False)
        
        import tools
        from torch import distributions as torchd

        config = self.configs

        acts = self.train_envs[0].action_space[0] # after 
        print("Action Space", acts)
        
        
        action_dim = acts.n if hasattr(acts, "n") else acts.shape[0]
        config.num_actions = action_dim

        combined_action_dim = action_dim * num_agents
        
        if hasattr(acts, "discrete"):
            # For discrete actions, create a random distribution for each agent
            # and combine them properly for the batch
            random_distributions = []
            for _ in range(num_agents):
                random_distributions.append(
                    tools.OneHotDist(
                        torch.zeros(batch_size, action_dim)
                    )
                )
        else:
            # For continuous actions, create a proper random distribution
            # that matches the shape of the agent's output
            combined_low = torch.tensor(acts.low).repeat(num_agents)
            combined_high = torch.tensor(acts.high).repeat(num_agents)
            
            # Create random distributions for each agent in the batch
            random_distributions = []
            for _ in range(num_agents):
                random_distributions.append(
                    torchd.independent.Independent(
                        torchd.uniform.Uniform(
                            torch.tensor(acts.low).repeat(batch_size, 1),
                            torch.tensor(acts.high).repeat(batch_size, 1),
                        ),
                        1,
                    )
                )
        
        def random_agent(o, d, s):
            # Sample actions for each agent
            actions = []
            logprobs = []
            
            for dist in random_distributions:
                action = dist.sample()
                logprob = dist.log_prob(action)
                actions.append(action)
                logprobs.append(logprob)
            
            # Stack actions and logprobs to match agent output format
            combined_action = torch.cat(actions, dim=1) if not hasattr(acts, "discrete") else torch.stack(actions, dim=1)
            combined_logprob = torch.stack(logprobs, dim=1)
            
            return {"action": combined_action, "logprob": combined_logprob}, None
        
        # Get random agent output
        random_agent_output, _ = random_agent(obs, reset, None)
        random_agent_logprob = random_agent_output["logprob"]
        random_agent_action = random_agent_output["action"]
        
        # Get main agent output
        agent_logprob = policy_output["logprob"]
        agent_action = policy_output["action"]
        
        # Print for debugging
        print("Random agent logprob shape:", random_agent_logprob.shape)
        print("Agent logprob shape:", agent_logprob.shape)
        print("Random agent action shape:", random_agent_action.shape)
        print("Agent action shape:", agent_action.shape)
        
        # Now you can properly compare the outputs since they should have the same shape
        # You might want to add assertions here to verify the shapes match
        assert random_agent_logprob.shape == agent_logprob.shape, "Logprob shapes don't match"
        assert random_agent_action.shape == agent_action.shape, "Action shapes don't match"



    ### ------------------------------------------------


    # Exploration vs explotation 
    # test_reward_processing_basic
    # Random agent in main script as same obs space of not random agent

    ### ------------------------------------------------



if __name__ == "__main__":
        unittest.main()