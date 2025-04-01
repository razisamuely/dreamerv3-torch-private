import torch
import numpy as np
import gym
from vmas import make_env
from vmas.simulator.core import Agent

# Add this at the beginning of your reset and step methods


class VmasSpread:
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, seed=0, device="cpu", **kwargs):
        self._name = name
        self._action_repeat = action_repeat
        self._size = size
        self._camera = camera
        self._device = device
        self._kwargs = kwargs
        self._random = np.random.RandomState(seed)
        
        # Create environment
        self._env = make_env(
            scenario=name,
            num_envs=1,  # No vectorization here since DreamerV3 handles that
            device=device,
            continuous_actions=True,
            dict_spaces=False,
            seed=seed,
            terminated_truncated=True,
            **kwargs
        )
        
        self._last_obs = None
        self.reward_range = [-np.inf, np.inf]

    def _ensure_batch_dimension(self, x):
        """Ensure input has the right format for DreamerV3"""
        if isinstance(x, np.ndarray):
            # For vector observations: Keep them as 1D arrays
            if len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1):
                return x.flatten()  # Make sure it's a flat 1D array
                
            # For images: Keep them as [H, W, C]
            elif len(x.shape) == 3 and x.shape[-1] in [1, 3]:
                return x  # Return as is
                
            # For 2D arrays that aren't batches of vectors
            elif len(x.shape) == 2 and x.shape[0] != 1:
                return x  # Return as is
                
        return x


    @property
    def observation_space(self):
        # Convert VMAS observation space to DreamerV3 format
        spaces = {}
        
        # Process each agent's observation space
        for agent_idx, agent_space in enumerate(self._env.observation_space):
            if isinstance(agent_space, gym.spaces.Dict):
                # If it's a Dict space, create keys for each sub-space
                for key, space in agent_space.spaces.items():
                    shape = space.shape if space.shape else (1,)
                    spaces[f"agent{agent_idx}_{key}"] = gym.spaces.Box(
                        -np.inf, np.inf, shape, dtype=np.float32
                    )
            else:
                # If it's a direct Box space
                shape = agent_space.shape if agent_space.shape else (1,)
                spaces[f"agent{agent_idx}_obs"] = gym.spaces.Box(
                    -np.inf, np.inf, shape, dtype=np.float32
                )
        
        # Add image observation
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        # Add required fields for DreamerV3
        spaces["is_first"] = gym.spaces.Box(0, 1, (1,), dtype=np.bool_)
        spaces["is_last"] = gym.spaces.Box(0, 1, (1,), dtype=np.bool_)
        spaces["is_terminal"] = gym.spaces.Box(0, 1, (1,), dtype=np.bool_)
        
        return gym.spaces.Dict(spaces)
    @property
    def action_space(self):
        # For continuous actions
        if hasattr(self._env.action_space[0], "shape"):
            low = np.concatenate([space.low for space in self._env.action_space])
            high = np.concatenate([space.high for space in self._env.action_space])
            return gym.spaces.Box(low, high, dtype=np.float32)
        else:
            raise ValueError("Action space is not continuous. Please check the environment. or fix this function to support discrete actions")

    def step(self, action):
        
        # if action is a numpy array, convert it to a list of tensors
        if isinstance(action, np.ndarray):
            action = [torch.tensor(np.array([action]))]
        elif isinstance(action, torch.Tensor):
            action = [action]
        # if kist of tensors, pass 
        if isinstance(action, list):
            pass
        else:
            print(action)
            raise ValueError("Action must be a numpy array or a list of tensors, currently: {},}".format(type(action)))
        

        # If the action is a list of one tensor, convert it to a list of tensors
        if len(action) == 1 and isinstance(action[0], torch.Tensor):
            actions  = []
            jump_length = self._env.action_space[0].shape[0]
            for i in range(0,len(action[0][0]), jump_length):
                actions.append(action[0][0][i:i+jump_length].unsqueeze_(0))
            
            action = actions
            
        
        # Now actions is a list of tensors, one for each agent
        # then convert action to number of agents
        # Step environment
        total_reward = 0
        for _ in range(self._action_repeat):
            obs, rewards, terminated, truncated, infos = self._env.step(action)

            total_reward += sum([r.item() for r in rewards])
            done = all(terminated) or all(truncated)
            if done:
                break

            
        formatted_obs = {}
        for agent_idx, agent_obs in enumerate(obs):
            if isinstance(agent_obs, torch.Tensor):
                value = agent_obs.cpu().numpy()
                formatted_obs[f"agent{agent_idx}_obs"] = self._ensure_batch_dimension(value)
            else:
                raise ValueError(f"Unsupported observation type: {type(agent_obs)}, expected torch.Tensor or dict, or just change this function to support it")
            
        image = self.render(mode="rgb_array")
        formatted_obs["image"] = self._ensure_batch_dimension(image)
                
        formatted_obs["is_terminal"] = all(terminated)
        formatted_obs["is_last"] = done
        formatted_obs["is_first"] = False
        
        return formatted_obs, total_reward, done, infos[0]

    def reset(self):
        obs = self._env.reset()
        
        # Format observations to match DreamerV3
        formatted_obs = {}
        
        # Handle the case when obs is a list of tensors
        for agent_idx, agent_obs in enumerate(obs):
            if isinstance(agent_obs, torch.Tensor):
                # Direct tensor output
                value = agent_obs.cpu().numpy()
                formatted_obs[f"agent{agent_idx}_obs"] = self._ensure_batch_dimension(value)
            else:
                # Dictionary output
                for key, value in agent_obs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    formatted_obs[f"agent{agent_idx}_{key}"] = self._ensure_batch_dimension(value)
        
        # Add rendered image
        image = self.render(mode="rgb_array")
        formatted_obs["image"] = self._ensure_batch_dimension(image)
        
        # Make sure boolean flags are consistent
        formatted_obs["is_terminal"] = False
        formatted_obs["is_first"] = True
        formatted_obs["is_last"] = False
        
        return formatted_obs

    def render(self, mode="rgb_array",get_orignal_frame = False):
        if mode != "rgb_array":
            return self._env.render(mode=mode)
        
        # Render at native resolution
        frame = self._env.render(
            mode="rgb_array",
            agent_index_focus=self._camera,
            visualize_when_rgb=False
        )
        
        # Only resize if explicitly required
        if self._size != (frame.shape[1], frame.shape[0]) and (not get_orignal_frame):
            import cv2
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_LANCZOS4)
        
        return frame
    
    def close(self):
        self._env.close()

    def get_random_actions(self):
        return self._env.get_random_actions()
    


 
