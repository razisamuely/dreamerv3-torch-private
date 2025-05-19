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
import networks
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
import random
import unittest


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

        if self.configs.dyn_discrete:
            self._feat_size = self.configs.dyn_stoch * self.configs.dyn_discrete + self.configs.dyn_deter
        else:
            self._feat_size = self.configs.dyn_stoch + self.configs.dyn_deter

        self.critic = networks.MLP(
            self._feat_size,
            (255,) if self.configs.critic["dist"] == "symlog_disc" else (),
            self.configs.critic["layers"],
            self.configs.units,
            self.configs.act,
            self.configs.norm,
            self.configs.critic["dist"],
            outscale=self.configs.critic["outscale"],
            device=self.configs.device,
            name="SharedValue",
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
    



    def test_critic_forward(self):
        # Create a dummy batch with the correct dimensions
        # random [15, 1024, 1536]
        image_feat = torch.randn((self.configs.imag_horizon, 
                                  random.randint(1, 10),
                                  self._feat_size ))
        output = self.critic(image_feat)
        print("Output shape:", output.logits.shape, output.probs.shape, output.buckets.shape )

        #  output.logits.shape, output.probs.shape, output.buckets.shape 
        #  (torch.Size([15, 1024, 255]), torch.Size([15, 1024, 255]), torch.Size([255]))
        #  Check output shape
        expected_shape = (self.configs.batch_size, self.configs.critic["layers"][-1])
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"


if __name__ == "__main__":
    unittest.main()