import unittest
import gym.spaces
import torch
import numpy as np
from pathlib import Path
import sys
import pathlib
import random

sys.path.append(str(Path(__file__).parent.parent))
from dreamer import Dreamer, count_steps, make_env
from tests.utils import load_config
from parallel import Damy
from tools import Logger
import networks

# Constants
NUM_AGENTS = 4
AGENT_OBS_SIZE = 18
IMAGE_SHAPE = (64, 64, 3)
ACTION_SPACE_SIZE = 8
CRITIC_OUTPUT_SIZE = 255
AGENT_OBS_KEYS = [f'agent{i}_obs' for i in range(NUM_AGENTS)]
BOOLEAN_KEYS = ['is_first', 'is_last', 'is_terminal']

ACTION_SPACE = gym.spaces.Box(low=-1, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.float32)
OBSERVATION_SPACE = gym.spaces.Dict({
    **{key: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(AGENT_OBS_SIZE,), dtype=np.float32) 
       for key in AGENT_OBS_KEYS},
    'image': gym.spaces.Box(low=0, high=255, shape=IMAGE_SHAPE, dtype=np.uint8),
    **{key: gym.spaces.Discrete(2) for key in BOOLEAN_KEYS}
})


class TestMultiAgentDreamer(unittest.TestCase):

    def setUp(self):
        self.config = load_config()
        self.config.logdir = str(pathlib.Path(__file__).parent.parent / "logs")
        
        logdir = pathlib.Path(self.config.logdir).expanduser()
        self.config.traindir = self.config.traindir or logdir / "train_eps"
        
        step = count_steps(self.config.traindir)
        logger = Logger(logdir, self.config.action_repeat * step)
        
        train_dataset = self._create_mock_dataset()
        
        make_env_fn = lambda mode, id: make_env(self.config, mode, id)
        train_envs = [Damy(make_env_fn("train", i)) for i in range(self.config.envs)]
        
        acts = train_envs[0].action_space[0]
        self.config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        
        self.agent = Dreamer(
            obs_space=OBSERVATION_SPACE,
            act_space=ACTION_SPACE,
            config=self.config,
            logger=logger,
            dataset=train_dataset
        )

        if self.config.dyn_discrete:
            feat_size = self.config.dyn_stoch * self.config.dyn_discrete + self.config.dyn_deter
        else:
            feat_size = self.config.dyn_stoch + self.config.dyn_deter

        self.critic = networks.MLP(
            feat_size,
            (CRITIC_OUTPUT_SIZE,) if self.config.critic["dist"] == "symlog_disc" else (),
            self.config.critic["layers"],
            self.config.units,
            self.config.act,
            self.config.norm,
            self.config.critic["dist"],
            outscale=self.config.critic["outscale"],
            device=self.config.device,
            name="SharedValue",
        )
        
        self.feat_size = feat_size

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

    def test_critic_forward_pass(self):
        batch_size = random.randint(1, 10)
        features = torch.randn((self.config.imag_horizon, batch_size, self.feat_size))
        
        output = self.critic(features)
        
        expected_logits_shape = (self.config.imag_horizon, batch_size, CRITIC_OUTPUT_SIZE)
        expected_probs_shape = (self.config.imag_horizon, batch_size, CRITIC_OUTPUT_SIZE)
        expected_buckets_shape = (CRITIC_OUTPUT_SIZE,)
        
        assert output.logits.shape == expected_logits_shape
        assert output.probs.shape == expected_probs_shape
        assert output.buckets.shape == expected_buckets_shape


if __name__ == "__main__":
    unittest.main()