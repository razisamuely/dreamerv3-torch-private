from envs.dmc import DeepMindControl
from envs.memorymaze import MemoryMaze
from envs.crafter_wrapper import Crafter
from envs.vmas_simple import Vmas
from envs.vmas_simple_spread import VmasSpread
import unittest

def test_observation_space_object_type():
    dmc_cartpol_env = DeepMindControl("cartpole_balance", size=(64, 64))
    print("\n\ndmc_cartpol_env.observation_space")
    print(dmc_cartpol_env.observation_space)

    dmc_walker_env = DeepMindControl("walker_walk", size=(64, 64))
    print("\n\ndmc_walker_env.observation space")
    print(dmc_walker_env.observation_space)

    memorynaze_env = MemoryMaze("9x9")
    print("\n\nmemorynaze_env.observation_space")
    print(memorynaze_env.observation_space)

    vmas_simple_env = Vmas("simple")
    print("\n\nvmas_simple_env.observation_space")
    print(vmas_simple_env.observation_space)

    vmas_simple_spread_env = VmasSpread("simple_spread")
    print("\n\nvmas_simple_spread_env.observation_space")
    print(vmas_simple_spread_env.observation_space)

def test_action_space_object_type():
    dmc_cartpol_env = DeepMindControl("cartpole_balance", size=(64, 64))
    print("\n\ndmc_cartpol_env.action_space")
    print(dmc_cartpol_env.action_space)

    dmc_walker_env = DeepMindControl("walker_walk", size=(64, 64))
    print("\n\ndmc_walker_env.action_space")
    print(dmc_walker_env.action_space)

    memorynaze_env = MemoryMaze("9x9")
    print("\n\nmemorynaze_env.action_space")
    print(memorynaze_env.action_space)

    vmas_simple_env = Vmas("simple")
    print("\n\nvmas_simple_env.action_space")
    print(vmas_simple_env.action_space)

    vmas_simple_spread_env = VmasSpread("simple_spread")
    print("\n\nvmas_simple_spread_env.action_space")
    print(vmas_simple_spread_env.action_space)
    

def print_obs(obs):
    if "image" not in obs:
        raise ValueError("No image in observation")
    
    for k, v in obs.items():
        if k == "image":
            print(k, v.shape)
        else:
            print(k, v,type(v))

def print_step(obs, reward, done, info):
    print("\n\nobs")
    print_obs(obs)
    print("\nreward")
    print(reward)
    print("\ndone")
    print(done)
    print("\ninfo")
    print(info)


def test_reset():
    # dmc_cartpol_env = DeepMindControl("cartpole_balance", size=(64, 64))
    # observation = dmc_cartpol_env.reset()
    # print("\n---- dmc_cartpol ---- ")
    # print_obs(observation)

    
    memorynaze_env = MemoryMaze("9x9")
    observation = memorynaze_env.reset()
    print("\n\n" +"-" * 30 + "\n---- memorynaze ---- ")
    print_obs(observation)

    vmas_simple_env = Vmas("simple")
    observation = vmas_simple_env.reset()
    print("\n\n" +"-" * 30 + "\n---- vmas_simple ---- ")
    print_obs(observation)

    vmas_simple_spread_env = VmasSpread("simple_spread")
    observation = vmas_simple_spread_env.reset()
    print("\n\n" +"-" * 30 + "\n---- vmas_simple_spread ---- ")
    print_obs(observation)



def test_step():
    # dmc_cartpol_env = DeepMindControl("cartpole_balance", size=(64, 64))
    # dmc_cartpol_env.reset()
    # obs, reward, done, info = dmc_cartpol_env.step(1)
    # dmc_cartpol_env.close()
    # print("\n---- dmc_cartpol ---- ")
    # print_step(obs, reward, done, info)

    # dmc_walker_env = DeepMindControl("walker_walk", size=(64, 64))
    # dmc_walker_env.reset()
    # obs, reward, done, info = dmc_walker_env.step(1)
    # dmc_walker_env.close()
    # print("\n---- dmc_walker ---- ")
    # print_step(obs, reward, done, info)

    memorynaze_env = MemoryMaze("9x9")
    memorynaze_env.reset()
    obs, reward, done, info = memorynaze_env.step(1)
    print("\n\n" +"-" * 30 + "\n---- memorynaze ---- ")
    print_step(obs, reward, done, info)

    vmas_simple_env = Vmas("simple")
    vmas_simple_env.reset()
    random_actions = vmas_simple_env.get_random_actions()
    random_action = random_actions[0]
    obs, reward, done, info = vmas_simple_env.step(random_action)
    print("\n\n" +"-" * 30 + "\n---- vmas_simple ---- ")
    print_step(obs, reward, done, info)

    vmas_simple_spread_env = VmasSpread("simple_spread")
    vmas_simple_spread_env.reset()
    random_actions = vmas_simple_spread_env.get_random_actions()
    obs, reward, done, info = vmas_simple_spread_env.step(random_actions)
    print("\n\n" +"-" * 30 + "\n---- vmas_simple_spread ---- ")
    print_step(obs, reward, done, info)


def test_no_random_action_is_taking_each_step():
    pass



if __name__ == '__main__':
    # test_observation_space_object_type()
    # test_action_space_object_type()
    # test_reset()
    test_step()
    
    unittest.main()
    
    

