import torch
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from TD3 import TD3
from utils import ReplayBuffer
from rsoccer_gym.vss.env_ma import *

def train():
    ######### Hyperparameters #########
    env_name = "SSLShootEnv-v0"
    number = 7
    random_seed = 0
    gamma = 0.99  # discount for future rewards
    batch_size = 100  # num of transitions sampled from replay buffer
    lr = 0.0001
    exploration_noise = 0.1
    polyak = 0.995  # target policy update parameter (1-tau)
    policy_noise = 0.2  # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2  # delayed policy updates parameter
    max_episodes = 10000000  # max num of episodes
    max_timesteps = 100  # max timesteps in one episode
    save_rate = 5000  # save the check point per ? episode
    ###################################
    # save setting
    directory = f"./models/{env_name}/{number}"
    os.makedirs(directory, exist_ok=True)
    hyperparameters = {
        'env_name': env_name,
        'number': number,
        'random_seed': random_seed,
        'gamma': gamma,
        'batch_size': batch_size,
        'lr': lr,
        'exploration_noise': exploration_noise,
        'polyak': polyak,
        'policy_noise': policy_noise,
        'noise_clip': noise_clip,
        'policy_delay': policy_delay,
        'max_episodes': max_episodes,
        'max_timesteps': max_timesteps,
        'save_rate': save_rate,
        'directory': directory
    }
    np.save(f"{directory}/args_num_{number}.npy", hyperparameters)


    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # logging variables:
    ep_reward = 0
    ep_step = 0
    total_step = 0
    writer = SummaryWriter(log_dir=f'./runs/{env_name}/{number}')

    # init sub-rewards dict
    reward_dict = {}
    for sub_reward in env.reward_shaping_total:
        if sub_reward.startswith("rw_"):
            reward_dict[sub_reward] = 0

    done_type = None
    # training procedure:
    for episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            total_step += 1
            ep_step +=1
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            action = action.clip(env.action_space.low, env.action_space.high)

            # take action in env:
            next_state, reward, done, info = env.step(action)
            for sub_reward in info:
                if sub_reward.startswith("rw_"):
                    reward_dict[sub_reward] += info[sub_reward]
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            ep_reward += reward

            # if episode is done then update policy:
            if done or t == (max_timesteps - 1):
                if info["goal"]!= 0:
                    done_type = "done_goal"
                elif t == (max_timesteps - 1):
                    done_type = "done_time_up"
                else:
                    for sub_reward in info:
                        if sub_reward.startswith("done_") and info[sub_reward] == 1:
                            done_type = sub_reward
                            break
                policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                break

        print("Episode: {}\tStep: {}k\tReward: {}\tGoal: {} \tDone Type: {} \tEpi_step: {} ".format(episode,int(total_step / 1000),round(ep_reward,2),info["goal"],done_type,ep_step))

        # logging updates:
        writer.add_scalar("reward", ep_reward, global_step=episode)
        writer.add_scalar('goal', info["goal"], global_step=episode)
        for r in reward_dict:
            writer.add_scalar(r, reward_dict[r], global_step=episode)
            reward_dict[r] = 0
        ep_reward = 0
        ep_step = 0
        # save checkpoint:
        if episode % save_rate == 0:
            policy.save(directory, int(total_step / 1000))



if __name__ == '__main__':
    train()
