import copy
import gym
import numpy as np
import torch
from TD3 import TD3
from rsoccer_gym.ssl import *


def match(env_name, number, step_k, max_episode, display):
    print(f"match for {env_name} number {number} {step_k}k model...")
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    args = np.load(f"./models/{env_name}/{number}/args_num_{number}.npy", allow_pickle=True)
    args = dict(args.tolist())
    agent = TD3(args["lr"], state_dim, action_dim, max_action)
    agent.actor.load_state_dict(
        torch.load(f"./models/{env_name}/{number}/{step_k}k_actor.pth"))
    agent.actor.eval()



    goal_num = 0
    opp_num = 0
    done_stats = {}
    for done_stat in env.reward_shaping_total:
        if done_stat.startswith("done_"):
            done_stats[done_stat] = 0
    avg_episode_step = 0
    for episode in range(max_episode):
        obs = env.reset()
        terminate = False
        done = False
        episode_step = 0
        episode_reward = 0

        while not (done or terminate):

            # For each step...
            action = agent.select_action(obs)
            obs_next, reward, done, info = env.step(copy.deepcopy(action))
            # print(action)
            # print(reward)
            if display:
                env.render()
            obs = obs_next

            episode_step += 1
            episode_reward += reward
            if episode_step >= args["max_timesteps"]:
                terminate = True
            # for i in info:
            #     print(i," ",info[i])
        # print(info)
        episode += 1

        avg_reward = episode_reward / episode_step
        # print(f"============epi={episode},avg_reward={avg_reward}==============")
        if info["goal"] == 1:
            goal_num += 1
        elif info["goal"] == -1:
            opp_num += 1

        for done_key in done_stats.keys():
            if info[done_key] == 1:
                done_stats[done_key] += 1

        avg_episode_step += episode_step
    avg_episode_step /= max_episode
    # print("goal", goal_num, "opp_goal", opp_num)
    return goal_num, opp_num, done_stats, avg_episode_step


if __name__ == '__main__':
    number = 7
    step_k = 2093
    max_episode = 1000
    display = True
    goal_num, opp_num, done_stats, avg_episode_step = match('SSLShootEnv-v0-v0',number, step_k, max_episode, display)
    print("goal ", goal_num,
          "opp_goal ", opp_num,
          "avg_episode_step ", avg_episode_step)
    for d in done_stats:
        print(d," ",done_stats[d])