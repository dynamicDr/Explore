import os
import re
import match


def extract_step_numbers(env_name,exp_number):
    step_numbers = []
    for file_name in os.listdir(f"./models/{env_name}/{exp_number}"):
        match = re.search("(\d+)._actor.pth", file_name)
        if match:
            step_numbers.append(int(match.group(1)))
    return sorted(step_numbers)

if __name__ == '__main__':
    number = 0
    max_episode = 100
    display = False
    env_name = "SSLPassEnv-v0"
    exp_numbers = extract_step_numbers(env_name,number)

    os.makedirs(f"./evaluate/{env_name}", exist_ok=True)
    if os.path.exists(f"./evaluate/{env_name}/output_{number}.txt"):
        os.remove(f"./evaluate/output_{number}.txt")
    for exp in exp_numbers:
        goal_num, opp_num, done_stats, avg_episode_step = match.match(env_name,number, exp,max_episode,display)
        with open(f'./evaluate/output_{number}.txt', 'a+', encoding='utf-8', errors='ignore') as f:
            text = f"\nexp_x_k_step {exp}\ntest_match_number {max_episode}\ngoal {goal_num}\nopp_goal {opp_num}\navg_episode_step {avg_episode_step}\n"
            f.write(text)
            for d in done_stats:
                text = f"{d} {done_stats[d]}\n"
                f.write(text)
            f.write("================================")