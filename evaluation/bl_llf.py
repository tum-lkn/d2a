from tx_assignment.rl_tx_assignment import ONVMMultiAgent
import numpy as np

config = {}
config["env_config"]= {
        "tx_cap": 8.4e6,
        "machine_count": 21,
        "cycles_per_second": 2.2e9,
        "num_of_sfcs": 8,
        "max_vnfs_per_sfc": 8,
        "max_vnfs": 40,
        "load_level": 0.9,
        "rate": 5e6}
single_env = ONVMMultiAgent(config["env_config"])
random = np.random.RandomState(seed=1)

all_actions = []
all_results = []

for i in range(100):
    results = []
    action_buffer = []
    obs = single_env.reset()
    done = {'__all__': False} 

    while done['__all__'] == False:
        action_dict = {
            k: 1 for k in obs.keys()
        }
        action_buffer.append(action_dict)
        obs, rew, done, info = single_env.step(action_dict)
        results.append(rew)

    all_actions.append(action_buffer)
    all_results.append(results)

episode_agent_reward = []
episode_reward = []

for results in all_results:
    rewards = {}
    x = 0
    for i, result in enumerate(results):
        if i == 0:
            for k in result.keys():
                rewards[k] = 0

        for agent, reward in result.items():
            rewards[agent] += reward
            x += reward
    episode_reward.append(x)

    for reward in rewards.values():
        episode_agent_reward.append(reward)


## Least loaded First baseline
