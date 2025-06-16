from amatrix_to_env import Environment
import numpy as np

M = np.array([[0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 6, 5, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 9, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

env = Environment(M, run_DP=True)


"""
print(env.optimal_path)
print(env.optimal_policy)

env.solve()

print(env.optimal_path)
print(env.optimal_policy)
 """


 
""" from random import randint
for i in range(20):
    action = randint(0,1)
    observation, reward, terminated = env.step(action)
    print("step number {}:".format(i+1))
    print("action: {}".format(action))
    print("obervation: {}".format(observation))
    print("reward: {}".format(reward))
    print("terminated: {}\n".format(terminated))
    
    if terminated:
        env.reset()
        print("environment reset")
 """
 
terminated = False
env.reset()
current_state = env.current_state
policy = env.optimal_policy
total_reward = 0

while not terminated:
    action = int(policy[current_state.index])
    _, reward, terminated = env.step(action)
    current_state = env.current_state
    total_reward += reward

print(total_reward)