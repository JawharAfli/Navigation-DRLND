import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment
from config import (
    BANANA_ENVIRONMENT,
    DEVICE,
    EPS_DECAY,
    EPS_END,
    EPS_START,
    MAX_T,
    N_EPISODES,
)

from dqn_agent import DQNAgent
from evaluate import evaluate
from qnetwork import QNetwork
from train import dqn

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    device = torch.device(DEVICE)

    env = UnityEnvironment(file_name=BANANA_ENVIRONMENT)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0)

    if args.train:

        scores = dqn(
            env=env,
            brain_name=brain_name,
            agent=agent,
            n_episodes=N_EPISODES,
            max_t=MAX_T,
            eps_start=EPS_START,
            eps_end=EPS_END,
            eps_decay=EPS_DECAY,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel("Score")
        plt.xlabel("Episode")
        plt.show()

    if args.eval:

        model = QNetwork(state_size, action_size, 0).to(device)
        model.load_state_dict(torch.load("model.pth"))
        model.eval()

        evaluate(env, brain_name, model, device, n_episodes=100, max_t=1000)
