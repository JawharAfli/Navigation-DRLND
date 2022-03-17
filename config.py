BANANA_ENVIRONMENT = "/home/jawhar/Desktop/udacity/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64"
DEVICE = "cuda:0"

N_EPISODES = 1400
MAX_T = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4
