# Navigation Project- DRL NanoDegree

This project is part of Udacity's Deep Reinforcement Learning NanoDegree. Its main objective is to train an agent to collect yellow bananas while avoiding the blue ones using Unity's ML-agents environment.

![env](banana.gif)

## Installation

1. Clone the repository.
```
git clone https://github.com/JawharAfli/Navigation-DRLND.git

cd Navigation-DRLND
```

2. Prepare the environment.
```
conda env create -f environment.yml

conda activate drlnd

pip install torch==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Download the Unity ML-Agents Banana Environment.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

4. Include the path of the banana environment into the configuration file `config.py`.

## Train your banana navigator

```
python banana_navigator.py --train
```

## Evaluate your banana navigator

```
python banana_navigator.py --eval
```
