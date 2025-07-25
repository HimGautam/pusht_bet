# BET_LEROBOT

> **Note:** This repository bundles the original LeRobot codebase (unmodified) by the LeRobot team, with credit given below. All core functionality and implementation belong to the LeRobot maintainers.

## Credit
- Original LeRobot project: [https://github.com/lerobot/lerobot ](https://github.com/huggingface/lerobot) 
- Upstream license: [Apache License 2.0](https://github.com/huggingface/lerobot?tab=Apache-2.0-1-ov-file#readme)
- Copyright © LeRobot Team


- We also use [nanoGPT](https://github.com/karpathy/nanoGPT) project by Andrej Karapathy in this work.

## My Code
In this repo you'll find:
- `src/lerobot/policies/bet` - my custom policy files
- `environment.yml` - exact conda environment spec I used in this project
- `Makefile` - for installing and running the package locally (from the Lerobot Team)

## Repository Structure
```
src/
└── lerobot/
    └── policies/
        ├── bet/      ← My implementation
        └── …         ← other policies

```

## Getting Started

### 1. Create Conda environment

Create a conda environment with `environment.yml` file.
```bash
conda env create -f environment.yml
conda activate lerobot
```

or you can create a conda new environment with `python 3.10` and install required dependencies.

### 2. Install Lerobot

Clone the current repository. Navigate to the root directory of this repo `cd BET_LEROBOT` and install lerobot with following command:

```
pip install -e ".[pusht]"
```

To test out the installation try:

```
python -c "import lerobot; print(lerobot.__version__)"
```

### 3. Behavior Transformer


This project is implementation of [Behavior Transformer](https://arxiv.org/abs/2206.11251) paper on the Pusht Environment. 

Behavior transformer tries to solve the problem of  multi-modal demonstration data for behavior cloning. 
- From the demonstration dataset, we cluster the actions into K action centers using K-means.
- Every continuous action is then split into a bin (nearest action center) and a residual (small offset from that center).
- The policy network predicts both action bin (trained with Focal loss) and residual offset (trained with Masked Multi-task loss)

For full details, see the original paper.


### 4. My Implementation

This project is built on top of the LeRobot framework. I used the PushT environment and the pusht_image dataset—both provided by LeRobot.

My custom policy is implemented under:
`src/lerobot/policies/bet/`

The implementation is split into two main files:

- `modeling_bet.py`: Contains the full neural network architecture, the forward() function used during training, and the select_action() method used during inference.

- `configuration_bet.py` :Stores all the hyperparameters (e.g. learning rate, model size, number of decoder layers, number of bins), and also defines how the LeRobot CLI loads and runs the policy.

<p align="center">
<img width="783" height="212" alt="Bet" src="https://github.com/user-attachments/assets/5b60b307-b2ae-4246-be6c-d7936127ef1f" />
</p>

### 5. Training

To train the Behavior Transformer policy, I leveraged LeRobot’s built‑in train.py script. Simply run:
```
python -m lerobot.scripts.train \
  --policy.type behavior_transformer \
  --dataset.repo_id     lerobot/pusht_image \
  --env.type            pusht \
  --env.obs_type        pixels_agent_pos \
  --output_dir          ./bet_run \
  --steps               20000 \
  --batch_size          64 \
  --log_freq            200 \
  --save_freq           1000 \
  --eval_freq           1000 \
  --eval.n_episodes     10 \
  --eval.batch_size     10 \
  --policy.push_to_hub  false \     #for training the policy locally
  --wandb.enable        true \
  --wandb.project       pusht_bet

```

![bet_pusht](https://github.com/user-attachments/assets/7c8f5193-5dfd-4ed2-8e1f-89d0dc59c253)


#### Hyperparams
You can also try tweaking learning rate, decoder layers, lambda_bin, etc., by editing configuration_bet.py.

#### Custom action bins

If you change num_bins, first regenerate the centroids with

```
python src/lerobot/policies/bet/compute_pushT_bins.py 
```
This will produce the .pt file your policy needs.

#### Run from repo root
Always execute the training command from the top‑level BET_LEROBOT directory to avoid path errors.
