# DARK


This is the official code repository for **Unifying Deductive and Abductive Reasoning in Knowledge Graphs with Masked Diffusion Model** (WWW 2026).

# Environment

```bash
conda create -n dark python=3.9
conda activate dark
pip install -r requirements.txt 
```

# Training

You can run the code in the following steps:

1. Sampling
2. Supervised training
3. Reinforcement learning

## Step 1: Sampling

```bash
bash scripts/sample/sample_full.sh
```

## Step 2: Supervised training
For the first-stage pretraining, set --training_mode unify.
For the second-stage supervised training for a single reasoning type, set --training_mode sft.

```bash
bash scripts/train/db.sh
```

or training with multi-gpu:

```bash
bash scripts/train/db-multi.sh
```



## Step 3: Reinforcement learning

Example scripts:

```bash
bash scripts/optim/db.sh
```





# Citation

Welcome to cite our work!

@article{gao2025unifying,
  title={Unifying Deductive and Abductive Reasoning in Knowledge Graphs with Masked Diffusion Model},
  author={Gao, Yisen and Bai, Jiaxin and Huang, Yi and Fu, Xingcheng and Sun, Qingyun and Song, Yangqiu},
  journal={arXiv preprint arXiv:2510.11462},
  year={2025}
}
