# DARK
This is the code repo for *Unifying Deductive and Abductive Reasoning in Knowledge Graphs with Masked Diffusion Model*

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
