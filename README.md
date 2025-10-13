# CtrlHGen
This is the code repo for *Controllable Logical Hypothesis Generation for Abductive Reasoning in Knowledge Graphs*

# Environment

```bash
conda create -n ctrlhgen python=3.9
conda activate ctrlhgen
pip install -r requirements.txt 
```

# Training

As described in the paper, you can run the code in the following steps:

1. Sampling
2. Supervised training
3. Reinforcement learning

## Step 1: Sampling

```bash
bash scripts/sample/sample_full.sh
```

## Step 2: Supervised training

1. Without condition

```bash
bash scripts/train/wn-g2.sh
```

2. With condition

```bash
bash scripts/cond-train/wn-g2-pattern.sh
```



## Step 3: Reinforcement learning

Example scripts:

```bash
bash scripts/optim/wn-g2.sh
```

For training with multi-gpu:

```bash
bash scripts/optim/wn-g2-multi.sh
```



# Evaluation

Example scripts:

```bash
bash scripts/test/wn-g2.sh
```

```bash
bash scripts/optim-test/wn-g2.sh
```



# Citation

Welcome to cite our work!
