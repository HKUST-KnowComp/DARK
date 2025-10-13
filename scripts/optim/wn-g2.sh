

CUDA_VISIBLE_DEVICES=1 python -m akgr.abduction_model.main \
    --modelname='mydream' \
    --data_root='./sampled_data/' \
    -d='WN18RR' \
    --checkpoint_root='./checkpoints/' \
    --save_frequency 5 \
    --mode='optimizing' \
    --merge_prob=0.0 \
    --explore_ratio=0.5 \
    --deductive_ratio=0.5 \
