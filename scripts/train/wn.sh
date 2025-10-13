

CUDA_VISIBLE_DEVICES=0 python -m akgr.abduction_model.main \
    --modelname='mydream' \
    --data_root='./sampled_data/' \
    -d='WN18RR' \
    --checkpoint_root='./checkpoints/' \
    --save_frequency 5 \
    --mode='training' \
    --training_mode='unify' \
    --merge_prob=0.0 \
    --attention_all
