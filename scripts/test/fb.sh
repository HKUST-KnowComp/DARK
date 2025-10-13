

CUDA_VISIBLE_DEVICES=3 python -m akgr.abduction_model.main \
    --modelname='mydream' \
    --data_root='./sampled_data/' \
    -d='FB15k-237' \
    --checkpoint_root='./checkpoints/' \
    --save_frequency 5 \
    --mode='testing' \
    --training_mode='sft' \
    --merge_prob=0.5 \
    --attention_all
