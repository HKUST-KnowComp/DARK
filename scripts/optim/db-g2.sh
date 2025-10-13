

CUDA_VISIBLE_DEVICES=3 python -m akgr.abduction_model.main \
    --modelname='mydream' \
    --data_root='./sampled_data/' \
    -d='DBpedia50' \
    --checkpoint_root='./checkpoints/' \
    --save_frequency 5 \
    --mode='optimizing' \
    --merge_prob=0.0 \
    --explore_ratio=1.0 \
    --deductive_ratio=0.0 \
