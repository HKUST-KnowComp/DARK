python -m akgr.sampling.sample_add \
    --modelname='GPT2_6_act_nt'\
    --data_root='./sampled_data/' -d='DBpedia50' --scale='full' -a=32  \
    --checkpoint_root='checkpoints/'\
    --result_root='./results/'\
    --save_frequency 1\
    --mode='training'