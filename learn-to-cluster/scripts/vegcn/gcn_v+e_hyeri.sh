config1=vegcn/configs/cfg_test_gcnv_hyeri.py
config2=vegcn/configs/cfg_test_gcne_hyeri.py
load_from1=data/pretrained_models/pretrained_gcn_v_ms1m.pth
load_from2=data/pretrained_models/pretrained_gcn_e_ms1m.pth

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

python vegcn/main.py \
    --config $config1 \
    --phase 'test' \
    --load_from $load_from1 \
    --save_output \
    --eval_interim \
    --force

python vegcn/main.py \
    --config $config2 \
    --phase 'test' \
    --load_from $load_from2 \
    --save_output \
    --eval_interim \
    --force
