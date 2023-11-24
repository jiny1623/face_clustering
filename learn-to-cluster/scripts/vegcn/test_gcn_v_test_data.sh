config=vegcn/configs/cfg_test_gcnv_test.py
load_from=data/pretrained_models/pretrained_gcn_v_ms1m.pth

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

python vegcn/main.py \
    --config $config \
    --phase 'test' \
    --load_from $load_from \
    --save_output \
    --eval_interim \
    --force
