# Face Clustering 추론 코드
## 입/출력 양식
* 입력: 각 이미지에 대해 face detection 모델이 추론한 Face Patch 모양대로 crop된 img
```json
[
    {
        "img_path": "/home/rippleai/data/video1/0001.png"
    },

    ...
]
```

* 출력: cluster id 정보가 추가된 json
```json
[
    {
        "img_path": "/home/rippleai/data/video1/0001.png", 
		"cluster_id": 3
    },

    ...
]
```

## GCN_V+E

0-1. 입력 json 파일을 통해 filelist.txt 를 만들어줍니다.

```bash
python input_totxt.py input.json
```

0-2. [GoogleDrive](https://drive.google.com/file/d/1eKsh7x-RUIHhIJ1R9AlUjsJdsdbh2qim/view?pli=1) 에서 Pretrained Model을 다운받아, `hfsoftmax/ckpt`에 넣어줍니다.

0-3. GCN-V+E의 Pretrained Model을 다운받아 줍니다.

```bash
cd learn-to-cluster
python tools/download_data.py
```

1. Face Recognition

```bash
conda create -n recognition python=3.6
conda activate recognition
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=9.2 -c pytorch
cd hfsoftmax
pip install -r requirements.txt

# arguments : prefix file_list ckpt_path output_path
sh ./scripts/extract_feat.sh prefix ../filelist.txt ckpt/resnet50_part0_train.pth.tar ../learn-to-cluster/data/features/test.bin
```

2. Clustering

```bash
conda create -n gcn_v+e python=3.7
conda activate gcn_v+e
pip install torch==1.2.0 torchvision==0.4.0
pip install faiss-gpu
cd ../learn-to-cluster
pip install -r requirements.txt
pip install pillow==6.2.1
pip install cmake
pip install -U openmim
mim install mmcv==1.3.3

# config : vegcn/configs/cfg_test_gcnv.py
# config의 test_name은 Face_Recognition에서 output_path의 bin file 이름과 동일해야 합니다. 
sh scripts/vegcn/test_gcn_v.sh
# config : vegcn/configs/cfg_test_gcne.py
sh scripts/vegcn/test_gcn_e.sh
```

3. save된 label을 통하여 output_gcn_v+e.json 을 생성합니다.

```bash
cd ..
# config : filelist.txt pred_labels.txt
python construct_output.py filelist.txt learn-to-cluster/data/work_dir/cfg_test_gcne/test_gcne_k_160_th_0.0_ig_0.8/tau_0.8_pred_labels.txt
```

## Ada-NETS

0~1 과정은 GCN_V+E와 동일합니다.
(다른 점은, 0-3 과정과, 마지막에 extract_feat.sh 실행 시 output_path를 알맞게 수정해주는 것 이외엔 없습니다.)

ex. `sh ./scripts/extract_feat.sh prefix ../filelist.txt ckpt/resnet50_part0_train.pth.tar ../Ada-NETS/data/feature/test.bin`

0-1. 입력 json 파일을 통해 filelist.txt 를 만들어줍니다.

```bash
python input_totxt.py input.json
```

0-2. [GoogleDrive](https://drive.google.com/file/d/1eKsh7x-RUIHhIJ1R9AlUjsJdsdbh2qim/view?pli=1) 에서 Pretrained Model을 다운받아, `hfsoftmax/ckpt`에 넣어줍니다.

0-3. Notion에 공유 드린 `Ada-NETS` 폴더를 대체합니다. 이 폴더에는 미리 training 해 둔 파일들이 `data/`, `AND/outpath/`, `GCN/outpath/` 에 존재합니다. 이를 통해 training 과정 없이 바로 inference 할 수 있게 해두었습니다.

1. Face Recognition

```bash
conda create -n recognition python=3.6
conda activate recognition
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=9.2 -c pytorch
cd hfsoftmax
pip install -r requirements.txt

# arguments : prefix file_list ckpt_path output_path
sh ./scripts/extract_feat.sh prefix ../filelist.txt ckpt/resnet50_part0_train.pth.tar ../Ada-NETS/data/feature/test.bin
```


2. Clustering

```bash
conda create -n adanets python=3.6 -y
conda activate adanets
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
cd Ada-NETS
rm -rf apex
git clone https://github.com/NVIDIA/apex --branch 22.03 --single-branch
cd apex
python setup.py install
cd ..
pip install -r requirements.txt

# ada-nets allows .npy file as input
# bin_to_npy.py는 (inference 시에 필요한) dummy label도 같이 생성해줍니다.
python bin_to_npy.py ./data/feature/test.bin

# 각 shell file의 featfile과 labelfile을 알맞게 수정한 다음 실행합니다. 현재는 test.npy로 되어있습니다.
sh inference/faiss_search.sh

sh inference/structure_space.sh

sh inference/max_Q_ind.sh

sh inference/test_AND.sh

sh inference/gene_adj.sh

sh inference/test_GCN.sh

# output_adanets.json이 저장됩니다.
sh inference/cluster.sh
```


### Debugging

1. gcn_v+e 적용 시 config에서 knn이, node의 개수 (=input jpg의 개수) 보다 많으면 오류가 발생합니다.
2. gcn_v+e 에서 `AssertionError: 0 vs 1` 가 뜨는 경우는 주로 input jpg의 개수가 적을 때 `ignore_ratio=0.8` 에 의하여 모든 vertices가 무시되는 경우입니다. `ignore_ratio=0`으로 config file을 수정하시면 잘 작동합니다.

