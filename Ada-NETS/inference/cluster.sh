#!/bin/bash
# set -euxo pipefail
set +x

cd GCN

outpath=outpath
featfile=outpath/fcfeat_ckpt_35000.npy
tag=fc
python -W ignore ../tool/faiss_search.py $featfile $featfile $outpath $tag

test_labelfile=../data/label/test.npy
Ifile=outpath/fcI.npy
Dfile=outpath/fcD.npy
filelist=../../filelist.txt 
python cluster.py $Ifile $Dfile $test_labelfile $filelist
