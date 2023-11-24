# On 1 TitanX, it takes around 22 hours for training
# metircs (pre, rec, pairwise fscore)
# test on 2 proposal params (gt iop, 0.2-0.8): (99.1, 67.29, 80.16)
# test on 20 proposal params (gt iop, 0.2-0.8): (97.83, 81.2, 88.74)

import os.path as osp
from functools import partial
from proposals import generate_proposals

use_random_seed = True
featureless = False

# model
model = dict(type='dsgcn',
             kwargs=dict(feature_dim=256,
                         featureless=featureless,
                         stage='seg',
                         reduce_method='no_pool',
                         use_random_seed=use_random_seed,
                         hidden_dims=[512, 64]))

# training args
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer_config = {}

lr_config = dict(
    policy='step',
    step=[15, 24, 28],
)

iter_size = 32
batch_size_per_gpu = 1
test_batch_size_per_gpu = 256
total_epochs = 30
workflow = [('train', 1)]

# misc
workers_per_gpu = 1

checkpoint_config = dict(interval=1)

log_level = 'INFO'
log_config = dict(interval=100, hooks=[
    dict(type='TextLoggerHook'),
])

# post_process
th_pos = -1
th_iou = 1
th_outlier = 0.5
keep_outlier = True

# testing metrics
metrics = ['pairwise', 'bcubed', 'nmi']

# data locations
prefix = './data'
train_name = 'part0_train'
test_name = 'part1_test'

th_iop_min=0.2
th_iop_max=0.8

knn_method = 'hnsw'

k_lst_i0 = [30, 60, 80]
step_i0 = 0.05
minsz_i0 = 3
maxsz_i0 = 300
th_lst_i0 = [0.6, 0.65, 0.7, 0.75]

k_th_lst_i0 = []
for k in k_lst_i0:
    for th in th_lst_i0:
        k_th_lst_i0.append((k, th))

th_i1 = 0.4
step_i1 = 0.05
minsz_i1 = 3
maxsz_i1 = 500
sv_minsz_i1 = 2
k_maxsz_lst_i1 = [(2, 8), (2, 12), (2, 16), (3, 5), (3, 10), (4, 4)]

proposal_params = [
    dict(k=k_i0,
         knn_method=knn_method,
         th_knn=th_i0,
         th_step=step_i0,
         minsz=minsz_i0,
         maxsz=maxsz_i0,
         iter1_params=[
             dict(k=k_i1,
                  knn_method=knn_method,
                  th_knn=th_i1,
                  th_step=step_i1,
                  minsz=minsz_i1,
                  maxsz=maxsz_i1,
                  sv_minsz=sv_minsz_i1,
                  sv_maxsz=sv_maxsz_i1) for k_i1, sv_maxsz_i1 in k_maxsz_lst_i1
         ]) for k_i0, th_i0 in k_th_lst_i0
]
feat_path = osp.join(prefix, 'features', '{}.bin'.format(train_name))
label_path = osp.join(prefix, 'labels', '{}.meta'.format(train_name))
proposal_path = osp.join(prefix, 'cluster_proposals')
train_data = dict(wo_weight=False,
                  feat_path=feat_path,
                  label_path=label_path,
                  use_random_seed=use_random_seed,
                  featureless=featureless,
                  th_iop_min=th_iop_min,
                  th_iop_max=th_iop_max,
                  proposal_folders=partial(generate_proposals,
                                           params=proposal_params,
                                           prefix=prefix,
                                           oprefix=proposal_path,
                                           name=train_name,
                                           dim=model['kwargs']['feature_dim'],
                                           no_normalize=False))

step_i0 = 0.05
minsz_i0 = 3
maxsz_i0 = 300
th_iter0_lst_i0 = [(0.6, True), (0.7, True), (0.75, False)]

th_i1 = 0.4
step_i1 = 0.05
minsz_i1 = 3
maxsz_i1 = 500
sv_minsz_i1 = 2
k_maxsz_lst_i1 = [(2, 8), (2, 12), (2, 16), (3, 5), (3, 10), (4, 4)]

proposal_params = [
    dict(k=k,
         knn_method=knn_method,
         th_knn=th_i0,
         th_step=step_i0,
         minsz=minsz_i0,
         maxsz=maxsz_i0,
         iter0=iter0,
         iter1_params=[
             dict(k=k_i1,
                  knn_method=knn_method,
                  th_knn=th_i1,
                  th_step=step_i1,
                  minsz=minsz_i1,
                  maxsz=maxsz_i1,
                  sv_minsz=sv_minsz_i1,
                  sv_maxsz=sv_maxsz_i1) for k_i1, sv_maxsz_i1 in k_maxsz_lst_i1
         ]) for th_i0, iter0 in th_iter0_lst_i0
]
# use the same proposal params as training
feat_path = osp.join(prefix, 'features', '{}.bin'.format(test_name))
label_path = osp.join(prefix, 'labels', '{}.meta'.format(test_name))
test_data = dict(wo_weight=True,
                 feat_path=feat_path,
                 label_path=label_path,
                 use_random_seed=use_random_seed,
                 use_max_degree_seed=True,
                 featureless=featureless,
                 th_iop_min=th_iop_min,
                 th_iop_max=th_iop_max,
                 proposal_folders=partial(generate_proposals,
                                          params=proposal_params,
                                          prefix=prefix,
                                          oprefix=proposal_path,
                                          name=test_name,
                                          dim=model['kwargs']['feature_dim'],
                                          no_normalize=False))
