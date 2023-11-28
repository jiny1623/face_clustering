import os
from pathlib import Path
import sys
import json

filelist_path = sys.argv[1]
pred_labels = sys.argv[2]
output_path = 'output_gcn_v+e.json'

# with open(json_path) as f:
#     data = json.load(f)

# # print(data[0]['img_path'])

# with open(output_path, 'w') as output_file:
#     for i in range(len(data)):
#         output_file.write(data[i]['img_path'] + '\n')
img_path = []
cluster_id = []

with open(filelist_path, 'r') as f:
    for line in f:
        img_path.append(line.strip())

with open(pred_labels, 'r') as l:
    for line in l:
        cluster_id.append((int)(line.strip()))

# print(img_path)
# print(cluster_id)

output = []

assert len(img_path) == len(cluster_id)

for i in range(len(img_path)):
    img = {'img_path': img_path[i], 'cluster_id': cluster_id[i]}
    output.append(img)

with open(output_path, 'w') as json_file:
    json.dump(output, json_file, indent=4)

# print(output)