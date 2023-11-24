import os
from pathlib import Path
import sys
import json

json_path = sys.argv[1]
output_path = 'filelist.txt'

with open(json_path) as f:
    data = json.load(f)

# print(data[0]['img_path'])

with open(output_path, 'w') as output_file:
    for i in range(len(data)):
        output_file.write(data[i]['img_path'] + '\n')
