import os
import sys
import json
from copy import deepcopy


in_paths, out_path = sys.argv[1:-1], sys.argv[-1]

all_lines = []
for path in in_paths:
    with open(path, 'r') as fin:
        lines = fin.readlines()
    all_lines.append(lines)


fout = open(out_path, 'w')

for i in range(len(all_lines[0])):
    item = deepcopy(json.loads(all_lines[0][i]))
    preds, confs = [], []
    for lines in all_lines:
        cur_item = json.loads(lines[i])
        assert cur_item['id'] == item['id']
        preds.append(cur_item['pred'])
        confs.append(cur_item['confidence'])
    item['pred'] = sum(preds) / len(preds)
    item['confidence'] = sum(confs) / len(confs)
    fout.write(json.dumps(item) + '\n')

fout.close()
