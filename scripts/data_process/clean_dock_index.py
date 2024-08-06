import sys
import ast

path = sys.argv[1]

with open(path, 'r') as fin:
    lines = fin.readlines()

new_lines = []
for line in lines:
    if 'pos' in ast.literal_eval(line.split('\t')[-2]): new_lines.append(line)
    else: print(line)


print(f'Original: {len(lines)}, new: {len(new_lines)}')
with open(path + '.cleaned', 'w') as fout:
    fout.writelines(new_lines)