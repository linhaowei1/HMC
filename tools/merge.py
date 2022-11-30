from tqdm.auto import tqdm
import os
import json
path = [
    './log/2021_epoch0',
    './log/2021_epoch1',
    './log/2021_epoch2',
    './log/111_epoch0',
    './log/111_epoch1',
    './log/111_epoch2',
    './log/222_epoch0',
    './log/222_epoch1',
    #'./log/222_epoch2',
    './log/333_epoch0',
    './log/333_epoch1',
    #'./log/333_epoch2',
]
DATA = []
for f_path in path:
    with open(os.path.join(f_path, 'pred'), 'r') as f:
        data = json.load(f)
    DATA.append(data)

out = []
for i in tqdm(range(len(DATA[0]))):
    comp = [DATA[j][i] for j in range(len(DATA))]
    out.append(max(comp, key=comp.count))

with open('./log/results.txt','w') as f:
    for line in out:
        f.write(str(line['aydj']) + ' ' + line['ay'] + '\n')
