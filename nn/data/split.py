import json
from sklearn.model_selection import train_test_split
import os
import numpy as np
from random import sample
import pandas as pd
from collections import defaultdict


# read json
with open('per_portion_data.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# train, test and val split
train, val, test = np.split(df.sample(frac=1), [int(.7*len(df)), int(.85*len(df))])

# save selected receipe names into distinct subfolders)
os.makedirs('train', exist_ok=True)
os.makedirs('val', exist_ok=True)
os.makedirs('test', exist_ok=True)

output = defaultdict(list)

for dataset, dsname in ((train, 'train'), (val, 'val'), (test, 'test')):
    for idx, row in dataset.iterrows():
        for i, img_name in enumerate(row['picture_files']):
            # print(img_name)
            name = row['id']+'_'+str(i)+'.jpg'
            output[dsname].append({'name': name, 'kcal': row['kcal_per_portion']})
            src = '../img/'+img_name
            dest = dsname+'/'+name
            os.symlink(src, dest)
    with open(dsname+'.json', 'w') as file:
        json.dump({'data': output[dsname]}, file)
