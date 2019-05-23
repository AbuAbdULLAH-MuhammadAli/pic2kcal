import json
from sklearn.model_selection import train_test_split
import os
import numpy as np
from random import sample
import pandas as pd


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

for dataset, dsname in ((train, 'train'), (val, 'val'), (test, 'test')):
    for idx, row in dataset.iterrows():
        for i, img_name in enumerate(row['picture_files']):
            # print(img_name)
            src = '../img/'+img_name
            dest = dsname+'/'+row['id']+'_'+str(i)+'.jpg'
            os.symlink(src, dest)

