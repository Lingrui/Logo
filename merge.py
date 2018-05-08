#!/usr/bin/env python3
import os
import sys
import simplejson as json
import numpy as np
from glob import glob
import cv2
import picpac
print(picpac.__file__)

def merge (path, dbout, label):
    db = picpac.Reader(path)
    for rec in db:
        anno = json.loads(rec.fields[1].decode('ascii'))
        for shape in anno['shapes']:
            shape['label'] = label
            pass
        dbout.append(label, rec.fields[0], json.dumps(anno).encode('ascii'))
        pass
    pass

def merge_dir (path, dbout, label):
    for p in glob(path + '/*.png'):
        print(p)
        with open(p, 'rb') as f:
            dbout.append(label, f.read())
            pass
        pass



db = picpac.Writer('db/all', picpac.OVERWRITE)

for i in [1,2,3,4,5,6]:
    merge('db/%d.db'% i, db, float(i))
    pass

merge_dir('/shared/s2/users/lcai/VideoAd/db/0', db, 0)


