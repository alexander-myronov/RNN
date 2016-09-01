from collections import OrderedDict
import json
import numpy as np

___author__ = 'amyronov'

with open(r'data/cache/filenames.json', 'r') as f:
            filenames = json.load(f, object_pairs_hook=OrderedDict)

# datasets = OrderedDict()
for name, (X_filename, y_filename) in filenames.iteritems():
    def loader():
        return np.load(X_filename+'.npy', mmap_mode='r'), \
                     np.load(y_filename+'.npy', mmap_mode='r')
    X, y = loader()
    print(name, (y==1).sum(), (y==-1).sum(), len(y))
