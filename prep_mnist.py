# Derived from https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook

import struct
import sys


features_path = sys.argv[1]
labels_path = sys.argv[2]


with open(features_path, 'rb') as f:
    features_data = f.read()


with open(labels_path, 'rb') as f:
    labels_data = f.read()


magic, labels_size = struct.unpack('>II', labels_data[:8])
assert magic == 2049


magic, features_size, rows, cols = struct.unpack('>IIII', features_data[:16])
assert magic == 2051
assert rows == 28
assert cols == 28


assert labels_size == features_size


label_it = struct.iter_unpack('B', labels_data[8:])
pixel_it = struct.iter_unpack('784B', features_data[16:])

for i, (label, pixel_data) in enumerate(zip(label_it, pixel_it)):
    print(f'{i=} {label=} {pixel_data=}')
    if i > 20:
        break
