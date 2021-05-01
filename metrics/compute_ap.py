import numpy as np
import argparse
import os 
import errno

try:
    os.mkdir('metrics/average_precision/')
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

parser = argparse.ArgumentParser(
    description='Compute Average Precision Per Class')
parser.add_argument('--conf_mat_path', type=str)
parser.add_argument('--fog_intensity', type=str)
args = parser.parse_args()
conf_mat = np.loadtxt(args.conf_mat_path, dtype=float, delimiter=',')

class_ap = np.zeros((conf_mat.shape[0], ))
class_ids = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation',
             'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motor cycle', 'bicycle', 'unlabeled']

f_ = open('metrics/average_precision/avg_precision_fog_intensity_' + args.fog_intensity + '.txt', 'w+')
for i in range(conf_mat.shape[0]):
    class_ap[i] = conf_mat[i, i]
    row_sum = np.sum(conf_mat[i])
    class_ap[i] /= row_sum
    f_.write(class_ids[i] + " " + str(class_ap[i]) + '\n')

f_.close()