from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

import config
from util.DataReader import DataReader

def main(unused_argv):
    # Data Params
    file_config = config.File_Config()
    # Set up datareader
    vis_feat_path = './data/vis_feats/%s_ann_vis_feats.pkl' % file_config.dataset
    if os.path.exists(vis_feat_path):
        print('Data prepared.')
    else:
        if ~os.path.exists('./data/vis_feats'):
            os.makedirs('./data/vis_feats/')
        reader = DataReader(file_config)
    
if __name__ == "__main__":
    tf.app.run()