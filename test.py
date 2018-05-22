from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import tqdm

import config
from util.DataReader import DataReader
from vc_model import VC_Model
from util import ciou

thresh_iou = 0.5

def main(unused_argv):
    # Initialize configs and parameters    
    model_config = config.Model_Config()
    file_config = config.File_Config()
    checkpoint = file_config.checkpoint
#    assert os.path.exists(checkpoint + '.*'), "checkpoint doesn't exist."

    # Set up model
    model = VC_Model(model_config, mode='eval')
    model.build()
    
    # Set up datareader
    reader = DataReader(file_config)

    # Set up session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())
    
    # Load parameters from checkpoint
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    
    # Set up evaluation splits
    if file_config.dataset in ['refcoco', 'refcoco+']:
        test_split = ['testA', 'testB']
    elif file_config.dataset == 'refcocog':
        test_split = ['val']  
    
    for split in test_split:
        num_correct = 0
        num_total = 0
        num_batch = reader.num_batch[split]
        for n_iter in tqdm.trange(num_batch):
            # Read one batch
            batch = reader.get_batch(split=split, shuffle=False, echo=False)   
            text_zseq = batch['text_zseq_batch']
            vis_feat = batch['vis_batch']
            visdif_feat = batch['visdif_batch']
            spa_feat    = batch['spa_batch']
            label_val   = batch['label_batch']

            scores_val = sess.run(model.scores,
                feed_dict={model.text_seqs:text_zseq,
                       model.region_vis_feat:vis_feat,
                       model.region_visdif_feat:visdif_feat,
                       model.region_spatial_feat:spa_feat
                       })

            predicts = np.argmax(scores_val, axis=1)
            labels = batch['label_batch']

            for i in range(len(labels)):
                gt_bbox = batch['coco_bboxes'][labels[i]]
                pred_bbox = batch['coco_bboxes'][predicts[i]]
                iou = ciou.iou_bboxes(pred_bbox, gt_bbox)
                if iou >= thresh_iou:
                    num_correct += 1
            num_total += len(labels)

        accuracy = num_correct/num_total
        print("%s overall_accuracy: %f" % (split, accuracy))
        
if __name__ == "__main__":
    tf.app.run()