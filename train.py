from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

import config
from util.DataReader import DataReader
from vc_model import VC_Model

max_iter = 180000

def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        print('Creating %s' % dir_path)

def init_word_embed(config):
    """Initialize word embedding matrix."""
    embedding_mat_val = np.load(config.wordembed_params)
    with tf.variable_scope('vc'):
        with tf.variable_scope('lstm', reuse=True):
            embedding_mat = tf.get_variable("embedding_mat", [config.num_vocab, config.embed_dim])
            init_we = tf.assign(embedding_mat, embedding_mat_val)
    return [init_we]

def main(unused_argv):
    # Initialize configs and parameters    
    model_config = config.Model_Config()
    file_config = config.File_Config()
    max_iter = model_config.max_iter
    snapshot_dir = file_config.snapshot_dir
    snapshot_file = file_config.snapshot_file
    snapshot_start = file_config.snapshot_start
    snapshot_interval = file_config.snapshot_interval
    log_dir = file_config.log_dir
    log_interval = file_config.log_interval

    # Set up model
    model = VC_Model(model_config, mode='train')
    model.build()
    
    # Set up datareader
    reader = DataReader(file_config)

    # Set up session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    # Initialize word embedding matrix.
    init_we = init_word_embed(file_config)
    sess.run(*init_we)
       
    # Check whether log and snapshot directories exist.
    check_dir(log_dir)
    check_dir(snapshot_dir)

    # log writer
    log_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    # snapshot saver
    snapshot_saver = tf.train.Saver()
    
    cls_loss_avg = 0
    acc_avg = 0
    initial_iter = sess.run(model.global_step)
    for n_iter in range(initial_iter, max_iter):
        # Read one batch
        batch = reader.get_batch(split='train')
        text_seq  = batch['text_zseq_batch']
        im_id       = batch['im_id']
        vis_feat      = batch['vis_batch']
        visdif_feat    = batch['visdif_batch']
        spa_feat      = batch['spa_batch']
        label_val    = batch['label_batch']
       
        print('\tthis batch: image %d, with %d sentences x %d proposal boxes = %d scores' %
                (im_id, text_seq.shape[1], vis_feat.shape[0],
                text_seq.shape[1]*vis_feat.shape[0]))
      
        scores_trn, cls_loss, reg_loss, acc, summary, _ = \
                sess.run([model.scores, model.cls_loss, model.reg_loss, model.accuracy,
                         model.summary_op, model.ops], 
                             feed_dict={model.text_seqs:text_seq,
                             model.region_vis_feat:vis_feat,
                             model.region_visdif_feat:visdif_feat,
                             model.region_spatial_feat:spa_feat,
                             model.labels:label_val
                             })
        
        cls_loss_avg = 0.99*cls_loss_avg + 0.01*cls_loss
        acc_avg = 0.99*acc_avg + 0.01*acc
        print('\titer = %d, cls_loss (cur) = %f, cls_loss (avg) = %f, acc (cur) = %f, acc (avg) = %f' %
                (n_iter, cls_loss, cls_loss_avg, acc, acc_avg))
    
        # Save log
        if ((n_iter+1) % log_interval) == 0:
            log_writer.add_summary(summary, n_iter)
        # save snapshot
        if ((n_iter+1) > snapshot_start) & (((n_iter+1) % snapshot_interval) ==0):
            snapshot_saver.save(sess, snapshot_file % (n_iter+1), write_meta_graph=False)
        if n_iter >= max_iter:
            break
if __name__ == "__main__":
    tf.app.run()
