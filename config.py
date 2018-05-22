from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import time

tf.flags.DEFINE_string("dataset", "refcoco",
                       "Dataset for training/test, option:refcoco/refcoco+/refcocog.")
tf.flags.DEFINE_string("vocab_file", "./data/word_embedding/vocabulary_72700.txt",
                       "Vocabulary file path.")
tf.flags.DEFINE_string("wordembed_params", "./data/word_embedding/embed_matrix.npy",
                       "word embedding initialization file path.")
tf.flags.DEFINE_string("checkpoint", "",
                       "checkpoint for evaluation (only used during evaluation stage).")
tf.flags.DEFINE_integer("log_interval", 500,
                       "Interval for saving log.")
tf.flags.DEFINE_integer("snapshot_start", 120000,
                       "Start step for saving snapshot.")
tf.flags.DEFINE_integer("snapshot_interval", 1000,
                       "Interval for saving snapshot.")
tf.flags.DEFINE_integer("max_iter", 180000,
                       "Maximum iterations for training.")

FLAGS = tf.app.flags.FLAGS

class Model_Config(object):
    """Wrapper class for model hyperparameters."""
    def __init__(self):
        """Sets the default model and training hyperparameters."""
        # LSTM input and output dimensionality, respectively.
        self.embed_dim = 300
        self.lstm_dim = 1000
        
        # Sequence maximum length and vocabulary length.
        self.L = 20
        self.num_vocab = 72704
        
        # Visual and spatial feature dimensionality
        self.vis_dim = 4096
        self.spa_dim = 5
        
        # Encoder, decoder and regularizer embedding dimensionality, respectively.
        self.enc_dim = 512
        self.dec_dim = 512
        self.reg_dim = 512

        # Training hyperparameters.
        # If True, the dropout applied to LSTM variables.
        self.lstm_dropout = False

        # Hyperparameters for learning rate and Momentum optimizer
        self.start_lr = 0.01
        self.lr_decay_step = 120000
        self.lr_decay_rate = 0.1
        self.momentum = 0.95
        
        # If not None, clip gradients to this value.
        self.clip_gradients = 10.0
        
        # Weight decay for regularization.
        self.weight_decay = 0.0005

        # Decay for averaging loss and accuracy .
        self.avg_decay = 0.99
 
class File_Config(object):
    """Data path for reader and main function."""
    def __init__(self, model='vc'):
        """Sets the data path."""
        # LSTM input and output dimensionality, respectively.

        # Dataset type.
        self.dataset = FLAGS.dataset # refcoco/refcoco+/refcocog

        # If True, print loading information. 
        self.info_print = True

        # Model type
        self.model = model
        
        # Set checkpoint (only useful in evaluation)
        self.checkpoint = FLAGS.checkpoint

        # Set split type for different datasets.
        self.setup()
        
    def set_split(self):
        assert self.dataset in ['refcoco', 'refcoco+', 'refcocog'], "Dataset should be refcoco/refcoco+/refcocog"

        if self.dataset in ['refcoco', 'refcoco+']:
            self.split = 'unc'
        else:
            self.split = 'google'

    def set_log_options(self):
        """Set tensorflow log and snapshot options."""
        self.log_dir = './tflog/%s/' % self.dataset
        self.log_interval = FLAGS.log_interval

        # Set snapshot options
        self.snapshot_dir = './tfmodel/%s/' % self.dataset
        self.snapshot_file =  os.path.join(self.snapshot_dir, 'iter_%d.tfmodel')
        self.snapshot_start = FLAGS.snapshot_start
        self.snapshot_interval  = FLAGS.snapshot_interval

    def set_init_params(self):
        """Set initialization parameters."""
        self.num_vocab = 72704
        self.embed_dim = 300
        self.vocab_file = FLAGS.vocab_file
        self.wordembed_params = FLAGS.wordembed_params

    def setup(self):
        """Set tensorflow log directory and so on."""
        self.set_split()
        self.set_log_options()
        self.set_init_params()