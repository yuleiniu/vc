from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from util.rnn import lstm_layer as lstm
from util.cnn import fc_layer as fc
from tensorflow.python.ops.nn import dropout as drop
from tensorflow import convert_to_tensor as to_T

from util.tf_eval_tools import compute_accuracy

class VC_Model(object):
    """
    Variational Context implementation based on https://arxiv.org/abs/1712.01892
    """
    def __init__(self, config, mode):
        """Basic setup.
        Args:
            config: Object containing configuration parameters.
            mode: "train", "eval" or "inference".
        """
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.is_supervised = config.is_supervised

        self.keys = 'vc'
        
        # A float32 Tensor with shape [batch_size, visual_length].
        self.region_visual_feat = None
        
        # A float32 Tensor with shape [batch_size, spatial_length].
        self.region_spatial_feat = None
        
        # An int32 Tensor with shape [padded_length, batch_size].
        self.text_seqs = None
                
        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None
        
        # Global step Tensor.
        self.global_step = None
    
    def build_inputs(self):
        """ Input batch.
        
        Outputs:            
            self.text_seqs
            self.region_visual_feat
            self.region_spatial_feat
            self.labels
        """
        with tf.variable_scope('inputs'):  
            self.text_seqs = tf.placeholder(dtype=tf.int32, 
                                            shape=[self.config.L, None], 
                                            name="text_seqs")
            self.region_vis_feat = tf.placeholder(dtype=tf.float32, 
                                            shape=[None, self.config.vis_dim], 
                                            name="region_vis_feat")
            self.region_visdif_feat = tf.placeholder(dtype=tf.float32, 
                                            shape=[None, self.config.vis_dim], 
                                            name="region_visdif_feat")
            self.region_spatial_feat = tf.placeholder(dtype=tf.float32, 
                                            shape=[None, self.config.spa_dim], 
                                            name="region_spatial_feat")
            self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
            
            self.region_visual_feat = tf.concat([self.region_vis_feat, self.region_visdif_feat], 
                                            axis=1, 
                                            name="region_visual_feat")
        
    def build_text_feature(self):
        """Generate text feature using bidirectional LSTM
        
        Outputs:
            self.text_bilstm_feat
            self.text_word_embed_feat
            self.word_is_not_pad
        """
        num_vocab = self.config.num_vocab
        embed_dim = self.config.embed_dim
        lstm_dim  = self.config.lstm_dim
        
        text_seq  = self.text_seqs         
        
        with tf.variable_scope('lstm'):           
            L = tf.shape(text_seq)[0] #seq length
            N1 = tf.shape(text_seq)[1] #batch size
            
            # Word embedding
            embedding_mat = tf.get_variable(name="embedding_mat", shape=[num_vocab, embed_dim])
            text_word_embed_feat = tf.nn.embedding_lookup(embedding_mat, text_seq) # [L, N1, embed_dim]
            
            # Encode the sentence into a vector representation, using the final
            # hidden states in a two-layer bidirectional LSTM network
            seq_length = tf.ones(to_T([N1]), dtype=tf.int32) * L
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=True)
            outputs1_raw, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell,
                                            cell_bw=lstm_cell,
                                            inputs=text_word_embed_feat,
                                            sequence_length=seq_length,
                                            dtype=tf.float32,
                                            time_major=True,
                                            scope="bidirectional_lstm1")
            outputs1 = tf.concat(outputs1_raw, axis=2)
            lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(lstm_dim, state_is_tuple=True)
            outputs2_raw, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell2, 
                                            cell_bw=lstm_cell2,
                                            inputs=outputs1, 
                                            sequence_length=seq_length, 
                                            dtype=tf.float32, 
                                            time_major=True, 
                                            scope="bidirectional_lstm2")
            outputs2 = tf.concat(outputs2_raw, axis=2)
            text_bilstm_feat = tf.concat([outputs1, outputs2], axis=2)
            if self.config.lstm_dropout:
                text_bilstm_feat = drop(text_bilstm_feat, 0.5)
                
            self.text_bilstm_feat = text_bilstm_feat
            self.text_word_embed_feat = text_word_embed_feat
            self.word_is_not_pad = tf.cast(tf.not_equal(text_seq, 0)[..., tf.newaxis], tf.float32)
    
    def build_encoder(self):
        """ Encoder, context estimated score
        
        Outputs:            
            self.enc_visual_feat
            self.enc_spatial_feat
            self.enc_score
        """        
        # text and region features
        text_bilstm_feat     = self.text_bilstm_feat
        text_word_embed_feat = self.text_word_embed_feat
        word_is_not_pad      = self.word_is_not_pad
        region_visual_feat   = self.region_visual_feat
        region_spatial_feat  = self.region_spatial_feat
        enc_dim = self.config.enc_dim
        
        # Tensor dimensionality
        L = tf.shape(text_bilstm_feat)[0]
        N1 = tf.shape(text_bilstm_feat)[1]
        N2 = tf.shape(region_spatial_feat)[0]
        D1 = text_bilstm_feat.get_shape().as_list()[-1] # lstm_dim*4
        D2 = text_word_embed_feat.get_shape().as_list()[-1] # embed_dim
        D3 = region_spatial_feat.get_shape().as_list()[-1] # spatial_dim
        D4 = region_visual_feat.get_shape().as_list()[-1] # visual_dim
    
        region_feat = tf.concat([region_visual_feat, region_spatial_feat], axis=1) # shape: [N2, D3+D4]
        
        with tf.variable_scope('encoder'):
            # 1. language-vision association between single RoI and the expression, represented by y^{c1} in the paper
            word_attention_single_score = fc('word_attention_single', tf.reshape(text_bilstm_feat, [-1, D1]), output_dim = 1) # shape: [L*N1, 1]
            word_attention_single_score = tf.reshape(word_attention_single_score, [L, N1, 1])
            word_prob = tf.nn.softmax(word_attention_single_score, dim = 0) * word_is_not_pad # shape: [L, N1, 1]
            word_prob = word_prob / tf.reduce_sum(word_prob, 0, keep_dims=True) # shape: [L, N1, 1]
            word_feat_single = tf.reduce_sum(word_prob * text_word_embed_feat, axis = 0) # shape: [N1, D2]

            # 2. language-vision association between pairwise RoI and the expression, represented by y^{c2} in the paper
            word_attention_pairwise_score = fc('word_attention_pairwise', tf.reshape(text_bilstm_feat, [-1, D1]), output_dim = 1)#shape: [L*N1, 1]
            word_attention_pairwise_score = tf.reshape(word_attention_pairwise_score, [L, N1, 1])  # shape: [L, N1, 1]
            word_prob = tf.nn.softmax(word_attention_pairwise_score, dim = 0) * word_is_not_pad # shape: [L, N1, 1]
            word_prob = word_prob / tf.reduce_sum(word_prob, 0, keep_dims=True) #shape: [L, N1, 1]
            word_feat_pairwise = tf.reduce_sum(word_prob * text_word_embed_feat, axis = 0) # shape: [N1, D2]

            # 3. context estimated score between single RoI and the expression
            region_embed = fc('region_visual_spatial_embed', region_feat, output_dim = D2) # shape: [N2, D2]
            region_embed = region_embed[tf.newaxis, ...] #shape: [1, N2, D2]
            mm_feat_norm = tf.nn.l2_normalize(region_embed * tf.reshape(word_feat_single, [N1, 1, D2]), dim = 2) #shape: [N1, N2, D2]
            single_score = fc('single_score', tf.reshape(mm_feat_norm, [-1, D2]), output_dim = 1) # shape: [N1*N2, 1]
            single_score = tf.reshape(single_score, [N1, N2, 1]) #shape[N1, N2, 1]

            # 4. context estimated score between pairwise RoI and the expression
            region_spatial_tile1 = tf.tile(tf.reshape(region_spatial_feat, [N2, 1, D3]), [1, N2, 1]) #shape: [N2, N2, D3]
            region_spatial_tile2 = tf.tile(tf.reshape(region_spatial_feat, [1, N2, D3]), [N2, 1, 1]) #shape: [N2, N2, D3]
            region_spatial_concat = tf.concat([region_spatial_tile1, region_spatial_tile2], axis = 2) #shape: [N2, N2, D3*2]
            region_embed = fc('region_spatial_embed', tf.reshape(region_spatial_concat, [-1, D3*2]), output_dim=D2) #shape: [N2*N2, D2]
            region_embed = region_embed[tf.newaxis, ...]
            mm_feat_norm = tf.nn.l2_normalize(region_embed * tf.reshape(word_feat_pairwise, [N1, 1, 1, D2]), 3) #shape: [N1, N2, N2, D2]
            pairwise_score = fc('pairwise_score', tf.reshape(mm_feat_norm, [-1, D2]), output_dim = 1) #shape: [N1*N2*N2, 1]
            pairwise_score = tf.reshape(pairwise_score, [N1, N2, N2]) # note that the semantic meaning of N2 and N2 swapped.
            pairwise_score = tf.transpose(pairwise_score, perm = [0, 2, 1]) # though it does not affect the result

            # 5. add single score and pairwise score
            alpha1 = tf.get_variable("scale_alpha1", shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1))
            alpha2 = tf.get_variable("scale_alpha2", shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1))
            single_score = single_score * alpha1
            pairwise_score = pairwise_score * alpha2
            score = single_score + pairwise_score # shape: [N1, N2, N2]
            score = tf.reshape(score, [-1, N2])
            score = tf.nn.softmax(score) # shape: [N1*N2, N2]            
            
            # 6. get softmax region feature, aka, the psudo object
            score = score[...,tf.newaxis] # shape: [N1*N2, N2, 1]
            z_spatial_feat = region_spatial_feat[tf.newaxis, ...] # shape: [1, N2, D3]
            z_visual_feat = region_visual_feat[tf.newaxis, ...] # shape, [1, N2, D4]
            z_spatial_feat = tf.reduce_sum(z_spatial_feat*score, axis = 1) # shape: [N1*N2, D3]
            z_visual_feat = tf.reduce_sum(z_visual_feat*score, axis = 1) # shape: [N1*N2, D4]
            z_spatial_feat = tf.reshape(z_spatial_feat, [N1, N2, D3])
            z_visual_feat = tf.reshape(z_visual_feat, [N1, N2, D4])
            z_region_feat = tf.concat([z_visual_feat, z_spatial_feat], axis=2)
            
            # 7. get the p(z|x) for the KL divergence
            # context estimated score between single RoI and the expression
            region_embed = fc('region_visual_spatial_embed', tf.reshape(z_region_feat, [-1, D3+D4]), output_dim=D2, reuse=True) # shape: [N1*N2, D2]
            region_embed = tf.reshape(region_embed, [N1, N2, D2]) #shape: [N1, N2, D2]
            mm_feat_norm = tf.nn.l2_normalize(region_embed * tf.reshape(word_feat_single, [N1, 1, D2]), 2)#shape: [N1, N2, D2]
            single_score = fc('single_score', tf.reshape(mm_feat_norm, [-1, D2]), output_dim=1, reuse=True) # shape: [N1*N2, 1]
            single_score = tf.reshape(single_score, [N1, N2])  # shape: [N1, N2]
            # context estimated score between pairwise RoI and the expression
            region_spatial_sub = tf.tile(tf.reshape(region_spatial_feat, [1, N2, D3]), [N1, 1, 1])  # shape: [N1, N2, D3]
            region_spatial_concat = tf.concat([region_spatial_sub, z_spatial_feat], axis=2)  # shape: [N1, N2, D3*2]
            region_embed = fc('region_spatial_embed', tf.reshape(region_spatial_concat, [-1, D3 * 2]), output_dim=D2, reuse=True) # shape: [N2*N2, D2]
            region_embed = tf.reshape(region_embed, [N1, N2, D2])
            mm_feat_norm = tf.nn.l2_normalize(region_embed * tf.reshape(word_feat_pairwise, [N1, 1, D2]), 2)  # shape: [N1, N2, D2]
            pairwise_score = fc('pairwise_score', tf.reshape(mm_feat_norm, [-1, D2]), output_dim=1, reuse=True)  # shape: [N1*N2, 1]
            pairwise_score = tf.reshape(pairwise_score, [N1, N2])  # shape: [N1, N2]
            5# add single score and pairwise score
            z_score = single_score * alpha1 + pairwise_score * alpha2
            
            self.enc_visual_feat  = z_visual_feat
            self.enc_spatial_feat = z_spatial_feat
            self.enc_score = z_score
        
    def build_decoder(self):
        """ referent grounding score
        """        
        # text and region features
        text_bilstm_feat     = self.text_bilstm_feat
        text_word_embed_feat = self.text_word_embed_feat
        word_is_not_pad      = self.word_is_not_pad
        region_visual_feat   = self.region_visual_feat
        region_spatial_feat  = self.region_spatial_feat
        enc_spatial_feat = self.enc_spatial_feat
        dec_dim = self.config.dec_dim
        
        # Tensor dimensionality
        L = tf.shape(text_bilstm_feat)[0]
        N1 = tf.shape(text_bilstm_feat)[1]
        N2 = tf.shape(region_spatial_feat)[0]
        D1 = text_bilstm_feat.get_shape().as_list()[-1] # lstm_dim*4
        D2 = text_word_embed_feat.get_shape().as_list()[-1] # embed_dim
        D3 = region_spatial_feat.get_shape().as_list()[-1] # spatial_dim
        D4 = region_visual_feat.get_shape().as_list()[-1] # visual_dim
    
        region_feat = tf.concat([region_visual_feat, region_spatial_feat], axis=1) # shape: [N2, D3+D4]
        with tf.variable_scope('decoder'):
            # 1. language-vision association between single RoI and the expression, represented by y^{r1} in the paper
            word_attention_single_score = fc('word_attention_single', tf.reshape(text_bilstm_feat, [-1, D1]), output_dim = 1) # shape: [L*N1, 1]
            word_attention_single_score = tf.reshape(word_attention_single_score, [L, N1, 1])
            word_prob = tf.nn.softmax(word_attention_single_score, dim = 0) * word_is_not_pad # shape: [L, N1, 1]
            word_prob = word_prob / tf.reduce_sum(word_prob, 0, keep_dims=True) # shape: [L, N1, 1]
            word_feat_single = tf.reduce_sum(word_prob * text_word_embed_feat, axis = 0) # shape: [N1, D2]

            # 2. language-vision association between single RoI and the expression, represented by y^{r2} in the paper
            word_attention_pairwise_score = fc('word_attention_pairwise', tf.reshape(text_bilstm_feat, [-1, D1]), output_dim = 1) # shape: [L*N1, 1]
            word_attention_pairwise_score = tf.reshape(word_attention_pairwise_score, [L, N1, 1]) # shape: [L, N1, 1]
            word_prob = tf.nn.softmax(word_attention_pairwise_score, dim = 0) * word_is_not_pad # shape: [L, N1, 1]
            word_prob = word_prob / tf.reduce_sum(word_prob, 0, keep_dims=True) # shape: [L, N1, 1]
            word_feat_pairwise = tf.reduce_sum(word_prob * text_word_embed_feat, axis = 0) # shape: [N1, D2]

            # 3. single region score: given every z region, check other region sub score induced by the z region
            region_embed = fc('region_visual_spatial_embed', region_feat, output_dim = D2) # shape: [N2, D2]
            region_embed = region_embed[tf.newaxis, ...]  # shape: [1, N2, D2]
            mm_feat_norm = tf.nn.l2_normalize(region_embed * tf.reshape(word_feat_single, [N1, 1, D2]), 2)  # shape: [N1, N2, D2]
            single_score = fc('single_score', tf.reshape(mm_feat_norm, [-1, D2]), output_dim=1)  # shape: [N1*N2, 1]
            single_score = tf.reshape(single_score, [N1, N2])  # shape: [N1, N2]

            # 4. pairwise region score: given every z region, check their relations to other regions.
            region_spatial_tile = tf.tile(region_spatial_feat[tf.newaxis, ...], [N1, 1, 1]) # shape: [N1, N2, D3]
            region_spatial_concat = tf.concat([region_spatial_tile, enc_spatial_feat], axis=2) # shape: [N1, N2, D3*2]
            region_embed = fc('region_spatial_embed', tf.reshape(region_spatial_concat, [-1, D3 * 2]), output_dim=D2) # shape: [N1*N2, D2]
            region_embed = tf.reshape(region_embed, [N1, N2, D2])
            mm_feat_norm = tf.nn.l2_normalize(region_embed * tf.reshape(word_feat_pairwise, [N1, 1, D2]), 2) # shape: [N1, N2, D2]
            pairwise_score = fc('pairwise_score', tf.reshape(mm_feat_norm, [-1, D2]), output_dim=1) # shape: [N1*N2, 1]
            pairwise_score = tf.reshape(pairwise_score, [N1, N2])

            # 5. add single score and pairwise score
            alpha1 = tf.get_variable("scale_alpha1", shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1))
            alpha2 = tf.get_variable("scale_alpha2", shape=[], dtype=tf.float32, initializer=tf.constant_initializer(1))
            single_score = single_score * alpha1
            pairwise_score = pairwise_score * alpha2
            score = single_score + pairwise_score  # shape: [N1, N2]
            
            self.localization_score = score
            
    def build_regulizer(self):
        """ context regularization score
        """             
        # text and region features
        text_bilstm_feat     = self.text_bilstm_feat
        text_word_embed_feat = self.text_word_embed_feat
        word_is_not_pad      = self.word_is_not_pad
        region_visual_feat   = self.region_visual_feat
        region_spatial_feat  = self.region_spatial_feat
        reg_dim = self.config.reg_dim 
        
        # Tensor dimensionality
        L = tf.shape(text_bilstm_feat)[0]
        N1 = tf.shape(text_bilstm_feat)[1]
        N2 = tf.shape(region_spatial_feat)[0]
        D1 = text_bilstm_feat.get_shape().as_list()[-1] # lstm_dim*4
        D2 = text_word_embed_feat.get_shape().as_list()[-1] # embed_dim
        D3 = region_spatial_feat.get_shape().as_list()[-1] # spatial_dim
        D4 = region_visual_feat.get_shape().as_list()[-1] # visual_dim
        
        region_feat = tf.concat([region_visual_feat, region_spatial_feat], axis=1) # shape: [N2, D3+D4]
        
        with tf.variable_scope('regularizer'):
            # 1. language-vision association between single RoI and the expression, represented by y^{g} in the paper
            word_obj_attention_score = fc('word_attention_obj', tf.reshape(text_bilstm_feat, [-1, D1]), output_dim = 1) # shape: [L*N1, 1]
            word_obj_attention_score = tf.reshape(word_obj_attention_score, [L, N1, 1])
            word_prob = tf.nn.softmax(word_obj_attention_score, dim = 0) * word_is_not_pad #shape: [L, N1, 1]
            word_prob = word_prob / tf.reduce_sum(word_prob, 0, keep_dims=True) #shape: [L, N1, 1]
            word_obj_feat = tf.reduce_sum(word_prob * text_word_embed_feat, axis = 0) #shape: [N1, D2]

            # 2. single score for subject
            region_embed = fc('region_obj_embed', region_feat, output_dim = D2) #shape: [N2, D2]
            mm_feat = tf.nn.l2_normalize(region_embed[tf.newaxis, ...] * tf.reshape(word_obj_feat, [N1, 1, D2]), dim = 2)#shape: [N1, N2, D2]
            score = fc('single_score', tf.reshape(mm_feat, [-1, D2]), output_dim = 1)  # shape: [N1*N2, 1]
            score = tf.reshape(score, [N1, N2]) #shape[N1, N2]
            
            self.prior_score = score
     
    def build_model(self):
        """Builds encoder, decoder and regulazier."""
        self.build_text_feature()
        self.build_encoder()
        self.build_decoder()
        self.build_regulizer()
        
        # final score = localization_score - enc_score + prior_score
        self.scores = self.localization_score - self.enc_score + self.prior_score
        self.preds  = tf.argmax(self.scores, axis=1)
    
    def evaluate(self):
        """Evaluation."""
        self.accuracy = compute_accuracy(self.region_spatial_feat, self.preds, self.labels)

    def setup_summary(self):
        """Set up summaries, such as loss, accuracy, and learning rate."""

        # Summary collection
        with tf.variable_scope('loss'):
            tf.summary.scalar('cls_loss', self.cls_loss_avg)
            tf.summary.scalar('reg_loss', self.reg_loss)
            tf.summary.scalar('learning_rate', self.learning_rate)

        with tf.variable_scope('accuracy'):
            tf.summary.scalar('acc_trn', self.accuracy_avg)

        # Add to update_ops collection.
        summary_op = tf.summary.merge_all()
        self.summary_op = summary_op

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"   
     
    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step
    
    def setup_loss(self):   
        """Sets up loss."""
        # Classification loss
        if self.is_supervised:
            cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.scores, labels=self.labels)
        else:
            eps = 1e-5
            cls_loss = -tf.log(tf.reduce_max(tf.maximum(tf.nn.softmax(self.scores), eps), axis=1))        
        cls_loss = tf.reduce_mean(cls_loss)    
        
        # Regularization Loss
        train_var_list = tf.trainable_variables()        
        reg_var_list = [var for var in train_var_list if
                (var.name[-9:-2] == 'weights' or var.name[-8:-2] == 'Matrix')]
        reg_loss = self.config.weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in reg_var_list])
        
        total_loss = cls_loss + reg_loss
        
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.total_loss = total_loss
    
    def setup_train_op(self):
        """Sets up the optimizer and train op."""
        config = self.config
        
        # Learning_rate
        learning_rate = tf.train.exponential_decay(learning_rate=config.start_lr, 
                                    global_step=self.global_step, 
                                    decay_steps=config.lr_decay_step,
                                    decay_rate=config.lr_decay_rate, staircase=True)
            
        # Optimizer
        solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=config.momentum)
        
        # Compute gradients
        train_var_list = tf.trainable_variables() 
        grads_and_vars = solver.compute_gradients(self.total_loss, var_list=train_var_list)
        # Clip gradient by L2 norm (set maximum L2 norm to 10).
        grads_and_vars = [(tf.clip_by_norm(g, clip_norm=config.clip_gradients), v) for g, v in grads_and_vars]
        # Apply gradients.
        solver_op = solver.apply_gradients(grads_and_vars, global_step=self.global_step)

        # Add to update_ops collection.
        tf.add_to_collection(self.keys, solver_op)

        self.learning_rate = learning_rate
    
    def setup_update_op(self):
        # Average classification loss
        cls_loss_avg = tf.Variable(initial_value=0,
                                    trainable=False, 
                                    dtype=tf.float32, 
                                    name='cls_loss_avg')
        cls_loss_op = tf.assign_add(cls_loss_avg,
                        (1-self.config.avg_decay)*(self.cls_loss-cls_loss_avg))
        tf.add_to_collection(self.keys, cls_loss_op)

        # Average accuracy
        accuracy_avg = tf.Variable(initial_value=0,
                                    trainable=False, 
                                    dtype=tf.float32, 
                                    name='accuracy_avg')
        accuracy_op = tf.assign_add(accuracy_avg,
                        (1-self.config.avg_decay)*(self.accuracy-accuracy_avg)) 
        tf.add_to_collection(self.keys, accuracy_op)

        self.cls_loss_avg = cls_loss_avg
        self.accuracy_avg = accuracy_avg

    def setup_ops(self):
        """Sets up all train_ops."""
        self.setup_loss()
        self.setup_train_op()
        self.setup_update_op()

        # Group all ops
        ops = tf.group(*tf.get_collection(self.keys))
        self.ops = ops


    def build(self):  
        with tf.variable_scope('vc'):
            self.build_inputs()
            self.build_model()
            self.evaluate()
            self.setup_global_step()
            if self.is_training():
                self.setup_ops()
                self.setup_summary()