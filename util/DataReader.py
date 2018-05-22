from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import json
import h5py
import tqdm
import pickle
from sklearn.preprocessing import normalize
from util import im_processing, text_processing as text_processing, fastrcnn_vgg_net
from util.spatial_feat import spatial_feature_from_bbox

import skimage.io
import skimage.transform
import tensorflow as tf

import random

# data paths
im_mean = fastrcnn_vgg_net.channel_mean
IMG_PATH = './data/images/mscoco/'

class DataReader:
    def __init__(self, config,
                 use_category=True):
        
        option = '%s_%s' % (config.dataset, config.split)
        data_path = './data/raw/%s/data.json' % option
        vis_feat_path = './data/vis_feats/%s_ann_vis_feats.pkl' % config.dataset
        vocab_file = './data/word_embedding/vocabulary_72700.txt'
        info_print = config.info_print

        # load data 
        self.use_category = use_category
        self.info_print = info_print
        with open(data_path) as data_file:
            self.data = json.load(data_file)
        self.vis_feat_path = vis_feat_path
        self.vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)
        
        # index to word
        self.ix_to_word = self.data['ix_to_word']
        self.word_to_ix = self.data['word_to_ix']
        
        # restruct refs, anns, images to dictionary
        self.refs = self.to_dict('refs', 'ref_id')
        self.anns = self.to_dict('anns', 'ann_id')
        self.sents = self.to_dict('sentences', 'sent_id')
        self.images = self.to_dict('images', 'image_id')
        
        # collect ref_ids, ann_ids, image_ids
        self.ref_ids   = list(self.refs.keys())
        self.ann_ids   = list(self.anns.keys())
        self.image_ids = list(self.images.keys())
        
        self.print_info('We have %d images.' % len(self.image_ids))
        self.print_info('We have %d anns.' % len(self.ann_ids))
        self.print_info('We have %d refs.' % len(self.ref_ids))
        
        # collect ref_to_ann, ref_to_sents, ann_to_image, image_to_anns, etc
        self.ref_to_ann    = self.key_to_key('ref', 'ann_id')
        if use_category:
            self.ref_to_cat = self.key_to_key('ref', 'category_id')
        self.ref_to_image  = self.key_to_key('ref', 'image_id')
        self.ref_to_sents  = self.key_to_key('ref', 'sent_ids')
        self.ann_to_image  = self.key_to_key('ann', 'image_id')
        if use_category:
            self.ann_to_cat = self.key_to_key('ann', 'category_id')
        self.ann_to_box    = self.key_to_key('ann', 'box')
        self.image_to_anns = self.key_to_key('image', 'ann_ids')
        self.image_to_refs = self.key_to_key('image', 'ref_ids')
        self.print_info('Mapping finished.')
        
        # collect visual and spatial features
        self.print_info('Collecting visual and spatial features...')
        self.ann_spa_feats = self.fetch_spa_feat() # spatial feature
        self.ann_vis_feats = self.fetch_vis_feat() # visual feature
        
        # collect same/different type(category) anns set
        if use_category:
            self.st_anns, self.dt_anns = self.fetch_nn_ids()
        
        # collect dif features
        if use_category:
            self.print_info('Calculating dif features...')
            self.ann_spadif_feats = self.fetch_spadif_feat() # spadif feature
            self.ann_visdif_feats = self.fetch_visdif_feat() # visdif feature
        
        # collect train/val split ids
        self.print_info('Splitting image ids...')
        self.image_split_ids = {}
        self.batch_list = {}
        self.num_batch = {}
        self.epoch = {}
        split_list = ['train', 'val', 'test', 'testA', 'testB']
        for split in split_list:
            self.image_split_ids[split] = self.get_split_ids(split)
            self.batch_list[split] = []
            self.num_batch[split] = len(self.image_split_ids[split])
            self.epoch[split] = -1
        
        self.print_info('Initialization finished.')
        
    def print_info(self, info):
        if self.info_print:
            print(info)
            
    def to_dict(self, attr, key):
        dictionary = {}
        for x in self.data[attr]:
            dictionary[x[key]] = x
        return dictionary
            
    def fetch_key_ids(self, key, split=None):
        key_ids = []
        if split == None:
            for dt in self.data[key+'s']:
                key_ids.append(dt[key+'_id'])
        else:
            for dt in self.data[key+'s']:
                if dt['split']==split:
                    key_ids.append(dt[key+'_id'])           
        return key_ids
    
    def key_to_key(self, key1, key2):
        dictionary = {}        
        for dt in self.data[key1+'s']:
            dictionary[dt[key1+'_id']] = dt[key2]
        return dictionary
    
    # get same/different type anns set, sorted by distance
    def fetch_nn_ids(self):
        ann_to_image  = self.ann_to_image
        ann_to_cat  = self.ann_to_cat
        image_to_anns = self.image_to_anns
        st_anns, dt_anns = {}, {} # same/different type anns
        for ann_id in self.ann_ids:
            image_id = ann_to_image[ann_id]
            candidates = image_to_anns[image_id]
            st_list, dt_list = [], []
            for other_id in candidates:
                if (ann_id != other_id) & (ann_to_image[ann_id] == ann_to_image[other_id]):
                    if ann_to_cat[ann_id] == ann_to_cat[other_id]:
                        st_list.append(other_id)
                    else:
                        dt_list.append(other_id)
            st_anns[ann_id], _ = self.sort_nn_ids(ann_id, st_list)
            dt_anns[ann_id], _ = self.sort_nn_ids(ann_id, dt_list)
        return st_anns, dt_anns
    def sort_nn_ids(self, ann_id, other_ids, order='ascending'):
        x1, y1, x2, y2, _ = self.fetch_ann_box_feat(ann_id)
        distance = []
        for other_id in other_ids:
            ox1, oy1, ox2, oy2, _ = self.fetch_ann_box_feat(other_id)
            dis = np.sqrt(((x1+x2-ox1-ox2)/2)**2 + ((y1+y2-oy1-oy2)/2)**2)
            distance.append(dis)
        dist_index = np.argsort(distance).tolist()
        distance = np.sort(distance).tolist()
        return [other_ids[x] for x in dist_index], distance
    
    # feature extraction
    def fetch_spa_feat(self):
        self.print_info('Collecting spatial features...')
        ann_spa_feats = {}
        for ann_id in self.ann_ids:
            ann_spa_feats[ann_id] = self.fetch_ann_spa_feat(ann_id)
        return ann_spa_feats
    def fetch_ann_spa_feat(self, ann_id, min_size=600, max_size=1000):
        # return x1, y1, x2, y2, area
        image_id = self.ann_to_image[ann_id]
        W, H = self.images[image_id]['width'], self.images[image_id]['height']
        x1, y1, w, h = self.ann_to_box[ann_id]
        x2 = max(x1+1, x1+w-1)
        y2 = max(y1+1, y1+h-1) 
        area = w*h
        
        # scale
        scale = min(max(min_size/H, min_size/W), max_size/H, max_size/W)
        new_h, new_w = int(scale*H), int(scale*W)
        region_bboxes = np.array([[x1, y1, x2, y2]], np.float32) * scale
        region_bboxes = im_processing.rectify_bboxes(region_bboxes, height=new_h, width=new_w)

        bbox_batch = np.zeros((len(region_bboxes), 5), np.float32)
        bbox_batch[:, 1:5] = region_bboxes
        spatial_batch = spatial_feature_from_bbox(region_bboxes, im_h=new_h, im_w=new_w)
        
        return spatial_batch[0]
    def fetch_ann_box_feat(self, ann_id):
        # return x1, y1, x2, y2, area
        image_id = self.ann_to_image[ann_id]
        W, H = self.images[image_id]['width'], self.images[image_id]['height']
        x1, y1, w, h = self.ann_to_box[ann_id]
        x2 = max(x1+1, x1+w-1)
        y2 = max(y1+1, y1+h-1)
        area = w*h
        return x1/W, y1/H, x2/W, y2/H, area/(W*H)
    def fetch_vis_feat(self):
        if os.path.isfile(self.vis_feat_path):
            with open(self.vis_feat_path, 'rb') as input:
                ann_vis_feats = pickle.load(input, encoding='bytes')
        else:
            ann_vis_feats = self.extract_ann_vis_feat()
        return ann_vis_feats
    # extract and save vis_feat
    def extract_ann_vis_feat(self):
        self.print_info('Extracting visual features...')
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            # load tensorflow vgg model
            img_batch = tf.placeholder(tf.float32, [1, None, None, 3])
            box_batch = tf.placeholder(tf.float32, [None, 5])
            vgg_feat = fastrcnn_vgg_net.vgg_roi_fc7(img_batch, box_batch, "vgg_local", apply_dropout = False)
            self.load_vgg(sess) # load pretrained cnn model
            ann_vis_feats = {}
            #for img_id in self.image_ids:
            for i in tqdm.trange(len(self.image_ids)):
                img_id = self.image_ids[i]
                # load image
                img_path = self.images[img_id]['file_name']
                img_name = os.path.join(IMG_PATH, img_path.split('_')[1], img_path)
                img = skimage.io.imread(img_name)
                if img.ndim == 2:
                    img = np.tile(img[..., np.newaxis], (1, 1, 3))

                # image to anns and box
                ann_ids = self.image_to_anns[img_id]
                box = []
                for ann_id in ann_ids:
                    box.append(self.ann_to_box[ann_id])
                
                # modify img and box
                img, box = self.img_box_modify(img, box)  
                
                # visual feature extraction
                vis_feat = sess.run(vgg_feat, feed_dict={img_batch:img, box_batch:box})
                for i in range(len(ann_ids)):
                    ann_vis_feats[ann_ids[i]] = vis_feat[i]
            # save vis_feat     
            with open(self.vis_feat_path, 'wb') as output:
                pickle.dump(ann_vis_feats, output)
        self.print_info('Visual feature extracted.')
        return ann_vis_feats
    def fetch_visdif_feat(self):
        ann_visdif_feats = {}
        for i in tqdm.trange(len(self.ann_ids)):
            ann_id = self.ann_ids[i]
            ann_vis_feat = self.ann_vis_feats[ann_id]
            other_ids = self.st_anns[ann_id]
            if len(other_ids) == 0:
                vis_dif_feat = np.zeros([len(ann_vis_feat)])
            else:
                vis_dif_feats = np.zeros([len(other_ids), len(ann_vis_feat)])
                for j, other_id in enumerate(other_ids):
                    vis_dif_feats[j] = ann_vis_feat - self.ann_vis_feats[other_id] # vis_dif: ai-aj
                vis_dif_feats = normalize(vis_dif_feats) # (ai-aj)/|ai-aj|
                vis_dif_feat = np.mean(vis_dif_feats, axis=0)
            ann_visdif_feats[ann_id] = vis_dif_feat
        return ann_visdif_feats            
    def fetch_spadif_feat(self):
        ann_spadif_feats = {}
        for i in tqdm.trange(len(self.ann_ids)):
            ann_id = self.ann_ids[i]
            ann_spa_feat = self.ann_spa_feats[ann_id] # [x1/w, y1/h, x2/w, y2/h, (w*h)/(W*H)]
            ann_w, ann_h = self.ann_to_box[ann_id][2:4] # w, h
            other_ids = self.st_anns[ann_id]
            # vis_dif
            spa_dif_feat = np.zeros([25])
            for j, other_id in enumerate(other_ids):
                if j>=5:
                    break
                other_spa_feat = self.ann_spa_feats[other_id]
                other_w, other_h = self.ann_to_box[other_id][2:4]
                other_spa_feat = other_spa_feat * np.array([other_w/ann_w, other_h/ann_h, other_w/ann_w, other_h/ann_h, 1])
                spa_dif_feat_tmp = ann_spa_feat - other_spa_feat # spa:dif
                spa_dif_feat_tmp[4] = other_spa_feat[4]/ann_spa_feat[4]
                spa_dif_feat[(j*5):(j*5+5)] = spa_dif_feat_tmp
            ann_spadif_feats[ann_id] = spa_dif_feat
        return ann_spadif_feats
    
    # other functions
    def get_split_ids(self, split='train'):
        # train
        image_ids = []
        for image_id in self.image_ids:
            for ref_id in self.image_to_refs[image_id]:
                ref = self.refs[ref_id]
                if ref['split'] == split:
                    image_ids.append(image_id)
                    break
        return image_ids
    
    def img_box_modify(self, im, box, min_size=600, max_size=1000):
        # box[x1, y1, w, h] -> [batch_index(0), x1, y1, x2, y2]

        # For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
        bbox_batch = np.zeros((len(box), 5), np.float32)
        bbox_batch[:,1:5] = box
        bbox_batch[:,3:5] = bbox_batch[:,3:5] + bbox_batch[:,1:3] - 1 # x2 = x1+w-1
        
        # calculate the resize scaling factor
        im_h, im_w = im.shape[:2]
        # make the short size equal to min_size but also the long size no bigger than max_size
        scale = min(max(min_size/im_h, min_size/im_w), max_size/im_h, max_size/im_w)
        
        # resize and process the image
        new_h, new_w = int(scale*im_h), int(scale*im_w)
        im_resized = skimage.img_as_float(skimage.transform.resize(im, [new_h, new_w]))
        im_processed = im_resized*255 - im_mean
        im_batch = im_processed[np.newaxis, ...].astype(np.float32)
        
        # resize and process the box
        bbox_batch[:, 1:] = im_processing.rectify_bboxes(bbox_batch[:, 1:]*scale, height=new_h, width=new_w)
  
        return im_batch, bbox_batch
    
    def load_vgg(self, sess):
        # Initialize CNN Parameters
        convnet_params = './data/models/fasterrcnn_vgg_coco_params.npz'
        convnet_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                          'conv3_1', 'conv3_2', 'conv3_3',
                          'conv4_1', 'conv4_2', 'conv4_3',
                          'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7']
        processed_params = np.load(convnet_params)
        processed_W = processed_params['processed_W'][()]#hw
        processed_B = processed_params['processed_B'][()]
        init_ops = []
        with tf.variable_scope('vgg_local', reuse=True):
            for l_name in convnet_layers:
                assign_W = tf.assign(tf.get_variable(l_name + '/weights'), processed_W[l_name])
                assign_B = tf.assign(tf.get_variable(l_name + '/biases'), processed_B[l_name])
                init_ops += [assign_W, assign_B]
        processed_params.close()
        sess.run(tf.group(*init_ops))
        
    def get_batch(self, split='train', shuffle=True, echo=True, image_id = None):
        if image_id == None:
            batch_list = self.batch_list[split][:]
            if len(batch_list) == 0:
                batch_list = self.image_split_ids[split][:]
                self.epoch[split] += 1
                if shuffle:
                    random.shuffle(batch_list)
            if echo:
                print('data reader: epoch = %d, batch = %d / %d' % (self.epoch[split], len(self.image_split_ids[split])-len(batch_list), len(self.image_split_ids[split])))
            image_id = batch_list.pop(0)
            self.batch_list[split] = batch_list
        ann_ids = self.image_to_anns[image_id]
        batch = {}
        batch['im_id'] = image_id
        # coco_bboxes, category_batch
        coco_bboxes = []
        vis_batch = []
        spa_batch = []
        if self.use_category:
            category_batch = []
            visdif_batch = []
            spadif_batch = []
        for ann_id in self.image_to_anns[image_id]:
            coco_bboxes.append(self.ann_to_box[ann_id])
            vis_batch.append(self.ann_vis_feats[ann_id])
            spa_batch.append(self.ann_spa_feats[ann_id])
            if self.use_category:
                category_batch.append(self.ann_to_cat[ann_id])
                visdif_batch.append(self.ann_visdif_feats[ann_id])
                spadif_batch.append(self.ann_spadif_feats[ann_id])
        batch['coco_bboxes'] = np.array(coco_bboxes, dtype=np.float32)
        batch['vis_batch'] = np.array(vis_batch, dtype=np.float32)
        batch['spa_batch'] = np.array(spa_batch, dtype=np.float32)
        if self.use_category:
            batch['category_batch'] = np.array(category_batch, dtype=np.int32)
            batch['visdif_batch'] = np.array(visdif_batch, dtype=np.float32)
            batch['spadif_batch'] = np.array(spadif_batch, dtype=np.float32)
        # coco_ann_ids, label_batch
        coco_ann_ids = []
        label_batch = []
        questions = []
        text_zseq_batch = []    # zero + seq for comprehension
        text_seqz_batch = []    # seq + zero for generation
        for ref_id in self.image_to_refs[image_id]:
            ref = self.refs[ref_id]
            ann_id = self.ref_to_ann[ref_id]
            if ref['split'] == split:
                for sent_id in ref['sent_ids']:
                    sent = self.sents[sent_id]['sent']
                    # refine sentence
                    coco_ann_ids.append(ann_id)
                    label_batch.append(ann_ids.index(ann_id))
                    questions.append(sent)
                    text_zseq_batch.append(
                            text_processing.preprocess_sentence(sent, self.vocab_dict, T=20, mode='zseq'))
                    text_seqz_batch.append(
                            text_processing.preprocess_sentence(sent, self.vocab_dict, T=20, mode='seqz'))
        text_zseq_batch = np.array(text_zseq_batch, dtype=np.int32).T
        text_seqz_batch = np.array(text_seqz_batch, dtype=np.int32).T
        batch['coco_ann_ids'] = coco_ann_ids
        batch['label_batch']  = np.array(label_batch, dtype=np.int32)
        batch['questions']    = questions
        batch['text_zseq_batch'] = np.array(text_zseq_batch, dtype=np.int32)
        batch['text_seqz_batch'] = np.array(text_seqz_batch, dtype=np.int32)
        return batch
