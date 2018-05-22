from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def compute_area(bbox):
    """Calculates area of the given box
    Input:
        bbox: bounding boxes, [N_batch, (x1, y1, x2, y2)]     
    Output:
        area: [N_batch, (area)]  
    """
    tl = tf.slice(bbox, begin=[0,0], size=[-1,2]) # top left
    br = tf.slice(bbox, begin=[0,2], size=[-1,2]) # bottom right
    area = tf.reduce_prod(tf.subtract(br, tl), axis=1)
    area = tf.maximum(area, 0)
    return area

def compute_iou(bbox1, bbox2):
    """Calculates IoU of the given box
    Input:
        bbox_det: detected bounding boxes, [N_batch, (x1, y1, x2, y2, ...)]
        bbox_gt: ground truth bounding boxes [N_batch, (x1, y1, x2, y2, ...)]        
    Output:
        IoU: [N_batch, (IoU)]    
    """
    
    # bbox should be [N_batch, (x1, y1, x2, y2)]
    bbox1 = tf.slice(bbox1, begin=[0,0], size=[-1,4])
    bbox2 = tf.slice(bbox2, begin=[0,0], size=[-1,4])
    
    # Compute area
    area1 = compute_area(bbox1)
    area2 = compute_area(bbox2)
    
    bbox_min = tf.minimum(bbox1, bbox2)
    bbox_max = tf.maximum(bbox1, bbox2)
    x1_min, y1_min, x2_min, y2_min = tf.unstack(bbox_min, axis=1)
    x1_max, y1_max, x2_max, y2_max = tf.unstack(bbox_max, axis=1)    
    
    # Compute intersection, union, IoU
    intersection = tf.maximum(x2_min-x1_max, 0) * tf.maximum(y2_min-y1_max, 0)
    union = area1 + area2 - intersection
    IoU = tf.maximum(tf.divide(intersection, union), 0)
    IoU = tf.cast(IoU, dtype=tf.float32)

    return IoU

def compute_accuracy(bbox, preds, labels, threshold=0.5):
    """Calculates accuracy
    Input:
        bbox: bounding boxes, [N_batch, (x1, y1, x2, y2, ...)]
        preds: predictions, [N_batch, (preds)]
        labels: labels, [N_batch, (labels)]
        threshold: IoU threshold, float
    Output:
        acc: [N_batch, (acc)]    
    """
    
    # Gather predicted and ground-truth bounding box.
    bbox_pred = tf.gather(bbox, preds)
    bbox_gt   = tf.gather(bbox, labels)

    # Compute IoU
    IoU = compute_iou(bbox_pred, bbox_gt)

    # Compute IoU>threshold
    threshold_ph = tf.constant(threshold, dtype=tf.float32)
    acc = tf.cast(tf.greater_equal(IoU, threshold), dtype=tf.float32)
    acc = tf.reduce_mean(acc) 

    return acc