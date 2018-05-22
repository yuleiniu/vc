"""Common util functions.

All the bounding boxes in this script should be represented by a 4-int list: 
\[x, y, w, h\]. 
(x, y) is the coordinates of the top left corner of the bounding box.
w and h are the width and height of the bounding box respectively.
"""

import numpy

def iou_bboxes(bbox1, bbox2):
  """Standard intersection over Union ratio between two bounding boxes."""
  bbox_ov_x = max(bbox1[0], bbox2[0])
  bbox_ov_y = max(bbox1[1], bbox2[1])
  bbox_ov_w = min(bbox1[0] + bbox1[2] - 1, bbox2[0] + bbox2[2] - 1) - bbox_ov_x + 1
  bbox_ov_h = min(bbox1[1] + bbox1[3] - 1, bbox2[1] + bbox2[3] - 1) - bbox_ov_y + 1
    
  area1 = area_bbox(bbox1)
  area2 = area_bbox(bbox2)
  area_o = area_bbox([bbox_ov_x, bbox_ov_y, bbox_ov_w, bbox_ov_h])
  area_u = area1 + area2 - area_o
  if area_u < 0.000001:
    return 0.0
  else:
    return area_o / area_u
      
def area_bbox(bbox):
  """Return the area of a bounding box."""
  if bbox[2] <= 0 or bbox[3] <= 0:
    return 0.0
  return float(bbox[2]) * float(bbox[3])