# Grounding Referring Expressions in Images by Variational Context

This repository contains the code for the following paper:

* Hanwang Zhang, Yulei Niu, Shih-Fu Chang, *Grounding Referring Expressions in Images by Variational Context*. In CVPR, 2018. ([PDF](https://arxiv.org/pdf/1712.01892.pdf))

```
@article{zhang2018grounding,
  title={Grounding Referring Expressions in Images by Variational Context},
  author={Zhang, Hanwang and Niu, Yulei and Chang, Shih-Fu},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```

Note: part of this repository is built upon [cmn](https://github.com/ronghanghu/cmn), [speaker_listener_reinforcer](https://github.com/lichengunc/speaker_listener_reinforcer) and [refer](https://github.com/lichengunc/refer).

## Requirements and Dependencies

* Python 3 ([Anaconda](https://www.continuum.io/downloads) recommended)
* [TensorFlow](https://www.tensorflow.org/install/) (v1.3.0 or higher)
* Clone 
```shell
# Make sure to clone with --recursive
git clone --recursive https://github.com/yuleiniu/vc.git
```
The ``recursive`` will help also clone the [refer API](https://github.com/lichengunc/refer) and [cmn API](https://github.com/ronghanghu/cmn) repo.
* Install other dependencies by simply run:
```shell
  pip install -r requirements.txt
```

## Preprocessing

* Download the model weights of Faster-RCNN VGG-16 network converted from Caffe model:
```shell
  ./data/models/download_vgg_params.sh
```

* Download the GloVe matrix for word embedding:
```shell
  ./data/word_embedding/download_embed_matrix.sh
```

* Re-build the NMS lib and the ROIPooling operation following [cmn](https://github.com/ronghanghu/cmn). Simply run:
```shell
  ./submodule/cmn.sh
```

* Preprocess data for the use of referring expression following [speaker_listener_reinforcer](https://github.com/lichengunc/speaker_listener_reinforcer) and [refer](https://github.com/lichengunc/refer) (implemented by Python 2) , and save the results into ``data/raw``. Simply run:
```shell
  ./submodule/refer.sh
```

## Extract features

* Extract region features for RefCOCO/RefCOCO+/RefCOCOg, run:
```shell
  python prepare_data.py --dataset refcoco  (for RefCOCO)
  python prepare_data.py --dataset refcoco+ (for RefCOCO+)
  python prepare_data.py --dataset refcocog (for RefCOCOg)
```

## Train

* To train the model, run:
```shell
  python train.py --dataset refcoco  (for RefCOCO)
  python train.py --dataset refcoco+ (for RefCOCO+)
  python train.py --dataset refcocog (for RefCOCOg)
```

## Evaluation

* To test the model, run:
```shell
  python test.py --dataset refcoco  (for RefCOCO)  --checkpoint /path/to/checkpoint
  python test.py --dataset refcoco+ (for RefCOCO+) --checkpoint /path/to/checkpoint
  python test.py --dataset refcocog (for RefCOCOg) --checkpoint /path/to/checkpoint
```