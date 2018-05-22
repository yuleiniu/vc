export VC_ROOT=$(pwd)
cd $VC_ROOT/submodule/cmn/util/faster_rcnn_lib/ && make
cd $VC_ROOT/submodule/cmn/util/roi_pooling/ && ./compile_roi_pooling.sh
cd $VC_ROOT
