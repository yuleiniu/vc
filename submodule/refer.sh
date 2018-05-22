export VC_ROOT=$(pwd)

### download data
export VC_ROOT=$(pwd)
if [ ! -d "./data/prepro/" ]; then
    mkdir ./data/prepro/
fi
cd $VC_ROOT/data/prepro/

## declare an array variable
declare -a arr=("refcoco" "refcoco+" "refcocog")
for dataset in "${arr[@]}"
do
    wget "http://bvisionweb1.cs.unc.edu/licheng/referit/data/${dataset}.zip"
    unzip "${dataset}.zip"
    rm "${dataset}.zip"
done
cd $VC_ROOT

### preprocess
cd $VC_ROOT/submodule/refer && make
cd $VC_ROOT

#declare -a arr=("refcoco" "refcoco+" "refcocog")
#for dataset in "${arr[@]}"
#do
#    python prepro.py --dataset ${dataset}
#done
python prepro.py --dataset refcoco  --splitBy unc
python prepro.py --dataset refcoco+ --splitBy unc
python prepro.py --dataset refcocog --splitBy google
