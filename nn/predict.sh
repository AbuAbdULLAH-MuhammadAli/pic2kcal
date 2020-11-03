export PYTHONPATH=..
for f in "$@"; do python predict.py --input-file $f --model densenet121 --weights ~/data/tmp/nobackup/pic2kcal-data/runs-from-cluster-2020/2020-03-15T12.58.07-no_portion_size_100g/epoch-018.pt --train-type kcal+nut+topings --bce-weight=400 --no-predict-portion-size; done
