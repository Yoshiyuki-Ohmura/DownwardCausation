

for f in `seq 1 100`
do
    echo $f
    PYTHONPATH=$(pwd) python script/evaluationInj.py -ldim 32 -bs 128 -l logdir/rep-pch16-nch32-cm1/$f -g 1 -sd 2  -of logdir/rep-pch16-nch32-cm1/eval.csv -pch 16 -nch 32 
done

for f in `seq 1 100`
do
    echo $f
    PYTHONPATH=$(pwd) python script/evaluationInj.py -ldim 32 -bs 128 -l logdir/rep-pch16-nch32-cm0/$f -g 1 -sd 2  -of logdir/rep-pch16-nch32-cm0/eval.csv -pch 16 -nch 32 
done
