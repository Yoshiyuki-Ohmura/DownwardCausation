

# epoch = 1000
#PYTHONPATH=$(pwd) python script/train_bisection_transInj.py -ldim 32 -bs 128  -g 1 -e 1000  -l logdir/p16-n64/01 -sd 1 -lr 1e-4 -pch 16 -nch 64 -cm 1.

#PYTHONPATH=$(pwd) python script/train_bisection_transInj.py -ldim 32 -bs 128  -g 1 -e 1000  -l logdir/p16-n64-cm0/01 -sd 1 -lr 1e-4 -pch 16 -nch 64 -cm 0.

#PYTHONPATH=$(pwd) python script/train_bisection_transInj.py -ldim 32 -bs 128  -g 1 -e 1000  -l logdir/p64-n32-cm0/01 -sd 1 -lr 1e-4 -pch 64 -nch 32 -cm 0.

#PYTHONPATH=$(pwd) python script/train_bisection_transInj.py -ldim 32 -bs 128  -g 1 -e 400  -l logdir/p64-n32-cm0/01 -sd 1 -lr 1e-4 -pch 64 -nch 32 -cm 0.

#PYTHONPATH=$(pwd) python script/train_bisection_transInj.py -ldim 32 -bs 128  -g 1 -e 1000  -l logdir/p32-n32-cm0/01 -sd 1 -lr 1e-4 -pch 32 -nch 32 -cm 0.


#for i in `seq 1 100`
#	 do
#	     PYTHONPATH=$(pwd) python script/train_bisection_transInj.py -ldim 32 -bs 128  -g 1 -e 400  -l logdir/rep-pch16-nch32-cm1/$i -sd $i -lr 1e-4 -pch 16 -nch 32 -cm 1. 
#done

#for i in `seq 1 100`
#	 do
#	     PYTHONPATH=$(pwd) python script/train_bisection_transInj.py -ldim 32 -bs 128  -g 1 -e 400  -l logdir/rep-pch16-nch32-cm0/$i -sd $i -lr 1e-4 -pch 16 -nch 32 -cm 0. 
#done


