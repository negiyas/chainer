export OOC=2
export BATCHSIZE=100000000
export USEPOOL=0
ITER=1 MEMCALL=1 GPU=3 ./go.sh 2>&1 | tee debug-OOC${OOC}-POOL${USEPOOL}-${BATCHSIZE}.log
./anal.sh  debug-OOC${OOC}-POOL${USEPOOL}-${BATCHSIZE}.log > debug-OOC${OOC}-POOL${USEPOOL}-${BATCHSIZE}.csv
