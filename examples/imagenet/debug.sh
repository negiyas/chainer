

ARCH=vgg16
BATCHSIZE=32
#OOC=2
OOC=0
USEPOOL=1
DEBUG=0

ARCH=${ARCH} BATCHSIZE=${BATCHSIZE} OOC=${OOC} USEPOOL=${USEPOOL} ITER=1 MEMPRINT=1 LOCAL=1 DEBUG=${DEBUG} GPU=2 ./go.sh 2>&1 | tee debug-${ARCH}-${BATCHSIZE}-${OOC}-${USEPOOL}.log | tee debug.log
./anal.sh debug-${ARCH}-${BATCHSIZE}-${OOC}-${USEPOOL}.log > debug-${ARCH}-${BATCHSIZE}-${OOC}-${USEPOOL}.csv

