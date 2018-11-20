
ARCH=vgg16
BATCHSIZE=128
ITER=10
MEMPRINT=1

LMS_LIST="0 2"
USEPOOL_LIST="0 1"

for LMS in $LMS_LIST
do
for USEPOOL in $USEPOOL_LIST
do

ARCH=$ARCH BATCHSIZE=$BATCHSIZE ITER=$ITER MEMPRINT=1 LMS=$LMS USEPOOL=$USEPOOL ./go.sh 2>&1 | tee go-arch${ARCH}-batch${BATCHSIZE}-lms${LMS}-pool${USEPOOL}.log
./anal.sh go-lms${LMS}-pool${USEPOOL}.log > go-arch${ARCH}-batch${BATCHSIZE}-lms${LMS}-pool${USEPOOL}.csv

done
done 2>&1 | tee go-all.log
