
# command to execute convnet. You can set the following varaibles .
#     ARCH: list of architectures (e.g. "alexnet googlenet")
#     INSIZE: list of input data sizes (e.g. "224 2240")
#     BATCHSIZE: list of batchsizes (e.g. "10 20 30")
#     ITER: number of iteration
#     QUEUE: queue for LSF (e.g. "shared" or "excl")
#     LOCAL: non-empty for direct execution (otherwise use LSF)
#     DEBUG: non-empty to enter debugger at first
#     PROFILE: non-empty for profiler
#     LOG: Log directory name (Do not include "-" in the name)

TIME=`date +%Y%m%d-%H%M%S`
DATE=`date +%Y%m%d`

export NCCL_ROOT="$HOME/usr/local"
export CUDNN_ROOT="$HOME/use/cudnn/cuda/targets/ppc64le-linux"
export CPATH="$NCCL_ROOT/include:$CUDNN_ROOT/include:$CPATH"
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$NCCL_ROOT/lib64:$CUDNN_ROOT/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$NCCL_ROOT/lib:$NCCL_ROOT/lib64:$CUDNN_ROOT/lib:$LIBRARY_PATH"

# imagenet 1k
DATA_ROOT=/home/negishi/use/imagenet
DATA_TRAIN_FILE=${DATA_ROOT}/training.txt
DATA_VAL_FILE=${DATA_ROOT}/validation.txt
DATA_MEAN_FILE=${DATA_ROOT}/mean.npy

#### Benchmark parameters
ARCH_LIST=${ARCH:-"alex vgga vgg16 vgg19 googlenet resnet50 resnet101 resnet152"}
ARCH_LIST=${ARCH:-"resnet50"}
BATCHSIZE_LIST=${BATCHSIZE:-"32 64 96 128 160 192 224 256 288 320 352 384 416 448 480 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 1024 1056 1088 1120 1152 1184 1216 1248 1280 1312 1344 1376 1408 1440 1472 1504 1536 1568 1600 1632 1664 1696 1728 1760 1792 1824 1856 1888 1920 1952 1984 2016 2048 2080 2112 2144 2176 2208 2240 2272 2304 2336 2368 2400 2432 2464 2496 2528 2560 2592 2624 2656 2688 2720 2752 2784 2816 2848 2880 2912 2944 2976 3008 3040 3072 3104 3136 3168 3200 3232 3264 3296 3328 3360 3392 3424 3456 3488 3520 3552 3584 3616 3648 3680 3712 3744 3776 3808 3840 3872 3904 3936 3968 4000 4032 4064 4096"}
BATCHSIZE_LIST=${BATCHSIZE:-"32"}
VAL_BATCHSIZE="10"
EPOCH=${EPOCH:-1}
GPU=${GPU:-0}
AUTO=${AUTO:-0}
AUTOWS=${AUTOWS:-0}
SYNTHETIC=${SYNTHETIC:-1}
MEMORYTRACE=${MEMTRACE:-0}
LOGDIR=${LOG:-"LOG-${DATE}"}
QUEUE=${QUEUE:-"excl"}
PROFILEOPT=${PROFILE:-0}
ITERATION=${ITER:-0}
LOCAL=${LOCAL:-1}
DEBUG=${DEBUG:-0}
TRACE=${TRACE:-0}
MEMPROF=${MEMPROF:-0}
MEMCALL=${MEMCALL:-0}
MEMPRINT=${MEMPRINT:-0}
TIMER=${TIMER:-0}
PRINTVAR=${PRINTVAR:-0}
LMS=${LMS:-0}
USEPOOL=${USEPOOL:-1}
INSIZE=224 # cannot be changed now

MAIN=${MAIN:-"train_imagenet_synthetic.py"}

if [ "${DEBUG}" != "0" -o "${LOCAL}" != "0" ]; then
    LSB_BATCH_JID=0
fi

if [ "${PROFILEOPT}" == "2" ]; then
    PROFILEPREFIX="nvprof --unified-memory-profiling per-process-device"
elif [ "${PROFILEOPT}" == "1" ]; then
    PROFILEPREFIX="nvprof"
fi

SYNTHOPT=""
if [ "${SYNTHETIC}" != "0" ]; then
    DATAARGS="--synthetic data val"
else
    DATAARGS="--root ${DATA_ROOT} --mean ${DATA_MEAN_FILE} ${DATA_TRAIN_FILE} ${DATA_VAL_FILE}"
fi

TRACEOPT=""
if [ "${TRACE}" != "0" ]; then
    TRACEOPT="--log"
fi

ITEROPT=""
if [ "${ITERATION}" != "0" ]; then
    ITEROPT="--iteration ${ITERATION}"
else
    ITEROPT="--epoch ${EPOCH}"
fi


ptile=${1:-4}
mkdir -p ${LOGDIR}
if [ -z $LSB_BATCH_JID ]; then
    pyenv local > ${LOGDIR}/${TIME}.pyenv
    /bin/cp -f $0 ${LOGDIR}/${TIME}.sh
    set -x
    bsub -n 1 \
     -e ${LOGDIR}/${TIME}-%J-lsf.err \
     -o ${LOGDIR}/${TIME}-%J-lsf.out \
     -q ${QUEUE} \
     -W 24:00 \
     -R "span[ptile=$ptile]" -R "rusage[ngpus_shared=4]" $0
else

for ARCH in ${ARCH_LIST}
do
for BATCHSIZE in ${BATCHSIZE_LIST}
do

LOGBASE="${LOGDIR}/${TIME}-${ARCH}-`printf %04d ${INSIZE}`-`printf %04d ${BATCHSIZE}`"
[ "${PROFILEOPT}" != "0" ] && PROFILECMD="${PROFILEPREFIX} -o $LOGBASE.nvvp"

CMD="$PROFILECMD python ${MAIN} --arch ${ARCH} --batchsize ${BATCHSIZE} --gpu ${GPU} ${LMSOPT} ${AUTOOPT} ${AUTOWSOPT} ${MEMTRACEOPT} ${ITEROPT} --loaderjob 4 --val_batchsize ${VAL_BATCHSIZE} ${TRACEOPT} --memprof ${MEMPROF} --memcallstack ${MEMCALL} --memprint ${MEMPRINT} --timer ${TIMER} --printvar ${PRINTVAR} --lms ${LMS} --usepool ${USEPOOL} -o ${LOGBASE}.result ${DATAARGS}"

if [ "${DEBUG}" != "0" ]; then
    $CMD --debug
else
    (echo $CMD; $CMD) 2>&1 | tee ${LOGBASE}.log | tee go.log
fi

done
done

fi
