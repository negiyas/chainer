
# command to execute convnet. You can set the following varaibles .
#     ARCH: list of architectures (e.g. "alexnet googlenet")
#     INSIZE: list of input data sizes (e.g. "224 2240")
#     BATCHSIZE: list of batchsizes (e.g. "10 20 30")
#     UNIFIED: list of unified memory modes (0: notuse, 1:use)
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
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$CUDNN_ROOT/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

#### Benchmark parameters
BATCHSIZE_LIST=${BATCHSIZE:-"32"}
GPU=${GPU:-0}
LOGDIR=${LOG:-"LOG-${DATE}"}
QUEUE=${QUEUE:-"excl"}
DEBUG=${DEBUG:-""}
PROFILEOPT=${PROFILE:-0}
LOCAL=1

MAIN="train_mnist.py"

if [ "${DEBUG}" != "" -o "${LOCAL}" != "" ]; then
    LSB_BATCH_JID=0
fi

if [ "${PROFILEOPT}" == "2" ]; then
    PROFILEPREFIX="nvprof --unified-memory-profiling per-process-device"
elif [ "${PROFILEOPT}" == "1" ]; then
    PROFILEPREFIX="nvprof"
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

for BATCHSIZE in ${BATCHSIZE_LIST}
do

LOGBASE="${LOGDIR}/${TIME}-`printf %04d ${BATCHSIZE}`"
[ "${PROFILEOPT}" != "0" ] && PROFILECMD="${PROFILEPREFIX} -o $LOGBASE.nvvp"

CMD="$PROFILECMD python ${MAIN} --batchsize ${BATCHSIZE} --gpu ${GPU} ${MEMTRACEOPT} --noplot -o ${LOGBASE}.result"

if [ "${DEBUG}" != "" ]; then
    echo $CMD
    $CMD --debug
else
    (echo $CMD; $CMD) 2>&1 | tee ${LOGBASE}.log | tee go.log
fi

done

fi
