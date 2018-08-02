
export ARCH=vgg16
export LOCAL=0
export ITER=5
export MEMPRINT=1
export QUEUE=shared

#OOC=0 USEPOOL=0 LOG=${ARCH}-ORG-nopool ./go.sh
#OOC=1 USEPOOL=0 LOG=${ARCH}-OOC-nopool ./go.sh
OOC=0 USEPOOL=1 LOG=${ARCH}-ORGNEW2-${QUEUE} ./go.sh
OOC=2 USEPOOL=1 LOG=${ARCH}-OOCNEW2-${QUEUE} ./go.sh
