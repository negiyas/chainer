
export NCCL_ROOT="$HOME/usr/local"
export CPATH="$NCCL_ROOT/include:$CPATH"
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$LD_LIBRARY_PATH"
export LDFLAGS="-L$NCCL_ROOT/lib"
export LIBRARY_PATH="$NCCL_ROOT/lib:$LIBRARY_PATH"

python setup.py install 2>&1 | tee -a build.log
pip install -e . 2>&1 | tee -a build.log
