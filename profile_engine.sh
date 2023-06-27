#!/bin/bash

ENGINE_PATH=${1:-'pallet_model.engine'}
SHAPE="1x3x256x256"

/usr/src/tensorrt/bin/trtexec \
  --loadEngine=$ENGINE_PATH \
  --shapes=input:$SHAPE \
  --useSpinWait \
  --useCudaGraph