#!/bin/bash

ONNX_PATH=${1:-'pallet_model_v1.onnx'}
OUTPUT_PATH=${2:-'pallet_model_v1.engine'}

/usr/src/tensorrt/bin/trtexec \
  --onnx=$ONNX_PATH \
  --minShapes=input:1x3x192x192 \
  --maxShapes=input:1x3x1536x1536 \
  --optShapes=input:1x3x256x256 \
  --saveEngine=$OUTPUT_PATH