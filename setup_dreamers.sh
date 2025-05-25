#!/bin/bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa  
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
