#!/bin/bash

# Simple wrapper script to run Makefile targets

if [ "$1" == "setup" ]; then
    make setup
elif [ "$1" == "train" ]; then
    make train
elif [ "$1" == "run" ]; then
    make run
elif [ "$1" == "clean" ]; then
    make clean
elif [ "$1" == "all" ]; then
    make all
else
    echo "Usage: ./run.sh {setup|train|run|clean|all}"
fi
