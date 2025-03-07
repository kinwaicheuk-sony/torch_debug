#!/bin/bash
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK

if [ $RANK -eq 0 ]; then
    echo RANK=$RANK
    echo LOCAL_RANK=$LOCAL_RANK
    echo WORLD_SIZE=$WORLD_SIZE
    echo MASTER_ADDR=$MASTER_ADDR
fi


exec python "$@"
