#! /bin/bash

JULIA_NUM_THREADS=auto JULIA_PROJECT=. julia src/run.jl "$@"
