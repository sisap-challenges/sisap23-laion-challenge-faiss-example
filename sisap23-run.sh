#! /bin/bash

for s in $@
do
  conda run -n faiss python -u search/search.py --size $s  
done
