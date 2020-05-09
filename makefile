#!/bin/bash

# number of threads:
for t in 1000 2000 4000 8000 16000 32000 64000 128000 256000 512000 1024000 2048000 4096000 8000000
do
	ARRAY_SIZE=$t
	g++ -DARRAY_SIZE=$t project4.cpp -o prog -lm -fopenmp
	./prog
done

