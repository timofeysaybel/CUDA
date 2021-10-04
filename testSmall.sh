#!/bin/bash

for (( i = 1; i <= 5; i++))
do
	./ImageConvolution gaussian small/small$i
	./ImageConvolution edge small/small$i
	./ImageConvolution sharpen small/small$i
done

