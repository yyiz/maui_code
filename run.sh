#!/bin/bash

if [ -z "$2" ]; then
	echo "Using default constants.h"
else
	cp settings/"$2" inc/constants.h
fi

if [ "$1" = "cleanrun" ]; then
    make clean 
    make
    ./maui
    rm inc/constants.h
elif [ "$1" = "run" ]; then
    ./maui
else
    echo "Invalid Argument"
fi