#!/bin/csh
mpiCC -c -DHYPRE_TIMING -I../../../hypre/include -I.. -I../../.. driver.c
mpiCC -o driver driver.o -L../../../hypre/lib -L/usr/lib -lHYPRE -lHYPRE_superlu -lHYPRE_LSI -lg2c -lm

