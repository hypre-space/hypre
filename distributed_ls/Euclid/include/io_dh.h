#ifndef IO_DH
#define IO_DH

#include "euclid_common.h"

extern void readEuclid(char *filename, Mat_dh *A);
extern void writeEuclid(int n, int *rp, int *cval, double *aval, char *filename);
extern void readSCR_seq(char *filename, Mat_dh *A);

void printTriplesToFile(int n, int *rp, int *cval, double *aval,
                           int *n2o_row, int *n2o_col, char *filename);




#endif
