#ifndef GET_ROW_DH
#define GET_ROW_DH

#include "euclid_common.h"

/* "row" refers to global row number */

extern void EuclidGetDimensions(void *A, int *beg_row, int *rowsLocal, int *rowsGlobal);
extern void EuclidGetRow(void *A, int row, int *len, int **ind, double **val);
extern void EuclidRestoreRow(void *A, int row, int *len, int **ind, double **val);

extern int EuclidReadLocalNz(void *A);

extern void PrintMatUsingGetRow(void* A, int beg_row, int m,
                          int *n2o_row, int *n2o_col, char *filename);


#endif

