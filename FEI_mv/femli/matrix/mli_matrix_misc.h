/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * utility functions
 *
 *****************************************************************************/

#include "utilities/utilities.h"
#include "matrix/mli_matrix.h"

extern int  MLI_Matrix_ComputePtAP(MLI_Matrix *P,MLI_Matrix *A,MLI_Matrix **);
extern int  MLI_Matrix_FormJacobi(MLI_Matrix *A, double alpha, MLI_Matrix **J);
extern int  MLI_Matrix_Compress(MLI_Matrix *A, int blksize, MLI_Matrix **A2);
extern int  MLI_Matrix_GetSubMatrix(MLI_Matrix *A, int nRows, int *rowIndices,
                       int *newNRows, double **newAA);
extern int  MLI_Matrix_GetOverlappedMatrix(MLI_Matrix *, int *offNRows, 
                       int **offRowLengs, int **offCols, double **offVals);

extern void MLI_Matrix_GetExtRows(MLI_Matrix *, MLI_Matrix *, int *extNRows,
                       int **extRowLengs, int **extCols, double **extVals);
extern void MLI_Matrix_MatMatMult(MLI_Matrix *, MLI_Matrix *, MLI_Matrix **);

