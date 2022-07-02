/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/***************************************************************************
  Module:  LLNL_FEI_Matrix.h
  Purpose: custom implementation of the FEI/Matrix 
 ***************************************************************************/

#ifndef _LLNL_FEI_MATRIX_H_
#define _LLNL_FEI_MATRIX_H_

#include "_hypre_utilities.h"
#include "HYPRE.h"

/**************************************************************************
 definition of the class to capture the FEI matrix information 
---------------------------------------------------------------------------*/

class LLNL_FEI_Matrix
{
   MPI_Comm mpiComm_;
   int      mypid_;
   int      outputLevel_;

   int    localNRows_;
   int    nConstraints_;
   int    extNRows_;
   int    *constrEqns_;
   int    *globalEqnOffsets_;
   int    *globalCROffsets_;
   int    *extColMap_;

   int    *diagIA_;
   int    *diagJA_;
   double *diagAA_;
   int    *offdIA_;
   int    *offdJA_;
   double *offdAA_;
   double *diagonal_;

   int    nRecvs_;
   int    *recvLengs_;
   int    *recvProcs_;
   int    *recvProcIndices_;
   double *dRecvBufs_;
   double *dExtBufs_;

   int    nSends_;
   int    *sendLengs_;
   int    *sendProcs_;
   int    *sendProcIndices_;
   double *dSendBufs_;
   MPI_Request *mpiRequests_;

   int    FLAG_PrintMatrix_;
   int    FLAG_MatrixOverlap_;

public :

   LLNL_FEI_Matrix(MPI_Comm comm);
   ~LLNL_FEI_Matrix();

   int     parameters(int numParams, char **paramString);

   int     resetMatrix(double s);

   int     setMatrix(int nRows, int *diagIA, int *diagJA, double *diagAA, 
                     int nExtRows, int *colMap, int *offdIA, int *offdJA, 
                     double *offdAA, double *diagonal, int *eqnOffsets,
                     int *crOffsets);

   int     setCommPattern(int nRecvs, int *recvLengs, int *recvProcs, 
                          int *recvProcIndices, int nSends, int *sendLengs, 
                          int *sendProcs, int *sendProcIndices);

   int     setComplete();

   int     setConstraints(int nConstraints, int *constEqns);

   int     residualNorm(int whichNorm, double *solnVector, double *rhsVector, 
                        double* norms);

   int     getNumLocalRows() {return localNRows_;}
   int     getNumExtRows() {return extNRows_;}
   int     *getEqnOffsets() {return globalEqnOffsets_;}
   double *getMatrixDiagonal() {return diagonal_;}
   int     getLocalMatrix(int *nrows, int **ia, int **ja, double **aa) 
                         {(*nrows) = localNRows_; (*ia) = diagIA_;
                          (*ja) = diagJA_; (*aa) = diagAA_; return 0; }
   int     getExtMatrix(int *nrows, int **ia, int **ja, double **aa, int **map) 
                         {(*nrows) = extNRows_; (*ia) = offdIA_;
                          (*ja) = offdJA_; (*aa) = offdAA_; 
                          (*map) = extColMap_; return 0; }

   void    matvec(double *x, double *y);

private:
   void scatterDData(double *x);
   void gatherAddDData(double *x);
   void printMatrix();
   void matMult(int ANRows, int ANCols, int *AIA, int *AJA, double *AAA, 
                int BNRows, int BNCols, int *BIA, int *BJA, double *BAA, 
                int *DNRows, int *DNCols, int **DIA, int **DJA, double **DAA);
   void exchangeSubMatrices();
   int  BinarySearch2(int *list, int start, int lsize, int ind);
   void IntSort(int *list1, int start, int theEnd);
   void IntSort2(int *list1, int *list2, int start, int theEnd);
   void IntSort2a(int *list1, double *list2, int start, int theEnd);
};

#endif /* endif for _LLNL_FEI_MATRIX_H_ */

