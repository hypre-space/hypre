/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/




/**************************************************************************
  Module:  LLNL_FEI_LSCore.h
  Purpose: custom implementation of the FEI/LSC
 **************************************************************************/

#ifndef _LLNL_FEI_LSCORE_H_
#define _LLNL_FEI_LSCORE_H_

#include "FEI_mv/fei-base/fei_defs.h"
#include "FEI_mv/fei-base/Data.h"
#include "FEI_mv/fei-base/Lookup.h"
#include "FEI_mv/fei-base/LinearSystemCore.h"
#include "cfei_hypre.h"

typedef enum {
  UNDEFINED_SolverLib = -1,
  HYPRE =2
} SolverLib_t;

/******************************************************************************
  This is the definition for the base LLNL_FEI_LSCore class. 
  *****************************************************************************/

class LLNL_FEI_LSCore
{
  private :

  public :

  LinearSystemCore *lsc_;

  LLNL_FEI_LSCore( SolverLib_t tmp );
  ~LLNL_FEI_LSCore();

  int setGlobalOffsets(int leng, int *nodeOffsets, int *eqnOffsets, 
                       int *blkEqnOffsets);
  
  int setMatrixStructure(int **ptColIndices, int *ptRowLengths, 
                         int **blkColIndices, int *blkRowLengths, 
                         int *ptRowsPerBlkRow);
  
  int sumIntoSystemMatrix(int nRows, const int *rows, int nCols, 
                          const int* cols, const double* const* vals);
  
  int putIntoSystemMatrix(int nRows, const int *rows, int nCols, 
                          const int* cols, const double* const* vals);
  
  int matrixLoadComplete();
  
  int sumIntoRHSVector(int num, const double *vals, const int *indices);
  
  int putIntoRHSVector(int num, const double *vals, const int *indices);
  
  int putInitialGuess(const int *eqnNumbers, const double *values, int leng);
  
  int parameters(int nParams, char **params);
  
  int solve(int *status, int *iterations);
  
  int formResidual( double* values, int leng);

  int getSolution(double *answers, int leng);
  
  int getSolnEntry(int eqnNum, double *answers);

};

#endif /* endif for _LLNL_FEI_LSCORE_H_ */

