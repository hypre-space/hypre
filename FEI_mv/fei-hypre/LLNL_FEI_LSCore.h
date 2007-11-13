/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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

