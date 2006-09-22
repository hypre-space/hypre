/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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
  Module:  LLNL_FEI_LSCore.cxx
  Purpose: custom implementation of the FEI/LSC
 **************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "utilities.h"
#include "HYPRE.h"
#include "LLNL_FEI_LSCore.h"

/**************************************************************************
 Constructor 
 -------------------------------------------------------------------------*/
LLNL_FEI_LSCore::LLNL_FEI_LSCore(SolverLib_t solverLib)
{ 
   lsc_ = NULL;

   switch (solverLib) 
   {
      case HYPRE:
           lsc_ = HYPRE_base_create(MPI_COMM_WORLD );
           if ( lsc_ == NULL ) printf("problem building HYPRE\n");
           break;                                                                                                                              
      default:
           printf("unable to determine library type in LLNL_FEI_LSCore.");
   }
}

/**************************************************************************
 destructor 
 -------------------------------------------------------------------------*/
LLNL_FEI_LSCore::~LLNL_FEI_LSCore()
{
   if (lsc_ != NULL) delete lsc_;
}

/**************************************************************************
 direct access to LSC functions 
 -------------------------------------------------------------------------*/
int LLNL_FEI_LSCore::setGlobalOffsets(int leng, int *nodeOffsets,
                                  int *eqnOffsets, int *blkEqnOffsets)
{
   return(lsc_->setGlobalOffsets(leng,nodeOffsets,eqnOffsets,blkEqnOffsets));
}

/**************************************************************************
 direct access to LSC functions 
 -------------------------------------------------------------------------*/
int LLNL_FEI_LSCore::setMatrixStructure(int **ptColIndices, int *ptRowLengths,
                 int **blkColIndices,int *blkRowLengths,int *ptRowsPerBlkRow)
{
   return(lsc_->setMatrixStructure(ptColIndices,ptRowLengths,blkColIndices,
                                   blkRowLengths,ptRowsPerBlkRow));
}

/**************************************************************************
 direct access to LSC functions 
 -------------------------------------------------------------------------*/
int LLNL_FEI_LSCore::sumIntoSystemMatrix(int nRows, const int *rows, 
				     int nCols, const int* cols,
				     const double* const* vals)
{
   return(lsc_->sumIntoSystemMatrix(nRows,rows,nCols,cols,vals));
}

/**************************************************************************
 direct access to LSC functions 
 -------------------------------------------------------------------------*/
int LLNL_FEI_LSCore::putIntoSystemMatrix(int nRows, const int *rows, 
				     int nCols, const int* cols,
				     const double* const* vals)
{
   return(lsc_->putIntoSystemMatrix(nRows,rows,nCols,cols,vals));
}

/**************************************************************************
 direct access to LSC functions 
 -------------------------------------------------------------------------*/
int LLNL_FEI_LSCore::matrixLoadComplete()
{
   return(lsc_->matrixLoadComplete());
}

/**************************************************************************
 direct access to LSC functions 
 -------------------------------------------------------------------------*/
int LLNL_FEI_LSCore::sumIntoRHSVector(int num, const double *vals, 
                                  const int *indices)
{
   return(lsc_->sumIntoRHSVector(num, vals, indices));
}

/**************************************************************************
 direct access to LSC functions 
 -------------------------------------------------------------------------*/
int LLNL_FEI_LSCore::putIntoRHSVector(int num, const double *vals, 
                                  const int *indices)
{
   return(lsc_->putIntoRHSVector(num, vals, indices));
}

/**************************************************************************
 direct access to LSC functions 
 -------------------------------------------------------------------------*/
int LLNL_FEI_LSCore::putInitialGuess(const int *eqnNumbers,
				 const double *values, int len)
{
   return(lsc_->putInitialGuess(eqnNumbers, values, len));
}

/**************************************************************************
 direct access to LSC functions 
 -------------------------------------------------------------------------*/
int LLNL_FEI_LSCore::parameters( int nParams, char **params)
{
   return(lsc_->parameters(nParams, params));
}

/**************************************************************************
 direct access to LSC functions 
 -------------------------------------------------------------------------*/
int LLNL_FEI_LSCore::solve( int *status, int *iterations)
{
   return(lsc_->launchSolver(*status, *iterations));
}

/**************************************************************************
 direct access to LSC functions 
 -------------------------------------------------------------------------*/
int LLNL_FEI_LSCore::formResidual( double* values, int leng)
{
   return(lsc_->formResidual(values, leng));
}

/**************************************************************************
 direct access to LSC functions 
 -------------------------------------------------------------------------*/
int LLNL_FEI_LSCore::getSolution( double *answers, int leng)
{
   return(lsc_->getSolution(answers, leng));
}

/**************************************************************************
 direct access to LSC functions 
 -------------------------------------------------------------------------*/
int LLNL_FEI_LSCore::getSolnEntry( int eqnNum, double *answers)
{
   return(lsc_->getSolnEntry(eqnNum, *answers));
}

