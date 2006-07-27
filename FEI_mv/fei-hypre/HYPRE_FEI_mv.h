/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/


#ifndef HYPRE_FEI_MV_HEADER
#define HYPRE_FEI_MV_HEADER

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FEI System Interface
 *
 * This interface represents a FEI linear algebaric conceptual view of a
 * linear system.  
 *
 * @memo A FEI linear-algebraic conceptual interface
 **/
/*@{*/

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/**
 * @name FEI LSI
 **/
/*@{*/

struct hypre_FEILSI_struct;

/**
 * The matrix object.
 **/

typedef struct hypre_FEILSI_struct *HYPRE_FEILSI;

/**
 * Create a LSI object.  
 **/

int HYPRE_FEILSICreate(MPI_Comm comm, HYPRE_FEILSI *lsi);

/**
 * Destroy a LSI object.  An object should be explicitly destroyed
 * using this destructor when the user's code no longer needs direct
 * access to it.  
 **/

int HYPRE_FEILSIDestroy(HYPRE_FEILSI lsi);

/**
 * Set up the row allocation information (which processor owns which rows).
 * So my processor owns row and column eqnOffsets[mypid] to
 * eqnOffsets[mypid+1]-1.
 **/

int HYPRE_FEILSISetGlobalOffsets(HYPRE_FEILSI lsi, 
                                 int *eqnOffsets);

/**
 * Set up the sparsity pattern of the local matrix. 
 **/

int HYPRE_FEILSISetMatrixStructure(HYPRE_FEILSI lsi,
                                   int** colIndices, 
                                   int* rowLengths);

/**
 * sumIntoSystemMatrix:
 * The coefficients 'values' are to be accumumlated into (added to any 
 * values already in place) 0-based equation 'row' of the matrix.
 **/

int HYPRE_FEILSISumIntoSystemMatrix(HYPRE_FEILSI lsi,
                                    int nRows, const int* rowIndices,
                                    int nCols, const int* colIndices,
                                    const double* const* values);

/**
 * putIntoSystemMatrix:
 * The coefficients 'values' are to be put the given row.
 **/

int HYPRE_FEILSIPutIntoSystemMatrix(HYPRE_FEILSI lsi,
                                    int nRows, const int* rowIndices,
                                    int nCols, const int* colIndices,
                                    const double* const* values);

/**
 * sumIntoRHSVector:
 * The coefficients 'values' are to be accumumlated into (added to any 
 * values already in place) 0-based equation 'row' of the vector.
 **/

int HYPRE_FEILSISumIntoRHSVector(HYPRE_FEILSI lsi,
                                 int nRows, const double* values,
                                 const int* colIndices);

/**
 * putIntoRHSVector:
 * The coefficients 'values' are to put into corresponding rows
 **/

int HYPRE_FEILSIPutIntoRHSVector(HYPRE_FEILSI lsi,
                                 int nRows, const double* values,
                                 const int* colIndices);

/**
 * Finalize the construction of the matrix before using.
 **/

int HYPRE_FEILSIAssemble(HYPRE_FEILSI lsi);
   
#ifdef __cplusplus
}
#endif

#endif


