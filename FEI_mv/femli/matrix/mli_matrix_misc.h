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





/******************************************************************************
 *
 * utility functions
 *
 *****************************************************************************/

#include "utilities/_hypre_utilities.h"
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
extern void MLI_Matrix_Transpose(MLI_Matrix *, MLI_Matrix **);

