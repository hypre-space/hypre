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


/******************************************************************************
 *
 * Header info for the hypre_DistributedMatrix structures
 *
 *****************************************************************************/

#ifndef HYPRE_DISTRIBUTED_MATRIX_MV_HEADER
#define HYPRE_DISTRIBUTED_MATRIX_MV_HEADER


typedef void *HYPRE_DistributedMatrix;

/* HYPRE_distributed_matrix.c */
int HYPRE_DistributedMatrixCreate (MPI_Comm context, HYPRE_DistributedMatrix *matrix );
int HYPRE_DistributedMatrixDestroy (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixLimitedDestroy (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixInitialize (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixAssemble (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixSetLocalStorageType (HYPRE_DistributedMatrix matrix , int type );
int HYPRE_DistributedMatrixGetLocalStorageType (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixSetLocalStorage (HYPRE_DistributedMatrix matrix , void *LocalStorage );
void *HYPRE_DistributedMatrixGetLocalStorage (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixSetTranslator (HYPRE_DistributedMatrix matrix , void *Translator );
void *HYPRE_DistributedMatrixGetTranslator (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixSetAuxiliaryData (HYPRE_DistributedMatrix matrix , void *AuxiliaryData );
void *HYPRE_DistributedMatrixGetAuxiliaryData (HYPRE_DistributedMatrix matrix );
MPI_Comm HYPRE_DistributedMatrixGetContext (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixGetDims (HYPRE_DistributedMatrix matrix , int *M , int *N );
int HYPRE_DistributedMatrixSetDims (HYPRE_DistributedMatrix matrix , int M , int N );
int HYPRE_DistributedMatrixPrint (HYPRE_DistributedMatrix matrix );
int HYPRE_DistributedMatrixGetLocalRange (HYPRE_DistributedMatrix matrix , int *row_start , int *row_end, int *col_start, int *col_end );
int HYPRE_DistributedMatrixGetRow (HYPRE_DistributedMatrix matrix , int row , int *size , int **col_ind , double **values );
int HYPRE_DistributedMatrixRestoreRow (HYPRE_DistributedMatrix matrix , int row , int *size , int **col_ind , double **values );

#endif
