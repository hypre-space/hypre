/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the hypre_DistributedMatrix structures
 *
 *****************************************************************************/

#ifndef HYPRE_DISTRIBUTED_MATRIX_MV_HEADER
#define HYPRE_DISTRIBUTED_MATRIX_MV_HEADER


typedef void *HYPRE_DistributedMatrix;

/* HYPRE_distributed_matrix.c */
HYPRE_Int HYPRE_DistributedMatrixCreate (MPI_Comm context, HYPRE_DistributedMatrix *matrix );
HYPRE_Int HYPRE_DistributedMatrixDestroy (HYPRE_DistributedMatrix matrix );
HYPRE_Int HYPRE_DistributedMatrixLimitedDestroy (HYPRE_DistributedMatrix matrix );
HYPRE_Int HYPRE_DistributedMatrixInitialize (HYPRE_DistributedMatrix matrix );
HYPRE_Int HYPRE_DistributedMatrixAssemble (HYPRE_DistributedMatrix matrix );
HYPRE_Int HYPRE_DistributedMatrixSetLocalStorageType (HYPRE_DistributedMatrix matrix , HYPRE_Int type );
HYPRE_Int HYPRE_DistributedMatrixGetLocalStorageType (HYPRE_DistributedMatrix matrix );
HYPRE_Int HYPRE_DistributedMatrixSetLocalStorage (HYPRE_DistributedMatrix matrix , void *LocalStorage );
void *HYPRE_DistributedMatrixGetLocalStorage (HYPRE_DistributedMatrix matrix );
HYPRE_Int HYPRE_DistributedMatrixSetTranslator (HYPRE_DistributedMatrix matrix , void *Translator );
void *HYPRE_DistributedMatrixGetTranslator (HYPRE_DistributedMatrix matrix );
HYPRE_Int HYPRE_DistributedMatrixSetAuxiliaryData (HYPRE_DistributedMatrix matrix , void *AuxiliaryData );
void *HYPRE_DistributedMatrixGetAuxiliaryData (HYPRE_DistributedMatrix matrix );
MPI_Comm HYPRE_DistributedMatrixGetContext (HYPRE_DistributedMatrix matrix );
HYPRE_Int HYPRE_DistributedMatrixGetDims (HYPRE_DistributedMatrix matrix , HYPRE_BigInt *M , HYPRE_BigInt *N );
HYPRE_Int HYPRE_DistributedMatrixSetDims (HYPRE_DistributedMatrix matrix , HYPRE_BigInt M , HYPRE_BigInt N );
HYPRE_Int HYPRE_DistributedMatrixPrint (HYPRE_DistributedMatrix matrix );
HYPRE_Int HYPRE_DistributedMatrixGetLocalRange (HYPRE_DistributedMatrix matrix , HYPRE_BigInt *row_start , HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int HYPRE_DistributedMatrixGetRow (HYPRE_DistributedMatrix matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Real **values );
HYPRE_Int HYPRE_DistributedMatrixRestoreRow (HYPRE_DistributedMatrix matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Real **values );

#endif
