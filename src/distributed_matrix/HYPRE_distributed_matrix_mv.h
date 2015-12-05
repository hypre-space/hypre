/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
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
HYPRE_Int HYPRE_DistributedMatrixGetDims (HYPRE_DistributedMatrix matrix , HYPRE_Int *M , HYPRE_Int *N );
HYPRE_Int HYPRE_DistributedMatrixSetDims (HYPRE_DistributedMatrix matrix , HYPRE_Int M , HYPRE_Int N );
HYPRE_Int HYPRE_DistributedMatrixPrint (HYPRE_DistributedMatrix matrix );
HYPRE_Int HYPRE_DistributedMatrixGetLocalRange (HYPRE_DistributedMatrix matrix , HYPRE_Int *row_start , HYPRE_Int *row_end, HYPRE_Int *col_start, HYPRE_Int *col_end );
HYPRE_Int HYPRE_DistributedMatrixGetRow (HYPRE_DistributedMatrix matrix , HYPRE_Int row , HYPRE_Int *size , HYPRE_Int **col_ind , double **values );
HYPRE_Int HYPRE_DistributedMatrixRestoreRow (HYPRE_DistributedMatrix matrix , HYPRE_Int row , HYPRE_Int *size , HYPRE_Int **col_ind , double **values );

#endif
