/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

/* distributed_matrix.c */
hypre_DistributedMatrix *hypre_DistributedMatrixCreate (MPI_Comm context );
HYPRE_Int hypre_DistributedMatrixDestroy (hypre_DistributedMatrix *matrix );
HYPRE_Int hypre_DistributedMatrixLimitedDestroy (hypre_DistributedMatrix *matrix );
HYPRE_Int hypre_DistributedMatrixInitialize (hypre_DistributedMatrix *matrix );
HYPRE_Int hypre_DistributedMatrixAssemble (hypre_DistributedMatrix *matrix );
HYPRE_Int hypre_DistributedMatrixSetLocalStorageType (hypre_DistributedMatrix *matrix , HYPRE_Int type );
HYPRE_Int hypre_DistributedMatrixGetLocalStorageType (hypre_DistributedMatrix *matrix );
HYPRE_Int hypre_DistributedMatrixSetLocalStorage (hypre_DistributedMatrix *matrix , void *local_storage );
void *hypre_DistributedMatrixGetLocalStorage (hypre_DistributedMatrix *matrix );
HYPRE_Int hypre_DistributedMatrixSetTranslator (hypre_DistributedMatrix *matrix , void *translator );
void *hypre_DistributedMatrixGetTranslator (hypre_DistributedMatrix *matrix );
HYPRE_Int hypre_DistributedMatrixSetAuxiliaryData (hypre_DistributedMatrix *matrix , void *auxiliary_data );
void *hypre_DistributedMatrixGetAuxiliaryData (hypre_DistributedMatrix *matrix );
HYPRE_Int hypre_DistributedMatrixPrint (hypre_DistributedMatrix *matrix );
HYPRE_Int hypre_DistributedMatrixGetLocalRange (hypre_DistributedMatrix *matrix , HYPRE_BigInt *row_start , HYPRE_BigInt *row_end, HYPRE_BigInt *col_start, HYPRE_BigInt *col_end );
HYPRE_Int hypre_DistributedMatrixGetRow (hypre_DistributedMatrix *matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Real **values );
HYPRE_Int hypre_DistributedMatrixRestoreRow (hypre_DistributedMatrix *matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Real **values );

/* distributed_matrix_ISIS.c */
HYPRE_Int hypre_InitializeDistributedMatrixISIS(hypre_DistributedMatrix *dm);
HYPRE_Int hypre_FreeDistributedMatrixISIS( hypre_DistributedMatrix *dm);
HYPRE_Int hypre_PrintDistributedMatrixISIS( hypre_DistributedMatrix *matrix );
HYPRE_Int hypre_GetDistributedMatrixLocalRangeISIS( hypre_DistributedMatrix *dm, HYPRE_BigInt *start, HYPRE_BigInt *end );
HYPRE_Int hypre_GetDistributedMatrixRowISIS( hypre_DistributedMatrix *dm, HYPRE_BigInt row, HYPRE_Int *size, HYPRE_BigInt **col_ind, HYPRE_Real **values );
HYPRE_Int hypre_RestoreDistributedMatrixRowISIS( hypre_DistributedMatrix *dm, HYPRE_BigInt row, HYPRE_Int *size, HYPRE_BigInt **col_ind, HYPRE_Real **values );

/* distributed_matrix_PETSc.c */
HYPRE_Int hypre_DistributedMatrixDestroyPETSc (hypre_DistributedMatrix *distributed_matrix );
HYPRE_Int hypre_DistributedMatrixPrintPETSc (hypre_DistributedMatrix *matrix );
HYPRE_Int hypre_DistributedMatrixGetLocalRangePETSc (hypre_DistributedMatrix *matrix , HYPRE_BigInt *start , HYPRE_BigInt *end );
HYPRE_Int hypre_DistributedMatrixGetRowPETSc (hypre_DistributedMatrix *matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Real **values );
HYPRE_Int hypre_DistributedMatrixRestoreRowPETSc (hypre_DistributedMatrix *matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Real **values );

/* distributed_matrix_parcsr.c */
HYPRE_Int hypre_DistributedMatrixDestroyParCSR ( hypre_DistributedMatrix *distributed_matrix );
HYPRE_Int hypre_DistributedMatrixInitializeParCSR ( hypre_DistributedMatrix *matrix );
HYPRE_Int hypre_DistributedMatrixPrintParCSR ( hypre_DistributedMatrix *matrix );
HYPRE_Int hypre_DistributedMatrixGetLocalRangeParCSR ( hypre_DistributedMatrix *matrix , HYPRE_BigInt *row_start , HYPRE_BigInt *row_end , HYPRE_BigInt *col_start , HYPRE_BigInt *col_end );
HYPRE_Int hypre_DistributedMatrixGetRowParCSR ( hypre_DistributedMatrix *matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Real **values );
HYPRE_Int hypre_DistributedMatrixRestoreRowParCSR ( hypre_DistributedMatrix *matrix , HYPRE_BigInt row , HYPRE_Int *size , HYPRE_BigInt **col_ind , HYPRE_Real **values );
