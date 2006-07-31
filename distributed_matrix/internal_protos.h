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

/* distributed_matrix.c */
hypre_DistributedMatrix *hypre_DistributedMatrixCreate (MPI_Comm context );
int hypre_DistributedMatrixDestroy (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixLimitedDestroy (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixInitialize (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixAssemble (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixSetLocalStorageType (hypre_DistributedMatrix *matrix , int type );
int hypre_DistributedMatrixGetLocalStorageType (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixSetLocalStorage (hypre_DistributedMatrix *matrix , void *local_storage );
void *hypre_DistributedMatrixGetLocalStorage (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixSetTranslator (hypre_DistributedMatrix *matrix , void *translator );
void *hypre_DistributedMatrixGetTranslator (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixSetAuxiliaryData (hypre_DistributedMatrix *matrix , void *auxiliary_data );
void *hypre_DistributedMatrixGetAuxiliaryData (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixPrint (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixGetLocalRange (hypre_DistributedMatrix *matrix , int *row_start , int *row_end, int *col_start, int *col_end );
int hypre_DistributedMatrixGetRow (hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values );
int hypre_DistributedMatrixRestoreRow (hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values );

/* distributed_matrix_ISIS.c */
int hypre_InitializeDistributedMatrixISIS(hypre_DistributedMatrix *dm);
int hypre_FreeDistributedMatrixISIS( hypre_DistributedMatrix *dm);
int hypre_PrintDistributedMatrixISIS( hypre_DistributedMatrix *matrix );
int hypre_GetDistributedMatrixLocalRangeISIS( hypre_DistributedMatrix *dm, int *start, int *end );
int hypre_GetDistributedMatrixRowISIS( hypre_DistributedMatrix *dm, int row, int *size, int **col_ind, double **values );
int hypre_RestoreDistributedMatrixRowISIS( hypre_DistributedMatrix *dm, int row, int *size, int **col_ind, double **values );

/* distributed_matrix_PETSc.c */
int hypre_DistributedMatrixDestroyPETSc (hypre_DistributedMatrix *distributed_matrix );
int hypre_DistributedMatrixPrintPETSc (hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixGetLocalRangePETSc (hypre_DistributedMatrix *matrix , int *start , int *end );
int hypre_DistributedMatrixGetRowPETSc (hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values );
int hypre_DistributedMatrixRestoreRowPETSc (hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values );

/* distributed_matrix_parcsr.c */
int hypre_DistributedMatrixDestroyParCSR ( hypre_DistributedMatrix *distributed_matrix );
int hypre_DistributedMatrixInitializeParCSR ( hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixPrintParCSR ( hypre_DistributedMatrix *matrix );
int hypre_DistributedMatrixGetLocalRangeParCSR ( hypre_DistributedMatrix *matrix , int *row_start , int *row_end , int *col_start , int *col_end );
int hypre_DistributedMatrixGetRowParCSR ( hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values );
int hypre_DistributedMatrixRestoreRowParCSR ( hypre_DistributedMatrix *matrix , int row , int *size , int **col_ind , double **values );
