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



/******************************************************************************
 *
 * Member functions for hypre_DistributedMatrix class for par_csr storage scheme.
 *
 *****************************************************************************/

#include "./distributed_matrix.h"

#include "HYPRE_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixDestroyParCSR
 *   Internal routine for freeing a matrix stored in Parcsr form.
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixDestroyParCSR( hypre_DistributedMatrix *distributed_matrix )
{

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixInitializeParCSR
 *--------------------------------------------------------------------------*/

  /* matrix must be set before calling this function*/

int 
hypre_DistributedMatrixInitializeParCSR(hypre_DistributedMatrix *matrix)
{
   
   return 0;
}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixPrintParCSR
 *   Internal routine for printing a matrix stored in Parcsr form.
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixPrintParCSR( hypre_DistributedMatrix *matrix )
{
   int  ierr=0;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(matrix);

   HYPRE_ParCSRMatrixPrint( Parcsr_matrix, "STDOUT" );
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetLocalRangeParCSR
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixGetLocalRangeParCSR( hypre_DistributedMatrix *matrix,
                             int *row_start,
                             int *row_end,
                             int *col_start,
                             int *col_end )
{
   int ierr=0;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(matrix);

   if (!Parcsr_matrix) return(-1);


   ierr = HYPRE_ParCSRMatrixGetLocalRange( Parcsr_matrix, row_start, row_end, 
					col_start, col_end );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetRowParCSR
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixGetRowParCSR( hypre_DistributedMatrix *matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   int ierr = 0;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(matrix);

   if (!Parcsr_matrix) return(-1);

   ierr = HYPRE_ParCSRMatrixGetRow( Parcsr_matrix, row, size, col_ind, values);

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixRestoreRowParCSR
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixRestoreRowParCSR( hypre_DistributedMatrix *matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   int ierr;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(matrix);

   if (Parcsr_matrix == NULL) return(-1);

   ierr = HYPRE_ParCSRMatrixRestoreRow( Parcsr_matrix, row, size, col_ind, values); 

   return(ierr);
}
