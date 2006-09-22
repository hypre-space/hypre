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
 * Member functions for hypre_DistributedMatrix class for PETSc storage scheme.
 *
 *****************************************************************************/

#include "./distributed_matrix.h"

/* Public headers and prototypes for PETSc matrix library */
#ifdef PETSC_AVAILABLE
#include "sles.h"
#endif

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixDestroyPETSc
 *   Internal routine for freeing a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixDestroyPETSc( hypre_DistributedMatrix *distributed_matrix )
{
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(distributed_matrix);

   MatDestroy( PETSc_matrix );
#endif

   return(0);
}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixPrintPETSc
 *   Internal routine for printing a matrix stored in PETSc form.
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixPrintPETSc( hypre_DistributedMatrix *matrix )
{
   int  ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(matrix);

   ierr = MatView( PETSc_matrix, VIEWER_STDOUT_WORLD );
#endif
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetLocalRangePETSc
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixGetLocalRangePETSc( hypre_DistributedMatrix *matrix,
                             int *start,
                             int *end )
{
   int ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(matrix);

   if (!PETSc_matrix) return(-1);


   ierr = MatGetOwnershipRange( PETSc_matrix, start, end ); CHKERRA(ierr);
/*

  Since PETSc's MatGetOwnershipRange actually returns 
  end = "one more than the global index of the last local row",
  we need to subtract one; hypre assumes we return the index
  of the last row itself.

*/
   *end = *end - 1;
#endif

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetRowPETSc
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixGetRowPETSc( hypre_DistributedMatrix *matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   int ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(matrix);

   if (!PETSc_matrix) return(-1);

   ierr = MatGetRow( PETSc_matrix, row, size, col_ind, values); CHKERRA(ierr);
#endif

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixRestoreRowPETSc
 *--------------------------------------------------------------------------*/

int 
hypre_DistributedMatrixRestoreRowPETSc( hypre_DistributedMatrix *matrix,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   int ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(matrix);

   if (PETSc_matrix == NULL) return(-1);

   ierr = MatRestoreRow( PETSc_matrix, row, size, col_ind, values); CHKERRA(ierr);
#endif

   return(ierr);
}
