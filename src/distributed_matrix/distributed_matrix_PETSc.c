/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

HYPRE_Int
hypre_DistributedMatrixDestroyPETSc( hypre_DistributedMatrix *dm )
{
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(dm);

   MatDestroy( PETSc_matrix );
#else
   HYPRE_UNUSED_VAR(dm);
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

HYPRE_Int
hypre_DistributedMatrixPrintPETSc( hypre_DistributedMatrix *dm )
{
   HYPRE_Int  ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(dm);

   ierr = MatView( PETSc_matrix, VIEWER_STDOUT_WORLD );
#else
   HYPRE_UNUSED_VAR(dm);
#endif

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetLocalRangePETSc
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DistributedMatrixGetLocalRangePETSc( hypre_DistributedMatrix *dm,
                             HYPRE_BigInt *start,
                             HYPRE_BigInt *end )
{
   HYPRE_Int ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(dm);

   if (!PETSc_matrix) return(-1);


   ierr = MatGetOwnershipRange( PETSc_matrix, start, end ); CHKERRA(ierr);
/*

  Since PETSc's MatGetOwnershipRange actually returns
  end = "one more than the global index of the last local row",
  we need to subtract one; hypre assumes we return the index
  of the last row itself.

*/
   *end = *end - 1;
#else
   HYPRE_UNUSED_VAR(dm);
   HYPRE_UNUSED_VAR(start);
   HYPRE_UNUSED_VAR(end);
#endif

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetRowPETSc
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DistributedMatrixGetRowPETSc( hypre_DistributedMatrix *dm,
                             HYPRE_BigInt row,
                             HYPRE_Int *size,
                             HYPRE_BigInt **col_ind,
                             HYPRE_Real **values )
{
   HYPRE_Int ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(dm);

   if (!PETSc_matrix) return(-1);

   ierr = MatGetRow( PETSc_matrix, row, size, col_ind, values); CHKERRA(ierr);
#else
   HYPRE_UNUSED_VAR(dm);
   HYPRE_UNUSED_VAR(row);
   HYPRE_UNUSED_VAR(size);
   HYPRE_UNUSED_VAR(col_ind);
   HYPRE_UNUSED_VAR(values);
#endif

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixRestoreRowPETSc
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DistributedMatrixRestoreRowPETSc( hypre_DistributedMatrix *dm,
                             HYPRE_BigInt row,
                             HYPRE_Int *size,
                             HYPRE_BigInt **col_ind,
                             HYPRE_Real **values )
{
   HYPRE_Int ierr=0;
#ifdef PETSC_AVAILABLE
   Mat PETSc_matrix = (Mat) hypre_DistributedMatrixLocalStorage(dm);

   if (PETSc_matrix == NULL) return(-1);

   ierr = MatRestoreRow( PETSc_matrix, row, size, col_ind, values); CHKERRA(ierr);
#else
   HYPRE_UNUSED_VAR(dm);
   HYPRE_UNUSED_VAR(row);
   HYPRE_UNUSED_VAR(size);
   HYPRE_UNUSED_VAR(col_ind);
   HYPRE_UNUSED_VAR(values);
#endif

   return(ierr);
}
