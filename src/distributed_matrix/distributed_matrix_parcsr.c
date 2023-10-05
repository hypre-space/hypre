/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

HYPRE_Int
hypre_DistributedMatrixDestroyParCSR( hypre_DistributedMatrix *dm )
{
   HYPRE_UNUSED_VAR(dm);

   return(0);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixInitializeParCSR
 *--------------------------------------------------------------------------*/

  /* matrix must be set before calling this function*/

HYPRE_Int
hypre_DistributedMatrixInitializeParCSR(hypre_DistributedMatrix *dm)
{
   HYPRE_UNUSED_VAR(dm);

   return 0;
}

/*--------------------------------------------------------------------------
 * Optional routines that depend on underlying storage type
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixPrintParCSR
 *   Internal routine for printing a matrix stored in Parcsr form.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DistributedMatrixPrintParCSR( hypre_DistributedMatrix *dm )
{
   HYPRE_Int  ierr=0;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(dm);

   HYPRE_ParCSRMatrixPrint( Parcsr_matrix, "STDOUT" );
   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetLocalRangeParCSR
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DistributedMatrixGetLocalRangeParCSR( hypre_DistributedMatrix *dm,
                                            HYPRE_BigInt            *row_start,
                                            HYPRE_BigInt            *row_end,
                                            HYPRE_BigInt            *col_start,
                                            HYPRE_BigInt            *col_end )
{
   HYPRE_Int ierr=0;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(dm);

   if (!Parcsr_matrix) return(-1);

   ierr = HYPRE_ParCSRMatrixGetLocalRange( Parcsr_matrix, row_start, row_end,
                                           col_start, col_end );

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixGetRowParCSR
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DistributedMatrixGetRowParCSR( hypre_DistributedMatrix *dm,
                                     HYPRE_BigInt             row,
                                     HYPRE_Int               *size,
                                     HYPRE_BigInt           **col_ind,
                                     HYPRE_Real             **values )
{
   HYPRE_Int ierr = 0;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(dm);

   if (!Parcsr_matrix) return(-1);

   ierr = HYPRE_ParCSRMatrixGetRow( Parcsr_matrix, row, size, col_ind, values);

   // RL: if HYPRE_ParCSRMatrixGetRow was on device, need the next line to guarantee it's done
#if defined(HYPRE_USING_GPU)
   hypre_SyncComputeStream(hypre_handle());
#endif

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_DistributedMatrixRestoreRowParCSR
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DistributedMatrixRestoreRowParCSR( hypre_DistributedMatrix *dm,
                                         HYPRE_BigInt             row,
                                         HYPRE_Int               *size,
                                         HYPRE_BigInt           **col_ind,
                                         HYPRE_Real             **values )
{
   HYPRE_Int ierr;
   HYPRE_ParCSRMatrix Parcsr_matrix = (HYPRE_ParCSRMatrix) hypre_DistributedMatrixLocalStorage(dm);

   if (Parcsr_matrix == NULL) return(-1);

   ierr = HYPRE_ParCSRMatrixRestoreRow( Parcsr_matrix, row, size, col_ind, values);

   return(ierr);
}
