/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParCSRMatrix Fortran interface to macros
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixGlobalNumRows
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixglobalnumrows, HYPRE_PARCSRMATRIXGLOBALNUMROWS)
( hypre_F90_Obj *matrix,
  hypre_F90_BigInt *num_rows,
  hypre_F90_Int *ierr      )
{
   *num_rows = (hypre_F90_BigInt)
               ( hypre_ParCSRMatrixGlobalNumRows(
                    (hypre_ParCSRMatrix *) *matrix ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixRowStarts
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrmatrixrowstarts, HYPRE_PARCSRMATRIXROWSTARTS)
( hypre_F90_Obj *matrix,
  hypre_F90_Obj *row_starts,
  hypre_F90_Int *ierr      )
{
   *row_starts = (hypre_F90_Obj)
                 ( hypre_ParCSRMatrixRowStarts(
                      (hypre_ParCSRMatrix *) *matrix ) );

   *ierr = 0;
}

#ifdef __cplusplus
}
#endif
