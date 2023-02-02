/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*****************************************************************************
 *
 * HYPRE_par_laplace Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * GenerateLaplacian
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_generatelaplacian, HYPRE_GENERATELAPLACIAN)
( hypre_F90_Comm *comm,
  hypre_F90_Int *nx,
  hypre_F90_Int *ny,
  hypre_F90_Int *nz,
  hypre_F90_Int *P,
  hypre_F90_Int *Q,
  hypre_F90_Int *R,
  hypre_F90_Int *p,
  hypre_F90_Int *q,
  hypre_F90_Int *r,
  hypre_F90_RealArray *value,
  hypre_F90_Obj *matrix,
  hypre_F90_Int *ierr   )

{
   *matrix = (hypre_F90_Obj)
             ( GenerateLaplacian(
                  hypre_F90_PassComm (comm),
                  hypre_F90_PassInt (nx),
                  hypre_F90_PassInt (ny),
                  hypre_F90_PassInt (nz),
                  hypre_F90_PassInt (P),
                  hypre_F90_PassInt (Q),
                  hypre_F90_PassInt (R),
                  hypre_F90_PassInt (p),
                  hypre_F90_PassInt (q),
                  hypre_F90_PassInt (r),
                  hypre_F90_PassRealArray (value) ) );

   *ierr = 0;
}
#ifdef __cplusplus
}
#endif
