/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/

/*****************************************************************************
 *
 * HYPRE_par_laplace Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

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
     hypre_F90_DblArray *value,
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
           hypre_F90_PassDblArray (value) ) );

   *ierr = 0;
}
