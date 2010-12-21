/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
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
                                    HYPRE_Int      *nx,
                                    HYPRE_Int      *ny,
                                    HYPRE_Int      *nz,
                                    HYPRE_Int      *P,
                                    HYPRE_Int      *Q,
                                    HYPRE_Int      *R,
                                    HYPRE_Int      *p,
                                    HYPRE_Int      *q,
                                    HYPRE_Int      *r,
                                    double   *value,
                                    hypre_F90_Obj *matrix,
                                    HYPRE_Int      *ierr   )

{
   *matrix = (hypre_F90_Obj) ( GenerateLaplacian( (MPI_Comm) *comm,
                                             (HYPRE_Int)      *nx,
                                             (HYPRE_Int)      *ny,
                                             (HYPRE_Int)      *nz,
                                             (HYPRE_Int)      *P,
                                             (HYPRE_Int)      *Q,
                                             (HYPRE_Int)      *R,
                                             (HYPRE_Int)      *p,
                                             (HYPRE_Int)      *q,
                                             (HYPRE_Int)      *r,
                                             (double *)  value ) );

   *ierr = 0;
}
