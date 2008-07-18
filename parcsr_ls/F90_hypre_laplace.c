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
                                  ( int      *comm,
                                    int      *nx,
                                    int      *ny,
                                    int      *nz,
                                    int      *P,
                                    int      *Q,
                                    int      *R,
                                    int      *p,
                                    int      *q,
                                    int      *r,
                                    double   *value,
                                    long int *matrix,
                                    int      *ierr   )

{
   *matrix = (long int) ( GenerateLaplacian( (MPI_Comm) *comm,
                                             (int)      *nx,
                                             (int)      *ny,
                                             (int)      *nz,
                                             (int)      *P,
                                             (int)      *Q,
                                             (int)      *R,
                                             (int)      *p,
                                             (int)      *q,
                                             (int)      *r,
                                             (double *)  value ) );

   *ierr = 0;
}
