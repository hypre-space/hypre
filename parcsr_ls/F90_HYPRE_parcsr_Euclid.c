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




/******************************************************************************
 *
 * HYPRE_Euclid Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_EuclidCreate - Return a Euclid "solver".  
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_euclidcreate, HYPRE_EUCLIDCREATE)
               (int *comm, long int *solver, int *ierr)
{
   *ierr = (int) HYPRE_EuclidCreate( (MPI_Comm)      *comm,
                                     (HYPRE_Solver *) solver );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidDestroy - Destroy a Euclid object.
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_eucliddestroy, HYPRE_EUCLIDDESTROY)
               (long int *solver, int *ierr)
{
   *ierr = (int) HYPRE_EuclidDestroy( (HYPRE_Solver) *solver );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetup - Set up function for Euclid.
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_euclidsetup, HYPRE_EUCLIDSETUP)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) HYPRE_EuclidSetup( (HYPRE_Solver)       *solver,
                                    (HYPRE_ParCSRMatrix) *A,
                                    (HYPRE_ParVector)    *b,
                                    (HYPRE_ParVector)    *x   );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSolve - Solve function for Euclid.
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_euclidsolve, HYPRE_EUCLIDSOLVE)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) HYPRE_EuclidSolve( (HYPRE_Solver)       *solver,
                                    (HYPRE_ParCSRMatrix) *A,
                                    (HYPRE_ParVector)    *b,
                                    (HYPRE_ParVector)    *x  );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetParams
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_euclidsetparams, HYPRE_EUCLIDSETPARAMS)
               (long int *solver, int *argc, char **argv, int *ierr)
{
   *ierr = (int) HYPRE_EuclidSetParams( (HYPRE_Solver) *solver, 
                                        (int)          *argc,
                                        (char **)       argv );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetParamsFromFile
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_euclidsetparamsfromfile, HYPRE_EUCLIDSETPARAMSFROMFILE)
               (long int *solver, char *filename, int *ierr)
{
   *ierr = (int) HYPRE_EuclidSetParamsFromFile( (HYPRE_Solver) *solver, 
                                                (char *)        filename );
}
