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
               (hypre_F90_Comm *comm, hypre_F90_Obj *solver, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_EuclidCreate( (MPI_Comm)      *comm,
                                     (HYPRE_Solver *) solver );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidDestroy - Destroy a Euclid object.
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_eucliddestroy, HYPRE_EUCLIDDESTROY)
               (hypre_F90_Obj *solver, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_EuclidDestroy( (HYPRE_Solver) *solver );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetup - Set up function for Euclid.
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_euclidsetup, HYPRE_EUCLIDSETUP)
               (hypre_F90_Obj *solver, hypre_F90_Obj *A, hypre_F90_Obj *b, hypre_F90_Obj *x, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_EuclidSetup( (HYPRE_Solver)       *solver,
                                    (HYPRE_ParCSRMatrix) *A,
                                    (HYPRE_ParVector)    *b,
                                    (HYPRE_ParVector)    *x   );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSolve - Solve function for Euclid.
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_euclidsolve, HYPRE_EUCLIDSOLVE)
               (hypre_F90_Obj *solver, hypre_F90_Obj *A, hypre_F90_Obj *b, hypre_F90_Obj *x, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_EuclidSolve( (HYPRE_Solver)       *solver,
                                    (HYPRE_ParCSRMatrix) *A,
                                    (HYPRE_ParVector)    *b,
                                    (HYPRE_ParVector)    *x  );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetParams
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_euclidsetparams, HYPRE_EUCLIDSETPARAMS)
               (hypre_F90_Obj *solver, HYPRE_Int *argc, char **argv, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_EuclidSetParams( (HYPRE_Solver) *solver, 
                                        (HYPRE_Int)          *argc,
                                        (char **)       argv );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetParamsFromFile
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_euclidsetparamsfromfile, HYPRE_EUCLIDSETPARAMSFROMFILE)
               (hypre_F90_Obj *solver, char *filename, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_EuclidSetParamsFromFile( (HYPRE_Solver) *solver, 
                                                (char *)        filename );
}
/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetlevel, HYPRE_EUCLIDSETLEVEL)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *eu_level,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_EuclidSetLevel( (HYPRE_Solver) *solver,
                                         (HYPRE_Int)          *eu_level ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetBJ
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetbj, HYPRE_EUCLIDSETBJ)(
   hypre_F90_Obj *solver,
   HYPRE_Int      *bj,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_EuclidSetBJ( (HYPRE_Solver) *solver,
                                         (HYPRE_Int)          *bj ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetSparseA
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetsparsea, HYPRE_EUCLIDSETSPARSEA)(
   hypre_F90_Obj *solver,
   double   *spa,
   HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_EuclidSetSparseA( (HYPRE_Solver) *solver,
                                           (double)          *spa ) );
}
