/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
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
   (hypre_F90_Comm *comm,
    hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_EuclidCreate(
      hypre_F90_PassComm (comm),
      hypre_F90_PassObjRef (HYPRE_Solver, solver) );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidDestroy - Destroy a Euclid object.
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_eucliddestroy, HYPRE_EUCLIDDESTROY)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_EuclidDestroy(
      hypre_F90_PassObj (HYPRE_Solver, solver) );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetup - Set up function for Euclid.
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_euclidsetup, HYPRE_EUCLIDSETUP)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_EuclidSetup(
      hypre_F90_PassObj (HYPRE_Solver, solver),
      hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
      hypre_F90_PassObj (HYPRE_ParVector, b),
      hypre_F90_PassObj (HYPRE_ParVector, x)   );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSolve - Solve function for Euclid.
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_euclidsolve, HYPRE_EUCLIDSOLVE)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_EuclidSolve(
      hypre_F90_PassObj (HYPRE_Solver, solver),
      hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
      hypre_F90_PassObj (HYPRE_ParVector, b),
      hypre_F90_PassObj (HYPRE_ParVector, x)  );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetParams
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_euclidsetparams, HYPRE_EUCLIDSETPARAMS)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *argc,
    char **argv,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_EuclidSetParams(
      hypre_F90_PassObj (HYPRE_Solver, solver), 
      hypre_F90_PassInt (argc),
      (char **)       argv );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetParamsFromFile
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_euclidsetparamsfromfile, HYPRE_EUCLIDSETPARAMSFROMFILE)
   (hypre_F90_Obj *solver,
    char *filename,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_EuclidSetParamsFromFile(
      hypre_F90_PassObj (HYPRE_Solver, solver), 
      (char *)        filename );
}
/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetlevel, HYPRE_EUCLIDSETLEVEL)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *eu_level,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_EuclidSetLevel(
      hypre_F90_PassObj (HYPRE_Solver, solver),
      hypre_F90_PassInt (eu_level) );
}


/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetBJ
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetbj, HYPRE_EUCLIDSETBJ)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *bj,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_EuclidSetBJ(
      hypre_F90_PassObj (HYPRE_Solver, solver),
      hypre_F90_PassInt (bj) );
}

/*--------------------------------------------------------------------------
 * HYPRE_EuclidSetSparseA
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_euclidsetsparsea, HYPRE_EUCLIDSETSPARSEA)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *spa,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_EuclidSetSparseA(
      hypre_F90_PassObj (HYPRE_Solver, solver),
      hypre_F90_PassDbl (spa) );
}
