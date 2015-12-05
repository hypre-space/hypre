/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_ParCSRPilut interface
 *
 *****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "./HYPRE_parcsr_ls.h"

#include "../distributed_matrix/HYPRE_distributed_matrix_types.h"
#include "../distributed_matrix/HYPRE_distributed_matrix_protos.h"

#include "../matrix_matrix/HYPRE_matrix_matrix_protos.h"

#include "../distributed_ls/pilut/HYPRE_DistributedMatrixPilutSolver_types.h"
#include "../distributed_ls/pilut/HYPRE_DistributedMatrixPilutSolver_protos.h"

/* Must include implementation definition for ParVector since no data access
  functions are publically provided. AJC, 5/99 */
/* Likewise for Vector. AJC, 5/99 */
#include "../seq_mv/vector.h"

/* AB 8/06 - replace header file */
/* #include "../parcsr_mv/par_vector.h" */
#include "../parcsr_mv/_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRPilutCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_NewDistributedMatrixPilutSolver( comm, NULL, 
            (HYPRE_DistributedMatrixPilutSolver *) solver);

   HYPRE_DistributedMatrixPilutSolverInitialize( 
      (HYPRE_DistributedMatrixPilutSolver) solver );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRPilutDestroy( HYPRE_Solver solver )
{
   HYPRE_DistributedMatrix mat = HYPRE_DistributedMatrixPilutSolverGetMatrix(
      (HYPRE_DistributedMatrixPilutSolver) solver );
   if ( mat ) HYPRE_DistributedMatrixDestroy( mat );

   HYPRE_FreeDistributedMatrixPilutSolver(
      (HYPRE_DistributedMatrixPilutSolver) solver );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRPilutSetup( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{
   HYPRE_DistributedMatrix matrix;
   HYPRE_DistributedMatrixPilutSolver distributed_solver = 
      (HYPRE_DistributedMatrixPilutSolver) solver;

   HYPRE_ConvertParCSRMatrixToDistributedMatrix(
             A, &matrix );

   HYPRE_DistributedMatrixPilutSolverSetMatrix( distributed_solver, matrix );

   HYPRE_DistributedMatrixPilutSolverSetup( distributed_solver );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRPilutSolve( HYPRE_Solver solver,
                   HYPRE_ParCSRMatrix A,
                   HYPRE_ParVector b,
                   HYPRE_ParVector x      )
{
   double *rhs, *soln;

   rhs = hypre_VectorData( hypre_ParVectorLocalVector( (hypre_ParVector *)b ) );
   soln = hypre_VectorData( hypre_ParVectorLocalVector( (hypre_ParVector *)x ) );

   HYPRE_DistributedMatrixPilutSolverSolve(
      (HYPRE_DistributedMatrixPilutSolver) solver,
      soln, rhs );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPilutSetMaxIter( HYPRE_Solver solver,
                        HYPRE_Int          max_iter  )
{

   HYPRE_DistributedMatrixPilutSolverSetMaxIts(
      (HYPRE_DistributedMatrixPilutSolver) solver, max_iter );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetDropTolerance
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPilutSetDropTolerance( HYPRE_Solver solver,
                    double       tol    )
{
   HYPRE_DistributedMatrixPilutSolverSetDropTolerance(
      (HYPRE_DistributedMatrixPilutSolver) solver, tol );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetFactorRowSize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPilutSetFactorRowSize( HYPRE_Solver solver,
                    HYPRE_Int       size    )
{
   HYPRE_DistributedMatrixPilutSolverSetFactorRowSize(
      (HYPRE_DistributedMatrixPilutSolver) solver, size );

   return hypre_error_flag;
}

