/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.6 $
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

int 
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

int 
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

int 
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

int 
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

int
HYPRE_ParCSRPilutSetMaxIter( HYPRE_Solver solver,
                        int          max_iter  )
{

   HYPRE_DistributedMatrixPilutSolverSetMaxIts(
      (HYPRE_DistributedMatrixPilutSolver) solver, max_iter );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetDropTolerance
 *--------------------------------------------------------------------------*/

int
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

int
HYPRE_ParCSRPilutSetFactorRowSize( HYPRE_Solver solver,
                    int       size    )
{
   HYPRE_DistributedMatrixPilutSolverSetFactorRowSize(
      (HYPRE_DistributedMatrixPilutSolver) solver, size );

   return hypre_error_flag;
}

