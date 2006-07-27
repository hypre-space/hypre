/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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
#include "../parcsr_mv/par_vector.h"


/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutCreate
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRPilutCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   int ierr = 0;
   
   ierr = HYPRE_NewDistributedMatrixPilutSolver( comm, NULL, 
            (HYPRE_DistributedMatrixPilutSolver *) solver);

   ierr = HYPRE_DistributedMatrixPilutSolverInitialize( 
      (HYPRE_DistributedMatrixPilutSolver) solver );

   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRPilutDestroy( HYPRE_Solver solver )
{
   int ierr = 0;

   ierr = HYPRE_DistributedMatrixDestroy( 
      HYPRE_DistributedMatrixPilutSolverGetMatrix(
         (HYPRE_DistributedMatrixPilutSolver) solver ) );
   if (ierr) return(ierr);

   ierr = HYPRE_FreeDistributedMatrixPilutSolver(
      (HYPRE_DistributedMatrixPilutSolver) solver );

   return( ierr );
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
   int ierr = 0;
   HYPRE_DistributedMatrix matrix;
   HYPRE_DistributedMatrixPilutSolver distributed_solver = 
      (HYPRE_DistributedMatrixPilutSolver) solver;

   ierr = HYPRE_ConvertParCSRMatrixToDistributedMatrix(
             A, &matrix );
   if (ierr) return(ierr);

   ierr = HYPRE_DistributedMatrixPilutSolverSetMatrix( distributed_solver, matrix );
   if (ierr) return(ierr);

   ierr = HYPRE_DistributedMatrixPilutSolverSetup( distributed_solver );

   return( ierr );
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
   int ierr = 0;
   double *rhs, *soln;

   rhs = hypre_VectorData( hypre_ParVectorLocalVector( (hypre_ParVector *)b ) );
   soln = hypre_VectorData( hypre_ParVectorLocalVector( (hypre_ParVector *)x ) );

   ierr = HYPRE_DistributedMatrixPilutSolverSolve(
      (HYPRE_DistributedMatrixPilutSolver) solver,
      soln, rhs );

   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPilutSetMaxIter( HYPRE_Solver solver,
                        int          max_iter  )
{
   int ierr = 0;

   ierr = HYPRE_DistributedMatrixPilutSolverSetMaxIts(
      (HYPRE_DistributedMatrixPilutSolver) solver, max_iter );

   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetDropTolerance
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPilutSetDropTolerance( HYPRE_Solver solver,
                    double       tol    )
{
   int ierr = 0;

   ierr = HYPRE_DistributedMatrixPilutSolverSetDropTolerance(
      (HYPRE_DistributedMatrixPilutSolver) solver, tol );


   return( ierr );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPilutSetFactorRowSize
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPilutSetFactorRowSize( HYPRE_Solver solver,
                    int       size    )
{
   int ierr = 0;

   ierr = HYPRE_DistributedMatrixPilutSolverSetFactorRowSize(
      (HYPRE_DistributedMatrixPilutSolver) solver, size );


   return( ierr );
}

