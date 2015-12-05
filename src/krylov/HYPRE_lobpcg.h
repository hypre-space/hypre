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
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/




#include "krylov.h"

#include "fortran_matrix.h"
#include "multivector.h"
#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"

#ifndef HYPRE_LOBPCG_SOLVER
#define HYPRE_LOBPCG_SOLVER

#ifdef __cplusplus
extern "C" {
#endif

  /* HYPRE_lobpcg.c */

  /* LOBPCG Constructor */
void
HYPRE_LOBPCGCreate( mv_InterfaceInterpreter*, HYPRE_MatvecFunctions*,
                    HYPRE_Solver* );

  /* LOBPCG Destructor */
int 
HYPRE_LOBPCGDestroy( HYPRE_Solver solver );

  /* Sets the preconditioner; if not called, preconditioning is not used */
int 
HYPRE_LOBPCGSetPrecond( HYPRE_Solver solver, 
			HYPRE_PtrToSolverFcn precond, 
			HYPRE_PtrToSolverFcn precond_setup, 
			HYPRE_Solver precond_solver );
int 
HYPRE_LOBPCGGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );

  /* Sets up A and the preconditioner, if there is one (see above) */
int 
HYPRE_LOBPCGSetup( HYPRE_Solver solver, 
		   HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );

  /* Sets up B; if not called, B = I */
int 
HYPRE_LOBPCGSetupB( HYPRE_Solver solver, 
		   HYPRE_Matrix B, HYPRE_Vector x );

  /* If called, makes the preconditionig to be applyed to Tx = b, not Ax = b */
int 
HYPRE_LOBPCGSetupT( HYPRE_Solver solver, 
		   HYPRE_Matrix T, HYPRE_Vector x );

  /* Solves A x = lambda B x, y'x = 0 */
int 
HYPRE_LOBPCGSolve( HYPRE_Solver data, mv_MultiVectorPtr y, 
		   mv_MultiVectorPtr x, double* lambda );

  /* Sets the absolute tolerance */
int 
HYPRE_LOBPCGSetTol( HYPRE_Solver solver, double tol );

  /* Sets the maximal number of iterations */
int 
HYPRE_LOBPCGSetMaxIter( HYPRE_Solver solver, int maxIter );

  /* Defines which initial guess for inner PCG iterations to use:
     mode = 0: use zero initial guess, otherwise use RHS */
int 
HYPRE_LOBPCGSetPrecondUsageMode( HYPRE_Solver solver, int mode );

  /* Sets the level of printout */
int 
HYPRE_LOBPCGSetPrintLevel( HYPRE_Solver solver , int level );

  /* Returns the pointer to residual norms matrix (blockSize x 1)*/
utilities_FortranMatrix*
HYPRE_LOBPCGResidualNorms( HYPRE_Solver solver );

  /* Returns the pointer to residual norms history matrix (blockSize x maxIter)*/
utilities_FortranMatrix*
HYPRE_LOBPCGResidualNormsHistory( HYPRE_Solver solver );

  /* Returns the pointer to eigenvalue history matrix (blockSize x maxIter)*/
utilities_FortranMatrix*
HYPRE_LOBPCGEigenvaluesHistory( HYPRE_Solver solver );

  /* Returns the number of iterations performed by LOBPCG */
int
HYPRE_LOBPCGIterations( HYPRE_Solver solver );

void
hypre_LOBPCGMultiOperatorB( void *data, void * x, void*  y );

void
lobpcg_MultiVectorByMultiVector(
mv_MultiVectorPtr x,
mv_MultiVectorPtr y,
utilities_FortranMatrix* xy);

#ifdef __cplusplus
}
#endif

#endif /* HYPRE_LOBPCG_SOLVER */
