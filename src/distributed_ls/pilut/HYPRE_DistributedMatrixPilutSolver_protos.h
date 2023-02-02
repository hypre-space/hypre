/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* HYPRE_DistributedMatrixPilutSolver.c */
HYPRE_Int HYPRE_NewDistributedMatrixPilutSolver (MPI_Comm comm , HYPRE_DistributedMatrix matrix, HYPRE_DistributedMatrixPilutSolver *solver );
HYPRE_Int HYPRE_FreeDistributedMatrixPilutSolver (HYPRE_DistributedMatrixPilutSolver in_ptr );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverInitialize (HYPRE_DistributedMatrixPilutSolver solver );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetMatrix (HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_DistributedMatrix matrix );
HYPRE_DistributedMatrix HYPRE_DistributedMatrixPilutSolverGetMatrix (HYPRE_DistributedMatrixPilutSolver in_ptr );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetNumLocalRow (HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_Int FirstLocalRow );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetFactorRowSize (HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_Int size );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetDropTolerance (HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_Real tolerance );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetMaxIts (HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_Int its );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetup (HYPRE_DistributedMatrixPilutSolver in_ptr );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSolve (HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_Real *x , HYPRE_Real *b );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetLogging( HYPRE_DistributedMatrixPilutSolver in_ptr, HYPRE_Int logging );

