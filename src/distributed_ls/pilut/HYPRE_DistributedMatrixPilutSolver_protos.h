/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/




 
/* HYPRE_DistributedMatrixPilutSolver.c */
int HYPRE_NewDistributedMatrixPilutSolver (MPI_Comm comm , HYPRE_DistributedMatrix matrix, HYPRE_DistributedMatrixPilutSolver *solver );
int HYPRE_FreeDistributedMatrixPilutSolver (HYPRE_DistributedMatrixPilutSolver in_ptr );
int HYPRE_DistributedMatrixPilutSolverInitialize (HYPRE_DistributedMatrixPilutSolver solver );
int HYPRE_DistributedMatrixPilutSolverSetMatrix (HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_DistributedMatrix matrix );
HYPRE_DistributedMatrix HYPRE_DistributedMatrixPilutSolverGetMatrix (HYPRE_DistributedMatrixPilutSolver in_ptr );
int HYPRE_DistributedMatrixPilutSolverSetNumLocalRow (HYPRE_DistributedMatrixPilutSolver in_ptr , int FirstLocalRow );
int HYPRE_DistributedMatrixPilutSolverSetFactorRowSize (HYPRE_DistributedMatrixPilutSolver in_ptr , int size );
int HYPRE_DistributedMatrixPilutSolverSetDropTolerance (HYPRE_DistributedMatrixPilutSolver in_ptr , double tolerance );
int HYPRE_DistributedMatrixPilutSolverSetMaxIts (HYPRE_DistributedMatrixPilutSolver in_ptr , int its );
int HYPRE_DistributedMatrixPilutSolverSetup (HYPRE_DistributedMatrixPilutSolver in_ptr );
int HYPRE_DistributedMatrixPilutSolverSolve (HYPRE_DistributedMatrixPilutSolver in_ptr , double *x , double *b );
 
