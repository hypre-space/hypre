/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




 
/* HYPRE_DistributedMatrixPilutSolver.c */
HYPRE_Int HYPRE_NewDistributedMatrixPilutSolver (MPI_Comm comm , HYPRE_DistributedMatrix matrix, HYPRE_DistributedMatrixPilutSolver *solver );
HYPRE_Int HYPRE_FreeDistributedMatrixPilutSolver (HYPRE_DistributedMatrixPilutSolver in_ptr );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverInitialize (HYPRE_DistributedMatrixPilutSolver solver );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetMatrix (HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_DistributedMatrix matrix );
HYPRE_DistributedMatrix HYPRE_DistributedMatrixPilutSolverGetMatrix (HYPRE_DistributedMatrixPilutSolver in_ptr );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetNumLocalRow (HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_Int FirstLocalRow );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetFactorRowSize (HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_Int size );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetDropTolerance (HYPRE_DistributedMatrixPilutSolver in_ptr , double tolerance );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetMaxIts (HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_Int its );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSetup (HYPRE_DistributedMatrixPilutSolver in_ptr );
HYPRE_Int HYPRE_DistributedMatrixPilutSolverSolve (HYPRE_DistributedMatrixPilutSolver in_ptr , double *x , double *b );
 
