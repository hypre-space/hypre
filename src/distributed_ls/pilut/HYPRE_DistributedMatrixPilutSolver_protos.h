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
 * $Revision: 2.2 $
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
 
