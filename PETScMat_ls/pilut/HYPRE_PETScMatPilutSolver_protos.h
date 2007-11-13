/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
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
 * $Revision$
 ***********************************************************************EHEADER*/




# define	P(s) s

/* HYPRE_PETScMatPilutSolver.c */
HYPRE_PETScMatPilutSolver HYPRE_NewPETScMatPilutSolver P((MPI_Comm comm , Mat matrix ));
int HYPRE_FreePETScMatPilutSolver P((HYPRE_PETScMatPilutSolver in_ptr ));
int HYPRE_PETScMatPilutSolverInitialize P((HYPRE_PETScMatPilutSolver in_ptr ));
int HYPRE_PETScMatPilutSolverSetMatrix P((HYPRE_PETScMatPilutSolver in_ptr , Mat matrix ));
Mat HYPRE_PETScMatPilutSolverGetMatrix P((HYPRE_PETScMatPilutSolver in_ptr ));
int HYPRE_PETScMatPilutSolverSetFactorRowSize P((HYPRE_PETScMatPilutSolver in_ptr , int size ));
int HYPRE_PETScMatPilutSolverSetDropTolerance P((HYPRE_PETScMatPilutSolver in_ptr , double tol ));
int HYPRE_PETScMatPilutSolverSetMaxIts P((HYPRE_PETScMatPilutSolver in_ptr , int its ));
int HYPRE_PETScMatPilutSolverSetup P((HYPRE_PETScMatPilutSolver in_ptr , Vec x , Vec b ));
int HYPRE_PETScMatPilutSolverApply P((HYPRE_PETScMatPilutSolver in_ptr , Vec b , Vec x ));
int HYPRE_PETScMatPilutSolverSolve P((HYPRE_PETScMatPilutSolver in_ptr , Vec x , Vec b ));

#undef P
