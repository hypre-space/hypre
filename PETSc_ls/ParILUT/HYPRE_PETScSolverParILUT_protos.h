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
 * $Revision$
 ***********************************************************************EHEADER*/



# define	P(s) s

/* HYPRE_PETScSolverParILUT.c */
HYPRE_PETScSolverParILUT HYPRE_NewPETScSolverParILUT P((MPI_Comm comm ));
int HYPRE_FreePETScSolverParILUT P((HYPRE_PETScSolverParILUT in_ptr ));
int HYPRE_PETScSolverParILUTInitialize P((HYPRE_PETScSolverParILUT in_ptr ));
int HYPRE_PETScSolverParILUTSetSystemSles P((HYPRE_PETScSolverParILUT in_ptr , SLES Sles ));
int HYPRE_PETScSolverParILUTSetSystemMatrix P((HYPRE_PETScSolverParILUT in_ptr , Mat matrix ));
int HYPRE_PETScSolverParILUTSetPreconditionerMatrix P((HYPRE_PETScSolverParILUT in_ptr , Mat matrix ));
Mat HYPRE_PETScSolverParILUTGetSystemMatrix P((HYPRE_PETScSolverParILUT in_ptr ));
Mat HYPRE_PETScSolverParILUTGetPreconditionerMatrix P((HYPRE_PETScSolverParILUT in_ptr ));
int HYPRE_PETScSolverParILUTSetFactorRowSize P((HYPRE_PETScSolverParILUT in_ptr , int size ));
int HYPRE_PETScSolverParILUTSetDropTolerance P((HYPRE_PETScSolverParILUT in_ptr , double tol ));
int HYPRE_PETScSolverParILUTSolve P((HYPRE_PETScSolverParILUT in_ptr , Vec x , Vec b ));

/* HYPRE_PETScSolverParILUTSetup.c */
int HYPRE_PETScSolverParILUTSetup P((HYPRE_PETScSolverParILUT in_ptr , Vec x , Vec b ));

#undef P
