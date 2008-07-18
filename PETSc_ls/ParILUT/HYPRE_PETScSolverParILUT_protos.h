/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
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
