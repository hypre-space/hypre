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

/* HYPRE_PETScMatPilutSolver.c */
HYPRE_PETScMatPilutSolver HYPRE_NewPETScMatPilutSolver P((MPI_Comm comm , Mat matrix ));
HYPRE_Int HYPRE_FreePETScMatPilutSolver P((HYPRE_PETScMatPilutSolver in_ptr ));
HYPRE_Int HYPRE_PETScMatPilutSolverInitialize P((HYPRE_PETScMatPilutSolver in_ptr ));
HYPRE_Int HYPRE_PETScMatPilutSolverSetMatrix P((HYPRE_PETScMatPilutSolver in_ptr , Mat matrix ));
Mat HYPRE_PETScMatPilutSolverGetMatrix P((HYPRE_PETScMatPilutSolver in_ptr ));
HYPRE_Int HYPRE_PETScMatPilutSolverSetFactorRowSize P((HYPRE_PETScMatPilutSolver in_ptr , HYPRE_Int size ));
HYPRE_Int HYPRE_PETScMatPilutSolverSetDropTolerance P((HYPRE_PETScMatPilutSolver in_ptr , double tol ));
HYPRE_Int HYPRE_PETScMatPilutSolverSetMaxIts P((HYPRE_PETScMatPilutSolver in_ptr , HYPRE_Int its ));
HYPRE_Int HYPRE_PETScMatPilutSolverSetup P((HYPRE_PETScMatPilutSolver in_ptr , Vec x , Vec b ));
HYPRE_Int HYPRE_PETScMatPilutSolverApply P((HYPRE_PETScMatPilutSolver in_ptr , Vec b , Vec x ));
HYPRE_Int HYPRE_PETScMatPilutSolverSolve P((HYPRE_PETScMatPilutSolver in_ptr , Vec x , Vec b ));

/* hypre.c */

#undef P
