#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


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
