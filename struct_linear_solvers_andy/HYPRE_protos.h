#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_struct_solver.c */
HYPRE_StructSolver HYPRE_NewStructSolver P((MPI_Comm context , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructSolver P((HYPRE_StructSolver struct_solver ));
int HYPRE_StructSolverSetType P((HYPRE_StructSolver solver , int type ));
int HYPRE_StructSolverInitialize P((HYPRE_StructSolver solver ));
int HYPRE_StructSolverSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix matrix , HYPRE_StructVector soln , HYPRE_StructVector rhs ));
int HYPRE_StructSolverSolve P((HYPRE_StructSolver solver ));
int HYPRE_StructSolverSetDropTolerance P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructSolverSetFactorRowSize P((HYPRE_StructSolver solver , int size ));

#undef P
