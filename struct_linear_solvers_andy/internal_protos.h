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

/* struct_solver.c */
hypre_StructSolver *hypre_NewStructSolver P((MPI_Comm context , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int hypre_FreeStructSolver P((hypre_StructSolver *struct_solver ));
int hypre_StructSolverSetType P((hypre_StructSolver *solver , int type ));
int hypre_StructSolverInitialize P((hypre_StructSolver *solver ));
int hypre_StructSolverSetup P((hypre_StructSolver *solver , HYPRE_StructMatrix matrix , HYPRE_StructVector soln , HYPRE_StructVector rhs ));
int hypre_StructSolverSolve P((hypre_StructSolver *solver ));
int hypre_StructSolverSetDropTolerance P((hypre_StructSolver *solver , double tol ));

/* struct_solver_PETSc.c */
int hypre_FreeStructSolverPETSc P((hypre_StructSolver *struct_solver ));
int hypre_StructSolverInitializePETSc P((hypre_StructSolver *struct_solver ));
int hypre_StructSolverSetupPETSc P((hypre_StructSolver *struct_solver ));
int hypre_StructSolverSolvePETSc P((hypre_StructSolver *struct_solver ));
int hypre_StructSolverPETScSetDropTolerance P((hypre_StructSolver *struct_solver , double tol ));

/* hypre.c */

#undef P
