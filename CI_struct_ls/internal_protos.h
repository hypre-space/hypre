# define	P(s) s

/* HYPRE_struct_solver.c */
HYPRE_StructInterfaceSolver HYPRE_NewStructInterfaceSolver P((MPI_Comm context , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int HYPRE_FreeStructInterfaceSolver P((HYPRE_StructInterfaceSolver struct_solver ));
int HYPRE_StructInterfaceSolverSetType P((HYPRE_StructInterfaceSolver solver , int type ));
int HYPRE_StructInterfaceSolverInitialize P((HYPRE_StructInterfaceSolver solver ));
int HYPRE_StructInterfaceSolverSetup P((HYPRE_StructInterfaceSolver solver , HYPRE_StructInterfaceMatrix matrix , HYPRE_StructInterfaceVector soln , HYPRE_StructInterfaceVector rhs ));
int HYPRE_StructInterfaceSolverSolve P((HYPRE_StructInterfaceSolver solver ));
int HYPRE_StructInterfaceSolverSetDropTolerance P((HYPRE_StructInterfaceSolver solver , double tol ));
int HYPRE_StructInterfaceSolverSetFactorRowSize P((HYPRE_StructInterfaceSolver solver , int size ));

/* hypre.c */

/* struct_solver.c */
hypre_StructInterfaceSolver *hypre_NewStructInterfaceSolver P((MPI_Comm context , HYPRE_StructGrid grid , HYPRE_StructStencil stencil ));
int hypre_FreeStructInterfaceSolver P((hypre_StructInterfaceSolver *struct_solver ));
int hypre_StructInterfaceSolverSetType P((hypre_StructInterfaceSolver *solver , int type ));
int hypre_StructInterfaceSolverInitialize P((hypre_StructInterfaceSolver *solver ));
int hypre_StructInterfaceSolverSetup P((hypre_StructInterfaceSolver *solver , HYPRE_StructInterfaceMatrix matrix , HYPRE_StructInterfaceVector soln , HYPRE_StructInterfaceVector rhs ));
int hypre_StructInterfaceSolverSolve P((hypre_StructInterfaceSolver *solver ));
int hypre_StructInterfaceSolverSetDropTolerance P((hypre_StructInterfaceSolver *solver , double tol ));
int hypre_StructInterfaceSolverSetFactorRowSize P((hypre_StructInterfaceSolver *solver , int size ));

/* struct_solver_PETSc.c */
int hypre_FreeStructInterfaceSolverPETSc P((hypre_StructInterfaceSolver *struct_solver ));
int hypre_StructInterfaceSolverInitializePETSc P((hypre_StructInterfaceSolver *struct_solver ));
int hypre_StructInterfaceSolverSetupPETSc P((hypre_StructInterfaceSolver *struct_solver ));
int hypre_StructInterfaceSolverSolvePETSc P((hypre_StructInterfaceSolver *struct_solver ));
int hypre_StructInterfaceSolverPETScSetDropTolerance P((hypre_StructInterfaceSolver *struct_solver , double tol ));
int hypre_StructInterfaceSolverPETScSetFactorRowSize P((hypre_StructInterfaceSolver *struct_solver , int size ));

#undef P
