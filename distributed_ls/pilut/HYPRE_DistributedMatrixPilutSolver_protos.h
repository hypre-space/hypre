# define        P(s) s
 
/* HYPRE_DistributedMatrixPilutSolver.c */
int HYPRE_NewDistributedMatrixPilutSolver P((MPI_Comm comm , HYPRE_DistributedMatrix matrix, HYPRE_DistributedMatrixPilutSolver *solver ));
int HYPRE_FreeDistributedMatrixPilutSolver P((HYPRE_DistributedMatrixPilutSolver in_ptr ));
int HYPRE_DistributedMatrixPilutSolverInitialize P((HYPRE_DistributedMatrixPilutSolver solver ));
int HYPRE_DistributedMatrixPilutSolverSetMatrix P((HYPRE_DistributedMatrixPilutSolver in_ptr , HYPRE_DistributedMatrix matrix ));
HYPRE_DistributedMatrix HYPRE_DistributedMatrixPilutSolverGetMatrix P((HYPRE_DistributedMatrixPilutSolver in_ptr ));
int HYPRE_DistributedMatrixPilutSolverSetNumLocalRow P((HYPRE_DistributedMatrixPilutSolver in_ptr , int FirstLocalRow ));
int HYPRE_DistributedMatrixPilutSolverSetFactorRowSize P((HYPRE_DistributedMatrixPilutSolver in_ptr , int size ));
int HYPRE_DistributedMatrixPilutSolverSetDropTolerance P((HYPRE_DistributedMatrixPilutSolver in_ptr , double tolerance ));
int HYPRE_DistributedMatrixPilutSolverSetMaxIts P((HYPRE_DistributedMatrixPilutSolver in_ptr , int its ));
int HYPRE_DistributedMatrixPilutSolverSetup P((HYPRE_DistributedMatrixPilutSolver in_ptr ));
int HYPRE_DistributedMatrixPilutSolverSolve P((HYPRE_DistributedMatrixPilutSolver in_ptr , double *x , double *b ));
 
#undef P

