 
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
 
