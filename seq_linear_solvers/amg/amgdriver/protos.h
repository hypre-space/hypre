 
#ifndef _PROTOS_HEADER
#define _PROTOS_HEADER
 
#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* amg.c */
void *NewAMGData P((Problem *problem , Solver *solver , char *log_file_name ));

/* globals.c */
void NewGlobals P((char *run_name ));
void FreeGlobals P((void ));

/* gmres.c */
int SPGMRATimes P((void *A_data_arg , N_Vector v_arg , N_Vector z_arg ));
int SPGMRPSolve P((void *P_data_arg , N_Vector r_arg , N_Vector z_arg , int lr_arg ));
void GMRES P((Vector *x_arg , Vector *b_arg , double tol_arg , void *data_arg ));
void GMRESSetup P((Matrix *A , int (*precond )(), void *precond_data , void *data ));
void *NewGMRESData P((Problem *problem , Solver *solver , char *log_file_name ));
void FreeGMRESData P((void *data ));

/* pcg.c */
void PCG P((Vector *x , Vector *b , double tol , void *data ));
void PCGSetup P((Matrix *A , int (*precond )(), void *precond_data , void *data ));
void *NewPCGData P((Problem *problem , Solver *solver , char *log_file_name ));
void FreePCGData P((void *data ));

/* problem.c */
Problem *NewProblem P((char *file_name ));
void FreeProblem P((Problem *problem ));
void WriteProblem P((char *file_name , Problem *problem ));

/* random.c */
void SeedRand P((int seed ));
double Rand P((void ));

/* read.c */
Matrix *ReadYSMP P((char *file_name ));
Vector *ReadVec P((char *file_name ));

/* solver.c */
Solver *NewSolver P((char *file_name ));
void FreeSolver P((Solver *solver ));
void WriteSolver P((char *file_name , Solver *solver ));

/* wjacobi.c */
int WJacobi P((Vector *x , Vector *b , double tol , void *data ));
void WJacobiSetup P((Matrix *A , void *data ));
void *NewWJacobiData P((Problem *problem , Solver *solver , char *log_file_name ));
void FreeWJacobiData P((void *data ));

#undef P
 
#endif
 
