 
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

/* pcg.c */
void PCG P((Vector *x , Vector *b , double tol , void *data ));
void PCGSetup P((Matrix *A , void (*precond )(), void *precond_data , void *data ));
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
void WJacobi P((Vector *x , Vector *b , double tol , void *data ));
void WJacobiSetup P((Matrix *A , void *data ));
void *NewWJacobiData P((Problem *problem , Solver *solver , char *log_file_name ));
void FreeWJacobiData P((void *data ));

/* write.c */
void WriteYSMP P((char *file_name , Matrix *matrix ));
void WriteVec P((char *file_name , Vector *vector ));

#undef P
 
#endif
 
