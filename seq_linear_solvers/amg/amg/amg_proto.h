#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* amgs01.c */
void AMGS01 P((Vector *u , Vector *f , double tol , Data *data ));
void AMGS01Setup P((Problem *problem , Data *data ));
Data *ReadAMGS01Params P((FILE *fp ));
Data *NewAMGS01Data P((int levmax , int ncg , double ecg , int nwt , double ewt , int nstr , int ncyc , int *mu , int *ntrlx , int *iprlx , int *ierlx , int *iurlx , int ioutdat , int ioutgrd , int ioutmat , int ioutres , int ioutsol , char *log_file_name ));
void FreeAMGS01Data P((Data *data ));

/* fortran.c */
void writeysm_ P((char *file_name , double *data , int *ia , int *ja , int *size , int file_name_len ));
void writevec_ P((char *file_name , double *data , int *size , int file_name_len ));

/* globals.c */
void NewGlobals P((char *run_name ));
void FreeGlobals P((void ));

/* matrix.c */
Matrix *NewMatrix P((double *data , int *ia , int *ja , int size ));
void FreeMatrix P((Matrix *matrix ));

/* pcg.c */
void PCG P((Vector *x , Vector *b , double tol , Data *data ));
Data *ReadPCGParams P((FILE *fp ));
Data *NewPCGData P((int max_iter , int two_norm , char *log_file_name ));
void FreePCGData P((Data *data ));

/* problem.c */
Problem *NewProblem P((char *file_name ));
void FreeProblem P((Problem *problem ));

/* random.c */
void SeedRand P((int seed ));
double Rand P((void ));

/* read.c */
Matrix *ReadYSMP P((char *file_name ));
Vector *ReadVec P((char *file_name ));

/* solver.c */
Solver *NewSolver P((char *file_name ));
void FreeSolver P((Solver *solver ));

/* vector.c */
Vector *NewVector P((double *data , int size ));
void FreeVector P((Vector *vector ));

/* write.c */
void WriteYSMP P((char *file_name , Matrix *matrix ));
void WriteVec P((char *file_name , Vector *vector ));

#undef P
