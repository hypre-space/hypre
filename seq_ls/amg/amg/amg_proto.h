#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* fortran.c */
void writeysm_ P((char *file_name , double *data , int *ia , int *ja , int *size , int file_name_len ));
void writevec_ P((char *file_name , double *data , int *size , int file_name_len ));

/* matrix.c */
Matrix *NewMatrix P((double *data , int *ia , int *ja , int size ));
void FreeMatrix P((Matrix *matrix ));

/* problem.c */
Problem *NewProblem P((char *file_name ));
void FreeProblem P((Problem *problem ));

/* random.c */
void SeedRand P((int seed ));
double Rand P((void ));

/* read.c */
Matrix *ReadYSMP P((char *file_name ));
Vector *ReadVec P((char *file_name ));

/* vector.c */
Vector *NewVector P((double *data , int size ));
void FreeVector P((Vector *vector ));

/* write.c */
void WriteYSMP P((char *file_name , Matrix *matrix ));
void WriteVec P((char *file_name , Vector *vector ));

#undef P
