
#ifndef _PROTOS_HEADER
#define _PROTOS_HEADER

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* amg.c */
void *amg_Initialize P((void *port_data ));
void amg_Finalize P((void *data ));

/* amg_cycle.c */
int amg_Cycle P((Vector **U_array , Vector **F_array , double tol , void *data ));

/* amg_params.c */
void amg_SetLevMax P((int levmax , void *data ));
void amg_SetNCG P((int ncg , void *data ));
void amg_SetECG P((double ecg , void *data ));
void amg_SetNWT P((int nwt , void *data ));
void amg_SetEWT P((double ewt , void *data ));
void amg_SetNSTR P((int nstr , void *data ));
void amg_SetNCyc P((int ncyc , void *data ));
void amg_SetMU P((int *mu , void *data ));
void amg_SetNTRLX P((int *ntrlx , void *data ));
void amg_SetIPRLX P((int *iprlx , void *data ));
void amg_SetLogging P((int ioutdat , char *log_file_name , void *data ));
void amg_SetNumUnknowns P((int num_unknowns , void *data ));
void amg_SetNumPoints P((int num_points , void *data ));
void amg_SetIU P((int *iu , void *data ));
void amg_SetIP P((int *ip , void *data ));
void amg_SetIV P((int *iv , void *data ));
void amg_SetXP P((double *xp , void *data ));
void amg_SetYP P((double *yp , void *data ));
void amg_SetZP P((double *zp , void *data ));

/* amg_relax.c */
int amg_Relax P((Vector *u , Vector *f , Matrix *A , VectorInt *ICG , VectorInt *IV , int min_point , int max_point , int point_type , int relax_type , double *D_mat , double *S_vec ));
int gselim P((double *A , double *x , int n ));

/* amg_setup.c */
int amg_Setup P((Matrix *A , void *data ));

/* amg_solve.c */
int amg_Solve P((Vector *u , Vector *f , double tol , void *data ));

/* config.cygwin32 */

/* data.c */
AMGData *amg_NewData P((int levmax , int ncg , double ecg , int nwt , double ewt , int nstr , int ncyc , int *mu , int *ntrlx , int *iprlx , int ioutdat , int cycle_op_count , char *log_file_name ));
void amg_FreeData P((AMGData *amg_data ));

/* matrix.c */
Matrix *NewMatrix P((double *data , int *ia , int *ja , int size ));
void FreeMatrix P((Matrix *matrix ));
void Matvec P((double alpha , Matrix *A , Vector *x , double beta , Vector *y ));
void MatvecT P((double alpha , Matrix *A , Vector *x , double beta , Vector *y ));

/* random.c */
void SeedRand P((int seed ));
double Rand P((void ));

/* vector.c */
Vector *NewVector P((double *data , int size ));
void FreeVector P((Vector *vector ));
VectorInt *NewVectorInt P((int *data , int size ));
void FreeVectorInt P((VectorInt *vector ));
void InitVector P((Vector *v , double value ));
void InitVectorRandom P((Vector *v ));
void CopyVector P((Vector *x , Vector *y ));
void ScaleVector P((double alpha , Vector *y ));
void Axpy P((double alpha , Vector *x , Vector *y ));
double InnerProd P((Vector *x , Vector *y ));

/* write.c */
void WriteYSMP P((char *file_name , Matrix *matrix ));
void WriteVec P((char *file_name , Vector *vector ));
void WriteVecInt P((char *file_name , VectorInt *vector ));
void WriteSetupParams P((void *data ));
void WriteSolverParams P((double tol , void *data ));

#undef P

#endif

