
#ifndef HYPRE_PROTOS_HEADER
#define HYPRE_PROTOS_HEADER

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* amg.c */
void *HYPRE_AMGInitialize P((void *port_data ));
void HYPRE_AMGFinalize P((void *data ));

/* amg_cycle.c */
int hypre_AMGCycle P((hypre_Vector **U_array , hypre_Vector **F_array , double tol , void *data ));

/* amg_params.c */
void HYPRE_AMGSetLevMax P((int levmax , void *data ));
void HYPRE_AMGSetNCG P((int ncg , void *data ));
void HYPRE_AMGSetECG P((double ecg , void *data ));
void HYPRE_AMGSetNWT P((int nwt , void *data ));
void HYPRE_AMGSetEWT P((double ewt , void *data ));
void HYPRE_AMGSetNSTR P((int nstr , void *data ));
void HYPRE_AMGSetNCyc P((int ncyc , void *data ));
void HYPRE_AMGSetMU P((int *mu , void *data ));
void HYPRE_AMGSetNTRLX P((int *ntrlx , void *data ));
void HYPRE_AMGSetIPRLX P((int *iprlx , void *data ));
void HYPRE_AMGSetLogging P((int ioutdat , char *log_file_name , void *data ));
void HYPRE_AMGSetNumUnknowns P((int num_unknowns , void *data ));
void HYPRE_AMGSetNumPoints P((int num_points , void *data ));
void HYPRE_AMGSetIU P((int *iu , void *data ));
void HYPRE_AMGSetIP P((int *ip , void *data ));
void HYPRE_AMGSetIV P((int *iv , void *data ));

/* amg_relax.c */
int hypre_AMGRelax P((hypre_Vector *u , hypre_Vector *f , hypre_Matrix *A , hypre_VectorInt *ICG , hypre_VectorInt *IV , int min_point , int max_point , int point_type , int relax_type , double *D_mat , double *S_vec ));
int gselim P((double *A , double *x , int n ));

/* amg_setup.c */
int HYPRE_AMGSetup P((hypre_Matrix *A , void *data ));

/* amg_solve.c */
int HYPRE_AMGSolve P((hypre_Vector *u , hypre_Vector *f , double tol , void *data ));

/* config.cygwin32 */

/* data.c */
hypre_AMGData *hypre_AMGNewData P((int levmax , int ncg , double ecg , int nwt , double ewt , int nstr , int ncyc , int *mu , int *ntrlx , int *iprlx , int ioutdat , int cycle_op_count , char *log_file_name ));
void hypre_AMGFreeData P((hypre_AMGData *amg_data ));

/* matrix.c */
hypre_Matrix *hypre_NewMatrix P((double *data , int *ia , int *ja , int size ));
void hypre_FreeMatrix P((hypre_Matrix *matrix ));
void hypre_Matvec P((double alpha , hypre_Matrix *A , hypre_Vector *x , double beta , hypre_Vector *y ));
void hypre_MatvecT P((double alpha , hypre_Matrix *A , hypre_Vector *x , double beta , hypre_Vector *y ));

/* random.c */
void hypre_SeedRand P((int seed ));
double hypre_Rand P((void ));

/* timing.c */
void HYPRE_AMGClock_init P((void ));
amg_Clock_t HYPRE_AMGClock P((void ));
amg_CPUClock_t HYPRE_AMGCPUClock P((void ));
void HYPRE_AMGPrintTiming P((double time_ticks , double cpu_ticks ));

/* vector.c */
hypre_Vector *hypre_NewVector P((double *data , int size ));
void hypre_FreeVector P((hypre_Vector *vector ));
hypre_VectorInt *hypre_NewVectorInt P((int *data , int size ));
void hypre_FreeVectorInt P((hypre_VectorInt *vector ));
void hypre_InitVector P((hypre_Vector *v , double value ));
void hypre_InitVectorRandom P((hypre_Vector *v ));
void hypre_CopyVector P((hypre_Vector *x , hypre_Vector *y ));
void hypre_ScaleVector P((double alpha , hypre_Vector *y ));
void hypre_Axpy P((double alpha , hypre_Vector *x , hypre_Vector *y ));
double hypre_InnerProd P((hypre_Vector *x , hypre_Vector *y ));

/* write.c */
void hypre_WriteYSMP P((char *file_name , hypre_Matrix *matrix ));
void hypre_WriteVec P((char *file_name , hypre_Vector *vector ));
void hypre_WriteVecInt P((char *file_name , hypre_VectorInt *vector ));
void hypre_WriteSetupParams P((void *data ));
void hypre_WriteSolverParams P((double tol , void *data ));

#undef P

#endif

