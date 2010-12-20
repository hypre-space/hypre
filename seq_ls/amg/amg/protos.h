
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
HYPRE_Int hypre_AMGCycle P((hypre_Vector **U_array , hypre_Vector **F_array , double tol , void *data ));

/* amg_params.c */
void HYPRE_AMGSetLevMax P((HYPRE_Int levmax , void *data ));
void HYPRE_AMGSetNCG P((HYPRE_Int ncg , void *data ));
void HYPRE_AMGSetECG P((double ecg , void *data ));
void HYPRE_AMGSetNWT P((HYPRE_Int nwt , void *data ));
void HYPRE_AMGSetEWT P((double ewt , void *data ));
void HYPRE_AMGSetNSTR P((HYPRE_Int nstr , void *data ));
void HYPRE_AMGSetNCyc P((HYPRE_Int ncyc , void *data ));
void HYPRE_AMGSetMU P((HYPRE_Int *mu , void *data ));
void HYPRE_AMGSetNTRLX P((HYPRE_Int *ntrlx , void *data ));
void HYPRE_AMGSetIPRLX P((HYPRE_Int *iprlx , void *data ));
void HYPRE_AMGSetLogging P((HYPRE_Int ioutdat , char *log_file_name , void *data ));
void HYPRE_AMGSetNumUnknowns P((HYPRE_Int num_unknowns , void *data ));
void HYPRE_AMGSetNumPoints P((HYPRE_Int num_points , void *data ));
void HYPRE_AMGSetIU P((HYPRE_Int *iu , void *data ));
void HYPRE_AMGSetIP P((HYPRE_Int *ip , void *data ));
void HYPRE_AMGSetIV P((HYPRE_Int *iv , void *data ));

/* amg_relax.c */
HYPRE_Int hypre_AMGRelax P((hypre_Vector *u , hypre_Vector *f , hypre_Matrix *A , hypre_VectorInt *ICG , hypre_VectorInt *IV , HYPRE_Int min_point , HYPRE_Int max_point , HYPRE_Int point_type , HYPRE_Int relax_type , double *D_mat , double *S_vec ));
HYPRE_Int gselim P((double *A , double *x , HYPRE_Int n ));

/* amg_setup.c */
HYPRE_Int HYPRE_AMGSetup P((hypre_Matrix *A , void *data ));

/* amg_solve.c */
HYPRE_Int HYPRE_AMGSolve P((hypre_Vector *u , hypre_Vector *f , double tol , void *data ));

/* config.cygwin32 */

/* data.c */
hypre_AMGData *hypre_AMGNewData P((HYPRE_Int levmax , HYPRE_Int ncg , double ecg , HYPRE_Int nwt , double ewt , HYPRE_Int nstr , HYPRE_Int ncyc , HYPRE_Int *mu , HYPRE_Int *ntrlx , HYPRE_Int *iprlx , HYPRE_Int ioutdat , HYPRE_Int cycle_op_count , char *log_file_name ));
void hypre_AMGFreeData P((hypre_AMGData *amg_data ));

/* matrix.c */
hypre_Matrix *hypre_NewMatrix P((double *data , HYPRE_Int *ia , HYPRE_Int *ja , HYPRE_Int size ));
void hypre_FreeMatrix P((hypre_Matrix *matrix ));
void hypre_Matvec P((double alpha , hypre_Matrix *A , hypre_Vector *x , double beta , hypre_Vector *y ));
void hypre_MatvecT P((double alpha , hypre_Matrix *A , hypre_Vector *x , double beta , hypre_Vector *y ));

/* random.c */
void hypre_SeedRand P((HYPRE_Int seed ));
double hypre_Rand P((void ));

/* timing.c */
void HYPRE_AMGClock_init P((void ));
amg_Clock_t HYPRE_AMGClock P((void ));
amg_CPUClock_t HYPRE_AMGCPUClock P((void ));
void HYPRE_AMGPrintTiming P((double time_ticks , double cpu_ticks ));

/* vector.c */
hypre_Vector *hypre_NewVector P((double *data , HYPRE_Int size ));
void hypre_FreeVector P((hypre_Vector *vector ));
hypre_VectorInt *hypre_NewVectorInt P((HYPRE_Int *data , HYPRE_Int size ));
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

