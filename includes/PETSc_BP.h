 
#ifndef _PROTOS_HEADER
#define _PROTOS_HEADER
 
#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* Amg_Apply.c */
int Amg_Apply P((void *solver_data , Vec b , Vec x ));

/* Apply.c */
int INCFACT_Apply P((void *solver_data , Vec b , Vec x ));

/* BlockJacobiAmgPcKsp.c */
int BlockJacobiAmgPcKsp P((Mat A , Vec x , Vec b , void *data ));

/* BlockJacobiAmgPcKspSetup.c */
void *BlockJacobiAmgPcKspSetup P((Mat A ));
int BlockJacobiAmgPcKspFinalize P((void *data ));

/* BlockJacobiICPcKspInitialize.c */
void *BlockJacobiICPcKspInitialize P((void *in_ptr ));
int BlockJacobiICPcKspFinalize P((void *data ));

/* BlockJacobiICPcKspSetup.c */
int BlockJacobiICPcKspSetup P((void *in_ptr , Mat A , Vec x , Vec b ));

/* BlockJacobiICPcKspSolve.c */
int BlockJacobiICPcKspSolve P((void *data , Mat A , Vec x , Vec b ));

/* BlockJacobiICPcKspSolver.c */
int BlockJacobiINCFACTPcKsp P((Mat A , Vec x , Vec b , void *data ));

/* BlockJacobiILUPcKspInitialize.c */
void *BlockJacobiILUPcKspInitialize P((void *in_ptr ));
int BlockJacobiILUPcKspFinalize P((void *data ));

/* BlockJacobiILUPcKspSetup.c */
int BlockJacobiILUPcKspSetup P((void *in_ptr , Mat A , Vec x , Vec b ));

/* BlockJacobiILUPcKspSolve.c */
int BlockJacobiILUPcKspSolve P((void *data , Mat A , Vec x , Vec b ));

/* BlockJacobiPcKspInitialize.c */
void *BlockJacobiINCFACTPcKspInitialize P((void *in_ptr ));
int BlockJacobiINCFACTPcKspFinalize P((void *data ));

/* BlockJacobiPcKspSetup.c */
int BlockJacobiINCFACTPcKspSetup P((void *in_ptr , Mat A , Vec x , Vec b ));

/* BlockJacobiPcKspSolve.c */
int BlockJacobiINCFACTPcKspSolve P((void *data , Mat A , Vec x , Vec b ));

/* CsrConvert.c */
int CsrGen_to_CsrDiagFirst P((Matrix *A , int **diag_loc_ret ));
int CsrDiagFirst_backto_CsrGen P((Matrix *A , int *diag_loc ));

/* IC_Apply.c */
int IC_Apply P((void *solver_data , Vec b , Vec x ));

/* ILU_Apply.c */
int ILU_Apply P((void *solver_data , Vec b , Vec x ));

/* ParMatrixAllocate.c */
void *ParMatrixAllocate P((void *port , int context ));
int ParMatrixFree P((void *data ));

/* ParMatrixAssemble.c */
int ParMatrixAssemble P((void *in_matrix ));

/* ParMatrixInitialize.c */
int ParMatrixInitialize P((void *in_ptr , int N , int n , int Nnz , int Lnnz ));

/* ReadMPIAIJ_Amg.c */
int ReadMPIAIJ_Amg P((Mat *A , char *file_name , int *n ));

/* ReadMPIAIJ_HB.c */
int ConHB2MPIAIJ P((Mat *A , char *file_name ));

/* ReadMPIAIJ_ILU.c */
int ReadMPIAIJ_ILU P((Mat *A , char *file_name , int *n ));

/* ReadMPIAIJ_UNSYM.c */
int ReadMPIAIJ_UNSYM P((Mat *A , char *file_name , int *n ));

/* ReadMPIVec.c */
int ReadMPIVec P((Vec *x , char *file_name ));

/* SetParMatrixCoeffsCoord.c */
int SetParMatrixCoeffsCoord P((void *in_matrix , int m , int n , int *im , int *in , double *data ));

/* SetParMatrixOptions.c */
int SetParMatrixImplementation P((void *in_matrix , int impl ));
int SetParMatrixMode P((void *in_matrix , int Mode ));

/* matrix.c */
Matrix *NewMatrix P((double *data , int *ia , int *ja , int size ));
void FreeMatrix P((Matrix *matrix ));

#undef P
 
#endif
 
