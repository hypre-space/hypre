/*BHEADER**********************************************************************
 * (c) 1996   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Public header file for AMG
 *
 *****************************************************************************/

#ifndef HYPRE_AMG_HEADER
#define HYPRE_AMG_HEADER


#include "matrix.h"
#include "vector.h"

/*--------------------------------------------------------------------------
 * Miscellaneous defines
 *--------------------------------------------------------------------------*/
 
#ifndef NULL
#define NULL 0
#endif

#define  hypre_NDIMU(nv)  (3*nv)
#define  hypre_NDIMP(np)  (3*np)
#define  hypre_NDIMA(na)  (4*na)
#define  hypre_NDIMB(na)  (3*na)

/*--------------------------------------------------------------------------
 * User prototypes
 *--------------------------------------------------------------------------*/
 
#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* amg.c */
void *HYPRE_AMGInitialize P((void *port_data ));
void HYPRE_AMGFinalize P((void *data ));

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
void HYPRE_AMGSetIOutDat P((int ioutdat , void *data ));
void HYPRE_AMGSetLogFileName P((char *log_file_name , void *data ));
void HYPRE_AMGSetNumUnknowns P((int num_unknowns , void *data ));
void HYPRE_AMGSetNumPoints P((int num_points , void *data ));
void HYPRE_AMGSetIU P((int *iu , void *data ));
void HYPRE_AMGSetIP P((int *ip , void *data ));
void HYPRE_AMGSetIV P((int *iv , void *data ));

/* amg_setup.c */
int HYPRE_AMGSetup P((hypre_Matrix *A , void *data ));

/* amg_solve.c */
int HYPRE_AMGSolve P((hypre_Vector *u , hypre_Vector *f , double tol , void *data ));

/* vector.c */
hypre_Vector *hypre_NewVector P((double *data , int size ));
void hypre_FreeVector P((hypre_Vector *vector ));
void hypre_InitVector P((hypre_Vector *v , double value ));
void hypre_InitVectorRandom P((hypre_Vector *v ));
void hypre_CopyVector P((hypre_Vector *x , hypre_Vector *y ));
void hypre_ScaleVector P((double alpha , hypre_Vector *y ));
void hypre_Axpy P((double alpha , hypre_Vector *x , hypre_Vector *y ));
double hypre_InnerProd P((hypre_Vector *x , hypre_Vector *y ));

/* matrix.c */
hypre_Matrix *hypre_NewMatrix P((double *data , int *ia , int *ja , int size ));
void hypre_FreeMatrix P((hypre_Matrix *matrix ));
void hypre_Matvec P((double alpha , hypre_Matrix *A , hypre_Vector *x , double beta , hypre_Vector *y ));

#undef P


#endif
