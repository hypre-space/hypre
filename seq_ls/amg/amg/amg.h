/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/





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
void HYPRE_AMGSetIOutDat P((HYPRE_Int ioutdat , void *data ));
void HYPRE_AMGSetLogFileName P((char *log_file_name , void *data ));
void HYPRE_AMGSetNumUnknowns P((HYPRE_Int num_unknowns , void *data ));
void HYPRE_AMGSetNumPoints P((HYPRE_Int num_points , void *data ));
void HYPRE_AMGSetIU P((HYPRE_Int *iu , void *data ));
void HYPRE_AMGSetIP P((HYPRE_Int *ip , void *data ));
void HYPRE_AMGSetIV P((HYPRE_Int *iv , void *data ));

/* amg_setup.c */
HYPRE_Int HYPRE_AMGSetup P((hypre_Matrix *A , void *data ));

/* amg_solve.c */
HYPRE_Int HYPRE_AMGSolve P((hypre_Vector *u , hypre_Vector *f , double tol , void *data ));

/* vector.c */
hypre_Vector *hypre_NewVector P((double *data , HYPRE_Int size ));
void hypre_FreeVector P((hypre_Vector *vector ));
void hypre_InitVector P((hypre_Vector *v , double value ));
void hypre_InitVectorRandom P((hypre_Vector *v ));
void hypre_CopyVector P((hypre_Vector *x , hypre_Vector *y ));
void hypre_ScaleVector P((double alpha , hypre_Vector *y ));
void hypre_Axpy P((double alpha , hypre_Vector *x , hypre_Vector *y ));
double hypre_InnerProd P((hypre_Vector *x , hypre_Vector *y ));

/* matrix.c */
hypre_Matrix *hypre_NewMatrix P((double *data , HYPRE_Int *ia , HYPRE_Int *ja , HYPRE_Int size ));
void hypre_FreeMatrix P((hypre_Matrix *matrix ));
void hypre_Matvec P((double alpha , hypre_Matrix *A , hypre_Vector *x , double beta , hypre_Vector *y ));

#undef P


#endif
