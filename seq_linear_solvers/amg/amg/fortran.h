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
 * C/Fortran interface macros
 *
 *****************************************************************************/

#ifndef HYPRE_AMG_FORTRAN_HEADER
#define HYPRE_AMG_FORTRAN_HEADER


#if defined(IRIX) || defined(DEC)
#define hypre_NAME_C_FOR_FORTRAN(name) name##_
#define hypre_NAME_FORTRAN_FOR_C(name) name##_
#else
#define hypre_NAME_C_FOR_FORTRAN(name) name##__
#define hypre_NAME_FORTRAN_FOR_C(name) name##_
#endif

/* setup */
#define hypre_CALL_SETUP(Setup_err_flag, A, amg_data) \
hypre_NAME_FORTRAN_FOR_C(setup)(&Setup_err_flag,\
			  &hypre_AMGDataNumLevels(amg_data),\
			  &hypre_AMGDataNSTR(amg_data),\
			  &hypre_AMGDataECG(amg_data),\
			  &hypre_AMGDataNCG(amg_data),\
			  &hypre_AMGDataEWT(amg_data),\
			  &hypre_AMGDataNWT(amg_data),\
			  hypre_AMGDataICDep(amg_data),\
			  &hypre_AMGDataIOutDat(amg_data),\
			  &hypre_AMGDataNumUnknowns(amg_data),\
			  hypre_AMGDataIMin(amg_data),\
			  hypre_AMGDataIMax(amg_data),\
			  hypre_MatrixData(A),\
			  hypre_MatrixIA(A),\
			  hypre_MatrixJA(A),\
			  hypre_AMGDataIU(amg_data),\
			  hypre_AMGDataIP(amg_data),\
			  hypre_AMGDataICG(amg_data),\
			  hypre_AMGDataIFG(amg_data),\
			  hypre_MatrixData(hypre_AMGDataP(amg_data)),\
			  hypre_MatrixIA(hypre_AMGDataP(amg_data)),\
			  hypre_MatrixJA(hypre_AMGDataP(amg_data)),\
			  hypre_AMGDataIPMN(amg_data),\
			  hypre_AMGDataIPMX(amg_data),\
			  hypre_AMGDataIV(amg_data),\
			  &hypre_AMGDataNDIMU(amg_data),\
			  &hypre_AMGDataNDIMP(amg_data),\
			  &hypre_AMGDataNDIMA(amg_data),\
			  &hypre_AMGDataNDIMB(amg_data),\
			  hypre_AMGDataLogFileName(amg_data),\
			  strlen(hypre_AMGDataLogFileName(amg_data)))

void hypre_NAME_FORTRAN_FOR_C(setup)(int *, int *, int *,
			       double *, int *, double *, int *,
			       int *, int *, int *, int *, int *,
			       double *, int *, int *,
			       int *, int *, int *, int *,
			       double *, int *, int *,
			       int *, int *, int *,
			       int *, int *, int *, int *,
			       char *, long);


/* idec */
void hypre_NAME_FORTRAN_FOR_C(idec)(int *, int *, int *, int *);

#endif

