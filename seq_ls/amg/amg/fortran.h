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

#ifndef _AMG_FORTRAN_HEADER
#define _AMG_FORTRAN_HEADER


#if defined(IRIX) || defined(DEC)
#define NAME_C_FOR_FORTRAN(name) name##_
#define NAME_FORTRAN_FOR_C(name) name##_
#else
#define NAME_C_FOR_FORTRAN(name) name##__
#define NAME_FORTRAN_FOR_C(name) name##_
#endif

/* setup */
#define CALL_SETUP(Setup_err_flag, A, amg_data) \
NAME_FORTRAN_FOR_C(setup)(&Setup_err_flag,\
			  &AMGDataNumLevels(amg_data),\
			  &AMGDataNSTR(amg_data),\
			  &AMGDataECG(amg_data),\
			  &AMGDataNCG(amg_data),\
			  &AMGDataEWT(amg_data),\
			  &AMGDataNWT(amg_data),\
			  AMGDataICDep(amg_data),\
			  &AMGDataIOutDat(amg_data),\
			  &AMGDataNumUnknowns(amg_data),\
			  AMGDataIMin(amg_data),\
			  AMGDataIMax(amg_data),\
			  MatrixData(A),\
			  MatrixIA(A),\
			  MatrixJA(A),\
			  AMGDataIU(amg_data),\
			  AMGDataIP(amg_data),\
			  AMGDataICG(amg_data),\
			  AMGDataIFG(amg_data),\
			  MatrixData(AMGDataP(amg_data)),\
			  MatrixIA(AMGDataP(amg_data)),\
			  MatrixJA(AMGDataP(amg_data)),\
			  AMGDataIPMN(amg_data),\
			  AMGDataIPMX(amg_data),\
			  AMGDataIV(amg_data),\
			  AMGDataXP(amg_data),\
			  AMGDataYP(amg_data),\
			  &AMGDataNDIMU(amg_data),\
			  &AMGDataNDIMP(amg_data),\
			  &AMGDataNDIMA(amg_data),\
			  &AMGDataNDIMB(amg_data),\
			  AMGDataLogFileName(amg_data),\
			  strlen(AMGDataLogFileName(amg_data)))

void NAME_FORTRAN_FOR_C(setup)(int *, int *, int *,
			       double *, int *, double *, int *,
			       int *, int *, int *, int *, int *,
			       double *, int *, int *,
			       int *, int *, int *, int *,
			       double *, int *, int *,
			       int *, int *, int *,
			       double *, double *,
			       int *, int *, int *, int *,
			       char *, long);


/* idec */
void NAME_FORTRAN_FOR_C(idec)(int *, int *, int *, int *);

#endif

