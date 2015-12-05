/*
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 1, 1997
 *
 * Changes made to this file addressing issue regarding calls to
 * blas/lapack functions (Dec 2003 at LLNL)
 */
#ifndef __SUPERLU_CNAMES /* allow multiple inclusions */
#define __SUPERLU_CNAMES

/*
 * These macros define how C routines will be called.  
 * They have been modified to use of the new autoconf Fortran
 * name mangling support, F77_FUNC, which is wrapped in the
 * hypre_F90_NAME_BLAS macro.
 */

/*
 * These defines set up the naming scheme required to have a fortran 77
 * routine call a C routine
 * No redefinition necessary to have following Fortran to C interface:
 *           FORTRAN CALL               C DECLARATION
 *           call dgemm(...)           void hypre_F90_NAME_BLAS(dgemm,DGEMM)(...)
 *
 * This is the default.
 */

#endif
/***
#define hypre_F90_NAME_BLAS(dasum,DASUM)    dasum
#define hypre_F90_NAME_BLAS(idamax,IDAMAX)   idamax
#define hypre_F90_NAME_BLAS(dcopy,DCOPY)    dcopy
#define hypre_F90_NAME_BLAS(dscal,DSCAL)    dscal
#define hypre_F90_NAME_BLAS(dger,DGER)     dger
#define hypre_F90_NAME_BLAS(dnrm2,DNRM2)    dnrm2
#define hypre_F90_NAME_BLAS(dsymv,DSYMV)    dsymv
#define hypre_F90_NAME_BLAS(ddot,DDOT)     ddot
#define hypre_F90_NAME_BLAS(daxpy,DAXPY)    daxpy
#define hypre_F90_NAME_BLAS(dsyr2,DSYR2)    dsyr2
#define hypre_F90_NAME_BLAS(drot,DROT)     drot
#define hypre_F90_NAME_BLAS(dgemv,DGEMV)    dgemv
#define hypre_F90_NAME_BLAS(dtrsv,DTRSV)    dtrsv
#define hypre_F90_NAME_BLAS(dgemm,DGEMM)    dgemm
#define hypre_F90_NAME_BLAS(dtrsm,DTRSM)    dtrsm

#define hypre_F90_NAME_BLAS(xerbla,XERBLA)  xerbla
#define hypre_F90_NAME_BLAS(dpotrf,DPOTRF)   dpotrf
#define hypre_F90_NAME_BLAS(dgels,DGELS)    dgels
#define hypre_F90_NAME_BLAS(dpotrs,DPOTRS)   dpotrs
#define hypre_F90_NAME_BLAS(lsame,LSAME)    lsame
#define hypre_F90_NAME_BLAS(dlamch,DLAMCH)  dlamch

#define c_bridge_dgssv_ c_bridge_dgssv
***/
#endif /* __SUPERLU_CNAMES */
