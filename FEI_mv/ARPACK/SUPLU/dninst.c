#include <stdio.h>
#include "dcomplex.h"
#include  <math.h>

#define z(i,j) z[(*ldz)*( (j) -1 )+ (i) - 1]
#define v(i,j) v[(*ldz)*( (j) -1 )+ (i) - 1]
#define dr(i) dr[(i)-1]
#define di(i) di[(i)-1]

#define max(a,b) ((a)> (b)? (a) : (b))

extern int dsparse_preprocess_(int *, int *, int *, int *, double *, int *);
extern int dsparse_factor_(int *);
extern int dsparse_solve_(int *, double *, double *);
extern int dsparse_destroy(int *);

#ifdef USE_COMPLEX
extern int zsparse_preprocess_(int *, int *, int *, int *, doublecomplex *, int *);
extern int zsparse_factor_(int *);
extern int zsparse_solve_(int *, doublecomplex *, doublecomplex *);
extern int zsparse_destroy(int *);
#endif

extern double ddot_(int *, double *, int *, double *, int *);
extern void dmvm_(int *, double *, int *, int *, double *, double *, int *);
extern void dcopy_(int *, double *, int *, double *, int *);
extern double dlapy2_(double *, double *);

void dninst_(int *n,         int *nev,    double *sigmar,
             double *sigmai, int *colptr, int *rowind, 
             double *nzvals, double *dr,  double *di,  
             double *z,      int *ldz,    int *info,     double *ptol)

/*  Arguement list:

    n       (int*)    Dimension of the problem.  (INPUT)

    nev     (int*)    Number of eigenvalues requested.  (INPUT/OUTPUT)
                      This routine is used to compute NEV eigenvalues
                      nearest to a shift (sigmar, sigmai).
                      On return, it gives the number of converged
                      eigenvalues.

    sigmar  (double*) Real part of the shift. (INPUT)

    sigmai  (double*) Imaginar part of the shift. (INPUT)

    colptr  (int*)    dimension n+1. (INPUT)
                      Column pointers for the sparse matrix.

    rowind  (int*)    dimension colptr[*n]-1. (INPUT)
                      Row indices for the sparse matrix.

    nzvals  (double*) dimension colptr[*n]-1. (INPUT)
                      Nonzero values of the sparse matrix.
                      The sparse matrix is represented by
                      the above three arrays colptr, rowind, nzvals.

    dr      (double*) dimension nev+1.  (OUTPUT)
                      Real part of the eigenvalue.

    di      (double*) dimension nev+1.  (OUTPUT)
                      Imaginar part of the eigenvalue.

    z       (double*) dimension ldz by nev+1. (OUTPUT)
                      Eigenvector matrix.
                      If the j-th eigenvalue is real, the j-th column
                      of z contains the corresponding eigenvector.
                      If the j-th and j+1st eigenvalues form a complex
                      conjuagate pair, then the j-th column of z contains
                      the real part of the eigenvector, and the j+1st column
                      of z contains the imaginary part of the eigenvector.

    ldz      (int*)   The leading dimension of z. (INPUT)

    info     (int*)   Error flag to indicate whether the eigenvalues 
                      calculation is successful. (OUTPUT)
                      *info = 0, successful exit
                      *info = 1, Maximum number of iteration is reached
                                 before all requested eigenvalues 
                                 have converged.
*/

{
    int    i, j, ibegin, iend, ncv, neqns, token, order=2;
    int    lworkl, ldv,  nnz, ione = 1;
    double tol=1.0e-10, zero = 0.0;
    double *workl, *workd, *resid, *workev, *v, *ax;
    double numr, numi, denr, deni;
    int    *select, first;
    int    ido, ishfts, maxitr, mode, rvec, ierr1, ierr2;
    int    iparam[11], ipntr[14];
    char   *which="LM", bmat[2], *all="A";
#ifdef USE_COMPLEX
    doublecomplex *cvals, *cx, *crhs;
#endif

    neqns = *n;
    *info = 0;
    tol   = *ptol;
    if ( tol < 1.0e-10 ) tol = 1.0e-10;
    if ( tol > 1.0e-1  ) tol = 1.0e-1;

    if (*n - *nev < 2) {
       *info = -1000;
       fprintf(stderr, " NEV must be less than N-2!\n");
       goto Error_handle; 
    }

    /* set parameters and allocate temp space for ARPACK*/
    ncv = max(*nev+20, 2*(*nev));
    if (ncv > neqns) ncv = neqns;

    /* Convert from 1-based index to 0-based index */
    nnz = colptr[neqns]-1;
    for (j=0;j<=neqns;j++) colptr[j]--;
    for (i=0;i<nnz;i++) rowind[i]--;

    /* Subtract shift from the matrix */
   
    if ( *sigmai == 0.0) {
       /* real shift */
       for (j = 0; j<neqns; j++) {
          ibegin = colptr[j];
          iend   = colptr[j+1]-1;
          for (i=ibegin;i<=iend;i++) 
             if (j == rowind[i]) nzvals[i] = nzvals[i] - *sigmar;
       }
    }
    else {
       printf("Arpack/SuperLU : complex sigma not supported.\n");
       exit(1);
#ifdef USE_COMPLEX
       /* complex shift need additional storage for the
          shifted matrix */
       cvals = (doublecomplex*)malloc(nnz*sizeof(doublecomplex));
       if (!cvals) {
          fprintf(stderr, " Fail to allocate cvals!\n");
          goto Error_handle;
       }

       for (i = 0; i<nnz; i++) {
          cvals[i].r = nzvals[i];
          cvals[i].i = 0.0;
       }
       for (j = 0; j<neqns; j++) {
          ibegin = colptr[j];
          iend   = colptr[j+1]-1;
          for (i=ibegin;i<=iend;i++) 
             if (j == rowind[i]) {
                cvals[i].r = cvals[i].r - *sigmar;
                cvals[i].i = -(*sigmai); 
             }
       }
#endif
    }

    /* order and factor the shifted matrix */
    token = 0;
    if (*sigmai == 0.0) {
       dsparse_preprocess_(&token, &neqns, colptr, rowind, nzvals, &order);
       dsparse_factor_(&token);
    }
#ifdef USE_COMPLEX
    else {
       zsparse_preprocess_(&token, &neqns, colptr, rowind, cvals, &order);
       zsparse_factor_(&token);
    }
#endif


    /* add the shift back if shift is real */
    if (*sigmai == 0) {
       for (j = 0; j<neqns; j++) {
          ibegin = colptr[j];
          iend   = colptr[j+1]-1;
          for (i=ibegin;i<=iend;i++)
             if (j == rowind[i]) nzvals[i] = nzvals[i] + *sigmar;
       }
    }

    /* change from 0-based index to 1-based index */
    for (j=0;j<=neqns;j++) colptr[j]++;
    for (i=0;i<nnz;i++) rowind[i]++;

    /* set parameters and allocate temp space for ARPACK*/
    lworkl = 3*ncv*ncv+6*ncv;
    ido    = 0;
    ierr1  = 0;
    ishfts = 1;
    maxitr = 300;
    mode   = 3;
    ldv    = neqns;

    iparam[0] = ishfts;
    iparam[2] = maxitr;
    iparam[6] = mode;

    resid = (double*) malloc(neqns*sizeof(double));
    if (!resid) {
       fprintf(stderr," Fail to allocate resid\n");
       goto Error_handle;
    }
    workl = (double*) malloc(lworkl*sizeof(double));
    if (!workl) {
       fprintf(stderr," Fail to allocate workl\n");
       goto Error_handle;
    }
    v     = (double*) malloc(ldv*ncv*sizeof(double));
    if (!v) {
       fprintf(stderr," Fail to allocate v\n");
       goto Error_handle;
    }
    workd = (double*) malloc(neqns*3*sizeof(double));
    if (!workd) {
       fprintf(stderr, " Fail to allocate workd\n");
       goto Error_handle;
    }
    workev= (double*) malloc(ncv*3*sizeof(double));
    if (!workev) {
       fprintf(stderr, " Fail to allocate workev\n");
       goto Error_handle;
    }
    select= (int*) malloc(ncv*sizeof(int));
    if (!select) {
       fprintf(stderr, " Fail to allocate select\n");
       goto Error_handle;
    }
#ifdef USE_COMPLEX
    if (*sigmai != 0.0) {
       cx    = (doublecomplex*)malloc(neqns*sizeof(doublecomplex));
       if (!cx) {
          fprintf(stderr, " Fail to allocate cx\n");
          goto Error_handle;
       }
       crhs  = (doublecomplex*)malloc(neqns*sizeof(doublecomplex));
       if (!crhs) {
          fprintf(stderr, " Fail to allocate crhs\n");
          goto Error_handle;
       }
    }
#endif

    /* intialize all work arrays */
    for (i=0;i<neqns;i++) resid[i] = 0.0;
    for (i=0;i<lworkl;i++) workl[i]=0.0;
    for (i=0;i<ldv*ncv;i++) v[i]=0.0;
    for (i=0;i<3*neqns;i++) workd[i]=0.0;
    for (i=0;i<3*ncv;i++) workev[i]=0.0;
    for (i=0;i<ncv;i++) select[i] = 0;

    if (*sigmai == 0.0) {
      bmat[0] = 'I';
    }
    else {
      bmat[0] = 'G';
    } 

    /* ARPACK reverse comm to compute eigenvalues and eigenvectors */
    if (*sigmai == 0.0) {
       while (ido != 99 ) {
          dnaupd_(&ido,    bmat, n,    which,  nev,   &tol,  resid, 
                  &ncv,    v,    &ldv, iparam, ipntr, workd, workl,
                  &lworkl, &ierr1);
          if (ido == -1 || ido == 1) {
             dsparse_solve_(&token, &workd[ipntr[1]-1],&workd[ipntr[0]-1]);
          }
       }
    }
#ifdef USE_COMPLEX
    else {
       while (ido != 99 ) {
          dnaupd_(&ido,    bmat, n,    which,  nev,   &tol,  resid,
                  &ncv,    v,    &ldv, iparam, ipntr, workd, workl,
                  &lworkl, &ierr1);
          if (ido == -1) {
              dcopy_(n, &workd[ipntr[0]-1], &ione, &workd[ipntr[1]-1], &ione);
              for (i=0;i<neqns;i++) {
                 crhs[i].r = workd[ipntr[1]-1+i];
                 crhs[i].i = 0.0; 
              } 
              zsparse_solve_(&token, cx, crhs);
              for (i=0;i<neqns;i++) {
                 workd[ipntr[1]-1+i] = cx[i].r;
              }
          }
          else if (ido == 1) {
              for (i=0;i<neqns;i++) {
                 crhs[i].r = workd[ipntr[2]-1+i];
                 crhs[i].i = 0.0;
              }
              zsparse_solve_(&token, cx, crhs);
              for (i=0;i<neqns;i++) {
                 workd[ipntr[1]-1+i] = cx[i].r;
              }
          }
          else if (ido == 2) {
              dcopy_(n, &workd[ipntr[0]-1], &ione, 
                        &workd[ipntr[1]-1], &ione);
          } 
       }
    }
#endif

    /* ARPACK postprocessing */
    if (ierr1 < 0) {
       fprintf(stderr, " Error with _naupd, ierr = %d\n", ierr1);
    }
    else {
       rvec = 1;

       dneupd_(&rvec, all,   select,  dr,   di,   z,      ldz, 
              sigmar, sigmai,workev,  bmat, n,    which,  nev,
              &tol,   resid, &ncv,    v,    &ldv, iparam, ipntr, 
              workd,  workl, &lworkl, &ierr2);

       *nev = iparam[4];

       if (ierr2 != 0) {
          fprintf(stderr," Error with _neupd, ierr = %d\n",ierr2);
          goto Error_handle;
       }
    } 

#ifdef USE_COMPLEX
    if (*sigmai != 0) {
       /* Use Rayleigh quotient to recover Ritz values */
       ax = (double*)malloc(neqns*sizeof(double));
       if (!ax) {
          fprintf(stderr, " Fail to allocate AX!\n");
          goto Error_handle;
       }

       for (i=0;i<neqns;i++) ax[i] = 0.0;
       first = 1;
       for (j = 1; j<=*nev; j++) {
          if (di(j) == 0.0) {
             dmvm_(n, nzvals, rowind, colptr, &z(1,j), ax, &ione);
             numr = ddot_(n, &z(1,j), &ione, ax, &ione);
             dcopy_(n, &z(1,j), &ione, ax, &ione);
             denr = ddot_(n, &z(1,j), &ione, ax, &ione);
             dr(j) = numr/denr;  
          }
          else if (first) {
             /* compute trans(x) A x */
             dmvm_(n, nzvals, rowind, colptr, &z(1,j), ax, &ione);
             numr = ddot_(n, &z(1,j), &ione, ax, &ione);
             numi = ddot_(n, &z(1,j+1), &ione, ax, &ione);
             dmvm_(n, nzvals, rowind, colptr, &z(1,j+1), ax, &ione);
             numr = numr  + ddot_(n, &z(1,j+1), &ione, ax, &ione);
             numi = -numi + ddot_(n, &z(1,j), &ione, ax, &ione);

             /* compute trans(x) M x */
             dcopy_(n, &z(1,j), &ione, ax, &ione);
             denr = ddot_(n, &z(1,j),   &ione, ax, &ione);
             deni = ddot_(n, &z(1,j+1), &ione, ax, &ione);
             dcopy_(n, &z(1,j+1), &ione, ax, &ione);
             denr = denr + ddot_(n, &z(1,j+1), &ione, ax, &ione);
             deni = -deni + ddot_(n, &z(1,j), &ione, ax, &ione); 

             dr(j) = (numr*denr+numi*deni)/dlapy2_(&denr, &deni);
             di(j) = (numi*denr-numr*deni)/dlapy2_(&denr, &deni);
             first = 0;
          }
          else {
             dr(j) = dr(j-1);
             di(j) = -di(j-1);
             first = 1;
          }
       } 
    }
#endif
  
    free(resid);
    free(workl);
    free(v);
    free(workd); 
    free(workev);
    free(select);
#ifdef USE_COMPLEX
    if (*sigmai != 0.0) {
       free(crhs);
       free(cx);
       free(cvals);
       free(ax);
       zsparse_destroy_(&token);
    }
    else {
#endif
       dsparse_destroy_(&token);
#ifdef USE_COMPLEX
    }
#endif

Error_handle:
    if (ierr1 != 0) *info = ierr1;
    if (ierr1 == 1) 
       fprintf(stderr, " Maxiumum number of iteration reached.\n");
}
