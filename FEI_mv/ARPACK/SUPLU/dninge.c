#include <stdio.h>
#include "dcomplex.h"
#include  <math.h>

#define z(i,j) z[(*ldz)*( (j) -1 )+ (i) - 1]
#define dr(i) dr[(i)-1]
#define di(i) di[(i)-1]

#define max(a,b) ((a)> (b)? (a) : (b))

extern int dsparse_preprocess_(int *, int *, int *, int *, double *, int *);
extern int dsparse_factor_(int *);
extern int dsparse_solve_(int *, double *, double *);
extern int dsparse_destroy(int *);

extern int zsparse_preprocess_(int *, int *, int *, int *, doublecomplex *, 
                               int *);
extern int zsparse_factor_(int *);
extern int zsparse_solve_(int *, doublecomplex *, doublecomplex *);
extern int zsparse_destroy(int *);

extern double ddot_(int *, double *, int *, double *, int *);
extern void   dmvm_(int *, double *, int *, int *, double *, double *, int *);
extern double dlapy2_(double *, double *);

void dninge_(int *n,         int *nev,   double *sigmar,
              double *sigmai, int *aptr,  int *aind, 
              double *aval,   int *bptr,  int *bind, 
              double *bval,   double *dr, double *di,
              double *z,      int *ldz,   int *info)

/*  This routine computes eigenvalues and eigenvectors of
    a matrix pair  (A,B).

    Arguement list:

    n      (int*)    Dimension of the problem. (INPUT)

    nev    (int*)    Number of eigenvalues requested. (INPUT/OUTPUT)
                     This routine is used to compute NEV eigenvalues
                     nearest to a shift (sigmar, sigmai).
                     On return, it gives the number of converged 
                     eigenvalues.

    sigmar (double*) Real part of the shift. (INPUT)

    sigmai (double*) Imaginar part of the shift. (INPUT)

    aptr   (int*)    dimension n+1. (INPUT)
                     Column pointers for the A matrix.

    aind   (int*)    dimension aptr[*n]-1. (INPUT)
                     Row indices for the A matrix.

    aval  (double*)  dimension aptr[*n]-1. (INPUT)
                     Nonzero values of the A matrix.
                     The sparse matrix A is represented by
                     the above three arrays aptr, aind, aval.

    bptr   (int*)    dimension n+1. (INPUT)
                     Column pointers for the B matrix.

    bind   (int*)    dimension bptr[*n]-1. (INPUT)
                     Row indices for the B matrix.

    bval   (double*) dimension bptr[*n]-1. (INPUT)
                     Nonzero values of the B matrix.
                     The sparse matrix B is represented by
                     the above three arrays bptr, bind, bval.

    dr     (double*) dimension nev+1. (OUTPUT)
                     Real part of the eigenvalue.

    di     (double*) dimension nev+1. (OUTPUT)
                     Imaginar part of the eigenvalue.

    z      (double*) dimension ldz by nev+1. (OUTPUT)
                     Eigenvector matrix.
                     If the j-th eigenvalue is real, the j-th column
                     of z contains the corresponding eigenvector.
                     If the j-th and j+1st eigenvalues form a complex
                     conjuagate pair, then the j-th column of z contains
                     the real part of the eigenvector, and the j+1st column
                     of z contains the imaginary part of the eigenvector.

    ldz     (int*)   The leading dimension of z. (INPUT)

    info     (int*)  Error flag to indicate whether the eigenvalues
                     calculation is successful. (OUTPUT)
                     *info = 0, successful exit
                     *info = 1, Maximum number of iteration is reached
                                before all requested eigenvalues
                                have converged.
*/

{
    int    i, j, ibegin, iend, ncv,  neqns, token, order=2;
    int    lworkl, ldv,  nnza, nnzb, nnzab, ione = 1;
    double tol=1.0e-10, zero = 0.0;
    double *workl, *workd, *resid, *workev, *v, *ax;
    double *dwork;
    int    *iwork;
    doublecomplex *cwork;
    double numr, numi, denr, deni;
    int    *select, first;
    int    ido, ishfts, maxitr, mode, rvec, ierr1, ierr2;
    int    iparam[11], ipntr[14];
    char   *which="LM", *bmat="G", *all="A";
    doublecomplex *cvals, *cx, *crhs;
    double        *rvals;
    int           *abptr, *abind;

    neqns = *n;
    nnza  = aptr[neqns]-1;
    nnzb  = bptr[neqns]-1;
    *info = 0;

    if (*n - *nev < 2) {
       *info = -1000;
       fprintf(stderr, " NEV must be less than N-2!\n");
       goto Error_handle;
    }

    /* set parameters and allocate temp space for ARPACK*/
    ncv = max(*nev+20, 2*(*nev));
    if (ncv > neqns) ncv = neqns;

    /* Count nonzeros in the shifted matrix */
    dwork = (double*)malloc(neqns*sizeof(double));
    if (!dwork) {
       fprintf(stderr, " Fail to allocate dwork\n");
       goto Error_handle;
    }
    iwork = (int*)malloc(neqns*sizeof(int));
    if (!iwork) {
       fprintf(stderr, " Fail to allocate iwork\n");
       goto Error_handle;
    }
    dnzcnt_(n, aptr, aind, aval, bptr, bind, bval, &nnzab, dwork, iwork);

    abptr = (int*)malloc((neqns+1)*sizeof(int));
    if (!abptr) { 
       fprintf(stderr, " Fail to allocate abptr\n");
       goto Error_handle;
    }
    abind = (int*)malloc(nnzab*sizeof(int));
    if (!abind) {
       fprintf(stderr, " Fail to allocate abind\n");
       goto Error_handle;
    }

    /* Subtract shift from the matrix */
    if ( *sigmai == 0.0) {
       /* real shift */
       rvals = (double*)malloc(nnzab*sizeof(double));
       if (!rvals) fprintf(stderr, " Fail to allocate rvals!\n");
       dshftab_(n,     sigmar, aptr,  aind,   aval, bptr, bind, bval,
                abptr, abind,  rvals, &nnzab, dwork);
    }
    else {
       /* complex shift */
       cwork = (doublecomplex*)malloc(neqns*sizeof(doublecomplex));
       if (!cwork) {
          fprintf(stderr, " Fail to allocate cwork!\n");
          goto Error_handle;
       }
         
       cvals = (doublecomplex*)malloc(nnzab*sizeof(doublecomplex));
       if (!cvals) {
          fprintf(stderr, " Fail to allocate cvals!\n");
          goto Error_handle;
       }

       dshftab2_(n,      sigmar, sigmai, aptr,  aind,  aval, 
                 bptr,   bind,   bval,   abptr, abind, cvals,
                 &nnzab, cwork);
    }

    for (j=0;j<=neqns;j++) abptr[j]--;
    for (i=0;i<nnzab;i++)  abind[i]--;

    /* order and factor the shifted matrix */
    token = 0;
    if (*sigmai == 0.0) {
       dsparse_preprocess_(&token, &neqns, abptr, abind, rvals, &order);
       dsparse_factor_(&token);
    }
    else {
       zsparse_preprocess_(&token, &neqns, abptr, abind, cvals, &order);
       zsparse_factor_(&token);
    }

    /* change from 0-based index to 1-based index */
    for (j=0;j<=neqns;j++) abptr[j]++;
    for (i=0;i<nnzab;i++)  abind[i]++;

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
       fprintf(stderr, " Fail to allocate resid\n");
       goto Error_handle;
    }
    workl = (double*) malloc(lworkl*sizeof(double));
    if (!workl) {
       fprintf(stderr, " Fail to allocate workl\n");
       goto Error_handle;
    }
    v     = (double*) malloc(ldv*ncv*sizeof(double));
    if (!v) {
       fprintf(stderr, " Fail to allocate v\n");
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
    ax    = (double*)malloc(neqns*sizeof(double));
    if (!ax) {
       fprintf(stderr, " Fail to allocate ax\n");
       goto Error_handle;
    }
    select= (int*) malloc(ncv*sizeof(int));
    if (!select) {
       fprintf(stderr, " Fail to allocate select\n");
       goto Error_handle;
    }

    if (*sigmai != 0.0) {
       cx    = (doublecomplex*)malloc(neqns*sizeof(doublecomplex));
       crhs  = (doublecomplex*)malloc(neqns*sizeof(doublecomplex));
    }

    /* intialize all work arrays */
    for (i=0;i<neqns;i++) resid[i] = 0.0;
    for (i=0;i<lworkl;i++) workl[i]=0.0;
    for (i=0;i<ldv*ncv;i++) v[i]=0.0;
    for (i=0;i<3*neqns;i++) workd[i]=0.0;
    for (i=0;i<3*ncv;i++) workev[i]=0.0;
    for (i=0;i<ncv;i++)   select[i] = 0;
    for (i=0;i<neqns;i++) ax[i] = 0.0;

    /* ARPACK reverse comm to compute eigenvalues and eigenvectors */
    if (*sigmai == 0.0) {
       while (ido != 99 ) {
          dnaupd_(&ido,    bmat, n,    which,  nev,   &tol,  resid, 
                  &ncv,    v,    &ldv, iparam, ipntr, workd, workl,
                  &lworkl, &ierr1);
          if (ido == -1) {
             dmvm_(n, bval, bind, bptr, &workd[ipntr[0]-1], ax, &ione);
             dsparse_solve_(&token, &workd[ipntr[1]-1], ax);
          }
          else if (ido == 1) {
             dsparse_solve_(&token, &workd[ipntr[1]-1], &workd[ipntr[2]-1]);
          }
          else {
             dmvm_(n,                  bval, bind, bptr, &workd[ipntr[0]-1],
                   &workd[ipntr[1]-1], &ione);
          }
       }
    }
    else {
       while (ido != 99 ) {
          dnaupd_(&ido,    bmat, n,    which,  nev,   &tol,  resid,
                  &ncv,    v,    &ldv, iparam, ipntr, workd, workl,
                  &lworkl, &ierr1);
          if (ido == -1) {
              dmvm_(n,                  bval,               bind,  bptr, 
                    &workd[ipntr[0]-1], &workd[ipntr[1]-1], &ione);
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
              dmvm_(n,                  bval,               bind, bptr, 
                    &workd[ipntr[0]-1], &workd[ipntr[1]-1], &ione);
          } 
       }
    }

    /* ARPACK postprocessing */
    if (ierr1 < 0) {
       fprintf(stderr, " Error with _naupd, ierr = %d\n", ierr1);
       goto Error_handle;
    }
    else {
       rvec = 1;
       dneupd_(&rvec, all,   select,  dr,   di,   z,      ldz, 
              sigmar, sigmai,workev,  bmat, n,    which,  nev,
              &tol,   resid, &ncv,    v,    &ldv, iparam, ipntr, 
              workd,  workl, &lworkl, &ierr2);

       if (ierr2 != 0) {
           fprintf(stderr," Error with _neupd, ierr = %d\n",ierr2);
           goto Error_handle;
       }
       *nev = iparam[4];
    } 

    if (*sigmai != 0) {
       /* Use Rayleigh quotient to recover Ritz values */

       for (i=0;i<neqns;i++) ax[i] = 0.0;
       first = 1;
       for (j = 1; j<=(*nev); j++) {
          if (di(j) == 0.0) {
             dmvm_(n, aval, aind, aptr, &z(1,j), ax, &ione);
             numr = ddot_(n, &z(1,j), &ione, ax, &ione);
             dmvm_(n, bval, bind, bptr, &z(1,j), ax, &ione);
             denr = ddot_(n, &z(1,j), &ione, ax, &ione);
             dr(j) = numr/denr;  
          }
          else if (first) {
             /* compute trans(x) A x */
             dmvm_(n, aval, aind, aptr, &z(1,j), ax, &ione);
             numr = ddot_(n, &z(1,j),   &ione, ax, &ione);
             numi = ddot_(n, &z(1,j+1), &ione, ax, &ione);
             dmvm_(n, aval, aind, aptr, &z(1,j+1), ax, &ione);
             numr = numr  + ddot_(n, &z(1,j+1), &ione, ax, &ione);
             numi = -numi + ddot_(n, &z(1,j), &ione, ax, &ione);

             /* compute trans(x) M x */
             dmvm_(n, bval, bind, bptr, &z(1,j), ax, &ione);
             denr = ddot_(n, &z(1,j),   &ione, ax, &ione);
             deni = ddot_(n, &z(1,j+1), &ione, ax, &ione);
             dmvm_(n, bval, bind, bptr, &z(1,j+1), ax, &ione);
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
  
    free(resid);
    free(workl);
    free(v);
    free(workd); 
    free(workev);
    free(select);
    free(abptr);
    free(abind);
    free(ax);
    free(dwork);
    free(iwork);
    if (*sigmai != 0.0) {
       free(crhs);
       free(cx);
       free(cvals);
       free(cwork);
       zsparse_destroy_(&token);
    }
    else {
       free(rvals);
       dsparse_destroy_(&token);
    }
Error_handle:
    if (ierr1 != 0) *info = ierr1;
    if (ierr1 == 1)
       fprintf(stderr, " Maxiumum number of iteration reached.\n");
}
