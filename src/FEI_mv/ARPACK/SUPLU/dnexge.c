#include <stdio.h>
#include "dcomplex.h"
#include  <math.h>

#define z(i,j) z[(*ldz)*( (j) -1 )+ (i) - 1]

#define max(a,b) ((a)> (b)? (a) : (b))

extern int dsparse_preprocess_(int *, int *, int *, int *, double *, int *);
extern int dsparse_factor_(int *);
extern int dsparse_solve_(int *, double *, double *);
extern int dsparse_destroy(int *);

extern void dmvm_(int *, double *, int *, int *, double *, double *, int *);

void dnexge_(int *n,     int *nev,   char *which,
             int *aptr,  int *aind,  double *aval,   
             int *bptr,  int *bind,  double *bval,
             double *dr, double *di, double *z,
             int *ldz,   int *info)

/*  This routine computes extreme eigenvalues and eigenvectors of
    a matrix pair  (A,B).

    Arguement list:

    n      (int*)    Dimension of the problem. (INPUT)

    nev    (int*)    Number of eigenvalues requested. (INPUT/OUTPUT)
                     This routine is used to compute NEV extreme
                     eigenvalues
                     On return, it gives the number of converged 
                     eigenvalues.

    which   (char*)  Specify which part of the spectrum is of interest.(INPUT)
                     which can be of the following type:
                     "LM" --- eigenvalues with the largest magnitude
                     "LR" --- eigenvalues with the largest real part.
                     "SR" --- eigenvalues with the smallest real part.
                     "LI" --- eigenvalues with the largest imag part.
                     "SI" --- eigenvalues with the largest imag part.
                     Note:
                     Eigenvalues with the smallest magnitude will 
                     be treated as interior eigenvalues.  One should
                     use dninge() with zero shift to find these eigenvalues.

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

    bval  (double*)  dimension bptr[*n]-1. (INPUT)
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

    info    (int*)   Error flag to indicate whether the eigenvalues 
                     calculation is successful.
                     *info = 0, successful exit.
                     *info = 1, maximum number of iteration reached before the
                                residual is below the default tolarance.

*/

{
    int    i, j, ibegin, iend, ncv,  neqns, token, order=2;
    int    lworkl, ldv,  nnzb, ione = 1;
    double tol=1.0e-10, zero = 0.0;
    double sigmar=0.0, sigmai=0.0;
    double *workl, *workd, *resid, *workev, *v, *ax;
    int    *select;
    int    ido, ishfts, maxitr, mode, rvec, ierr1, ierr2;
    int    iparam[11], ipntr[14];
    char   *bmat="I", *all="A";

    neqns = *n;
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

    /* change from 1-based index to 0-based index */
    for (j=0;j<=neqns;j++) bptr[j]--;
    for (i=0;i<nnzb;i++)  bind[i]--;

    /* order and factor the B matrix */
    token = 0;
    dsparse_preprocess_(&token, &neqns, bptr, bind, bval, &order);
    dsparse_factor_(&token);

    /* change from 0-based index to 1-based index */
    for (j=0;j<=neqns;j++) bptr[j]++;
    for (i=0;i<nnzb;i++)  bind[i]++;

    lworkl = 3*ncv*ncv+6*ncv;
    ido    = 0;
    ierr1  = 0;
    ishfts = 1;
    maxitr = 300;
    mode   = 1;
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
       fprintf(stderr," Fail to allocate workd\n");
       goto Error_handle;
    }

    workev= (double*) malloc(ncv*3*sizeof(double));
    if (!workev) {
       fprintf(stderr," Fail to allocate workev\n");
       goto Error_handle;
    }

    ax    = (double*)malloc(neqns*sizeof(double));
    if (!ax) {
       fprintf(stderr," Fail to allocate ax\n");
       goto Error_handle;
    }

    select= (int*) malloc(ncv*sizeof(int));
    if (!select) {
       fprintf(stderr," Fail to allocate select\n");
       goto Error_handle;
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
    while (ido != 99 ) {
       dnaupd_(&ido,    bmat, n,    which,  nev,   &tol,  resid, 
               &ncv,    v,    &ldv, iparam, ipntr, workd, workl,
               &lworkl, &ierr1);
       if (ido == -1 || ido == 1) {
          dmvm_(n, aval, aind, aptr, &workd[ipntr[0]-1], ax, &ione);
          dsparse_solve_(&token, &workd[ipntr[1]-1], ax);
       }
    }

    /* ARPACK postprocessing */
    if (ierr1 < 0) {
       fprintf(stderr, " Error with _naupd, info = %d\n", info);
       goto Error_handle;
    }
    else {
       rvec = 1;
       dneupd_(&rvec,  all,     select,  dr,   di,   z,      ldz, 
              &sigmar, &sigmai, workev,  bmat, n,    which,  nev,
              &tol,    resid,   &ncv,    v,    &ldv, iparam, ipntr, 
              workd,   workl,   &lworkl, &ierr2);

       if (ierr2 != 0) {
          fprintf(stderr," Error with _neupd, ierr = %d\n",ierr2);
          goto Error_handle;
       }
       *nev = iparam[4];
    } 

    free(resid);
    free(workl);
    free(v);
    free(workd); 
    free(workev);
    free(select);
    free(ax);
    dsparse_destroy_(&token);

Error_handle:
    if (ierr1 != 0) *info = ierr1;
    if (ierr1 == 1) 
       fprintf(stderr, " Maxiumum number of iteration reached.\n");
}
