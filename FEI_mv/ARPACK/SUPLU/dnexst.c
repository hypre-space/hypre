#include <stdio.h>
#include "dcomplex.h"

#define z(i,j) z[(*ldz)*( (j) -1 )+ (i) - 1]
#define v(i,j) v[(*ldz)*( (j) -1 )+ (i) - 1]
#define dr(i) dr[(i)-1]
#define di(i) di[(i)-1]

#define max(a,b) ((a)> (b)? (a) : (b))

extern void dmvm_(int *, double *, int *, int *, double *, double *, int *);

void dnexst_(int *n,      int *nev,    char *which,
             int *colptr, int *rowind, double *nzvals, 
             double *dr,  double *di,  double *z,
             int *ldz,    int *info)

/*  Arguement list:

    n       (int*)    Dimension of the problem.  (INPUT)

    nev     (int*)    Number of eigenvalues requested.  (INPUT)
                      This routine is used to compute NEV extreme eigenvalues.

    which   (char*)   Specify which part of the spectrum is of interest.(INPUT)
                      which can be of the following type:
                      "LM" --- eigenvalues with the largest magnitude
                      "LR" --- eigenvalues with the largest real part.
                      "SR" --- eigenvalues with the smallest real part.
                      "LI" --- eigenvalues with the largest imag part.
                      "SI" --- eigenvalues with the largest imag part.
                      Note:
                      Eigenvalues with the smallest magnitude will 
                      be treated as interior eigenvalues.  One should
                      use dninst() with zero shift to find these eigenvalues.

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
                      calculation is successful.
                      *info = 0, successful exit
*/

{
    int    i, j, ibegin, iend, ncv, neqns, token, order=2;
    int    lworkl, ldv,  nnz, ione = 1;
    double tol=1.0e-10, zero = 0.0;
    double *workl, *workd, *resid, *workev, *v, *ax;
    double sigmar=0.0, sigmai=0.0;
    int    *select, first;
    int    ido, ierr1, ierr2, ishfts, maxitr, mode, rvec;
    int    iparam[11], ipntr[14];
    char   *bmat="I", *all="A";

    neqns = *n;
    nnz   = colptr[neqns]-1;

    *info = 0;

    if (*n - *nev < 2) {
       *info = -1000;
       fprintf(stderr, " NEV must be less than N-2!\n");
       goto Error_handle; 
    }

    /* set parameters and allocate temp space for ARPACK*/
    ncv = max(*nev+20, 2*(*nev));
    if (ncv > neqns) ncv = neqns;
   
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

    /* intialize all work arrays */
    for (i=0;i<neqns;i++) resid[i] = 0.0;
    for (i=0;i<lworkl;i++) workl[i]=0.0;
    for (i=0;i<ldv*ncv;i++) v[i]=0.0;
    for (i=0;i<3*neqns;i++) workd[i]=0.0;
    for (i=0;i<3*ncv;i++) workev[i]=0.0;
    for (i=0;i<ncv;i++) select[i] = 0;

    /* ARPACK reverse comm to compute eigenvalues and eigenvectors */
    while (ido != 99 ) {
       dnaupd_(&ido,    bmat, n,    which,  nev,   &tol,  resid, 
               &ncv,    v,    &ldv, iparam, ipntr, workd, workl,
               &lworkl, &ierr1);
       if (ido == -1 || ido == 1) {
          dmvm_(n, nzvals, rowind, colptr, &workd[ipntr[0]-1],
                &workd[ipntr[1]-1], &ione);
       }
    }

    /* ARPACK postprocessing */
    if (ierr1 < 0) {
       fprintf(stderr, " Error with _naupd, ierr = %d\n", ierr1);
       goto Error_handle;
    }
    else {
       rvec = 1;
       dneupd_(&rvec,  all,   select,  dr,   di,   z,      ldz, 
              &sigmar, &sigmai,workev, bmat, n,    which,  nev,
              &tol,    resid, &ncv,    v,    &ldv, iparam, ipntr, 
              workd,   workl, &lworkl, &ierr2);

       *nev = iparam[4];

       if (ierr2 != 0) {
          fprintf(stderr," Error with _neupd, ierr = %d\n",ierr2);
          goto Error_handle;
       }
    } 

    free(resid);
    free(workl);
    free(v);
    free(workd); 
    free(workev);
    free(select);

Error_handle:
    if (ierr1 != 0) *info = ierr1;
    if (ierr1 == 1) 
       fprintf(stderr, " Maxiumum number of iteration reached.\n");
}
