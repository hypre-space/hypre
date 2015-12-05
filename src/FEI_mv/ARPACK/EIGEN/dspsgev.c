#include  <stdio.h>
#include  <math.h>

#define z(i,j) z[(*ldz)*( (j) -1 )+ (i) - 1]
#define d(i)   d[(i)-1]

extern int ldlt_preprocess__(int *, int *, int *, int *, int *, int *);
extern int ldlt_factor__(int *, int *, int *, double *);
extern int ldlt_solve__(int *, double *, double *);
extern int ldlt_free__(int *);

#ifdef TIMING
   #include <sys/types.h>
   #include <sys/times.h>
   #include <time.h>
   #include <sys/time.h>

   #ifndef CLK_TCK
      #define CLK_TCK 60
   #endif

   double dspsgev()
   {
      struct tms use;
      double tmp;
      times(&use);
      tmp = use.tms_utime;
      /* tmp += use.tms_stime; */
      return (double)(tmp) / CLK_TCK;
   }

   double tprep, tfact, tsolv=0.0, tarpk, tmv=0.0, t0, t1, t00, t11;
  
#endif

void dspsgev_(int *n,    int *nev,  double *sigma,
              int *aptr, int *aind, double *aval,
              int *bptr, int *bind, double *bval, 
              double *d, double *z, int *ldz)

/*  Arguement list:

    n       (int*)    Dimension of the problem.

    nev     (int*)    Number of eigenvalues requested.
                      This routine is used to compute NEV eigenvalues
                      nearest to a shift sigma.

    sigma   (double*) Target shift.

    aptr    (int*)    dimension n+1.
                      Column pointers for the A matrix.

    aind    (int*)    dimension aptr[*n]-1.
                      Row indices for the A matrix.

    aval    (double*) dimension aptr[*n]-1.
                      Nonzero values of the A matrix.
                      The A matrix is represented by
                      the above three arrays aptr, aind, aval.

    bptr    (int*)    dimension n+1.
                      Column pointers for the B matrix.

    bind    (int*)    dimension bptr[*n]-1.
                      Row indices for the B matrix.

    bval    (double*) dimension bptr[*n]-1.
                      Nonzero values of the B matrix.
                      The B matrix is represented by
                      the above three arrays bptr, bind, bval.

    d       (double*) dimension nev.
                      Converged eigenvalues.

    z       (double*) dimension ldz by nev.
                      Eigenvector matrix.

    ldz      (int*)   The leading dimension of z.
*/

{
    int    i, j, ibegin, iend, ncv, neqns, token, Lnnz, order=2;
    int    lworkl, ldv,  nnz, ione = 1;
    double tol=0.0, zero = 0.0;
    double *workl, *workd, *resid, *v, *ax, *dwork;
    double *abval;
    int    *abptr, *abind, *iwork;
    int    *select, first;
    int    ido, info, ishfts, maxitr, mode, rvec, ierr, nnzab;
    int    iparam[11], ipntr[11];
    char   *which="LM", bmat[1]="G", *all="A";

    neqns = *n;

    /* perform A-sigma*M */

    dwork = (double*)malloc(neqns*sizeof(double));
    if (!dwork) fprintf(stderr, " Fail to allocate dwork\n");
    iwork = (int*)malloc(neqns*sizeof(int));
    if (!iwork) fprintf(stderr, " Fail to allocate iwork\n");

    dnzcnt_(n, aptr, aind, aval, bptr, bind, bval, &nnzab, dwork, iwork);

    if ( *sigma != 0.0) {
       abptr = (int*)malloc((neqns+1)*sizeof(int));
       if (!abptr) fprintf(stderr, " Fail to allocate abptr\n");
       abind = (int*)malloc(nnzab*sizeof(int));
       if (!abind) fprintf(stderr, " Fail to allocate abind\n");
       abval = (double*)malloc(nnzab*sizeof(double));
       if (!abval) fprintf(stderr, " Fail to allocate abval!\n");
       dshftab_(n,     sigma, aptr,  aind,   aval, bptr, bind, bval,
                abptr, abind, abval, &nnzab, dwork);
    }
    else {
       abptr = aptr;
       abind = aind;
       abval = aval;
    }

    /* order and factor the shifted matrix */
    token = 0;
#ifdef TIMING
    t0    = dspsgev();
#endif
    ldlt_preprocess__(&token, &neqns, abptr, abind, &Lnnz, &order);
#ifdef TIMING
    t1    = dspsgev();
    tprep = t1 - t0;
    t0    = dspsgev(); 
#endif
    ldlt_fact__(&token, abptr, abind, abval);
#ifdef TIMING
    t1    = dspsgev();
    tfact = t1 - t0;
#endif

    /* set parameters and allocate temp space for ARPACK*/
    ncv = *nev + 20; /* use a 20-th degree polynomial for implicit restart */
    lworkl = ncv*(ncv+8);
    ido    = 0;
    info   = 0;
    ishfts = 1;
    maxitr = 300;
    mode   = 3;
    ldv    = neqns;

    iparam[0] = ishfts;
    iparam[2] = maxitr;
    iparam[6] = mode;

    resid = (double*) malloc(neqns*sizeof(double));
    if (!resid) fprintf(stderr," Fail to allocate resid\n");
    workl = (double*) malloc(lworkl*sizeof(double));
    if (!workl) fprintf(stderr," Fail to allocate workl\n");
    v     = (double*) malloc(ldv*ncv*sizeof(double));
    if (!v) fprintf(stderr," Fail to allocate v\n");
    workd = (double*) malloc(neqns*3*sizeof(double));
    if (!workd) fprintf(stderr, " Fail to allocate workd\n");
    select= (int*) malloc(ncv*sizeof(int));
    if (!select) fprintf(stderr, " Fail to allocate select\n");
    ax= (double*) malloc(neqns*sizeof(double));
    if (!ax) fprintf(stderr, " Fail to allocate ax\n");

    /* intialize all work arrays */
    for (i=0;i<neqns;i++)   resid[i] = 0.0;
    for (i=0;i<lworkl;i++)  workl[i]=0.0;
    for (i=0;i<ldv*ncv;i++) v[i]=0.0;
    for (i=0;i<3*neqns;i++) workd[i]=0.0;
    for (i=0;i<ncv;i++)     select[i] = 0;

#ifdef TIMING
    t0 = dspsgev();
#endif
    /* ARPACK reverse comm to compute eigenvalues and eigenvectors */
    while (ido != 99 ) {
       dsaupd_(&ido,    bmat, n,    which,  nev,   &tol,  resid, 
               &ncv,    v,    &ldv, iparam, ipntr, workd, workl,
               &lworkl, &info);
       if (ido == -1 ) {
#ifdef TIMING
   t00 = dspsgev();
#endif 
          dsymmv_(n, bval, bptr, bind, &workd[ipntr[0]-1], ax);
#ifdef TIMING
   t11 = dspsgev();
   tmv = tmv + (t11 - t00);
#endif
#ifdef TIMING
    t00 = dspsgev();
#endif
          ldlt_solve__(&token, &workd[ipntr[1]-1], ax);
#ifdef TIMING
    t11   = dspsgev();
    tsolv = tsolv + (t11-t00); 
#endif

       }
       else if (ido == 1) {
#ifdef TIMING
    t00 = dspsgev();
#endif
          ldlt_solve__(&token, &workd[ipntr[1]-1], &workd[ipntr[2]-1]);
#ifdef TIMING
    t11   = dspsgev();
    tsolv = tsolv + (t11-t00);
#endif

       }
       else if (ido == 2) {
#ifdef TIMING
   t00 = dspsgev();
#endif 
          dsymmv_(n, bval, bptr, bind, &workd[ipntr[0]-1], &workd[ipntr[1]-1]);
#ifdef TIMING
   t11 = dspsgev();
   tmv = tmv + (t11 - t00);
#endif 
       }
    }

    /* ARPACK postprocessing */
    if (info < 0) {
       fprintf(stderr, " Error with _naupd, info = %d\n", info);
    }
    else {
       rvec = 1;
       *nev = iparam[4];
       dseupd_(&rvec, all,   select,  d,     z,    ldz, 
               sigma, bmat,  n,       which, nev,  &tol,
               resid, &ncv,  v,       &ldv,  iparam, ipntr, 
               workd, workl, &lworkl, &ierr);

       if (ierr != 0) fprintf(stderr," Error with _neupd, ierr = %d\n",ierr);
    } 
#ifdef TIMING
    t1    = dspsgev();
    tarpk = t1 - t0 - tsolv;

    printf(" LDLT preprocess    time = %10.2e\n", tprep);
    printf(" LDLT factorization time = %10.2e\n", tfact);
    printf(" LDLT solve         time = %10.2e\n", tsolv);
    printf(" SPARSE MATVEC      time = %10.2e\n", tmv);
    printf(" ARPACK Internal    time = %10.2e\n", tarpk);
#endif


    free(resid);
    free(workl);
    free(v);
    free(workd); 
    free(select);
    free(ax);
    if (*sigma != 0.0) {
       free(abptr);
       free(abind);
       free(abval);
    }
    free(dwork);
    free(iwork);
    ldlt_free__(&token);
}
