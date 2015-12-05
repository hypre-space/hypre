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

   double dspssev_timing()
   {
      struct tms use;
      double tmp;
      times(&use);
      tmp = use.tms_utime;
      /* tmp += use.tms_stime; */
      return (double)(tmp) / CLK_TCK;
   }

   double tprep, tfact, tsolv, tarpk, t0, t1, t00, t11;
  
#endif


void dspssev_(int *n,      int *nev,    double *sigma,
              int *colptr, int *rowind, double *nzvals, 
              double *d,   double *z,   int *ldz)

/*  Arguement list:

    n       (int*)    Dimension of the problem.

    nev     (int*)    Number of eigenvalues requested.
                      This routine is used to compute NEV eigenvalues
                      nearest to a shift (sigmar, sigmai).

    sigma   (double*) Target shift.

    colptr  (int*)    dimension n+1.
                      Column pointers for the sparse matrix.

    rowind  (int*)    dimension colptr[*n]-1.
                      Row indices for the sparse matrix.

    nzvals  (double*) dimension colptr[*n]-1.
                      Nonzero values of the sparse matrix.
                      The sparse matrix is represented by
                      the above three arrays colptr, rowind, nzvals.

    d       (double*) dimension nev.
                      Converged eigenvalues.

    z       (double*) dimension ldz by nev.
                      Eigenvector matrix.

    ldz      (int*)   The leading dimension of z.
*/

{
    int    i, j, ibegin, iend, ncv, neqns, token, Lnnz, order=0;
    int    lworkl, ldv,  nnz, ione = 1;
    double tol=0.0, zero = 0.0;
    double *workl, *workd, *resid, *v, *ax;
    int    *select, first;
    int    ido, info, ishfts, maxitr, mode, rvec, ierr;
    int    iparam[11], ipntr[11];
    char   *which="SM", bmat[1]="I", *all="A";

    neqns = *n;

    /* Subtract shift from the matrix */
    for (j = 0; j<neqns; j++) 
       nzvals[colptr[j]-1] = nzvals[colptr[j]-1] - *sigma; 
   
    /* order and factor the shifted matrix */
    token = 0;

#ifdef TIMING
    t0    = dspssev_timing();
    tsolv = 0.0;
#endif
/*
    ldlt_preprocess__(&token, &neqns, colptr, rowind, &Lnnz, &order);
*/
#ifdef TIMING
    t1    = dspssev_timing();
    tprep = t1 - t0;
    t0    = dspssev_timing(); 
#endif
/*
    ldlt_fact__(&token, colptr, rowind, nzvals);
*/
#ifdef TIMING
    t1    = dspssev_timing();
    tfact = t1 - t0;
#endif

    /* add the shift back if shift is real */
    for (j = 0; j<neqns; j++) 
       nzvals[colptr[j]-1] = nzvals[colptr[j]-1] + *sigma; 

    /* set parameters and allocate temp space for ARPACK*/
    ncv = *nev + 20; /* use a 20-th degree polynomial for implicit restart */
    lworkl = ncv*(ncv+8);
    ido    = 0;
    info   = 0;
    ishfts = 1;
    maxitr = 500;
    mode   = 1;
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

    /* intialize all work arrays */
    for (i=0;i<neqns;i++)   resid[i] = 0.0;
    for (i=0;i<lworkl;i++)  workl[i]=0.0;
    for (i=0;i<ldv*ncv;i++) v[i]=0.0;
    for (i=0;i<3*neqns;i++) workd[i]=0.0;
    for (i=0;i<ncv;i++)     select[i] = 0;

#ifdef TIMING
    t0 = dspssev_timing();
#endif
    /* ARPACK reverse comm to compute eigenvalues and eigenvectors */
    while (ido != 99 ) {
       dsaupd_(&ido,    bmat, n,    which,  nev,   &tol,  resid, 
               &ncv,    v,    &ldv, iparam, ipntr, workd, workl,
               &lworkl, &info);
       if (ido == -1 || ido == 1) {
#ifdef TIMING
    t00   = dspssev_timing();
#endif
/*
          ldlt_solve__(&token, &workd[ipntr[1]-1],&workd[ipntr[0]-1]);
*/
        for ( i = 0; i < neqns; i++ ) 
        {
           workd[ipntr[1]-1+i] = 0.0;
           for ( j = colptr[i]-1; j < colptr[i+1]-1; j++ ) 
              workd[ipntr[1]-1+i] += nzvals[j] * workd[ipntr[0]-2+rowind[j]];
        }      
#ifdef TIMING
    t11   = dspssev_timing();
    tsolv = tsolv + (t11 - t00);
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
    t1    = dspssev_timing();
    tarpk = t1 - t0 - tsolv;

    printf(" LDLT preprocess    time = %10.2e\n", tprep);
    printf(" LDLT factorization time = %10.2e\n", tfact);
    printf(" LDLT solve         time = %10.2e\n", tsolv);
    printf(" ARPACK Internal    time = %10.2e\n", tarpk);
#endif

    free(resid);
    free(workl);
    free(v);
    free(workd); 
    free(select);
    ldlt_free__(&token);
}
