#include "Euclid_dh.h"

static void apply_seq_nat_private(Euclid_dh ctx, double *xx, double *yy);
static void apply_seq_nat_private_D(Euclid_dh ctx, double *xx, double *yy);
static void apply_seq_nat_threaded_private(Euclid_dh ctx, double *xx, double *yy);

static void apply_seq_nat_threaded_private_D(Euclid_dh ctx, double *xx, double *yy)
{}
static void apply_seq_perm_threaded_private(Euclid_dh ctx, double *xx, double *yy)
{}
static void apply_seq_perm_threaded_private_D(Euclid_dh ctx, double *xx, double *yy)
{}


/* stop compiler from complaining about unreferenced static
   functions when configured --with-strict-checking.
 */   
void junk(Euclid_dh ctx)
{
  double *x = NULL, *y = NULL;
  apply_seq_nat_threaded_private_D(ctx,x,y);
  apply_seq_perm_threaded_private(ctx,x,y);
  apply_seq_perm_threaded_private_D(ctx,x,y);
}

#undef __FUNC__ 
#define __FUNC__ "not_implemented_private"
void not_implemented_private(Euclid_dh ctx)
{
  START_FUNC_DH
  SET_V_ERROR("not implemented");
  END_FUNC_DH
}


/* this is just a big switch; this was once done otherwise, with
 * v-tables (as in PETSc) --- but that made it really difficult
 * to track control flow, so I removed all V-tables.
 *
 * - D. Hysom, 10/2K
 */
#undef __FUNC__ 
#define __FUNC__ "Euclid_dhApply_private"
void Euclid_dhApply_private(Euclid_dh ctx, double *xx, double *yy)
{
  START_FUNC_DH
  bool single = (ctx->avalF != NULL) ? true : false;

  /* case 1: no preconditioning */
  if (ctx->algo_ilu == NONE_ILU) {
    int i, m = ctx->m;
    for (i=0; i<m; ++i) yy[i] = xx[i];
  } 

  /* shared memory cases */
  else if (np_dh == 1) {

    /* natural ording -> block-jacobi */
    if (ctx->isNaturallyOrdered) {
      if (single) {
        apply_seq_nat_threaded_private(ctx, xx, yy); CHECK_V_ERROR;
      } else {
        /* apply_seq_nat_threaded_private_D(ctx, xx, yy); CHECK_V_ERROR; */
        not_implemented_private(ctx); CHECK_V_ERROR;
      }

    } else {
      if (single) {
        not_implemented_private(ctx); CHECK_V_ERROR;
      } else {
        not_implemented_private(ctx); CHECK_V_ERROR;
      }
    }
  }


  /* MPI cases */
  else {

    /* MPI Block Jacobi cases */
    if (ctx->algo_par == BJILU_PAR) {

      /* matrix has NOT been reordered */
      if (ctx->isNaturallyOrdered) {
        if (single) {
          apply_seq_nat_private(ctx, xx, yy); CHECK_V_ERROR;
        } else {
          apply_seq_nat_private_D(ctx, xx, yy); CHECK_V_ERROR;
        } 

      /* matrix HAS been reordered */
      }  else {
        if (single) {
          not_implemented_private(ctx); CHECK_V_ERROR;
        } else {
          not_implemented_private(ctx); CHECK_V_ERROR;
        }
      }
    }

    /* MPI Parallel ILU cases (unless user has set stupid switches,
     * like a single subdomain, ordering is never natural, so we
     * needn't worry about that).
     */
    else if (ctx->algo_par == PILU_PAR) {
      if (single) {
        not_implemented_private(ctx); CHECK_V_ERROR;
      } else {
        not_implemented_private(ctx); CHECK_V_ERROR;
      }
    } 


    /* error checking: shouldn't ever be here! */
    else {
      sprintf(msgBuf_dh, "unsupported setting for ctx->algo_par = %i", ctx->algo_par);
      SET_V_ERROR(msgBuf_dh);
    }
  }
  END_FUNC_DH
}



/* for single precision and:
 *
 *      (1) natural ordering, 1 subdomain (mpi task), 1 processor;
 *      (2) mpi parallel block jacobi, natural ordering 
 *
 * (unthreaded)
 */
#undef __FUNC__ 
#define __FUNC__ "apply_seq_nat_private"
void apply_seq_nat_private(Euclid_dh ctx, double *xx, double *yy)
{
  START_FUNC_DH
  int m = ctx->m, i;
  int *rp = ctx->rpF, *cval = ctx->cvalF;
  float *aval = ctx->avalF;
  int nz, *vi, *diag = ctx->diagF;
  float *scale = ctx->scale;
  float sum, *v, *work = ctx->work;

  /* if matrix was scaled, must scale the rhs */
  if (scale != NULL) {
    for (i=0; i<m; ++i) { xx[i] *= scale[i]; }
  } 

  work[0] = xx[0];
  for ( i=0; i<m; i++ ) {
    v   = aval + rp[i];
    vi  = cval + rp[i];
    nz  = diag[i] - rp[i];
    sum = xx[i];
    while (nz--) sum -= (*v++ * work[*vi++]);
    work[i] = sum;
  }

  /* backward solve the upper triangular */
  for ( i=m-1; i>=0; i-- ){
    v   = aval + diag[i] + 1;
    vi  = cval + diag[i] + 1;
    nz  = rp[i+1] - diag[i] - 1;
    sum = work[i];
    while (nz--) sum -= (*v++ * work[*vi++]);
    yy[i] = work[i] = sum*aval[diag[i]];
  }

  /* put rhs back the way it was */
  if (scale != NULL) {
    for (i=0; i<m; ++i) { xx[i] /= scale[i]; }
  } 

  END_FUNC_DH
}


/* for double precision and:
 *
 *      (1) natural ordering, 1 subdomain (mpi task), 1 processor;
 *      (2) mpi parallel block jacobi, natural ordering 
 *
 * (unthreaded)
 */
#undef __FUNC__ 
#define __FUNC__ "apply_seq_nat_private_D"
void apply_seq_nat_private_D(Euclid_dh ctx, double *xx, double *yy)
{
  START_FUNC_DH

  int m = ctx->m, i;
  int *rp = ctx->rpF, *cval = ctx->cvalF;
  double *aval = ctx->avalFD;
  int nz, *vi, *diag = ctx->diagF;
  double *scale = ctx->scaleD;
  double sum, *v, *work = ctx->workD;

  /* if matrix was scaled, must scale the rhs */
  if (scale != NULL) {
    for (i=0; i<m; ++i) { xx[i] *= scale[i]; }
  } 

    work[0] = xx[0];
    for ( i=0; i<m; i++ ) {
      v   = aval + rp[i];
      vi  = cval + rp[i];
      nz  = diag[i] - rp[i];
      sum = xx[i];
      while (nz--) sum -= (*v++ * work[*vi++]);
      work[i] = sum;
    }

    /* backward solve the upper triangular */
    for ( i=m-1; i>=0; i-- ){
      v   = aval + diag[i] + 1;
      vi  = cval + diag[i] + 1;
      nz  = rp[i+1] - diag[i] - 1;
      sum = work[i];
      while (nz--) sum -= (*v++ * work[*vi++]);
      yy[i] = work[i] = sum*aval[diag[i]];
    }

    /* put rhs back the way it was */
    if (scale != NULL) {
      for (i=0; i<m; ++i) { xx[i] /= scale[i]; }
    } 

  END_FUNC_DH
}



/* for single precision, single mpi task, threaded, block-jacobi
 */
#undef __FUNC__
#define __FUNC__ "apply_seq_nat_threaded_private"
void apply_seq_nat_threaded_private(Euclid_dh ctx, double *xx, double *yy)
{
  START_FUNC_DH
  int       h, n;
  int       *rp, *cval, *diag;
  float     *aval, *scale;
  int       pn = ctx->blockCount;
  PartNode  *part = ctx->block;
  float     *v, *work;
  double    sum;
  int       i, *vi, nz;

    n = ctx->n;
    rp = ctx->rpF;
    cval = ctx->cvalF;
    aval = ctx->avalF;
    diag = ctx->diagF;
    scale = ctx->scale;
    work = ctx->work;

  #pragma omp parallel 
  {
     /* if matrix was scaled, must scale the rhs */
     if (scale != NULL) {
       #pragma omp for schedule(static)
       for (i=0; i<n; ++i) { xx[i] *= scale[i]; }
     } 

   #pragma omp for schedule(static) private(v,vi,nz,sum,i)
    for (h=0; h<pn; ++h) {    /* fwd solve diagonal block h */
      int from = part[h].beg_row;
      int to = part[h].end_row;

      work[from] = xx[from];
      for ( i=from+1; i<to; i++ ) {
        v   = aval + rp[i];
        vi  = cval + rp[i];
        nz  = diag[i] - rp[i];
        sum = xx[i];
        while (nz--) sum -= (*v++ * work[*vi++]);
        work[i] = sum;
      }

      /* backward solve the upper triangular */
      for ( i=to-1; i>=from; i-- ){
        v   = aval + diag[i] + 1;
        vi  = cval + diag[i] + 1;
        nz  = rp[i+1] - diag[i] - 1;
        sum = work[i];
        while (nz--) sum -= (*v++ * work[*vi++]);
        yy[i] = work[i] = sum*aval[diag[i]];
      }
    }

     /* put rhs back the way it was */
     if (scale != NULL) {
       #pragma omp for schedule(static) 
       for (i=0; i<n; ++i) { xx[i] /= scale[i]; }
     } 

    } /* #pragma omp parallel */

  END_FUNC_DH
}

#if 0

#undef __FUNC__
#define __FUNC__ "Euclid_dhApplyPermuted_private"
void Euclid_dhApplyPermuted_private(Euclid_dh ctx, double *xx, double *yy)
{
  START_FUNC_DH
  int       h, n;
  int       *rp, *cval, *diag;
  REAL_DH   *aval, *scale;
  int       *n2o_row, *n2o_col, pn = ctx->blockCount;
  PartNode  *part = ctx->block;
  REAL_DH *v, *work;
  double sum;
  int i, *vi, nz;

  n2o_row = ctx->n2o_row;
  n2o_col = ctx->n2o_col;
  n = ctx->n;
  rp = ctx->rpF;
  cval = ctx->cvalF;
  aval = ctx->avalF;
  diag = ctx->diagF;
  scale = ctx->scale;
  work = ctx->work;

  #pragma omp parallel 
  {
    /* if matrix was scaled, must scale the rhs */
    if (scale != NULL) {
      #pragma omp for schedule(static)
        for (i=0; i<n; ++i) { xx[i] *= scale[i]; }
    } 

   #pragma omp for schedule(static) private(v,vi,nz,sum,i)
    for (h=0; h<pn; ++h) {    /* fwd solve diagonal block h */
      int from = part[h].beg_row;
      int to = part[h].end_row;

      /* forward solve the lower triangle */
      work[from] = xx[n2o_row[from]];
      for ( i=from+1; i<to; i++ ) {
        v   = aval + rp[i];
        vi  = cval + rp[i];
        nz  = diag[i] - rp[i];
        sum = xx[n2o_row[i]];
        while (nz--) sum -= (*v++ * work[*vi++]);
        work[i] = sum;
      }

      /* backward solve the upper triangular */
      for ( i=to-1; i>=from; i-- ){
        v   = aval + diag[i] + 1;
        vi  = cval + diag[i] + 1;
        nz  = rp[i+1] - diag[i] - 1;
        sum = work[i];
        while (nz--) sum -= (*v++ * work[*vi++]);
        yy[n2o_col[i]] = work[i] = sum*aval[diag[i]];
      }
    }

    /* put rhs back the way it was */
    if (scale != NULL) {
      #pragma omp for schedule(static) 
      for (i=0; i<n; ++i) { xx[i] /= scale[i]; }
    } 

  } /* #pragma omp parallel */

  END_FUNC_DH
}


/* there is room for additional parallelism here, when
   factoring boundary rows.  For possible future development . . .
   (as written, factoring of boundary rows is entirely sequential)
*/
#undef __FUNC__
#define __FUNC__ "Euclid_dhApplyPILU_private"
void Euclid_dhApplyPILU_private(Euclid_dh ctx, double *xx, double *yy)
{
  START_FUNC_DH
  int       h, n;
  int       *rp, *cval, *diag;
  REAL_DH   *aval, *scale;
  int       *n2o_row, *n2o_col, pn = ctx->blockCount;
  REAL_DH *v, *work;
  double sum;
  int i, *vi, nz, from, to;
  PartNode  *part = ctx->block;

  n2o_row = ctx->n2o_row;
  n2o_col = ctx->n2o_col;
  n = ctx->n;
  rp = ctx->rpF;
  cval = ctx->cvalF;
  aval = ctx->avalF;
  diag = ctx->diagF;
  scale = ctx->scale;
  work = ctx->work;

  /* if matrix was scaled, must scale the rhs */
  if (scale != NULL) {
    #pragma omp for schedule(static)
    for (i=0; i<n; ++i) { xx[i] *= scale[i]; }
  } 

    /* forward solve lower triangle interiors (parallel) */
   #pragma omp for schedule(static) private(from,to,v,vi,nz,sum,i)
    for (h=0; h<pn; ++h) { 
      from = part[h].beg_row;
      to = part[h].first_bdry;

      work[from] = xx[n2o_row[from]];
      for ( i=from+1; i<to; i++ ) {
        v   = aval + rp[i];
        vi  = cval + rp[i];
        nz  = diag[i] - rp[i];
        sum = xx[n2o_row[i]];
        while (nz--) sum -= (*v++ * work[*vi++]);
        work[i] = sum;
      }
    }

    /* forward solve lower triangle boundaries (sequential) */
    for (h=0; h<pn; ++h) { 
      from = part[h].first_bdry;
      to   = part[h].end_row;
      for ( i=from; i<to; i++ ) {
        v   = aval + rp[i];
        vi  = cval + rp[i];
        nz  = diag[i] - rp[i];
        sum = xx[n2o_row[i]];
        while (nz--) sum -= (*v++ * work[*vi++]);
        work[i] = sum;
      }
    }

    /* backward solve upper triangular boundaries (sequential) */
    for (h=pn-1; h>=0; --h) { 
      from = part[h].end_row-1;
      to   = part[h].first_bdry;
      for ( i=from; i>=to; i-- ){
        v   = aval + diag[i] + 1;
        vi  = cval + diag[i] + 1;
        nz  = rp[i+1] - diag[i] - 1;
        sum = work[i];
        while (nz--) sum -= (*v++ * work[*vi++]);
        yy[n2o_col[i]] = work[i] = sum*aval[diag[i]];
      }
    }

    /* backward solve upper triangular interiors (parallel) */
   #pragma omp for schedule(static) private(from,to,v,vi,nz,sum,i)
    for (h=pn-1; h>=0; --h) { 
      from = part[h].first_bdry - 1;
      to   = part[h].beg_row;
      for ( i=from; i>=to; i-- ){
        v   = aval + diag[i] + 1;
        vi  = cval + diag[i] + 1;
        nz  = rp[i+1] - diag[i] - 1;
        sum = work[i];
        while (nz--) sum -= (*v++ * work[*vi++]);
        yy[n2o_col[i]] = work[i] = sum*aval[diag[i]];
      }
    }

  /* put rhs back the way it was */
  if (scale != NULL) {
    #pragma omp for schedule(static) 
    for (i=0; i<n; ++i) { xx[i] /= scale[i]; }
  } 

  END_FUNC_DH
}

#endif /* #if 0 */
