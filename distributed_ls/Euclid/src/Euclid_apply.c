#include "Euclid_dh.h"
 
static void apply_blockJacobi_private(Euclid_dh ctx, double *xx, double *yy);
static void apply_pilu_seq_private(Euclid_dh ctx, double *xx, double *yy);

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
  /* default; for everything except PILU */
  ctx->from = 0;
  ctx->to = ctx->m;

  /* case 1: no preconditioning */
  if (ctx->algo_ilu == NONE_ILU) {
    int i, m = ctx->m;
    for (i=0; i<m; ++i) yy[i] = xx[i];
  } 

  /* shared memory cases */
  else if (np_dh == 1) {

    /* natural ording -> block-jacobi */
    if (ctx->algo_par == BJILU_PAR) {
      if (ctx->isNaturallyOrdered) {
        apply_blockJacobi_private(ctx, xx, yy); CHECK_V_ERROR;
      } else {
        not_implemented_private(ctx); CHECK_V_ERROR;
      }
    } else if (ctx->algo_par == PILU_PAR) {
      apply_pilu_seq_private(ctx, xx, yy); CHECK_V_ERROR;
    }
  }


  /* MPI cases */
  else {

    /* MPI Block Jacobi cases */
    if (ctx->algo_par == BJILU_PAR) {
      if (ctx->isNaturallyOrdered) {
        apply_blockJacobi_private(ctx, xx, yy); CHECK_V_ERROR;
      }  else {
        not_implemented_private(ctx); CHECK_V_ERROR;
      }
    }

    /* MPI Parallel ILU cases (unless user has set stupid switches,
     * like a single subdomain, ordering is never natural, so we
     * needn't worry about that).
     */
    else if (ctx->algo_par == PILU_PAR) {
      not_implemented_private(ctx); CHECK_V_ERROR;
    } 


    /* error checking: shouldn't ever be here! */
    else {
      sprintf(msgBuf_dh, "unsupported setting for ctx->algo_par = %i", ctx->algo_par);
      SET_V_ERROR(msgBuf_dh);
    }
  }
  END_FUNC_DH
}



/*  For blockJacobi, single or multiple MPI tasks,
 *  one or more diagonal blocks per MPI task
 *
 */
#undef __FUNC__
#define __FUNC__ "apply_blockJacobi_private"
void apply_blockJacobi_private(Euclid_dh ctx, double *xx, double *yy)
{
  START_FUNC_DH
  int       h, i, m, *vi, nz;
  int       *rp, *cval, *diag;
  float     *avalF, *scaleF, *workF, *vF, sumF;
  double    *avalD, *scaleD, *workD, *vD, sumD;
  int       pn = ctx->blockCount;
  PartNode  *part;
  bool      isSingle = ctx->isSinglePrecision;

  m = ctx->m;
  rp = ctx->rpF;
  cval = ctx->cvalF;
  avalF = ctx->avalF;
  avalD = ctx->avalD;
  diag = ctx->diagF;
  scaleF = ctx->scaleF;
  scaleD = ctx->scaleD;
  workF = ctx->workF;
  workD = ctx->workD;
  part = ctx->block;

  #ifdef USING_OPENMP_DH
  #pragma omp parallel 
  #endif
  {

    /* if matrix was scaled, must scale the rhs */
    if (scaleF != NULL && scaleD != NULL) {
      if (isSingle) {
        #ifdef USING_OPENMP_DH
        #pragma omp for schedule(static)
        #endif
        for (i=0; i<m; ++i) { xx[i] *= scaleF[i]; }
      } else {
        #ifdef USING_OPENMP_DH
        #pragma omp for schedule(static)
        #endif
        for (i=0; i<m; ++i) { xx[i] *= scaleD[i]; }
      }
    }

   #ifdef USING_OPENMP_DH
   #pragma omp for schedule(static) private(vF,vD,vi,nz,sumF,sumD,i)
   #endif
    for (h=0; h<pn; ++h) {    /* fwd solve diagonal block h */
      int from = part[h].beg_row;
      int to = part[h].end_row;

      if (isSingle) {
        workF[from] = xx[from];
        for ( i=from+1; i<to; i++ ) {
          vF  = avalF + rp[i];
          vi  = cval + rp[i];
          nz  = diag[i] - rp[i];
          sumF = xx[i];
          while (nz--) sumF -= (*vF++ * workF[*vi++]);
          workF[i] = sumF;
        }
      } else {
        workD[from] = xx[from];
        for ( i=from+1; i<to; i++ ) {
          vD  = avalD + rp[i];
          vi  = cval + rp[i];
          nz  = diag[i] - rp[i];
          sumD = xx[i];
          while (nz--) sumD -= (*vD++ * workD[*vi++]);
          workD[i] = sumD;
        }
      }

      /* backward solve the upper triangular */
      if (isSingle) {
        for ( i=to-1; i>=from; i-- ){
          vF   = avalF + diag[i] + 1;
          vi  = cval + diag[i] + 1;
          nz  = rp[i+1] - diag[i] - 1;
          sumF = workF[i];
          while (nz--) sumF -= (*vF++ * workF[*vi++]);
          yy[i] = workF[i] = sumF*avalF[diag[i]];
        }
      } else {
        for ( i=to-1; i>=from; i-- ){
          vD  = avalD + diag[i] + 1;
          vi  = cval + diag[i] + 1;
          nz  = rp[i+1] - diag[i] - 1;
          sumD = workD[i];
          while (nz--) sumD -= (*vD++ * workD[*vi++]);
          yy[i] = workD[i] = sumD*avalD[diag[i]];
        }
      }
    }

    /* put rhs back the way it was */
    if (scaleF != NULL && scaleD != NULL) {
      if (isSingle) {
        #ifdef USING_OPENMP_DH
        #pragma omp for schedule(static)
        #endif
        for (i=0; i<m; ++i) { xx[i] /= scaleF[i]; }
      } else {
        #ifdef USING_OPENMP_DH
        #pragma omp for schedule(static)
        #endif
        for (i=0; i<m; ++i) { xx[i] /= scaleD[i]; }
      }
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

  #ifdef USING_OPENMP_DH
  #pragma omp parallel 
  #endif
  {
    /* if matrix was scaled, must scale the rhs */
    if (scale != NULL) {
      #ifdef USING_OPENMP_DH
      #pragma omp for schedule(static)
       #endif
        for (i=0; i<n; ++i) { xx[i] *= scale[i]; }
    }

   #ifdef USING_OPENMP_DH
   #pragma omp for schedule(static) private(v,vi,nz,sum,i)
   #endif
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
      #ifdef USING_OPENMP_DH
      #pragma omp for schedule(static) 
      #endif
      for (i=0; i<n; ++i) { xx[i] /= scale[i]; }
    } 

  } /* #pragma omp parallel */

  END_FUNC_DH
}

#endif /* #if 0 */


/* 
   For single MPI task, shared-memory version of PILU.

   There is room for additional parallelism here, when
   factoring boundary rows.  For possible future development . . .
   (as written, factoring of boundary rows is entirely sequential)
*/
#undef __FUNC__
#define __FUNC__ "apply_pilu_seq_private"
void apply_pilu_seq_private(Euclid_dh ctx, double *xx, double *yy)
{
  START_FUNC_DH

  not_implemented_private(ctx); CHECK_V_ERROR;

#if 0
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
    #ifdef USING_OPENMP_DH
    #pragma omp for schedule(static)
    #endif
    for (i=0; i<n; ++i) { xx[i] *= scale[i]; }
  } 

    /* forward solve lower triangle interiors (parallel) */
   #ifdef USING_OPENMP_DH
   #pragma omp for schedule(static) private(from,to,v,vi,nz,sum,i)
   #endif
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
   #ifdef USING_OPENMP_DH
   #pragma omp for schedule(static) private(from,to,v,vi,nz,sum,i)
   #endif
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
    #ifdef USING_OPENMP_DH
    #pragma omp for schedule(static) 
    #endif
    for (i=0; i<n; ++i) { xx[i] /= scale[i]; }
  } 

#endif
  END_FUNC_DH
}

