/* to do: re-integrate fix-smalll-pivots */

/*
   There are two versions of everything: one for single and
   one for double precision.  Boy, is that lousy!  This would
   be a perfect place for templates, if only they worked a
   bit better . . .
*/


#include "ilu_dh.h"
#include "Mem_dh.h"
#include "Parser_dh.h"
#include "Euclid_dh.h"

#if 0
 static  double fixPivot_private(int row, int len, float *vals); 
#endif

/*------------------------------------------------------------------*
 *  single precision functions start here
 *------------------------------------------------------------------*/

int symbolic_row_private(int localRow, 
                 int *list, int *marker, int *tmpFill,
                 int len, int *CVAL, double *AVAL,
                 int *o2n_col, float *work, Euclid_dh ctx);

int numeric_row_private(int localRow, 
                        int len, int *CVAL, double *AVAL,
                        float *work, Euclid_dh ctx);

#if 0

/* not used ? */
#undef __FUNC__
#define __FUNC__ "fixPivot_private"
double fixPivot_private(int row, int len, float *vals)
{
  START_FUNC_DH
  int i;
  float max = 0.0;
  bool debug = false;

  for (i=0; i<len; ++i) {
    float tmp = fabs(vals[i]);
    max = MAX(max, tmp);
  }
  END_FUNC_VAL(max* ctxPrivate->pivotFix)
}

#endif

#undef __FUNC__
#define __FUNC__ "iluk_seq"
void iluk_seq(Euclid_dh ctx)
{
  START_FUNC_DH
  int      *rp, *cval, *diag;
  int      *CVAL;
  int      i, j, len, count, col, idx = 0;
  int      *list, *marker, *fill, *tmpFill;
  int      temp, m = ctx->m, from = ctx->from, to = ctx->to;
  int      *n2o_row = ctx->n2o_row, *n2o_col = ctx->n2o_col, *o2n_col;
  int      beg_row = ctx->beg_row;
  double   *AVAL;
  float    *work, *aval = ctx->avalF;

  rp = ctx->rpF;
  cval = ctx->cvalF;
  fill = ctx->fillF;
  diag = ctx->diagF;

  /* allocate and initialize working space */
  o2n_col = (int*)MALLOC_DH((m+1)*sizeof(int)); CHECK_V_ERROR;
  list   = (int*)MALLOC_DH((m+1)*sizeof(int)); CHECK_V_ERROR;
  marker = (int*)MALLOC_DH(m*sizeof(int)); CHECK_V_ERROR;
  tmpFill = (int*)MALLOC_DH(m*sizeof(int)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) marker[i] = -1;
  rp[0] = 0;

  /* working space for values */
  work = (float*)MALLOC_DH(m*sizeof(float)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) work[i] = 0.0;

  /* form inverse permutation for column ordering */
  for (i=0; i<m; ++i) o2n_col[n2o_col[i]] = i;

  /*---------- main loop: factor row(i)  ----------*/
  for (i=from; i<to; ++i) {
    int row = n2o_row[i];             /* local row number */
    int globalRow = row + beg_row;    /* global row number */
    EuclidGetRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* compute scaling value for row(i) */
    if (ctx->isScaled) {
      double tmp = 0.0;
      int j;
      for (j=0; j<len; ++j) tmp = MAX( tmp, fabs(AVAL[j]) );
      if (tmp) {
        ctx->scale[i] = 1.0/tmp;
      }
    }

    /* Compute symbolic factor for row(i);
       this also initializes value in the working vector.
     */
    count = symbolic_row_private(i, list, marker, tmpFill, 
                                 len, CVAL, AVAL,
                                 o2n_col, work, ctx); CHECK_V_ERROR;

    /* Ensure adequate storage; reallocate, if necessary. */
    if (idx + count > ctx->allocF) {
      reallocate_private(i, count, &(ctx->allocF), &rp, &cval, &aval, NULL, &fill); CHECK_V_ERROR;
      ctx->cvalF = cval;
      ctx->avalF = aval;
      ctx->fillF = fill;
    }

    /* Copy factored symbolic row to permanent storage */
    col = list[m];
    while (count--) {
      cval[idx] = col;  
      fill[idx] = tmpFill[col];
      ++idx;
      col = list[col];
    }

    /* add row-pointer to start of next row. */
    rp[i+1] = idx;

    /* Insert pointer to diagonal */
    temp = rp[i]; 
    while (cval[temp] != i) ++temp;
    diag[i] = temp;

    /* compute numeric factor for current row */
     numeric_row_private(i, len, CVAL, AVAL, 
                          work, ctx); CHECK_V_ERROR
    EuclidRestoreRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* Copy factored numeric row to permanent storage,
       and re-zero work vector
     */

    for (j=rp[i]; j<rp[i+1]; ++j) {
      col = cval[j];
      aval[j] = work[col];
      work[col] = 0.0;
    }
  }

  FREE_DH(o2n_col); CHECK_V_ERROR;
  FREE_DH(list); CHECK_V_ERROR;
  FREE_DH(tmpFill); CHECK_V_ERROR;
  FREE_DH(marker); CHECK_V_ERROR;
  FREE_DH(work); CHECK_V_ERROR;
  END_FUNC_DH
}


/* Computes ILU(K) factor of a single row; returns fill 
   count for the row.  Explicitly inserts diag if not already 
   present.  On return, all column indices are local 
   (i.e, referenced to 0).
*/
#undef __FUNC__
#define __FUNC__ "symbolic_row_private"
int symbolic_row_private(int localRow, 
                 int *list, int *marker, int *tmpFill,
                 int len, int *CVAL, double *AVAL,
                 int *o2n_col, float *work, Euclid_dh ctx)
{
  START_FUNC_DH
  int level = ctx->level, m = ctx->m;
  int *cval = ctx->cvalF, *diag = ctx->diagF, *rp = ctx->rpF; 
  int *fill = ctx->fillF;
  int count = 0;
  int j, node, tmp, col, head;
  int fill1, fill2;
  int beg_row = ctx->beg_row;
  float val;
  double thresh = ctx->sparseTolA;

  /* Insert col indices in linked list, and values in work vector.
   * List[m] points to the first (smallest) col in the linked list.
   * Column values are adjusted from global to local numbering.
   */
  list[m] = m;
  for (j=0; j<len; ++j) {
    tmp = m;
    col = *CVAL++ - beg_row;
    val = *AVAL++;

    if (col >= 0 && col < m) { /* this statement is for block jacobi */
      if (fabs(val) > thresh) {  /* sparsification */
        ++count;
        col = o2n_col[col];
        while (col > list[tmp]) tmp = list[tmp];
        list[col]   = list[tmp];
        list[tmp]   = col;
        tmpFill[col] = 0;
        marker[col] = localRow;
      }
    }
  }

  /* insert diag if not already present */
  if (marker[localRow] != localRow) {
    ctx->symbolicZeroDiags += 1;
    tmp = m;
    while (localRow > list[tmp]) tmp = list[tmp];
    list[localRow]    = list[tmp];
    list[tmp]    = localRow;
    tmpFill[localRow] = 0;
    marker[localRow]  = localRow;
    ++count;
  }

  /* update row from previously factored rows */
  head = m;
  if (level > 0) {
    while (list[head] < localRow) {
      node = list[head];
      fill1 = tmpFill[node];

      if (fill1 < level) {
        for (j = diag[node]+1; j<rp[node+1]; ++j) {
          col = cval[j];
          fill2 = fill1 + fill[j] + 1;

          if (fill2 <= level) {
            /* if newly discovered fill entry, mark it as discovered;
             * if entry has level <= K, add it to the linked-list.
             */
            if (marker[col] < localRow) {
              tmp = head;
              marker[col] = localRow;
              tmpFill[col] = fill2;
              while (col > list[tmp]) tmp = list[tmp];
              list[col] = list[tmp];
              list[tmp]    = col;
              ++count; /* increment fill count */
            }

          /* if previously-discovered fill, update the entry's level. */
            else {
              tmpFill[col] = (fill2 < tmpFill[col]) ? fill2 : tmpFill[col];
            }
          }
        }
      } /* fill1 < level  */
      head = list[head];  /* advance to next item in linked list */
    }
  }
  END_FUNC_VAL(count)
}


#undef __FUNC__
#define __FUNC__ "numeric_row_private"
int numeric_row_private(int localRow,
                        int len, int *CVAL, double *AVAL,
                        float *work, Euclid_dh ctx)
{
  START_FUNC_DH
  float pc, pv, multiplier;
  int     j, k, row;
  int     m = ctx->m, *rp = ctx->rpF, *cval = ctx->cvalF;
  int     *diag = ctx->diagF;
  int     beg_row = ctx->beg_row;
  float   *aval = ctx->avalF;
  float   scale = ctx->scale[localRow];

  /* zero work vector */
  for (j=rp[localRow]; j<rp[localRow+1]; ++j) {
    int col = cval[j];
    work[col] = 0.0;
  }

  /* init work vector with values from A */
  /* (note: some values may be na due to sparsification; this is O.K.) */
  for (j=0; j<len; ++j) {
    int col = *CVAL++ - beg_row;
    float val = *AVAL++;
    if (col >= 0 && col < m) work[col] = val*scale;
  }

  for (j=rp[localRow]; j<diag[localRow]; ++j) {
    row = cval[j];
    pc = work[row];

    if (pc != 0.0) {
      pv = aval[diag[row]];
      multiplier = pc / pv;

      work[row] = multiplier;
      for (k=diag[row]+1; k<rp[row+1]; ++k) {
        int col = cval[k];
        work[col] -= (multiplier * aval[k]);
      }
    }
  }

  /* check for zero or too small of a pivot */
#if 0
  if (fabs(work[i]) <= pivotTol) {
    /* yuck! assume row scaling, and just stick in a value */
    aval[diag[i]] = pivotFix;
  }
#endif

  END_FUNC_VAL(0)
}

/*------------------------------------------------------------------*
 *  double precision functions start here
 *------------------------------------------------------------------*/

int symbolic_row_private_D(int localRow, 
                 int *list, int *marker, int *tmpFill,
                 int len, int *CVAL, double *AVAL,
                 int *o2n_col, double *work, Euclid_dh ctx);

int numeric_row_private_D(int localRow, 
                        int len, int *CVAL, double *AVAL,
                        double *work, Euclid_dh ctx);

#undef __FUNC__
#define __FUNC__ "iluk_seq_D"
void iluk_seq_D(Euclid_dh ctx)
{
  START_FUNC_DH
  int      *rp, *cval, *diag;
  int      *CVAL;
  int      i, j, len, count, col, idx = 0;
  int      *list, *marker, *fill, *tmpFill;
  int      temp, m = ctx->m, from = ctx->from, to = ctx->to;
  int      *n2o_row = ctx->n2o_row, *n2o_col = ctx->n2o_col, *o2n_col;
  int      beg_row = ctx->beg_row;
  double   *AVAL;
  double    *work, *aval = ctx->avalFD;

  rp = ctx->rpF;
  cval = ctx->cvalF;
  fill = ctx->fillF;
  diag = ctx->diagF;

  /* allocate and initialize working space */
  o2n_col = (int*)MALLOC_DH((m+1)*sizeof(int)); CHECK_V_ERROR;
  list   = (int*)MALLOC_DH((m+1)*sizeof(int)); CHECK_V_ERROR;
  marker = (int*)MALLOC_DH(m*sizeof(int)); CHECK_V_ERROR;
  tmpFill = (int*)MALLOC_DH(m*sizeof(int)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) marker[i] = -1;
  rp[0] = 0;

  /* working space for values */
  work = (double*)MALLOC_DH(m*sizeof(double)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) work[i] = 0.0;

  /* form inverse permutation for column ordering */
  for (i=0; i<m; ++i) o2n_col[n2o_col[i]] = i;

  /*---------- main loop: factor row(i)  ----------*/
  for (i=from; i<to; ++i) {
    int row = n2o_row[i];             /* local row number */
    int globalRow = row + beg_row;    /* global row number */
    EuclidGetRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* compute scaling value for row(i) */
    if (ctx->isScaled) {
      double tmp = 0.0;
      int j;
      for (j=0; j<len; ++j) tmp = MAX( tmp, fabs(AVAL[j]) );
      if (tmp) {
        ctx->scaleD[i] = 1.0/tmp;
      }
    }

    /* Compute symbolic factor for row(i);
       this also initializes value in the working vector.
     */
    count = symbolic_row_private_D(i, list, marker, tmpFill, 
                                 len, CVAL, AVAL,
                                 o2n_col, work, ctx); CHECK_V_ERROR;

    /* Ensure adequate storage; reallocate, if necessary. */
    if (idx + count > ctx->allocF) {
      reallocate_private(i, count, &(ctx->allocF), &rp, &cval, NULL, &aval, &fill); CHECK_V_ERROR;
      ctx->cvalF = cval;
      ctx->avalFD = aval;
      ctx->fillF = fill;
    }

    /* Copy factored symbolic row to permanent storage */
    col = list[m];
    while (count--) {
      cval[idx] = col;  
      fill[idx] = tmpFill[col];
      ++idx;
      col = list[col];
    }

    /* add row-pointer to start of next row. */
    rp[i+1] = idx;

    /* Insert pointer to diagonal */
    temp = rp[i]; 
    while (cval[temp] != i) ++temp;
    diag[i] = temp;

    /* compute numeric factor for current row */
     numeric_row_private_D(i, len, CVAL, AVAL, 
                          work, ctx); CHECK_V_ERROR
    EuclidRestoreRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* Copy factored numeric row to permanent storage,
       and re-zero work vector
     */
    for (j=rp[i]; j<rp[i+1]; ++j) {
      col = cval[j];
      aval[j] = work[col];
      work[col] = 0.0;
    }
  }

  FREE_DH(o2n_col); CHECK_V_ERROR;
  FREE_DH(list); CHECK_V_ERROR;
  FREE_DH(tmpFill); CHECK_V_ERROR;
  FREE_DH(marker); CHECK_V_ERROR;
  FREE_DH(work); CHECK_V_ERROR;
  END_FUNC_DH
}


/* Computes ILU(K) factor of a single row; returns fill 
   count for the row.  Explicitly inserts diag if not already 
   present.  On return, all column indices are local 
   (i.e, referenced to 0).
*/
#undef __FUNC__
#define __FUNC__ "symbolic_row_private_D"
int symbolic_row_private_D(int localRow, 
                 int *list, int *marker, int *tmpFill,
                 int len, int *CVAL, double *AVAL,
                 int *o2n_col, double *work, Euclid_dh ctx)
{
  START_FUNC_DH
  int level = ctx->level, m = ctx->m;
  int *cval = ctx->cvalF, *diag = ctx->diagF, *rp = ctx->rpF; 
  int *fill = ctx->fillF;
  int count = 0;
  int j, node, tmp, col, head;
  int fill1, fill2;
  int beg_row = ctx->beg_row;
  double val;
  double thresh = ctx->sparseTolA;

  /* Insert col indices in linked list, and values in work vector.
   * List[m] points to the first (smallest) col in the linked list.
   * Column values are adjusted from global to local numbering.
   */
  list[m] = m;
  for (j=0; j<len; ++j) {
    tmp = m;
    col = *CVAL++ - beg_row;
    val = *AVAL++;

    if (col >= 0 && col < m) { /* this statement is for block jacobi */
      if (fabs(val) > thresh) {  /* sparsification */
        ++count;
        col = o2n_col[col];
        while (col > list[tmp]) tmp = list[tmp];
        list[col]   = list[tmp];
        list[tmp]   = col;
        tmpFill[col] = 0;
        marker[col] = localRow;
      }
    }
  }

  /* insert diag if not already present */
  if (marker[localRow] != localRow) {
    ctx->symbolicZeroDiags += 1;
    tmp = m;
    while (localRow > list[tmp]) tmp = list[tmp];
    list[localRow]    = list[tmp];
    list[tmp]    = localRow;
    tmpFill[localRow] = 0;
    marker[localRow]  = localRow;
    ++count;
  }

  /* update row from previously factored rows */
  head = m;
  if (level > 0) {
    while (list[head] < localRow) {
      node = list[head];
      fill1 = tmpFill[node];

      if (fill1 < level) {
        for (j = diag[node]+1; j<rp[node+1]; ++j) {
          col = cval[j];
          fill2 = fill1 + fill[j] + 1;

          if (fill2 <= level) {
            /* if newly discovered fill entry, mark it as discovered;
             * if entry has level <= K, add it to the linked-list.
             */
            if (marker[col] < localRow) {
              tmp = head;
              marker[col] = localRow;
              tmpFill[col] = fill2;
              while (col > list[tmp]) tmp = list[tmp];
              list[col] = list[tmp];
              list[tmp]    = col;
              ++count; /* increment fill count */
            }

          /* if previously-discovered fill, update the entry's level. */
            else {
              tmpFill[col] = (fill2 < tmpFill[col]) ? fill2 : tmpFill[col];
            }
          }
        }
      } /* fill1 < level  */
      head = list[head];  /* advance to next item in linked list */
    }
  }
  END_FUNC_VAL(count)
}


#undef __FUNC__
#define __FUNC__ "numeric_row_private_D"
int numeric_row_private_D(int localRow,
                        int len, int *CVAL, double *AVAL,
                        double *work, Euclid_dh ctx)
{
  START_FUNC_DH
  double pc, pv, multiplier;
  int     j, k, row;
  int     m = ctx->m, *rp = ctx->rpF, *cval = ctx->cvalF;
  int     *diag = ctx->diagF;
  int     beg_row = ctx->beg_row;
  double  *aval = ctx->avalFD;
  double  scale = ctx->scaleD[localRow];

  /* zero work vector */
  for (j=rp[localRow]; j<rp[localRow+1]; ++j) {
    work[cval[j]] = 0.0;
  }

  /* init work vector with values from A */
  /* (note: some values may be na due to sparsification; this is O.K.) */
  for (j=0; j<len; ++j) {
    int col = *CVAL++ - beg_row;
    float val = *AVAL++;
    if (col >= 0 && col < m) work[col] = val*scale;
  }

  for (j=rp[localRow]; j<diag[localRow]; ++j) {
    row = cval[j];
    pc = work[row];

    if (pc != 0.0) {
      pv = aval[diag[row]];
      multiplier = pc / pv;

      work[row] = multiplier;
      for (k=diag[row]+1; k<rp[row+1]; ++k) {
        int col = cval[k];
        work[col] -= (multiplier * aval[k]);
      }
    }
  }

  /* check for zero or too small of a pivot */
#if 0
  if (fabs(work[i]) <= pivotTol) {
    /* yuck! assume row scaling, and just stick in a value */
    aval[diag[i]] = pivotFix;
  }
#endif

  END_FUNC_VAL(0)
}
