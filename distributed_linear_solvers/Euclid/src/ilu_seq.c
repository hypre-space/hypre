/* to do: re-integrate fix-smalll-pivots */

#include "ilu_dh.h"
#include "Mem_dh.h"
#include "Parser_dh.h"
#include "Euclid_dh.h"

void ilut_seq_D(Euclid_dh ctx) {}
static void compute_scaling_private(int row, int len, double *AVAL, Euclid_dh ctx);

#if 0
 static  double fixPivot_private(int row, int len, float *vals); 
#endif

/*------------------------------------------------------------------*
 *  single precision functions start here
 *------------------------------------------------------------------*/

int symbolic_row_private(int localRow, 
                 int *list, int *marker, int *tmpFill,
                 int len, int *CVAL, double *AVAL,
                 int *o2n_col, Euclid_dh ctx);

int numeric_row_private(int localRow, 
                        int len, int *CVAL, double *AVAL,
                        float *workF, double *workD, Euclid_dh ctx);

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
  float    *workF, *avalF;
  double   *workD, *avalD;
  bool     isSingle;

  rp = ctx->rpF;
  cval = ctx->cvalF;
  fill = ctx->fillF;
  diag = ctx->diagF;
  avalF = ctx->avalF;
  avalD = ctx->avalD;
  workF = ctx->workF;
  workD = ctx->workD;
  isSingle = ctx->isSinglePrecision;

  /* allocate and initialize working space */
  o2n_col = (int*)MALLOC_DH((m+1)*sizeof(int)); CHECK_V_ERROR;
  list   = (int*)MALLOC_DH((m+1)*sizeof(int)); CHECK_V_ERROR;
  marker = (int*)MALLOC_DH(m*sizeof(int)); CHECK_V_ERROR;
  tmpFill = (int*)MALLOC_DH(m*sizeof(int)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) marker[i] = -1;
  rp[0] = 0;

  /* working space for values */
  if (isSingle) { for (i=0; i<m; ++i) workF[i] = 0.0; }
  else { for (i=0; i<m; ++i) workD[i] = 0.0; }

  /* form inverse permutation for column ordering */
  for (i=0; i<m; ++i) o2n_col[n2o_col[i]] = i;

  /*---------- main loop: factor row(i)  ----------*/
  for (i=from; i<to; ++i) {
    int row = n2o_row[i];             /* local row number */
    int globalRow = row + beg_row;    /* global row number */
    EuclidGetRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* compute scaling value for row(i) */
    if (ctx->isScaled) { 
      compute_scaling_private(i, len, AVAL, ctx); CHECK_V_ERROR; 
    }

    /* Compute symbolic factor for row(i);
       this also initializes value in the working vector.
     */
    count = symbolic_row_private(i, list, marker, tmpFill, 
                                 len, CVAL, AVAL,
                                 o2n_col, ctx); CHECK_V_ERROR;

    /* Ensure adequate storage; reallocate, if necessary. */
    if (idx + count > ctx->allocF) {
      reallocate_private(i, count, &(ctx->allocF), &rp, &cval, &avalF, &avalD, &fill); CHECK_V_ERROR;
      ctx->cvalF = cval;
      ctx->avalF = avalF;
      ctx->avalD = avalD;
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
                          workF, workD, ctx); CHECK_V_ERROR
    EuclidRestoreRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* Copy factored numeric row to permanent storage,
       and re-zero work vector
     */
    if (isSingle) {
      for (j=rp[i]; j<rp[i+1]; ++j) 
        { col = cval[j];  avalF[j] = workF[col];  workF[col] = 0.0; }
    } else {
      for (j=rp[i]; j<rp[i+1]; ++j) 
        { col = cval[j];  avalD[j] = workD[col];  workD[col] = 0.0; }
    }
  }

  FREE_DH(o2n_col); CHECK_V_ERROR;
  FREE_DH(list); CHECK_V_ERROR;
  FREE_DH(tmpFill); CHECK_V_ERROR;
  FREE_DH(marker); CHECK_V_ERROR;
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
                 int *o2n_col, Euclid_dh ctx)
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
                        float *workF, double *workD, Euclid_dh ctx)
{
  START_FUNC_DH
  double  pc, pv, multiplier;
  int     j, k, row;
  int     m = ctx->m, *rp = ctx->rpF, *cval = ctx->cvalF;
  int     *diag = ctx->diagF;
  int     beg_row = ctx->beg_row;
  float   *avalF = ctx->avalF;
  double  *avalD = ctx->avalD;
  float   scale;
  bool    isSingle = ctx->isSinglePrecision;

  if (isSingle) { scale = ctx->scaleF[localRow]; }
  else          { scale = ctx->scaleD[localRow]; }

  /* zero work vector */
  if (isSingle) {
    for (j=rp[localRow]; j<rp[localRow+1]; ++j) 
      { int col = cval[j];  workF[col] = 0.0; }
  } else {
    for (j=rp[localRow]; j<rp[localRow+1]; ++j) 
      { int col = cval[j];  workD[col] = 0.0; }
  }

  /* init work vector with values from A */
  /* (note: some values may be na due to sparsification; this is O.K.) */
  if (isSingle) {
    for (j=0; j<len; ++j) {
      int col = *CVAL++ - beg_row;
      float val = *AVAL++;
      if (col >= 0 && col < m) workF[col] = val*scale;
    }
  } else {
    for (j=0; j<len; ++j) {
      int col = *CVAL++ - beg_row;
      float val = *AVAL++;
      if (col >= 0 && col < m) workD[col] = val*scale;
    }
  }

  for (j=rp[localRow]; j<diag[localRow]; ++j) {
    row = cval[j];
    if (isSingle) { pc = workF[row]; }
    else          { pc = workD[row]; }

    if (pc != 0.0) {
      if (isSingle) { pv = avalF[diag[row]]; }
      else          { pv = avalD[diag[row]]; }
      multiplier = pc / pv;
      if (isSingle) { workF[row] = multiplier; }
      else          { workD[row] = multiplier; }

      if (isSingle) {
        for (k=diag[row]+1; k<rp[row+1]; ++k) {
          int col = cval[k];
          workF[col] -= (multiplier * avalF[k]);
        }
      } else {
        for (k=diag[row]+1; k<rp[row+1]; ++k) {
          int col = cval[k];
          workD[col] -= (multiplier * avalD[k]);
        }
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


/*-----------------------------------------------------------------------*
 * ILUT starts here
 *-----------------------------------------------------------------------*/

/* single precision version */

int ilut_row_private(int localRow, int *list, int *o2n_col, int *marker,
                     int len, int *CVAL, double *AVAL,
                     float *workF, double *workD, Euclid_dh ctx);

#undef __FUNC__
#define __FUNC__ "ilut_seq"
void ilut_seq(Euclid_dh ctx)
{
  START_FUNC_DH
  int      *rp, *cval, *diag, *CVAL;
  int      i, len, count, col, idx = 0;
  int      *list, *marker;
  int      temp, m, from, to;
  int      *n2o_row, *n2o_col, *o2n_col, beg_row;
  double   *AVAL;
  float    *workF, *avalF;
  double   *workD, *avalD;
  bool     isSingle;

  m = ctx->m;
  rp = ctx->rpF;
  cval = ctx->cvalF;
  diag = ctx->diagF;
  avalF = ctx->avalF;
  avalD = ctx->avalD;
  workF = ctx->workF;
  workD = ctx->workD;
  isSingle = ctx->isSinglePrecision;
  n2o_row = ctx->n2o_row;
  n2o_col = ctx->n2o_col;
  from = ctx->from;
  to = ctx->to;
  beg_row = ctx->beg_row;

  /* allocate and initialize working space */
  o2n_col = (int*)MALLOC_DH((m+1)*sizeof(int)); CHECK_V_ERROR;
  list   = (int*)MALLOC_DH((m+1)*sizeof(int)); CHECK_V_ERROR;
  marker = (int*)MALLOC_DH(m*sizeof(int)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) marker[i] = -1;
  rp[0] = 0;

  /* working space for values */
  if (isSingle) { for (i=0; i<m; ++i) workF[i] = 0.0; }
  else { for (i=0; i<m; ++i) workD[i] = 0.0; }

  /* form inverse permutation for column ordering */
  for (i=0; i<m; ++i) o2n_col[n2o_col[i]] = i;

  /* ----- main loop start ----- */
  for (i=from; i<to; ++i) {
    int row = n2o_row[i];             /* local row number */
    int globalRow = row + beg_row;    /* global row number */
    EuclidGetRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* compute scaling value for row(i) */
    if (ctx->isScaled) { 
      compute_scaling_private(i, len, AVAL, ctx); CHECK_V_ERROR; 
    }

    /* compute factor for row i */
    count = ilut_row_private(i, list, o2n_col, marker,
                         len, CVAL, AVAL, workF, workD, ctx); CHECK_V_ERROR;

    EuclidRestoreRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* Ensure adequate storage; reallocate, if necessary. */
    if (idx + count > ctx->allocF) {
      reallocate_private(i, count, &(ctx->allocF), &rp, &cval, &avalF, &avalD, NULL); CHECK_V_ERROR;
      ctx->cvalF = cval;
      ctx->avalF = avalF;
      ctx->avalD = avalD;
    }

    /* Copy factored row to permanent storage 
       and re-zero work vector
     */
    col = list[m];
    if (isSingle) {
      while (count--) {
        cval[idx] = col;   avalF[idx++] = workF[col];  workF[col] = 0.0;
        col = list[col];
      }
    } else {
      while (count--) {
        cval[idx] = col;   avalD[idx++] = workD[col];  workD[col] = 0.0;
        col = list[col];
      }
    }

    /* add row-pointer to start of next row. */
    rp[i+1] = idx;

    /* Insert pointer to diagonal */
    temp = rp[i]; 
    while (cval[temp] != i) ++temp;
    diag[i] = temp;
  }
  /* ----- main loop end ----- */

  FREE_DH(list);
  FREE_DH(marker);
  FREE_DH(o2n_col);
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "ilut_row_private"
int ilut_row_private(int localRow, int *list, int *o2n_col, int *marker,
                     int len, int *CVAL, double *AVAL,
                     float *workF, double *workD, Euclid_dh ctx)
{
  START_FUNC_DH
  int     j, col, m = ctx->m, *rp = ctx->rpF, *cval = ctx->cvalF;
  int     tmp, *diag = ctx->diagF;
  int     head;
  int     count = 0, beg_row = ctx->beg_row;
  double  val;
  float   multF, *avalF = ctx->avalF, scaleF;
  double  multD, *avalD = ctx->avalD, scaleD;
  double  pv, pc, thresh = ctx->sparseTolA;
  bool    isSingle = ctx->isSinglePrecision;
  double  droptol = ctx->droptol;

  if (isSingle) { scaleF = ctx->scaleF[localRow]; }
  else          { scaleD = ctx->scaleD[localRow]; }

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
        if (isSingle) { workF[col] = val*scaleF; }
        else          { workD[col] = val*scaleD; }
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
    marker[localRow]  = localRow;
    ++count;
  }

  /* update current row from previously factored rows */
  head = m;
  while (list[head] < localRow) {
    int row = list[head];

    /* get the multiplier, and apply 1st drop tolerance test */
    if (isSingle) { pc = workF[row]; }
    else          { pc = workD[row]; }
    if (pc != 0.0) {
      if (isSingle) { pv = avalF[diag[row]]; }
      else          { pv = avalD[diag[row]]; }
      multF = multD = pc / pv;

      if (fabs(multD) > droptol) {
        if (isSingle) { workF[row] = multF; }
        else          { workD[row] = multD; }

        /* single precision row update */
        if (isSingle) {
          for (j=diag[row]+1; j<rp[row+1]; ++j) {
            col = cval[j];
            workF[col] -= (multF * avalF[j]);

            /* if col isn't already present in the linked-list, insert it.  */
            if (marker[col] < localRow) {
              marker[col] = localRow;     /* mark the column as known fill */
              tmp = head;             /* insert in list [this and next 3 lines] */
              while (col > list[tmp]) tmp = list[tmp];
              list[col] = list[tmp];
              list[tmp] = col;
              ++count;                /* increment fill count */
            }
          } 
        } 

        /* double precision row update */
        else {
          for (j=diag[row]+1; j<rp[row+1]; ++j) {
            col = cval[j];
            workD[col] -= (multD * avalD[j]);

            /* if col isn't already present in the linked-list, insert it.  */
            if (marker[col] < localRow) {
              marker[col] = localRow;        /* mark the column as known fill */
              tmp = head;             /* insert in list [this and next 3 lines] */
              while (col > list[tmp]) tmp = list[tmp];
              list[col] = list[tmp];
              list[tmp] = col;
              ++count;                /* increment fill count */
            }
          }
        }
      }
    }
    head = list[head];  /* advance to next item in linked list */
  }

  END_FUNC_VAL(count)
}


#undef __FUNC__
#define __FUNC__ "compute_scaling_private"
void compute_scaling_private(int row, int len, double *AVAL, Euclid_dh ctx)
{
  START_FUNC_DH
  double tmp = 0.0;
  int j;
  bool isSingle = ctx->isSinglePrecision;

  for (j=0; j<len; ++j) tmp = MAX( tmp, fabs(AVAL[j]) );
  if (tmp) {
    if (isSingle) { ctx->scaleF[row] = 1.0/tmp; }
    else          { ctx->scaleD[row] = 1.0/tmp; }
  }
  END_FUNC_DH
}
