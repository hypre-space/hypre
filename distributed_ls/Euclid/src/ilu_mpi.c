#include "ilu_dh.h"
#include "getRow_dh.h"
#include "Mem_dh.h"
#include "Parser_dh.h"
#include "Hash_dh.h"
#include "Euclid_dh.h"
#include "SortedList_dh.h"

#undef __FUNC__
#define __FUNC__ "iluk_mpi"
void iluk_mpi(Euclid_dh ctx)
{
  START_FUNC_DH
  SET_V_ERROR("need to fix REAL_DH!!!!!");
  END_FUNC_DH
}


#if 0

#define IS_LOCAL(col, beg, end) (((col) >= (beg) && (col) < (end)) ? true : false)

void iluk_symbolic_row_private(int localRow, int len, int *RP, int *CVAL, double *AVAL, 
                 Hash_dh externalRows, SortedList_dh sList, Euclid_dh ctx);

void iluk_numeric_row_private(int new_row, 
                 Hash_dh externalRows, SortedList_dh sList, Euclid_dh ctx);

#undef __FUNC__
#define __FUNC__ "iluk_mpi"
void iluk_mpi(Euclid_dh ctx)
{
  START_FUNC_DH
  int      *RP, *CVAL;
  int      *rp = ctx->rpF, *cval = ctx->cvalF, *fill = ctx->fillF; 
  int      *diag = ctx->diagF;
  REAL_DH  *aval = ctx->avalF, *AVAL, scale;
  int      i, len, count, m = ctx->m, from = ctx->from, to = ctx->to;
  int      idx = 0;
  int      *n2o_row = ctx->n2o_row;
  int      row, beg_row = ctx->beg_row;
  SortedList_dh sList;

  /* initialize sorted list thingy */
  SortedList_dhCreate(&sList); CHECK_V_ERROR;
  SortedList_dhInit(sList, m, beg_row, ctx->n2o_col, ctx->o2n_nonLocal); CHECK_V_ERROR;

  /* loop over rows to be factored */
  for (i=from; i<to; ++i) {
    int row = n2o_row[i];             /* local row number */
    int globalRow = row + beg_row;    /* global row number */
    EuclidGetRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;
EuclidRestoreRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

fprintf(logFile, "--------------------------------------- i= %i, row= %i\n", i+1, row+1);

    /* compute scaling value for row(i) */
    scale = 1.0;
    if (ctx->isScaled) {
      double tmp = 0.0;
      int j;
      for (j=0; j<len; ++j) tmp = MAX( tmp, fabs(AVAL[j]) );
      if (tmp) {
        ctx->scale[i] = scale = 1.0/tmp;
      }
    }

    SortedList_dhReset(sList, i); CHECK_V_ERROR;

    /* compute ILU(k) symbolic factor for row */
    /* Compute symbolic factor for row(i);
       this also initializes value in the working vector.
     */
    iluk_symbolic_row_private(i, len, RP, CVAL, AVAL, 
                            ctx->externalRows, sList, ctx); CHECK_V_ERROR;

    /* compute numeric factor for row */
    iluk_numeric_row_private(i, ctx->externalRows, sList, ctx); CHECK_V_ERROR;

    /* Ensure adequate storage; reallocate, if necessary. */
    count = SortedList_dhReadCount(sList); CHECK_V_ERROR;

/* ? */ fprintf(logFile, "##  idx = %i\n", idx); 
idx = rp[i]; 
fprintf(logFile, "@@  idx = %i\n", idx);

    if (idx + count > ctx->allocF) {
fprintf(logFile, "REALLOCATING\n");
      reallocate_private(i, count, &(ctx->allocF), &rp, &cval, &aval, &fill); CHECK_V_ERROR;
      ctx->cvalF = cval;
      ctx->avalF = aval;
      ctx->fillF = fill;
    }

    /* Copy factor to permanent storage */
    while (count--) {
      SRecord *sr = SortedList_dhGetSmallest(sList); CHECK_V_ERROR;
fprintf(logFile, "  final: cval[%i] = %i  val= %g\n", idx, 1+sr->col, sr->val);
      cval[idx] = sr->col;
      aval[idx] = sr->val;
      fill[idx] = sr->level;
      ++idx;
    }
fprintf(logFile, "\n");

    /* add row-pointer to start of next row. */
    rp[i+1] = idx;

    /* Insert pointer to diagonal */
    { int j;
      bool flag = true;
      for (j=rp[i]; j<rp[i+1]; ++j) {
        if (cval[j] == i+beg_row) {
          diag[i] = j;
          flag = false;
          break;
        }
      }
      if (flag) {
        sprintf(msgBuf_dh, "failed to find diag for row= %i", 1+i);
        SET_V_ERROR(msgBuf_dh);
      }
    }

#if 0
    /* check for zero or too small of a pivot */
    if (fabs(aval[diag[i]]) <= pivotTol) {
      /* yuck! assume row scaling, and just stick in a value */
      aval[diag[i]] = pivotFix;
    }
#endif

  }

  SortedList_dhDestroy(sList); CHECK_V_ERROR;
  END_FUNC_DH
}

/* Computes ILU(K) factor of a single row; returns fill 
   count for the row.  Explicitly inserts diag if not already 
   present.  Inserts values from A in work vector (this is so
   we don't have to call GET_ROW more than once)
*/
#undef __FUNC__
#define __FUNC__ "iluk_symbolic_row_private"
void iluk_symbolic_row_private(int localRow, int len, int *RP, int *CVAL, double *AVAL, 
                 Hash_dh externalRows, SortedList_dh sList, Euclid_dh ctx)
{
  START_FUNC_DH
  int level = ctx->level, m = ctx->m;
  int beg_row = ctx->beg_row;
  int *cval = ctx->cvalF, *diag = ctx->diagF, *rp = ctx->rpF; 
  int *fill = ctx->fillF;
  int j, node, col, end_row = beg_row + m;
  int len, level_1, level_2;
  int *cvalPtr, *fillPtr;
  SRecord sr, *srPtr;
  REAL_DH scale = ctx->scale[localRow];
  double thresh = ctx->sparseTolA;

  /* insert col indices in sorted linked list */
  sr.level = 0;
  for (j=0; j<len; ++j) {
    sr.col = *CVAL++;
    sr.val = scale * *AVAL++;
    if (sr.val > thresh) {
      SortedList_dhPermuteAndInsert(sList, &sr); CHECK_V_ERROR;
    }
  }

  /* ensure diagonal entry is inserted */
  sr.val = 0.0; 
  sr.col = localRow+beg_row;
  srPtr = SortedList_dhFind(sList, &sr); CHECK_V_ERROR;
  if (srPtr == NULL) {
    sr.col = localRow+beg_row;
    SortedList_dhPermuteAndInsert(sList, &sr); CHECK_V_ERROR;
  }

 SortedList_dhPrint(sList, logFile);

  /* update row from previously factored rows */
  if (level > 0) {
    while(1) {
      srPtr = SortedList_dhGetSmallestLowerTri(sList); CHECK_V_ERROR;

if (srPtr == NULL) fprintf(logFile, "symbolic: got row= NULL\n");

      if (srPtr == NULL) break;    

fprintf(logFile, "symbolic: got row= %i\n", srPtr->col + 1);

      node = srPtr->col;
      level_1 = srPtr->level;
      if (level_1 < level) {

        if (IS_LOCAL(node, beg_row, end_row)) {
          node -= beg_row;
fprintf(logFile, "symbolic: MERGING IN LOCAL ROW= %i\n", node+1);

          len = rp[node+1] - diag[node] - 1; 
          cvalPtr = cval + diag[node] + 1;
          fillPtr = fill + diag[node] + 1;
        } else {
          len = 0;
          fprintf(logFile, "can't merge non-local row %i", node+1);
        }

        /* merge in strict upper triangular portion of row */
        for (j = 0; j<len; ++j) {
          col = *cvalPtr++;
          level_2 = 1+ level_1 + *fillPtr++;
          if (level_2 <= level) {
            /* Insert new element, or update level if already inserted. */
            sr.col = col;
            sr.level = level_2;
            SortedList_dhInsertOrUpdate(sList, &sr); CHECK_V_ERROR;
          }
        }
      } 
    } 
  } 
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "iluk_numeric_row_private"
void iluk_numeric_row_private(int new_row, 
                 Hash_dh externalRows, SortedList_dh sList, Euclid_dh ctx)
{
  START_FUNC_DH
  int m = ctx->m, beg_row = ctx->beg_row;
  int *rp = ctx->rpF, *cval = ctx->cvalF, *diag = ctx->diagF;
  REAL_DH *aval = ctx->avalF;
  int     len, row, end_row = beg_row+m;
  int     *cvalPtr;
  REAL_DH *avalPtr, multiplier, pc, pv;
  SRecord sr, *srPtr;

  /* note: non-zero entries from A were inserted in list during iluk_symbolic_row_private */

  SortedList_dhResetGetSmallest(sList); CHECK_V_ERROR;
  while (1) {
    srPtr = SortedList_dhGetSmallestLowerTri(sList); CHECK_V_ERROR;

if (srPtr == NULL) fprintf(logFile, "numeric: got row= NULL\n");

    if (srPtr == NULL) break;    

fprintf(logFile, "numeric: got row= %i\n", srPtr->col + 1);

    /* update new_row's values from upper triangular portion of previously
       factored row 
     */
    row = srPtr->col;

    if (IS_LOCAL(row, beg_row, end_row)) {
      row -= beg_row;

fprintf(logFile, "numeric: MERGING IN LOCAL ROW= %i\n", row+1);

      len = rp[row+1] - diag[row]; 
      cvalPtr = cval + diag[row];
      avalPtr = aval + diag[row];
    } else {
      if (externalRows == NULL) {
        len = 0;
        sprintf(msgBuf_dh, "externalRows == NULL; can't merge non-local row %i", row+1);
        SET_INFO(msgBuf_dh);
      } else {
        len = 0;
        sprintf(msgBuf_dh, "not implemented; can't merge non-local row %i\n", row+1);
        SET_INFO(msgBuf_dh);
      }
    }


    if (len) {
      /* first, form and store pivot */
      sr.col = row+beg_row;
      srPtr = SortedList_dhFind(sList, &sr); CHECK_V_ERROR;
      pc = srPtr->val;
      if (pc != 0.0) {
        pv = *avalPtr++;
        --len;
        ++cvalPtr;
        multiplier = pc / pv;
        srPtr->val = multiplier;

fprintf(logFile, "pc= %g  pv= %g  multiplier = %g\n", pc,pv, multiplier);


        /* second, update from strict upper triangular portion of row */
        while (len--) {
          sr.col = *cvalPtr++;
          srPtr = SortedList_dhFind(sList, &sr); CHECK_V_ERROR;
          if (srPtr != NULL) {
            srPtr->val -= (multiplier * *avalPtr++);
          }
        }
      }
    }    
  }
  END_FUNC_DH
}

#endif /* #if 0 */
