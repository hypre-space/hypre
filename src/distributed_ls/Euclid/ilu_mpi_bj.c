/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_Euclid.h"

/* to do: re-integrate fix-smalll-pivots */

/* #include "ilu_dh.h" */
/* #include "Mem_dh.h" */
/* #include "Parser_dh.h" */
/* #include "Euclid_dh.h" */
/* #include "getRow_dh.h" */
/* #include "Factor_dh.h" */
/* #include "SubdomainGraph_dh.h" */

HYPRE_Int symbolic_row_private(HYPRE_Int localRow, HYPRE_Int beg_row, HYPRE_Int end_row,
                 HYPRE_Int *list, HYPRE_Int *marker, HYPRE_Int *tmpFill,
                 HYPRE_Int len, HYPRE_Int *CVAL, HYPRE_Real *AVAL,
                 HYPRE_Int *o2n_col, Euclid_dh ctx);

static HYPRE_Int numeric_row_private(HYPRE_Int localRow, HYPRE_Int beg_row, HYPRE_Int end_row,
                        HYPRE_Int len, HYPRE_Int *CVAL, HYPRE_Real *AVAL,
                        REAL_DH *work, HYPRE_Int *o2n_col, Euclid_dh ctx);


/* all non-local column indices are discarded in symbolic_row_private() */
#undef __FUNC__
#define __FUNC__ "iluk_mpi_bj"
void iluk_mpi_bj(Euclid_dh ctx)
{
  START_FUNC_DH
  HYPRE_Int      *rp, *cval, *diag;
  HYPRE_Int      *CVAL;
  HYPRE_Int      i, j, len, count, col, idx = 0;
  HYPRE_Int      *list, *marker, *fill, *tmpFill;
  HYPRE_Int      temp, m, from = ctx->from, to = ctx->to;
  HYPRE_Int      *n2o_row, *o2n_col;
  HYPRE_Int      first_row, last_row;
  HYPRE_Real   *AVAL;
  REAL_DH  *work, *aval;
  Factor_dh F = ctx->F;
  SubdomainGraph_dh sg = ctx->sg;

if (ctx->F == NULL) {
  SET_V_ERROR("ctx->F is NULL");
}
if (ctx->F->rp == NULL) {
  SET_V_ERROR("ctx->F->rp is NULL");
}

/*  printf_dh("====================== starting iluk_mpi_bj; level= %i\n\n", ctx->level);
*/

  m    = F->m;
  rp   = F->rp;
  cval = F->cval;
  fill = F->fill;
  diag = F->diag;
  aval = F->aval;
  work = ctx->work;

  n2o_row = sg->n2o_row;
  o2n_col = sg->o2n_col;

  /* allocate and initialize working space */
  list   = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  marker = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  tmpFill = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) {
    marker[i] = -1;
    work[i] = 0.0;
  }

  /*---------- main loop ----------*/

  /* global numbers of first and last locally owned rows,
     with respect to A
   */
  first_row = sg->beg_row[myid_dh];
  last_row  = first_row + sg->row_count[myid_dh];
  for (i=from; i<to; ++i) {

    HYPRE_Int row = n2o_row[i];            /* local row number */
    HYPRE_Int globalRow = row + first_row; /* global row number */

    EuclidGetRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* compute scaling value for row(i) */
    if (ctx->isScaled) {
      compute_scaling_private(i, len, AVAL, ctx); CHECK_V_ERROR;
    }

    /* Compute symbolic factor for row(i);
       this also performs sparsification
     */
    count = symbolic_row_private(i, first_row, last_row,
                                 list, marker, tmpFill,
                                 len, CVAL, AVAL,
                                 o2n_col, ctx); CHECK_V_ERROR;

    /* Ensure adequate storage; reallocate, if necessary. */
    if (idx + count > F->alloc) {
      Factor_dhReallocate(F, idx, count); CHECK_V_ERROR;
      SET_INFO("REALLOCATED from lu_mpi_bj");
      cval = F->cval;
      fill = F->fill;
      aval = F->aval;
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
     numeric_row_private(i, first_row, last_row,
                          len, CVAL, AVAL,
                          work, o2n_col, ctx); CHECK_V_ERROR
    EuclidRestoreRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* Copy factored numeric row to permanent storage,
       and re-zero work vector
     */
    for (j=rp[i]; j<rp[i+1]; ++j) {
      col = cval[j];
      aval[j] = work[col];
      work[col] = 0.0;
    }

    /* check for zero diagonal */
    if (! aval[diag[i]]) {
      hypre_sprintf(msgBuf_dh, "zero diagonal in local row %i", i+1);
      SET_V_ERROR(msgBuf_dh);
    }
  }

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
HYPRE_Int symbolic_row_private(HYPRE_Int localRow, HYPRE_Int beg_row, HYPRE_Int end_row,
                 HYPRE_Int *list, HYPRE_Int *marker, HYPRE_Int *tmpFill,
                 HYPRE_Int len, HYPRE_Int *CVAL, HYPRE_Real *AVAL,
                 HYPRE_Int *o2n_col, Euclid_dh ctx)
{
  START_FUNC_DH
  HYPRE_Int level = ctx->level, m = ctx->F->m;
  HYPRE_Int *cval = ctx->F->cval, *diag = ctx->F->diag, *rp = ctx->F->rp;
  HYPRE_Int *fill = ctx->F->fill;
  HYPRE_Int count = 0;
  HYPRE_Int j, node, tmp, col, head;
  HYPRE_Int fill1, fill2;
  HYPRE_Real val;
  HYPRE_Real thresh = ctx->sparseTolA;
  REAL_DH scale;

  scale = ctx->scale[localRow];
  ctx->stats[NZA_STATS] += (HYPRE_Real)len;

  /* Insert col indices in linked list, and values in work vector.
   * List[m] points to the first (smallest) col in the linked list.
   * Column values are adjusted from global to local numbering.
   */
  list[m] = m;
  for (j=0; j<len; ++j) {
    tmp = m;
    col = *CVAL++;
    val = *AVAL++;

    /* throw out nonlocal columns */
    if (col >= beg_row && col < end_row) {
        col -= beg_row;        /* adjust column to local zero-based */
        col = o2n_col[col];    /* permute column */
      if (hypre_abs(scale*val) > thresh || col == localRow) {  /* sparsification */
        ++count;
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
/*     ctx->symbolicZeroDiags += 1; */
    tmp = m;
    while (localRow > list[tmp]) tmp = list[tmp];
    list[localRow]    = list[tmp];
    list[tmp]    = localRow;
    tmpFill[localRow] = 0;
    marker[localRow]  = localRow;
    ++count;
  }
  ctx->stats[NZA_USED_STATS] += (HYPRE_Real)count;

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
      }
      head = list[head];  /* advance to next item in linked list */
    }
  }
  END_FUNC_VAL(count)
}


#undef __FUNC__
#define __FUNC__ "numeric_row_private"
HYPRE_Int numeric_row_private(HYPRE_Int localRow, HYPRE_Int beg_row, HYPRE_Int end_row,
                        HYPRE_Int len, HYPRE_Int *CVAL, HYPRE_Real *AVAL,
                        REAL_DH *work, HYPRE_Int *o2n_col, Euclid_dh ctx)
{
  START_FUNC_DH
  HYPRE_Real  pc, pv, multiplier;
  HYPRE_Int     j, k, col, row;
  HYPRE_Int     *rp = ctx->F->rp, *cval = ctx->F->cval;
  HYPRE_Int     *diag = ctx->F->diag;
  HYPRE_Real  val;
  REAL_DH *aval = ctx->F->aval, scale;

  scale = ctx->scale[localRow];

  /* zero work vector */
  /* note: indices in col[] are already permuted, and are
           local (zero-based)
   */
  for (j=rp[localRow]; j<rp[localRow+1]; ++j) {
    col = cval[j];
    work[col] = 0.0;
  }

  /* init work vector with values from A */
  /* (note: some values may be na due to sparsification; this is O.K.) */
  for (j=0; j<len; ++j) {
    col = *CVAL++;
    val = *AVAL++;

    if (col >= beg_row && col < end_row) {
      col -= beg_row;        /* adjust column to local zero-based */
      col = o2n_col[col];    /* we permute the indices from A */
      work[col] = val*scale;
    }
  }

  for (j=rp[localRow]; j<diag[localRow]; ++j) {
    row = cval[j];
    pc = work[row];

    if (pc != 0.0) {
      pv = aval[diag[row]];
      multiplier = pc / pv;
      work[row] = multiplier;

      for (k=diag[row]+1; k<rp[row+1]; ++k) {
        col = cval[k];
        work[col] -= (multiplier * aval[k]);
      }
    }
  }

  /* check for zero or too small of a pivot */
#if 0
  if (hypre_abs(work[i]) <= pivotTol) {
    /* yuck! assume row scaling, and just stick in a value */
    aval[diag[i]] = pivotFix;
  }
#endif

  END_FUNC_VAL(0)
}
