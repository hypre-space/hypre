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

static bool check_constraint_private(Euclid_dh ctx, HYPRE_Int b, HYPRE_Int j);

static HYPRE_Int symbolic_row_private(HYPRE_Int localRow,
                 HYPRE_Int *list, HYPRE_Int *marker, HYPRE_Int *tmpFill,
                 HYPRE_Int len, HYPRE_Int *CVAL, HYPRE_Real *AVAL,
                 HYPRE_Int *o2n_col, Euclid_dh ctx, bool debug);

static HYPRE_Int numeric_row_private(HYPRE_Int localRow,
                        HYPRE_Int len, HYPRE_Int *CVAL, HYPRE_Real *AVAL,
                        REAL_DH *work, HYPRE_Int *o2n_col, Euclid_dh ctx, bool debug);


#undef __FUNC__
#define __FUNC__ "compute_scaling_private"
void compute_scaling_private(HYPRE_Int row, HYPRE_Int len, HYPRE_Real *AVAL, Euclid_dh ctx)
{
  START_FUNC_DH
  HYPRE_Real tmp = 0.0;
  HYPRE_Int j;

  for (j=0; j<len; ++j) tmp = MAX( tmp, hypre_abs(AVAL[j]) );
  if (tmp) {
    ctx->scale[row] = 1.0/tmp;
  }
  END_FUNC_DH
}

#if 0

/* not used ? */
#undef __FUNC__
#define __FUNC__ "fixPivot_private"
HYPRE_Real fixPivot_private(HYPRE_Int row, HYPRE_Int len, float *vals)
{
  START_FUNC_DH
  HYPRE_Int i;
  float max = 0.0;
  bool debug = false;

  for (i=0; i<len; ++i) {
    float tmp = hypre_abs(vals[i]);
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
  HYPRE_Int      *rp, *cval, *diag;
  HYPRE_Int      *CVAL;
  HYPRE_Int      i, j, len, count, col, idx = 0;
  HYPRE_Int      *list, *marker, *fill, *tmpFill;
  HYPRE_Int      temp, m, from = ctx->from, to = ctx->to;
  HYPRE_Int      *n2o_row, *o2n_col, beg_row, beg_rowP;
  HYPRE_Real   *AVAL;
  REAL_DH  *work, *aval;
  Factor_dh F = ctx->F;
  SubdomainGraph_dh sg = ctx->sg;
  bool debug = false;

  if (logFile != NULL  &&  Parser_dhHasSwitch(parser_dh, "-debug_ilu")) debug = true;

  m = F->m;
  rp = F->rp;
  cval = F->cval;
  fill = F->fill;
  diag = F->diag;
  aval = F->aval;
  work = ctx->work;
  count = rp[from];

  if (sg == NULL) {
    SET_V_ERROR("subdomain graph is NULL");
  }

  n2o_row = ctx->sg->n2o_row;
  o2n_col = ctx->sg->o2n_col;
  beg_row  = ctx->sg->beg_row[myid_dh];
  beg_rowP  = ctx->sg->beg_rowP[myid_dh];

  /* allocate and initialize working space */
  list   = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  marker = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  tmpFill = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) marker[i] = -1;

  /* working space for values */
  for (i=0; i<m; ++i) work[i] = 0.0;

/*    printf_dh("====================== starting iluk_seq; level= %i\n\n", ctx->level);
*/


  /*---------- main loop ----------*/

  for (i=from; i<to; ++i) {
    HYPRE_Int row = n2o_row[i];             /* local row number */
    HYPRE_Int globalRow = row+beg_row;      /* global row number */

/*hypre_fprintf(logFile, "--------------------------------- localRow= %i\n", 1+i);
*/

    if (debug) {
	hypre_fprintf(logFile, "ILU_seq ================================= starting local row: %i, (global= %i) level= %i\n", i+1, i+1+sg->beg_rowP[myid_dh], ctx->level);
    }

    EuclidGetRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* compute scaling value for row(i) */
    if (ctx->isScaled) {
      compute_scaling_private(i, len, AVAL, ctx); CHECK_V_ERROR;
    }

    /* Compute symbolic factor for row(i);
       this also performs sparsification
     */
    count = symbolic_row_private(i, list, marker, tmpFill,
                                 len, CVAL, AVAL,
                                 o2n_col, ctx, debug); CHECK_V_ERROR;

    /* Ensure adequate storage; reallocate, if necessary. */
    if (idx + count > F->alloc) {
      Factor_dhReallocate(F, idx, count); CHECK_V_ERROR;
      SET_INFO("REALLOCATED from ilu_seq");
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
/*hypre_fprintf(logFile, "  col= %i\n", 1+col);
*/
      col = list[col];
    }

    /* add row-pointer to start of next row. */
    rp[i+1] = idx;

    /* Insert pointer to diagonal */
    temp = rp[i];
    while (cval[temp] != i) ++temp;
    diag[i] = temp;

/*hypre_fprintf(logFile, "  diag[i]= %i\n", diag);
*/

    /* compute numeric factor for current row */
     numeric_row_private(i, len, CVAL, AVAL,
                          work, o2n_col, ctx, debug); CHECK_V_ERROR
    EuclidRestoreRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* Copy factored numeric row to permanent storage,
       and re-zero work vector
     */
    if (debug) {
      hypre_fprintf(logFile, "ILU_seq:  ");
      for (j=rp[i]; j<rp[i+1]; ++j) {
        col = cval[j];
        aval[j] = work[col];
        work[col] = 0.0;
        hypre_fprintf(logFile, "%i,%i,%g ; ", 1+cval[j], fill[j], aval[j]);
        fflush(logFile);
      }
      hypre_fprintf(logFile, "\n");
    } else {
      for (j=rp[i]; j<rp[i+1]; ++j) {
        col = cval[j];
        aval[j] = work[col];
        work[col] = 0.0;
      }
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

  /* adjust column indices back to global */
  if (beg_rowP) {
    HYPRE_Int start = rp[from];
    HYPRE_Int stop = rp[to];
    for (i=start; i<stop; ++i) cval[i] += beg_rowP;
  }

  /* for debugging: this is so the Print methods will work, even if
     F hasn't been fully factored
  */
  for (i=to+1; i<m; ++i) rp[i] = 0;

  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "iluk_seq_block"
void iluk_seq_block(Euclid_dh ctx)
{
  START_FUNC_DH
  HYPRE_Int      *rp, *cval, *diag;
  HYPRE_Int      *CVAL;
  HYPRE_Int      h, i, j, len, count, col, idx = 0;
  HYPRE_Int      *list, *marker, *fill, *tmpFill;
  HYPRE_Int      temp, m;
  HYPRE_Int      *n2o_row, *o2n_col, *beg_rowP, *n2o_sub, blocks;
  HYPRE_Int      *row_count, *dummy = NULL, dummy2[1];
  HYPRE_Real   *AVAL;
  REAL_DH  *work, *aval;
  Factor_dh F = ctx->F;
  SubdomainGraph_dh sg = ctx->sg;
  bool bj = false, constrained = false;
  //HYPRE_Int discard = 0;
  HYPRE_Int gr = -1;  /* globalRow */
  bool debug = false;

  if (logFile != NULL  &&  Parser_dhHasSwitch(parser_dh, "-debug_ilu")) debug = true;

/*hypre_fprintf(stderr, "====================== starting iluk_seq_block; level= %i\n\n", ctx->level);
*/

  if (!strcmp(ctx->algo_par, "bj")) bj = true;
  constrained = ! Parser_dhHasSwitch(parser_dh, "-unconstrained");

  m    = F->m;
  rp   = F->rp;
  cval = F->cval;
  fill = F->fill;
  diag = F->diag;
  aval = F->aval;
  work = ctx->work;

  if (sg != NULL) {
    n2o_row   = sg->n2o_row;
    o2n_col   = sg->o2n_col;
    row_count = sg->row_count;
    /* beg_row   = sg->beg_row ; */
    beg_rowP  = sg->beg_rowP;
    n2o_sub   = sg->n2o_sub;
    blocks    = sg->blocks;
  }

  else {
    dummy = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
    for (i=0; i<m; ++i) dummy[i] = i;
    n2o_row   = dummy;
    o2n_col   = dummy;
    dummy2[0] = m; row_count = dummy2;
    /* beg_row   = 0; */
    beg_rowP  = dummy;
    n2o_sub   = dummy;
    blocks    = 1;
  }

  /* allocate and initialize working space */
  list   = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  marker = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  tmpFill = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) marker[i] = -1;

  /* working space for values */
  for (i=0; i<m; ++i) work[i] = 0.0;

  /*---------- main loop ----------*/

 for (h=0; h<blocks; ++h) {
  /* 1st and last row in current block, with respect to A */
  HYPRE_Int curBlock = n2o_sub[h];
  HYPRE_Int first_row = beg_rowP[curBlock];
  HYPRE_Int end_row   = first_row + row_count[curBlock];

    if (debug) {
        hypre_fprintf(logFile, "\n\nILU_seq BLOCK: %i @@@@@@@@@@@@@@@ \n", curBlock);
    }

  for (i=first_row; i<end_row; ++i) {
    HYPRE_Int row = n2o_row[i];
    ++gr;

    if (debug) {
      hypre_fprintf(logFile, "ILU_seq  global: %i  local: %i =================================\n", 1+gr, 1+i-first_row);
    }

/*prinft("first_row= %i  end_row= %i\n", first_row, end_row);
*/

    EuclidGetRow(ctx->A, row, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* compute scaling value for row(i) */
    if (ctx->isScaled) {
      compute_scaling_private(i, len, AVAL, ctx); CHECK_V_ERROR;
    }

    /* Compute symbolic factor for row(i);
       this also performs sparsification
     */
    count = symbolic_row_private(i, list, marker, tmpFill,
                                 len, CVAL, AVAL,
                                 o2n_col, ctx, debug); CHECK_V_ERROR;

    /* Ensure adequate storage; reallocate, if necessary. */
    if (idx + count > F->alloc) {
      Factor_dhReallocate(F, idx, count); CHECK_V_ERROR;
      SET_INFO("REALLOCATED from ilu_seq");
      cval = F->cval;
      fill = F->fill;
      aval = F->aval;
    }

    /* Copy factored symbolic row to permanent storage */
    col = list[m];
    while (count--) {

      /* constrained pilu */
      if (constrained && !bj) {
        if (col >= first_row && col < end_row) {
          cval[idx] = col;
          fill[idx] = tmpFill[col];
          ++idx;
        } else {
          if (check_constraint_private(ctx, curBlock, col)) {
            cval[idx] = col;
            fill[idx] = tmpFill[col];
            ++idx;
          } else {
             //++discard;
          }
        }
        col = list[col];
      }

      /* block jacobi case */
      else if (bj) {
        if (col >= first_row && col < end_row) {
          cval[idx] = col;
          fill[idx] = tmpFill[col];
          ++idx;
        } else {
           //++discard;
        }
        col = list[col];
      }

      /* general case */
      else {
        cval[idx] = col;
        fill[idx] = tmpFill[col];
        ++idx;
        col = list[col];
      }
    }

    /* add row-pointer to start of next row. */
    rp[i+1] = idx;

    /* Insert pointer to diagonal */
    temp = rp[i];
    while (cval[temp] != i) ++temp;
    diag[i] = temp;

    /* compute numeric factor for current row */
    numeric_row_private(i, len, CVAL, AVAL,
                          work, o2n_col, ctx, debug); CHECK_V_ERROR
    EuclidRestoreRow(ctx->A, row, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* Copy factored numeric row to permanent storage,
       and re-zero work vector
     */
    if (debug) {
      hypre_fprintf(logFile, "ILU_seq: ");
      for (j=rp[i]; j<rp[i+1]; ++j) {
        col = cval[j];
        aval[j] = work[col];
        work[col] = 0.0;
        hypre_fprintf(logFile, "%i,%i,%g ; ", 1+cval[j], fill[j], aval[j]);
      }
      hypre_fprintf(logFile, "\n");
     }

     /* normal operation */
     else {
      for (j=rp[i]; j<rp[i+1]; ++j) {
        col = cval[j];
        aval[j] = work[col];
        work[col] = 0.0;
      }
    }

    /* check for zero diagonal */
    if (! aval[diag[i]]) {
      hypre_sprintf(msgBuf_dh, "zero diagonal in local row %i", i+1);
      SET_V_ERROR(msgBuf_dh);
    }
  }
 }

/*  hypre_printf("bj= %i  constrained= %i  discarded= %i\n", bj, constrained, discard); */

  if (dummy != NULL) { FREE_DH(dummy); CHECK_V_ERROR; }
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
HYPRE_Int symbolic_row_private(HYPRE_Int localRow,
                 HYPRE_Int *list, HYPRE_Int *marker, HYPRE_Int *tmpFill,
                 HYPRE_Int len, HYPRE_Int *CVAL, HYPRE_Real *AVAL,
                 HYPRE_Int *o2n_col, Euclid_dh ctx, bool debug)
{
  START_FUNC_DH
  HYPRE_Int level = ctx->level, m = ctx->F->m;
  HYPRE_Int *cval = ctx->F->cval, *diag = ctx->F->diag, *rp = ctx->F->rp;
  HYPRE_Int *fill = ctx->F->fill;
  HYPRE_Int count = 0;
  HYPRE_Int j, node, tmp, col, head;
  HYPRE_Int fill1, fill2, beg_row;
  HYPRE_Real val;
  HYPRE_Real thresh = ctx->sparseTolA;
  REAL_DH scale;

  scale = ctx->scale[localRow];
  ctx->stats[NZA_STATS] += (HYPRE_Real)len;
  beg_row  = ctx->sg->beg_row[myid_dh];

  /* Insert col indices in linked list, and values in work vector.
   * List[m] points to the first (smallest) col in the linked list.
   * Column values are adjusted from global to local numbering.
   */
  list[m] = m;
  for (j=0; j<len; ++j) {
    tmp = m;
    col = *CVAL++;
    col -= beg_row;     /* adjust to zero based */
    col = o2n_col[col]; /* permute the column */
    val = *AVAL++;
    val *= scale;       /* scale the value */

    if (hypre_abs(val) > thresh || col == localRow) {  /* sparsification */
      ++count;
      while (col > list[tmp]) tmp = list[tmp];
      list[col]   = list[tmp];
      list[tmp]   = col;
      tmpFill[col] = 0;
      marker[col] = localRow;
    }
  }

  /* insert diag if not already present */
  if (marker[localRow] != localRow) {
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

      if (debug) {
        hypre_fprintf(logFile, "ILU_seq   sf updating from row: %i\n", 1+node);
      }

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
HYPRE_Int numeric_row_private(HYPRE_Int localRow,
                        HYPRE_Int len, HYPRE_Int *CVAL, HYPRE_Real *AVAL,
                        REAL_DH *work, HYPRE_Int *o2n_col, Euclid_dh ctx, bool debug)
{
  START_FUNC_DH
  HYPRE_Real  pc, pv, multiplier;
  HYPRE_Int     j, k, col, row;
  HYPRE_Int     *rp = ctx->F->rp, *cval = ctx->F->cval;
  HYPRE_Int     *diag = ctx->F->diag;
  HYPRE_Int     beg_row;
  HYPRE_Real  val;
  REAL_DH *aval = ctx->F->aval, scale;

  scale = ctx->scale[localRow];
  beg_row  = ctx->sg->beg_row[myid_dh];

  /* zero work vector */
  /* note: indices in col[] are already permuted. */
  for (j=rp[localRow]; j<rp[localRow+1]; ++j) {
    col = cval[j];
    work[col] = 0.0;
  }

  /* init work vector with values from A */
  /* (note: some values may be na due to sparsification; this is O.K.) */
  for (j=0; j<len; ++j) {
    col = *CVAL++;
    col -= beg_row;
    val = *AVAL++;
    col = o2n_col[col];  /* note: we permute the indices from A */
    work[col] = val*scale;
  }



/*hypre_fprintf(stderr, "local row= %i\n", 1+localRow);
*/


  for (j=rp[localRow]; j<diag[localRow]; ++j) {
    row = cval[j];     /* previously factored row */
    pc = work[row];


      pv = aval[diag[row]]; /* diagonal of previously factored row */

/*
if (pc == 0.0 || pv == 0.0) {
hypre_fprintf(stderr, "pv= %g; pc= %g\n", pv, pc);
}
*/

    if (pc != 0.0 && pv != 0.0) {
      multiplier = pc / pv;
      work[row] = multiplier;

      if (debug) {
        hypre_fprintf(logFile, "ILU_seq   nf updating from row: %i; multiplier= %g\n", 1+row, multiplier);
      }

      for (k=diag[row]+1; k<rp[row+1]; ++k) {
        col = cval[k];
        work[col] -= (multiplier * aval[k]);
      }
    } else  {
      if (debug) {
        hypre_fprintf(logFile, "ILU_seq   nf NO UPDATE from row %i; pc = %g; pv = %g\n", 1+row, pc, pv);
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


/*-----------------------------------------------------------------------*
 * ILUT starts here
 *-----------------------------------------------------------------------*/
HYPRE_Int ilut_row_private(HYPRE_Int localRow, HYPRE_Int *list, HYPRE_Int *o2n_col, HYPRE_Int *marker,
                     HYPRE_Int len, HYPRE_Int *CVAL, HYPRE_Real *AVAL,
                     REAL_DH *work, Euclid_dh ctx, bool debug);

#undef __FUNC__
#define __FUNC__ "ilut_seq"
void ilut_seq(Euclid_dh ctx)
{
  START_FUNC_DH
  HYPRE_Int      *rp, *cval, *diag, *CVAL;
  HYPRE_Int      i, len, count, col, idx = 0;
  HYPRE_Int      *list, *marker;
  HYPRE_Int      temp, m, from, to;
  HYPRE_Int      *n2o_row, *o2n_col, beg_row, beg_rowP;
  HYPRE_Real   *AVAL, droptol;
  REAL_DH *work, *aval, val;
  Factor_dh F = ctx->F;
  SubdomainGraph_dh sg = ctx->sg;
  bool debug = false;

  if (logFile != NULL  &&  Parser_dhHasSwitch(parser_dh, "-debug_ilu")) debug = true;

  m = F->m;
  rp = F->rp;
  cval = F->cval;
  diag = F->diag;
  aval = F->aval;
  work = ctx->work;
  from = ctx->from;
  to = ctx->to;
  count = rp[from];
  droptol = ctx->droptol;

  if (sg == NULL) {
    SET_V_ERROR("subdomain graph is NULL");
  }

  n2o_row = ctx->sg->n2o_row;
  o2n_col = ctx->sg->o2n_col;
  beg_row  = ctx->sg->beg_row[myid_dh];
  beg_rowP  = ctx->sg->beg_rowP[myid_dh];


  /* allocate and initialize working space */
  list   = (HYPRE_Int*)MALLOC_DH((m+1)*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  marker = (HYPRE_Int*)MALLOC_DH(m*sizeof(HYPRE_Int)); CHECK_V_ERROR;
  for (i=0; i<m; ++i) marker[i] = -1;
  rp[0] = 0;

  /* working space for values */
  for (i=0; i<m; ++i) work[i] = 0.0;

  /* ----- main loop start ----- */
  for (i=from; i<to; ++i) {
    HYPRE_Int row = n2o_row[i];             /* local row number */
    HYPRE_Int globalRow = row + beg_row;    /* global row number */
    EuclidGetRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* compute scaling value for row(i) */
    compute_scaling_private(i, len, AVAL, ctx); CHECK_V_ERROR;

    /* compute factor for row i */
    count = ilut_row_private(i, list, o2n_col, marker,
                         len, CVAL, AVAL, work, ctx, debug); CHECK_V_ERROR;

    EuclidRestoreRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* Ensure adequate storage; reallocate, if necessary. */
    if (idx + count > F->alloc) {
      Factor_dhReallocate(F, idx, count); CHECK_V_ERROR;
      SET_INFO("REALLOCATED from ilu_seq");
      cval = F->cval;
      aval = F->aval;
    }

    /* Copy factored row to permanent storage,
       apply 2nd drop test,
       and re-zero work vector
     */
    col = list[m];
    while (count--) {
      val = work[col];
      if (col == i || hypre_abs(val) > droptol) {
        cval[idx] = col;
        aval[idx++] = val;
        work[col] = 0.0;
      }
      col = list[col];
    }

    /* add row-pointer to start of next row. */
    rp[i+1] = idx;

    /* Insert pointer to diagonal */
    temp = rp[i];
    while (cval[temp] != i) ++temp;
    diag[i] = temp;

    /* check for zero diagonal */
    if (! aval[diag[i]]) {
      hypre_sprintf(msgBuf_dh, "zero diagonal in local row %i", i+1);
      SET_V_ERROR(msgBuf_dh);
    }
  } /* --------- main loop end --------- */

  /* adjust column indices back to global */
  if (beg_rowP) {
    HYPRE_Int start = rp[from];
    HYPRE_Int stop = rp[to];
    for (i=start; i<stop; ++i) cval[i] += beg_rowP;
  }

  FREE_DH(list);
  FREE_DH(marker);
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "ilut_row_private"
HYPRE_Int ilut_row_private(HYPRE_Int localRow, HYPRE_Int *list, HYPRE_Int *o2n_col, HYPRE_Int *marker,
                     HYPRE_Int len, HYPRE_Int *CVAL, HYPRE_Real *AVAL,
                     REAL_DH *work, Euclid_dh ctx, bool debug)
{
  HYPRE_UNUSED_VAR(debug);

  START_FUNC_DH
  Factor_dh F = ctx->F;
  HYPRE_Int     j, col, m = ctx->m, *rp = F->rp, *cval = F->cval;
  HYPRE_Int     tmp, *diag = F->diag;
  HYPRE_Int     head;
  HYPRE_Int     count = 0, beg_row;
  HYPRE_Real  val;
  HYPRE_Real  mult, *aval = F->aval;
  HYPRE_Real  scale, pv, pc;
  HYPRE_Real  droptol = ctx->droptol;
  HYPRE_Real thresh = ctx->sparseTolA;

  scale = ctx->scale[localRow];
  ctx->stats[NZA_STATS] += (HYPRE_Real)len;
  beg_row  = ctx->sg->beg_row[myid_dh];


  /* Insert col indices in linked list, and values in work vector.
   * List[m] points to the first (smallest) col in the linked list.
   * Column values are adjusted from global to local numbering.
   */
  list[m] = m;
  for (j=0; j<len; ++j) {
    tmp = m;
    col = *CVAL++;
    col -= beg_row;     /* adjust to zero based */
    col = o2n_col[col]; /* permute the column */
    val = *AVAL++;
    val *= scale;       /* scale the value */

    if (hypre_abs(val) > thresh || col == localRow) {  /* sparsification */
      ++count;
      while (col > list[tmp]) tmp = list[tmp];
      list[col]   = list[tmp];
      list[tmp]   = col;
      work[col] = val;
      marker[col] = localRow;
    }
  }

  /* insert diag if not already present */
  if (marker[localRow] != localRow) {
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
    HYPRE_Int row = list[head];

    /* get the multiplier, and apply 1st drop tolerance test */
    pc = work[row];
    if (pc != 0.0) {
      pv = aval[diag[row]];  /* diagonal (pivot) of previously factored row */
      mult = pc / pv;

      /* update localRow from previously factored "row" */
      if (hypre_abs(mult) > droptol) {
        work[row] = mult;

        for (j=diag[row]+1; j<rp[row+1]; ++j) {
          col = cval[j];
          work[col] -= (mult * aval[j]);

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
    }
    head = list[head];  /* advance to next item in linked list */
  }

  END_FUNC_VAL(count)
}


#undef __FUNC__
#define __FUNC__ "check_constraint_private"
bool check_constraint_private(Euclid_dh ctx, HYPRE_Int p1, HYPRE_Int j)
{
  START_FUNC_DH
  bool retval = false;
  HYPRE_Int i, p2;
  HYPRE_Int *nabors, count;
  SubdomainGraph_dh sg = ctx->sg;

  if (sg == NULL) {
    SET_ERROR(-1, "ctx->sg == NULL");
  }

  p2 = SubdomainGraph_dhFindOwner(ctx->sg, j, true);


  nabors = sg->adj + sg->ptrs[p1];
  count = sg->ptrs[p1+1]  - sg->ptrs[p1];

/*
hypre_printf("p1= %i, p2= %i;  p1's nabors: ", p1, p2);
for (i=0; i<count; ++i) hypre_printf("%i ", nabors[i]);
hypre_printf("\n");
*/

  for (i=0; i<count; ++i) {
/* hypre_printf("  @@@ next nabor= %i\n", nabors[i]);
*/
    if (nabors[i] == p2) {
      retval = true;
      break;
    }
  }

  END_FUNC_VAL(retval)
}
