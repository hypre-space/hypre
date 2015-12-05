/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




#include "Euclid_dh.h"
#include "Factor_dh.h"
#include "Mat_dh.h"
#include "ilu_dh.h"
#include "Mem_dh.h"
#include "Parser_dh.h"
#include "Hash_dh.h"
#include "getRow_dh.h"
#include "SortedList_dh.h"
#include "ExternalRows_dh.h"
#include "SubdomainGraph_dh.h"


static void iluk_symbolic_row_private(HYPRE_Int localRow, HYPRE_Int len, HYPRE_Int *CVAL, 
                                      double *AVAL, ExternalRows_dh extRows, 
                                      SortedList_dh sList, Euclid_dh ctx, 
                                      bool debug);

static void iluk_numeric_row_private(HYPRE_Int new_row, ExternalRows_dh extRows, 
                                      SortedList_dh slist, Euclid_dh ctx,
                                      bool debug);

#undef __FUNC__
#define __FUNC__ "iluk_mpi_pilu"
void iluk_mpi_pilu(Euclid_dh ctx)
{
  START_FUNC_DH
  HYPRE_Int from = ctx->from, to = ctx->to;
  HYPRE_Int i, m; 
  HYPRE_Int *n2o_row; /* *o2n_col; */
  HYPRE_Int *rp, *cval, *diag, *fill;
  HYPRE_Int beg_row, beg_rowP, end_rowP;
  SubdomainGraph_dh sg = ctx->sg;
  HYPRE_Int *CVAL, len, idx = 0, count;
  double *AVAL;
  REAL_DH *aval;
  Factor_dh F = ctx->F;
  SortedList_dh slist = ctx->slist;
  ExternalRows_dh extRows = ctx->extRows;
  bool bj, noValues, debug = false;

  /* for debugging */
  if (logFile != NULL && Parser_dhHasSwitch(parser_dh, "-debug_ilu")) debug = true;
  noValues = Parser_dhHasSwitch(parser_dh, "-noValues");
  bj = ctx->F->blockJacobi;

  m    = F->m;
  rp   = F->rp;
  cval = F->cval;
  fill = F->fill;
  diag = F->diag;
  aval = F->aval;
  /* work = ctx->work; */

  n2o_row = sg->n2o_row;
  /* o2n_col = sg->o2n_col; */

  if (from != 0) idx = rp[from];

  /* global numbers of first and last locally owned rows,
     with respect to A 
   */
  beg_row = sg->beg_row[myid_dh];
  /* end_row  = beg_row + sg->row_count[myid_dh]; */

  /* global number or first locally owned row, after reordering */
  beg_rowP = sg->beg_rowP[myid_dh];
  end_rowP  = beg_rowP + sg->row_count[myid_dh];


  /* loop over rows to be factored (i references local rows) */
  for (i=from; i<to; ++i) {

    HYPRE_Int row = n2o_row[i];            /* local row number */
    HYPRE_Int globalRow = row + beg_row;   /* global row number */

    if (debug) {
      hypre_fprintf(logFile, "\nILU_pilu global: %i  old_Local: %i =========================================================\n", i+1+beg_rowP, row+1);
    }

    EuclidGetRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    if (debug) {
      HYPRE_Int h;
      hypre_fprintf(logFile, "ILU_pilu  EuclidGetRow:\n");
      for (h=0; h<len; ++h) hypre_fprintf(logFile, "    %i   %g\n", 1+CVAL[h], AVAL[h]);
    }


    /* compute scaling value for row(i) */
    if (ctx->isScaled) { 
      compute_scaling_private(i, len, AVAL, ctx); CHECK_V_ERROR; 
    }

    SortedList_dhReset(slist, i); CHECK_V_ERROR;

    /* Compute symbolic factor for row(i);
       this also performs sparsification
     */
    iluk_symbolic_row_private(i, len, CVAL, AVAL, 
                              extRows, slist, ctx, debug); CHECK_V_ERROR;

    /* enforce subdomain constraint */
    SortedList_dhEnforceConstraint(slist, sg);

    /* compute numeric factor for row */
    if (! noValues) {
      iluk_numeric_row_private(i, extRows, slist, ctx, debug); CHECK_V_ERROR;
    }

    EuclidRestoreRow(ctx->A, globalRow, &len, &CVAL, &AVAL); CHECK_V_ERROR;

    /* Ensure adequate storage; reallocate, if necessary. */
    count = SortedList_dhReadCount(slist); CHECK_V_ERROR;

    /* Ensure adequate storage; reallocate, if necessary. */
    if (idx + count > F->alloc) {
      Factor_dhReallocate(F, idx, count); CHECK_V_ERROR;
      SET_INFO("REALLOCATED from ilu_mpi_pilu");
      cval = F->cval;
      fill = F->fill;
      aval = F->aval;
    }

    /* Copy factor to permanent storage */
    if (bj) {   /* for debugging: blockJacobi case */
      HYPRE_Int col;
      while (count--) {
        SRecord *sr = SortedList_dhGetSmallest(slist); CHECK_V_ERROR;
        col = sr->col;
        if (col >= beg_rowP && col < end_rowP) {
          cval[idx] = col;
          if (noValues) { aval[idx] = 0.0; }
          else          { aval[idx] = sr->val; }
          fill[idx] = sr->level;
          ++idx;
        }
      }
    } 

    if (debug) {
      hypre_fprintf(logFile, "ILU_pilu  ");
      while (count--) {
        SRecord *sr = SortedList_dhGetSmallest(slist); CHECK_V_ERROR;
        cval[idx] = sr->col;
        aval[idx] = sr->val;
        fill[idx] = sr->level;
        hypre_fprintf(logFile, "%i,%i,%g ; ", 1+cval[idx], fill[idx], aval[idx]);
        ++idx;
      }
      hypre_fprintf(logFile, "\n");
    }

    else {
      while (count--) {
        SRecord *sr = SortedList_dhGetSmallest(slist); CHECK_V_ERROR;
        cval[idx] = sr->col;
        aval[idx] = sr->val;
        fill[idx] = sr->level;
        ++idx;
      }
    }

    /* add row-pointer to start of next row. */
    rp[i+1] = idx;

    /* Insert pointer to diagonal */
    { HYPRE_Int temp = rp[i];
      bool flag = true;
      while (temp < idx) {
        if (cval[temp] == i+beg_rowP) {
          diag[i] = temp;
          flag = false;
          break;
        }
        ++temp;
      }
      if (flag) {
        if (logFile != NULL) {
          HYPRE_Int k;
          hypre_fprintf(logFile, "Failed to find diag in localRow %i (globalRow %i; ct= %i)\n   ", 
                                1+i, i+1+beg_rowP, rp[i+1] - rp[i]);
          for (k=rp[i]; k<rp[i+1]; ++k) {
            hypre_fprintf(logFile, "%i ", cval[i]+1);
          }
          hypre_fprintf(logFile, "\n\n");
        }
        hypre_sprintf(msgBuf_dh, "failed to find diagonal for localRow: %i", 1+i);
        SET_V_ERROR(msgBuf_dh);
      }
    }
/*
    { HYPRE_Int temp = rp[i]; 
      while (cval[temp] != i+beg_row) ++temp;
      diag[i] = temp;
    }
*/

    /* check for zero diagonal */
    if (! aval[diag[i]]) {
      hypre_sprintf(msgBuf_dh, "zero diagonal in local row %i", i+1);
      SET_V_ERROR(msgBuf_dh);
    }

  }

  /* adjust to local (zero) based, if block jacobi factorization */
  if (bj) {
    HYPRE_Int nz = rp[m];
    for (i=0; i<nz; ++i) cval[i] -= beg_rowP;
  }

  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "iluk_symbolic_row_private"
void iluk_symbolic_row_private(HYPRE_Int localRow, HYPRE_Int len, HYPRE_Int *CVAL, 
                               double *AVAL, ExternalRows_dh extRows, 
                               SortedList_dh slist, Euclid_dh ctx, bool debug)
{
  START_FUNC_DH
  HYPRE_Int       level = ctx->level, m = ctx->m;
  HYPRE_Int       beg_row = ctx->sg->beg_row[myid_dh];
  HYPRE_Int       beg_rowP = ctx->sg->beg_rowP[myid_dh];
  HYPRE_Int       *cval = ctx->F->cval, *diag = ctx->F->diag; 
  HYPRE_Int       *rp = ctx->F->rp, *fill = ctx->F->fill;
  HYPRE_Int       j, node, col;
  HYPRE_Int       end_rowP = beg_rowP + m;
  HYPRE_Int       level_1, level_2;
  HYPRE_Int       *cvalPtr, *fillPtr;
  SRecord   sr, *srPtr;
  REAL_DH   scale, *avalPtr;
  double    thresh = ctx->sparseTolA;
  bool      wasInserted;
  HYPRE_Int       count = 0;

  scale = ctx->scale[localRow];
  ctx->stats[NZA_STATS] += (double)len;

  /* insert col indices in sorted linked list */
  sr.level = 0;
  for (j=0; j<len; ++j) {
    sr.col = CVAL[j];
    sr.val = scale * AVAL[j];
/*    if (fabs(sr.val) > thresh) { */
      wasInserted = SortedList_dhPermuteAndInsert(slist, &sr, thresh); CHECK_V_ERROR;
      if (wasInserted) ++count;
/*    } */
    if (debug) {
      hypre_fprintf(logFile, "ILU_pilu   inserted from A: col= %i  val= %g\n",
                                        1+CVAL[j], sr.val);
    }
  }

  /* ensure diagonal entry is inserted */
  sr.val = 0.0; 
  sr.col = localRow+beg_rowP;
  srPtr = SortedList_dhFind(slist, &sr); CHECK_V_ERROR;
  if (srPtr == NULL) {
    SortedList_dhInsert(slist, &sr); CHECK_V_ERROR;
    ++count;
    if (debug) {
      hypre_fprintf(logFile, "ILU_pilu   inserted missing diagonal: %i\n", 1+localRow+beg_row);
    }
  }
  ctx->stats[NZA_USED_STATS] += (double)count;

  /* update row from previously factored rows */
  sr.val = 0.0;
  if (level > 0) {
    while(1) {
      srPtr = SortedList_dhGetSmallestLowerTri(slist); CHECK_V_ERROR;
      if (srPtr == NULL) break;

      node = srPtr->col;

        if (debug) {
          hypre_fprintf(logFile, "ILU_pilu   sf updating from row: %i\n", 1+srPtr->col);
        }

      level_1 = srPtr->level;
      if (level_1 < level) {

        /* case 1: locally owned row */
        if (node >= beg_rowP && node < end_rowP) {
          node -= beg_rowP;
          len = rp[node+1] - diag[node] - 1;
          cvalPtr = cval + diag[node] + 1;
          fillPtr = fill + diag[node] + 1;
        }

        /* case 2: external row */
        else {
          len = 0;
          ExternalRows_dhGetRow(extRows, node, &len, &cvalPtr, 
                                            &fillPtr, &avalPtr); CHECK_V_ERROR;
          if (debug && len == 0) {
            hypre_fprintf(stderr, "ILU_pilu  sf failed to get extern row: %i\n", 1+node);
          }
        }


        /* merge in strict upper triangular portion of row */
        for (j = 0; j<len; ++j) {
          col = *cvalPtr++;
          level_2 = 1+ level_1 + *fillPtr++;
          if (level_2 <= level) {
            /* Insert new element, or update level if already inserted. */
            sr.col = col;
            sr.level = level_2;
            sr.val = 0.0;
            SortedList_dhInsertOrUpdate(slist, &sr); CHECK_V_ERROR;
          }
        }
      }
    }
  }
  END_FUNC_DH
}


#undef __FUNC__
#define __FUNC__ "iluk_numeric_row_private"
void iluk_numeric_row_private(HYPRE_Int new_row, ExternalRows_dh extRows, 
                                SortedList_dh slist, Euclid_dh ctx, bool debug)
{
  START_FUNC_DH
  HYPRE_Int    m = ctx->m;
  HYPRE_Int    beg_rowP = ctx->sg->beg_rowP[myid_dh];
  HYPRE_Int    end_rowP = beg_rowP + m;
  HYPRE_Int    len, row;
  HYPRE_Int    *rp = ctx->F->rp, *cval = ctx->F->cval, *diag = ctx->F->diag;
  REAL_DH *avalPtr, *aval = ctx->F->aval;
  HYPRE_Int     *cvalPtr;
  double  multiplier, pc, pv;
  SRecord sr, *srPtr;

  /* note: non-zero entries from A were inserted in list during iluk_symbolic_row_private */

  SortedList_dhResetGetSmallest(slist); CHECK_V_ERROR;
  while (1) {
    srPtr = SortedList_dhGetSmallestLowerTri(slist); CHECK_V_ERROR;
    if (srPtr == NULL) break;

    /* update new_row's values from upper triangular portion of previously
       factored row
     */
    row = srPtr->col;

    if (row >= beg_rowP && row < end_rowP) {
      HYPRE_Int local_row = row - beg_rowP;

      len = rp[local_row+1] - diag[local_row];
      cvalPtr = cval + diag[local_row];
      avalPtr = aval + diag[local_row]; 
    } else {
      len = 0;
      ExternalRows_dhGetRow(extRows, row, &len, &cvalPtr, 
                                            NULL, &avalPtr); CHECK_V_ERROR;
      if (debug && len == 0) {
        hypre_fprintf(stderr, "ILU_pilu  failed to get extern row: %i\n", 1+row);
      }

    }

    if (len) {
      /* first, form and store pivot */
      sr.col = row;
      srPtr = SortedList_dhFind(slist, &sr); CHECK_V_ERROR;
      if (srPtr == NULL) {
        hypre_sprintf(msgBuf_dh, "find failed for sr.col = %i while factoring local row= %i \n", 1+sr.col, new_row+1);
        SET_V_ERROR(msgBuf_dh);
      }

      pc = srPtr->val;

      if (pc != 0.0) {
        pv = *avalPtr++; 
        --len;
        ++cvalPtr;
        multiplier = pc / pv;
        srPtr->val = multiplier;

        if (debug) {
          hypre_fprintf(logFile, "ILU_pilu   nf updating from row: %i; multiplier = %g\n", 1+srPtr->col, multiplier);
        }

        /* second, update from strict upper triangular portion of row */
        while (len--) {
          sr.col = *cvalPtr++;
          sr.val = *avalPtr++;
          srPtr = SortedList_dhFind(slist, &sr); CHECK_V_ERROR;
          if (srPtr != NULL) {
            srPtr->val -= (multiplier * sr.val);
          }
        }
      }

       else {
        if (debug) {
          hypre_fprintf(logFile, "ILU_pilu   NO UPDATE from row: %i; srPtr->val = 0.0\n", 1+srPtr->col);
        }
       }

    }
  }
  END_FUNC_DH
}
