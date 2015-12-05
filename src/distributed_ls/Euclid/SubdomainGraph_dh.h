/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




#ifndef SUBDOMAIN_GRAPH_DH
#define SUBDOMAIN_GRAPH_DH

#include "euclid_common.h"

#define MAX_SUBDOMAIN_COLOR 100
  /* could be done better: if we can't color the subdomain graph
     with this many colors, an error is thrown in SubdomainGraph_dhColor().
   */

/* for internal timing */
#define TIMING_BINS_SG 10
enum{ TOTAL_SGT,  /* total Init (setup) time */
      FIND_NABORS_SGT,
      ORDER_BDRY_SGT,
      FORM_GRAPH_SGT,
      EXCHANGE_PERMS_SGT
    };

struct _subdomain_dh {
  HYPRE_Int blocks;          /* number of subdomains */
  HYPRE_Int *ptrs, *adj;     /* csr structure for representing subdomain graph */
  HYPRE_Int *o2n_sub;        /* subdomain graph permutation; */
  HYPRE_Int *n2o_sub;        /* inverse permutation; */
  HYPRE_Int colors;    /* number of colors used  for coloring the subdomain graph */
  bool doNotColor; /* if true, subdomain graph is not colored and reordered */
  HYPRE_Int *colorVec;  /* if colorVec[i] = x, then subdomain i was colored "x".
                     this array is probably only useful for debugging.
                   */

  HYPRE_Int *beg_row;   /* global ordering of first local row owned by P_i */
  HYPRE_Int *beg_rowP;  /* global ordering of first local row owned by P_i after
                     subdomain reordering 
                   */
  HYPRE_Int *row_count; /* P_i owns row_count[i] local rows */
  HYPRE_Int *bdry_count; /* bdry_count[i] of P_i's rows are boundary rows */

  /* Nearest neighbors in subdomain graph, before reordering;
     "self" is not included.  Not used for sequential case.
   */
  HYPRE_Int *loNabors, loCount;
  HYPRE_Int *hiNabors, hiCount;
  HYPRE_Int *allNabors, allCount;


  /* permutation information for global unknowns (matrix rows) */
  HYPRE_Int m;               /* length of n2o_row and o2n_col */
  HYPRE_Int *n2o_row;        /* permutation for locally owned matrix rows */
  HYPRE_Int *o2n_col;        /* permutation for locally owned matrix columns */

  Hash_i_dh o2n_ext;   /* permutation for external columns */
  Hash_i_dh n2o_ext;   /* inverse permutation for external columns */

  double timing[TIMING_BINS_SG];
  bool debug;
};

extern void SubdomainGraph_dhCreate(SubdomainGraph_dh *s);
extern void SubdomainGraph_dhDestroy(SubdomainGraph_dh s);

extern void SubdomainGraph_dhInit(SubdomainGraph_dh s, HYPRE_Int blocks, bool bj, void *A);
  /* Partitions matrix A into the specified number of blocks,
     if there is a single MPI task; for mpi use, "blocks" must be the same 
     as the number of mpi tasks; for sequential, it may vary.
     On completion, the subdomain graph will be fully formed,
     (all fields valid); o2n_row[] and n2o_col[] will be permutations
     for the locally owned portion of A such that A's interior nodes are
     ordered first.
     This function may call a partitioner, such as METIS (currently, only for sequential).
     On completion, "o2n" contains a natural ordering, beg_row is identical to
     beg_rowP, and_rowP is identical to end_rowP.

     if "bj" is true, the following setup steps are NOT performed:
     form subdomain graph; find neighbors; order boundary nodes
  */

extern void SubdomainGraph_dhColor(SubdomainGraph_dh s);
  /*
     Colors and orders subdomain graph; on completion, o2n[], beg_rowP[], and
     end_rowP[] may be altered.
  */

extern HYPRE_Int SubdomainGraph_dhFindOwner(SubdomainGraph_dh s, HYPRE_Int idx, bool permuted);
  /* Returns the subdomain block to which row idx belongs, or throws an error.
     If "permuted" is true, it's assumed the graph has been permuted (i.e.,
     'globally reordering phase' in PILU algorithm).
  */

extern void SubdomainGraph_dhExchangePerms(SubdomainGraph_dh s);
  /*
     exchange permutation information for external columns with nearest neighbors;
     caller must ensure SubdomainGraph_dhInit() has completed before calling.
  */

extern void SubdomainGraph_dhPrintSubdomainGraph(SubdomainGraph_dh s, FILE *fp);

extern void SubdomainGraph_dhPrintStatsLong(SubdomainGraph_dh s, FILE *fp);
  /* similar to Short, but prints complete list of interior/bdry node ratios;
     also prints subdomain permutation
   */

extern void SubdomainGraph_dhDump(SubdomainGraph_dh s, char *filename);
  /* for testing */

extern void SubdomainGraph_dhPrintRatios(SubdomainGraph_dh s, FILE *fp);
  /* prints ratios of interior/boundary node for all subdomains */


extern void SubdomainGraph_dhPrintStats(SubdomainGraph_dh sg, FILE *fp);

#endif
