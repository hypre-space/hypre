/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef EUCLID_MPI_INTERFACE_DH
#define EUCLID_MPI_INTERFACE_DH

#define DEFAULT_DROP_TOL 0.01

/* #include "euclid_common.h" */

/*======================================================================
 * Naming convention: functions ending in _mpi are located in
 * src/Euclid_mpi.c; those ending in _seq are in src/Euclid_seq.c;
 * most others should be in Euclid_all.c.
 *
 * Exceptions: all Apply() (triangular solves) are in src/Euclid_apply.c;
 *             except for the Apply for MPI PILU, which is called
 *             Mat_dhSolve, and is in src/Mat_dh.c
 *
 * Users should only need to call functions with names of the form
 * Euclid_dhXXX (public functions). 
 *
 * Some of the functions whose names are of the form XXX_private_XXX,
 * as could easily be static functions; similarly, the enums and
 * structs do need to be public.  They are, primarily, for ease in
 * debugging and ready reference.
 *
 * Exceptions: the apply_private functions aren't listed here --- they're
 * all static in src/Euclid_apply.c
 *======================================================================*/

extern void Euclid_dhCreate(Euclid_dh *ctxOUT);
extern void Euclid_dhDestroy(Euclid_dh ctx);
extern void Euclid_dhSetup(Euclid_dh ctx);
extern void Euclid_dhSolve(Euclid_dh ctx, Vec_dh lhs, Vec_dh rhs, HYPRE_Int *its);
extern void Euclid_dhApply(Euclid_dh ctx, HYPRE_Real *lhs, HYPRE_Real *rhs);

extern void Euclid_dhPrintTestData(Euclid_dh ctx, FILE *fp);
extern void Euclid_dhPrintScaling(Euclid_dh ctx, FILE *fp);

extern void Euclid_dhPrintStatsShort(Euclid_dh ctx, HYPRE_Real setup, HYPRE_Real solve, FILE *fp);


extern void Euclid_dhPrintStatsShorter(Euclid_dh ctx, FILE *fp);
  /* on-line reporting, for making quick tables */

extern void Euclid_dhPrintHypreReport(Euclid_dh ctx, FILE *fp);

extern void Euclid_dhPrintStats(Euclid_dh ctx, FILE *fp);
  /* prints same info as Euclid_dhPrintParams(), but also
     prints timing information, number of iterations, etc;
     may be called after solve is completed.
   */

#ifdef HYPRE_MODE
/* is this still needed? */
extern void Euclid_dhInputHypreMat(Euclid_dh ctx, HYPRE_ParCSRMatrix A);
#endif

/*----------------------------------------------------------------------
 * Private data structures
 *----------------------------------------------------------------------*/

#define MAX_OPT_LEN 20

/* for internal timing */
#define TIMING_BINS 10
enum{ SOLVE_START_T,
      TRI_SOLVE_T,  /* triangular solves */
      SETUP_T,      /* total setup */
      SUB_GRAPH_T,  /* setup SubdomainGraph_dh */
      FACTOR_T,     /* factorization */
      SOLVE_SETUP_T, /* setup for solves */
      COMPUTE_RHO_T,
      /* note: SETUP_T - (FACTOR_T + SUB_GRAPH_T) should be small! */
      TOTAL_SOLVE_TEMP_T,
      TOTAL_SOLVE_T
    };

/* for statistical reporting */
#define STATS_BINS 10
enum{ NZA_STATS,       /* cumulative nonzeros for all systems solved */
      NZF_STATS,       /* cumulative nonzeros for all systems solved */
      NZA_USED_STATS,  /* cumulative nonzeros NOT dropped by sparseA */ 
      NZA_RATIO_STATS  /* NZA_USED_STATS/NZA_STATS, over all processors */
    };


/* primary data structure: this is monstrously long; but it works. 
   Users must ensure the following fields are initialized prior
   to calling Euclid_dhSetup(): m, n, beg_row, A
*/
struct _mpi_interface_dh {
  bool isSetup;

  HYPRE_Real rho_init;  
  HYPRE_Real rho_final;  
    /* Memory allocation for factor; will initially allocate space for 
       rho_init*nzA nonzeros; rho_final is computed after factorization,
       and is the minimum that rho_init whoulc have been to avoid
       memory reallocation; rho_final is a maximum across all processors.
    */

  HYPRE_Int m;         /* local rows in matrix */
  HYPRE_Int n;         /* global rows in matrix */
  HYPRE_Real *rhs;   /* used for debugging; this vector is not owned! */
  void *A;       /*  PETSc, HYPRE, Euclid, or other matrix object. */
  Factor_dh F;   /* data structure for the factor, F = L+U-I */
  SubdomainGraph_dh sg; 

  REAL_DH *scale;      /* row scaling vector */
  bool    isScaled;    /* set at runtime, turns scaling on or off */

  /* workspace for factorization and triangular solves */
  HYPRE_Real *work;
  HYPRE_Real *work2;
  HYPRE_Int from, to;  /* which local rows to factor or solve */

  /* runtime parameters (mostly) */
  char algo_par[MAX_OPT_LEN]; /* parallelization strategy */
  char algo_ilu[MAX_OPT_LEN]; /* ILU factorization method */
  HYPRE_Int level;      /* for ILU(k) */
  HYPRE_Real droptol;     /* for ILUT */
  HYPRE_Real sparseTolA;  /* for sparsifying A */
  HYPRE_Real sparseTolF;  /* for sparsifying the factors */
  HYPRE_Real pivotMin;    /* if pivots are <= to this value, fix 'em */
  HYPRE_Real pivotFix;    /* multiplier for adjusting small pivots */
  HYPRE_Real maxVal;      /* largest abs. value in matrix */

  /* data structures for parallel ilu (pilu) */
  SortedList_dh   slist;
  ExternalRows_dh extRows;

  /* for use with Euclid's internal krylov solvers; */
  char    krylovMethod[MAX_OPT_LEN];
  HYPRE_Int     maxIts;
  HYPRE_Real  rtol;
  HYPRE_Real  atol;
  HYPRE_Int     its; /* number of times preconditioner was applied since last call to Setup */
  HYPRE_Int     itsTotal; /* cululative number of times preconditioner was applied */

  /* internal statistics */
  HYPRE_Int setupCount;
  HYPRE_Int logging;    /* added in support of Hypre */
  HYPRE_Real timing[TIMING_BINS];
  HYPRE_Real stats[STATS_BINS];
  bool timingsWereReduced;
  bool   printStats; /* if true, on 2nd and subsequent calls to Setup,
                        calls Euclid_dhPrintStatsShorter().  Intent is to
                        print out stats for each setup phase when 
                        using Euclid, e.g, for nonlinear solves.
                     */
}; 

#endif /*  #ifndef EUCLID_MPI_INTERFACE_DH */
