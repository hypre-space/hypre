/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef FACTOR_DH
#define FACTOR_DH

/* #include "euclid_common.h" */

struct _factor_dh {
  /* dimensions of local rectangular submatrix; global matrix is n*n */
  HYPRE_Int m, n;    

  HYPRE_Int id;          /* this subdomain's id after reordering */
  HYPRE_Int beg_row;     /* global number of 1st locally owned row */
  HYPRE_Int first_bdry;  /* local number of first boundary row */
  HYPRE_Int bdry_count;  /* m - first_boundary */

  /* if true, factorization was block jacobi, in which case all
     column indices are zero-based; else, they are global.
  */
  bool blockJacobi;

  /* sparse row-oriented storage for locally owned submatrix */
  HYPRE_Int *rp;       
  HYPRE_Int *cval;
  REAL_DH *aval;
  HYPRE_Int *fill;
  HYPRE_Int *diag;
  HYPRE_Int alloc; /* currently allocated length of cval, aval, and fill arrays */

  /* used for PILU solves (Apply) */
  HYPRE_Int          num_recvLo, num_recvHi;
  HYPRE_Int          num_sendLo, num_sendHi;  /* used in destructor */
  HYPRE_Real   *work_y_lo;  /* recv values from lower nabors; also used as
                               work vector when solving Ly=b for y.
                            */
  HYPRE_Real   *work_x_hi;  /* recv values from higher nabors; also used as
                               work vector when solving Ux=y for x.
                            */
  HYPRE_Real   *sendbufLo, *sendbufHi;
  HYPRE_Int          *sendindLo, *sendindHi;
  HYPRE_Int          sendlenLo, sendlenHi;
  bool         solveIsSetup;
  Numbering_dh numbSolve;

  hypre_MPI_Request  recv_reqLo[MAX_MPI_TASKS], recv_reqHi[MAX_MPI_TASKS]; /* used for persistent comms */
  hypre_MPI_Request  send_reqLo[MAX_MPI_TASKS], send_reqHi[MAX_MPI_TASKS]; /* used for persistent comms */
  hypre_MPI_Request  requests[MAX_MPI_TASKS];
  hypre_MPI_Status   status[MAX_MPI_TASKS];  

  bool debug;
};

extern void Factor_dhCreate(Factor_dh *mat);
extern void Factor_dhDestroy(Factor_dh mat);

extern void Factor_dhTranspose(Factor_dh matIN, Factor_dh *matOUT);

extern void Factor_dhInit(void *A, bool fillFlag, bool avalFlag,
                          HYPRE_Real rho, HYPRE_Int id, HYPRE_Int beg_rowP, Factor_dh *F);

extern void Factor_dhReallocate(Factor_dh F, HYPRE_Int used, HYPRE_Int additional);
  /* ensures fill, cval, and aval arrays can accomodate
     at least "c" additional entrie
   */

  /* adopted from ParaSails, by Edmond Chow */
extern void Factor_dhSolveSetup(Factor_dh mat, SubdomainGraph_dh sg);


extern void Factor_dhSolve(HYPRE_Real *rhs, HYPRE_Real *lhs, Euclid_dh ctx);
extern void Factor_dhSolveSeq(HYPRE_Real *rhs, HYPRE_Real *lhs, Euclid_dh ctx);

  /* functions for monitoring stability */
extern HYPRE_Real Factor_dhCondEst(Factor_dh mat, Euclid_dh ctx);
extern HYPRE_Real Factor_dhMaxValue(Factor_dh mat);
extern HYPRE_Real Factor_dhMaxPivotInverse(Factor_dh mat);

extern HYPRE_Int Factor_dhReadNz(Factor_dh mat);
extern void Factor_dhPrintTriples(Factor_dh mat, char *filename);

extern void Factor_dhPrintGraph(Factor_dh mat, char *filename);
 /* seq only */


extern void Factor_dhPrintDiags(Factor_dh mat, FILE *fp);
extern void Factor_dhPrintRows(Factor_dh mat, FILE *fp);
  /* prints local matrix to logfile, if open */

#endif
