
/*
 * -- Distributed symbolic factorization auxialiary routine  (version 2.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley - July 2003
 * INRIA France - January 2004
 * Laura Grigori
 *
 * November 1, 2007
 */

/* limits.h:  the largest positive integer (INT_MAX) */
#include <limits.h>
#include <math.h>
#include "superlu_ddefs.h"

/*
 * Internal protypes
 */

static float
a_plus_at_CompRow_loc
(int, int_t *, int, int_t *, int_t , int_t *, int_t *,  
 int, int_t *, int_t *, int_t **,  int_t **, gridinfo_t *);

float
get_perm_c_parmetis (SuperMatrix *A, int_t *perm_r, int_t *perm_c,
		     int nprocs_i, int noDomains, 
		     int_t **sizes, int_t **fstVtxSep,
		     gridinfo_t *grid, MPI_Comm *metis_comm)
/*
 * Purpose
 * =======
 *
 * GET_PERM_C_PARMETIS obtains a permutation matrix Pc, by applying a
 * graph partitioning algorithm to the symmetrized graph A+A'.  The
 * multilevel graph partitioning algorithm used is the
 * ParMETIS_V3_NodeND routine available in the parallel graph
 * partitioning package parMETIS.  
 *
 * The number of independent sub-domains noDomains computed by this
 * algorithm has to be a power of 2.  Hence noDomains is the larger
 * number power of 2 that is smaller than nprocs_i, where nprocs_i = nprow
 * * npcol is the number of processors used in SuperLU_DIST.
 *
 * Arguments
 * =========
 *
 * A       (input) SuperMatrix*
 *         Matrix A in A*X=B, of dimension (A->nrow, A->ncol). The number
 *         of the linear equations is A->nrow.  Matrix A is distributed
 *         in NRformat_loc format.
 *
 * perm_r  (input) int_t*
 *         Row permutation vector of size A->nrow, which defines the 
 *         permutation matrix Pr; perm_r[i] = j means row i of A is in 
 *         position j in Pr*A.
 *
 * perm_c  (output) int_t*
 *	   Column permutation vector of size A->ncol, which defines the 
 *         permutation matrix Pc; perm_c[i] = j means column i of A is 
 *         in position j in A*Pc.
 *
 * nprocs_i (input) int*
 *         Number of processors the input matrix is distributed on in a block
 *         row format.  It corresponds to number of processors used in
 *         SuperLU_DIST.
 *
 * noDomains (input) int*, must be power of 2
 *         Number of independent domains to be computed by the graph
 *         partitioning algorithm.  ( noDomains <= nprocs_i )
 *
 * sizes   (output) int_t**, of size 2 * noDomains
 *         Returns pointer to an array containing the number of nodes
 *         for each sub-domain and each separator.  Separators are stored 
 *         from left to right.
 *         Memory for the array is allocated in this routine.
 *
 * fstVtxSep (output) int_t**, of size 2 * noDomains
 *         Returns pointer to an array containing first node for each
 *         sub-domain and each separator.
 *         Memory for the array is allocated in this routine.
 *
 * Return value
 * ============
 *   < 0, number of bytes allocated on return from the symbolic factorization.
 *   > 0, number of bytes allocated when out of memory.
 *
 */
{
  NRformat_loc *Astore;
  int   iam, p;
  int   *b_rowptr_int, *b_colind_int, *l_sizes_int, *dist_order_int, *vtxdist_o_int;
  int   *options, numflag;
  int_t m_loc, nnz_loc, fst_row;
  int_t m, n, bnz, i, j;
  int_t *rowptr, *colind, *l_fstVtxSep, *l_sizes;
  int_t *b_rowptr, *b_colind;
  int_t *dist_order;
  int  *recvcnts, *displs;
  /* first row index on each processor when the matrix is distributed
     on nprocs (vtxdist_i) or noDomains processors (vtxdist_o) */
  int_t  *vtxdist_i, *vtxdist_o; 
  int_t szSep, k, noNodes;
  float apat_mem_l; /* memory used during the computation of the graph of A+A' */
  float mem;  /* Memory used during this routine */
  MPI_Status status;

  /* Initialization. */
  MPI_Comm_rank (grid->comm, &iam);
  n = A->ncol;
  m = A->nrow;
  if ( m != n ) ABORT("Matrix is not square");
  mem = 0.;

#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Enter get_perm_c_parmetis()");
#endif

  Astore = (NRformat_loc *) A->Store;
  nnz_loc = Astore->nnz_loc; /* number of nonzeros in the local submatrix */
  m_loc = Astore->m_loc;     /* number of rows local to this processor */
  fst_row = Astore->fst_row; /* global index of the first row */
  rowptr = Astore->rowptr;   /* pointer to rows and column indices */
  colind = Astore->colind;
  
#if ( PRNTlevel>=1 )
  if ( !iam ) printf(".. Use parMETIS ordering on A'+A with %d sub-domains.\n",
		     noDomains);
#endif

  numflag = 0;
  /* determine first row on each processor */
  vtxdist_i = (int_t *) SUPERLU_MALLOC((nprocs_i+1) * sizeof(int_t));
  if ( !vtxdist_i ) ABORT("SUPERLU_MALLOC fails for vtxdist_i.");
  vtxdist_o = (int_t *) SUPERLU_MALLOC((nprocs_i+1) * sizeof(int_t));
  if ( !vtxdist_o ) ABORT("SUPERLU_MALLOC fails for vtxdist_o.");

  MPI_Allgather (&fst_row, 1, mpi_int_t, vtxdist_i, 1, mpi_int_t,
		 grid->comm);
  vtxdist_i[nprocs_i] = m;

  if (noDomains == nprocs_i) {
    /* keep the same distribution of A */
    for (p = 0; p <= nprocs_i; p++)
      vtxdist_o[p] = vtxdist_i[p];
  }
  else {
    i = n / noDomains;
    j = n % noDomains;
    for (k = 0, p = 0; p < noDomains; p++) {
      vtxdist_o[p] = k;
      k += i;
      if (p < j)  k++;
    }
    /* The remaining non-participating processors get the same 
       first-row-number as the last processor.   */
    for (p = noDomains; p <= nprocs_i; p++)
      vtxdist_o[p] = k;
  }

#if ( DEBUGlevel>=2 )
  if (!iam)
    PrintInt10 ("vtxdist_o", nprocs_i + 1, vtxdist_o);
#endif  

  /* Compute distributed A + A' */
  if ((apat_mem_l = 
       a_plus_at_CompRow_loc(iam, perm_r, nprocs_i, vtxdist_i,
			     n, rowptr, colind, noDomains, vtxdist_o,
			     &bnz, &b_rowptr, &b_colind, grid)) > 0)
    return (apat_mem_l);
  mem += -apat_mem_l;
  
  /* Initialize and allocate storage for parMetis. */    
  (*sizes) = (int_t *) SUPERLU_MALLOC(2 * noDomains * sizeof(int_t));
  if (!(*sizes)) ABORT("SUPERLU_MALLOC fails for sizes.");
  l_sizes = *sizes;
  (*fstVtxSep) = (int_t *) SUPERLU_MALLOC(2 * noDomains * sizeof(int_t));
  if (!(*fstVtxSep)) ABORT("SUPERLU_MALLOC fails for fstVtxSep.");
  l_fstVtxSep = *fstVtxSep;
  m_loc = vtxdist_o[iam+1] - vtxdist_o[iam];
  
  if ( iam < noDomains) 
    /* dist_order_int is the perm returned by parMetis, distributed */
    if (! (dist_order_int = (int *) SUPERLU_MALLOC(m_loc * sizeof(int))))
      ABORT("SUPERLU_MALLOC fails for dist_order_int.");

  /* ParMETIS represents the column pointers and row indices of *
   * the input matrix using integers. When SuperLU_DIST uses    *
   * long int for the int_t type, then several supplementary    *
   * copies need to be performed in order to call ParMETIS.     */
#if defined (_LONGINT)
  l_sizes_int = (int *) SUPERLU_MALLOC(2 * noDomains * sizeof(int));
  if (!(l_sizes_int)) ABORT("SUPERLU_MALLOC fails for l_sizes_int.");
  
  /* Allocate storage */
  if ( !(b_rowptr_int = (int*) SUPERLU_MALLOC((m_loc+1) * sizeof(int))))
    ABORT("SUPERLU_MALLOC fails for b_rowptr_int[]");
  for (i = 0; i <= m_loc; i++)
    b_rowptr_int[i] = b_rowptr[i];
  SUPERLU_FREE (b_rowptr);
  
  if ( bnz ) {
    if ( !(b_colind_int = (int *) SUPERLU_MALLOC( bnz * sizeof(int))))
      ABORT("SUPERLU_MALLOC fails for b_colind_int[]");
    for (i = 0; i < bnz; i++)
      b_colind_int[i] = b_colind[i];
    SUPERLU_FREE (b_colind);
  }
  
  if ( !(vtxdist_o_int = 
	 (int *) SUPERLU_MALLOC((nprocs_i+1) * sizeof(int))))
    ABORT("SUPERLU_MALLOC fails for vtxdist_o_int.");
  for (i = 0; i <= nprocs_i; i++)
    vtxdist_o_int[i] = vtxdist_o[i];
  SUPERLU_FREE (vtxdist_o);

#else  /* Default */

  vtxdist_o_int = vtxdist_o;
  b_rowptr_int = b_rowptr; b_colind_int = b_colind;
  l_sizes_int = l_sizes;

#endif
    
  if ( iam < noDomains) {
    options = (int *) SUPERLU_MALLOC(4 * sizeof(int));
    options[0] = 0;
    options[1] = 0;
    options[2] = 0;
    options[3] = 1;

#if 0
    ParMETIS_V3_NodeND(vtxdist_o_int, b_rowptr_int, b_colind_int, 
		       &numflag, options,
		       dist_order_int, l_sizes_int, metis_comm);
#endif
  }
  
  if (bnz) 
    SUPERLU_FREE (b_colind_int);
  if ( iam < noDomains) {
    SUPERLU_FREE (options);
  }
  SUPERLU_FREE (b_rowptr_int);
  
#if defined (_LONGINT)
  /* Copy data from dist_order_int to dist_order */
  if ( iam < noDomains) {
    /* dist_order is the perm returned by parMetis, distributed */
    if (!(dist_order = (int_t *) SUPERLU_MALLOC(m_loc * sizeof(int_t))))
      ABORT("SUPERLU_MALLOC fails for dist_order.");
    for (i = 0; i < m_loc; i++)
      dist_order[i] = dist_order_int[i];
    SUPERLU_FREE(dist_order_int);
    
    for (i = 0; i < 2*noDomains; i++)
      l_sizes[i] = l_sizes_int[i];
    SUPERLU_FREE(l_sizes_int);
  }
#else 
  dist_order = dist_order_int;
#endif
  
  /* Allgatherv dist_order to get perm_c */
  if (!(displs = (int *) SUPERLU_MALLOC (nprocs_i * sizeof(int))))
    ABORT ("SUPERLU_MALLOC fails for displs.");
  if ( !(recvcnts = (int *) SUPERLU_MALLOC (nprocs_i * sizeof(int))))
    ABORT ("SUPERLU_MALLOC fails for recvcnts.");
  for (i = 0; i < nprocs_i; i++)
    recvcnts[i] = vtxdist_o_int[i+1] - vtxdist_o_int[i];
  displs[0]=0;
  for(i=1; i < nprocs_i; i++) 
    displs[i] = displs[i-1] + recvcnts[i-1];
  
  MPI_Allgatherv (dist_order, m_loc, mpi_int_t, perm_c, recvcnts, displs, 
		  mpi_int_t, grid->comm);

  if ( iam < noDomains) {
    SUPERLU_FREE (dist_order);
  }
  SUPERLU_FREE (vtxdist_i);
  SUPERLU_FREE (vtxdist_o_int);
  SUPERLU_FREE (recvcnts);
  SUPERLU_FREE (displs);
  
  /* send l_sizes to every processor p >= noDomains */
  if (!iam)
    for (p = noDomains; p < nprocs_i; p++)
      MPI_Send (l_sizes, 2*noDomains, mpi_int_t, p, 0, grid->comm);
  if (noDomains <= iam && iam < nprocs_i)
    MPI_Recv (l_sizes, 2*noDomains, mpi_int_t, 0, 0, grid->comm,
	      &status);
  
  /* Determine the first node in each separator, store it in l_fstVtxSep */  
  for (j = 0; j < 2 * noDomains; j++)
    l_fstVtxSep[j] = 0;
  l_fstVtxSep[2*noDomains - 2] = l_sizes[2*noDomains - 2];
  szSep = noDomains;
  i = 0;
  while (szSep != 1) {
    for (j = i; j < i + szSep; j++) {
      l_fstVtxSep[j] += l_sizes[j]; 	      
    }
    for (j = i; j < i + szSep; j++) {
      k = i + szSep + (j-i) / 2;
      l_fstVtxSep[k] += l_fstVtxSep[j]; 
    }
    i += szSep;
    szSep = szSep / 2;
  }
  
  l_fstVtxSep[2 * noDomains - 2] -= l_sizes[2 * noDomains - 2];
  i = 2 * noDomains - 2;
  szSep = 1;
  while (i > 0) {
    for (j = i; j < i + szSep; j++) {
      k = (i - 2 * szSep) + (j-i) * 2 + 1;
      noNodes = l_fstVtxSep[k];
      l_fstVtxSep[k] = l_fstVtxSep[j] - l_sizes[k];
      l_fstVtxSep[k-1] = l_fstVtxSep[k] + l_sizes[k] - 
	noNodes - l_sizes[k-1];
    }
    szSep *= 2;
    i -= szSep;
  }

#if ( PRNTlevel>=2 )
  if (!iam ) {
    PrintInt10 ("Sizes of separators", 2 * noDomains-1, l_sizes);
    PrintInt10 ("First Vertex Separator", 2 * noDomains-1, l_fstVtxSep);
  }
#endif

#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Exit get_perm_c_parmetis()");
#endif
  
  return (-mem);

} /* get_perm_c_parmetis */


static float
a_plus_at_CompRow_loc
(
 int   iam,         /* Input - my processor number */
 int_t *perm_r,     /* Input - row permutation vector Pr */
 int   nprocs_i,    /* Input - number of processors the input matrix
		       is distributed on */
 int_t *vtxdist_i,  /* Input - index of first row on each processor of the input matrix */
 int_t n,           /* Input - number of columns in matrix A. */
 int_t *rowptr,     /* Input - row pointers of size m_loc+1 for matrix A. */
 int_t *colind,     /* Input - column indices of size nnz_loc for matrix A. */
 int   nprocs_o,    /* Input - number of processors the output matrix
		       is distributed on */
 int_t *vtxdist_o,  /* Input - index of first row on each processor of the output matrix */
 int_t *p_bnz,      /* Output - on exit, returns the actual number of
		       local nonzeros in matrix A'+A. */
 int_t **p_b_rowptr, /* Output - output matrix, row pointers of size m_loc+1 */
 int_t **p_b_colind, /* Output - output matrix, column indices of size *p_bnz */
 gridinfo_t *grid    /* Input - grid of processors information */
 )
{
/*
 * Purpose
 * =======
 *
 * Form the structure of Pr*A +A'Pr'. A is an n-by-n matrix in
 * NRformat_loc format, represented by (rowptr, colind). The output
 * B=Pr*A +A'Pr' is in NRformat_loc format (symmetrically, also row
 * oriented), represented by (b_rowptr, b_colind).
 *
 * The input matrix A is distributed in block row format on nprocs_i
 * processors.  The output matrix B is distributed in block row format
 * on nprocs_o processors, where nprocs_o <= nprocs_i.  On output, the
 * matrix B has its rows permuted according to perm_r.
 *
 * Sketch of the algorithm
 * =======================
 *
 * Let iam by my process number.  Let fst_row, lst_row = m_loc +
 * fst_row be the first/last row stored on iam.
 * 
 * Compute Pr' - the inverse row permutation, stored in iperm_r.
 *
 * Compute the transpose  of the block row of Pr*A that iam owns:
 *    T[:,Pr(fst_row:lst_row)] = Pr' * A[:,fst_row:lst_row] * Pr'
 *
 *
 * All to all communication such that every processor iam receives all
 * the blocks of the transpose matrix that it needs, that is
 *           T[fst_row:lst_row, :]
 *
 * Compute B = A[fst_row:lst_row, :] + T[fst_row:lst_row, :]
 *
 * If Pr != I or nprocs_i != nprocs_o then permute the rows of B (that
 * is compute Pr*B) and redistribute from nprocs_i to nprocs_o
 * according to the block row distribution in vtxdist_i, vtxdist_o.
 */
  
  int_t i, j, k, col, num_nz, nprocs;
  int_t *tcolind_recv; /* temporary receive buffer */
  int_t *tcolind_send; /* temporary send buffer */
  int_t sz_tcolind_send, sz_tcolind_loc, sz_tcolind_recv;
  int_t ind, ind_tmp, ind_rcv;
  int redist_pra; /* TRUE if Pr != I or nprocs_i != nprocs_o */
  int_t *marker, *iperm_r;
  int_t *sendCnts, *recvCnts;
  int_t *sdispls, *rdispls;
  int_t bnz, *b_rowptr, *b_colind, bnz_t, *b_rowptr_t, *b_colind_t;
  int_t p, t_ind, nelts, ipcol;
  int_t m_loc, m_loc_o;      /* number of local rows */ 
  int_t fst_row, fst_row_o;  /* index of first local row */
  int_t nnz_loc;    /* number of local nonzeros in matrix A */
  float apat_mem, apat_mem_max;
  int   *intBuf1, *intBuf2, *intBuf3, *intBuf4;  

#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Enter a_plus_at_CompRow_loc()");
#endif
  
  fst_row    = vtxdist_i[iam];
  m_loc      = vtxdist_i[iam+1] - vtxdist_i[iam];
  nnz_loc    = rowptr[m_loc];
  redist_pra = FALSE;  
  nprocs     = SUPERLU_MAX(nprocs_i, nprocs_o);
  apat_mem_max = 0.;
  
  if (!(marker = (int_t*) SUPERLU_MALLOC( (n+1) * sizeof(int_t))))
    ABORT("SUPERLU_MALLOC fails for marker[]");
  if (!(iperm_r = (int_t*) SUPERLU_MALLOC( n * sizeof(int_t))))
    ABORT("SUPERLU_MALLOC fails for iperm_r[]");
  if (!(sendCnts = (int_t*) SUPERLU_MALLOC(nprocs * sizeof(int_t))))
    ABORT("SUPERLU_MALLOC fails for sendCnts[]");
  if (!(recvCnts = (int_t*) SUPERLU_MALLOC(nprocs * sizeof(int_t))))
    ABORT("SUPERLU_MALLOC fails for recvCnts[]");
  if (!(sdispls = (int_t*) SUPERLU_MALLOC((nprocs+1) * sizeof(int_t))))
    ABORT("SUPERLU_MALLOC fails for sdispls[]");
  if (!(rdispls = (int_t*) SUPERLU_MALLOC((nprocs+1) * sizeof(int_t))))
    ABORT("SUPERLU_MALLOC fails for rdispls[]");
  apat_mem = 2 * n + 4 * nprocs + 3;

#if defined (_LONGINT)
  intBuf1 = (int *) SUPERLU_MALLOC(4 * nprocs * sizeof(int));
  intBuf2 = intBuf1 + nprocs;
  intBuf3 = intBuf1 + 2 * nprocs;
  intBuf4 = intBuf1 + 3 * nprocs;
  apat_mem += 4*nprocs*sizeof(int) / sizeof(int_t);
#endif  

  /* compute the inverse row permutation vector */
  for (i = 0; i < n; i++) {
    marker[i] = 1;
    if (perm_r[i] != i)
      redist_pra = TRUE;
    iperm_r[perm_r[i]] = i;
  }

  /* TRANSPOSE LOCAL ROWS ON MY PROCESSOR iam.         */
  /* THE RESULT IS STORED IN TCOLIND_SEND.             */
  /* THIS COUNTS FOR TWO PASSES OF THE LOCAL MATRIX.   */

  /* First pass to get counts of each row of T, and set up column pointers */
  for (j = 0; j < m_loc; j++) {
    for (i = rowptr[j]; i < rowptr[j+1]; i++){
      marker[iperm_r[colind[i]]]++;
    }
  }
  /* determine number of elements to be sent to each processor */
  for (p = 0; p < nprocs_i; p++) {
    sendCnts[p] = 0;
    for (i = vtxdist_i[p]; i < vtxdist_i[p+1]; i++) 
      sendCnts[p] += marker[i];
  }
  /* exchange send/receive counts information in between all processors */
  MPI_Alltoall (sendCnts, 1, mpi_int_t,
		recvCnts, 1, mpi_int_t, grid->comm);
  sendCnts[iam] = 0;
  sz_tcolind_loc = recvCnts[iam];
  
  for (i = 0, j = 0, p = 0; p < nprocs_i; p++) {
    rdispls[p] = j;
    j += recvCnts[p];
    sdispls[p] = i;  
    i += sendCnts[p];
  }
  recvCnts[iam] = 0;
  sz_tcolind_recv = j;
  sz_tcolind_send = i;
  
  /* allocate memory to receive necessary blocks of transpose matrix T */
  if (sz_tcolind_recv) {
    if ( !(tcolind_recv = (int_t*) SUPERLU_MALLOC( sz_tcolind_recv 
						   * sizeof(int_t) )))
      ABORT("SUPERLU_MALLOC fails tcolind_recv[]");
    apat_mem += sz_tcolind_recv;
  }
  /* allocate memory to send blocks of local transpose matrix T to other processors */
  if (sz_tcolind_send) {
    if (!(tcolind_send = (int_t*) SUPERLU_MALLOC( (sz_tcolind_send) 
						  * sizeof(int_t))))
      ABORT("SUPERLU_MALLOC fails for tcolind_send[]");
    apat_mem += sz_tcolind_send;
  }

  /* Set up marker[] to point at the beginning of each row in the
     send/receive buffer.  For each row, we store first its number of
     elements, and then the elements. */  
  ind_rcv = rdispls[iam];
  for (p = 0; p < nprocs_i; p++) {
    for (i = vtxdist_i[p]; i < vtxdist_i[p+1]; i++) {
      nelts = marker[i] - 1;
      if (p == iam) {
	tcolind_recv[ind_rcv] = nelts;
	marker[i] = ind_rcv + 1;
	ind_rcv += nelts + 1;
      }
      else {
	tcolind_send[sdispls[p]] = nelts;
	marker[i] = sdispls[p] + 1;
	sdispls[p] += nelts + 1;
      }
    }
  }
  /* reset sdispls vector */
  for (i = 0, p = 0; p < nprocs_i; p++) {
    sdispls[p] = i;  
    i += sendCnts[p];
  }
  /* Second pass of the local matrix A to copy data to be send */
  for (j = 0; j < m_loc; j++)
    for (i = rowptr[j]; i < rowptr[j+1]; i++) {
      col = colind[i];
      ipcol = iperm_r[col];      
      if (ipcol >= fst_row && ipcol < fst_row + m_loc)  /* local data */
	tcolind_recv[marker[ipcol]] = perm_r[j + fst_row];      
      else /* remote */ 
	tcolind_send[marker[ipcol]] = perm_r[j + fst_row];
      marker[ipcol] ++;
    }
  sendCnts[iam] = 0;
  recvCnts[iam] = 0;

#if defined (_LONGINT)
  for (p=0; p<nprocs; p++) {
    if (sendCnts[p] > INT_MAX || sdispls[p] > INT_MAX ||
	recvCnts[p] > INT_MAX || rdispls[p] > INT_MAX)
      ABORT("ERROR in dist_symbLU size to send > INT_MAX\n");
    intBuf1[p] = (int) sendCnts[p];
    intBuf2[p] = (int) sdispls[p];
    intBuf3[p] = (int) recvCnts[p];
    intBuf4[p] = (int) rdispls[p];
  }
#else  /* Default */
  intBuf1 = sendCnts;  intBuf2 = sdispls;
  intBuf3 = recvCnts;  intBuf4 = rdispls;
#endif
  
  /* send/receive transpose matrix T */
  MPI_Alltoallv (tcolind_send, intBuf1, intBuf2, mpi_int_t,
		 tcolind_recv, intBuf3, intBuf4, mpi_int_t,
		 grid->comm);
  /* ------------------------------------------------------------
     DEALLOCATE SEND COMMUNICATION STORAGE
     ------------------------------------------------------------*/
  if (sz_tcolind_send) {
    SUPERLU_FREE( tcolind_send );
    apat_mem_max = apat_mem;
    apat_mem -= sz_tcolind_send;
  }

  /* ----------------------------------------------------------------
     FOR LOCAL ROWS:
       compute B = A + T, where row j of B is:
       Struct (B(j,:)) = Struct (A(j,:)) UNION Struct (T(j,:))
       do not include the diagonal entry
     THIS COUNTS FOR TWO PASSES OF THE LOCAL ROWS OF A AND T.   
     ------------------------------------------------------------------ */
  
  /* Reset marker to EMPTY */
  for (i = 0; i < n; ++i) marker[i] = EMPTY;
  /* save rdispls information */
  for (p = 0; p < nprocs_i; p++)
    sdispls[p] = rdispls[p];

  /* First pass determines number of nonzeros in B */
  num_nz = 0;
  for (j = 0; j < m_loc; j++) {
    /* Flag the diagonal so it's not included in the B matrix */
    marker[perm_r[j + fst_row]] = j;
    
    /* Add pattern of row A(j,:) to B(j,:) */
    for (i = rowptr[j]; i < rowptr[j+1]; i++) {
      k = colind[i];
      if ( marker[k] != j ) {
	marker[k] = j;
	++num_nz;
      }
    }
    
    /* Add pattern of row T(j,:) to B(j,:) */
    for (p = 0; p < nprocs_i; p++) {
      t_ind = rdispls[p];
      nelts = tcolind_recv[t_ind]; t_ind ++;
      for (i = t_ind; i < t_ind + nelts; i++) {
	k = tcolind_recv[i];
	if ( marker[k] != j ) {
	  marker[k] = j;
	  ++num_nz;
	}
      }
      t_ind += nelts;
      rdispls[p] = t_ind;
    }
  }
  bnz_t = num_nz;

  /* Allocate storage for B=Pr*A+A'*Pr' */
  if ( !(b_rowptr_t = (int_t*) SUPERLU_MALLOC((m_loc+1) * sizeof(int_t))))
    ABORT("SUPERLU_MALLOC fails for b_rowptr_t[]");
  if ( bnz_t ) {
    if ( !(b_colind_t = (int_t*) SUPERLU_MALLOC( bnz_t * sizeof(int_t))))
      ABORT("SUPERLU_MALLOC fails for b_colind_t[]");
  }
  apat_mem += m_loc + 1 + bnz_t;
  if (apat_mem > apat_mem_max)
    apat_mem_max = apat_mem;
  
  /* Reset marker to EMPTY */
  for (i = 0; i < n; i++) marker[i] = EMPTY;
  /* restore rdispls information */
  for (p = 0; p < nprocs_i; p++)
    rdispls[p] = sdispls[p];
  
  /* Second pass, compute each row of B, one at a time */
  num_nz = 0;
  t_ind = 0;
  for (j = 0; j < m_loc; j++) {
    b_rowptr_t[j] = num_nz;
    
    /* Flag the diagonal so it's not included in the B matrix */
    marker[perm_r[j + fst_row]] = j;

    /* Add pattern of row A(j,:) to B(j,:) */
    for (i = rowptr[j]; i < rowptr[j+1]; i++) {
      k = colind[i];
      if ( marker[k] != j ) {
	marker[k] = j;
	b_colind_t[num_nz] = k; num_nz ++;
      }
    }

    /* Add pattern of row T(j,:) to B(j,:) */
    for (p = 0; p < nprocs_i; p++) {
      t_ind = rdispls[p];
      nelts = tcolind_recv[t_ind]; t_ind++;
      for (i = t_ind; i < t_ind + nelts; i++) {
	k = tcolind_recv[i];
	if ( marker[k] != j ) {
	  marker[k] = j;
	  b_colind_t[num_nz] = k; num_nz++;
	}
      }
      t_ind += nelts;
      rdispls[p] = t_ind;
    }
  }
  b_rowptr_t[m_loc] = num_nz;

  for (p = 0; p <= SUPERLU_MIN(nprocs_i, nprocs_o); p++) 
    if (vtxdist_i[p] != vtxdist_o[p])
      redist_pra = TRUE;
  
  if (sz_tcolind_recv) {
    SUPERLU_FREE (tcolind_recv);
    apat_mem -= sz_tcolind_recv;
  }
  SUPERLU_FREE (marker);
  SUPERLU_FREE (iperm_r);
  apat_mem -= 2 * n + 1;
  
  /* redistribute permuted matrix (by rows) from nproc_i processors
     to nproc_o processors */
  if (redist_pra) {
    m_loc_o = vtxdist_o[iam+1] - vtxdist_o[iam];
    fst_row_o = vtxdist_o[iam];
    nnz_loc = 0;
    
    if ( !(b_rowptr = intMalloc_dist(m_loc_o + 1)) )
      ABORT("Malloc fails for *b_rowptr[].");
    apat_mem += m_loc_o + 1;
    if (apat_mem > apat_mem_max)
      apat_mem_max = apat_mem;

    for (p = 0; p < nprocs_i; p++) {
      sendCnts[p] = 0;
      recvCnts[p] = 0;
    }

    for (i = 0; i < m_loc; i++) {
      k = perm_r[i+fst_row];
      /* find the processor to which row k belongs */
      j = FALSE; p = 0;
      while (!j) {
	if (vtxdist_o[p] <= k && k < vtxdist_o[p+1])
	  j = TRUE;
	else 
	  p ++;
      }
      if (p == iam) {
	b_rowptr[k-fst_row_o] = b_rowptr_t[i + 1] - b_rowptr_t[i];
	nnz_loc += b_rowptr[k-fst_row_o];
      }
      else
	sendCnts[p] += b_rowptr_t[i + 1] - b_rowptr_t[i] + 2;
    }
    /* exchange send/receive counts information in between all processors */
    MPI_Alltoall (sendCnts, 1, mpi_int_t,
		  recvCnts, 1, mpi_int_t, grid->comm);
    
    for (i = 0, j = 0, p = 0; p < nprocs_i; p++) {
      rdispls[p] = j;
      j += recvCnts[p];
      sdispls[p] = i;  
      i += sendCnts[p];
    }
    rdispls[p] = j;
    sdispls[p] = i;
    sz_tcolind_recv = j;
    sz_tcolind_send = i;

    /* allocate memory for local data */
    tcolind_recv = NULL;
    tcolind_send = NULL;
    if (sz_tcolind_recv) {
      if ( !(tcolind_recv = (int_t*) SUPERLU_MALLOC( sz_tcolind_recv 
						     * sizeof(int_t) )))
	ABORT("SUPERLU_MALLOC fails tcolind_recv[]");
      apat_mem += sz_tcolind_recv;
    }
    /* allocate memory to receive necessary data */
    if (sz_tcolind_send) {
      if (!(tcolind_send = (int_t*) SUPERLU_MALLOC( (sz_tcolind_send) 
						    * sizeof(int_t))))
	ABORT("SUPERLU_MALLOC fails for tcolind_send[]");
      apat_mem += sz_tcolind_send;
    }
    if (apat_mem > apat_mem_max)
      apat_mem_max = apat_mem;

    /* Copy data to be send */
    ind_rcv = rdispls[iam];
    for (i = 0; i < m_loc; i++) {
      k = perm_r[i+fst_row];
      /* find the processor to which row k belongs */
      j = FALSE; p = 0;
      while (!j) {
	if (vtxdist_o[p] <= k && k < vtxdist_o[p+1])
	  j = TRUE;
	else 
	  p ++;
      }
      if (p != iam) { /* remote */ 
	tcolind_send[sdispls[p]] = k;
	tcolind_send[sdispls[p]+1] = b_rowptr_t[i+1] - b_rowptr_t[i];
	sdispls[p] += 2;
	for (j = b_rowptr_t[i]; j < b_rowptr_t[i+1]; j++) {
	  tcolind_send[sdispls[p]] = b_colind_t[j]; sdispls[p] ++;
	}
      }
    }
  
    /* reset sdispls vector */
    for (i = 0, p = 0; p < nprocs_i; p++) {
      sdispls[p] = i;  
      i += sendCnts[p];
    }
    sendCnts[iam] = 0;
    recvCnts[iam] = 0;
    
#if defined (_LONGINT)
    for (p=0; p<nprocs; p++) {
      if (sendCnts[p] > INT_MAX || sdispls[p] > INT_MAX ||
	  recvCnts[p] > INT_MAX || rdispls[p] > INT_MAX)
	ABORT("ERROR in dist_symbLU size to send > INT_MAX\n");
      intBuf1[p] = (int) sendCnts[p];
      intBuf2[p] = (int) sdispls[p];
      intBuf3[p] = (int) recvCnts[p];
      intBuf4[p] = (int) rdispls[p];
    }
#else  /* Default */
    intBuf1 = sendCnts;  intBuf2 = sdispls;
    intBuf3 = recvCnts;  intBuf4 = rdispls;
#endif

    /* send/receive permuted matrix T by rows */
    MPI_Alltoallv (tcolind_send, intBuf1, intBuf2, mpi_int_t,
		   tcolind_recv, intBuf3, intBuf4, mpi_int_t,
		   grid->comm);
    /* ------------------------------------------------------------
       DEALLOCATE COMMUNICATION STORAGE
       ------------------------------------------------------------*/
    if (sz_tcolind_send) {
      SUPERLU_FREE( tcolind_send );
      apat_mem -= sz_tcolind_send;
    }
    
    /* ------------------------------------------------------------
       STORE ROWS IN ASCENDING ORDER OF THEIR NUMBER
       ------------------------------------------------------------*/
    for (p = 0; p < nprocs; p++) {
      if (p != iam) {
	i = rdispls[p];
	while (i < rdispls[p+1]) {
	  j = tcolind_recv[i];
	  nelts = tcolind_recv[i+1];
	  i += 2 + nelts;
	  b_rowptr[j-fst_row_o] = nelts;
	  nnz_loc += nelts;
	}
      }
    }

    if (nnz_loc)
      if ( !(b_colind = intMalloc_dist(nnz_loc)) ) {
	ABORT("Malloc fails for bcolind[].");
	apat_mem += nnz_loc;
	if (apat_mem > apat_mem_max)
	  apat_mem_max = apat_mem;
      }

    /* Initialize the array of row pointers */
    k = 0;
    for (j = 0; j < m_loc_o; j++) {
      i = b_rowptr[j];
      b_rowptr[j] = k;
      k += i;
    }
    if (m_loc_o) b_rowptr[j] = k;
    
    /* Copy the data into the row oriented storage */
    for (p = 0; p < nprocs; p++) {
      if (p != iam) {
	i = rdispls[p];
	while (i < rdispls[p+1]) {
	  j = tcolind_recv[i];
	  nelts = tcolind_recv[i+1];
	  for (i += 2, k = b_rowptr[j-fst_row_o]; 
	       k < b_rowptr[j-fst_row_o+1]; i++, k++) 
	    b_colind[k] = tcolind_recv[i];
	}
      }
    }
    for (i = 0; i < m_loc; i++) {
      k = perm_r[i+fst_row];
      if (k >= vtxdist_o[iam] && k < vtxdist_o[iam+1]) {
	ind = b_rowptr[k-fst_row_o];
	for (j = b_rowptr_t[i]; j < b_rowptr_t[i+1]; j++, ind++)
	  b_colind[ind] = b_colind_t[j];
      }
    }
    
    SUPERLU_FREE(b_rowptr_t);
    if ( bnz_t )
      SUPERLU_FREE(b_colind_t);
    if (sz_tcolind_recv)
      SUPERLU_FREE(tcolind_recv);
    apat_mem -= bnz_t + m_loc + sz_tcolind_recv;
    
    *p_bnz = nnz_loc;
    *p_b_rowptr = b_rowptr;
    *p_b_colind = b_colind;
  }
  else {
    *p_bnz = bnz_t;
    *p_b_rowptr = b_rowptr_t;
    *p_b_colind = b_colind_t;
  }
  
  SUPERLU_FREE (rdispls);
  SUPERLU_FREE (sdispls);
  SUPERLU_FREE (sendCnts);
  SUPERLU_FREE (recvCnts);
  apat_mem -= 4 * nprocs + 2;
#if defined (_LONGINT)
  SUPERLU_FREE (intBuf1);
  apat_mem -= 4*nprocs*sizeof(int) / sizeof(int_t);
#endif
  
#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Exit a_plus_at_CompRow_loc()");
#endif
  
  return (- apat_mem_max * sizeof(int_t));
} /* a_plus_at_CompRow_loc */


