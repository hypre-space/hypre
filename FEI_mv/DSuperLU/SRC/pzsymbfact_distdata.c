
/*
 * -- Parallel symbolic factorization auxialiary routine (version 2.2) --
 * -- Distributes the data from parallel symbolic factorization 
 * -- to numeric factorization
 * INRIA France -  July 1, 2004
 * Laura Grigori
 *
 * November 1, 2007
 * Feburary 20, 2008
 */

/* limits.h:  the largest positive integer (INT_MAX) */
#include <limits.h>

#include "superlu_zdefs.h"
#include "psymbfact.h"

static float
dist_symbLU (int_t n, Pslu_freeable_t *Pslu_freeable, 
	     Glu_persist_t *Glu_persist, 
	     int_t **p_xlsub, int_t **p_lsub, int_t **p_xusub, int_t **p_usub,
	     gridinfo_t *grid
	     )
/*
 * Purpose
 * =======
 * 
 * Redistribute the symbolic structure of L and U from the distribution
 * used in the parallel symbolic factorization step to the distdibution
 * used in the parallel numeric factorization step.  On exit, the L and U
 * structure for the 2D distribution used in the numeric factorization step is
 * stored in p_xlsub, p_lsub, p_xusub, p_usub.  The global supernodal 
 * information is also computed and it is stored in Glu_persist->supno
 * and Glu_persist->xsup.
 *
 * This routine allocates memory for storing the structure of L and U
 * and the supernodes information.  This represents the arrays:
 * p_xlsub, p_lsub, p_xusub, p_usub,
 * Glu_persist->supno,  Glu_persist->xsup.
 *
 * This routine also deallocates memory allocated during symbolic 
 * factorization routine.  That is, the folloing arrays are freed:
 * Pslu_freeable->xlsub,  Pslu_freeable->lsub, 
 * Pslu_freeable->xusub, Pslu_freeable->usub, 
 * Pslu_freeable->globToLoc, Pslu_freeable->supno_loc, 
 * Pslu_freeable->xsup_beg_loc, Pslu_freeable->xsup_end_loc.
 *
 * Arguments
 * =========
 *
 * n      (Input) int_t
 *        Order of the input matrix
 * Pslu_freeable  (Input) Pslu_freeable_t *
 *        Local L and U structure, 
 *        global to local indexing information.
 * 
 * Glu_persist (Output) Glu_persist_t *
 *        Stores on output the information on supernodes mapping.
 * 
 * p_xlsub (Output) int_t **
 *         Pointer to structure of L distributed on a 2D grid 
 *         of processors, stored by columns.
 * 
 * p_lsub  (Output) int_t **
 *         Structure of L distributed on a 2D grid of processors, 
 *         stored by columns.
 *
 * p_xusub (Output) int_t **
 *         Pointer to structure of U distributed on a 2D grid 
 *         of processors, stored by rows.
 * 
 * p_usub  (Output) int_t **
 *         Structure of U distributed on a 2D grid of processors, 
 *         stored by rows.
 * 
 * grid   (Input) gridinfo_t*
 *        The 2D process mesh.
 *
 * Return value
 * ============
 *   < 0, number of bytes allocated on return from the dist_symbLU.
 *   > 0, number of bytes allocated in this routine when out of memory.
 *        (an approximation).
 */
{
  int   iam, nprocs, pc, pr, p, np, p_diag;
  int_t *nnzToSend, *nnzToRecv, *nnzToSend_l, *nnzToSend_u, 
    *tmp_ptrToSend, *mem;
  int_t *nnzToRecv_l, *nnzToRecv_u;
  int_t *send_1, *send_2, nsend_1, nsend_2;
  int_t *ptrToSend, *ptrToRecv, sendL, sendU, *snd_luind, *rcv_luind;
  int_t nsupers, nsupers_i, nsupers_j;
  int *nvtcs, *intBuf1, *intBuf2, *intBuf3, *intBuf4, intNvtcs_loc;
  int_t maxszsn, maxNvtcsPProc;
  int_t *xsup_n, *supno_n, *temp, *xsup_beg_s, *xsup_end_s, *supno_s;
  int_t *xlsub_s, *lsub_s, *xusub_s, *usub_s;
  int_t *xlsub_n, *lsub_n, *xusub_n, *usub_n;
  int_t *xsub_s, *sub_s, *xsub_n, *sub_n;
  int_t *globToLoc, nvtcs_loc;
  int_t SendCnt_l, SendCnt_u, nnz_loc_l, nnz_loc_u, nnz_loc,
    RecvCnt_l, RecvCnt_u, ind_loc;
  int_t i, k, j, gb, szsn,  gb_n, gb_s, gb_l, fst_s, fst_s_l, lst_s, i_loc;
  int_t nelts, isize;
  float memAux;  /* Memory used during this routine and freed on return */
  float memRet; /* Memory allocated and not freed on return */
  int_t iword, dword;
  
  /* ------------------------------------------------------------
     INITIALIZATION.
     ------------------------------------------------------------*/
  iam = grid->iam;
#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Enter dist_symbLU()");
#endif
  nprocs = (int) grid->nprow * grid->npcol;
  xlsub_s = Pslu_freeable->xlsub; lsub_s = Pslu_freeable->lsub;
  xusub_s = Pslu_freeable->xusub; usub_s = Pslu_freeable->usub;
  maxNvtcsPProc = Pslu_freeable->maxNvtcsPProc;
  globToLoc     = Pslu_freeable->globToLoc;
  nvtcs_loc     = Pslu_freeable->nvtcs_loc;
  xsup_beg_s    = Pslu_freeable->xsup_beg_loc;
  xsup_end_s    = Pslu_freeable->xsup_end_loc;
  supno_s       = Pslu_freeable->supno_loc;
  rcv_luind     = NULL;
  iword = sizeof(int_t);
  dword = sizeof(doublecomplex);
  memAux = 0.; memRet = 0.;
  
  mem           = intCalloc_dist(12 * nprocs);
  if (!mem)
    return (ERROR_RET);
  memAux     = (float) (12 * nprocs * sizeof(int_t));
  nnzToRecv     = mem;
  nnzToSend     = nnzToRecv + 2*nprocs;
  nnzToSend_l   = nnzToSend + 2 * nprocs;
  nnzToSend_u   = nnzToSend_l + nprocs;
  send_1        = nnzToSend_u + nprocs;
  send_2        = send_1 + nprocs;
  tmp_ptrToSend = send_2 + nprocs;
  nnzToRecv_l   = tmp_ptrToSend + nprocs;
  nnzToRecv_u   = nnzToRecv_l + nprocs;
  
  ptrToSend = nnzToSend;
  ptrToRecv = nnzToSend + nprocs;

  nvtcs = (int *) SUPERLU_MALLOC(5 * nprocs * sizeof(int));
  intBuf1 = nvtcs + nprocs;
  intBuf2 = nvtcs + 2 * nprocs;
  intBuf3 = nvtcs + 3 * nprocs;
  intBuf4 = nvtcs + 4 * nprocs;
  memAux += 5 * nprocs * sizeof(int);

  maxszsn   = sp_ienv_dist(3);
  
  /* Allocate space for storing Glu_persist_n. */
  if ( !(supno_n = intMalloc_dist(n+1)) ) {
    fprintf (stderr, "Malloc fails for supno_n[].");
    return (memAux);
  }
  memRet += (float) ((n+1) * sizeof(int_t));

  /* ------------------------------------------------------------
     DETERMINE SUPERNODES FOR NUMERICAL FACTORIZATION
     ------------------------------------------------------------*/
  
  if (nvtcs_loc > INT_MAX)
    ABORT("ERROR in dist_symbLU nvtcs_loc > INT_MAX\n");
  intNvtcs_loc = (int) nvtcs_loc;
  MPI_Gather (&intNvtcs_loc, 1, MPI_INT, nvtcs, 1, MPI_INT,
	      0, grid->comm);

  if (!iam) {
    /* set ptrToRecv to point to the beginning of the data for
       each processor */
    for (k = 0, p = 0; p < nprocs; p++) {
      ptrToRecv[p] = k;
      k += nvtcs[p];
    }
  }
  
  if (nprocs > 1) {
    temp = NULL;
    if (!iam ) {
      if ( !(temp = intMalloc_dist (n+1)) ) {
	fprintf (stderr, "Malloc fails for temp[].");
	return (memAux + memRet);
      }
      memAux += (float) (n+1) * iword;
    }
#if defined (_LONGINT)
    for (p=0; p<nprocs; p++) {
      if (ptrToRecv[p] > INT_MAX)
	ABORT("ERROR in dist_symbLU size to send > INT_MAX\n");
      intBuf1[p] = (int) ptrToRecv[p];
    }
#else  /* Default */
    intBuf1 = ptrToRecv;
#endif
    MPI_Gatherv (supno_s, (int) nvtcs_loc, mpi_int_t, 
		 temp, nvtcs, intBuf1, mpi_int_t, 0, grid->comm);
  }
  else
    temp = supno_s;

  if (!iam) {
    nsupers = 0;
    p = (int) OWNER( globToLoc[0] );
    gb = temp[ptrToRecv[p]];
    supno_n[0] = nsupers;
    ptrToRecv[p] ++;
    szsn = 1;
    for (j = 1; j < n; j ++) {
      if (p != (int) OWNER( globToLoc[j] ) || szsn >= maxszsn || gb != temp[ptrToRecv[p]]) {
	nsupers ++;
	p  = (int) OWNER( globToLoc[j] );
	gb = temp[ptrToRecv[p]];
	szsn = 1;
      }
      else {
	szsn ++;
      }
      ptrToRecv[p] ++;
      supno_n[j] = nsupers;
    }
    nsupers++;
    if (nprocs > 1) {
      SUPERLU_FREE (temp);
      memAux -= (float) (n+1) * iword;
    }
    supno_n[n] = nsupers;
  }

  /* reset to 0 nnzToSend */
  for (p = 0; p < 2 *nprocs; p++)
    nnzToSend[p] = 0;
  
  MPI_Bcast (supno_n, n+1, mpi_int_t, 0, grid->comm);
  nsupers = supno_n[n];
  /* Allocate space for storing Glu_persist_n. */
  if ( !(xsup_n = intMalloc_dist(nsupers+1)) ) {
    fprintf (stderr, "Malloc fails for xsup_n[].");
    return (memAux + memRet);
  }
  memRet += (float) (nsupers+1) * iword;  

  /* ------------------------------------------------------------
     COUNT THE NUMBER OF NONZEROS TO BE SENT TO EACH PROCESS,
     THEN ALLOCATE SPACE.
     THIS ACCOUNTS FOR THE FIRST PASS OF L and U.
     ------------------------------------------------------------*/
  gb = EMPTY;
  for (i = 0; i < n; i++) {
    if (gb != supno_n[i]) {
      /* a new supernode starts */
      gb = supno_n[i];
      xsup_n[gb] = i;
    }
  }
  xsup_n[nsupers] = n;
  
  for (p = 0; p < nprocs; p++) {
    send_1[p] = FALSE;
    send_2[p] = FALSE;
  }
  for (gb_n = 0; gb_n < nsupers; gb_n ++) {
    i = xsup_n[gb_n];
    if (iam == (int) OWNER( globToLoc[i] )) {
      pc = PCOL( gb_n, grid );
      pr = PROW( gb_n, grid );
      p_diag = PNUM( pr, pc, grid);
      
      i_loc = LOCAL_IND( globToLoc[i] );
      gb_s  = supno_s[i_loc];
      fst_s = xsup_beg_s[gb_s];
      lst_s = xsup_end_s[gb_s];
      fst_s_l = LOCAL_IND( globToLoc[fst_s] );
      for (j = xlsub_s[fst_s_l]; j < xlsub_s[fst_s_l+1]; j++) {
	k = lsub_s[j];
	if (k >= i) {
	  gb = supno_n[k];
	  p = (int) PNUM( PROW(gb, grid), pc, grid );
	  nnzToSend[2*p] ++;
	  send_1[p] = TRUE;
	}
      }
      for (j = xusub_s[fst_s_l]; j < xusub_s[fst_s_l+1]; j++) {
	k = usub_s[j];
	if (k >= i + xsup_n[gb_n+1] - xsup_n[gb_n]) {
	  gb = supno_n[k];
	  p = PNUM( pr, PCOL(gb, grid), grid);
	  nnzToSend[2*p+1] ++;	
	  send_2[p] = TRUE;
	}
      }
      
      nsend_2 = 0;
      for (p = pr * grid->npcol; p < (pr + 1) * grid->npcol; p++) {
	nnzToSend[2*p+1] += 2;
	if (send_2[p])  nsend_2 ++;	  
      }
      for (p = pr * grid->npcol; p < (pr + 1) * grid->npcol; p++) 
	if (send_2[p] || p == p_diag) {
	  if (p == p_diag && !send_2[p])
	    nnzToSend[2*p+1] += nsend_2;
	  else
	    nnzToSend[2*p+1] += nsend_2-1;
	  send_2[p] = FALSE;
	}
      nsend_1 = 0;
      for (p = pc; p < nprocs; p += grid->npcol) {
	nnzToSend[2*p] += 2;
	if (send_1[p]) nsend_1 ++;
      }
      for (p = pc; p < nprocs; p += grid->npcol) 
	if (send_1[p]) {
	  nnzToSend[2*p] += nsend_1-1;
	  send_1[p] = FALSE;
	}
	else
	  nnzToSend[2*p] += nsend_1;
    }
  }
  
  /* All-to-all communication */
  MPI_Alltoall( nnzToSend, 2, mpi_int_t, nnzToRecv, 2, mpi_int_t,
		grid->comm);
  
  nnz_loc_l = nnz_loc_u = 0;
  SendCnt_l = SendCnt_u = RecvCnt_l = RecvCnt_u = 0;  
  for (p = 0; p < nprocs; p++) {
    if ( p != iam ) {
      SendCnt_l += nnzToSend[2*p];   nnzToSend_l[p] = nnzToSend[2*p];
      SendCnt_u += nnzToSend[2*p+1]; nnzToSend_u[p] = nnzToSend[2*p+1]; 
      RecvCnt_l += nnzToRecv[2*p];   nnzToRecv_l[p] = nnzToRecv[2*p];
      RecvCnt_u += nnzToRecv[2*p+1]; nnzToRecv_u[p] = nnzToRecv[2*p+1];
    } else {
      nnz_loc_l += nnzToRecv[2*p];
      nnz_loc_u += nnzToRecv[2*p+1];
      nnzToSend_l[p] = 0; nnzToSend_u[p] = 0;
      nnzToRecv_l[p] = nnzToRecv[2*p]; 
      nnzToRecv_u[p] = nnzToRecv[2*p+1];
    }
  }
  
  /* Allocate space for storing the symbolic structure after redistribution. */
  nsupers_i = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
  nsupers_j = CEILING( nsupers, grid->npcol ); /* Number of local block columns */
  if ( !(xlsub_n = intCalloc_dist(nsupers_j+1)) ) {
    fprintf (stderr, "Malloc fails for xlsub_n[].");
    return (memAux + memRet);
  }
  memRet += (float) (nsupers_j+1) * iword;

  if ( !(xusub_n = intCalloc_dist(nsupers_i+1)) ) {
    fprintf (stderr, "Malloc fails for xusub_n[].");
    return (memAux + memRet);
  }
  memRet += (float) (nsupers_i+1) * iword;  

  /* Allocate temp storage for sending/receiving the L/U symbolic structure. */
  if ( (RecvCnt_l + nnz_loc_l) || (RecvCnt_u + nnz_loc_u) ) {
    if (!(rcv_luind = 
	  intMalloc_dist(SUPERLU_MAX(RecvCnt_l+nnz_loc_l, RecvCnt_u+nnz_loc_u))) ) {
      fprintf (stderr, "Malloc fails for rcv_luind[].");
      return (memAux + memRet);
    }
    memAux += (float) SUPERLU_MAX(RecvCnt_l+nnz_loc_l, RecvCnt_u+nnz_loc_u) 
      * iword;
  }
  if ( nprocs > 1 && (SendCnt_l || SendCnt_u) ) {
    if (!(snd_luind = intMalloc_dist(SUPERLU_MAX(SendCnt_l, SendCnt_u))) ) {
      fprintf (stderr, "Malloc fails for index[].");
      return (memAux + memRet);
    }
    memAux += (float) SUPERLU_MAX(SendCnt_l, SendCnt_u) * iword;
  } 
  
  /* ------------------------------------------------------------------
     LOAD THE SYMBOLIC STRUCTURE OF L AND U INTO THE STRUCTURES TO SEND.
     THIS ACCOUNTS FOR THE SECOND PASS OF L and U.
     ------------------------------------------------------------------*/
  sendL = TRUE;
  sendU = FALSE;
  while (sendL || sendU) {
    if (sendL) {
      xsub_s = xlsub_s; sub_s = lsub_s; xsub_n = xlsub_n;
      nnzToSend = nnzToSend_l; nnzToRecv = nnzToRecv_l;
    }
    if (sendU) {
      xsub_s = xusub_s; sub_s = usub_s; xsub_n = xusub_n;
      nnzToSend = nnzToSend_u; nnzToRecv = nnzToRecv_u;
    }
    for (i = 0, j = 0, p = 0; p < nprocs; p++) {
      if ( p != iam ) {
	ptrToSend[p] = i;  i += nnzToSend[p];
      }
      ptrToRecv[p] = j;  j += nnzToRecv[p];
    }
    nnzToRecv[iam] = 0;
    
    ind_loc = ptrToRecv[iam];
    for (gb_n = 0; gb_n < nsupers; gb_n++) {
      nsend_2 = 0;    
      i = xsup_n[gb_n];
      if (iam == OWNER( globToLoc[i] )) {
	pc = PCOL( gb_n, grid );
	pr = PROW( gb_n, grid );
	p_diag = PNUM( pr, pc, grid );
	
	i_loc = LOCAL_IND( globToLoc[i] );
	gb_s  = supno_s[i_loc];
	fst_s = xsup_beg_s[gb_s];
	lst_s = xsup_end_s[gb_s];
	fst_s_l = LOCAL_IND( globToLoc[fst_s] );

	if (sendL) {
	  p = pc;                np = grid->nprow;	  
	} else {
	  p = pr * grid->npcol;  np = grid->npcol;
	}
	for (j = 0; j < np; j++) {
	  if (p == iam) {
	    rcv_luind[ind_loc] = gb_n;
	    rcv_luind[ind_loc+1] = 0;
	    tmp_ptrToSend[p] = ind_loc + 1;
	    ind_loc += 2;	 
	  }
	  else {
	    snd_luind[ptrToSend[p]] = gb_n;
	    snd_luind[ptrToSend[p]+1] = 0;
	    tmp_ptrToSend[p] = ptrToSend[p] + 1;
	    ptrToSend[p] += 2;	 
	  }
	  if (sendL) p += grid->npcol;
	  if (sendU) p++;
	}
	for (j = xsub_s[fst_s_l]; j < xsub_s[fst_s_l+1]; j++) {
	  k = sub_s[j];
	  if ((sendL && k >= i) || (sendU && k >= i + xsup_n[gb_n+1] - xsup_n[gb_n])) {
	    gb = supno_n[k];
	    if (sendL)
	      p = PNUM( PROW(gb, grid), pc, grid );
	    else 
	      p = PNUM( pr, PCOL(gb, grid), grid);
	    if (send_1[p] == FALSE) {
	      send_1[p] = TRUE;
	      send_2[nsend_2] = k; nsend_2 ++;
	    }
	    if (p == iam) {
	      rcv_luind[ind_loc] = k;  ind_loc++;
	      if (sendL)
		xsub_n[LBj( gb_n, grid )] ++;
	      else
		xsub_n[LBi( gb_n, grid )] ++;
	    }
	    else {
	      snd_luind[ptrToSend[p]] = k;
	      ptrToSend[p] ++; snd_luind[tmp_ptrToSend[p]] ++;
	    }
	  }
	}
	if (sendL)
	  for (p = pc; p < nprocs; p += grid->npcol) {
	      for (k = 0; k < nsend_2; k++) {
		gb = supno_n[send_2[k]];
		if (PNUM(PROW(gb, grid), pc, grid) != p) {
		  if (p == iam) {
		    rcv_luind[ind_loc] = send_2[k];  ind_loc++;
		    xsub_n[LBj( gb_n, grid )] ++;
		  }
		  else {
		    snd_luind[ptrToSend[p]] = send_2[k];
		    ptrToSend[p] ++; snd_luind[tmp_ptrToSend[p]] ++;
		  }
		}
	      }
	      send_1[p] = FALSE;
	  }  
	if (sendU)
	  for (p = pr * grid->npcol; p < (pr + 1) * grid->npcol; p++) {
	    if (send_1[p] || p == p_diag) {	      
	      for (k = 0; k < nsend_2; k++) {
		gb = supno_n[send_2[k]];
		if(PNUM( pr, PCOL(gb, grid), grid) != p) {
		  if (p == iam) {
		    rcv_luind[ind_loc] = send_2[k];  ind_loc++;
		    xsub_n[LBi( gb_n, grid )] ++;
		  }
		  else {
		    snd_luind[ptrToSend[p]] = send_2[k];
		    ptrToSend[p] ++; snd_luind[tmp_ptrToSend[p]] ++;
		  }	     
		}
	      } 
	      send_1[p] = FALSE;
	    }
	  }
      }
    }
    
    /* reset ptrToSnd to point to the beginning of the data for
       each processor (structure needed in MPI_Alltoallv) */
    for (i = 0, p = 0; p < nprocs; p++) {
      ptrToSend[p] = i;  i += nnzToSend[p];
    }

    /* ------------------------------------------------------------
       PERFORM REDISTRIBUTION. THIS INVOLVES ALL-TO-ALL COMMUNICATION.
       Note: it uses MPI_Alltoallv.
       ------------------------------------------------------------*/
    if (nprocs > 1) {
#if defined (_LONGINT)
      nnzToSend[iam] = 0;
      for (p=0; p<nprocs; p++) {
	if (nnzToSend[p] > INT_MAX || ptrToSend[p] > INT_MAX ||
	    nnzToRecv[p] > INT_MAX || ptrToRecv[p] > INT_MAX)
	  ABORT("ERROR in dist_symbLU size to send > INT_MAX\n");
	intBuf1[p] = (int) nnzToSend[p];
	intBuf2[p] = (int) ptrToSend[p];
	intBuf3[p] = (int) nnzToRecv[p];
	intBuf4[p] = (int) ptrToRecv[p];
      }
#else  /* Default */
      intBuf1 = nnzToSend;  intBuf2 = ptrToSend;
      intBuf3 = nnzToRecv;  intBuf4 = ptrToRecv;
#endif

      MPI_Alltoallv (snd_luind, intBuf1, intBuf2, mpi_int_t, 
		     rcv_luind, intBuf3, intBuf4, mpi_int_t,
		     grid->comm);
    }
    if (sendL)
      nnzToRecv[iam] = nnz_loc_l;
    else 
      nnzToRecv[iam] = nnz_loc_u;
    
    /* ------------------------------------------------------------
       DEALLOCATE TEMPORARY STORAGE.
       -------------------------------------------------------------*/
    if (sendU) 
      if ( nprocs > 1 && (SendCnt_l || SendCnt_u) ) {
	SUPERLU_FREE (snd_luind);
	memAux -= (float) SUPERLU_MAX(SendCnt_l, SendCnt_u) * iword;
      }
    
    /* ------------------------------------------------------------
       CONVERT THE FORMAT.
       ------------------------------------------------------------*/
    /* Initialize the array of column of L/ row of U pointers */
    k = 0;
    for (p = 0; p < nprocs; p ++) {
      if (p != iam) {
	i = k;
	while (i < k + nnzToRecv[p]) {
	  gb = rcv_luind[i];
	  nelts = rcv_luind[i+1];
	  if (sendL)
	    xsub_n[LBj( gb, grid )] = nelts;
	  else
	    xsub_n[LBi( gb, grid )] = nelts;
	  i += nelts + 2;
	}
      }
      k += nnzToRecv[p];
    }

    if (sendL) j = nsupers_j;
    else j = nsupers_i;
    k = 0; 
    isize = xsub_n[0];
    xsub_n[0] = 0; 
    for (gb_l = 1; gb_l < j; gb_l++) {
      k += isize;
      isize = xsub_n[gb_l];
      xsub_n[gb_l] = k;
    }
    xsub_n[gb_l] = k + isize;
    nnz_loc = xsub_n[gb_l];
    if (sendL) {
      lsub_n = NULL;
      if (nnz_loc) {
	if ( !(lsub_n = intMalloc_dist(nnz_loc)) ) {
	  fprintf (stderr, "Malloc fails for lsub_n[].");
	  return (memAux + memRet);
	}
	memRet += (float) (nnz_loc * iword);
      }
      sub_n = lsub_n;
    }
    if (sendU) {
      usub_n = NULL;
      if (nnz_loc) {
	if ( !(usub_n = intMalloc_dist(nnz_loc)) ) {
	  fprintf (stderr, "Malloc fails for usub_n[].");
	  return (memAux + memRet);
	}
	memRet += (float) (nnz_loc * iword);
      }
      sub_n = usub_n;
    }
    
    /* Copy the data into the L column / U row oriented storage */
    k = 0;
    for (p = 0; p < nprocs; p++) {
      i = k;
      while (i < k + nnzToRecv[p]) {
	gb = rcv_luind[i];
	if (gb >= nsupers)
	  printf ("Pe[%d] p %d gb %d nsupers %d i %d i-k %d\n",
		  iam, p, gb, nsupers, i, i-k);
	i += 2;
	if (sendL) gb_l = LBj( gb, grid );
	if (sendU) gb_l = LBi( gb, grid );
	for (j = xsub_n[gb_l]; j < xsub_n[gb_l+1]; i++, j++) {
	  sub_n[j] = rcv_luind[i];
	}
      }      
      k += nnzToRecv[p];
    }
    if (sendL) {
      sendL = FALSE;  sendU = TRUE;
    }
    else
      sendU = FALSE;
  }

  /* deallocate memory allocated during symbolic factorization routine */
  if (rcv_luind != NULL) {
    SUPERLU_FREE (rcv_luind);
    memAux -= (float) SUPERLU_MAX(RecvCnt_l+nnz_loc_l, RecvCnt_u+nnz_loc_u) * iword;
  }
  SUPERLU_FREE (mem);  
  memAux -= (float) (12 * nprocs * iword);
  SUPERLU_FREE(nvtcs);
  memAux -= (float) (5 * nprocs * sizeof(int));
  
  if (xlsub_s != NULL) {
    SUPERLU_FREE (xlsub_s); SUPERLU_FREE (lsub_s);
  }
  if (xusub_s != NULL) {
    SUPERLU_FREE (xusub_s); SUPERLU_FREE (usub_s);
  }
  SUPERLU_FREE (globToLoc); 
  if (supno_s != NULL) {
    SUPERLU_FREE (xsup_beg_s); SUPERLU_FREE (xsup_end_s);
    SUPERLU_FREE (supno_s);
  }
  
  Glu_persist->supno = supno_n;  Glu_persist->xsup  = xsup_n;
  *p_xlsub = xlsub_n; *p_lsub = lsub_n;
  *p_xusub = xusub_n; *p_usub = usub_n;

#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Exit dist_symbLU()");
#endif
  
  return (-memRet);
}
  
static float
zdist_A(SuperMatrix *A, ScalePermstruct_t *ScalePermstruct,
	Glu_persist_t *Glu_persist, gridinfo_t *grid, 
	int_t **p_ainf_colptr, int_t **p_ainf_rowind, doublecomplex **p_ainf_val,
	int_t **p_asup_rowptr, int_t **p_asup_colind, doublecomplex **p_asup_val,
	int_t *ilsum_i, int_t *ilsum_j
	)
{
/*
 *
 * Purpose
 * =======
 *   Re-distribute A on the 2D process mesh.  The lower part is
 *   stored using a column format and the upper part
 *   is stored using a row format.
 * 
 * Arguments
 * =========
 * 
 * A      (Input) SuperMatrix*
 *	  The distributed input matrix A of dimension (A->nrow, A->ncol).
 *        The type of A can be: Stype = SLU_NR_loc; Dtype = SLU_Z; Mtype = SLU_GE.
 *
 * ScalePermstruct (Input) ScalePermstruct_t*
 *        The data structure to store the scaling and permutation vectors
 *        describing the transformations performed to the original matrix A.
 *
 * Glu_persist  (Input) Glu_persist_t *
 *        Information on supernodes mapping.
 * 
 * grid   (Input) gridinfo_t*
 *        The 2D process mesh.
 *
 * p_ainf_colptr (Output) int_t**
 *         Pointer to the lower part of A distributed on a 2D grid 
 *         of processors, stored by columns.
 *
 * p_ainf_rowind (Output) int_t**
 *         Structure of of the lower part of A distributed on a 
 *         2D grid of processors, stored by columns.
 *
 * p_ainf_val    (Output) doublecomplex**
 *         Numerical values of the lower part of A, distributed on a 
 *         2D grid of processors, stored by columns.
 *
 * p_asup_rowptr (Output) int_t**
 *         Pointer to the upper part of A distributed on a 2D grid 
 *         of processors, stored by rows.
 *
 * p_asup_colind (Output) int_t**
 *         Structure of of the upper part of A distributed on a 
 *         2D grid of processors, stored by rows.
 *
 * p_asup_val    (Output) doublecomplex**
 *         Numerical values of the upper part of A, distributed on a 
 *         2D grid of processors, stored by rows.
 *
 * ilsum_i  (Input) int_t *
 *       Starting position of each supernode in 
 *       the full array (local, block row wise).
 *
 * ilsum_j  (Input) int_t *
 *       Starting position of each supernode in 
 *       the full array (local, block column wise).
 *
 * Return value
 * ============
 *   < 0, number of bytes allocated on return from the dist_symbLU
 *   > 0, number of bytes allocated when out of memory.
 *        (an approximation).
 *
 */
  int    iam, p, procs;
  NRformat_loc *Astore;
  int_t  *perm_r; /* row permutation vector */
  int_t  *perm_c; /* column permutation vector */
  int_t  i, it, irow, fst_row, j, jcol, k, gbi, gbj, n, m_loc, jsize, isize;
  int_t  nsupers, nsupers_i, nsupers_j;
  int_t  nnz_loc, nnz_loc_ainf, nnz_loc_asup;    /* number of local nonzeros */
  int_t  nnz_remote; /* number of remote nonzeros to be sent */
  int_t  SendCnt; /* number of remote nonzeros to be sent */
  int_t  RecvCnt; /* number of remote nonzeros to be sent */
  int_t *ainf_colptr, *ainf_rowind, *asup_rowptr, *asup_colind;
  doublecomplex *asup_val, *ainf_val;
  int_t  *nnzToSend, *nnzToRecv, maxnnzToRecv;
  int_t  *ia, *ja, **ia_send, *index, *itemp;
  int_t  *ptr_to_send;
  doublecomplex *aij, **aij_send, *nzval, *dtemp;
  doublecomplex *nzval_a;
  MPI_Request *send_req;
  MPI_Status  status;
  int_t *xsup = Glu_persist->xsup;    /* supernode and column mapping */
  int_t *supno = Glu_persist->supno;   
  float memAux;  /* Memory used during this routine and freed on return */
  float memRet; /* Memory allocated and not freed on return */
  int_t iword, dword, szbuf;

  /* ------------------------------------------------------------
     INITIALIZATION.
     ------------------------------------------------------------*/
  iam = grid->iam;
#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Enter zdist_A()");
#endif
  iword = sizeof(int_t);
  dword = sizeof(double);
  
  perm_r = ScalePermstruct->perm_r;
  perm_c = ScalePermstruct->perm_c;
  procs = grid->nprow * grid->npcol;
  Astore = (NRformat_loc *) A->Store;
  n = A->ncol;
  m_loc = Astore->m_loc;
  fst_row = Astore->fst_row;
  if (!(nnzToRecv = intCalloc_dist(2*procs))) {
    fprintf (stderr, "Malloc fails for nnzToRecv[].");
    return (ERROR_RET);
  }
  memAux = (float) (2 * procs * iword);
  memRet = 0.;
  nnzToSend = nnzToRecv + procs;
  nsupers  = supno[n-1] + 1;  

  /* ------------------------------------------------------------
     COUNT THE NUMBER OF NONZEROS TO BE SENT TO EACH PROCESS,
     THEN ALLOCATE SPACE.
     THIS ACCOUNTS FOR THE FIRST PASS OF A.
     ------------------------------------------------------------*/
  for (i = 0; i < m_loc; ++i) {
    for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j) {
      irow = perm_c[perm_r[i+fst_row]];  /* Row number in Pc*Pr*A */
      jcol = Astore->colind[j];
      gbi = BlockNum( irow );
      gbj = BlockNum( jcol );
      p = PNUM( PROW(gbi,grid), PCOL(gbj,grid), grid );
      ++nnzToSend[p]; 
    }
  }
  
  /* All-to-all communication */
  MPI_Alltoall( nnzToSend, 1, mpi_int_t, nnzToRecv, 1, mpi_int_t,
		grid->comm);
  
  maxnnzToRecv = 0;
  nnz_loc = SendCnt = RecvCnt = 0;
  
  for (p = 0; p < procs; ++p) {
    if ( p != iam ) {
      SendCnt += nnzToSend[p];
      RecvCnt += nnzToRecv[p];
      maxnnzToRecv = SUPERLU_MAX( nnzToRecv[p], maxnnzToRecv );
    } else {
      nnz_loc += nnzToRecv[p];
      /*assert(nnzToSend[p] == nnzToRecv[p]);*/
    }
  }
  k = nnz_loc + RecvCnt; /* Total nonzeros ended up in my process. */
  szbuf = k;

  /* Allocate space for storing the triplets after redistribution. */
  if ( !(ia = intMalloc_dist(2*k)) ) {
    fprintf (stderr, "Malloc fails for ia[].");
    return (memAux);
  }
  memAux += (float) (2*k*iword);
  ja = ia + k;
  if ( !(aij = doublecomplexMalloc_dist(k)) ) {
    fprintf (stderr, "Malloc fails for aij[].");
    return (memAux);
  }
  memAux += (float) (k*dword);
  
  /* Allocate temporary storage for sending/receiving the A triplets. */
  if ( procs > 1 ) {
    if ( !(send_req = (MPI_Request *)
	   SUPERLU_MALLOC(2*procs *sizeof(MPI_Request))) ) {
      fprintf (stderr, "Malloc fails for send_req[].");
      return (memAux);
    }
    memAux += (float) (2*procs *sizeof(MPI_Request));
    if ( !(ia_send = (int_t **) SUPERLU_MALLOC(procs*sizeof(int_t*))) ) {
      fprintf(stderr, "Malloc fails for ia_send[].");
      return (memAux);
    }
    memAux += (float) (procs*sizeof(int_t*));
    if ( !(aij_send = (doublecomplex **)SUPERLU_MALLOC(procs*sizeof(doublecomplex*))) ) {
      fprintf(stderr, "Malloc fails for aij_send[].");
      return (memAux);
    }
    memAux += (float) (procs*sizeof(doublecomplex*));    
    if ( !(index = intMalloc_dist(2*SendCnt)) ) {
      fprintf(stderr, "Malloc fails for index[].");
      return (memAux);
    }
    memAux += (float) (2*SendCnt*iword);
    if ( !(nzval = doublecomplexMalloc_dist(SendCnt)) ) {
      fprintf(stderr, "Malloc fails for nzval[].");
      return (memAux);
    }
    memAux += (float) (SendCnt * dword);
    if ( !(ptr_to_send = intCalloc_dist(procs)) ) {
      fprintf(stderr, "Malloc fails for ptr_to_send[].");
      return (memAux);
    }
    memAux += (float) (procs * iword);
    if ( !(itemp = intMalloc_dist(2*maxnnzToRecv)) ) {
      fprintf(stderr, "Malloc fails for itemp[].");
      return (memAux);
    }
    memAux += (float) (2*maxnnzToRecv*iword);
    if ( !(dtemp = doublecomplexMalloc_dist(maxnnzToRecv)) ) {
      fprintf(stderr, "Malloc fails for dtemp[].");
      return (memAux);
    }
    memAux += (float) (maxnnzToRecv * dword);
    
    for (i = 0, j = 0, p = 0; p < procs; ++p) {
      if ( p != iam ) {
	ia_send[p] = &index[i];
	i += 2 * nnzToSend[p]; /* ia/ja indices alternate */
	aij_send[p] = &nzval[j];
	j += nnzToSend[p];
      }
    }
  } /* if procs > 1 */
  
  nsupers_i = CEILING( nsupers, grid->nprow ); /* Number of local block rows */
  nsupers_j = CEILING( nsupers, grid->npcol ); /* Number of local block columns */
  if ( !(ainf_colptr = intCalloc_dist(ilsum_j[nsupers_j] + 1)) ) {
    fprintf (stderr, "Malloc fails for *ainf_colptr[].");
    return (memAux);
  }
  memRet += (float) (ilsum_j[nsupers_j] + 1) * iword;
  if ( !(asup_rowptr = intCalloc_dist(ilsum_i[nsupers_i] + 1)) ) {
    fprintf (stderr, "Malloc fails for *asup_rowptr[].");
    return (memAux+memRet);
  }
  memRet += (float) (ilsum_i[nsupers_i] + 1) * iword;
  
  /* ------------------------------------------------------------
     LOAD THE ENTRIES OF A INTO THE (IA,JA,AIJ) STRUCTURES TO SEND.
     THIS ACCOUNTS FOR THE SECOND PASS OF A.
     ------------------------------------------------------------*/
  nnz_loc = 0; /* Reset the local nonzero count. */
  nnz_loc_ainf = nnz_loc_asup = 0;
  nzval_a = Astore->nzval;
  for (i = 0; i < m_loc; ++i) {
    for (j = Astore->rowptr[i]; j < Astore->rowptr[i+1]; ++j) {
      irow = perm_c[perm_r[i+fst_row]];  /* Row number in Pc*Pr*A */
      jcol = Astore->colind[j];
      gbi = BlockNum( irow );
      gbj = BlockNum( jcol );
      p = PNUM( PROW(gbi,grid), PCOL(gbj,grid), grid );
      
      if ( p != iam ) { /* remote */
	k = ptr_to_send[p];
	ia_send[p][k] = irow;
	ia_send[p][k + nnzToSend[p]] = jcol;
	aij_send[p][k] = nzval_a[j];
	++ptr_to_send[p]; 
      } else {          /* local */
	ia[nnz_loc] = irow;
	ja[nnz_loc] = jcol;
	aij[nnz_loc] = nzval_a[j];
	++nnz_loc;
	/* Count nonzeros in each column of L / row of U */
	if (gbi >= gbj) {
	  ainf_colptr[ilsum_j[LBj( gbj, grid )] + jcol - FstBlockC( gbj )] ++;
	  nnz_loc_ainf ++;
	}
	else {
	  asup_rowptr[ilsum_i[LBi( gbi, grid )] + irow - FstBlockC( gbi )] ++;
	  nnz_loc_asup ++;
	}
      }
    }
  }

  /* ------------------------------------------------------------
     PERFORM REDISTRIBUTION. THIS INVOLVES ALL-TO-ALL COMMUNICATION.
     NOTE: Can possibly use MPI_Alltoallv.
     ------------------------------------------------------------*/
  for (p = 0; p < procs; ++p) {
    if ( p != iam ) {
      it = 2*nnzToSend[p];
      MPI_Isend( ia_send[p], it, mpi_int_t,
		 p, iam, grid->comm, &send_req[p] );
      it = nnzToSend[p];
      MPI_Isend( aij_send[p], it, SuperLU_MPI_DOUBLE_COMPLEX,
		 p, iam+procs, grid->comm, &send_req[procs+p] ); 
    }
  }
  
  for (p = 0; p < procs; ++p) {
    if ( p != iam ) {
      it = 2*nnzToRecv[p];
      MPI_Recv( itemp, it, mpi_int_t, p, p, grid->comm, &status ); 
      it = nnzToRecv[p];
      MPI_Recv( dtemp, it, SuperLU_MPI_DOUBLE_COMPLEX, p, p+procs,
		grid->comm, &status );
      for (i = 0; i < nnzToRecv[p]; ++i) {
	ia[nnz_loc] = itemp[i];
	irow = itemp[i];
	jcol = itemp[i + nnzToRecv[p]];
	/* assert(jcol<n); */
	ja[nnz_loc] = jcol;
	aij[nnz_loc] = dtemp[i];
	++nnz_loc;
	
	gbi = BlockNum( irow );
	gbj = BlockNum( jcol );
	/* Count nonzeros in each column of L / row of U */
	if (gbi >= gbj) {
	  ainf_colptr[ilsum_j[LBj( gbj, grid )] + jcol - FstBlockC( gbj )] ++;
	  nnz_loc_ainf ++;
	}
	else {
	  asup_rowptr[ilsum_i[LBi( gbi, grid )] + irow - FstBlockC( gbi )] ++;
	  nnz_loc_asup ++;
	}
      }
    }
  }
  
  for (p = 0; p < procs; ++p) {
    if ( p != iam ) {
      MPI_Wait( &send_req[p], &status);
      MPI_Wait( &send_req[procs+p], &status);
    }
  }
  
  /* ------------------------------------------------------------
     DEALLOCATE TEMPORARY STORAGE
     ------------------------------------------------------------*/
  
  SUPERLU_FREE(nnzToRecv);
  memAux -= 2 * procs * iword;
  if ( procs > 1 ) {
    SUPERLU_FREE(send_req);
    SUPERLU_FREE(ia_send);
    SUPERLU_FREE(aij_send);
    SUPERLU_FREE(index);
    SUPERLU_FREE(nzval);
    SUPERLU_FREE(ptr_to_send);
    SUPERLU_FREE(itemp);
    SUPERLU_FREE(dtemp);
    memAux -= 2*procs *sizeof(MPI_Request) + procs*sizeof(int_t*) +
      procs*sizeof(doublecomplex*) + 2*SendCnt * iword +
      SendCnt* dword + procs*iword +
      2*maxnnzToRecv*iword + maxnnzToRecv*dword;
  }
  
  /* ------------------------------------------------------------
     CONVERT THE TRIPLET FORMAT.
     ------------------------------------------------------------*/
  if (nnz_loc_ainf != 0) {
    if ( !(ainf_rowind = intMalloc_dist(nnz_loc_ainf)) ) {
      fprintf (stderr, "Malloc fails for *ainf_rowind[].");
      return (memAux+memRet);
    }
    memRet += (float) (nnz_loc_ainf * iword);
    if ( !(ainf_val = doublecomplexMalloc_dist(nnz_loc_ainf)) ) {
      fprintf (stderr, "Malloc fails for *ainf_val[].");
      return (memAux+memRet);
    }
    memRet += (float) (nnz_loc_ainf * dword);
  }
  else {
    ainf_rowind = NULL;
    ainf_val = NULL;
  }
  if (nnz_loc_asup != 0) {
    if ( !(asup_colind = intMalloc_dist(nnz_loc_asup)) ) {
      fprintf (stderr, "Malloc fails for *asup_colind[].");
      return (memAux + memRet);
    }
    memRet += (float) (nnz_loc_asup * iword);
    if ( !(asup_val = doublecomplexMalloc_dist(nnz_loc_asup)) ) {
      fprintf (stderr, "Malloc fails for *asup_val[].");
      return (memAux  + memRet);
    }
    memRet += (float) (nnz_loc_asup * dword);
  }
  else {
    asup_colind = NULL;
    asup_val = NULL;
  }

  /* Initialize the array of column pointers */
  k = 0; 
  jsize = ainf_colptr[0];  ainf_colptr[0] = 0; 
  for (j = 1; j < ilsum_j[nsupers_j]; j++) {
    k += jsize;              
    jsize = ainf_colptr[j];  
    ainf_colptr[j] = k;
  }
  ainf_colptr[ilsum_j[nsupers_j]] = k + jsize;
  i = 0;
  isize = asup_rowptr[0];  asup_rowptr[0] = 0;
  for (j = 1; j < ilsum_i[nsupers_i]; j++) {
    i += isize;
    isize = asup_rowptr[j];  
    asup_rowptr[j] = i;
  }
  asup_rowptr[ilsum_i[nsupers_i]] = i + isize;

  /* Copy the triplets into the column oriented storage */
  for (i = 0; i < nnz_loc; ++i) {
    jcol = ja[i];
    irow = ia[i];
    gbi = BlockNum( irow );
    gbj = BlockNum( jcol );
    /* Count nonzeros in each column of L / row of U */
    if (gbi >= gbj) {
      j = ilsum_j[LBj( gbj, grid )] + jcol - FstBlockC( gbj );
      k = ainf_colptr[j];
      ainf_rowind[k] = irow;
      ainf_val[k] = aij[i];
      ainf_colptr[j] ++;
    }
    else {
      j = ilsum_i[LBi( gbi, grid )] + irow - FstBlockC( gbi );
      k = asup_rowptr[j];
      asup_colind[k] = jcol;
      asup_val[k] = aij[i];
      asup_rowptr[j] ++;
    }
  }

  /* Reset the column pointers to the beginning of each column */
  for (j = ilsum_j[nsupers_j]; j > 0; j--) 
    ainf_colptr[j] = ainf_colptr[j-1];
  for (j = ilsum_i[nsupers_i]; j > 0; j--) 
    asup_rowptr[j] = asup_rowptr[j-1];
  ainf_colptr[0] = 0;
  asup_rowptr[0] = 0;
  
  SUPERLU_FREE(ia);
  SUPERLU_FREE(aij);
  memAux -= 2*szbuf*iword + szbuf*dword;
  
  *p_ainf_colptr = ainf_colptr;
  *p_ainf_rowind = ainf_rowind; 
  *p_ainf_val    = ainf_val;
  *p_asup_rowptr = asup_rowptr;
  *p_asup_colind = asup_colind;
  *p_asup_val    = asup_val;

#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Exit zdist_A()");
  fprintf (stdout, "Size of allocated memory (MB) %.3f\n", memRet*1e-6);
#endif

  return (-memRet);
} /* dist_A */

int_t
zdist_psymbtonum(fact_t fact, int_t n, SuperMatrix *A,
		ScalePermstruct_t *ScalePermstruct,
		Pslu_freeable_t *Pslu_freeable, 
		LUstruct_t *LUstruct, gridinfo_t *grid)
/*
 *
 *
 * Purpose
 * =======
 *   Distribute the input matrix onto the 2D process mesh.
 * 
 * Arguments
 * =========
 * 
 * fact (input) fact_t
 *        Specifies whether or not the L and U structures will be re-used.
 *        = SamePattern_SameRowPerm: L and U structures are input, and
 *                                   unchanged on exit.
 *          This routine should not be called for this case, an error
 *          is generated.  Instead, pddistribute routine should be called.
 *        = DOFACT or SamePattern: L and U structures are computed and output.
 *
 * n      (Input) int
 *        Dimension of the matrix.
 *
 * A      (Input) SuperMatrix*
 *	  The distributed input matrix A of dimension (A->nrow, A->ncol).
 *        A may be overwritten by diag(R)*A*diag(C)*Pc^T.
 *        The type of A can be: Stype = NR; Dtype = SLU_D; Mtype = GE.
 *
 * ScalePermstruct (Input) ScalePermstruct_t*
 *        The data structure to store the scaling and permutation vectors
 *        describing the transformations performed to the original matrix A.
 *
 * Glu_freeable (Input) *Glu_freeable_t
 *        The global structure describing the graph of L and U.
 * 
 * LUstruct (Input) LUstruct_t*
 *        Data structures for L and U factors.
 *
 * grid   (Input) gridinfo_t*
 *        The 2D process mesh.
 *
 * Return value
 * ============
 *   < 0, number of bytes allocated on return from the dist_symbLU
 *   > 0, number of bytes allocated for performing the distribution
 *       of the data, when out of memory.
 *        (an approximation).
 *
 */
{
  Glu_persist_t *Glu_persist = LUstruct->Glu_persist;
  Glu_freeable_t Glu_freeable_n;
  LocalLU_t *Llu = LUstruct->Llu;
  int_t bnnz, fsupc, i, irow, istart, j, jb, jj, k, 
    len, len1, nsupc, nsupc_gb, ii, nprocs;
  int_t ljb;  /* local block column number */
  int_t nrbl; /* number of L blocks in current block column */
  int_t nrbu; /* number of U blocks in current block column */
  int_t gb;   /* global block number; 0 < gb <= nsuper */
  int_t lb;   /* local block number; 0 < lb <= ceil(NSUPERS/Pr) */
  int iam, jbrow, jbcol, jcol, kcol, mycol, myrow, pc, pr, ljb_i, ljb_j, p;
  int_t mybufmax[NBUFFERS];
  NRformat_loc *Astore;
  doublecomplex *a;
  int_t *asub, *xa;
  int_t *ainf_colptr, *ainf_rowind, *asup_rowptr, *asup_colind;
  doublecomplex *asup_val, *ainf_val;
  int_t *xsup, *supno;    /* supernode and column mapping */
  int_t *lsub, *xlsub, *usub, *xusub;
  int_t nsupers, nsupers_i, nsupers_j, nsupers_ij;
  int_t next_ind;      /* next available position in index[*] */
  int_t next_val;      /* next available position in nzval[*] */
  int_t *index;         /* indices consist of headers and row subscripts */
  doublecomplex *lusup, *uval; /* nonzero values in L and U */
  int_t *recvBuf;
  int *ptrToRecv, *nnzToRecv, *ptrToSend, *nnzToSend;
  doublecomplex **Lnzval_bc_ptr;  /* size ceil(NSUPERS/Pc) */
  int_t  **Lrowind_bc_ptr; /* size ceil(NSUPERS/Pc) */
  doublecomplex **Unzval_br_ptr;  /* size ceil(NSUPERS/Pr) */
  int_t  **Ufstnz_br_ptr;  /* size ceil(NSUPERS/Pr) */
  
  /*-- Counts to be used in factorization. --*/
  int_t  *ToRecv, *ToSendD, **ToSendR;
  
  /*-- Counts to be used in lower triangular solve. --*/
  int_t  *fmod;          /* Modification count for L-solve.        */
  int_t  **fsendx_plist; /* Column process list to send down Xk.   */
  int_t  nfrecvx = 0;    /* Number of Xk I will receive.           */
  int_t  nfsendx = 0;    /* Number of Xk I will send               */
  int_t  kseen;
  
  /*-- Counts to be used in upper triangular solve. --*/
  int_t  *bmod;          /* Modification count for U-solve.        */
  int_t  **bsendx_plist; /* Column process list to send down Xk.   */
  int_t  nbrecvx = 0;    /* Number of Xk I will receive.           */
  int_t  nbsendx = 0;    /* Number of Xk I will send               */  
  int_t  *ilsum;         /* starting position of each supernode in 
			    the full array (local)                 */  
  int_t  *ilsum_j, ldaspa_j; /* starting position of each supernode in 
				the full array (local, block column wise) */  
  /*-- Auxiliary arrays; freed on return --*/
  int_t *Urb_marker;  /* block hit marker; size ceil(NSUPERS/Pr)           */
  int_t *LUb_length; /* L,U block length; size nsupers_ij */
  int_t *LUb_indptr; /* pointers to L,U index[]; size nsupers_ij */
  int_t *LUb_number; /* global block number; size nsupers_ij */
  int_t *LUb_valptr; /* pointers to U nzval[]; size ceil(NSUPERS/Pc)      */
  int_t *Lrb_marker;  /* block hit marker; size ceil(NSUPERS/Pr)           */
  doublecomplex *dense, *dense_col; /* SPA */
  doublecomplex zero = {0.0, 0.0};
  int_t ldaspa;     /* LDA of SPA */
  int_t iword, dword;
  float memStrLU, memA,
    memDist = 0.; /* memory used for redistributing the data, which does
		   not include the memory for the numerical values of L and U */
  float  memNLU = 0.; /* memory allocated for storing the numerical values of 
		    L and U, that will be used in the numeric factorization */

#if ( PRNTlevel>=1 )
  int_t nLblocks = 0, nUblocks = 0;
#endif
  
  /* Initialization. */
  iam = grid->iam;
#if ( DEBUGlevel>=1 )
  CHECK_MALLOC(iam, "Enter dist_psymbtonum()");
#endif
  myrow = MYROW( iam, grid );
  mycol = MYCOL( iam, grid );
  nprocs = grid->npcol * grid->nprow;
  for (i = 0; i < NBUFFERS; ++i) mybufmax[i] = 0;
  Astore   = (NRformat_loc *) A->Store;
  
  iword = sizeof(int_t);
  dword = sizeof(doublecomplex);

  if (fact == SamePattern_SameRowPerm) {
    ABORT ("ERROR: call of dist_psymbtonum with fact equals SamePattern_SameRowPerm.");  
  }

  if ((memStrLU = 
       dist_symbLU (n, Pslu_freeable, 
		    Glu_persist, &xlsub, &lsub, &xusub, &usub,	grid)) > 0)
    return (memStrLU);
  memDist += (-memStrLU);
  xsup  = Glu_persist->xsup;    /* supernode and column mapping */
  supno = Glu_persist->supno;   
  nsupers  = supno[n-1] + 1;
  nsupers_i = CEILING( nsupers, grid->nprow );/* No of local row blocks */
  nsupers_j = CEILING( nsupers, grid->npcol );/* No of local column blocks */
  nsupers_ij = SUPERLU_MAX(nsupers_i, nsupers_j);
  if ( !(ilsum = intMalloc_dist(nsupers_i+1)) ) {
    fprintf (stderr, "Malloc fails for ilsum[].");  
    return (memDist + memNLU);
  }
  memNLU += (nsupers_i+1) * iword;
  if ( !(ilsum_j = intMalloc_dist(nsupers_j+1)) ) {
    fprintf (stderr, "Malloc fails for ilsum_j[].");
    return (memDist + memNLU);
  }
  memDist += (nsupers_j+1) * iword;

  /* Compute ldaspa and ilsum[], ldaspa_j and ilsum_j[]. */
  ilsum[0] = 0;
  ldaspa = 0;
  for (gb = 0; gb < nsupers; gb++) 
    if ( myrow == PROW( gb, grid ) ) {
      i = SuperSize( gb );
      ldaspa += i;
      lb = LBi( gb, grid );
      ilsum[lb + 1] = ilsum[lb] + i;
    }
  ilsum[nsupers_i] = ldaspa;

  ldaspa_j = 0; ilsum_j[0] = 0;  
  for (gb = 0; gb < nsupers; gb++) 
    if (mycol == PCOL( gb, grid )) {
      i = SuperSize( gb );
      ldaspa_j += i;
      lb = LBj( gb, grid );
      ilsum_j[lb + 1] = ilsum_j[lb] + i;
    }
  ilsum_j[nsupers_j] = ldaspa_j;
  
  if ((memA = zdist_A(A, ScalePermstruct, Glu_persist,
		      grid, &ainf_colptr, &ainf_rowind, &ainf_val,
		      &asup_rowptr, &asup_colind, &asup_val,
		      ilsum, ilsum_j)) > 0)
    return (memDist + memA + memNLU);
  memDist += (-memA);

  /* ------------------------------------------------------------
     FIRST TIME CREATING THE L AND U DATA STRUCTURES.
     ------------------------------------------------------------*/
  
  /* We first need to set up the L and U data structures and then
   * propagate the values of A into them.
   */
  if ( !(ToRecv = intCalloc_dist(nsupers)) ) {
    fprintf(stderr, "Calloc fails for ToRecv[].");
    return (memDist + memNLU);
  }
  memNLU += nsupers * iword;
  
  k = CEILING( nsupers, grid->npcol ); /* Number of local column blocks */
  if ( !(ToSendR = (int_t **) SUPERLU_MALLOC(k*sizeof(int_t*))) ) {
    fprintf(stderr, "Malloc fails for ToSendR[].");
    return (memDist + memNLU);
  }
  memNLU += k*sizeof(int_t*);
  j = k * grid->npcol;
  if ( !(index = intMalloc_dist(j)) ) {
    fprintf(stderr, "Malloc fails for index[].");
    return (memDist + memNLU);
  }
  memNLU += j*iword;
  
  for (i = 0; i < j; ++i) index[i] = EMPTY;
  for (i = 0,j = 0; i < k; ++i, j += grid->npcol) ToSendR[i] = &index[j];
  
  /* Auxiliary arrays used to set up L and U block data structures.
     They are freed on return. */
  if ( !(LUb_length = intCalloc_dist(nsupers_ij)) ) {
    fprintf(stderr, "Calloc fails for LUb_length[].");
    return (memDist + memNLU);
  }
  if ( !(LUb_indptr = intMalloc_dist(nsupers_ij)) ) {
    fprintf(stderr, "Malloc fails for LUb_indptr[].");
    return (memDist + memNLU);
  }
  if ( !(LUb_number = intCalloc_dist(nsupers_ij)) ) {
    fprintf(stderr, "Calloc fails for LUb_number[].");
    return (memDist + memNLU);
  }    
  if ( !(LUb_valptr = intCalloc_dist(nsupers_ij)) ) {
    fprintf(stderr, "Calloc fails for LUb_valptr[].");
    return (memDist + memNLU);
  }
  memDist += 4 * nsupers_ij * iword;
  
  k = CEILING( nsupers, grid->nprow ); 
  /* Pointers to the beginning of each block row of U. */
  if ( !(Unzval_br_ptr = 
	 (doublecomplex**)SUPERLU_MALLOC(nsupers_i * sizeof(doublecomplex*))) ) {
    fprintf(stderr, "Malloc fails for Unzval_br_ptr[].");
    return (memDist + memNLU);
  }
  if ( !(Ufstnz_br_ptr = (int_t**)SUPERLU_MALLOC(nsupers_i * sizeof(int_t*))) ) {
    fprintf(stderr, "Malloc fails for Ufstnz_br_ptr[].");
    return (memDist + memNLU);
  }
  memNLU += nsupers_i*sizeof(doublecomplex*) + nsupers_i*sizeof(int_t*);
  Unzval_br_ptr[nsupers_i-1] = NULL;
  Ufstnz_br_ptr[nsupers_i-1] = NULL;

  if ( !(ToSendD = intCalloc_dist(nsupers_i)) ) {
    fprintf(stderr, "Malloc fails for ToSendD[].");
    return (memDist + memNLU);
  }
  memNLU += nsupers_i*iword;  
  if ( !(Urb_marker = intCalloc_dist(nsupers_j))) {
    fprintf(stderr, "Calloc fails for rb_marker[].");
    return (memDist + memNLU);
  }
  if ( !(Lrb_marker = intCalloc_dist( nsupers_i ))) {
    fprintf(stderr, "Calloc fails for rb_marker[].");
    return (memDist + memNLU);
  }
  memDist += (nsupers_i + nsupers_j)*iword;
  
  /* Auxiliary arrays used to set up L, U block data structures.
     They are freed on return.
     k is the number of local row blocks.   */
  if ( !(dense = doublecomplexCalloc_dist(SUPERLU_MAX(ldaspa, ldaspa_j) 
				   * sp_ienv_dist(3))) ) {
    fprintf(stderr, "Calloc fails for SPA dense[].");
    return (memDist + memNLU);
  }
  /* These counts will be used for triangular solves. */
  if ( !(fmod = intCalloc_dist(nsupers_i)) ) {
    fprintf(stderr, "Calloc fails for fmod[].");
    return (memDist + memNLU);
  }
  if ( !(bmod = intCalloc_dist(nsupers_i)) ) {
    fprintf(stderr, "Calloc fails for bmod[].");
    return (memDist + memNLU);
  }
  /* ------------------------------------------------ */
  memNLU += 2*nsupers_i*iword + 
    SUPERLU_MAX(ldaspa, ldaspa_j)*sp_ienv_dist(3)*dword; 
  
  /* Pointers to the beginning of each block column of L. */
  if ( !(Lnzval_bc_ptr = 
	 (doublecomplex**)SUPERLU_MALLOC(nsupers_j * sizeof(doublecomplex*))) ) {
    fprintf(stderr, "Malloc fails for Lnzval_bc_ptr[].");
    return (memDist + memNLU);
  }
  if ( !(Lrowind_bc_ptr = (int_t**)SUPERLU_MALLOC(nsupers_j * sizeof(int_t*))) ) {
    fprintf(stderr, "Malloc fails for Lrowind_bc_ptr[].");
    return (memDist + memNLU);
  }
  memNLU += nsupers_j * sizeof(doublecomplex*) + nsupers_j * sizeof(int_t*);
  Lnzval_bc_ptr[nsupers_j-1] = NULL;
  Lrowind_bc_ptr[nsupers_j-1] = NULL;
  
  /* These lists of processes will be used for triangular solves. */
  if ( !(fsendx_plist = (int_t **) SUPERLU_MALLOC(nsupers_j*sizeof(int_t*))) ) {
    fprintf(stderr, "Malloc fails for fsendx_plist[].");
    return (memDist + memNLU);
  }
  len = nsupers_j * grid->nprow;
  if ( !(index = intMalloc_dist(len)) ) {
    fprintf(stderr, "Malloc fails for fsendx_plist[0]");
    return (memDist + memNLU);
  }
  for (i = 0; i < len; ++i) index[i] = EMPTY;
  for (i = 0, j = 0; i < nsupers_j; ++i, j += grid->nprow)
    fsendx_plist[i] = &index[j];
  if ( !(bsendx_plist = (int_t **) SUPERLU_MALLOC(nsupers_j*sizeof(int_t*))) ) {
    fprintf(stderr, "Malloc fails for bsendx_plist[].");
    return (memDist + memNLU);
  }
  if ( !(index = intMalloc_dist(len)) ) {
    fprintf(stderr, "Malloc fails for bsendx_plist[0]");
    return (memDist + memNLU);
  }
  for (i = 0; i < len; ++i) index[i] = EMPTY;
  for (i = 0, j = 0; i < nsupers_j; ++i, j += grid->nprow)
    bsendx_plist[i] = &index[j];
  /* -------------------------------------------------------------- */
  memNLU += 2*nsupers_j*sizeof(int_t*) + 2*len*iword;
  
  /*------------------------------------------------------------
    PROPAGATE ROW SUBSCRIPTS AND VALUES OF A INTO L AND U BLOCKS.
    THIS ACCOUNTS FOR ONE-PASS PROCESSING OF A, L AND U.
    ------------------------------------------------------------*/
  for (jb = 0; jb < nsupers; jb++) {
    jbcol = PCOL( jb, grid );
    jbrow = PROW( jb, grid );
    ljb_j = LBj( jb, grid ); /* Local block number column wise */
    ljb_i = LBi( jb, grid);  /* Local block number row wise */
    fsupc = FstBlockC( jb );
    nsupc = SuperSize( jb );
    
    if ( myrow == jbrow ) { /* Block row jb in my process row */
      /* Scatter A into SPA. */
      for (j = ilsum[ljb_i], dense_col = dense; j < ilsum[ljb_i]+nsupc; j++) {
	for (i = asup_rowptr[j]; i < asup_rowptr[j+1]; i++) {
	  if (i >= asup_rowptr[ilsum[nsupers_i]]) 
	    printf ("ERR7\n");
	  jcol = asup_colind[i];
	  if (jcol >= n)
	    printf ("Pe[%d] ERR distsn jb %d gb %d j %d jcol %d\n",
		    iam, jb, gb, j, jcol);
	  gb = BlockNum( jcol );
	  lb = LBj( gb, grid );
	  if (gb >= nsupers || lb >= nsupers_j) printf ("ERR8\n");
	  jcol = ilsum_j[lb] + jcol - FstBlockC( gb );
	  if (jcol >= ldaspa_j)
	    printf ("Pe[%d] ERR1 jb %d gb %d j %d jcol %d\n",
		    iam, jb, gb, j, jcol);
	  dense_col[jcol] = asup_val[i];
	}
	dense_col += ldaspa_j;
      }
      
      /*------------------------------------------------
       * SET UP U BLOCKS.
       *------------------------------------------------*/
      /* Count number of blocks and length of each block. */
      nrbu = 0;
      len = 0; /* Number of column subscripts I own. */
      len1 = 0; /* number of fstnz subscripts */
      for (i = xusub[ljb_i]; i < xusub[ljb_i+1]; i++) {
	if (i >= xusub[nsupers_i]) printf ("ERR10\n");
	jcol = usub[i];
	gb = BlockNum( jcol ); /* Global block number */
	
	/*if (fsupc <= 146445 && 146445 < fsupc + nsupc && jcol == 397986)
	  printf ("Pe[%d] [%d %d] elt [%d] jbcol %d pc %d\n",
	  iam, jb, gb, jcol, jbcol, pc); */
	
	lb = LBj( gb, grid );  /* Local block number */
	pc = PCOL( gb, grid ); /* Process col owning this block */
	if (mycol == jbcol) ToSendR[ljb_j][pc] = YES;
	/* if (mycol == jbcol && mycol != pc) ToSendR[ljb_j][pc] = YES; */
	pr = PROW( gb, grid );
	if ( pr != jbrow  && mycol == pc)
	  bsendx_plist[lb][jbrow] = YES; 
	if (mycol == pc) {
	  len += nsupc;
	  LUb_length[lb] += nsupc;
	  ToSendD[ljb_i] = YES;
	  if (Urb_marker[lb] <= jb) { /* First see this block */
	    if (Urb_marker[lb] == FALSE && gb != jb && myrow != pr) nbrecvx ++;
	    Urb_marker[lb] = jb + 1;
	    LUb_number[nrbu] = gb;
	    /* if (gb == 391825 && jb == 145361)
	       printf ("Pe[%d] T1 [%d %d] nrbu %d \n",
	       iam, jb, gb, nrbu); */
	    nrbu ++;
	    len1 += SuperSize( gb );
	    if ( gb != jb )/* Exclude diagonal block. */
	      ++bmod[ljb_i];/* Mod. count for back solve */
#if ( PRNTlevel>=1 )
	    ++nUblocks;
#endif
	  }
	}
      } /* for i ... */
      
      if ( nrbu ) { 
	/* Sort the blocks of U in increasing block column index.
	   SuperLU_DIST assumes this is true */
	/* simple insert sort algorithm */
	/* to be transformed in quick sort */
	for (j = 1; j < nrbu; j++) {
	  k = LUb_number[j];
	  for (i=j-1; i>=0 && LUb_number[i] > k; i--) {
	    LUb_number[i+1] = LUb_number[i];
	  }
	  LUb_number[i+1] = k;
	} 
	
	/* Set up the initial pointers for each block in
	   index[] and nzval[]. */
	/* Add room for descriptors */
	len1 += BR_HEADER + nrbu * UB_DESCRIPTOR;
	if ( !(index = intMalloc_dist(len1+1)) ) {
	  fprintf (stderr, "Malloc fails for Uindex[]");
	  return (memDist + memNLU);
	}
	Ufstnz_br_ptr[ljb_i] = index;
	if (!(Unzval_br_ptr[ljb_i] =
	      doublecomplexMalloc_dist(len))) {
	  fprintf (stderr, "Malloc fails for Unzval_br_ptr[*][]");
	  return (memDist + memNLU);
	}
	memNLU += (len1+1)*iword + len*dword;
	uval = Unzval_br_ptr[ljb_i];
	mybufmax[2] = SUPERLU_MAX( mybufmax[2], len1 );
	mybufmax[3] = SUPERLU_MAX( mybufmax[3], len );
	index[0] = nrbu;  /* Number of column blocks */
	index[1] = len;   /* Total length of nzval[] */
	index[2] = len1;  /* Total length of index */
	index[len1] = -1; /* End marker */
	next_ind = BR_HEADER;
	next_val = 0;
	for (k = 0; k < nrbu; k++) {
	  gb = LUb_number[k];
	  lb = LBj( gb, grid );
	  len = LUb_length[lb];
	  LUb_length[lb] = 0;  /* Reset vector of block length */
	  index[next_ind++] = gb; /* Descriptor */
	  index[next_ind++] = len;
	  LUb_indptr[lb] = next_ind;
	  for (; next_ind < LUb_indptr[lb] + SuperSize( gb ); next_ind++)
	    index[next_ind] = FstBlockC( jb + 1 );
	  LUb_valptr[lb] = next_val;
	  next_val += len;
	}
	/* Propagate the fstnz subscripts to Ufstnz_br_ptr[],
	   and the initial values of A from SPA into Unzval_br_ptr[]. */
	for (i = xusub[ljb_i]; i < xusub[ljb_i+1]; i++) {
	  jcol = usub[i];
	  gb = BlockNum( jcol );
	  
	  if ( mycol == PCOL( gb, grid ) ) {
	    lb = LBj( gb, grid );
	    k = LUb_indptr[lb]; /* Start fstnz in index */
	    index[k + jcol - FstBlockC( gb )] = FstBlockC( jb );
	  }
	}  /* for i ... */
	
	for (i = 0; i < nrbu; i++) {
	  gb = LUb_number[i];
	  lb = LBj( gb, grid );   
	  next_ind = LUb_indptr[lb];
	  k = FstBlockC( jb + 1);
	  jcol = ilsum_j[lb];
	  for (jj = 0; jj < SuperSize( gb ); jj++, jcol++) {
	    dense_col = dense;
	    j = index[next_ind+jj];
	    for (ii = j; ii < k; ii++) {
	      uval[LUb_valptr[lb]++] = dense_col[jcol];
	      dense_col[jcol] = zero;
	      dense_col += ldaspa_j;	      
	    }
	  }
	}
      } else {
	Ufstnz_br_ptr[ljb_i] = NULL;
	Unzval_br_ptr[ljb_i] = NULL;
      } /* if nrbu ... */	
    } /* if myrow == jbrow */
    
      /*------------------------------------------------
       * SET UP L BLOCKS.
       *------------------------------------------------*/
    if (mycol == jbcol) {  /* Block column jb in my process column */
      /* Scatter A_inf into SPA. */
      for (j = ilsum_j[ljb_j], dense_col = dense; j < ilsum_j[ljb_j] + nsupc; j++) {
	for (i = ainf_colptr[j]; i < ainf_colptr[j+1]; i++) {
	  irow = ainf_rowind[i];
	  if (irow >= n) printf ("Pe[%d] ERR1\n", iam);
	  gb = BlockNum( irow );
	  if (gb >= nsupers) printf ("Pe[%d] ERR5\n", iam);
	  if ( myrow == PROW( gb, grid ) ) {
	    lb = LBi( gb, grid );
	    irow = ilsum[lb] + irow - FstBlockC( gb );
	    if (irow >= ldaspa) printf ("Pe[%d] ERR0\n", iam);
	    dense_col[irow] = ainf_val[i];
	  }
	}
	dense_col += ldaspa;
      }      
      
      /* sort the indices of the diagonal block at the beginning of xlsub */
      if (myrow == jbrow) {
	k = xlsub[ljb_j];
	for (i = xlsub[ljb_j]; i < xlsub[ljb_j+1]; i++) {
	  irow = lsub[i];
	  if (irow < nsupc + fsupc && i != k+irow-fsupc) {
	    lsub[i] = lsub[k + irow - fsupc];
	    lsub[k + irow - fsupc] = irow;
	    i --;
	  }
	}
      }
      
      /* Count number of blocks and length of each block. */
      nrbl = 0;
      len = 0; /* Number of row subscripts I own. */
      kseen = 0;
      for (i = xlsub[ljb_j]; i < xlsub[ljb_j+1]; i++) {
	irow = lsub[i];
	gb = BlockNum( irow ); /* Global block number */	  
	pr = PROW( gb, grid ); /* Process row owning this block */
	if ( pr != jbrow && fsendx_plist[ljb_j][pr] == EMPTY &&
	     myrow == jbrow) {
	  fsendx_plist[ljb_j][pr] = YES;
	  ++nfsendx;
	}
	if ( myrow == pr ) {
	  lb = LBi( gb, grid );  /* Local block number */
	  if (Lrb_marker[lb] <= jb) { /* First see this block */
	    Lrb_marker[lb] = jb + 1;
	    LUb_length[lb] = 1;
	    LUb_number[nrbl++] = gb;
	    if ( gb != jb ) /* Exclude diagonal block. */
	      ++fmod[lb]; /* Mod. count for forward solve */
	    if ( kseen == 0 && myrow != jbrow ) {
	      ++nfrecvx;
	      kseen = 1;
	    }
#if ( PRNTlevel>=1 )
	    ++nLblocks;
#endif
	  } else 
	    ++LUb_length[lb];	    
	  ++len;
	}
      } /* for i ... */
      
      if ( nrbl ) { /* Do not ensure the blocks are sorted! */
	/* Set up the initial pointers for each block in 
	   index[] and nzval[]. */
	/* If I am the owner of the diagonal block, order it first in LUb_number.
	   Necessary for SuperLU_DIST routines */
	kseen = EMPTY;
	for (j = 0; j < nrbl; j++) {
	  if (LUb_number[j] == jb)
	    kseen = j;
	}
	if (kseen != EMPTY && kseen != 0) {
	  LUb_number[kseen] = LUb_number[0];
	  LUb_number[0] = jb;
	}
	
	/* Add room for descriptors */
	len1 = len + BC_HEADER + nrbl * LB_DESCRIPTOR;
	if ( !(index = intMalloc_dist(len1)) ) {
	  fprintf (stderr, "Malloc fails for index[]");
	  return (memDist + memNLU);
	}
	Lrowind_bc_ptr[ljb_j] = index;
	if (!(Lnzval_bc_ptr[ljb_j] = 
	      doublecomplexMalloc_dist(len*nsupc))) {
	  fprintf(stderr, "Malloc fails for Lnzval_bc_ptr[*][] col block %d ", jb);
	  return (memDist + memNLU);
	}
	memNLU += len1*iword + len*nsupc*dword;
	
	lusup = Lnzval_bc_ptr[ljb_j];
	mybufmax[0] = SUPERLU_MAX( mybufmax[0], len1 );
	mybufmax[1] = SUPERLU_MAX( mybufmax[1], len*nsupc );
	mybufmax[4] = SUPERLU_MAX( mybufmax[4], len );
	index[0] = nrbl;  /* Number of row blocks */
	index[1] = len;   /* LDA of the nzval[] */
	next_ind = BC_HEADER;
	next_val = 0;
	for (k = 0; k < nrbl; ++k) {
	  gb = LUb_number[k];
	  lb = LBi( gb, grid );
	  len = LUb_length[lb];
	  LUb_length[lb] = 0;
	  index[next_ind++] = gb; /* Descriptor */
	  index[next_ind++] = len; 
	  LUb_indptr[lb] = next_ind;
	    LUb_valptr[lb] = next_val;
	    next_ind += len;
	    next_val += len;
	  }
	  /* Propagate the compressed row subscripts to Lindex[],
	     and the initial values of A from SPA into Lnzval[]. */
	  len = index[1];  /* LDA of lusup[] */
	  for (i = xlsub[ljb_j]; i < xlsub[ljb_j+1]; i++) {
	    irow = lsub[i];
	    gb = BlockNum( irow );
	    if ( myrow == PROW( gb, grid ) ) {
	      lb = LBi( gb, grid );
	      k = LUb_indptr[lb]++; /* Random access a block */
	      index[k] = irow;
	      k = LUb_valptr[lb]++;
	      irow = ilsum[lb] + irow - FstBlockC( gb );
	      for (j = 0, dense_col = dense; j < nsupc; ++j) {
		lusup[k] = dense_col[irow];
		dense_col[irow] = zero;
		k += len;
		dense_col += ldaspa;
	      }
	    }
	  } /* for i ... */
	} else {
	  Lrowind_bc_ptr[ljb_j] = NULL;
	  Lnzval_bc_ptr[ljb_j] = NULL;
	} /* if nrbl ... */		  
      } /* if mycol == pc */
  } /* for jb ... */

  SUPERLU_FREE(ilsum_j);
  SUPERLU_FREE(Urb_marker);
  SUPERLU_FREE(LUb_length);
  SUPERLU_FREE(LUb_indptr);
  SUPERLU_FREE(LUb_number);
  SUPERLU_FREE(LUb_valptr);
  SUPERLU_FREE(Lrb_marker);
  SUPERLU_FREE(dense);
  
  /* Free the memory used for storing L and U */
  SUPERLU_FREE(xlsub); SUPERLU_FREE(xusub);
  if (lsub != NULL)
    SUPERLU_FREE(lsub);  
  if (usub != NULL)
    SUPERLU_FREE(usub);
  
  /* Free the memory used for storing A */
  SUPERLU_FREE(ainf_colptr);
  if (ainf_rowind != NULL) {
    SUPERLU_FREE(ainf_rowind);
    SUPERLU_FREE(ainf_val);
  }
  SUPERLU_FREE(asup_rowptr);
  if (asup_colind != NULL) {
    SUPERLU_FREE(asup_colind);	
    SUPERLU_FREE(asup_val);	
  }
  
  /* exchange information about bsendx_plist in between column of processors */
  k = SUPERLU_MAX( grid->nprow, grid->npcol);
  if ( !(recvBuf = (int_t *) SUPERLU_MALLOC(nsupers*k*iword)) ) {
    fprintf (stderr, "Malloc fails for recvBuf[].");
    return (memDist + memNLU);
  }
  if ( !(nnzToRecv = (int *) SUPERLU_MALLOC(nprocs*sizeof(int))) ) {
    fprintf (stderr, "Malloc fails for nnzToRecv[].");
    return (memDist + memNLU);
  }
  if ( !(ptrToRecv = (int *) SUPERLU_MALLOC(nprocs*sizeof(int))) ) {
    fprintf (stderr, "Malloc fails for ptrToRecv[].");
    return (memDist + memNLU);
  }
  if ( !(nnzToSend = (int *) SUPERLU_MALLOC(nprocs*sizeof(int))) ) {
    fprintf (stderr, "Malloc fails for nnzToRecv[].");
    return (memDist + memNLU);
  }
  if ( !(ptrToSend = (int *) SUPERLU_MALLOC(nprocs*sizeof(int))) ) {
    fprintf (stderr, "Malloc fails for ptrToRecv[].");
    return (memDist + memNLU);
  }
  
  if (memDist < (nsupers*k*iword +4*nprocs * sizeof(int)))
    memDist = nsupers*k*iword +4*nprocs * sizeof(int);
  
  for (p = 0; p < nprocs; p++)
    nnzToRecv[p] = 0;
  
  for (jb = 0; jb < nsupers; jb++) {
    jbcol = PCOL( jb, grid );
    jbrow = PROW( jb, grid );
    p = PNUM(jbrow, jbcol, grid);
    nnzToRecv[p] += grid->npcol;
  }    
  i = 0;
  for (p = 0; p < nprocs; p++) {
    ptrToRecv[p] = i;
    i += nnzToRecv[p];
    ptrToSend[p] = 0;
    if (p != iam)
      nnzToSend[p] = nnzToRecv[iam];
    else
      nnzToSend[p] = 0;
  }
  nnzToRecv[iam] = 0;
  i = ptrToRecv[iam];
  for (jb = 0; jb < nsupers; jb++) {
    jbcol = PCOL( jb, grid );
    jbrow = PROW( jb, grid );
    p = PNUM(jbrow, jbcol, grid);
    if (p == iam) {
      ljb_j = LBj( jb, grid ); /* Local block number column wise */	
      for (j = 0; j < grid->npcol; j++, i++)
	recvBuf[i] = ToSendR[ljb_j][j];
    }
  }   
  
  MPI_Alltoallv (&(recvBuf[ptrToRecv[iam]]), nnzToSend, ptrToSend, mpi_int_t,
		 recvBuf, nnzToRecv, ptrToRecv, mpi_int_t, grid->comm);
  
  for (jb = 0; jb < nsupers; jb++) {
    jbcol = PCOL( jb, grid );
    jbrow = PROW( jb, grid );
    p = PNUM(jbrow, jbcol, grid);
    ljb_j = LBj( jb, grid ); /* Local block number column wise */	
    ljb_i = LBi( jb, grid ); /* Local block number row wise */	
    /* (myrow == jbrow) {
       if (ToSendD[ljb_i] == YES)
       ToRecv[jb] = 1;
       }
       else {
       if (recvBuf[ptrToRecv[p] + mycol] == YES)
       ToRecv[jb] = 2;
       } */
    if (recvBuf[ptrToRecv[p] + mycol] == YES) {
      if (myrow == jbrow)
	ToRecv[jb] = 1;
      else
	ToRecv[jb] = 2;
    }
    if (mycol == jbcol) {
      for (i = 0, j = ptrToRecv[p]; i < grid->npcol; i++, j++) 
	ToSendR[ljb_j][i] = recvBuf[j];  
      ToSendR[ljb_j][mycol] = EMPTY;
    }
    ptrToRecv[p] += grid->npcol;
  }   
  
  /* exchange information about bsendx_plist in between column of processors */
  MPI_Allreduce ((*bsendx_plist), recvBuf, nsupers_j * grid->nprow, mpi_int_t,
		 MPI_MAX, grid->cscp.comm);
  
  for (jb = 0; jb < nsupers; jb ++) {
    jbcol = PCOL( jb, grid);
    jbrow = PROW( jb, grid);
    if (mycol == jbcol) {
      ljb_j = LBj( jb, grid ); /* Local block number column wise */	
      if (myrow == jbrow ) {
	for (k = ljb_j * grid->nprow; k < (ljb_j+1) * grid->nprow; k++) {
	  (*bsendx_plist)[k] = recvBuf[k];
	  if ((*bsendx_plist)[k] != EMPTY)
	    nbsendx ++;
	}
      }
      else {
	for (k = ljb_j * grid->nprow; k < (ljb_j+1) * grid->nprow; k++) 
	  (*bsendx_plist)[k] = EMPTY;
      }
    }
  }
  
  SUPERLU_FREE(nnzToRecv);
  SUPERLU_FREE(ptrToRecv);
  SUPERLU_FREE(nnzToSend);
  SUPERLU_FREE(ptrToSend);
  SUPERLU_FREE(recvBuf);
  
  Llu->Lrowind_bc_ptr = Lrowind_bc_ptr;
  Llu->Lnzval_bc_ptr = Lnzval_bc_ptr;
  Llu->Ufstnz_br_ptr = Ufstnz_br_ptr;
  Llu->Unzval_br_ptr = Unzval_br_ptr;
  Llu->ToRecv = ToRecv;
  Llu->ToSendD = ToSendD;
  Llu->ToSendR = ToSendR;
  Llu->fmod = fmod;
  Llu->fsendx_plist = fsendx_plist;
  Llu->nfrecvx = nfrecvx;
  Llu->nfsendx = nfsendx;
  Llu->bmod = bmod;
  Llu->bsendx_plist = bsendx_plist;
  Llu->nbrecvx = nbrecvx;
  Llu->nbsendx = nbsendx;
  Llu->ilsum = ilsum;
  Llu->ldalsum = ldaspa;
  LUstruct->Glu_persist = Glu_persist;	
#if ( PRNTlevel>=1 )
  if ( !iam ) printf(".. # L blocks %d\t# U blocks %d\n",
		     nLblocks, nUblocks);
#endif
  
  /* Find the maximum buffer size. */
  MPI_Allreduce(mybufmax, Llu->bufmax, NBUFFERS, mpi_int_t, 
		MPI_MAX, grid->comm);
  
#if ( DEBUGlevel>=1 )
  /* Memory allocated but not freed:
     ilsum, fmod, fsendx_plist, bmod, bsendx_plist,
     ToRecv, ToSendR, ToSendD
  */
  CHECK_MALLOC(iam, "Exit dist_psymbtonum()");
#endif
    
  return (- (memDist+memNLU));
} /* dist_psymbtonum */

