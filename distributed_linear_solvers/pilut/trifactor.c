/*
 * trifactor.c
 *
 * This file contains a number of fuction that are used in solving
 * the triangular systems resulting from the ILUT
 *
 * Started 11/13/95
 * George
 *
 * 7/8
 *  - seperate SetUpFactor from SetUpLUFactor and verify
 * 7/9
 *  - MPI support, adding to the comm structure
 *  - timing of the LDUSolve. The computation is very scalable, but the
 *    communication and sync is not. Partially this comes from sending
 *    zero length messages. I'll fix that.
 * 7/10
 *  - MPI and validation. Doesn't seem to work with Edinburgh, but
 *    I haven't the slightest idea why not. (Artifact of running
 *    along with shmem?)
 * 7/11
 *  - cleaned up code a little. Added timer macros.
 *
 * $Id$
 */

#include "./DistributedMatrixPilutSolver.h"
#include "ilu.h"


/*************************************************************************
* This function performs the forward and backward substitution.
* It solves the system LDUx = b.
**************************************************************************/
void LDUSolve(DataDistType *ddist, FactorMatType *ldu, double *x, double *b,
                   hypre_PilutSolverGlobals *globals)
{
  int ii, i, j, k, l, TAG;
  int nlevels, snbrpes, rnbrpes;
  int *perm, *iperm, *nnodes, *rowptr, *colind,
    *spes, *sptr, *sind, *auxsptr, *rpes, *rdone, *rnum;
  double *lx, *ux, *values, *dvalues, *gatherbuf, **raddr, xx;
  MPI_Status Status;

  /* PrintLine("LDUSolve start", globals); */

  lnrows    = ddist->ddist_lnrows;
  perm      = ldu->perm;
  iperm     = ldu->iperm;
  nnodes    = ldu->nnodes;
  nlevels   = ldu->nlevels;
  dvalues   = ldu->dvalues;
  gatherbuf = ldu->gatherbuf;

  lx = ldu->lx;
  ux = ldu->ux;

  /******************************************************************
  * Do the L(lx) = b, first 
  *******************************************************************/
  snbrpes = ldu->lcomm.snbrpes;
  spes    = ldu->lcomm.spes;
  sptr    = ldu->lcomm.sptr;
  sind    = ldu->lcomm.sind;
  auxsptr = ldu->lcomm.auxsptr;
  memcpy_idx(auxsptr, sptr, snbrpes+1);

  rnbrpes = ldu->lcomm.rnbrpes;
  raddr   = ldu->lcomm.raddr;
  rpes    = ldu->lcomm.rpes;
  rdone   = ldu->lcomm.rdone;
  for (i=0; i<rnbrpes; i++)
    rdone[i] = 0 ;

  rowptr = ldu->lrowptr;
  colind = ldu->lcolind;
  values = ldu->lvalues;


  /* Do the local first.
   * For forward substitution we do local+1st MIS == nnodes[1] (NOT [0]!) */
  for (i=0; i<nnodes[max(0,min(1,nlevels))]; i++) {
    xx = 0.0;
    for (j=rowptr[i]; j<rowptr[i+1]; j++) 
      xx += values[j]*lx[colind[j]];
    lx[i] = b[perm[i]] - xx;
  }

  /* Do the distributed next */
  for (ii=1; ii<nlevels; ii++) {
    /* make MPI LX tags unique for this level (so we don't have to sync) */
    TAG = (TAG_LDU_lx | ii);

    /* Send the required lx elements to the appropriate processors */
    for (i=0; i<snbrpes; i++) {
      if (sptr[i+1] > auxsptr[i]  &&  sind[auxsptr[i]]<nnodes[ii]) { /* Something to send */
        for (j=auxsptr[i], l=0;   j<sptr[i+1] && sind[j]<nnodes[ii];   j++, l++) 
          gatherbuf[l] = lx[sind[j]];

	MPI_Send( gatherbuf, l, MPI_DOUBLE_PRECISION,
		  spes[i], TAG, pilut_comm );

        auxsptr[i] = j;
      }
    }

    /* get number of recieves for this level */
    rnum = &(ldu->lcomm.rnum[(ii-1)*rnbrpes]) ;

    /* Recv the required lx elements from the appropriate processors */
    for (i=0; i<rnbrpes; i++) {
      if ( rnum[i] > 0 ) { /* Something to recv */
	MPI_Recv( raddr[i]+rdone[i], rnum[i], MPI_DOUBLE_PRECISION,
		  rpes[i], TAG, pilut_comm, &Status );

	rdone[i] += rnum[i] ;
      }
    }

    /* solve for this MIS set
     * by construction all remote lx elements needed are filled in */
    for (i=nnodes[ii]; i<nnodes[ii+1]; i++) {
      xx = 0.0;
      for (j=rowptr[i]; j<rowptr[i+1]; j++) {
        xx += values[j]*lx[colind[j]];
      }
      lx[i] = b[perm[i]] - xx;
    }
  }


  /******************************************************************
  * Do the U(ly) = (lx), next 
  *******************************************************************/
  snbrpes = ldu->ucomm.snbrpes;
  spes    = ldu->ucomm.spes;
  sptr    = ldu->ucomm.sptr;
  sind    = ldu->ucomm.sind;
  auxsptr = ldu->ucomm.auxsptr;
  memcpy_idx(auxsptr, sptr, snbrpes+1);

  rnbrpes = ldu->ucomm.rnbrpes;
  raddr   = ldu->ucomm.raddr;
  rpes    = ldu->ucomm.rpes;
  rdone   = ldu->ucomm.rdone;
  for (i=0; i<rnbrpes; i++)
    rdone[i] = 0 ;

  rowptr = ldu->urowptr;
  colind = ldu->ucolind;
  values = ldu->uvalues;

  /* Do the distributed */
  for (ii=nlevels; ii>0; ii--) {
    /* Solve for this MIS set
     * by construction all remote lx elements needed are filled in */
    for (i=nnodes[ii]-1; i>=nnodes[ii-1]; i--) {
      xx = 0.0;
      for (j=rowptr[i]; j<rowptr[i+1]; j++) 
        xx += values[j]*ux[colind[j]];
      ux[i] = dvalues[i]*(lx[i] - xx);
    }

    /* make MPI UX tags unique for this level (so we don't have to sync) */
    TAG = (TAG_LDU_ux | ii);

    /* Send the required ux elements to the appropriate processors */
    for (i=0; i<snbrpes; i++) {
      if (sptr[i+1] > auxsptr[i]  &&  sind[auxsptr[i]]>=nnodes[ii-1]) { /* Something to send */
        for (j=auxsptr[i], l=0;   j<sptr[i+1] && sind[j]>=nnodes[ii-1];   j++, l++) 
          gatherbuf[l] = ux[sind[j]];

	MPI_Send( gatherbuf, l, MPI_DOUBLE_PRECISION,
		  spes[i], TAG, pilut_comm );

        auxsptr[i] = j;
      }
    }

    /* get number of recieves for this level */
    rnum = &(ldu->ucomm.rnum[(ii-1)*rnbrpes]);

    /* Recv the required ux elements from the appropriate processors */
    for (i=0; i<rnbrpes; i++) {
      if ( rnum[i] > 0 ) { /* Something to recv */
	MPI_Recv( raddr[i]+rdone[i], rnum[i], MPI_DOUBLE_PRECISION,
		  rpes[i], TAG, pilut_comm, &Status );

	rdone[i] += rnum[i] ;
      }
    }
  }

  /* Do the local next */
  for (i=nnodes[0]-1; i>=0; i--) {
    xx = 0.0;
    for (j=rowptr[i]; j<rowptr[i+1]; j++) 
      xx += values[j]*ux[colind[j]];
    ux[i] = dvalues[i]*(lx[i] - xx);
  }


  /* Permute the solution to back to x */
  for (i=0; i<lnrows; i++)
    x[i] = ux[iperm[i]];
}


/*************************************************************************
* This function sets-up the communication parameters for the forward
* and backward substitution, and relabels the L and U matrices 
**************************************************************************/
void SetUpLUFactor(DataDistType *ddist, FactorMatType *ldu, int maxnz,
                   hypre_PilutSolverGlobals *globals )
{
  int i, maxsend;
  int *petotal, *rind, *imap;

  petotal = idx_malloc(npes+1,       "SetUpLUFactor: petotal");
  rind    = idx_malloc(ddist->ddist_nrows, "SetUpLUFactor: rind"   );
  imap    = idx_malloc_init(ddist->ddist_nrows, -1, "SetUpLUFactor: imap");

  /* This is the global maximum for both L and U */
  maxsend = 0;

  /* Work on L first */
  SetUpFactor( ddist, ldu, maxnz,   petotal, rind, imap, &maxsend,   true,
               globals  );

  /* Now work on U   */
  SetUpFactor( ddist, ldu, maxnz,   petotal, rind, imap, &maxsend,   false,
               globals );

  /* Allocate memory for the gather buffer. This is an overestimate */
  ldu->gatherbuf = fp_malloc(maxsend, "SetUpLUFactor: ldu->gatherbuf");

  free_multi(petotal, rind, imap, -1);
}

/*************************************************************************
* This function sets-up the communication parameters for the forward
* and backward substitution, and relabels the L and U matrices.
* This function is called twice--once for L and once for U. DoingL
* differentiates the two calls for the minor differences between them.
* These differences are marked by **** in comments
**************************************************************************/
void SetUpFactor(DataDistType *ddist, FactorMatType *ldu, int maxnz,
		 int *petotal, int *rind, int *imap,
		 int *maxsendP, bool DoingL,
                   hypre_PilutSolverGlobals *globals )
{
  int i, ii, j, k, l, 
    nlevels, nrecv, nsend, snbrpes, rnbrpes;
  int *rowdist, *sptr, *sind, *spes, *rpes,
    *perm, *iperm, *newrowptr, *newcolind,
    *srowptr, *erowptr, *colind, *rnum ;
  double *newvalues, *values, *x, **raddr;
  TriSolveCommType *TriSolveComm;
  MPI_Status Status;
  MPI_Datatype MyColType_rnbr;

  /* data common to L and U */
  lnrows   = ddist->ddist_lnrows;
  nlevels  = ldu->nlevels;
  rowdist  = ddist->ddist_rowdist;
  firstrow = rowdist[mype];
  lastrow  = rowdist[mype+1];
  perm     = ldu->perm;
  iperm    = ldu->iperm;

  /**** choose between L and U data ****/
  srowptr = (DoingL  ?  ldu->lsrowptr  :  ldu->usrowptr);
  erowptr = (DoingL  ?  ldu->lerowptr  :  ldu->uerowptr);
  colind  = (DoingL  ?  ldu->lcolind   :  ldu->ucolind );
  values  = (DoingL  ?  ldu->lvalues   :  ldu->uvalues );
  TriSolveComm    = (DoingL  ?  &(ldu->lcomm)  :  &(ldu->ucomm));

  /* Determine the needed indices for L (U) */
  nrecv   = 0;
  for (ii=ldu->nnodes[0]; ii<lnrows; ii++) {
    i = perm[ii];
    for (j=srowptr[i]; j<erowptr[i]; j++) {
      k = colind[j];
      if ((k < firstrow || k >= lastrow) && imap[k] == -1) {
        rind[nrecv++] = k;
        imap[k] = -2;
      }
    }
  }
  sincsort_fast(nrecv, rind);

  /**** select between L and U "x" vectors ****/
  if ( DoingL ) {
    ldu->lxlen = GlobalSEMax(lnrows+nrecv, pilut_comm );
    x = ldu->lx  = fp_malloc_init(ldu->lxlen, 0, "SetUpFactor: ldu->lx");
  }
  else {
    ldu->uxlen = GlobalSEMax(lnrows+nrecv, pilut_comm);
    x = ldu->ux  = fp_malloc_init(ldu->uxlen, 0, "SetUpFactor: ldu->ux");
  }

  /* Determine processor boundaries */
  j = 0;
  for (i=0; i<npes; i++) {
    k = j;
    for (; j<nrecv; j++) {
      if (rind[j] >= rowdist[i+1])
        break;
    }
    petotal[i] = j-k;
  }

  /* Tell the processors how many elements I'll be sending */
  rnbrpes = 0;
  for (i=0; i<npes; i++) {
    if (petotal[i] > 0) {
      rnbrpes++;
    }
  }
  TriSolveComm->rnbrpes = rnbrpes ;

  MPI_Alltoall( petotal, 1, MPI_INTEGER,
		lu_recv, 1, MPI_INTEGER, pilut_comm );

  /* Determine to how many processors you will be sending data */
  snbrpes = 0;
  nsend = 0;
  for (i=0; i<npes; i++) {
    if (lu_recv[i] > 0) {
      snbrpes++;
      nsend += lu_recv[i];
      if ((*maxsendP) < lu_recv[i])
        (*maxsendP) = lu_recv[i];
    }
  }
  TriSolveComm->snbrpes = snbrpes;

  /* Allocate sufficient memory for the various data structures for TriSolveComm */
          TriSolveComm->auxsptr = idx_malloc(snbrpes+1, "SetUpFactor: TriSolveComm->auxsptr");
  spes  = TriSolveComm->spes    = idx_malloc(snbrpes,   "SetUpFactor: TriSolveComm->spes"   );
  sptr  = TriSolveComm->sptr    = idx_malloc(snbrpes+1, "SetUpFactor: TriSolveComm->sptr"   );
  sind  = TriSolveComm->sind    = idx_malloc(GlobalSEMax(nsend, pilut_comm), "SetUpFactor: TriSolveComm->sind");

          TriSolveComm->rdone   = idx_malloc(rnbrpes,  "SetUpFactor: TriSolveComm->rpes");
  rpes  = TriSolveComm->rpes    = idx_malloc(rnbrpes,  "SetUpFactor: TriSolveComm->rpes" );
  raddr = TriSolveComm->raddr   = (double**) mymalloc( sizeof(double*)*(rnbrpes+1),
					       "SetUpFactor: TriSolveComm->raddr");

  /* Save send addresses, lengths, and construct spes */
  snbrpes = 0;
  for (i=0; i<npes; i++) {
    if (lu_recv[i] > 0) {
      spes[snbrpes] = i;
      sptr[snbrpes] = lu_recv[i];
      snbrpes++;

      lu_recv[i] = 0;
    }
  }
  assert( TriSolveComm->snbrpes == snbrpes );

  /* Create a sptr array into sind */
  for (i=1; i<snbrpes; i++)
    sptr[i] += sptr[i-1];
  for (i=snbrpes; i>0; i--)
    sptr[i] = sptr[i-1];
  sptr[0] = 0;

  /* Send the rind sets to the processors */
  rnbrpes = 0;
  k = 0;
  for (i=0; i<npes; i++) {
    if (petotal[i] > 0) {
      MPI_Send( rind+k, petotal[i], MPI_INTEGER ,
		i, TAG_SetUp_rind, pilut_comm );

      /* recv info for LDUSolve */
      raddr[rnbrpes] = x + k + lnrows;
      rpes [rnbrpes] = i;
      rnbrpes++;
      k += petotal[i];

      assert( k < ddist->ddist_nrows );
    }
  }
  /* this last one is to compute (raddr[i+1] - raddr[i]) */
  raddr[rnbrpes] = x + k + lnrows;
  assert( TriSolveComm->rnbrpes == rnbrpes );

  for (i=0; i<snbrpes; i++) {
    MPI_Recv( sind+sptr[i], sptr[i+1]-sptr[i], MPI_INTEGER,
	      spes[i], TAG_SetUp_rind, pilut_comm, &Status );
  }

  /* At this point, the set of indexes that you need to send to processors are
     stored in (sptr, sind) */
  /* Apply the iperm[] onto the sind in order to sort them according to MIS */
  for (i=0; i<nsend; i++) {
    CheckBounds(firstrow, sind[i], lastrow, globals);
    sind[i] = iperm[sind[i]-firstrow];
  }

  /**** Go and do a segmented sort of the elements of the sind.
   **** L is sorted increasing, U is sorted decreasing. ****/
  if ( DoingL ) {
    for (i=0; i<snbrpes; i++) 
      sincsort_fast(sptr[i+1]-sptr[i], sind+sptr[i]);
  }
  else {
    for (i=0; i<snbrpes; i++)
      sdecsort_fast(sptr[i+1]-sptr[i], sind+sptr[i]);
  }

  /* Apply the perm[] onto the sind to take it back to the original index space */
  for (i=0; i<nsend; i++) {
    CheckBounds(0, sind[i], lnrows, globals);
    sind[i] = perm[sind[i]]+firstrow;
  }

  /* Write them back to the processors that send them to me */
  for (i=0; i<snbrpes; i++) {
    MPI_Send( sind+sptr[i], sptr[i+1]-sptr[i], MPI_INTEGER,
	      spes[i], TAG_SetUp_reord, pilut_comm );
  }

  /* Recv them from the processors that send them to me */
  k = 0;
  for (i=0; i<npes; i++) {
    if (petotal[i] > 0) {
      MPI_Recv( rind+k, petotal[i], MPI_INTEGER,
	        i, TAG_SetUp_reord, pilut_comm, &Status );
      k += petotal[i];
    }
  }

  /* Apply the iperm[] onto the sind for easy indexing during solution */
  for (i=0; i<nsend; i++) 
    sind[i] = iperm[sind[i]-firstrow];

  /* Create imap array for relabeling L */
  for (i=0; i<nrecv; i++) {
    assert(imap[rind[i]] == -2);
    imap[rind[i]] = lnrows+i;
  }

  /* Construct the IMAP array of the locally stored rows */
  for (i=0; i<lnrows; i++) 
    imap[firstrow+perm[i]] = i;

  /* rnum is a 2D array of nlevels rows of rnbrpes columns each */
  TriSolveComm->rnum = idx_malloc(nlevels * rnbrpes, "SetUpFactor: TriSolveComm->rnum");
        rnum = idx_malloc(nlevels, "SetUpFactor: rnum"      );
  memcpy_idx(TriSolveComm->auxsptr, sptr, snbrpes+1);

  /**** send the number of elements we are going to send to each PE.
   **** Note the inner for loop has no body, and L and U differ slightly.
   **** For L, rnum[nlevels-1] is undefined and rnum only has (nlevels-1) entries ****/
  for (i=0; i<snbrpes; i++) {
    if ( DoingL ) {
      for (ii=1; ii<nlevels; ii++) {
	for (j=TriSolveComm->auxsptr[i], l=0;   j<sptr[i+1] && sind[j]<ldu->nnodes[ii];     j++, l++)
	  ;

	rnum[ii-1] = l;
	TriSolveComm->auxsptr[i] = j;
      }
      rnum[nlevels-1] = 0; /* never used */
    }
    else {
      for (ii=nlevels; ii>0; ii--) {
	for (j=TriSolveComm->auxsptr[i], l=0;   j<sptr[i+1] && sind[j]>=ldu->nnodes[ii-1];  j++, l++)
	  ;

	rnum[ii-1] = l;
	TriSolveComm->auxsptr[i] = j;
      }
    }

    MPI_Send( rnum, nlevels, MPI_INTEGER,
	      spes[i], TAG_SetUp_rnum, pilut_comm );
  }

  if (rnum) free(rnum);

  /* recieve data as columns rather than rows */
  MPI_Type_vector( nlevels, 1, rnbrpes, MPI_INTEGER, &MyColType_rnbr );
  MPI_Type_commit( &MyColType_rnbr );

  /* recieve each column */
  for (i=0; i<rnbrpes; i++) {
    MPI_Recv( TriSolveComm->rnum+i, 1, MyColType_rnbr,
	      rpes[i], TAG_SetUp_rnum, pilut_comm, &Status );
  }

  MPI_Type_free( &MyColType_rnbr );

  /* Now, go and create the renumbered L (U) that is also in CSR format */
  newrowptr = idx_malloc(lnrows+1,     "SetUpFactor: rowptr");
  newcolind = idx_malloc(lnrows*maxnz, "SetUpFactor: colind");
  newvalues =  fp_malloc(lnrows*maxnz, "SetUpFactor: values");

  newrowptr[0] = 0;
  k = 0;
  for (ii=0; ii<lnrows; ii++) {
    i = perm[ii];
    for (j=srowptr[i]; j<erowptr[i]; j++) {
      assert(imap[colind[j]] != -1);
      newcolind[k] = imap[colind[j]];
      newvalues[k] = values[j];
      k++;
    }
    newrowptr[ii+1] = k;
  }

  /**** Store new L (DU) into LDU ****/
  if ( DoingL ) {
    /* Free memory that stored the L so far and relink the data structures */
    free_multi(ldu->lsrowptr, ldu->lerowptr, ldu->lcolind, ldu->lvalues, -1);
    ldu->lrowptr = newrowptr;
    ldu->lcolind = newcolind;
    ldu->lvalues = newvalues;
  }
  else {
    /* Use uvalues as a buffer to permute the dvalues */
    for (i=0; i<lnrows; i++)
      values[i] = ldu->dvalues[perm[i]];
    memcpy_fp(ldu->dvalues, values, lnrows);

    /* Free memory that stored the U so far and relink the data structures */
    free_multi(ldu->usrowptr, ldu->uerowptr, ldu->ucolind, ldu->uvalues, -1);
    ldu->urowptr = newrowptr;
    ldu->ucolind = newcolind;
    ldu->uvalues = newvalues;
  }

  /* Reset the imap by only touching the appropriate elements */
  for (i=0; i<nrecv; i++)
    imap[rind[i]] = -1;
  for (i=0; i<lnrows; i++) 
    imap[firstrow+i] = -1;
}

