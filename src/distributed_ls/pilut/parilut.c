/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*
 * parilut.c
 *
 * This file implements the parallel phase of the hypre_ILUT algorithm
 *
 * Started 10/21/95
 * George
 *
 * Taken over by MRGates 7/1/97.
 *
 * 7/8
 *  - added rowlen to rmat and verified
 * 7/11
 *  - MPI and validated
 *  - fixed one more problem with rowlen (a rcolind--)
 * 7/25
 *  - replaced George's reduction and second drop functions with my own
 *    The biggest difference is since I allow non-diagonal MIS sets then
 *    there is fill into L and the L must be processed in the correct
 *    order. Therefore I reverted to using the workspace as in serilut.
 *    (Note that this changes our answer so it is hard to verify.)
 *  - seperated the second drop function into four stages:
 *     1) drop below rtol
 *     2) seperate LU entries
 *     3) update L for the row
 *     4) form nrmat or DU for the row
 * 7/28
 *  - finished the local factorization to reduce non-diagonal sets.
 *    This allows fillin, but the remote reduction still does not since
 *    all the necesary rows are not recieved yet (and thus it doesn't
 *    know what rows are actually in the MIS--otherwise we could just
 *    ignore MIS fillin for the moment).
 * 7/29
 *  - send all factored rows, not just the requested ones (changes hypre_EraseMap also)
 *  - add the (maxnz+2) factor to the map, so for outside nodes, l is the exact index
 *  - removed inrval, instead using the new permutation to know what rows are MIS
 *  - removed map from the cinfo, since it was never refered to but globally
 * 8/1
 *  - implemented split PE numbering. change VPE(x) to (x) to get back unsplit numbering.
 * 8/6
 *  - Removed split PE numbering. After further testing, this does not seem to be an
 *    improvement, since it increases the number of levels. See par_split.c for that code.
 */

#include "./DistributedMatrixPilutSolver.h"
#include "ilu.h"

/*************************************************************************
* This function performs hypre_ILUT on the boundary nodes via MIS computation
**************************************************************************/
void hypre_ParILUT(DataDistType *ddist, FactorMatType *ldu,
             ReduceMatType *rmat, HYPRE_Int gmaxnz, HYPRE_Real tol,
             hypre_PilutSolverGlobals *globals )
{
  HYPRE_Int nmis, nlevel;
  CommInfoType cinfo;
  HYPRE_Int *perm, *iperm, *newiperm, *newperm;
  ReduceMatType *rmats[2], nrmat;

#ifdef HYPRE_DEBUG
  hypre_PrintLine("hypre_ILUT start", globals);
#endif

  /* Initialize globals */
  global_maxnz = gmaxnz;

  nrows    = ddist->ddist_nrows;
  lnrows   = ddist->ddist_lnrows;
  firstrow = ddist->ddist_rowdist[mype];
  lastrow  = ddist->ddist_rowdist[mype+1];
  perm  = ldu->perm;
  iperm = ldu->iperm;

  ndone = rmat->rmat_ndone;
  ntogo = rmat->rmat_ntogo;
  nleft = hypre_GlobalSESum(ntogo, pilut_comm);

  rmats[0] = rmat;
  rmats[1] = &nrmat;

  /* Initialize and allocate structures, including global workspace */
  hypre_ParINIT( &nrmat, &cinfo, ddist->ddist_rowdist, globals );

  /* Copy the old perm into new perm vectors at the begining.
   * After that this is done more or less automatically */
  newperm  = hypre_idx_malloc(lnrows, "hypre_ParILUT: newperm");
  newiperm = hypre_idx_malloc(lnrows, "hypre_ParILUT: newiperm");

  hypre_memcpy_idx(newperm,   perm, lnrows);
  hypre_memcpy_idx(newiperm, iperm, lnrows);

  ldu->nnodes[0] = ndone;
  nlevel = 0;

  while( nleft > 0 ) {
    /* hypre_printf("PE %d Nlevel: %d, Nleft: %d, (%d,%d)\n",
     * mype, nlevel, nleft, ndone, ntogo); fflush(stdout); */

    hypre_ComputeCommInfo(rmats[nlevel%2], &cinfo, ddist->ddist_rowdist, globals );
    nmis = hypre_SelectSet(rmats[nlevel%2], &cinfo, perm, iperm, newperm, newiperm, globals );

    hypre_FactorLocal(ldu, rmats[nlevel%2], rmats[(nlevel+1)%2], &cinfo,
          perm, iperm, newperm, newiperm, nmis, tol, globals );

    fflush(stdout); hypre_MPI_Barrier(pilut_comm);
    hypre_SendFactoredRows(ldu, &cinfo, newperm, nmis, globals);
    fflush(stdout); hypre_MPI_Barrier(pilut_comm);

    hypre_ComputeRmat(ldu, rmats[nlevel%2], rmats[(nlevel+1)%2], &cinfo,
          perm, iperm, newperm, newiperm, nmis, tol, globals);

    hypre_EraseMap(&cinfo, newperm, nmis, globals);

    /* copy the new portion of the permutation, and the entire inverse
     * (since updates to the inverse are scattered throughout.) */
    hypre_memcpy_idx(perm+ndone, newperm+ndone,  ntogo );
    hypre_memcpy_idx(iperm,      newiperm,       lnrows);

    /* setup next rmat */
    nlevel++;
    ndone = rmats[nlevel%2]->rmat_ndone = ndone+nmis;
    ntogo = rmats[nlevel%2]->rmat_ntogo = ntogo-nmis;

    nleft = hypre_GlobalSESum(ntogo, pilut_comm);

    if (nlevel > MAXNLEVEL)
      hypre_errexit("Maximum number of levels exceeded!\n", globals);
    ldu->nnodes[nlevel] = ndone;
  }
  ldu->nlevels = nlevel;

  /*hypre_free_multi(jr, jw, lr, w, map,
    nrmat.rmat_rnz,        nrmat.rmat_rrowlen,  nrmat.rmat_rcolind,
             nrmat.rmat_rvalues,
             cinfo.gatherbuf,  cinfo.rrowind,  cinfo.rnbrind,   cinfo.rnbrptr,
             cinfo.snbrind, cinfo.srowind, cinfo.snbrptr,
             cinfo.incolind,  cinfo.invalues,
             newperm, newiperm, vrowdist, -1);*/
  hypre_TFree(jr, HYPRE_MEMORY_HOST);
  hypre_TFree(jw, HYPRE_MEMORY_HOST);
  hypre_TFree(hypre_lr, HYPRE_MEMORY_HOST);
  hypre_TFree(w, HYPRE_MEMORY_HOST);
  hypre_TFree(pilut_map, HYPRE_MEMORY_HOST);
  hypre_TFree(nrmat.rmat_rnz, HYPRE_MEMORY_HOST);
  hypre_TFree(nrmat.rmat_rrowlen, HYPRE_MEMORY_HOST);
  hypre_TFree(nrmat.rmat_rcolind, HYPRE_MEMORY_HOST);
  hypre_TFree(nrmat.rmat_rvalues, HYPRE_MEMORY_HOST);
  hypre_TFree(cinfo.gatherbuf, HYPRE_MEMORY_HOST);
  hypre_TFree(cinfo.rrowind, HYPRE_MEMORY_HOST);
  hypre_TFree(cinfo.rnbrind, HYPRE_MEMORY_HOST);
  hypre_TFree(cinfo.rnbrptr, HYPRE_MEMORY_HOST);
  hypre_TFree(cinfo.snbrind, HYPRE_MEMORY_HOST);
  hypre_TFree(cinfo.srowind, HYPRE_MEMORY_HOST);
  hypre_TFree(cinfo.snbrptr, HYPRE_MEMORY_HOST);
  hypre_TFree(cinfo.incolind, HYPRE_MEMORY_HOST);
  hypre_TFree(cinfo.invalues, HYPRE_MEMORY_HOST);
  hypre_TFree(newperm, HYPRE_MEMORY_HOST);
  hypre_TFree(newiperm, HYPRE_MEMORY_HOST);
  hypre_TFree(vrowdist, HYPRE_MEMORY_HOST);

  jr = NULL;
  jw = NULL;
  hypre_lr = NULL;
  w  = NULL;

#ifdef HYPRE_DEBUG
  hypre_PrintLine("hypre_ParILUT done", globals);
#endif
}


/*************************************************************************
* This function determines communication info. It assumes (and leaves)
* the map in a zero state. If memory requirements increase, it will
* free and reallocate memory for send/recieve buffers. Usually memory
* doesn't increase since the problem size is decreasing each iteration.
*
* The rrowind and srowind now have two bits packed into them, so
* (rowind>>2) is the index, rowind & 0x1 is lo, rowind & 0x2 is hi,
* where lo==1 means the lower half has this nonzero col index, hi==1 means
* the upper half has this nonzero col index.
**************************************************************************/
void hypre_ComputeCommInfo(ReduceMatType *rmat, CommInfoType *cinfo, HYPRE_Int *rowdist,
             hypre_PilutSolverGlobals *globals)
{
  HYPRE_Int i, ir, j, k, penum;
  HYPRE_Int nrecv, nsend, rnnbr, snnbr, maxnrecv, maxnsend;
  HYPRE_Int *rnz, *rcolind;
  HYPRE_Int *rrowind,  *rnbrptr,  *rnbrind, *srowind, *snbrind, *snbrptr;
  hypre_MPI_Status Status ;
  hypre_MPI_Request *index_requests;

#ifdef HYPRE_DEBUG
  hypre_PrintLine("hypre_ComputeCommInfo", globals);
#endif
#ifdef HYPRE_TIMING
  hypre_BeginTiming( globals->CCI_timer  );
#endif

  rnz = rmat->rmat_rnz;

  rnbrind  = cinfo->rnbrind;
  rnbrptr  = cinfo->rnbrptr;
  rrowind  = cinfo->rrowind;

  snbrind  = cinfo->snbrind;
  snbrptr  = cinfo->snbrptr;

  /* Determine the indices that are needed */
  nrecv  = 0;
  for (ir=0; ir<ntogo; ir++) {
    rcolind = rmat->rmat_rcolind[ir];
    for (j=1; j<rnz[ir]; j++) {
      k = rcolind[j];
      hypre_CheckBounds(0, k, nrows, globals);
      if ((k < firstrow || k >= lastrow) && pilut_map[k] == 0) {
        pilut_map[k] = 1;
        rrowind[nrecv++] = k;
      }
    }
  }

  /* Sort the indices to be received in increasing order */
  hypre_sincsort_fast(nrecv, rrowind);

  /* Determine processor boundaries in the rowind */
  rnnbr = 0;
  rnbrptr[0] = 0;
  for (penum=0, j=0;   penum<npes && j<nrecv;   penum++) {
    k = j;
    for (; j<nrecv; j++) {
      if (rrowind[j] >= rowdist[penum+1])
        break;
    }
    if (j-k > 0) { /* Something for pe penum */
      rnbrind[rnnbr] = penum;
      rnbrptr[++rnnbr] = j;
    }
  }
  cinfo->rnnbr = rnnbr;

  /* reset the map afterwards */
  for (i=0; i<nrecv; i++)
    pilut_map[rrowind[i]] = 0;

  /* Now you know from which processors, and what you need. */
  cinfo->maxntogo = hypre_GlobalSEMax(ntogo, pilut_comm);
  maxnrecv = rnnbr*(cinfo->maxntogo);  /*hypre_GlobalSEMax(nrecv);*/

  /* If memory requirements change, allocate new memory.
   * The first iteration this always occurs -- see hypre_ParINIT */
  if (cinfo->maxnrecv < maxnrecv)
  {
     hypre_TFree(cinfo->incolind, HYPRE_MEMORY_HOST);
     hypre_TFree(cinfo->invalues, HYPRE_MEMORY_HOST);
     cinfo->incolind = hypre_idx_malloc(maxnrecv*(global_maxnz+2)+1, "hypre_ComputeCommInfo: cinfo->incolind");
     cinfo->invalues =  hypre_fp_malloc(maxnrecv*(global_maxnz+2)+1, "hypre_ComputeCommInfo: cinfo->invalues");
     cinfo->maxnrecv = maxnrecv;
  }
  hypre_assert( cinfo->incolind != NULL );
  hypre_assert( cinfo->invalues != NULL );

  /* Zero our send buffer */
  for(i=0; i<npes; i++)
    pilu_send[i] = 0;

  /* tell the processors in nbrind what I'm going to send them. */
  for (i=0; i<rnnbr; i++)
    pilu_send[rnbrind[i]] = rnbrptr[i+1]-rnbrptr[i];    /* The # of rows I need */

  hypre_MPI_Alltoall( pilu_send, 1, HYPRE_MPI_INT,
        pilu_recv, 1, HYPRE_MPI_INT, pilut_comm );

  nsend = 0;
  snnbr = 0;
  snbrptr[0] = 0;
  for (penum=0; penum<npes; penum++) {
    if (pilu_recv[penum] > 0) {
      nsend += pilu_recv[penum];
      snbrind[snnbr] = penum;
      snbrptr[++snnbr] = nsend;
    }
  }
  cinfo->snnbr = snnbr;

  /* Allocate requests */
  index_requests = hypre_CTAlloc( hypre_MPI_Request,  snnbr , HYPRE_MEMORY_HOST);

  maxnsend = hypre_GlobalSEMax(nsend, pilut_comm);

  /* If memory requirements change, allocate new memory.
   * The first iteration this always occurs -- see hypre_ParINIT */
  if (cinfo->maxnsend < maxnsend) {
     hypre_TFree(cinfo->srowind, HYPRE_MEMORY_HOST);
     cinfo->srowind  = hypre_idx_malloc(maxnsend, "hypre_ComputeCommInfo: cinfo->srowind");
     cinfo->maxnsend = maxnsend;
  }
  hypre_assert( cinfo->srowind  != NULL );
  srowind = cinfo->srowind;

  /* issue asynchronous recieves */
  for (i=0; i<snnbr; i++) {
    hypre_MPI_Irecv( srowind+snbrptr[i], snbrptr[i+1]-snbrptr[i], HYPRE_MPI_INT,
          snbrind[i], TAG_Comm_rrowind, pilut_comm, &index_requests[i] ) ;
  }
  /* OK, now I go and send the rrowind to the processor */
  for (i=0; i<rnnbr; i++) {
    hypre_MPI_Send( rrowind+rnbrptr[i], rnbrptr[i+1]-rnbrptr[i], HYPRE_MPI_INT,
          rnbrind[i], TAG_Comm_rrowind, pilut_comm );
  }

  /* finalize  receives */
  for (i=0; i<snnbr; i++) {
    hypre_MPI_Wait( &index_requests[i], &Status ) ;
  }

#ifdef HYPRE_TIMING
  hypre_EndTiming( globals->CCI_timer  );
#endif
  /* clean up memory */
  hypre_TFree(index_requests, HYPRE_MEMORY_HOST);
}


/*************************************************************************
* This function returns what virtual PE the given row idx is located on.
**************************************************************************/
HYPRE_Int hypre_Idx2PE(HYPRE_Int idx,
             hypre_PilutSolverGlobals *globals)
{
  HYPRE_Int penum = 0;
  while (idx >= vrowdist[penum+1]) {  /* idx >= lastrow? */
    penum++;
    hypre_assert( penum < npes );
  }

  return penum;
}

/*************************************************************************
* This function computes a set that is independant between PEs but may
* contain dependencies within a PE. This variant simply gives rows to
* the lowest PE possible, which creates some load imbalancing between
* the highest and lowest PEs. It also forms the new permutation and
* marks the _local_ rows that are in the set (but not remote rows).
* For historical reasons the set is called a maximal indep. set (MIS).
**************************************************************************/
HYPRE_Int hypre_SelectSet(ReduceMatType *rmat, CommInfoType *cinfo,
              HYPRE_Int *perm,    HYPRE_Int *iperm,
              HYPRE_Int *newperm, HYPRE_Int *newiperm,
              hypre_PilutSolverGlobals *globals)
{
  HYPRE_UNUSED_VAR(iperm);

  HYPRE_Int ir, i, j, k, l, num;
  HYPRE_Int nnz, snnbr;
  HYPRE_Int *rcolind, *snbrind, *snbrptr, *srowind;

#ifdef HYPRE_DEBUG
  hypre_PrintLine("hypre_SelectSet", globals);
#endif
#ifdef HYPRE_TIMING
  hypre_BeginTiming( globals->SS_timer  );
#endif

  snnbr    = cinfo->snnbr;
  snbrind  = cinfo->snbrind;
  snbrptr  = cinfo->snbrptr;
  srowind  = cinfo->srowind;

  /* determine local rows that do not have non-zeros on higher numbered PEs. */
  num = 0;
  for (ir=0; ir<ntogo; ir++) {
    i = perm[ir+ndone]+firstrow;

    rcolind = rmat->rmat_rcolind[ir];
    nnz     = rmat->rmat_rnz[ir];

    for (j=1; j<nnz; j++) {
      if ((rcolind[j] < firstrow  ||  rcolind[j] >= lastrow)  &&
            mype > hypre_Idx2PE(rcolind[j], globals))
         break ;
    }
    if ( j == nnz ) {    /* passed test; put into set */
      jw[num++] = i;
      pilut_map[i]    = 1;     /* local doesn't need info in high bits */
    }
  }

  /* check for asymetries -- the triangular solves depend on the set being block diagonal */
  for (k=0; k<snnbr; k++)
    if (snbrind[k] < mype)
      for (i=snbrptr[k]; i<snbrptr[k+1]; i++)
         for (j=0; j<num; j++)
            if (srowind[i] == jw[j]) {
               hypre_CheckBounds(firstrow, jw[j], lastrow, globals);
               pilut_map[jw[j]] = 0;
               jw[j] = jw[--num];
            }

  /* Compute the new permutation with MIS at beginning */
  j = ndone;
  k = ndone+num;
  for (ir=ndone; ir<lnrows; ir++) {
    l = perm[ir];
    hypre_CheckBounds(0, l, lnrows, globals);
    if (pilut_map[l+firstrow] == 1) {  /* This is in MIS, put it into ldu */
      hypre_CheckBounds(ndone, j, ndone+num, globals);
      newperm[j]  = l;
      newiperm[l] = j++;
    }
    else {
      hypre_CheckBounds(ndone+num, k, lnrows, globals);
      newperm[k]  = l;
      newiperm[l] = k++;
    }
  }

#ifdef HYPRE_TIMING
  hypre_EndTiming( globals->SS_timer  );
#endif
#ifndef NDEBUG
  /* DEBUGGING: check map is zero outside of local rows */
  for (i=0; i<firstrow; i++)
    hypre_assert(pilut_map[i] == 0);
  for (i=lastrow; i<nrows; i++)
    hypre_assert(pilut_map[i] == 0);
#endif

  return num;
}

/*************************************************************************
* This function sends the factored rows to the appropriate processors. The
* rows are sent in the order of the _new_ MIS permutation. Each PE then
* uses the recieved information to mark _remote_ rows in the MIS. It takes
* as input the factored rows in LDU, the new permutation vectors, and the
* global map with local MIS rows already marked. This also updates the
* rnbrptr[i] to be the actual number of rows recieved from PE rnbrind[i].
* 3/20/98: Bug fix, lengths input to sgatherbuf increased by one to reflect
*   fact that diagonal element is also transmitted. -AJC
**************************************************************************/
void hypre_SendFactoredRows(FactorMatType *ldu, CommInfoType *cinfo,
                            HYPRE_Int *newperm, HYPRE_Int nmis, hypre_PilutSolverGlobals *globals)
{
  HYPRE_Int i, j, k, ku, kg, l, penum, snnbr, rnnbr, cnt, inCnt;
  HYPRE_Int *snbrind, *rnbrind, *rnbrptr, *sgatherbuf, *incolind;
  HYPRE_Int *usrowptr, *uerowptr, *ucolind;
  HYPRE_Real *dgatherbuf, *uvalues, *dvalues, *invalues;
  hypre_MPI_Status Status;
  hypre_MPI_Request *index_requests, *value_requests ;

#ifdef HYPRE_DEBUG
  hypre_PrintLine("hypre_SendFactoredRows", globals);
#endif
#ifdef HYPRE_TIMING
  hypre_BeginTiming( globals->SFR_timer  );
#endif

  snnbr   = cinfo->snnbr;
  snbrind = cinfo->snbrind;

  rnnbr   = cinfo->rnnbr;
  rnbrind = cinfo->rnbrind;
  rnbrptr = cinfo->rnbrptr;

  /* NOTE we cast a (HYPRE_Real*) to an (HYPRE_Int*) */
  sgatherbuf = (HYPRE_Int *)cinfo->gatherbuf;
  dgatherbuf = cinfo->gatherbuf;

  incolind = cinfo->incolind;
  invalues = cinfo->invalues;

  usrowptr = ldu->usrowptr;
  uerowptr = ldu->uerowptr;
  ucolind  = ldu->ucolind;
  uvalues  = ldu->uvalues;
  dvalues  = ldu->dvalues;

  /* Allocate requests */
  index_requests = hypre_CTAlloc( hypre_MPI_Request,  rnnbr , HYPRE_MEMORY_HOST);
  value_requests = hypre_CTAlloc( hypre_MPI_Request,  rnnbr , HYPRE_MEMORY_HOST);

  /* Issue asynchronous receives for rows from other processors.
     Asynchronous receives needed to avoid overflowing comm buffers. */
  j = 0;
  cnt = (cinfo->maxntogo)*(global_maxnz+2) ;
  for (i=0; i<rnnbr; i++) {
    penum = rnbrind[i];

    hypre_MPI_Irecv( incolind+j, cnt, HYPRE_MPI_INT,
          penum, TAG_Send_colind, pilut_comm, &index_requests[i] );

    hypre_MPI_Irecv( invalues+j, cnt, hypre_MPI_REAL,
          penum, TAG_Send_values, pilut_comm, &value_requests[i] );

    j += cnt;
  }

  /* pack the colind for sending*/
  l = 0;
  for (j=ndone; j<ndone+nmis; j++) {
    k = newperm[j];
    hypre_CheckBounds(firstrow, k+firstrow, lastrow, globals);
    hypre_assert(IsInMIS(pilut_map[k+firstrow]));
    hypre_CheckBounds(0, uerowptr[k]-usrowptr[k], global_maxnz+1, globals);

    /* sgatherbuf[l++] = uerowptr[k]-usrowptr[k]; */  /* store length */
    /* Bug fix, 3/20/98 */
    sgatherbuf[l++] = uerowptr[k]-usrowptr[k]+1;  /* store length */
    sgatherbuf[l++] = k+firstrow;               /* store row #  */

    for (ku=usrowptr[k], kg=l;   ku<uerowptr[k];   ku++, kg++)
      sgatherbuf[kg] = ucolind[ku];
    l += global_maxnz;
  }

  /* send colind to each neighbor */
  for (i=0; i<snnbr; i++) {
    hypre_MPI_Send( sgatherbuf, l, HYPRE_MPI_INT,
          snbrind[i], TAG_Send_colind, pilut_comm );
  }

  /* pack the values */
  l = 0;
  for (j=ndone; j<ndone+nmis; j++) {
    k = newperm[j];
    hypre_CheckBounds(firstrow, k+firstrow, lastrow, globals);
    hypre_assert(IsInMIS(pilut_map[k+firstrow]));

    l++;                          /* first element undefined */
    dgatherbuf[l++] = dvalues[k]; /* store diagonal */

    for (ku=usrowptr[k], kg=l;   ku<uerowptr[k];   ku++, kg++)
      dgatherbuf[kg] = uvalues[ku];
    l += global_maxnz;
  }

  /* send values to each neighbor */
  for (i=0; i<snnbr; i++) {
    hypre_MPI_Send( dgatherbuf, l, hypre_MPI_REAL,
          snbrind[i], TAG_Send_values, pilut_comm );
  }

  /* Finish receiving rows */
  j = 0;
  cnt = (cinfo->maxntogo)*(global_maxnz+2) ;
  for (i=0; i<rnnbr; i++) {
    penum = rnbrind[i];

    hypre_MPI_Wait( &index_requests[i], &Status);

    /* save where each row is received into the map */
    hypre_MPI_Get_count( &Status, HYPRE_MPI_INT, &inCnt );
    rnbrptr[i] = inCnt;
    for (k=0; k<inCnt; k += global_maxnz+2)
      pilut_map[incolind[j+k+1]] = ((j+k)<<1) + 1; /* pack MIS flag in LSB */

    hypre_MPI_Wait( &value_requests[i], &Status);

    j += cnt;
    hypre_CheckBounds(0, j, (cinfo->maxnrecv)*(global_maxnz+2)+2, globals);
  }
#ifdef HYPRE_TIMING
  hypre_EndTiming( globals->SFR_timer  );
#endif

  /* clean up memory */
  hypre_TFree(index_requests, HYPRE_MEMORY_HOST);
  hypre_TFree(value_requests, HYPRE_MEMORY_HOST);
}


/*************************************************************************
* This function creates the new reduce matrix. It takes as input the
* current reduced matrix and the outside nodes sent from other PEs.
* Also both the old permutation (which applies to this rmat) and the new
* permutation (which applies to the new rmat) are taken as input. After
* each row is computed, the number of non-zeros is kept the same.
*
* Note that all fill elements into the L portion mus fill unto the same
* processor as the row being subtracted is, since it is block diagonal.
**************************************************************************/
void hypre_ComputeRmat(FactorMatType *ldu, ReduceMatType *rmat,
                 ReduceMatType *nrmat, CommInfoType *cinfo,
                 HYPRE_Int *perm,    HYPRE_Int *iperm,
                 HYPRE_Int *newperm, HYPRE_Int *newiperm, HYPRE_Int nmis, HYPRE_Real tol,
                 hypre_PilutSolverGlobals *globals)
{
  HYPRE_UNUSED_VAR(perm);

  HYPRE_Int i, ir, inr, start, k, kk, l, m, end, nnz;
  HYPRE_Int *usrowptr, *uerowptr, *ucolind, *incolind, *rcolind, rrowlen;
  HYPRE_Real *uvalues, *nrm2s, *invalues, *rvalues, *dvalues;
  HYPRE_Real mult, rtol;

#ifdef HYPRE_DEBUG
  hypre_PrintLine("hypre_ComputeRmat", globals);
#endif
#ifdef HYPRE_TIMING
  hypre_BeginTiming( globals->CR_timer  );
#endif

  usrowptr = ldu->usrowptr;
  uerowptr = ldu->uerowptr;
  ucolind  = ldu->ucolind;
  uvalues  = ldu->uvalues;
  dvalues  = ldu->dvalues;
  nrm2s    = ldu->nrm2s;

  incolind = cinfo->incolind;
  invalues = cinfo->invalues;

  /* OK, now reduce the remaining rows.
   * inr counts the rows actually factored as an index for the nrmat */
  inr = 0;
  for (ir=ndone+nmis; ir<lnrows; ir++) {
    i = newperm[ir];
    hypre_CheckBounds(0, i, lnrows, globals);
    hypre_assert(!IsInMIS(pilut_map[i+firstrow]));

    rtol = nrm2s[i]*tol;

    /* get the row according to the _previous_ permutation */
    k = iperm[i]-ndone;
    hypre_CheckBounds(0, k, ntogo, globals);
    nnz     = rmat->rmat_rnz[k];
              rmat->rmat_rnz[k] = 0;
    rcolind = rmat->rmat_rcolind[k];
              rmat->rmat_rcolind[k] = NULL;
    rvalues = rmat->rmat_rvalues[k];
              rmat->rmat_rvalues[k] = NULL;
    rrowlen = rmat->rmat_rrowlen[k];
              rmat->rmat_rrowlen[k] = 0;

    /* Initialize workspace and determine the L indices (ie., MIS).
     * The L indices are stored as either the row's new local permutation
     * or the permuted order we recieved the row. The LSB is a flag
     * for being local (==0) or remote (==1). */
    jr[rcolind[0]] = 0;  /* store diagonal first */
    jw[0] = rcolind[0];
     w[0] = rvalues[0];

    lastlr = 0;
    for (lastjr=1; lastjr<nnz; lastjr++) {
      hypre_CheckBounds(0, rcolind[lastjr], nrows, globals);

      /* record L elements */
      if (IsInMIS(pilut_map[rcolind[lastjr]])) {
         if (rcolind[lastjr] >= firstrow  &&  rcolind[lastjr] < lastrow)
            hypre_lr[lastlr] = (newiperm[rcolind[lastjr]-firstrow] << 1);
         else {
            hypre_lr[lastlr] = pilut_map[rcolind[lastjr]];  /* map[] == (l<<1) | 1 */
            hypre_assert(incolind[StripMIS(pilut_map[rcolind[lastjr]])+1] ==
                 rcolind[lastjr]);
         }
        lastlr++;
      }

      jr[rcolind[lastjr]] = lastjr;
      jw[lastjr] = rcolind[lastjr];
       w[lastjr] = rvalues[lastjr];
    }
    hypre_assert(lastjr == nnz);
    hypre_assert(lastjr > 0);

    /* Go through the L nonzeros and pull in the contributions */
    while( lastlr != 0 ) {
      k = hypre_ExtractMinLR( globals );

      if ( IsLocal(k) ) {  /* Local node -- row is in DU */
         hypre_CheckBounds(0, StripLocal(k), lnrows, globals);
         kk = newperm[ StripLocal(k) ];  /* remove the local bit (LSB) */
         k  = kk+firstrow;

         hypre_CheckBounds(0, kk, lnrows, globals);
         hypre_CheckBounds(0, jr[k], lastjr, globals);
         hypre_assert(jw[jr[k]] == k);

        mult = w[jr[k]]*dvalues[kk];
        w[jr[k]] = mult;

        if (hypre_abs(mult) < rtol)
           continue; /* First drop test */

        for (l=usrowptr[kk]; l<uerowptr[kk]; l++) {
          hypre_CheckBounds(0, ucolind[l], nrows, globals);
          m = jr[ucolind[l]];
          if (m == -1) {
            if (hypre_abs(mult*uvalues[l]) < rtol)
              continue;  /* Don't worry. The fill has too small of a value */

            /* record L elements -- these must be local */
            if (IsInMIS(pilut_map[ucolind[l]])) {
               hypre_assert(ucolind[l] >= firstrow  &&  ucolind[l] < lastrow);
               hypre_lr[lastlr] = (newiperm[ucolind[l]-firstrow] << 1);
               lastlr++;
            }

            /* Create fill */
            jr[ucolind[l]] = lastjr;
            jw[lastjr] = ucolind[l];
             w[lastjr] = -mult*uvalues[l];
             lastjr++;
          }
          else
            w[m] -= mult*uvalues[l];
        }
      }
      else { /* Outside node -- row is in incolind/invalues */
        start = StripLocal(k);             /* Remove the local bit (LSB) */
        end   = start + incolind[start];   /* get length */
        start++;
        k     = incolind[start];           /* get diagonal colind == row index */

        hypre_CheckBounds(0, k, nrows, globals);
        hypre_CheckBounds(0, jr[k], lastjr, globals);
        hypre_assert(jw[jr[k]] == k);

        mult = w[jr[k]]*invalues[start];
        w[jr[k]] = mult;

        if (hypre_abs(mult) < rtol)
           continue; /* First drop test */

        for (l=++start; l<=end; l++) {
          hypre_CheckBounds(0, incolind[l], nrows, globals);
          m = jr[incolind[l]];
          if (m == -1) {
            if (hypre_abs(mult*invalues[l]) < rtol)
              continue;  /* Don't worry. The fill has too small of a value */

            /* record L elements -- these must be remote */
            if (IsInMIS(pilut_map[incolind[l]])) {
               hypre_assert(incolind[l] < firstrow  ||  incolind[l] >= lastrow);
               hypre_lr[lastlr] = pilut_map[incolind[l]];  /* map[] == (l<<1) | 1 */
               lastlr++;
            }

            /* Create fill */
            jr[incolind[l]] = lastjr;
            jw[lastjr] = incolind[l];
             w[lastjr] = -mult*invalues[l];
             lastjr++;
          }
          else
            w[m] -= mult*invalues[l];
        }
      }
    } /* L non-zeros */

    /* perform SecondDrops and store in appropriate places */
    hypre_SecondDropSmall( rtol, globals );
    m = hypre_SeperateLU_byMIS( globals);
    hypre_UpdateL( i, m, ldu, globals );
    hypre_FormNRmat( inr++, m, nrmat, global_maxnz, rrowlen, rcolind, rvalues, globals );
    /* hypre_FormNRmat( inr++, m, nrmat, 3*global_maxnz, rcolind, rvalues, globals ); */
  }
#ifdef HYPRE_TIMING
  hypre_EndTiming( globals->CR_timer  );
#endif

}


/*************************************************************************
* This function performs a serial hypre_ILUT on the local MIS rows, then calls
* hypre_SecondDrop to drop some elements and create LDU. If the set is truly
* independant, then this just puts the row into DU. If there are
* dependencies within a PE this factors those, adding to L, and forms DU.
**************************************************************************/
void hypre_FactorLocal(FactorMatType *ldu, ReduceMatType *rmat,
                 ReduceMatType *nrmat, CommInfoType *cinfo,
                 HYPRE_Int *perm,    HYPRE_Int *iperm,
                 HYPRE_Int *newperm, HYPRE_Int *newiperm, HYPRE_Int nmis, HYPRE_Real tol,
                 hypre_PilutSolverGlobals *globals)
{
  HYPRE_UNUSED_VAR(cinfo);

  HYPRE_Int i, ir, k, kk, l, m, nnz, diag;
  HYPRE_Int *usrowptr, *uerowptr, *ucolind, *rcolind;
  HYPRE_Real *uvalues, *nrm2s, *rvalues, *dvalues;
  HYPRE_Real mult, rtol;

#ifdef HYPRE_DEBUG
  hypre_PrintLine("hypre_FactorLocal", globals);
#endif
#ifdef HYPRE_TIMING
  hypre_BeginTiming( globals->FL_timer  );
#endif


  hypre_assert( rmat  != nrmat    );
  hypre_assert( perm  != newperm  );
  hypre_assert( iperm != newiperm );

  usrowptr = ldu->usrowptr;
  uerowptr = ldu->uerowptr;
  ucolind  = ldu->ucolind;
  uvalues  = ldu->uvalues;
  dvalues  = ldu->dvalues;
  nrm2s    = ldu->nrm2s;

  /* OK, now factor the nmis rows */
  for (ir=ndone; ir<ndone+nmis; ir++) {
    i = newperm[ir];
    hypre_CheckBounds(0, i, lnrows, globals);
    hypre_assert(IsInMIS(pilut_map[i+firstrow]));

    rtol = nrm2s[i]*tol;  /* Compute relative tolerance */
    diag = newiperm[i];

    /* get the row according to the _previous_ permutation */
    k = iperm[i]-ndone;
    hypre_CheckBounds(0, k, ntogo, globals);
    nnz     = rmat->rmat_rnz[k];
    rcolind = rmat->rmat_rcolind[k];
    rvalues = rmat->rmat_rvalues[k];

    /* Initialize workspace and determines the L indices.
     * Since there are only local nodes, we just store the
     * row's new permutation into lr, without any flags. */
    jr[rcolind[0]] = 0;  /* store diagonal first */
    jw[0] = rcolind[0];
     w[0] = rvalues[0];
    hypre_assert(jw[0] == i+firstrow);

    lastlr = 0;
    for (lastjr=1; lastjr<nnz; lastjr++) {
      hypre_CheckBounds(0, rcolind[lastjr], nrows, globals);

      /* record L elements */
      if (rcolind[lastjr] >= firstrow  &&
            rcolind[lastjr] <  lastrow   &&
            newiperm[rcolind[lastjr]-firstrow] < diag) {
         hypre_lr[lastlr] = newiperm[rcolind[lastjr]-firstrow];
        lastlr++;
      }

      jr[rcolind[lastjr]] = lastjr;
      jw[lastjr] = rcolind[lastjr];
       w[lastjr] = rvalues[lastjr];
    }

    /* Go through the L nonzeros and pull in the contributions */
    while( lastlr != 0 ) {
      k = hypre_ExtractMinLR(globals);

      hypre_CheckBounds(0, k, lnrows, globals);
      kk = newperm[ k ];
      k  = kk+firstrow;

      hypre_CheckBounds(0, kk, lnrows, globals);
      hypre_CheckBounds(0, jr[k], lastjr, globals);
      hypre_assert(jw[jr[k]] == k);

      mult = w[jr[k]]*dvalues[kk];
      w[jr[k]] = mult;

      if (hypre_abs(mult) < rtol)
         continue; /* First drop test */

      for (l=usrowptr[kk]; l<uerowptr[kk]; l++) {
         hypre_CheckBounds(0, ucolind[l], nrows, globals);
         m = jr[ucolind[l]];
         if (m == -1) {
            if (hypre_abs(mult*uvalues[l]) < rtol)
             continue;  /* Don't worry. The fill has too small of a value */

            /* record L elements */
            if (ucolind[l] >= firstrow  &&
                  ucolind[l] <  lastrow   &&
                  newiperm[ucolind[l]-firstrow] < diag) {
               hypre_assert(IsInMIS(pilut_map[ucolind[l]]));
               hypre_lr[lastlr] = newiperm[ucolind[l]-firstrow];
               lastlr++;
            }

            /* Create fill */
            jr[ucolind[l]]  = lastjr;
            jw[lastjr] = ucolind[l];
            w[lastjr] = -mult*uvalues[l];
            lastjr++;
         }
         else
            w[m] -= mult*uvalues[l];
      }
    } /* L non-zeros */

    /* perform SecondDrops and store in appropriate places */
    hypre_SecondDropSmall( rtol, globals );
    m = hypre_SeperateLU_byDIAG( diag, newiperm, globals );
    hypre_UpdateL( i, m, ldu, globals );
    hypre_FormDU( i, m, ldu, rcolind, rvalues, tol, globals );
  }
#ifdef HYPRE_TIMING
  hypre_EndTiming( globals->FL_timer  );
#endif
}


/*************************************************************************
* This function drops small values from the workspace, and also resets
* the jr[] array to all -1's.
**************************************************************************/
void hypre_SecondDropSmall( HYPRE_Real rtol,
             hypre_PilutSolverGlobals *globals )
{
  HYPRE_Int i;

  /* Reset the jr array. */
  for (i=0; i<lastjr; i++) {
    hypre_CheckBounds(0, jw[i], nrows, globals);
    jr[jw[i]] = -1;
  }

  /* Remove any (off-diagonal) elements of the row below the tolerance */
  for (i=1; i<lastjr;) {
    if (hypre_abs(w[i]) < rtol) {
      jw[i] = jw[--lastjr];
       w[i] =  w[lastjr];
    }
    else
      i++;
  }
}



/*****************************************************************
* This function seperates the L and U portions of the workspace
* and returns the point at which they seperate, so
*  L entries are between [1     .. point)
*  U or rmat entries are [point .. lastjr)
* We assume the diagonal D is index [0].
*
* This version compares the (new) permuted order of entries to the
* given permuted order of the row (diag) to determine entries in L.
* This is suitable for local factorizations.
******************************************************************/
HYPRE_Int hypre_SeperateLU_byDIAG( HYPRE_Int diag, HYPRE_Int *newiperm,
             hypre_PilutSolverGlobals *globals )
{
  HYPRE_Int first, last, itmp;
  HYPRE_Real dtmp;

#ifdef HYPRE_TIMING
  hypre_BeginTiming( globals->SLUD_timer  );
#endif

  /* Perform a Qsort type pass to seperate L and U (rmat) entries. */
  if (lastjr == 1)
    last = first = 1;
  else {
    last  = 1;
    first = lastjr-1;
    while (true) {
      while (last < first  &&  /* while (last < first  AND  [last] is in L) */
            (jw[last] >= firstrow &&
             jw[last] <  lastrow  &&
             newiperm[jw[last]-firstrow] < diag))
        last++;
      while (last < first  &&  /* while (last < first  AND  [first] is not in L) */
            ! (jw[first] >= firstrow &&
               jw[first] <  lastrow  &&
               newiperm[jw[first]-firstrow] < diag))
        first--;

      if (last < first) {
        SWAP(jw[first], jw[last], itmp);
        SWAP( w[first],  w[last], dtmp);
        last++; first--;
      }

      if (last == first) {
        if ((jw[last] >= firstrow &&  /* if [last] is in L */
                 jw[last] <  lastrow  &&
                 newiperm[jw[last]-firstrow] < diag)) {
          first++;
          last++;
        }
        break;
      }
      else if (last > first) {
        first++;
        break;
      }
    }
  }

#ifndef NDEBUG
  /* DEBUGGING: verify sorting to some extent */
  for (itmp=1; itmp<last; itmp++) {
    hypre_assert((jw[itmp] >= firstrow &&   /* [itmp] is in L -- must be MIS */
             jw[itmp] <  lastrow  &&
             newiperm[jw[itmp]-firstrow] < diag));
    hypre_assert(IsInMIS(pilut_map[jw[itmp]]));
  }
  for (itmp=first; itmp<lastjr; itmp++) {
    hypre_assert(!(jw[itmp] >= firstrow &&  /* [itmp] is not in L -- may be MIS still */
             jw[itmp] <  lastrow  &&
             newiperm[jw[itmp]-firstrow] < diag));
  }
  hypre_assert(last == first);
#endif
#ifdef HYPRE_TIMING
  hypre_EndTiming( globals->SLUD_timer  );
#endif


  return first;
}


/*****************************************************************
* This function seperates the L and U portions of the workspace
* and returns the point at which they seperate, so
*  L entries are between [1     .. point)
*  U or rmat entries are [point .. lastjr)
* We assume the diagonal D is index [0].
*
* This version simply uses the MIS to determine entries in L.
* This is suitable for reductions involving rows on other PEs,
* where -every- row in the MIS will be part of L.
******************************************************************/
HYPRE_Int hypre_SeperateLU_byMIS( hypre_PilutSolverGlobals *globals )
{
  HYPRE_Int first, last, itmp;
  HYPRE_Real dtmp;

#ifdef HYPRE_TIMING
  hypre_BeginTiming( globals->SLUM_timer  );
#endif

  /* Perform a Qsort type pass to seperate L and U (rmat) entries. */
  if (lastjr == 1)
    last = first = 1;
  else {
    last  = 1;
    first = lastjr-1;
    while (true) {
      while (last < first  &&    IsInMIS(pilut_map[jw[last ]]))  /* and [last] is in L */
        last++;
      while (last < first  &&  ! IsInMIS(pilut_map[jw[first]]))  /* and [first] is not in L */
        first--;

      if (last < first) {
        SWAP(jw[first], jw[last], itmp);
        SWAP( w[first],  w[last], dtmp);
        last++; first--;
      }

      if (last == first) {
        if (IsInMIS(pilut_map[jw[last]])) {
          first++;
          last++;
        }
        break;
      }
      else if (last > first) {
        first++;
        break;
      }
    }
  }

#ifndef NDEBUG
  /* DEBUGGING: verify sorting to some extent */
  for (itmp=1; itmp<last; itmp++)
    hypre_assert(IsInMIS(pilut_map[jw[itmp]]));
  for (itmp=first; itmp<lastjr; itmp++)
    hypre_assert(!IsInMIS(pilut_map[jw[itmp]]));
  hypre_assert(last == first);
#endif

#ifdef HYPRE_TIMING
  hypre_EndTiming( globals->SLUM_timer  );
#endif


  return first;
}


/*************************************************************************
* This function updates the L part of the given row, assuming that the
* workspace has already been split into L and U entries. L may already
* be partially or completely full--this fills it and then starts to
* replace the min value.
**************************************************************************/
void hypre_UpdateL(HYPRE_Int lrow, HYPRE_Int last, FactorMatType *ldu,
             hypre_PilutSolverGlobals *globals)
{
  HYPRE_Int i, j, min, start, end;
  HYPRE_Int *lcolind;
  HYPRE_Real *lvalues;

#ifdef HYPRE_TIMING
  hypre_BeginTiming( globals->UL_timer  );
#endif

  lcolind = ldu->lcolind;
  lvalues = ldu->lvalues;

  start = ldu->lsrowptr[lrow];
  end   = ldu->lerowptr[lrow];

  /* The entries between [1, last) are part of L */
  for (i=1; i<last; i++) {
    if (end-start < global_maxnz) {  /* In case we did not have maxnz in L */
      lcolind[end] = jw[i];
      lvalues[end] =  w[i];
      end++;
    }
    else {
      min = start;  /* find min and replace if i is larger */
      for (j=start+1; j<end; j++) {
         if (hypre_abs(lvalues[j]) < hypre_abs(lvalues[min]))
            min = j;
      }

      if (hypre_abs(lvalues[min]) < hypre_abs(w[i])) {
         lcolind[min] = jw[i];
         lvalues[min] =  w[i];
      }
    }
  }
  ldu->lerowptr[lrow] = end;
  hypre_CheckBounds(0, end-start, global_maxnz+1, globals);
#ifdef HYPRE_TIMING
  hypre_EndTiming( globals->UL_timer  );
#endif

}


/*************************************************************************
* This function forms the new reduced row corresponding to
* the given row, assuming that the
* workspace has already been split into L and U (rmat) entries. It reuses
* the memory for the row in the reduced matrix, storing the new row into
* nrmat->*[rrow].
* New version allows new row to be larger than original row, so it does not
* necessarily reuse the same memory. AC 3-18
**************************************************************************/
void hypre_FormNRmat(HYPRE_Int rrow, HYPRE_Int first, ReduceMatType *nrmat,
               HYPRE_Int max_rowlen,
               HYPRE_Int in_rowlen, HYPRE_Int *in_colind, HYPRE_Real *in_values,
               hypre_PilutSolverGlobals *globals )
{
  HYPRE_Int nz, max, j, out_rowlen, *rcolind;
  HYPRE_Real *rvalues;

#ifdef HYPRE_TIMING
  hypre_BeginTiming( globals->FNR_timer  );
#endif

  hypre_assert(in_colind[0] == jw[0]);  /* diagonal at the beginning */

  /* check to see if we need to reallocate space */
  out_rowlen = hypre_min( max_rowlen, lastjr-first+1 );
  if( out_rowlen > in_rowlen )
  {
    /*hypre_free_multi( in_colind, in_values, -1 );*/
    hypre_TFree(in_colind, HYPRE_MEMORY_HOST);
    hypre_TFree(in_values, HYPRE_MEMORY_HOST);
    in_colind = NULL; in_values = NULL;
    rcolind = hypre_idx_malloc( out_rowlen, "FornNRmat: rcolind");
    rvalues = hypre_fp_malloc( out_rowlen, "FornNRmat: rvalues");
  }else
  {
    rcolind = in_colind;
    rvalues = in_values;
  }

  rcolind[0] = jw[0];
  rvalues[0] = w[0];

  /* The entries [first, lastjr) are part of U (rmat) */
  if (lastjr-first+1 <= max_rowlen) { /* Simple copy */
    for (nz=1, j=first;   j<lastjr;   nz++, j++) {
      rcolind[nz] = jw[j];
      rvalues[nz] =  w[j];
    }
    hypre_assert(nz == lastjr-first+1);
  }
  else { /* Keep largest out_rowlen elements in the reduced row */
    for (nz=1; nz<out_rowlen; nz++) {
      max = first;
      for (j=first+1; j<lastjr; j++) {
         if (hypre_abs(w[j]) > hypre_abs(w[max]))
            max = j;
      }

      rcolind[nz] = jw[max];   /* store max */
      rvalues[nz] =  w[max];

      jw[max] = jw[--lastjr];  /* swap max out */
       w[max] =  w[  lastjr];
    }
    hypre_assert(nz == out_rowlen);
  }
  hypre_assert(nz <= max_rowlen);

  /* link the reused storage to the new reduced system */
  nrmat->rmat_rnz[rrow]     = nz;
  nrmat->rmat_rrowlen[rrow] = out_rowlen;
  nrmat->rmat_rcolind[rrow] = rcolind;
  nrmat->rmat_rvalues[rrow] = rvalues;

#ifdef HYPRE_TIMING
  hypre_EndTiming( globals->FNR_timer  );
#endif

}



/*************************************************************************
* This function forms the DU part of the given row, assuming that the
* workspace has already been split into L and U entries. It disposes of
* the memory used by the row in the reduced matrix.
**************************************************************************/
void hypre_FormDU(HYPRE_Int lrow, HYPRE_Int first, FactorMatType *ldu,
      HYPRE_Int *rcolind, HYPRE_Real *rvalues, HYPRE_Real tol,
             hypre_PilutSolverGlobals *globals )
{
  HYPRE_Int nz, max, j, end;
  HYPRE_Int *ucolind, *uerowptr;
  HYPRE_Real *uvalues;

  ucolind  = ldu->ucolind;
  uerowptr = ldu->uerowptr;
  uvalues  = ldu->uvalues;

  /*
   * Take care of the diagonal
   */
  if (w[0] == 0.0) {
    hypre_printf("Zero pivot in row %d, adding e to proceed!\n", lrow);
    ldu->dvalues[lrow] = 1.0/tol;
  }
  else
    ldu->dvalues[lrow] = 1.0/w[0];

  /*
   * Take care of the elements of U
   * Note U is completely empty beforehand.
   */
  end = ldu->uerowptr[lrow];

  hypre_assert(ldu->usrowptr[lrow] == ldu->uerowptr[lrow]);
  for (nz=0; nz<global_maxnz && lastjr>first; nz++) {
    /* The entries [first, lastjr) are part of U */
    max = first;
    for (j=first+1; j<lastjr; j++) {
      if (hypre_abs(w[j]) > hypre_abs(w[max]))
         max = j;
    }

    ucolind[end] = jw[max];  /* store max */
    uvalues[end] =  w[max];
    end++;

    jw[max] = jw[--lastjr];  /* swap max out */
     w[max] =  w[  lastjr];
  }
  uerowptr[lrow] = end;

  /* free the row storage */
  hypre_TFree( rcolind ,HYPRE_MEMORY_HOST);
  hypre_TFree( rvalues ,HYPRE_MEMORY_HOST);
}


/*************************************************************************
* This function zeros the map for all local rows and rows we recieved.
* During debugging it checks the entire map to ensure other entries remain
* zero as expected. cinfo->rnbrptr[i] has the _actual_ number of rows
* recieved from PE rnbrind[i], which is set in hypre_SendFactoredRows.
**************************************************************************/
void hypre_EraseMap(CommInfoType *cinfo, HYPRE_Int *newperm, HYPRE_Int nmis,
             hypre_PilutSolverGlobals *globals)
{
  HYPRE_Int i, j, k, cnt, rnnbr;
  HYPRE_Int *rnbrptr, *incolind;

  rnnbr    = cinfo->rnnbr;
  rnbrptr  = cinfo->rnbrptr;
  incolind = cinfo->incolind;

#ifdef HYPRE_DEBUG
  hypre_PrintLine("hypre_EraseMap", globals);
#endif

  /* clear map of all MIS rows */
  for (i=ndone; i<ndone+nmis; i++)
    pilut_map[newperm[i]+firstrow] = 0;

  /* clear map of all received rows. see hypre_SendFactoredRows code */
  j = 1;  /* row index in [1] */
  cnt = (cinfo->maxntogo)*(global_maxnz+2) ;
  for (i=0; i<rnnbr; i++) {
    for (k=0; k<rnbrptr[i]; k += global_maxnz+2)
      pilut_map[incolind[j+k]] = 0;
    j += cnt;
  }

#ifndef NDEBUG
  /* DEBUGGING: check entire map */
  for (i=0; i<nrows; i++)
    if ( pilut_map[i] != 0 ) {
      hypre_printf("PE %d BAD ERASE %d [%d %d]\n", mype, i, firstrow, lastrow);
      pilut_map[i] = 0;
    }
#endif
}


/*************************************************************************
* This function allocates datastructures for the new reduced matrix (nrmat),
* the global workspace, and the communication info. Some parts of the
* comm info are allocated dynamically so we just initialize their size to
* zero here, forcing an allocation the first time hypre_ComputeCommInfo is called.
* Comments indicate where in George's code these originally existed.
**************************************************************************/
void hypre_ParINIT( ReduceMatType *nrmat, CommInfoType *cinfo, HYPRE_Int *rowdist,
              hypre_PilutSolverGlobals *globals )
{
  HYPRE_Int i;

#ifdef HYPRE_DEBUG
  hypre_PrintLine("hypre_ParINIT", globals);
#endif

  /* save a global copy of the row distribution */
  vrowdist = hypre_idx_malloc(npes+1, "hypre_ParINIT: vrowdist");
  hypre_memcpy_idx(vrowdist, rowdist, npes+1);

  /* ---- hypre_ParILUT ---- */
  /* Allocate the new rmat */
  nrmat->rmat_rnz     = hypre_idx_malloc(ntogo, "hypre_ParILUT: nrmat->rmat_rnz"    );
  nrmat->rmat_rrowlen = hypre_idx_malloc(ntogo, "hypre_ParILUT: nrmat->rmat_rrowlen");
  nrmat->rmat_rcolind = (HYPRE_Int **) hypre_mymalloc( sizeof(HYPRE_Int*)*ntogo, "hypre_ParILUT: nrmat->rmat_rcolind");
  nrmat->rmat_rvalues = (HYPRE_Real **)  hypre_mymalloc( sizeof(HYPRE_Real*) *ntogo, "hypre_ParILUT: nrmat->rmat_rvalues");
  for ( i=0; i < ntogo; i++ )
  {
     nrmat->rmat_rcolind[ i ] = NULL;
     nrmat->rmat_rvalues[ i ] = NULL;
  }

  /* Allocate work space */
  hypre_TFree(jr, HYPRE_MEMORY_HOST);
  jr = hypre_idx_malloc_init(nrows, -1, "hypre_ParILUT: jr");
  hypre_TFree(hypre_lr, HYPRE_MEMORY_HOST);
  hypre_lr = hypre_idx_malloc_init(nleft, -1, "hypre_ParILUT: lr");
  hypre_TFree(jw, HYPRE_MEMORY_HOST);
  jw = hypre_idx_malloc(nleft, "hypre_ParILUT: jw");
  hypre_TFree(w, HYPRE_MEMORY_HOST);
  w  =  hypre_fp_malloc(nleft, "hypre_ParILUT: w");

  /* ---- hypre_ComputeCommInfo ---- */
  /* Allocate global map */
  pilut_map = hypre_idx_malloc_init(nrows, 0, "hypre_ComputeCommInfo: map");

  /* Allocate cinfo */
  cinfo->rnbrind  = hypre_idx_malloc(npes,   "hypre_ComputeCommInfo: cinfo->rnbrind");
  cinfo->rrowind  = hypre_idx_malloc(nleft,  "hypre_ComputeCommInfo: cinfo->rrowind");
  cinfo->rnbrptr  = hypre_idx_malloc(npes+1, "hypre_ComputeCommInfo: cinfo->rnbrptr");

  cinfo->snbrind  = hypre_idx_malloc(npes,   "hypre_ComputeCommInfo: cinfo->snbrind");
  cinfo->snbrptr  = hypre_idx_malloc(npes+1, "hypre_ComputeCommInfo: cinfo->snbrptr");

  /* force allocates within hypre_ComputeCommInfo */
  cinfo->incolind = NULL;
  cinfo->invalues = NULL;
  cinfo->srowind  = NULL;
  cinfo->maxnrecv = 0;
  cinfo->maxnsend = 0;

  /* ---- ComputeMIS ---- */
  /*cinfo->gatherbuf = hypre_fp_malloc(ntogo*(global_maxnz+2), "ComputeMIS: gatherbuf");*/
  /* RDF: There is a purify UMR problem that a calloc gets rid of.
   * Don't know if this is actually an indication of a bug */
  cinfo->gatherbuf = hypre_CTAlloc(HYPRE_Real,  ntogo*(global_maxnz+2), HYPRE_MEMORY_HOST);

}
