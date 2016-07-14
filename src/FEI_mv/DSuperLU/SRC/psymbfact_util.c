
/*
 * -- Distributed symbolic factorization auxialiary routine  (version 1.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley - July 2003
 * INRIA France - January 2004
 * Laura Grigori
 *
 * November 1, 2007
 */

#include "superlu_ddefs.h"
#include "psymbfact.h"

static void
copy_mem_int(int_t howmany, int_t* old, int_t* new)
{
  register int_t i;
  for (i = 0; i < howmany; i++) new[i] = old[i];
}


/*
 * Expand the existing storage to accommodate more fill-ins.
 */
/************************************************************************/
static int_t *expand
/************************************************************************/
(
 int_t prev_len,    /* length used from previous call */
 int_t min_new_len, /* minimum new length to allocate */
 int_t *prev_mem,    /* pointer to the previous memory */
 int_t *p_new_len,      /* length of the new memory allocated */
 int_t len_tcopy_fbeg,  /* size of the memory to be copied to new store 
			     starting from the beginning of the memory */
 int_t len_tcopy_fend,  /* size of the memory to be copied to new store,
			    starting from the end of the memory */
 psymbfact_stat_t *PS
 )
{
  float exp = 2.0;
  float alpha;
  int_t *new_mem;
  int_t new_len, tries, lword, extra, bytes_to_copy;
  
  alpha = exp;
  lword = sizeof(int_t);
  
  new_len = alpha * prev_len;
  if (min_new_len > 0 && new_len < min_new_len)
    new_len = min_new_len;
  
  new_mem = (void *) SUPERLU_MALLOC(new_len * lword);
  PS->allocMem += new_len * lword;
  
  if (new_mem) {
    if (len_tcopy_fbeg != 0)
      copy_mem_int(len_tcopy_fbeg, prev_mem, new_mem);
    if (len_tcopy_fend != 0)  
      copy_mem_int(len_tcopy_fend, &(prev_mem[prev_len-len_tcopy_fend]), 
		   &(new_mem[new_len-len_tcopy_fend]));
  }
  *p_new_len = new_len;
  return new_mem;
  
} /* EXPAND */


/*
 * Expand the data structures for L and U during the factorization.
 * Return value:   0 - successful return
 *               > 0 - number of bytes allocated when run out of space
 */
/************************************************************************/
int_t psymbfact_LUXpandMem
/************************************************************************/
(
 int_t iam,
 int_t n,           /* total number of columns */
 int_t vtxXp,       /* current vertex */
 int_t next,        /* number of elements currently in the factors */
 int_t min_new_len, /* minimum new length to allocate */
 int_t mem_type,    /* which type of memory to expand  */
 int_t rout_type,   /* during which type of factorization */
 int_t free_prev_mem, /* =1 if prev_mem has to be freed */
 Pslu_freeable_t *Pslu_freeable,
 Llu_symbfact_t *Llu_symbfact,  /* modified - global LU data structures */
 vtcsInfo_symbfact_t *VInfo,
 psymbfact_stat_t *PS
 )
{
  int_t  *new_mem, *prev_mem, *xsub;
  /* size of the memory to be copied to new store starting from the 
     beginning/end of the memory */
  int_t xsub_nextLvl;  
  int_t exp, prev_xsub_nextLvl, vtxXp_lid;
  int_t *globToLoc, maxNvtcsPProc, nvtcs_loc;
  int_t fstVtx_nextLvl, fstVtx_nextLvl_lid, vtx_lid, i, j;
  int_t len_tcopy_fbeg, len_tcopy_fend, new_len, prev_len;  

  exp  = 2;
  globToLoc = Pslu_freeable->globToLoc;
  nvtcs_loc = VInfo->nvtcs_loc;
  maxNvtcsPProc  = Pslu_freeable->maxNvtcsPProc;
  fstVtx_nextLvl = VInfo->fstVtx_nextLvl;
  vtxXp_lid      = LOCAL_IND( globToLoc[vtxXp] );
  len_tcopy_fbeg = next;
  if (fstVtx_nextLvl == n)
    fstVtx_nextLvl_lid = nvtcs_loc;
  else
    fstVtx_nextLvl_lid = LOCAL_IND( globToLoc[fstVtx_nextLvl] );  

  if ( mem_type == LSUB ) {
    prev_mem = Llu_symbfact->lsub;
    prev_len = Llu_symbfact->szLsub;
    xsub = Llu_symbfact->xlsub;
    if (rout_type == DOMAIN_SYMB)
      prev_xsub_nextLvl = xsub[vtxXp_lid+1];
    else
      prev_xsub_nextLvl = VInfo->xlsub_nextLvl;
  } else if ( mem_type == USUB ) {
    prev_mem = Llu_symbfact->usub;
    prev_len = Llu_symbfact->szUsub;
    xsub = Llu_symbfact->xusub;
    if (rout_type == DOMAIN_SYMB)
      prev_xsub_nextLvl = xsub[vtxXp_lid+1];
    else
      prev_xsub_nextLvl = VInfo->xusub_nextLvl;
  }
  
  len_tcopy_fend = prev_len - prev_xsub_nextLvl;  
  if (rout_type == DNS_UPSEPS || rout_type == DNS_CURSEP) {
    fstVtx_nextLvl = n;
    fstVtx_nextLvl_lid = nvtcs_loc;
    len_tcopy_fend = 0;
  }
#ifdef TEST_SYMB
  printf ("Pe[%d] LUXpand mem_t %d vtxXp %d\n", 
	  iam, mem_type, vtxXp); 
#endif
  new_mem = expand (prev_len, min_new_len, prev_mem,
		    &new_len, len_tcopy_fbeg, len_tcopy_fend, PS);
  if ( !new_mem ) {
    fprintf(stderr, "Pe[%d] Can't exp MemType %d: prv_len %d min_new %d new_l %d\n",
	   iam, mem_type, prev_len, min_new_len, new_len);
    return ERROR_RET;
  }
  
  xsub_nextLvl = new_len - len_tcopy_fend;
  
  /* reset xsub information pointing to A data */
  if (fstVtx_nextLvl != n || rout_type == DOMAIN_SYMB) {
    if (rout_type == DOMAIN_SYMB)
      vtx_lid = vtxXp_lid + 1;
    else {
      vtx_lid = fstVtx_nextLvl_lid +1;
    }
    i = xsub_nextLvl + xsub[vtx_lid] - prev_xsub_nextLvl;
    for (; vtx_lid < nvtcs_loc; vtx_lid ++) {
      j = xsub[vtx_lid+1] - xsub[vtx_lid];
      xsub[vtx_lid] = i;
      i += j;
    }
    xsub[vtx_lid] = i;
  }

  if (free_prev_mem) {
    SUPERLU_FREE (prev_mem);
    PS->allocMem -= 0;
  }
  
  if ( mem_type == LSUB ) {
    Llu_symbfact->lsub   = new_mem;
    Llu_symbfact->szLsub = new_len;
    VInfo->xlsub_nextLvl = xsub_nextLvl;
  } else if ( mem_type == USUB ) {
    Llu_symbfact->usub   = new_mem;
    Llu_symbfact->szUsub = new_len;
    VInfo->xusub_nextLvl = xsub_nextLvl;
  }
  
  Llu_symbfact->no_expand ++;
  return SUCCES_RET;
}

/*
 * Expand the data structures for L and U during the factorization.
 * Return value: SUCCES_RET - successful return
 *               ERROR_RET - error due to a memory alocation failure
 */
/************************************************************************/
int_t psymbfact_LUXpand
/************************************************************************/
(
 int_t iam, 
 int_t n,           /* total number of columns */
 int_t fstVtxLvl_loc, /* first vertex in the level to update */
 int_t vtxXp,         /* current vertex */
 int_t *p_next,        /* number of elements currently in the factors */
 int_t min_new_len, /* minimum new length to allocate */
 int_t mem_type,   /* which type of memory to expand  */
 int_t rout_type,  /* during which type of factorization */
 int_t free_prev_mem, /* =1 if free prev_mem memory */
 Pslu_freeable_t *Pslu_freeable, 
 Llu_symbfact_t *Llu_symbfact,  /* modified - global LU data structures */
 vtcsInfo_symbfact_t *VInfo,
 psymbfact_stat_t *PS
 )
{
  int mem_error;
  int_t  *new_mem, *prev_mem, *xsub, sz_prev_mem;
  /* size of the memory to be copied to new store starting from the 
     beginning/end of the memory */
  int_t exp, prev_xsub_nextLvl, vtxXp_lid, xsub_nextLvl;
  int_t *globToLoc, nvtcs_loc, maxNvtcsPProc;
  int_t fstVtx_nextLvl, fstVtx_nextLvl_lid;
  int_t i, j, k, vtx_lid, len_texp, nelts, nel;
  int_t fstVtxLvl_loc_lid, prev_len, next;
  
  exp  = 2;
  next = *p_next;
  globToLoc = Pslu_freeable->globToLoc;
  nvtcs_loc = VInfo->nvtcs_loc;
  maxNvtcsPProc  = Pslu_freeable->maxNvtcsPProc;
  fstVtx_nextLvl = VInfo->fstVtx_nextLvl;
  
  vtxXp_lid = LOCAL_IND( globToLoc[vtxXp] );
  if (fstVtx_nextLvl == n)
    fstVtx_nextLvl_lid = VInfo->nvtcs_loc;
  else
    fstVtx_nextLvl_lid = LOCAL_IND( globToLoc[fstVtx_nextLvl] );  
  if (rout_type == RL_SYMB)
    fstVtxLvl_loc_lid = LOCAL_IND( globToLoc[fstVtxLvl_loc] );

  if ( mem_type == LSUB ) {
    xsub = Llu_symbfact->xlsub;
    prev_mem = Llu_symbfact->lsub;
    prev_xsub_nextLvl = VInfo->xlsub_nextLvl;
    sz_prev_mem = Llu_symbfact->szLsub;
  } else if ( mem_type == USUB ) {
    xsub = Llu_symbfact->xusub;
    prev_mem = Llu_symbfact->usub;
    prev_xsub_nextLvl = VInfo->xusub_nextLvl;
    sz_prev_mem = Llu_symbfact->szUsub;
  }
#ifdef TEST_SYMB
  printf ("Pe[%d] Expand LU mem_t %d vtxXp %d\n", 
	  iam, mem_type, vtxXp); 
#endif
  /* Try to expand the size of xsub in the existing memory */
  if (rout_type == RL_SYMB) {
    len_texp = 0;
    for (vtx_lid = fstVtxLvl_loc_lid; vtx_lid < fstVtx_nextLvl_lid; vtx_lid ++) {
      nelts = xsub[vtx_lid+1] - xsub[vtx_lid];
      if (nelts == 0) nelts = 1;
      nelts = 2 * nelts;
      if (nelts > Llu_symbfact->cntelt_vtcs[vtx_lid])
	nelts = Llu_symbfact->cntelt_vtcs[vtx_lid];
      len_texp += nelts;
    }
/*     len_texp = 2 * (xsub[fstVtx_nextLvl_lid] - xsub[fstVtxLvl_loc_lid]); */
    prev_len = xsub[fstVtxLvl_loc_lid];
    next = prev_len;
  }
  else {
    nelts = xsub[vtxXp_lid+1] - xsub[vtxXp_lid];
    if (nelts == 0) nelts = 1;
    len_texp = xsub[fstVtx_nextLvl_lid] - xsub[vtxXp_lid+1] +
      4 * nelts;
    prev_len = xsub[vtxXp_lid];
  }
  
  if (prev_len + len_texp >= prev_xsub_nextLvl) {
    /* not enough memory */
    min_new_len = prev_len + len_texp + (sz_prev_mem - prev_xsub_nextLvl);
    if (mem_error = 
	psymbfact_LUXpandMem (iam, n, vtxXp, next, min_new_len, 
			      mem_type, rout_type, 0, Pslu_freeable, Llu_symbfact,
			      VInfo, PS))
      return (mem_error);
    if ( mem_type == LSUB ) 
      new_mem = Llu_symbfact->lsub;
    else if ( mem_type == USUB ) 
      new_mem = Llu_symbfact->usub;
  }
  else 
    new_mem = prev_mem;

  if (mem_type == LSUB && PS->estimLSz < (prev_len + len_texp))
    PS->estimLSz = prev_len + len_texp;
  if (mem_type == USUB && PS->estimUSz < (prev_len + len_texp))
    PS->estimUSz = prev_len;

  /* expand the space */
  if (rout_type == LL_SYMB) {
    i = xsub[vtxXp_lid] + len_texp;
    vtx_lid = fstVtx_nextLvl_lid - 1;
    for (; vtx_lid > vtxXp_lid; vtx_lid --) {
      j = xsub[vtx_lid];  
      nel = 0;
      while (j < xsub[vtx_lid+1] && prev_mem[j] != EMPTY) {
	nel ++; j ++;
      }
      j = xsub[vtx_lid] + nel - 1;  
      k = i - (xsub[vtx_lid+1] - xsub[vtx_lid]) + nel - 1;
      if (k+1 < i)  new_mem[k+1] = EMPTY; 
      while (j >= xsub[vtx_lid]) {
	new_mem[k] = prev_mem[j]; k--; j--;
      }
      k = i;
      i -= (xsub[vtx_lid+1] - xsub[vtx_lid]);
      xsub[vtx_lid+1] = k;
    }
    xsub[vtx_lid+1] = i;
    k = *p_next;
    if (k < xsub[vtx_lid+1])
      new_mem[k] = EMPTY;
  }

  if (rout_type == RL_SYMB) {
    *p_next -= xsub[vtxXp_lid];
    i = xsub[fstVtxLvl_loc_lid] + len_texp;
    vtx_lid = fstVtx_nextLvl_lid - 1;
    for (; vtx_lid >= fstVtxLvl_loc_lid; vtx_lid --) {
      nelts = 2 * (xsub[vtx_lid+1] - xsub[vtx_lid]);
      if (nelts == 0) nelts = 2;
      if (nelts > Llu_symbfact->cntelt_vtcs[vtx_lid])
	nelts = Llu_symbfact->cntelt_vtcs[vtx_lid];
      j = xsub[vtx_lid];  
      nel = 0;
      while (j < xsub[vtx_lid+1] && prev_mem[j] != EMPTY) {
	nel ++; j ++;
      }
      j = xsub[vtx_lid] + nel - 1;  
      k = i - nelts + nel - 1;
      if (k+1 < i) new_mem[k+1] = EMPTY; 
      while (j >= xsub[vtx_lid]) {
	new_mem[k] = prev_mem[j]; k--; j--;
      }
      k = i;
      i -= nelts;
      xsub[vtx_lid+1] = k;
    }
    *p_next += xsub[vtxXp_lid];
  }  

  if (free_prev_mem && new_mem != prev_mem)
    SUPERLU_FREE (prev_mem);
  Llu_symbfact->no_expcp ++;
  
  return SUCCES_RET;
}

/*
 * Expand the data structures for L and U during the factorization.
 * Return value:   0 - successful return
 *               > 0 - number of bytes allocated when run out of space
 */
/************************************************************************/
int_t psymbfact_LUXpand_RL
/************************************************************************/
(
 int_t iam, 
 int_t n,           /* total number of columns */
 int_t vtxXp,       /* current vertex */
 int_t next,        /* number of elements currently in the factors */
 int_t len_texp,    /* length to expand */
 int_t mem_type,    /* which type of memory to expand  */
 Pslu_freeable_t *Pslu_freeable, 
 Llu_symbfact_t *Llu_symbfact,  /* modified - global LU data structures */
 vtcsInfo_symbfact_t *VInfo,
 psymbfact_stat_t *PS
 )
{
  int_t  *new_mem, *prev_mem, *xsub, mem_error, sz_prev_mem;
  /* size of the memory to be copied to new store starting from the 
     beginning/end of the memory */
  int_t exp, prev_xsub_nextLvl, vtxXp_lid, xsub_nextLvl;
  int_t *globToLoc, nvtcs_loc, maxNvtcsPProc;
  int_t fstVtx_nextLvl, fstVtx_nextLvl_lid;
  int_t i, j, k, vtx_lid, nel;
  int_t fstVtxLvl_loc_lid, prev_len, min_new_len;

#ifdef TEST_SYMB
  printf ("Pe[%d] Expand LU_RL mem_t %d vtxXp %d\n", 
	  iam, mem_type, vtxXp); 
#endif
  globToLoc = Pslu_freeable->globToLoc;
  nvtcs_loc = VInfo->nvtcs_loc;
  maxNvtcsPProc  = Pslu_freeable->maxNvtcsPProc;
  fstVtx_nextLvl = VInfo->fstVtx_nextLvl;
  
  vtxXp_lid = LOCAL_IND( globToLoc[vtxXp] );
  if (fstVtx_nextLvl == n)
    fstVtx_nextLvl_lid = VInfo->nvtcs_loc;
  else
    fstVtx_nextLvl_lid = LOCAL_IND( globToLoc[fstVtx_nextLvl] );  

  if ( mem_type == LSUB ) {
    xsub = Llu_symbfact->xlsub;
    prev_mem = Llu_symbfact->lsub;
    prev_xsub_nextLvl = VInfo->xlsub_nextLvl;
    sz_prev_mem = Llu_symbfact->szLsub;
  } else if ( mem_type == USUB ) {
    xsub = Llu_symbfact->xusub;
    prev_mem = Llu_symbfact->usub;
    prev_xsub_nextLvl = VInfo->xusub_nextLvl;
    sz_prev_mem = Llu_symbfact->szUsub;
  }
  else ABORT("Tries to expand nonexisting memory type.\n");
  
  /* Try to expand the size of xsub in the existing memory */
  prev_len = xsub[vtxXp_lid];
  
  if (prev_len + len_texp >= prev_xsub_nextLvl) {
    /* not enough memory */
    min_new_len = prev_len + len_texp + (sz_prev_mem - prev_xsub_nextLvl);
    if (mem_error = 
	psymbfact_LUXpandMem (iam, n, vtxXp, next, min_new_len, 
			      mem_type, RL_SYMB, 0, Pslu_freeable, Llu_symbfact,
			      VInfo, PS))
      return (mem_error);
    if ( mem_type == LSUB ) 
      new_mem = Llu_symbfact->lsub;
    else if ( mem_type == USUB ) 
      new_mem = Llu_symbfact->usub;
  }
  else 
    new_mem = prev_mem;

  /* expand the space */
  if (mem_type == LSUB && PS->estimLSz < (prev_len + len_texp))
    PS->estimLSz = prev_len + len_texp;
  if (mem_type == USUB && PS->estimUSz < (prev_len + len_texp))
    PS->estimUSz = prev_len;

  i = xsub[vtxXp_lid] + len_texp;
  vtx_lid = fstVtx_nextLvl_lid - 1;
  for (; vtx_lid > vtxXp_lid; vtx_lid --) {
    j = xsub[vtx_lid];  
    nel = 0;
    while (j < xsub[vtx_lid+1] && prev_mem[j] != EMPTY) {
      nel ++; j++;
    }
    j = xsub[vtx_lid] + nel - 1;  
    k = i - Llu_symbfact->cntelt_vtcs[vtx_lid] + nel - 1;
    if (k+1 < i) 
      new_mem[k+1] = EMPTY; 
    while (j >= xsub[vtx_lid]) {
      new_mem[k] = prev_mem[j];
      k--; j--;
    }
    k = i;
    i -= Llu_symbfact->cntelt_vtcs[vtx_lid];
    xsub[vtx_lid+1] = k;
  }
  xsub[vtx_lid+1] = i;
  k = next;
  if (k < xsub[vtx_lid+1])
    new_mem[k] = EMPTY;
  
  if (new_mem != prev_mem)
    SUPERLU_FREE (prev_mem);
  Llu_symbfact->no_expcp ++;
  
  return SUCCES_RET;
}

/*
 * Expand the data structures for L and U pruned during the factorization.
 * Return value: SUCCES_RET - successful return
 *               ERROR_RET - error when run out of space
 */
/************************************************************************/
int_t psymbfact_prLUXpand
/************************************************************************/
(
 int_t iam, 
 int_t min_new_len, /* minimum new length to allocate */ 
 MemType mem_type,  /* which type of memory to expand  */
 Llu_symbfact_t *Llu_symbfact, /* modified L/U pruned structures */
 psymbfact_stat_t *PS
 )
{
  int_t *prev_mem, *new_mem;
  int_t prev_len, new_len, len_tcopy_fbeg;
  
  if ( mem_type == LSUB_PR ) {
    prev_len = Llu_symbfact->szLsubPr;
    prev_mem = Llu_symbfact->lsubPr;
    len_tcopy_fbeg = Llu_symbfact->indLsubPr;
  } else if ( mem_type == USUB_PR ) {
    prev_len = Llu_symbfact->szUsubPr;
    prev_mem = Llu_symbfact->usubPr;
    len_tcopy_fbeg = Llu_symbfact->indUsubPr;
  } else ABORT("Tries to expand nonexisting memory type.\n");
  
#ifdef TEST_SYMB
  printf ("Pe[%d] Expand prmem prev_len %d min_new_l %d len_tfbeg %d\n", 
	  iam, prev_len, min_new_len, len_tcopy_fbeg);
#endif
  
  new_mem = expand (prev_len, min_new_len, prev_mem, 
		    &new_len, len_tcopy_fbeg, 0, PS);
  
  if ( !new_mem ) {
    fprintf(stderr, "Can't expand MemType %d: \n", mem_type);
    return (ERROR_RET);
  }
  
  Llu_symbfact->no_expand_pr ++;
  if ( mem_type == LSUB_PR ) {
    Llu_symbfact->lsubPr  = new_mem;
    Llu_symbfact->szLsubPr = new_len;
  } else if ( mem_type == USUB_PR ) {
    Llu_symbfact->usubPr  = new_mem;
    Llu_symbfact->szUsubPr = new_len;
  } else ABORT("Tries to expand nonexisting memory type.\n");
  
  SUPERLU_FREE (prev_mem);

  return SUCCES_RET;
}
