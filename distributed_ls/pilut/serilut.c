/*
 * serilut.c
 *
 * This file implements hypre_ILUT in the local part of the matrix
 *
 * Started 10/18/95
 * George
 *
 * 7/8 MRG
 * - added rrowlen and verified
 * 7/22 MRG
 * - removed hypre_SelectInterior function form hypre_SerILUT code
 * - changed FindMinGreater to hypre_ExtractMinLR
 * - changed lr to using permutation; this allows reorderings like RCM.
 *
 * 12/4 AJC
 * - Changed code to handle modified matrix storage format with multiple blocks
 *
 * 1/13 AJC
 * - Modified code with macros to allow both 0 and 1-based indexing
 *
 * $Id$
 *
 */

#include "./DistributedMatrixPilutSolver.h"
#include "ilu.h"


/*************************************************************************
* This function takes a matrix and performs an hypre_ILUT of the internal nodes
**************************************************************************/
int hypre_SerILUT(DataDistType *ddist, HYPRE_DistributedMatrix matrix,
             FactorMatType *ldu,
	     ReduceMatType *rmat, int maxnz, double tol, 
             hypre_PilutSolverGlobals *globals)
{
  int i, ii, j, k, kk, l, m, ierr, diag_present;
  int *perm, *iperm, 
          *usrowptr, *uerowptr, *ucolind,
          *rnz, **rcolind;
  int row_size, *col_ind;
  double *values, *uvalues, *dvalues, *nrm2s, **rvalues;
  int nlocal, nbnd;
  double mult, rtol;
  int *structural_union;


  nrows    = ddist->ddist_nrows;
  lnrows   = ddist->ddist_lnrows;
  firstrow = ddist->ddist_rowdist[mype];
  lastrow  = ddist->ddist_rowdist[mype+1];

  usrowptr = ldu->usrowptr;
  uerowptr = ldu->uerowptr;
  ucolind  = ldu->ucolind;
  uvalues  = ldu->uvalues;
  dvalues  = ldu->dvalues;
  nrm2s    = ldu->nrm2s;
  perm     = ldu->perm;
  iperm    = ldu->iperm;

  /* Allocate work space */
  if (jr) { free(jr); jr = NULL; }
  jr = hypre_idx_malloc_init(nrows, -1, "hypre_SerILUT: jr");
  if (lr) { free(lr); lr = NULL; }
  lr = hypre_idx_malloc_init(nrows, -1, "hypre_SerILUT: lr");
  if (jw) { free(jw); jw = NULL; }
  jw = hypre_idx_malloc(nrows, "hypre_SerILUT: jw");
  if (w) { free(w); w = NULL; }
  w  =  hypre_fp_malloc(nrows, "hypre_SerILUT: w" );

  /* Find structural union of local rows */

#ifdef HYPRE_TIMING
{
   int           FSUtimer;
   FSUtimer = hypre_InitializeTiming( "FindStructuralUnion");
   hypre_BeginTiming( FSUtimer );
#endif

  ierr = FindStructuralUnion( matrix, &structural_union, globals );

#ifdef HYPRE_TIMING
   hypre_EndTiming( FSUtimer );
   /* hypre_FinalizeTiming( FSUtimer ); */
}
#endif

  if(ierr) return(ierr);

  /* Exchange structural unions with other processors */
  ierr = ExchangeStructuralUnions( ddist, &structural_union, globals );
  if(ierr) return(ierr);

  /* Select the rows to be factored */
#ifdef HYPRE_TIMING
  {
   int           SItimer;
   SItimer = hypre_InitializeTiming( "hypre_SelectInterior");
   hypre_BeginTiming( SItimer );
#endif
  nlocal = hypre_SelectInterior( lnrows, matrix, structural_union,
                           perm, iperm, globals );
#ifdef HYPRE_TIMING
   hypre_EndTiming( SItimer );
   /* hypre_FinalizeTiming( SItimer ); */
  }
#endif

  /* Structural Union no longer required */
  hypre_TFree( structural_union );

  nbnd = lnrows - nlocal ;
#ifdef HYPRE_DEBUG
  printf("nbnd = %d, lnrows=%d, nlocal=%d\n", nbnd, lnrows, nlocal );
#endif

  ldu->nnodes[0] = nlocal;

#ifdef HYPRE_TIMING
   globals->SDSeptimer = hypre_InitializeTiming("hypre_SecondDrop Separation");
   globals->SDKeeptimer = hypre_InitializeTiming("hypre_SecondDrop extraction of kept elements");
   globals->SDUSeptimer = hypre_InitializeTiming("hypre_SecondDropUpdate Separation");
   globals->SDUKeeptimer = hypre_InitializeTiming("hypre_SecondDropUpdate extraction of kept elements");
#endif

#ifdef HYPRE_TIMING
  {
   int           LFtimer;
   LFtimer = hypre_InitializeTiming( "Local factorization computational stage");
   hypre_BeginTiming( LFtimer );
#endif

  /* myprintf("Nlocal: %d, Nbnd: %d\n", nlocal, nbnd); */

  /*******************************************************************/
  /* Go and factor the nlocal rows                                   */
  /*******************************************************************/
  for (ii=0; ii<nlocal; ii++) {
    i = perm[ii];
    rtol = nrm2s[i]*tol;  /* Compute relative tolerance */

    /* Initialize work space  */
    ierr = HYPRE_DistributedMatrixGetRow( matrix, firstrow+i, &row_size,
               &col_ind, &values);
    if (ierr) return(ierr);

    for (lastjr=1, lastlr=0, j=0, diag_present=0; j<row_size; j++) {
      if (iperm[ col_ind[j] - firstrow ] < iperm[i]) 
        lr[lastlr++] = iperm[ col_ind[j]-firstrow]; /* Copy the L elements separately */

      if (col_ind[j] != i+firstrow) { /* Off-diagonal element */
        jr[col_ind[j]] = lastjr;
        jw[lastjr] = col_ind[j];
        w[lastjr] = values[j];
        lastjr++;
      }
      else { /* Put the diagonal element at the beginning */
        diag_present = 1;
        jr[i+firstrow] = 0;
        jw[0] = i+firstrow;
        w[0] = values[j];
      }
    }

    if( !diag_present ) /* No diagonal element was found; insert a zero */
    {
      jr[i+firstrow] = 0;
      jw[0] = i+firstrow;
      w[0] = 0.0;
    }

    ierr = HYPRE_DistributedMatrixRestoreRow( matrix, firstrow+i, &row_size,
               &col_ind, &values);

    k = -1;
    while (lastlr != 0) {
      /* since fill may create new L elements, and they must by done in order
       * of the permutation, search for the min each time.
       * Note that we depend on the permutation order following natural index
       * order for the interior rows. */
      kk = perm[hypre_ExtractMinLR( globals )];
      k  = kk+firstrow;

      mult = w[jr[k]]*dvalues[kk];
      w[jr[k]] = mult;

      if (fabs(mult) < rtol)
        continue;	/* First drop test */

      for (l=usrowptr[kk]; l<uerowptr[kk]; l++) {
        m = jr[ucolind[l]];

        if (m == -1 && fabs(mult*uvalues[l]) < rtol*0.5)
          continue;  /* Don't add fill if the element is too small */

        if (m == -1) {  /* Create fill */
          if (iperm[ucolind[l]-firstrow] < iperm[i]) 
            lr[lastlr++] = iperm[ucolind[l]-firstrow]; /* Copy the L elements separately */

          jr[ucolind[l]] = lastjr;
          jw[lastjr] = ucolind[l];
          w[lastjr] = 0.0;
          m = lastjr++;
        }
        w[m] -= mult*uvalues[l];
      }
    }

    /* Apply 2nd dropping rule -- forms L and U */
    hypre_SecondDrop(maxnz, rtol, i+firstrow, perm, iperm, ldu, globals );
  }

#ifdef HYPRE_TIMING
   hypre_EndTiming( LFtimer );
   /* hypre_FinalizeTiming( LFtimer ); */
  }
#endif
#ifdef HYPRE_TIMING
  {
   int           FRtimer;
   FRtimer = hypre_InitializeTiming( "Local factorization Schur complement stage");
   hypre_BeginTiming( FRtimer );
#endif

  /******************************************************************/
  /* Form the reduced matrix                                        */
  /******************************************************************/
  /* Allocate memory for the reduced matrix */
  rnz =
    rmat->rmat_rnz     = hypre_idx_malloc(nbnd, "hypre_SerILUT: rmat->rmat_rnz"    );
  rmat->rmat_rrowlen   = hypre_idx_malloc(nbnd, "hypre_SerILUT: rmat->rmat_rrowlen");
  rcolind =
    rmat->rmat_rcolind = (int **)hypre_mymalloc(sizeof(int *)*nbnd, "hypre_SerILUT: rmat->rmat_rcolind");
  rvalues =
    rmat->rmat_rvalues =  (double **)hypre_mymalloc(sizeof(double *)*nbnd, "hypre_SerILUT: rmat->rmat_rvalues");
  rmat->rmat_ndone = nlocal;
  rmat->rmat_ntogo = nbnd;

  for (ii=nlocal; ii<lnrows; ii++) {
    i = perm[ii];
    rtol = nrm2s[i]*tol;  /* Compute relative tolerance */

    /* Initialize work space */
    ierr = HYPRE_DistributedMatrixGetRow( matrix, firstrow+i, &row_size,
               &col_ind, &values);
    if (ierr) return(ierr);

    for (lastjr=1, lastlr=0, j=0, diag_present=0; j<row_size; j++) {
      if (col_ind[j] >= firstrow  &&
	  col_ind[j] < lastrow    &&
	  iperm[col_ind[j]-firstrow] < nlocal) 
        lr[lastlr++] = iperm[col_ind[j]-firstrow]; /* Copy the L elements separately */

      if (col_ind[j] != i+firstrow) { /* Off-diagonal element */
        jr[col_ind[j]] = lastjr;
        jw[lastjr] = col_ind[j];
        w[lastjr] = values[j];
        lastjr++;
      }
      else { /* Put the diagonal element at the begining */
        diag_present = 1;
        jr[i+firstrow] = 0;
        jw[0] = i+firstrow;
        w[0] = values[j];
      }
    }

     if( !diag_present ) /* No diagonal element was found; insert a zero */
    {
      jr[i+firstrow] = 0;
      jw[0] = i+firstrow;
      w[0] = 0.0;
    }

    ierr = HYPRE_DistributedMatrixRestoreRow( matrix, firstrow+i, &row_size,
               &col_ind, &values);

    k = -1;
    while (lastlr != 0) {
      kk = perm[hypre_ExtractMinLR(globals)];
      k  = kk+firstrow;

      mult = w[jr[k]]*dvalues[kk];
      w[jr[k]] = mult;

      if (fabs(mult) < rtol)
        continue;	/* First drop test */

      for (l=usrowptr[kk]; l<uerowptr[kk]; l++) {
        m = jr[ucolind[l]];

        if (m == -1 && fabs(mult*uvalues[l]) < rtol*0.5)
          continue;  /* Don't add fill if the element is too small */

        if (m == -1) {  /* Create fill */
	  hypre_CheckBounds(firstrow, ucolind[l], lastrow, globals);
          if (iperm[ucolind[l]-firstrow] < nlocal) 
            lr[lastlr++] = iperm[ucolind[l]-firstrow]; /* Copy the L elements separately */

          jr[ucolind[l]] = lastjr;
          jw[lastjr] = ucolind[l];
          w[lastjr] = 0.0;
          m = lastjr++;
        }
        w[m] -= mult*uvalues[l];
      }
    }

    /* Apply 2nd dropping rule -- forms partial L and rmat */
    hypre_SecondDropUpdate(maxnz, MAX(3*maxnz, row_size),
		     rtol, i+firstrow,
		     nlocal, perm, iperm, ldu, rmat, globals);
  }

#ifdef HYPRE_TIMING
   hypre_EndTiming( FRtimer );
   /* hypre_FinalizeTiming( FRtimer ); */
  }
#endif

  hypre_free_multi(jr, jw, lr, w, -1);
  jr = NULL;
  jw = NULL;
  lr = NULL;
  w = NULL;

  return(ierr);
}


/*************************************************************************
* This function selects the interior nodes (ones w/o nonzeros corresponding
* to other PEs) and permutes them first, then boundary nodes last.
* It takes a vector that marks rows as being forced to not be in the interior.
* For full generality this would also mark them in the map, but it doesn't.
**************************************************************************/
int hypre_SelectInterior( int local_num_rows, 
                    HYPRE_DistributedMatrix matrix, 
                    int *external_rows,
		    int *newperm, int *newiperm, 
                    hypre_PilutSolverGlobals *globals )
{
  int nbnd, nlocal, i, j, ierr;
  int break_loop; /* marks finding an element making this row exterior. -AC */
  int row_size, *col_ind;
  double *values;

  /* Determine which vertices are in the boundary,
   * permuting interior rows first then boundary nodes. */
  nbnd = 0;
  nlocal = 0;
  for (i=0; i<local_num_rows; i++) 
  {
    if (external_rows[i])
    {
      newperm[local_num_rows-nbnd-1] = i;
      newiperm[i] = local_num_rows-nbnd-1;
      nbnd++;
    } else
    {
      ierr = HYPRE_DistributedMatrixGetRow( matrix, firstrow+i, &row_size,
               &col_ind, &values);
      if (ierr) return(ierr);

      for (j=0, break_loop=0; ( j<row_size )&& (break_loop == 0); j++) 
      {
        if (col_ind[j] < firstrow || col_ind[j] >= lastrow) 
        {
          newperm[local_num_rows-nbnd-1] = i;
          newiperm[i] = local_num_rows-nbnd-1;
          nbnd++;
          break_loop = 1;
        }
      }

      ierr = HYPRE_DistributedMatrixRestoreRow( matrix, firstrow+i, &row_size,
               &col_ind, &values);

      if ( break_loop == 0 ) 
      {
        newperm[nlocal] = i;
        newiperm[i] = nlocal;
        nlocal++;
      }
    }
  }

  return nlocal;
}


/*************************************************************************
* FindStructuralUnion
*   Produces a vector of length n that marks the union of the nonzero
*   structure of all locally stored rows, not including locally stored columns.
**************************************************************************/
int FindStructuralUnion( HYPRE_DistributedMatrix matrix, 
                    int **structural_union,
                    hypre_PilutSolverGlobals *globals )
{ 
  int ierr=0, i, j, row_size, *col_ind;

  /* Allocate and clear structural_union vector */
  *structural_union = hypre_CTAlloc( int, nrows );

  /* Loop through rows */
  for ( i=0; i< lnrows; i++ )
  {
    /* Get row structure; no values needed */
    ierr = HYPRE_DistributedMatrixGetRow( matrix, firstrow+i, &row_size,
               &col_ind, NULL );
    if (ierr) return(ierr);

    /* Loop through nonzeros in this row */
    for ( j=0; j<row_size; j++)
    {
      if (col_ind[j] < firstrow || col_ind[j] >= lastrow) 
      {
        (*structural_union)[ col_ind[j] ] = 1;
      }
    }

    /* Restore row structure */
    ierr = HYPRE_DistributedMatrixRestoreRow( matrix, firstrow+i, &row_size,
               &col_ind, NULL );
    if (ierr) return(ierr);

  }

  return(ierr);
}


/*************************************************************************
* ExchangeStructuralUnions
*   Exchanges structural union vectors with other processors and produces
*   a vector the size of the number of locally stored rows that marks
*   whether any exterior processor has a nonzero in the column corresponding
*   to each row. This is used to determine if a local row might have to
*   update an off-processor row.
**************************************************************************/
int ExchangeStructuralUnions( DataDistType *ddist,
                    int **structural_union,
                    hypre_PilutSolverGlobals *globals )
{ 
  int ierr=0, i, j, *recv_unions;

  /* allocate space for receiving unions */
  recv_unions = hypre_CTAlloc( int, nrows );

  /* combine my structural_union with all other processors */
  MPI_Allreduce( *structural_union, recv_unions, nrows, 
                 MPI_INT, MPI_LOR, pilut_comm );

  /* free and reallocate structural union so that is of local size */
  hypre_TFree( *structural_union );
  *structural_union = hypre_TAlloc( int, lnrows );

  hypre_memcpy_int( *structural_union, &recv_unions[firstrow], lnrows );

  /* deallocate recv_unions */
  hypre_TFree( recv_unions );

  return(ierr);
}


/*************************************************************************
* This function applies the second droping rule where maxnz elements 
* greater than tol are kept. The elements are stored into LDU.
**************************************************************************/
void hypre_SecondDrop(int maxnz, double tol, int row,
		int *perm, int *iperm,
		FactorMatType *ldu, hypre_PilutSolverGlobals *globals)
{
  int i, j, ierr;
  int max, nz, diag, lrow;
  int first, last, itmp;
  double dtmp;

  /* Reset the jr array, it is not needed any more */
  for (i=0; i<lastjr; i++) 
    jr[jw[i]] = -1;

  lrow = row-firstrow;
  diag = iperm[lrow];

  /* Deal with the diagonal element first */
  assert(jw[0] == row);
  if (w[0] != 0.0) 
    ldu->dvalues[lrow] = 1.0/w[0];
  else { /* zero pivot */
    printf("Zero pivot in row %d, adding e to proceed!\n", row);
    ldu->dvalues[lrow] = 1.0/tol;
  }
  jw[0] = jw[--lastjr];
  w[0] = w[lastjr];


  /* First go and remove any off diagonal elements bellow the tolerance */
  for (i=0; i<lastjr;) {
    if (fabs(w[i]) < tol) {
      jw[i] = jw[--lastjr];
      w[i] = w[lastjr];
    }
    else
      i++;
  }

#ifdef HYPRE_TIMING
  hypre_BeginTiming( globals->SDSeptimer );
#endif

  if (lastjr == 0)
    last = first = 0;
  else { /* Perform a Qsort type pass to seperate L and U entries */
    last = 0, first = lastjr-1;
    while (1) {
      while (last < first && iperm[jw[last]-firstrow] < diag)
        last++;
      while (last < first && iperm[jw[first]-firstrow] > diag)
        first--;

      if (last < first) {
        SWAP(jw[first], jw[last], itmp);
        SWAP(w[first], w[last], dtmp);
        last++; first--;
      }

      if (last == first) {
        if (iperm[jw[last]-firstrow] < diag) {
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
#ifdef HYPRE_TIMING
  hypre_EndTiming( globals->SDSeptimer );
#endif

  /*****************************************************************
  * The entries between [0, last) are part of L
  * The entries [first, lastjr) are part of U
  ******************************************************************/

#ifdef HYPRE_TIMING
  hypre_BeginTiming(globals-> SDKeeptimer );
#endif

  /* Now, I want to keep maxnz elements of L. Go and extract them */

  ierr = hypre_DoubleQuickSplit( w, jw, last, maxnz ); 
  if (ierr) exit;
  for ( j= hypre_max(0,last-maxnz); j< last; j++ ) 
  {
     ldu->lcolind[ldu->lerowptr[lrow]] = jw[ j ];
     ldu->lvalues[ldu->lerowptr[lrow]++] = w[ j ];
  }


  /* This was the previous insertion sort that was replaced with
     the QuickSplit routine above. AJC, 5/00 
  for (nz=0; nz<maxnz && last>0; nz++) {
    for (max=0, j=1; j<last; j++) {
      if (fabs(w[j]) > fabs(w[max]))
        max = j;
    }

    ldu->lcolind[ldu->lerowptr[lrow]] = jw[max];
    ldu->lvalues[ldu->lerowptr[lrow]] = w[max];
    ldu->lerowptr[lrow]++;

    jw[max] = jw[--last];
    w[max] = w[last];
  }
  */


  /* Now, I want to keep maxnz elements of U. Go and extract them */
  ierr = hypre_DoubleQuickSplit( w+first, jw+first, lastjr-first, maxnz ); 
  if (ierr) exit;
  for ( j=hypre_max(first, lastjr-maxnz); j< lastjr; j++ ) 
  {
     ldu->ucolind[ldu->uerowptr[lrow]] = jw[ j ];
     ldu->uvalues[ldu->uerowptr[lrow]++] = w[ j ];
  }

  /*
     This was the previous insertion sort that was replaced with
     the QuickSplit routine above. AJC, 5/00 
  for (nz=0; nz<maxnz && lastjr>first; nz++) {
    for (max=first, j=first+1; j<lastjr; j++) {
      if (fabs(w[j]) > fabs(w[max]))
        max = j;
    }

    ldu->ucolind[ldu->uerowptr[lrow]] = jw[max];
    ldu->uvalues[ldu->uerowptr[lrow]] = w[max];
    ldu->uerowptr[lrow]++;

    jw[max] = jw[--lastjr];
    w[max] = w[lastjr];
  }
  */


#ifdef HYPRE_TIMING
  hypre_EndTiming( globals->SDKeeptimer );
#endif


}


/*************************************************************************
* This function applyies the second droping rule whre maxnz elements 
* greater than tol are kept. The elements are stored into L and the Rmat.
* This version keeps only maxnzkeep 
**************************************************************************/
void hypre_SecondDropUpdate(int maxnz, int maxnzkeep, double tol, int row,
		      int nlocal, int *perm, int *iperm, 
		      FactorMatType *ldu, ReduceMatType *rmat,
                      hypre_PilutSolverGlobals *globals )
{
  int i, j, k, nl;
  int max, nz, lrow, rrow;
  int last, first, itmp;
  double dtmp;
  int ierr=0;


  /* Reset the jr array, it is not needed any more */
  for (i=0; i<lastjr; i++) 
    jr[jw[i]] = -1;

  lrow = row-firstrow;
  rrow = iperm[lrow] - nlocal;

  /* First go and remove any elements of the row bellow the tolerance */
  for (i=1; i<lastjr;) {
    if (fabs(w[i]) < tol) {
      jw[i] = jw[--lastjr];
      w[i] = w[lastjr];
    }
    else
      i++;
  }


#ifdef HYPRE_TIMING
  hypre_BeginTiming( globals->SDUSeptimer );
#endif

   if (lastjr == 1)
    last = first = 1;
  else { /* Perform a Qsort type pass to seperate L and U entries */
    last = 1, first = lastjr-1;
    while (1) {
      while (last < first         &&     /* and [last] is L */
	     jw[last] >= firstrow &&
	     jw[last] < lastrow   &&
	     iperm[jw[last]-firstrow] < nlocal)
        last++;
      while (last < first            &&  /* and [first] is not L */
	     !(jw[first] >= firstrow &&
	       jw[first] < lastrow   &&
	       iperm[jw[first]-firstrow] < nlocal))
        first--;

      if (last < first) {
        SWAP(jw[first], jw[last], itmp);
        SWAP( w[first],  w[last], dtmp);
        last++; first--;
      }

      if (last == first) {
        if (jw[last] >= firstrow &&
	    jw[last] < lastrow   &&
	    iperm[jw[last]-firstrow] < nlocal) {
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
#ifdef HYPRE_TIMING
  hypre_EndTiming( globals->SDUSeptimer );
#endif

  /*****************************************************************
  * The entries between [1, last) are part of L
  * The entries [first, lastjr) are part of U
  ******************************************************************/

#ifdef HYPRE_TIMING
  hypre_BeginTiming( globals->SDUKeeptimer );
#endif


  /* Keep large maxnz elements of L */
  ierr = hypre_DoubleQuickSplit( w+1, jw+1, last-1, maxnz ); 
  if (ierr) exit;
  for ( j= hypre_max(1,last-maxnz); j< last; j++ ) 
  {
     ldu->lcolind[ldu->lerowptr[lrow]] = jw[ j ];
     ldu->lvalues[ldu->lerowptr[lrow]++] = w[ j ];
  }


  /* This was the previous insertion sort that was replaced with
     the QuickSplit routine above. AJC, 5/00 
  for (nz=0; nz<maxnz && last>1; nz++) {
    for (max=1, j=2; j<last; j++) {
      if (fabs(w[j]) > fabs(w[max]))
        max = j;
    }

    ldu->lcolind[ldu->lerowptr[lrow]] = jw[max];
    ldu->lvalues[ldu->lerowptr[lrow]] =  w[max];
    ldu->lerowptr[lrow]++;

    jw[max] = jw[--last];
    w[max] = w[last];
  }
  */

  /* Allocate appropriate amount of memory for the reduced row */
  nl = MIN(lastjr-first+1, maxnzkeep);
  rmat->rmat_rnz[rrow] = nl;
  rmat->rmat_rcolind[rrow] = hypre_idx_malloc(nl, "hypre_SecondDropUpdate: rmat->rmat_rcolind[rrow]");
  rmat->rmat_rvalues[rrow] =  hypre_fp_malloc(nl, "hypre_SecondDropUpdate: rmat->rmat_rvalues[rrow]");

  rmat->rmat_rrowlen[rrow]    = nl;
  rmat->rmat_rcolind[rrow][0] = row;  /* Put the diagonal at the begining */
  rmat->rmat_rvalues[rrow][0] = w[0];

  if (nl == lastjr-first+1) { /* Simple copy */
    for (i=1,j=first; j<lastjr; j++,i++) {
      rmat->rmat_rcolind[rrow][i] = jw[j];
      rmat->rmat_rvalues[rrow][i] = w[j];
    }
  }
  else { /* Keep large nl elements in the reduced row */
    for (nz=1; nz<nl; nz++) {
      for (max=first, j=first+1; j<lastjr; j++) {
        if (fabs(w[j]) > fabs(w[max]))
          max = j;
      }

      rmat->rmat_rcolind[rrow][nz] = jw[max];
      rmat->rmat_rvalues[rrow][nz] = w[max];

      jw[max] = jw[--lastjr];
      w[max] = w[lastjr];
    }
  }
#ifdef HYPRE_TIMING
  hypre_EndTiming( globals->SDUKeeptimer );
#endif

}

