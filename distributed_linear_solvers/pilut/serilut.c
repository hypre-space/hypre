/*
 * serilut.c
 *
 * This file implements ILUT in the local part of the matrix
 *
 * Started 10/18/95
 * George
 *
 * 7/8 MRG
 * - added rrowlen and verified
 * 7/22 MRG
 * - removed SelectInterior function form SerILUT code
 * - changed FindMinGreater to ExtractMinLR
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
* This function takes a matrix and performs an ILUT of the internal nodes
**************************************************************************/
int SerILUT(DataDistType *ddist, HYPRE_DistributedMatrix matrix,
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
  jr = idx_malloc_init(nrows, -1, "SerILUT: jr");
  lr = idx_malloc_init(nrows, -1, "SerILUT: lr");
  jw = idx_malloc(nrows, "SerILUT: jw");
  w  =  fp_malloc(nrows, "SerILUT: w" );

  /* Select the rows to be factored */
  nlocal = SelectInterior( lnrows, matrix,
                           perm, iperm, globals );
  nbnd = lnrows - nlocal ;
  ldu->nnodes[0] = nlocal;

  /* myprintf("Nlocal: %d, Nbnd: %d\n", nlocal, nbnd); */

  /*******************************************************************/
  /* Go and factor the nlocal rows                                   */
  /*******************************************************************/
  for (ii=0; ii<nlocal; ii++) {
    i = perm[ii];
    rtol = nrm2s[i]*tol;  /* Compute relative tolerance */

    /* Initialize work space  */
    ierr = HYPRE_GetDistributedMatrixRow( matrix, firstrow+i, &row_size,
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

    ierr = HYPRE_RestoreDistributedMatrixRow( matrix, firstrow+ii, &row_size,
               &col_ind, &values);

    k = -1;
    while (lastlr != 0) {
      /* since fill may create new L elements, and they must by done in order
       * of the permutation, search for the min each time.
       * Note that we depend on the permutation order following natural index
       * order for the interior rows. */
      kk = perm[ExtractMinLR( globals )];
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
    SecondDrop(maxnz, rtol, i+firstrow, perm, iperm, ldu, globals );
  }

  /******************************************************************/
  /* Form the reduced matrix                                        */
  /******************************************************************/
  /* Allocate memory for the reduced matrix */
  rnz =
    rmat->rmat_rnz     = idx_malloc(nbnd, "SerILUT: rmat->rmat_rnz"    );
  rmat->rmat_rrowlen   = idx_malloc(nbnd, "SerILUT: rmat->rmat_rrowlen");
  rcolind =
    rmat->rmat_rcolind = (int **)mymalloc(sizeof(int *)*nbnd, "SerILUT: rmat->rmat_rcolind");
  rvalues =
    rmat->rmat_rvalues =  (double **)mymalloc(sizeof(double *)*nbnd, "SerILUT: rmat->rmat_rvalues");
  rmat->rmat_ndone = nlocal;
  rmat->rmat_ntogo = nbnd;

  for (ii=nlocal; ii<lnrows; ii++) {
    i = perm[ii];
    rtol = nrm2s[i]*tol;  /* Compute relative tolerance */

    /* Initialize work space */
    ierr = HYPRE_GetDistributedMatrixRow( matrix, firstrow+i, &row_size,
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

    ierr = HYPRE_RestoreDistributedMatrixRow( matrix, firstrow+ii, &row_size,
               &col_ind, &values);

    k = -1;
    while (lastlr != 0) {
      kk = perm[ExtractMinLR(globals)];
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
	  CheckBounds(firstrow, ucolind[l], lastrow, globals);
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
    SecondDropUpdate(maxnz, MAX(3*maxnz, row_size),
		     rtol, i+firstrow,
		     nlocal, perm, iperm, ldu, rmat, globals);
  }

  free_multi(jr, jw, lr, w, -1);

  return(ierr);
}


/*************************************************************************
* This function selects the interior nodes (ones w/o nonzeros corresponding
* to other PEs) and permutes them first, then boundary nodes last.
* For full generality this would also mark them in the map, but it doesn't.
**************************************************************************/
int SelectInterior( int local_num_rows, HYPRE_DistributedMatrix matrix, 
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
  for (i=0; i<local_num_rows; i++) {
    break_loop = 0;

    ierr = HYPRE_GetDistributedMatrixRow( matrix, firstrow+i, &row_size,
               &col_ind, &values);
    if (ierr) return(ierr);

    for (j=0; ( j<row_size )&& (break_loop == 0); j++) 
    {
      if (col_ind[j] < firstrow || col_ind[j] >= lastrow) 
      {
        newperm[local_num_rows-nbnd-1] = i;
        newiperm[i] = local_num_rows-nbnd-1;
        nbnd++;
        break_loop = 1;
      }
    }

    ierr = HYPRE_RestoreDistributedMatrixRow( matrix, firstrow+i, &row_size,
               &col_ind, &values);

    if ( break_loop == 0 ) {
      newperm[nlocal] = i;
      newiperm[i] = nlocal;
      nlocal++;
    }
  }

  return nlocal;
}


/*************************************************************************
* This function applies the second droping rule where maxnz elements 
* greater than tol are kept. The elements are stored into LDU.
**************************************************************************/
void SecondDrop(int maxnz, double tol, int row,
		int *perm, int *iperm,
		FactorMatType *ldu, hypre_PilutSolverGlobals *globals)
{
  int i, j;
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
  /*****************************************************************
  * The entries between [0, last) are part of L
  * The entries [first, lastjr) are part of U
  ******************************************************************/


  /* Now, I want to keep maxnz elements of L. Go and extract them */
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


  /* Now, I want to keep maxnz elements of U. Go and extract them */
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

}


/*************************************************************************
* This function applyies the second droping rule whre maxnz elements 
* greater than tol are kept. The elements are stored into L and the Rmat.
* This version keeps only maxnzkeep 
**************************************************************************/
void SecondDropUpdate(int maxnz, int maxnzkeep, double tol, int row,
		      int nlocal, int *perm, int *iperm, 
		      FactorMatType *ldu, ReduceMatType *rmat,
                      hypre_PilutSolverGlobals *globals )
{
  int i, j, k, nl;
  int max, nz, lrow, rrow;
  int last, first, itmp;
  double dtmp;


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
  /*****************************************************************
  * The entries between [1, last) are part of L
  * The entries [first, lastjr) are part of U
  ******************************************************************/


  /* Keep large maxnz elements of L */
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

  /* Allocate appropriate amount of memory for the reduced row */
  nl = MIN(lastjr-first+1, maxnzkeep);
  rmat->rmat_rnz[rrow] = nl;
  rmat->rmat_rcolind[rrow] = idx_malloc(nl, "SecondDropUpdate: rmat->rmat_rcolind[rrow]");
  rmat->rmat_rvalues[rrow] =  fp_malloc(nl, "SecondDropUpdate: rmat->rmat_rvalues[rrow]");

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

}

