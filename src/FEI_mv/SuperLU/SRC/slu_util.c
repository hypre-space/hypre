/*
 * -- SuperLU routine (version 3.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * October 15, 2003
 *
 */
/*
  Copyright (c) 1994 by Xerox Corporation.  All rights reserved.
 
  THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
  EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.
 
  Permission is hereby granted to use or copy this program for any
  purpose, provided the above notices are retained on all copies.
  Permission to modify the code and to distribute modified code is
  granted, provided the above notices are retained, and a notice that
  the code was modified is included with the above copyright notice.
*/
/*
  This file has been modified to be compatible with the HYPRE
  linear solver
*/

#include <math.h>
#include "slu_ddefs.h"

/* 
 * Global statistics variale
 */

void superlu_abort_and_exit(char* msg)
{
    fprintf(stderr, msg);
    exit (-1);
}

/*
 * Set the default values for the options argument.
 */
void set_default_options(superlu_options_t *options)
{
    options->Fact = DOFACT;
    options->Equil = YES;
    options->ColPerm = COLAMD;
    options->DiagPivotThresh = 1.0;
    options->Trans = NOTRANS;
    options->IterRefine = NOREFINE;
    options->SymmetricMode = NO;
    options->PivotGrowth = NO;
    options->ConditionNumber = NO;
    options->PrintStat = YES;
}

/*
 * Print the options setting.
 */
void print_options(superlu_options_t *options)
{
    printf(".. options:\n");
    printf("\tFact\t %8d\n", options->Fact);
    printf("\tEquil\t %8d\n", options->Equil);
    printf("\tColPerm\t %8d\n", options->ColPerm);
    printf("\tDiagPivotThresh %8.4f\n", options->DiagPivotThresh);
    printf("\tTrans\t %8d\n", options->Trans);
    printf("\tIterRefine\t%4d\n", options->IterRefine);
    printf("\tSymmetricMode\t%4d\n", options->SymmetricMode);
    printf("\tPivotGrowth\t%4d\n", options->PivotGrowth);
    printf("\tConditionNumber\t%4d\n", options->ConditionNumber);
    printf("..\n");
}

/* Deallocate the structure pointing to the actual storage of the matrix. */
void
Destroy_SuperMatrix_Store(SuperMatrix *A)
{
    SUPERLU_FREE ( A->Store );
}

void
Destroy_CompCol_Matrix(SuperMatrix *A)
{
    SUPERLU_FREE( ((NCformat *)A->Store)->rowind );
    SUPERLU_FREE( ((NCformat *)A->Store)->colptr );
    SUPERLU_FREE( ((NCformat *)A->Store)->nzval );
    SUPERLU_FREE( A->Store );
}

void
Destroy_CompRow_Matrix(SuperMatrix *A)
{
    SUPERLU_FREE( ((NRformat *)A->Store)->colind );
    SUPERLU_FREE( ((NRformat *)A->Store)->rowptr );
    SUPERLU_FREE( ((NRformat *)A->Store)->nzval );
    SUPERLU_FREE( A->Store );
}

void
Destroy_SuperNode_Matrix(SuperMatrix *A)
{
    SUPERLU_FREE ( ((SCformat *)A->Store)->rowind );
    SUPERLU_FREE ( ((SCformat *)A->Store)->rowind_colptr );
    SUPERLU_FREE ( ((SCformat *)A->Store)->nzval );
    SUPERLU_FREE ( ((SCformat *)A->Store)->nzval_colptr );
    SUPERLU_FREE ( ((SCformat *)A->Store)->col_to_sup );
    SUPERLU_FREE ( ((SCformat *)A->Store)->sup_to_col );
    SUPERLU_FREE ( A->Store );
}

/* A is of type Stype==NCP */
void
Destroy_CompCol_Permuted(SuperMatrix *A)
{
    SUPERLU_FREE ( ((NCPformat *)A->Store)->colbeg );
    SUPERLU_FREE ( ((NCPformat *)A->Store)->colend );
    SUPERLU_FREE ( A->Store );
}

/* A is of type Stype==DN */
void
Destroy_Dense_Matrix(SuperMatrix *A)
{
    DNformat* Astore = A->Store;
    SUPERLU_FREE (Astore->nzval);
    SUPERLU_FREE ( A->Store );
}

/*
 * Reset repfnz[] for the current column 
 */
void
resetrep_col (const int nseg, const int *segrep, int *repfnz)
{
    int i, irep;
    
    for (i = 0; i < nseg; i++) {
	irep = segrep[i];
	repfnz[irep] = EMPTY;
    }
}


/*
 * Count the total number of nonzeros in factors L and U,  and in the 
 * symmetrically reduced L. 
 */
void
countnz(const int n, int *xprune, int *nnzL, int *nnzU, GlobalLU_t *Glu)
{
    int          nsuper, fsupc, i, j;
    int          nnzL0, jlen, irep;
    int          *xsup, *xlsub;

    xsup   = Glu->xsup;
    xlsub  = Glu->xlsub;
    *nnzL  = 0;
    *nnzU  = (Glu->xusub)[n];
    nnzL0  = 0;
    nsuper = (Glu->supno)[n];

    if ( n <= 0 ) return;

    /* 
     * For each supernode
     */
    for (i = 0; i <= nsuper; i++) {
	fsupc = xsup[i];
	jlen = xlsub[fsupc+1] - xlsub[fsupc];

	for (j = fsupc; j < xsup[i+1]; j++) {
	    *nnzL += jlen;
	    *nnzU += j - fsupc + 1;
	    jlen--;
	}
	irep = xsup[i+1] - 1;
	nnzL0 += xprune[irep] - xlsub[irep];
    }
    
    /* printf("\tNo of nonzeros in symm-reduced L = %d\n", nnzL0);*/
}



/*
 * Fix up the data storage lsub for L-subscripts. It removes the subscript
 * sets for structural pruning,	and applies permuation to the remaining
 * subscripts.
 */
void
fixupL(const int n, const int *perm_r, GlobalLU_t *Glu)
{
    register int nsuper, fsupc, nextl, i, j, k, jstrt;
    int          *xsup, *lsub, *xlsub;

    if ( n <= 1 ) return;

    xsup   = Glu->xsup;
    lsub   = Glu->lsub;
    xlsub  = Glu->xlsub;
    nextl  = 0;
    nsuper = (Glu->supno)[n];
    
    /* 
     * For each supernode ...
     */
    for (i = 0; i <= nsuper; i++) {
	fsupc = xsup[i];
	jstrt = xlsub[fsupc];
	xlsub[fsupc] = nextl;
	for (j = jstrt; j < xlsub[fsupc+1]; j++) {
	    lsub[nextl] = perm_r[lsub[j]]; /* Now indexed into P*A */
	    nextl++;
  	}
	for (k = fsupc+1; k < xsup[i+1]; k++) 
	    	xlsub[k] = nextl;	/* Other columns in supernode i */

    }

    xlsub[n] = nextl;
}


/*
 * Diagnostic print of segment info after panel_dfs().
 */
void print_panel_seg(int n, int w, int jcol, int nseg, 
		     int *segrep, int *repfnz)
{
    int j, k;
    
    for (j = jcol; j < jcol+w; j++) {
	printf("\tcol %d:\n", j);
	for (k = 0; k < nseg; k++)
	    printf("\t\tseg %d, segrep %d, repfnz %d\n", k, 
			segrep[k], repfnz[(j-jcol)*n + segrep[k]]);
    }

}


void
StatInit(SuperLUStat_t *stat)
{
    register int i, w, panel_size, relax;

    panel_size = sp_ienv(1);
    relax = sp_ienv(2);
    w = SUPERLU_MAX(panel_size, relax);
    stat->panel_histo = intCalloc(w+1);
    stat->utime = (double *) SUPERLU_MALLOC(NPHASES * sizeof(double));
    if (!stat->utime) ABORT("SUPERLU_MALLOC fails for stat->utime");
    stat->ops = (flops_t *) SUPERLU_MALLOC(NPHASES * sizeof(flops_t));
    if (!stat->ops) ABORT("SUPERLU_MALLOC fails for stat->ops");
    for (i = 0; i < NPHASES; ++i) {
        stat->utime[i] = 0.;
        stat->ops[i] = 0.;
    }
}


void
StatPrint(SuperLUStat_t *stat)
{
    double         *utime;
    flops_t        *ops;

    utime = stat->utime;
    ops   = stat->ops;
    printf("Factor time  = %8.2f\n", utime[FACT]);
    if ( utime[FACT] != 0.0 )
      printf("Factor flops = %e\tMflops = %8.2f\n", ops[FACT],
	     ops[FACT]*1e-6/utime[FACT]);

    printf("Solve time   = %8.2f\n", utime[SOLVE]);
    if ( utime[SOLVE] != 0.0 )
      printf("Solve flops = %e\tMflops = %8.2f\n", ops[SOLVE],
	     ops[SOLVE]*1e-6/utime[SOLVE]);

}


void
StatFree(SuperLUStat_t *stat)
{
    SUPERLU_FREE(stat->panel_histo);
    SUPERLU_FREE(stat->utime);
    SUPERLU_FREE(stat->ops);
}


flops_t
LUFactFlops(SuperLUStat_t *stat)
{
    return (stat->ops[FACT]);
}

flops_t
LUSolveFlops(SuperLUStat_t *stat)
{
    return (stat->ops[SOLVE]);
}





/* 
 * Fills an integer array with a given value.
 */
void ifill(int *a, int alen, int ival)
{
    register int i;
    for (i = 0; i < alen; i++) a[i] = ival;
}



/* 
 * Get the statistics of the supernodes 
 */
#define NBUCKS 10
static 	int	max_sup_size;

void super_stats(int nsuper, int *xsup)
{
    register int nsup1 = 0;
    int          i, isize, whichb, bl, bh;
    int          bucket[NBUCKS];

    max_sup_size = 0;

    for (i = 0; i <= nsuper; i++) {
	isize = xsup[i+1] - xsup[i];
	if ( isize == 1 ) nsup1++;
	if ( max_sup_size < isize ) max_sup_size = isize;	
    }

    printf("    Supernode statistics:\n\tno of super = %d\n", nsuper+1);
    printf("\tmax supernode size = %d\n", max_sup_size);
    printf("\tno of size 1 supernodes = %d\n", nsup1);

    /* Histogram of the supernode sizes */
    ifill (bucket, NBUCKS, 0);

    for (i = 0; i <= nsuper; i++) {
        isize = xsup[i+1] - xsup[i];
        whichb = (float) isize / max_sup_size * NBUCKS;
        if (whichb >= NBUCKS) whichb = NBUCKS - 1;
        bucket[whichb]++;
    }
    
    printf("\tHistogram of supernode sizes:\n");
    for (i = 0; i < NBUCKS; i++) {
        bl = (float) i * max_sup_size / NBUCKS;
        bh = (float) (i+1) * max_sup_size / NBUCKS;
        printf("\tsnode: %d-%d\t\t%d\n", bl+1, bh, bucket[i]);
    }

}


float SpaSize(int n, int np, float sum_npw)
{
    return (sum_npw*8 + np*8 + n*4)/1024.;
}

float DenseSize(int n, float sum_nw)
{
    return (sum_nw*8 + n*8)/1024.;;
}



/*
 * Check whether repfnz[] == EMPTY after reset.
 */
void check_repfnz(int n, int w, int jcol, int *repfnz)
{
    int jj, k;

    for (jj = jcol; jj < jcol+w; jj++) 
	for (k = 0; k < n; k++)
	    if ( repfnz[(jj-jcol)*n + k] != EMPTY ) {
		fprintf(stderr, "col %d, repfnz_col[%d] = %d\n", jj,
			k, repfnz[(jj-jcol)*n + k]);
		ABORT("check_repfnz");
	    }
}


/* Print a summary of the testing results. */
void
PrintSumm(char *type, int nfail, int nrun, int nerrs)
{
    if ( nfail > 0 )
	printf("%3s driver: %d out of %d tests failed to pass the threshold\n",
	       type, nfail, nrun);
    else
	printf("All tests for %3s driver passed the threshold (%6d tests run)\n", type, nrun);

    if ( nerrs > 0 )
	printf("%6d error messages recorded\n", nerrs);
}


int print_int_vec(char *what, int n, int *vec)
{
    int i;
    printf("%s\n", what);
    for (i = 0; i < n; ++i) printf("%d\t%d\n", i, vec[i]);
    return 0;
}

int superlu_lsame(char *ca, char *cb)
{
/*  -- LAPACK auxiliary routine (version 2.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994

    Purpose
    =======

    LSAME returns .TRUE. if CA is the same letter as CB regardless of case.

    Arguments
    =========

    CA      (input) CHARACTER*1
    CB      (input) CHARACTER*1
            CA and CB specify the single characters to be compared.

   =====================================================================
*/

    /* System generated locals */
    int ret_val;

    /* Local variables */
    int inta, intb, zcode;

    ret_val = *(unsigned char *)ca == *(unsigned char *)cb;
    if (ret_val) {
        return ret_val;
    }

    /* Now test for equivalence if both characters are alphabetic. */

    zcode = 'Z';

    /* Use 'Z' rather than 'A' so that ASCII can be detected on Prime
       machines, on which ICHAR returns a value with bit 8 set.
       ICHAR('A') on Prime machines returns 193 which is the same as
       ICHAR('A') on an EBCDIC machine. */

    inta = *(unsigned char *)ca;
    intb = *(unsigned char *)cb;

    if (zcode == 90 || zcode == 122) {
        /* ASCII is assumed - ZCODE is the ASCII code of either lower or
          upper case 'Z'. */
        if (inta >= 97 && inta <= 122) inta += -32;
        if (intb >= 97 && intb <= 122) intb += -32;

    } else if (zcode == 233 || zcode == 169) {
        /* EBCDIC is assumed - ZCODE is the EBCDIC code of either lower or
          upper case 'Z'. */
        if ((inta >= 129 && inta <= 137) || (inta >= 145 && inta <= 153) ||
                (inta >= 162 && inta <= 169))
            inta += 64;
        if ((intb >= 129 && intb <= 137) || (intb >= 145 && intb <= 153) ||
                (intb >= 162 && intb <= 169))
            intb += 64;
    } else if (zcode == 218 || zcode == 250) {
        /* ASCII is assumed, on Prime machines - ZCODE is the ASCII code
          plus 128 of either lower or upper case 'Z'. */
        if (inta >= 225 && inta <= 250) inta += -32;
        if (intb >= 225 && intb <= 250) intb += -32;
    }
    ret_val = inta == intb;
    return ret_val;

} /* superlu_lsame */

/* Subroutine */ int superlu_xerbla(char *srname, int *info)
{
/*  -- LAPACK auxiliary routine (version 2.0) --
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,
       Courant Institute, Argonne National Lab, and Rice University
       September 30, 1994


    Purpose
    =======

    XERBLA  is an error handler for the LAPACK routines.
    It is called by an LAPACK routine if an input parameter has an
    invalid value.  A message is printed and execution stops.

    Installers may consider modifying the STOP statement in order to
    call system-specific exception-handling facilities.

    Arguments
    =========

    SRNAME  (input) CHARACTER*6
            The name of the routine which called XERBLA.

    INFO    (input) INT
            The position of the invalid parameter in the parameter list

            of the calling routine.

   =====================================================================
*/

    printf("** On entry to %6s, parameter number %2d had an illegal value\n",
                srname, *info);

/*     End of XERBLA */

    return 0;
} /* superlu_xerbla */


/*
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
 *
 */
/*
 * File name:		dmyblas2.c
 * Purpose:
 *     Level 2 BLAS operations: solves and matvec, written in C.
 * Note:
 *     This is only used when the system lacks an efficient BLAS library.
 */

/*
 * Solves a dense UNIT lower triangular system. The unit lower 
 * triangular matrix is stored in a 2D array M(1:nrow,1:ncol). 
 * The solution will be returned in the rhs vector.
 */
void sludlsolve ( int ldm, int ncol, double *M, double *rhs )
{
    int k;
    double x0, x1, x2, x3, x4, x5, x6, x7;
    double *M0;
    register double *Mki0, *Mki1, *Mki2, *Mki3, *Mki4, *Mki5, *Mki6, *Mki7;
    register int firstcol = 0;

    M0 = &M[0];

    while ( firstcol < ncol - 7 ) { /* Do 8 columns */
      Mki0 = M0 + 1;
      Mki1 = Mki0 + ldm + 1;
      Mki2 = Mki1 + ldm + 1;
      Mki3 = Mki2 + ldm + 1;
      Mki4 = Mki3 + ldm + 1;
      Mki5 = Mki4 + ldm + 1;
      Mki6 = Mki5 + ldm + 1;
      Mki7 = Mki6 + ldm + 1;

      x0 = rhs[firstcol];
      x1 = rhs[firstcol+1] - x0 * *Mki0++;
      x2 = rhs[firstcol+2] - x0 * *Mki0++ - x1 * *Mki1++;
      x3 = rhs[firstcol+3] - x0 * *Mki0++ - x1 * *Mki1++ - x2 * *Mki2++;
      x4 = rhs[firstcol+4] - x0 * *Mki0++ - x1 * *Mki1++ - x2 * *Mki2++
	                   - x3 * *Mki3++;
      x5 = rhs[firstcol+5] - x0 * *Mki0++ - x1 * *Mki1++ - x2 * *Mki2++
	                   - x3 * *Mki3++ - x4 * *Mki4++;
      x6 = rhs[firstcol+6] - x0 * *Mki0++ - x1 * *Mki1++ - x2 * *Mki2++
	                   - x3 * *Mki3++ - x4 * *Mki4++ - x5 * *Mki5++;
      x7 = rhs[firstcol+7] - x0 * *Mki0++ - x1 * *Mki1++ - x2 * *Mki2++
	                   - x3 * *Mki3++ - x4 * *Mki4++ - x5 * *Mki5++
			   - x6 * *Mki6++;

      rhs[++firstcol] = x1;
      rhs[++firstcol] = x2;
      rhs[++firstcol] = x3;
      rhs[++firstcol] = x4;
      rhs[++firstcol] = x5;
      rhs[++firstcol] = x6;
      rhs[++firstcol] = x7;
      ++firstcol;
    
      for (k = firstcol; k < ncol; k++)
	rhs[k] = rhs[k] - x0 * *Mki0++ - x1 * *Mki1++
	                - x2 * *Mki2++ - x3 * *Mki3++
                        - x4 * *Mki4++ - x5 * *Mki5++
			- x6 * *Mki6++ - x7 * *Mki7++;
 
      M0 += 8 * ldm + 8;
    }

    while ( firstcol < ncol - 3 ) { /* Do 4 columns */
      Mki0 = M0 + 1;
      Mki1 = Mki0 + ldm + 1;
      Mki2 = Mki1 + ldm + 1;
      Mki3 = Mki2 + ldm + 1;

      x0 = rhs[firstcol];
      x1 = rhs[firstcol+1] - x0 * *Mki0++;
      x2 = rhs[firstcol+2] - x0 * *Mki0++ - x1 * *Mki1++;
      x3 = rhs[firstcol+3] - x0 * *Mki0++ - x1 * *Mki1++ - x2 * *Mki2++;

      rhs[++firstcol] = x1;
      rhs[++firstcol] = x2;
      rhs[++firstcol] = x3;
      ++firstcol;
    
      for (k = firstcol; k < ncol; k++)
	rhs[k] = rhs[k] - x0 * *Mki0++ - x1 * *Mki1++
	                - x2 * *Mki2++ - x3 * *Mki3++;
 
      M0 += 4 * ldm + 4;
    }

    if ( firstcol < ncol - 1 ) { /* Do 2 columns */
      Mki0 = M0 + 1;
      Mki1 = Mki0 + ldm + 1;

      x0 = rhs[firstcol];
      x1 = rhs[firstcol+1] - x0 * *Mki0++;

      rhs[++firstcol] = x1;
      ++firstcol;
    
      for (k = firstcol; k < ncol; k++)
	rhs[k] = rhs[k] - x0 * *Mki0++ - x1 * *Mki1++;
 
    }
    
}

/*
 * Solves a dense upper triangular system. The upper triangular matrix is
 * stored in a 2-dim array M(1:ldm,1:ncol). The solution will be returned
 * in the rhs vector.
 */
void
sludusolve ( ldm, ncol, M, rhs )
int ldm;	/* in */
int ncol;	/* in */
double *M;	/* in */
double *rhs;	/* modified */
{
    double xj;
    int jcol, j, irow;

    jcol = ncol - 1;

    for (j = 0; j < ncol; j++) {

	xj = rhs[jcol] / M[jcol + jcol*ldm]; 		/* M(jcol, jcol) */
	rhs[jcol] = xj;
	
	for (irow = 0; irow < jcol; irow++)
	    rhs[irow] -= xj * M[irow + jcol*ldm];	/* M(irow, jcol) */

	jcol--;

    }
}


/*
 * Performs a dense matrix-vector multiply: Mxvec = Mxvec + M * vec.
 * The input matrix is M(1:nrow,1:ncol); The product is returned in Mxvec[].
 */
void sludmatvec ( ldm, nrow, ncol, M, vec, Mxvec )

int ldm;	/* in -- leading dimension of M */
int nrow;	/* in */ 
int ncol;	/* in */
double *M;	/* in */
double *vec;	/* in */
double *Mxvec;	/* in/out */

{
    double vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi7;
    double *M0;
    register double *Mki0, *Mki1, *Mki2, *Mki3, *Mki4, *Mki5, *Mki6, *Mki7;
    register int firstcol = 0;
    int k;

    M0 = &M[0];
    while ( firstcol < ncol - 7 ) {	/* Do 8 columns */

	Mki0 = M0;
	Mki1 = Mki0 + ldm;
        Mki2 = Mki1 + ldm;
        Mki3 = Mki2 + ldm;
	Mki4 = Mki3 + ldm;
	Mki5 = Mki4 + ldm;
	Mki6 = Mki5 + ldm;
	Mki7 = Mki6 + ldm;

	vi0 = vec[firstcol++];
	vi1 = vec[firstcol++];
	vi2 = vec[firstcol++];
	vi3 = vec[firstcol++];	
	vi4 = vec[firstcol++];
	vi5 = vec[firstcol++];
	vi6 = vec[firstcol++];
	vi7 = vec[firstcol++];	

	for (k = 0; k < nrow; k++) 
	    Mxvec[k] += vi0 * *Mki0++ + vi1 * *Mki1++
		      + vi2 * *Mki2++ + vi3 * *Mki3++ 
		      + vi4 * *Mki4++ + vi5 * *Mki5++
		      + vi6 * *Mki6++ + vi7 * *Mki7++;

	M0 += 8 * ldm;
    }

    while ( firstcol < ncol - 3 ) {	/* Do 4 columns */

	Mki0 = M0;
	Mki1 = Mki0 + ldm;
	Mki2 = Mki1 + ldm;
	Mki3 = Mki2 + ldm;

	vi0 = vec[firstcol++];
	vi1 = vec[firstcol++];
	vi2 = vec[firstcol++];
	vi3 = vec[firstcol++];	
	for (k = 0; k < nrow; k++) 
	    Mxvec[k] += vi0 * *Mki0++ + vi1 * *Mki1++
		      + vi2 * *Mki2++ + vi3 * *Mki3++ ;

	M0 += 4 * ldm;
    }

    while ( firstcol < ncol ) {		/* Do 1 column */

 	Mki0 = M0;
	vi0 = vec[firstcol++];
	for (k = 0; k < nrow; k++)
	    Mxvec[k] += vi0 * *Mki0++;

	M0 += ldm;
    }
	
}

