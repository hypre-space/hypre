/*
 * -- SuperLU routine (version 2.0) --
 * Univ. of California Berkeley, Xerox Palo Alto Research Center,
 * and Lawrence Berkeley National Lab.
 * November 15, 1997
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

#include <stdio.h>
#include <math.h>
#include <HYPRE_config.h>
#include "dsp_defs.h"
#include "util.h"

/* 
 * Global statistics variale
 */
SuperLUStat_t SuperLUStat;

void superlu_abort_and_exit(char* msg)
{
    fprintf(stderr, msg);
    exit (-1);
}

void *superlu_malloc(int size)
{
    void *buf;
    buf = (void *) malloc(size);
    return (buf);
}

void superlu_free(void *addr)
{
    free (addr);
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
StatInit(int panel_size, int relax)
{
    register int i, w;
    w = MAX(panel_size, relax);
    SuperLUStat.panel_histo = intCalloc(w+1);
    SuperLUStat.utime = (double *) SUPERLU_MALLOC(NPHASES * sizeof(double));
    if (!SuperLUStat.utime) ABORT("SUPERLU_MALLOC fails for SuperLUStat.utime");
    SuperLUStat.ops = (flops_t *) SUPERLU_MALLOC(NPHASES * sizeof(flops_t));
    if (!SuperLUStat.ops) ABORT("SUPERLU_MALLOC fails for SuperLUStat.ops");
    for (i = 0; i < NPHASES; ++i) {
        SuperLUStat.utime[i] = 0.;
        SuperLUStat.ops[i] = 0.;
    }
}


void
PrintStat(SuperLUStat_t *SuperLUStat)
{
    double         *utime;
    flops_t        *ops;

    utime = SuperLUStat->utime;
    ops   = SuperLUStat->ops;
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
StatFree()
{
    SUPERLU_FREE(SuperLUStat.panel_histo);
    SUPERLU_FREE(SuperLUStat.utime);
    SUPERLU_FREE(SuperLUStat.ops);
}


flops_t
LUFactFlops()
{
    return (SuperLUStat.ops[FACT]);
}

flops_t
LUSolveFlops()
{
    return (SuperLUStat.ops[SOLVE]);
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

int PrintInt10(char *name, int len, int *x)
{
    register int i;
    
    printf("%s:", name);
    for (i = 0; i < len; ++i) {
	if ( i % 10 == 0 ) printf("\n[%4d-%4d]", i, i+9);
	printf("%6d", x[i]);
    }
    printf("\n");
    return 0;
}


#ifndef HYPRE_USING_HYPRE_BLAS
int lsame_(char *ca, char *cb)
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
	if (inta >= 129 && inta <= 137 || inta >= 145 && inta <= 153 || inta 
		>= 162 && inta <= 169)
	    inta += 64;
	if (intb >= 129 && intb <= 137 || intb >= 145 && intb <= 153 || intb 
		>= 162 && intb <= 169)
	    intb += 64;
    } else if (zcode == 218 || zcode == 250) {
	/* ASCII is assumed, on Prime machines - ZCODE is the ASCII code   
          plus 128 of either lower or upper case 'Z'. */
	if (inta >= 225 && inta <= 250) inta += -32;
	if (intb >= 225 && intb <= 250) intb += -32;
    }
    ret_val = inta == intb;
    return ret_val;
    
} /* lsame_ */

/* Subroutine */ int xerbla_(char *srname, int *info)
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
} /* xerbla_ */

#define TRUE_ (1)
#define FALSE_ (0)
#ifndef abs
#define abs(x) ((x) >= 0 ? (x) : -(x))
#endif
#ifndef min
#define min(a,b) ((a) <= (b) ? (a) : (b))
#endif
#ifndef max
#define max(a,b) ((a) >= (b) ? (a) : (b))
#endif

double dlamch_(char *cmach)
{
/*  -- LAPACK auxiliary routine (version 2.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       October 31, 1992   

    Purpose   
    =======   

    DLAMCH determines double precision machine parameters.   

    Arguments   
    =========   

    CMACH   (input) CHARACTER*1   
            Specifies the value to be returned by DLAMCH:   
            = 'E' or 'e',   DLAMCH := eps   
            = 'S' or 's ,   DLAMCH := sfmin   
            = 'B' or 'b',   DLAMCH := base   
            = 'P' or 'p',   DLAMCH := eps*base   
            = 'N' or 'n',   DLAMCH := t   
            = 'R' or 'r',   DLAMCH := rnd   
            = 'M' or 'm',   DLAMCH := emin   
            = 'U' or 'u',   DLAMCH := rmin   
            = 'L' or 'l',   DLAMCH := emax   
            = 'O' or 'o',   DLAMCH := rmax   

            where   

            eps   = relative machine precision   
            sfmin = safe minimum, such that 1/sfmin does not overflow   
            base  = base of the machine   
            prec  = eps*base   
            t     = number of (base) digits in the mantissa   
            rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise   
            emin  = minimum exponent before (gradual) underflow   
            rmin  = underflow threshold - base**(emin-1)   
            emax  = largest exponent before overflow   
            rmax  = overflow threshold  - (base**emax)*(1-eps)   

   ===================================================================== 
*/

    static int first = TRUE_;

    /* System generated locals */
    int i__1;
    double ret_val;
    /* Builtin functions */
    double pow_di(double *, int *);
    /* Local variables */
    static double base;
    static int beta;
    static double emin, prec, emax;
    static int imin, imax;
    static int lrnd;
    static double rmin, rmax, t, rmach;
/*    extern int lsame_(char *, char *);*/
    static double small, sfmin;
    extern /* Subroutine */ int dlamc2_(int *, int *, int *, 
	    double *, int *, double *, int *, double *);
    static int it;
    static double rnd, eps;

    if (first) {
	first = FALSE_;
	dlamc2_(&beta, &it, &lrnd, &eps, &imin, &rmin, &imax, &rmax);
	base = (double) beta;
	t = (double) it;
	if (lrnd) {
	    rnd = 1.;
	    i__1 = 1 - it;
	    eps = pow_di(&base, &i__1) / 2;
	} else {
	    rnd = 0.;
	    i__1 = 1 - it;
	    eps = pow_di(&base, &i__1);
	}
	prec = eps * base;
	emin = (double) imin;
	emax = (double) imax;
	sfmin = rmin;
	small = 1. / rmax;
	if (small >= sfmin) {

	/* Use SMALL plus a bit, to avoid the possibility of rounding   
             causing overflow when computing  1/sfmin. */
	    sfmin = small * (eps + 1.);
	}
    }

    if (lsame_(cmach, "E")) {
	rmach = eps;
    } else if (lsame_(cmach, "S")) {
	rmach = sfmin;
    } else if (lsame_(cmach, "B")) {
	rmach = base;
    } else if (lsame_(cmach, "P")) {
	rmach = prec;
    } else if (lsame_(cmach, "N")) {
	rmach = t;
    } else if (lsame_(cmach, "R")) {
	rmach = rnd;
    } else if (lsame_(cmach, "M")) {
	rmach = emin;
    } else if (lsame_(cmach, "U")) {
	rmach = rmin;
    } else if (lsame_(cmach, "L")) {
	rmach = emax;
    } else if (lsame_(cmach, "O")) {
	rmach = rmax;
    }

    ret_val = rmach;
    return ret_val;

/*     End of DLAMCH */

} /* dlamch_ */


/* Subroutine */ int dlamc1_(int *beta, int *t, int *rnd, int 
	*ieee1)
{
/*  -- LAPACK auxiliary routine (version 2.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       October 31, 1992   


    Purpose   
    =======   

    DLAMC1 determines the machine parameters given by BETA, T, RND, and   
    IEEE1.   

    Arguments   
    =========   

    BETA    (output) INT   
            The base of the machine.   

    T       (output) INT   
            The number of ( BETA ) digits in the mantissa.   

    RND     (output) INT   
            Specifies whether proper rounding  ( RND = .TRUE. )  or   
            chopping  ( RND = .FALSE. )  occurs in addition. This may not 
  
            be a reliable guide to the way in which the machine performs 
  
            its arithmetic.   

    IEEE1   (output) INT   
            Specifies whether rounding appears to be done in the IEEE   
            'round to nearest' style.   

    Further Details   
    ===============   

    The routine is based on the routine  ENVRON  by Malcolm and   
    incorporates suggestions by Gentleman and Marovich. See   

       Malcolm M. A. (1972) Algorithms to reveal properties of   
          floating-point arithmetic. Comms. of the ACM, 15, 949-951.   

       Gentleman W. M. and Marovich S. B. (1974) More on algorithms   
          that reveal properties of floating point arithmetic units.   
          Comms. of the ACM, 17, 276-277.   

   ===================================================================== 
*/
    /* Initialized data */
    static int first = TRUE_;
    /* System generated locals */
    double d__1, d__2;
    /* Local variables */
    static int lrnd;
    static double a, b, c, f;
    static int lbeta;
    static double savec;
    extern double dlamc3_(double *, double *);
    static int lieee1;
    static double t1, t2;
    static int lt;
    static double one, qtr;

    if (first) {
	first = FALSE_;
	one = 1.;

/*        LBETA,  LIEEE1,  LT and  LRND  are the  local values  of  BE
TA,   
          IEEE1, T and RND.   

          Throughout this routine  we use the function  DLAMC3  to ens
ure   
          that relevant values are  stored and not held in registers, 
 or   
          are not affected by optimizers.   

          Compute  a = 2.0**m  with the  smallest positive integer m s
uch   
          that   

             fl( a + 1.0 ) = a. */

	a = 1.;
	c = 1.;

/* +       WHILE( C.EQ.ONE )LOOP */
L10:
	if (c == one) {
	    a *= 2;
	    c = dlamc3_(&a, &one);
	    d__1 = -a;
	    c = dlamc3_(&c, &d__1);
	    goto L10;
	}
/* +       END WHILE   

          Now compute  b = 2.0**m  with the smallest positive integer 
m   
          such that   

             fl( a + b ) .gt. a. */

	b = 1.;
	c = dlamc3_(&a, &b);

/* +       WHILE( C.EQ.A )LOOP */
L20:
	if (c == a) {
	    b *= 2;
	    c = dlamc3_(&a, &b);
	    goto L20;
	}
/* +       END WHILE   

          Now compute the base.  a and c  are neighbouring floating po
int   
          numbers  in the  interval  ( beta**t, beta**( t + 1 ) )  and
 so   
          their difference is beta. Adding 0.25 to c is to ensure that
 it   
          is truncated to beta and not ( beta - 1 ). */

	qtr = one / 4;
	savec = c;
	d__1 = -a;
	c = dlamc3_(&c, &d__1);
	lbeta = (int) (c + qtr);

/*        Now determine whether rounding or chopping occurs,  by addin
g a   
          bit  less  than  beta/2  and a  bit  more  than  beta/2  to 
 a. */

	b = (double) lbeta;
	d__1 = b / 2;
	d__2 = -b / 100;
	f = dlamc3_(&d__1, &d__2);
	c = dlamc3_(&f, &a);
	if (c == a) {
	    lrnd = TRUE_;
	} else {
	    lrnd = FALSE_;
	}
	d__1 = b / 2;
	d__2 = b / 100;
	f = dlamc3_(&d__1, &d__2);
	c = dlamc3_(&f, &a);
	if (lrnd && c == a) {
	    lrnd = FALSE_;
	}

/*        Try and decide whether rounding is done in the  IEEE  'round
 to   
          nearest' style. B/2 is half a unit in the last place of the 
two   
          numbers A and SAVEC. Furthermore, A is even, i.e. has last  
bit   
          zero, and SAVEC is odd. Thus adding B/2 to A should not  cha
nge   
          A, but adding B/2 to SAVEC should change SAVEC. */

	d__1 = b / 2;
	t1 = dlamc3_(&d__1, &a);
	d__1 = b / 2;
	t2 = dlamc3_(&d__1, &savec);
	lieee1 = t1 == a && t2 > savec && lrnd;

/*        Now find  the  mantissa, t.  It should  be the  integer part
 of   
          log to the base beta of a,  however it is safer to determine
  t   
          by powering.  So we find t as the smallest positive integer 
for   
          which   

             fl( beta**t + 1.0 ) = 1.0. */

	lt = 0;
	a = 1.;
	c = 1.;

/* +       WHILE( C.EQ.ONE )LOOP */
L30:
	if (c == one) {
	    ++lt;
	    a *= lbeta;
	    c = dlamc3_(&a, &one);
	    d__1 = -a;
	    c = dlamc3_(&c, &d__1);
	    goto L30;
	}
/* +       END WHILE */

    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *ieee1 = lieee1;
    return 0;

/*     End of DLAMC1 */

} /* dlamc1_ */


/* Subroutine */ int dlamc2_(int *beta, int *t, int *rnd, 
	double *eps, int *emin, double *rmin, int *emax, 
	double *rmax)
{
/*  -- LAPACK auxiliary routine (version 2.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       October 31, 1992   


    Purpose   
    =======   

    DLAMC2 determines the machine parameters specified in its argument   
    list.   

    Arguments   
    =========   

    BETA    (output) INT   
            The base of the machine.   

    T       (output) INT   
            The number of ( BETA ) digits in the mantissa.   

    RND     (output) INT   
            Specifies whether proper rounding  ( RND = .TRUE. )  or   
            chopping  ( RND = .FALSE. )  occurs in addition. This may not 
  
            be a reliable guide to the way in which the machine performs 
  
            its arithmetic.   

    EPS     (output) DOUBLE PRECISION   
            The smallest positive number such that   

               fl( 1.0 - EPS ) .LT. 1.0,   

            where fl denotes the computed value.   

    EMIN    (output) INT   
            The minimum exponent before (gradual) underflow occurs.   

    RMIN    (output) DOUBLE PRECISION   
            The smallest normalized number for the machine, given by   
            BASE**( EMIN - 1 ), where  BASE  is the floating point value 
  
            of BETA.   

    EMAX    (output) INT   
            The maximum exponent before overflow occurs.   

    RMAX    (output) DOUBLE PRECISION   
            The largest positive number for the machine, given by   
            BASE**EMAX * ( 1 - EPS ), where  BASE  is the floating point 
  
            value of BETA.   

    Further Details   
    ===============   

    The computation of  EPS  is based on a routine PARANOIA by   
    W. Kahan of the University of California at Berkeley.   

   ===================================================================== 
*/
    /* Table of constant values */
    
    /* Initialized data */
    static int first = TRUE_;
    static int iwarn = FALSE_;
    /* System generated locals */
    int i__1;
    double d__1, d__2, d__3, d__4, d__5;
    /* Builtin functions */
    double pow_di(double *, int *);
    /* Local variables */
    static int ieee;
    static double half;
    static int lrnd;
    static double leps, zero, a, b, c;
    static int i, lbeta;
    static double rbase;
    static int lemin, lemax, gnmin;
    static double small;
    static int gpmin;
    static double third, lrmin, lrmax, sixth;
    extern /* Subroutine */ int dlamc1_(int *, int *, int *, 
	    int *);
    extern double dlamc3_(double *, double *);
    static int lieee1;
    extern /* Subroutine */ int dlamc4_(int *, double *, int *), 
	    dlamc5_(int *, int *, int *, int *, int *, 
	    double *);
    static int lt, ngnmin, ngpmin;
    static double one, two;

    if (first) {
	first = FALSE_;
	zero = 0.;
	one = 1.;
	two = 2.;

/*        LBETA, LT, LRND, LEPS, LEMIN and LRMIN  are the local values
 of   
          BETA, T, RND, EPS, EMIN and RMIN.   

          Throughout this routine  we use the function  DLAMC3  to ens
ure   
          that relevant values are stored  and not held in registers, 
 or   
          are not affected by optimizers.   

          DLAMC1 returns the parameters  LBETA, LT, LRND and LIEEE1. 
*/

	dlamc1_(&lbeta, &lt, &lrnd, &lieee1);

/*        Start to find EPS. */

	b = (double) lbeta;
	i__1 = -lt;
	a = pow_di(&b, &i__1);
	leps = a;

/*        Try some tricks to see whether or not this is the correct  E
PS. */

	b = two / 3;
	half = one / 2;
	d__1 = -half;
	sixth = dlamc3_(&b, &d__1);
	third = dlamc3_(&sixth, &sixth);
	d__1 = -half;
	b = dlamc3_(&third, &d__1);
	b = dlamc3_(&b, &sixth);
	b = abs(b);
	if (b < leps) {
	    b = leps;
	}

	leps = 1.;

/* +       WHILE( ( LEPS.GT.B ).AND.( B.GT.ZERO ) )LOOP */
L10:
	if (leps > b && b > zero) {
	    leps = b;
	    d__1 = half * leps;
/* Computing 5th power */
	    d__3 = two, d__4 = d__3, d__3 *= d__3;
/* Computing 2nd power */
	    d__5 = leps;
	    d__2 = d__4 * (d__3 * d__3) * (d__5 * d__5);
	    c = dlamc3_(&d__1, &d__2);
	    d__1 = -c;
	    c = dlamc3_(&half, &d__1);
	    b = dlamc3_(&half, &c);
	    d__1 = -b;
	    c = dlamc3_(&half, &d__1);
	    b = dlamc3_(&half, &c);
	    goto L10;
	}
/* +       END WHILE */

	if (a < leps) {
	    leps = a;
	}

/*        Computation of EPS complete.   

          Now find  EMIN.  Let A = + or - 1, and + or - (1 + BASE**(-3
)).   
          Keep dividing  A by BETA until (gradual) underflow occurs. T
his   
          is detected when we cannot recover the previous A. */

	rbase = one / lbeta;
	small = one;
	for (i = 1; i <= 3; ++i) {
	    d__1 = small * rbase;
	    small = dlamc3_(&d__1, &zero);
/* L20: */
	}
	a = dlamc3_(&one, &small);
	dlamc4_(&ngpmin, &one, &lbeta);
	d__1 = -one;
	dlamc4_(&ngnmin, &d__1, &lbeta);
	dlamc4_(&gpmin, &a, &lbeta);
	d__1 = -a;
	dlamc4_(&gnmin, &d__1, &lbeta);
	ieee = FALSE_;

	if (ngpmin == ngnmin && gpmin == gnmin) {
	    if (ngpmin == gpmin) {
		lemin = ngpmin;
/*            ( Non twos-complement machines, no gradual under
flow;   
                e.g.,  VAX ) */
	    } else if (gpmin - ngpmin == 3) {
		lemin = ngpmin - 1 + lt;
		ieee = TRUE_;
/*            ( Non twos-complement machines, with gradual und
erflow;   
                e.g., IEEE standard followers ) */
	    } else {
		lemin = min(ngpmin,gpmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if (ngpmin == gpmin && ngnmin == gnmin) {
	    if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1) {
		lemin = max(ngpmin,ngnmin);
/*            ( Twos-complement machines, no gradual underflow
;   
                e.g., CYBER 205 ) */
	    } else {
		lemin = min(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else if ((i__1 = ngpmin - ngnmin, abs(i__1)) == 1 && gpmin == gnmin)
		 {
	    if (gpmin - min(ngpmin,ngnmin) == 3) {
		lemin = max(ngpmin,ngnmin) - 1 + lt;
/*            ( Twos-complement machines with gradual underflo
w;   
                no known machine ) */
	    } else {
		lemin = min(ngpmin,ngnmin);
/*            ( A guess; no known machine ) */
		iwarn = TRUE_;
	    }

	} else {
/* Computing MIN */
	    i__1 = min(ngpmin,ngnmin), i__1 = min(i__1,gpmin);
	    lemin = min(i__1,gnmin);
/*         ( A guess; no known machine ) */
	    iwarn = TRUE_;
	}
/* **   
   Comment out this if block if EMIN is ok */
	if (iwarn) {
	    first = TRUE_;
	    printf("\n\n WARNING. The value EMIN may be incorrect:- ");
	    printf("EMIN = %8i\n",lemin);
	    printf("If, after inspection, the value EMIN looks acceptable");
            printf("please comment out \n the IF block as marked within the"); 
            printf("code of routine DLAMC2, \n otherwise supply EMIN"); 
            printf("explicitly.\n");
	}
/* **   

          Assume IEEE arithmetic if we found denormalised  numbers abo
ve,   
          or if arithmetic seems to round in the  IEEE style,  determi
ned   
          in routine DLAMC1. A true IEEE machine should have both  thi
ngs   
          true; however, faulty machines may have one or the other. */

	ieee = ieee || lieee1;

/*        Compute  RMIN by successive division by  BETA. We could comp
ute   
          RMIN as BASE**( EMIN - 1 ),  but some machines underflow dur
ing   
          this computation. */

	lrmin = 1.;
	i__1 = 1 - lemin;
	for (i = 1; i <= 1-lemin; ++i) {
	    d__1 = lrmin * rbase;
	    lrmin = dlamc3_(&d__1, &zero);
/* L30: */
	}

/*        Finally, call DLAMC5 to compute EMAX and RMAX. */

	dlamc5_(&lbeta, &lt, &lemin, &ieee, &lemax, &lrmax);
    }

    *beta = lbeta;
    *t = lt;
    *rnd = lrnd;
    *eps = leps;
    *emin = lemin;
    *rmin = lrmin;
    *emax = lemax;
    *rmax = lrmax;

    return 0;


/*     End of DLAMC2 */

} /* dlamc2_ */


double dlamc3_(double *a, double *b)
{
/*  -- LAPACK auxiliary routine (version 2.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       October 31, 1992   


    Purpose   
    =======   

    DLAMC3  is intended to force  A  and  B  to be stored prior to doing 
  
    the addition of  A  and  B ,  for use in situations where optimizers 
  
    might hold one of these in a register.   

    Arguments   
    =========   

    A, B    (input) DOUBLE PRECISION   
            The values A and B.   

   ===================================================================== 
*/
/* >>Start of File<<   
       System generated locals */
    double ret_val;

    ret_val = *a + *b;

    return ret_val;

/*     End of DLAMC3 */

} /* dlamc3_ */


/* Subroutine */ int dlamc4_(int *emin, double *start, int *base)
{
/*  -- LAPACK auxiliary routine (version 2.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       October 31, 1992   


    Purpose   
    =======   

    DLAMC4 is a service routine for DLAMC2.   

    Arguments   
    =========   

    EMIN    (output) EMIN   
            The minimum exponent before (gradual) underflow, computed by 
  
            setting A = START and dividing by BASE until the previous A   
            can not be recovered.   

    START   (input) DOUBLE PRECISION   
            The starting point for determining EMIN.   

    BASE    (input) INT   
            The base of the machine.   

   ===================================================================== 
*/
    /* System generated locals */
    double d__1;
    /* Local variables */
    static double zero, a;
    static int i;
    static double rbase, b1, b2, c1, c2, d1, d2;
    extern double dlamc3_(double *, double *);
    static double one;

    a = *start;
    one = 1.;
    rbase = one / *base;
    zero = 0.;
    *emin = 1;
    d__1 = a * rbase;
    b1 = dlamc3_(&d__1, &zero);
    c1 = a;
    c2 = a;
    d1 = a;
    d2 = a;
/* +    WHILE( ( C1.EQ.A ).AND.( C2.EQ.A ).AND.   
      $       ( D1.EQ.A ).AND.( D2.EQ.A )      )LOOP */
L10:
    if (c1 == a && c2 == a && d1 == a && d2 == a) {
	--(*emin);
	a = b1;
	d__1 = a / *base;
	b1 = dlamc3_(&d__1, &zero);
	d__1 = b1 * *base;
	c1 = dlamc3_(&d__1, &zero);
	d1 = zero;
	for (i = 1; i <= *base; ++i) {
	    d1 += b1;
/* L20: */
	}
	d__1 = a * rbase;
	b2 = dlamc3_(&d__1, &zero);
	d__1 = b2 / rbase;
	c2 = dlamc3_(&d__1, &zero);
	d2 = zero;
	for (i = 1; i <= *base; ++i) {
	    d2 += b2;
/* L30: */
	}
	goto L10;
    }
/* +    END WHILE */

    return 0;

/*     End of DLAMC4 */

} /* dlamc4_ */


/* Subroutine */ int dlamc5_(int *beta, int *p, int *emin, 
	int *ieee, int *emax, double *rmax)
{
/*  -- LAPACK auxiliary routine (version 2.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       October 31, 1992   


    Purpose   
    =======   

    DLAMC5 attempts to compute RMAX, the largest machine floating-point   
    number, without overflow.  It assumes that EMAX + abs(EMIN) sum   
    approximately to a power of 2.  It will fail on machines where this   
    assumption does not hold, for example, the Cyber 205 (EMIN = -28625, 
  
    EMAX = 28718).  It will also fail if the value supplied for EMIN is   
    too large (i.e. too close to zero), probably with overflow.   

    Arguments   
    =========   

    BETA    (input) INT   
            The base of floating-point arithmetic.   

    P       (input) INT   
            The number of base BETA digits in the mantissa of a   
            floating-point value.   

    EMIN    (input) INT   
            The minimum exponent before (gradual) underflow.   

    IEEE    (input) INT   
            A int flag specifying whether or not the arithmetic   
            system is thought to comply with the IEEE standard.   

    EMAX    (output) INT   
            The largest exponent before overflow   

    RMAX    (output) DOUBLE PRECISION   
            The largest machine floating-point number.   

   ===================================================================== 
  


       First compute LEXP and UEXP, two powers of 2 that bound   
       abs(EMIN). We then assume that EMAX + abs(EMIN) will sum   
       approximately to the bound that is closest to abs(EMIN).   
       (EMAX is the exponent of the required number RMAX). */
    /* Table of constant values */
    static double c_b5 = 0.;
    
    /* System generated locals */
    double d__1;
    /* Local variables */
    static int lexp;
    static double oldy;
    static int uexp, i;
    static double y, z;
    static int nbits;
    extern double dlamc3_(double *, double *);
    static double recbas;
    static int exbits, expsum, try__;



    lexp = 1;
    exbits = 1;
L10:
    try__ = lexp << 1;
    if (try__ <= -(*emin)) {
	lexp = try__;
	++exbits;
	goto L10;
    }
    if (lexp == -(*emin)) {
	uexp = lexp;
    } else {
	uexp = try__;
	++exbits;
    }

/*     Now -LEXP is less than or equal to EMIN, and -UEXP is greater   
       than or equal to EMIN. EXBITS is the number of bits needed to   
       store the exponent. */

    if (uexp + *emin > -lexp - *emin) {
	expsum = lexp << 1;
    } else {
	expsum = uexp << 1;
    }

/*     EXPSUM is the exponent range, approximately equal to   
       EMAX - EMIN + 1 . */

    *emax = expsum + *emin - 1;
    nbits = exbits + 1 + *p;

/*     NBITS is the total number of bits needed to store a   
       floating-point number. */

    if (nbits % 2 == 1 && *beta == 2) {

/*        Either there are an odd number of bits used to store a   
          floating-point number, which is unlikely, or some bits are 
  
          not used in the representation of numbers, which is possible
,   
          (e.g. Cray machines) or the mantissa has an implicit bit,   
          (e.g. IEEE machines, Dec Vax machines), which is perhaps the
   
          most likely. We have to assume the last alternative.   
          If this is true, then we need to reduce EMAX by one because 
  
          there must be some way of representing zero in an implicit-b
it   
          system. On machines like Cray, we are reducing EMAX by one 
  
          unnecessarily. */

	--(*emax);
    }

    if (*ieee) {

/*        Assume we are on an IEEE machine which reserves one exponent
   
          for infinity and NaN. */

	--(*emax);
    }

/*     Now create RMAX, the largest machine number, which should   
       be equal to (1.0 - BETA**(-P)) * BETA**EMAX .   

       First compute 1.0 - BETA**(-P), being careful that the   
       result is less than 1.0 . */

    recbas = 1. / *beta;
    z = *beta - 1.;
    y = 0.;
    for (i = 1; i <= *p; ++i) {
	z *= recbas;
	if (y < 1.) {
	    oldy = y;
	}
	y = dlamc3_(&y, &z);
/* L20: */
    }
    if (y >= 1.) {
	y = oldy;
    }

/*     Now multiply by BETA**EMAX to get RMAX. */

    for (i = 1; i <= *emax; ++i) {
	d__1 = y * *beta;
	y = dlamc3_(&d__1, &c_b5);
/* L30: */
    }

    *rmax = y;
    return 0;

/*     End of DLAMC5 */

} /* dlamc5_ */

double pow_di(double *ap, int *bp)
{
    double pow, x;
    int n;

    pow = 1;
    x = *ap;
    n = *bp;

    if(n != 0){
	if(n < 0) {
	    n = -n;
	    x = 1/x;
	}
	for( ; ; ) {
	    if(n & 01) pow *= x;
	    if(n >>= 1)	x *= x;
	    else break;
	}
    }
    return(pow);
}
#endif /*HYPRE_USING_HYPRE_BLAS*/
