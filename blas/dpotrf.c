#include "f2c.h"

/* Subroutine */ int dpotrf_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info)
{
/*  -- LAPACK routine (version 2.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       March 31, 1993   


    Purpose   
    =======   

    DPOTRF computes the Cholesky factorization of a real symmetric   
    positive definite matrix A.   

    The factorization has the form   
       A = U**T * U,  if UPLO = 'U', or   
       A = L  * L**T,  if UPLO = 'L',   
    where U is an upper triangular matrix and L is lower triangular.   

    This is the block version of the algorithm, calling Level 3 BLAS.   

    Arguments   
    =========   

    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of A is stored;   
            = 'L':  Lower triangle of A is stored.   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)   
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading 
  
            N-by-N upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of A contains the lower 
  
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization A = U**T*U or A = L*L**T.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  if INFO = i, the leading minor of order i is not   
                  positive definite, and the factorization could not be   
                  completed.   

    ===================================================================== 
  


       Test the input parameters.   

    
   Parameter adjustments   
       Function Body */
    /* Table of constant values */
    static integer c__1 = 1;
    static integer c_n1 = -1;
    static doublereal c_b13 = -1.;
    static doublereal c_b14 = 1.;
    
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;
    /* Local variables */
    static integer j;
    extern /* Subroutine */ int dgemm_(char *, char *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int dtrsm_(char *, char *, char *, char *, 
	    integer *, integer *, doublereal *, doublereal *, integer *, 
	    doublereal *, integer *);
    static logical upper;
    extern /* Subroutine */ int dsyrk_(char *, char *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *,
	     integer *), dpotf2_(char *, integer *, 
	    doublereal *, integer *, integer *);
    static integer jb, nb;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);




#define A(I,J) a[(I)-1 + ((J)-1)* ( *lda)]

    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DPOTRF", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "DPOTRF", uplo, n, &c_n1, &c_n1, &c_n1, 6L, 1L);
    if (nb <= 1 || nb >= *n) {

/*        Use unblocked code. */

	dpotf2_(uplo, n, &A(1,1), lda, info);
    } else {

/*        Use blocked code. */

	if (upper) {

/*           Compute the Cholesky factorization A = U'*U. */

	    i__1 = *n;
	    i__2 = nb;
	    for (j = 1; nb < 0 ? j >= *n : j <= *n; j += nb) {

/*              Update and factorize the current diagonal bloc
k and test   
                for non-positive-definiteness.   

   Computing MIN */
		i__3 = nb, i__4 = *n - j + 1;
		jb = min(i__3,i__4);
		i__3 = j - 1;
		dsyrk_("Upper", "Transpose", &jb, &i__3, &c_b13, &A(1,j), lda, &c_b14, &A(j,j), lda);
		dpotf2_("Upper", &jb, &A(j,j), lda, info);
		if (*info != 0) {
		    goto L30;
		}
		if (j + jb <= *n) {

/*                 Compute the current block row. */

		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
		    dgemm_("Transpose", "No transpose", &jb, &i__3, &i__4, &
			    c_b13, &A(1,j), lda, &A(1,j+jb), lda, &c_b14, &A(j,j+jb), lda);
		    i__3 = *n - j - jb + 1;
		    dtrsm_("Left", "Upper", "Transpose", "Non-unit", &jb, &
			    i__3, &c_b14, &A(j,j), lda, &A(j,j+jb), lda);
		}
/* L10: */
	    }

	} else {

/*           Compute the Cholesky factorization A = L*L'. */

	    i__2 = *n;
	    i__1 = nb;
	    for (j = 1; nb < 0 ? j >= *n : j <= *n; j += nb) {

/*              Update and factorize the current diagonal bloc
k and test   
                for non-positive-definiteness.   

   Computing MIN */
		i__3 = nb, i__4 = *n - j + 1;
		jb = min(i__3,i__4);
		i__3 = j - 1;
		dsyrk_("Lower", "No transpose", &jb, &i__3, &c_b13, &A(j,1), lda, &c_b14, &A(j,j), lda);
		dpotf2_("Lower", &jb, &A(j,j), lda, info);
		if (*info != 0) {
		    goto L30;
		}
		if (j + jb <= *n) {

/*                 Compute the current block column. */

		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
		    dgemm_("No transpose", "Transpose", &i__3, &jb, &i__4, &
			    c_b13, &A(j+jb,1), lda, &A(j,1), 
			    lda, &c_b14, &A(j+jb,j), lda);
		    i__3 = *n - j - jb + 1;
		    dtrsm_("Right", "Lower", "Transpose", "Non-unit", &i__3, &
			    jb, &c_b14, &A(j,j), lda, &A(j+jb,j), lda);
		}
/* L20: */
	    }
	}
    }
    goto L40;

L30:
    *info = *info + j - 1;

L40:
    return 0;

/*     End of DPOTRF */

} /* dpotrf_ */

#include "f2c.h"

/* Subroutine */ int dpotf2_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info)
{
/*  -- LAPACK routine (version 2.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       February 29, 1992   


    Purpose   
    =======   

    DPOTF2 computes the Cholesky factorization of a real symmetric   
    positive definite matrix A.   

    The factorization has the form   
       A = U' * U ,  if UPLO = 'U', or   
       A = L  * L',  if UPLO = 'L',   
    where U is an upper triangular matrix and L is lower triangular.   

    This is the unblocked version of the algorithm, calling Level 2 BLAS. 
  

    Arguments   
    =========   

    UPLO    (input) CHARACTER*1   
            Specifies whether the upper or lower triangular part of the   
            symmetric matrix A is stored.   
            = 'U':  Upper triangular   
            = 'L':  Lower triangular   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)   
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading 
  
            n by n upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading n by n lower triangular part of A contains the lower 
  
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization A = U'*U  or A = L*L'.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    INFO    (output) INTEGER   
            = 0: successful exit   
            < 0: if INFO = -k, the k-th argument had an illegal value   
            > 0: if INFO = k, the leading minor of order k is not   
                 positive definite, and the factorization could not be   
                 completed.   

    ===================================================================== 
  


       Test the input parameters.   

    
   Parameter adjustments   
       Function Body */
    /* Table of constant values */
    static integer c__1 = 1;
    static doublereal c_b10 = -1.;
    static doublereal c_b12 = 1.;
    
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;
    doublereal d__1;
    /* Builtin functions */
    double sqrt(doublereal);
    /* Local variables */
    extern doublereal ddot_(integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    static integer j;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int dgemv_(char *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, integer *);
    static logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static doublereal ajj;




#define A(I,J) a[(I)-1 + ((J)-1)* ( *lda)]

    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DPOTF2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    if (upper) {

/*        Compute the Cholesky factorization A = U'*U. */

	i__1 = *n;
	for (j = 1; j <= *n; ++j) {

/*           Compute U(J,J) and test for non-positive-definiteness
. */

	    i__2 = j - 1;
	    ajj = A(j,j) - ddot_(&i__2, &A(1,j), &c__1, 
		    &A(1,j), &c__1);
	    if (ajj <= 0.) {
		A(j,j) = ajj;
		goto L30;
	    }
	    ajj = sqrt(ajj);
	    A(j,j) = ajj;

/*           Compute elements J+1:N of row J. */

	    if (j < *n) {
		i__2 = j - 1;
		i__3 = *n - j;
		dgemv_("Transpose", &i__2, &i__3, &c_b10, &A(1,j+1), lda, &A(1,j), &c__1, &c_b12, &A(j,j+1), lda);
		i__2 = *n - j;
		d__1 = 1. / ajj;
		dscal_(&i__2, &d__1, &A(j,j+1), lda);
	    }
/* L10: */
	}
    } else {

/*        Compute the Cholesky factorization A = L*L'. */

	i__1 = *n;
	for (j = 1; j <= *n; ++j) {

/*           Compute L(J,J) and test for non-positive-definiteness
. */

	    i__2 = j - 1;
	    ajj = A(j,j) - ddot_(&i__2, &A(j,1), lda, &A(j,1), lda);
	    if (ajj <= 0.) {
		A(j,j) = ajj;
		goto L30;
	    }
	    ajj = sqrt(ajj);
	    A(j,j) = ajj;

/*           Compute elements J+1:N of column J. */

	    if (j < *n) {
		i__2 = *n - j;
		i__3 = j - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b10, &A(j+1,1), lda, &A(j,1), lda, &c_b12, &A(j+1,j), &c__1);
		i__2 = *n - j;
		d__1 = 1. / ajj;
		dscal_(&i__2, &d__1, &A(j+1,j), &c__1);
	    }
/* L20: */
	}
    }
    goto L40;

L30:
    *info = j;

L40:
    return 0;

/*     End of DPOTF2 */

} /* dpotf2_ */

/* f2c.h  --  Standard Fortran to C header file */

/**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

	- From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */

#ifndef F2C_INCLUDE
#define F2C_INCLUDE

typedef long int integer;
typedef char *address;
typedef short int shortint;
typedef float real;
typedef double doublereal;
typedef struct { real r, i; } complex;
typedef struct { doublereal r, i; } doublecomplex;
typedef long int logical;
typedef short int shortlogical;
typedef char logical1;
typedef char integer1;
/* typedef long long longint; */ /* system-dependent */

#define TRUE_ (1)
#define FALSE_ (0)

/* Extern is for use with -E */
#ifndef Extern
#define Extern extern
#endif

/* I/O stuff */

#ifdef f2c_i2
/* for -i2 */
typedef short flag;
typedef short ftnlen;
typedef short ftnint;
#else
typedef long flag;
typedef long ftnlen;
typedef long ftnint;
#endif

/*external read, write*/
typedef struct
{	flag cierr;
	ftnint ciunit;
	flag ciend;
	char *cifmt;
	ftnint cirec;
} cilist;

/*internal read, write*/
typedef struct
{	flag icierr;
	char *iciunit;
	flag iciend;
	char *icifmt;
	ftnint icirlen;
	ftnint icirnum;
} icilist;

/*open*/
typedef struct
{	flag oerr;
	ftnint ounit;
	char *ofnm;
	ftnlen ofnmlen;
	char *osta;
	char *oacc;
	char *ofm;
	ftnint orl;
	char *oblnk;
} olist;

/*close*/
typedef struct
{	flag cerr;
	ftnint cunit;
	char *csta;
} cllist;

/*rewind, backspace, endfile*/
typedef struct
{	flag aerr;
	ftnint aunit;
} alist;

/* inquire */
typedef struct
{	flag inerr;
	ftnint inunit;
	char *infile;
	ftnlen infilen;
	ftnint	*inex;	/*parameters in standard's order*/
	ftnint	*inopen;
	ftnint	*innum;
	ftnint	*innamed;
	char	*inname;
	ftnlen	innamlen;
	char	*inacc;
	ftnlen	inacclen;
	char	*inseq;
	ftnlen	inseqlen;
	char 	*indir;
	ftnlen	indirlen;
	char	*infmt;
	ftnlen	infmtlen;
	char	*inform;
	ftnint	informlen;
	char	*inunf;
	ftnlen	inunflen;
	ftnint	*inrecl;
	ftnint	*innrec;
	char	*inblank;
	ftnlen	inblanklen;
} inlist;

#define VOID void

union Multitype {	/* for multiple entry points */
	integer1 g;
	shortint h;
	integer i;
	/* longint j; */
	real r;
	doublereal d;
	complex c;
	doublecomplex z;
	};

typedef union Multitype Multitype;

typedef long Long;	/* No longer used; formerly in Namelist */

struct Vardesc {	/* for Namelist */
	char *name;
	char *addr;
	ftnlen *dims;
	int  type;
	};
typedef struct Vardesc Vardesc;

struct Namelist {
	char *name;
	Vardesc **vars;
	int nvars;
	};
typedef struct Namelist Namelist;

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) (doublereal)abs(x)
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#define dmin(a,b) (doublereal)min(a,b)
#define dmax(a,b) (doublereal)max(a,b)

/* procedure parameter types for -A and -C++ */

#define F2C_proc_par_types 1
#ifdef __cplusplus
typedef int /* Unknown procedure type */ (*U_fp)(...);
typedef shortint (*J_fp)(...);
typedef integer (*I_fp)(...);
typedef real (*R_fp)(...);
typedef doublereal (*D_fp)(...), (*E_fp)(...);
typedef /* Complex */ VOID (*C_fp)(...);
typedef /* Double Complex */ VOID (*Z_fp)(...);
typedef logical (*L_fp)(...);
typedef shortlogical (*K_fp)(...);
typedef /* Character */ VOID (*H_fp)(...);
typedef /* Subroutine */ int (*S_fp)(...);
#else
typedef int /* Unknown procedure type */ (*U_fp)();
typedef shortint (*J_fp)();
typedef integer (*I_fp)();
typedef real (*R_fp)();
typedef doublereal (*D_fp)(), (*E_fp)();
typedef /* Complex */ VOID (*C_fp)();
typedef /* Double Complex */ VOID (*Z_fp)();
typedef logical (*L_fp)();
typedef shortlogical (*K_fp)();
typedef /* Character */ VOID (*H_fp)();
typedef /* Subroutine */ int (*S_fp)();
#endif
/* E_fp is for real functions when -R is not specified */
typedef VOID C_f;	/* complex function */
typedef VOID H_f;	/* character function */
typedef VOID Z_f;	/* double complex function */
typedef doublereal E_f;	/* real function with -R not specified */

/* undef any lower-case symbols that your C compiler predefines, e.g.: */

#ifndef Skip_f2c_Undefs
#undef cray
#undef gcos
#undef mc68010
#undef mc68020
#undef mips
#undef pdp11
#undef sgi
#undef sparc
#undef sun
#undef sun2
#undef sun3
#undef sun4
#undef u370
#undef u3b
#undef u3b2
#undef u3b5
#undef unix
#undef vax
#endif
#endif
#include "f2c.h"

integer ilaenv_(integer *ispec, char *name, char *opts, integer *n1, integer *
	n2, integer *n3, integer *n4, ftnlen name_len, ftnlen opts_len)
{
/*  -- LAPACK auxiliary routine (version 2.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       September 30, 1994   


    Purpose   
    =======   

    ILAENV is called from the LAPACK routines to choose problem-dependent 
  
    parameters for the local environment.  See ISPEC for a description of 
  
    the parameters.   

    This version provides a set of parameters which should give good,   
    but not optimal, performance on many of the currently available   
    computers.  Users are encouraged to modify this subroutine to set   
    the tuning parameters for their particular machine using the option   
    and problem size information in the arguments.   

    This routine will not function correctly if it is converted to all   
    lower case.  Converting it to all upper case is allowed.   

    Arguments   
    =========   

    ISPEC   (input) INTEGER   
            Specifies the parameter to be returned as the value of   
            ILAENV.   
            = 1: the optimal blocksize; if this value is 1, an unblocked 
  
                 algorithm will give the best performance.   
            = 2: the minimum block size for which the block routine   
                 should be used; if the usable block size is less than   
                 this value, an unblocked routine should be used.   
            = 3: the crossover point (in a block routine, for N less   
                 than this value, an unblocked routine should be used)   
            = 4: the number of shifts, used in the nonsymmetric   
                 eigenvalue routines   
            = 5: the minimum column dimension for blocking to be used;   
                 rectangular blocks must have dimension at least k by m, 
  
                 where k is given by ILAENV(2,...) and m by ILAENV(5,...) 
  
            = 6: the crossover point for the SVD (when reducing an m by n 
  
                 matrix to bidiagonal form, if max(m,n)/min(m,n) exceeds 
  
                 this value, a QR factorization is used first to reduce   
                 the matrix to a triangular form.)   
            = 7: the number of processors   
            = 8: the crossover point for the multishift QR and QZ methods 
  
                 for nonsymmetric eigenvalue problems.   

    NAME    (input) CHARACTER*(*)   
            The name of the calling subroutine, in either upper case or   
            lower case.   

    OPTS    (input) CHARACTER*(*)   
            The character options to the subroutine NAME, concatenated   
            into a single character string.  For example, UPLO = 'U',   
            TRANS = 'T', and DIAG = 'N' for a triangular routine would   
            be specified as OPTS = 'UTN'.   

    N1      (input) INTEGER   
    N2      (input) INTEGER   
    N3      (input) INTEGER   
    N4      (input) INTEGER   
            Problem dimensions for the subroutine NAME; these may not all 
  
            be required.   

   (ILAENV) (output) INTEGER   
            >= 0: the value of the parameter specified by ISPEC   
            < 0:  if ILAENV = -k, the k-th argument had an illegal value. 
  

    Further Details   
    ===============   

    The following conventions have been used when calling ILAENV from the 
  
    LAPACK routines:   
    1)  OPTS is a concatenation of all of the character options to   
        subroutine NAME, in the same order that they appear in the   
        argument list for NAME, even if they are not used in determining 
  
        the value of the parameter specified by ISPEC.   
    2)  The problem dimensions N1, N2, N3, N4 are specified in the order 
  
        that they appear in the argument list for NAME.  N1 is used   
        first, N2 second, and so on, and unused problem dimensions are   
        passed a value of -1.   
    3)  The parameter value returned by ILAENV is checked for validity in 
  
        the calling subroutine.  For example, ILAENV is used to retrieve 
  
        the optimal blocksize for STRTRI as follows:   

        NB = ILAENV( 1, 'STRTRI', UPLO // DIAG, N, -1, -1, -1 )   
        IF( NB.LE.1 ) NB = MAX( 1, N )   

    ===================================================================== 
*/
/* >>Start of File<<   
       System generated locals */
    integer ret_val;
    /* Builtin functions   
       Subroutine */ int s_copy(char *, char *, ftnlen, ftnlen);
    integer s_cmp(char *, char *, ftnlen, ftnlen);
    /* Local variables */
    static integer i;
    static logical cname, sname;
    static integer nbmin;
    static char c1[1], c2[2], c3[3], c4[2];
    static integer ic, nb, iz, nx;
    static char subnam[6];



    switch (*ispec) {
	case 1:  goto L100;
	case 2:  goto L100;
	case 3:  goto L100;
	case 4:  goto L400;
	case 5:  goto L500;
	case 6:  goto L600;
	case 7:  goto L700;
	case 8:  goto L800;
    }

/*     Invalid value for ISPEC */

    ret_val = -1;
    return ret_val;

L100:

/*     Convert NAME to upper case if the first character is lower case. */

    ret_val = 1;
    s_copy(subnam, name, 6L, name_len);
    ic = *(unsigned char *)subnam;
    iz = 'Z';
    if (iz == 90 || iz == 122) {

/*        ASCII character set */

	if (ic >= 97 && ic <= 122) {
	    *(unsigned char *)subnam = (char) (ic - 32);
	    for (i = 2; i <= 6; ++i) {
		ic = *(unsigned char *)&subnam[i - 1];
		if (ic >= 97 && ic <= 122) {
		    *(unsigned char *)&subnam[i - 1] = (char) (ic - 32);
		}
/* L10: */
	    }
	}

    } else if (iz == 233 || iz == 169) {

/*        EBCDIC character set */

	if (ic >= 129 && ic <= 137 || ic >= 145 && ic <= 153 || ic >= 162 && 
		ic <= 169) {
	    *(unsigned char *)subnam = (char) (ic + 64);
	    for (i = 2; i <= 6; ++i) {
		ic = *(unsigned char *)&subnam[i - 1];
		if (ic >= 129 && ic <= 137 || ic >= 145 && ic <= 153 || ic >= 
			162 && ic <= 169) {
		    *(unsigned char *)&subnam[i - 1] = (char) (ic + 64);
		}
/* L20: */
	    }
	}

    } else if (iz == 218 || iz == 250) {

/*        Prime machines:  ASCII+128 */

	if (ic >= 225 && ic <= 250) {
	    *(unsigned char *)subnam = (char) (ic - 32);
	    for (i = 2; i <= 6; ++i) {
		ic = *(unsigned char *)&subnam[i - 1];
		if (ic >= 225 && ic <= 250) {
		    *(unsigned char *)&subnam[i - 1] = (char) (ic - 32);
		}
/* L30: */
	    }
	}
    }

    *(unsigned char *)c1 = *(unsigned char *)subnam;
    sname = *(unsigned char *)c1 == 'S' || *(unsigned char *)c1 == 'D';
    cname = *(unsigned char *)c1 == 'C' || *(unsigned char *)c1 == 'Z';
    if (! (cname || sname)) {
	return ret_val;
    }
    s_copy(c2, subnam + 1, 2L, 2L);
    s_copy(c3, subnam + 3, 3L, 3L);
    s_copy(c4, c3 + 1, 2L, 2L);

    switch (*ispec) {
	case 1:  goto L110;
	case 2:  goto L200;
	case 3:  goto L300;
    }

L110:

/*     ISPEC = 1:  block size   

       In these examples, separate code is provided for setting NB for   
       real and complex.  We assume that NB will take the same value in   
       single or double precision. */

    nb = 1;

    if (s_cmp(c2, "GE", 2L, 2L) == 0) {
	if (s_cmp(c3, "TRF", 3L, 3L) == 0) {
	    if (sname) {
		nb = 64;
	    } else {
		nb = 64;
	    }
	} else if (s_cmp(c3, "QRF", 3L, 3L) == 0 || s_cmp(c3, "RQF", 3L, 3L) 
		== 0 || s_cmp(c3, "LQF", 3L, 3L) == 0 || s_cmp(c3, "QLF", 3L, 
		3L) == 0) {
	    if (sname) {
		nb = 32;
	    } else {
		nb = 32;
	    }
	} else if (s_cmp(c3, "HRD", 3L, 3L) == 0) {
	    if (sname) {
		nb = 32;
	    } else {
		nb = 32;
	    }
	} else if (s_cmp(c3, "BRD", 3L, 3L) == 0) {
	    if (sname) {
		nb = 32;
	    } else {
		nb = 32;
	    }
	} else if (s_cmp(c3, "TRI", 3L, 3L) == 0) {
	    if (sname) {
		nb = 64;
	    } else {
		nb = 64;
	    }
	}
    } else if (s_cmp(c2, "PO", 2L, 2L) == 0) {
	if (s_cmp(c3, "TRF", 3L, 3L) == 0) {
	    if (sname) {
		nb = 64;
	    } else {
		nb = 64;
	    }
	}
    } else if (s_cmp(c2, "SY", 2L, 2L) == 0) {
	if (s_cmp(c3, "TRF", 3L, 3L) == 0) {
	    if (sname) {
		nb = 64;
	    } else {
		nb = 64;
	    }
	} else if (sname && s_cmp(c3, "TRD", 3L, 3L) == 0) {
	    nb = 1;
	} else if (sname && s_cmp(c3, "GST", 3L, 3L) == 0) {
	    nb = 64;
	}
    } else if (cname && s_cmp(c2, "HE", 2L, 2L) == 0) {
	if (s_cmp(c3, "TRF", 3L, 3L) == 0) {
	    nb = 64;
	} else if (s_cmp(c3, "TRD", 3L, 3L) == 0) {
	    nb = 1;
	} else if (s_cmp(c3, "GST", 3L, 3L) == 0) {
	    nb = 64;
	}
    } else if (sname && s_cmp(c2, "OR", 2L, 2L) == 0) {
	if (*(unsigned char *)c3 == 'G') {
	    if (s_cmp(c4, "QR", 2L, 2L) == 0 || s_cmp(c4, "RQ", 2L, 2L) == 0 
		    || s_cmp(c4, "LQ", 2L, 2L) == 0 || s_cmp(c4, "QL", 2L, 2L)
		     == 0 || s_cmp(c4, "HR", 2L, 2L) == 0 || s_cmp(c4, "TR", 
		    2L, 2L) == 0 || s_cmp(c4, "BR", 2L, 2L) == 0) {
		nb = 32;
	    }
	} else if (*(unsigned char *)c3 == 'M') {
	    if (s_cmp(c4, "QR", 2L, 2L) == 0 || s_cmp(c4, "RQ", 2L, 2L) == 0 
		    || s_cmp(c4, "LQ", 2L, 2L) == 0 || s_cmp(c4, "QL", 2L, 2L)
		     == 0 || s_cmp(c4, "HR", 2L, 2L) == 0 || s_cmp(c4, "TR", 
		    2L, 2L) == 0 || s_cmp(c4, "BR", 2L, 2L) == 0) {
		nb = 32;
	    }
	}
    } else if (cname && s_cmp(c2, "UN", 2L, 2L) == 0) {
	if (*(unsigned char *)c3 == 'G') {
	    if (s_cmp(c4, "QR", 2L, 2L) == 0 || s_cmp(c4, "RQ", 2L, 2L) == 0 
		    || s_cmp(c4, "LQ", 2L, 2L) == 0 || s_cmp(c4, "QL", 2L, 2L)
		     == 0 || s_cmp(c4, "HR", 2L, 2L) == 0 || s_cmp(c4, "TR", 
		    2L, 2L) == 0 || s_cmp(c4, "BR", 2L, 2L) == 0) {
		nb = 32;
	    }
	} else if (*(unsigned char *)c3 == 'M') {
	    if (s_cmp(c4, "QR", 2L, 2L) == 0 || s_cmp(c4, "RQ", 2L, 2L) == 0 
		    || s_cmp(c4, "LQ", 2L, 2L) == 0 || s_cmp(c4, "QL", 2L, 2L)
		     == 0 || s_cmp(c4, "HR", 2L, 2L) == 0 || s_cmp(c4, "TR", 
		    2L, 2L) == 0 || s_cmp(c4, "BR", 2L, 2L) == 0) {
		nb = 32;
	    }
	}
    } else if (s_cmp(c2, "GB", 2L, 2L) == 0) {
	if (s_cmp(c3, "TRF", 3L, 3L) == 0) {
	    if (sname) {
		if (*n4 <= 64) {
		    nb = 1;
		} else {
		    nb = 32;
		}
	    } else {
		if (*n4 <= 64) {
		    nb = 1;
		} else {
		    nb = 32;
		}
	    }
	}
    } else if (s_cmp(c2, "PB", 2L, 2L) == 0) {
	if (s_cmp(c3, "TRF", 3L, 3L) == 0) {
	    if (sname) {
		if (*n2 <= 64) {
		    nb = 1;
		} else {
		    nb = 32;
		}
	    } else {
		if (*n2 <= 64) {
		    nb = 1;
		} else {
		    nb = 32;
		}
	    }
	}
    } else if (s_cmp(c2, "TR", 2L, 2L) == 0) {
	if (s_cmp(c3, "TRI", 3L, 3L) == 0) {
	    if (sname) {
		nb = 64;
	    } else {
		nb = 64;
	    }
	}
    } else if (s_cmp(c2, "LA", 2L, 2L) == 0) {
	if (s_cmp(c3, "UUM", 3L, 3L) == 0) {
	    if (sname) {
		nb = 64;
	    } else {
		nb = 64;
	    }
	}
    } else if (sname && s_cmp(c2, "ST", 2L, 2L) == 0) {
	if (s_cmp(c3, "EBZ", 3L, 3L) == 0) {
	    nb = 1;
	}
    }
    ret_val = nb;
    return ret_val;

L200:

/*     ISPEC = 2:  minimum block size */

    nbmin = 2;
    if (s_cmp(c2, "GE", 2L, 2L) == 0) {
	if (s_cmp(c3, "QRF", 3L, 3L) == 0 || s_cmp(c3, "RQF", 3L, 3L) == 0 || 
		s_cmp(c3, "LQF", 3L, 3L) == 0 || s_cmp(c3, "QLF", 3L, 3L) == 
		0) {
	    if (sname) {
		nbmin = 2;
	    } else {
		nbmin = 2;
	    }
	} else if (s_cmp(c3, "HRD", 3L, 3L) == 0) {
	    if (sname) {
		nbmin = 2;
	    } else {
		nbmin = 2;
	    }
	} else if (s_cmp(c3, "BRD", 3L, 3L) == 0) {
	    if (sname) {
		nbmin = 2;
	    } else {
		nbmin = 2;
	    }
	} else if (s_cmp(c3, "TRI", 3L, 3L) == 0) {
	    if (sname) {
		nbmin = 2;
	    } else {
		nbmin = 2;
	    }
	}
    } else if (s_cmp(c2, "SY", 2L, 2L) == 0) {
	if (s_cmp(c3, "TRF", 3L, 3L) == 0) {
	    if (sname) {
		nbmin = 8;
	    } else {
		nbmin = 8;
	    }
	} else if (sname && s_cmp(c3, "TRD", 3L, 3L) == 0) {
	    nbmin = 2;
	}
    } else if (cname && s_cmp(c2, "HE", 2L, 2L) == 0) {
	if (s_cmp(c3, "TRD", 3L, 3L) == 0) {
	    nbmin = 2;
	}
    } else if (sname && s_cmp(c2, "OR", 2L, 2L) == 0) {
	if (*(unsigned char *)c3 == 'G') {
	    if (s_cmp(c4, "QR", 2L, 2L) == 0 || s_cmp(c4, "RQ", 2L, 2L) == 0 
		    || s_cmp(c4, "LQ", 2L, 2L) == 0 || s_cmp(c4, "QL", 2L, 2L)
		     == 0 || s_cmp(c4, "HR", 2L, 2L) == 0 || s_cmp(c4, "TR", 
		    2L, 2L) == 0 || s_cmp(c4, "BR", 2L, 2L) == 0) {
		nbmin = 2;
	    }
	} else if (*(unsigned char *)c3 == 'M') {
	    if (s_cmp(c4, "QR", 2L, 2L) == 0 || s_cmp(c4, "RQ", 2L, 2L) == 0 
		    || s_cmp(c4, "LQ", 2L, 2L) == 0 || s_cmp(c4, "QL", 2L, 2L)
		     == 0 || s_cmp(c4, "HR", 2L, 2L) == 0 || s_cmp(c4, "TR", 
		    2L, 2L) == 0 || s_cmp(c4, "BR", 2L, 2L) == 0) {
		nbmin = 2;
	    }
	}
    } else if (cname && s_cmp(c2, "UN", 2L, 2L) == 0) {
	if (*(unsigned char *)c3 == 'G') {
	    if (s_cmp(c4, "QR", 2L, 2L) == 0 || s_cmp(c4, "RQ", 2L, 2L) == 0 
		    || s_cmp(c4, "LQ", 2L, 2L) == 0 || s_cmp(c4, "QL", 2L, 2L)
		     == 0 || s_cmp(c4, "HR", 2L, 2L) == 0 || s_cmp(c4, "TR", 
		    2L, 2L) == 0 || s_cmp(c4, "BR", 2L, 2L) == 0) {
		nbmin = 2;
	    }
	} else if (*(unsigned char *)c3 == 'M') {
	    if (s_cmp(c4, "QR", 2L, 2L) == 0 || s_cmp(c4, "RQ", 2L, 2L) == 0 
		    || s_cmp(c4, "LQ", 2L, 2L) == 0 || s_cmp(c4, "QL", 2L, 2L)
		     == 0 || s_cmp(c4, "HR", 2L, 2L) == 0 || s_cmp(c4, "TR", 
		    2L, 2L) == 0 || s_cmp(c4, "BR", 2L, 2L) == 0) {
		nbmin = 2;
	    }
	}
    }
    ret_val = nbmin;
    return ret_val;

L300:

/*     ISPEC = 3:  crossover point */

    nx = 0;
    if (s_cmp(c2, "GE", 2L, 2L) == 0) {
	if (s_cmp(c3, "QRF", 3L, 3L) == 0 || s_cmp(c3, "RQF", 3L, 3L) == 0 || 
		s_cmp(c3, "LQF", 3L, 3L) == 0 || s_cmp(c3, "QLF", 3L, 3L) == 
		0) {
	    if (sname) {
		nx = 128;
	    } else {
		nx = 128;
	    }
	} else if (s_cmp(c3, "HRD", 3L, 3L) == 0) {
	    if (sname) {
		nx = 128;
	    } else {
		nx = 128;
	    }
	} else if (s_cmp(c3, "BRD", 3L, 3L) == 0) {
	    if (sname) {
		nx = 128;
	    } else {
		nx = 128;
	    }
	}
    } else if (s_cmp(c2, "SY", 2L, 2L) == 0) {
	if (sname && s_cmp(c3, "TRD", 3L, 3L) == 0) {
	    nx = 1;
	}
    } else if (cname && s_cmp(c2, "HE", 2L, 2L) == 0) {
	if (s_cmp(c3, "TRD", 3L, 3L) == 0) {
	    nx = 1;
	}
    } else if (sname && s_cmp(c2, "OR", 2L, 2L) == 0) {
	if (*(unsigned char *)c3 == 'G') {
	    if (s_cmp(c4, "QR", 2L, 2L) == 0 || s_cmp(c4, "RQ", 2L, 2L) == 0 
		    || s_cmp(c4, "LQ", 2L, 2L) == 0 || s_cmp(c4, "QL", 2L, 2L)
		     == 0 || s_cmp(c4, "HR", 2L, 2L) == 0 || s_cmp(c4, "TR", 
		    2L, 2L) == 0 || s_cmp(c4, "BR", 2L, 2L) == 0) {
		nx = 128;
	    }
	}
    } else if (cname && s_cmp(c2, "UN", 2L, 2L) == 0) {
	if (*(unsigned char *)c3 == 'G') {
	    if (s_cmp(c4, "QR", 2L, 2L) == 0 || s_cmp(c4, "RQ", 2L, 2L) == 0 
		    || s_cmp(c4, "LQ", 2L, 2L) == 0 || s_cmp(c4, "QL", 2L, 2L)
		     == 0 || s_cmp(c4, "HR", 2L, 2L) == 0 || s_cmp(c4, "TR", 
		    2L, 2L) == 0 || s_cmp(c4, "BR", 2L, 2L) == 0) {
		nx = 128;
	    }
	}
    }
    ret_val = nx;
    return ret_val;

L400:

/*     ISPEC = 4:  number of shifts (used by xHSEQR) */

    ret_val = 6;
    return ret_val;

L500:

/*     ISPEC = 5:  minimum column dimension (not used) */

    ret_val = 2;
    return ret_val;

L600:

/*     ISPEC = 6:  crossover point for SVD (used by xGELSS and xGESVD) */

    ret_val = (integer) ((real) min(*n1,*n2) * 1.6f);
    return ret_val;

L700:

/*     ISPEC = 7:  number of processors (not used) */

    ret_val = 1;
    return ret_val;

L800:

/*     ISPEC = 8:  crossover point for multishift (used by xHSEQR) */

    ret_val = 50;
    return ret_val;

/*     End of ILAENV */

} /* ilaenv_ */

#include "f2c.h"

logical lsame_(char *ca, char *cb)
{
/*  -- LAPACK auxiliary routine (version 2.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       September 30, 1994   


    Purpose   
    =======   

    LSAME returns .TRUE. if CA is the same letter as CB regardless of   
    case.   

    Arguments   
    =========   

    CA      (input) CHARACTER*1   
    CB      (input) CHARACTER*1   
            CA and CB specify the single characters to be compared.   

   ===================================================================== 
  


       Test if the characters are equal */
    /* System generated locals */
    logical ret_val;
    /* Local variables */
    static integer inta, intb, zcode;


    ret_val = *(unsigned char *)ca == *(unsigned char *)cb;
    if (ret_val) {
	return ret_val;
    }

/*     Now test for equivalence if both characters are alphabetic. */

    zcode = 'Z';

/*     Use 'Z' rather than 'A' so that ASCII can be detected on Prime   
       machines, on which ICHAR returns a value with bit 8 set.   
       ICHAR('A') on Prime machines returns 193 which is the same as   
       ICHAR('A') on an EBCDIC machine. */

    inta = *(unsigned char *)ca;
    intb = *(unsigned char *)cb;

    if (zcode == 90 || zcode == 122) {

/*        ASCII is assumed - ZCODE is the ASCII code of either lower o
r   
          upper case 'Z'. */

	if (inta >= 97 && inta <= 122) {
	    inta += -32;
	}
	if (intb >= 97 && intb <= 122) {
	    intb += -32;
	}

    } else if (zcode == 233 || zcode == 169) {

/*        EBCDIC is assumed - ZCODE is the EBCDIC code of either lower
 or   
          upper case 'Z'. */

	if (inta >= 129 && inta <= 137 || inta >= 145 && inta <= 153 || inta 
		>= 162 && inta <= 169) {
	    inta += 64;
	}
	if (intb >= 129 && intb <= 137 || intb >= 145 && intb <= 153 || intb 
		>= 162 && intb <= 169) {
	    intb += 64;
	}

    } else if (zcode == 218 || zcode == 250) {

/*        ASCII is assumed, on Prime machines - ZCODE is the ASCII cod
e   
          plus 128 of either lower or upper case 'Z'. */

	if (inta >= 225 && inta <= 250) {
	    inta += -32;
	}
	if (intb >= 225 && intb <= 250) {
	    intb += -32;
	}
    }
    ret_val = inta == intb;

/*     RETURN   

       End of LSAME */

    return ret_val;
} /* lsame_ */

#include "f2c.h"

/* Subroutine */ int xerbla_(char *srname, integer *info)
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

    INFO    (input) INTEGER   
            The position of the invalid parameter in the parameter list   

            of the calling routine.   

   ===================================================================== 
*/

    printf("** On entry to %6s, parameter number %2i had an illegal value\n",
		srname, *info);

/*     End of XERBLA */

    return 0;
} /* xerbla_ */

