/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* f2c.h  --  Standard Fortran to C header file */

/**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

	- From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */

#ifndef F2C_INCLUDE
#define F2C_INCLUDE

/* MPI is not needed here, so don't include mpi.h */
#include "HYPRE_config.h"
#ifndef HYPRE_SEQUENTIAL
#define HYPRE_SEQUENTIAL
#endif
#include "_hypre_utilities.h"
#include "math.h"

#define sqrt hypre_sqrt
#define log hypre_log
#define pow hypre_pow

#ifdef HYPRE_BIGINT
typedef long long int HYPRE_LongInt;
typedef unsigned long long int HYPRE_ULongInt;
#else 
typedef long int HYPRE_LongInt;
typedef unsigned long int HYPRE_ULongInt;
#endif

/* F2C_INTEGER will normally be `HYPRE_Int' but would be `long' on 16-bit systems */
/* we assume short, float are OK */

/* integer changed to HYPRE_Int - edmond 1/12/00 */

typedef HYPRE_Int integer;
typedef HYPRE_ULongInt uinteger;
typedef char *address;
typedef short int shortint;
typedef float real;
typedef HYPRE_Real doublereal;
typedef struct { real r, i; } complex;
typedef struct { doublereal r, i; } doublecomplex;
typedef HYPRE_LongInt logical;
typedef short int shortlogical;
typedef char logical1;
typedef char integer1;
/* integer*8 support from f2c not currently supported: */
#if 0
typedef @F2C_LONGINT@ /* long long */ longint; /* system-dependent */
typedef unsigned @F2C_LONGINT@ ulongint;	/* system-dependent */
#define qbit_clear(a,b)	((a) & ~((ulongint)1 << (b)))
#define qbit_set(a,b)	((a) |  ((ulongint)1 << (b)))
#endif
/* typedef long long HYPRE_Int longint; */ /* RDF: removed */

#define TRUE_ (1)
#define FALSE_ (0)

/* Extern is for use with -E */
#ifndef Extern
#define Extern extern
#endif

/* I/O stuff */

#ifdef f2c_i2
  #error f2c_i2 will not work with g77!!!!
/* for -i2 */
typedef short flag;
typedef short ftnlen;
typedef short ftnint;
#else
typedef HYPRE_LongInt /* HYPRE_Int or long HYPRE_Int */ flag;
typedef HYPRE_Int /* HYPRE_Int or long HYPRE_Int */ ftnlen; /* changed by edmond */
typedef HYPRE_LongInt /* HYPRE_Int or long HYPRE_Int */ ftnint;
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

/*typedef long HYPRE_Int Long;*/	/* No longer used; formerly in Namelist */

struct Vardesc {	/* for Namelist */
	char *name;
	char *addr;
	ftnlen *dims;
	HYPRE_Int  type;
	};
typedef struct Vardesc Vardesc;

struct Namelist {
	char *name;
	Vardesc **vars;
	HYPRE_Int nvars;
	};
typedef struct Namelist Namelist;

/* The following undefs are to prevent conflicts with external libraries */
#undef abs
#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) (doublereal)abs(x)
#ifndef min
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#endif
#define dmin(a,b) (doublereal)min(a,b)
#define dmax(a,b) (doublereal)max(a,b)
#define bit_test(a,b)	((a) >> (b) & 1)
#define bit_clear(a,b)	((a) & ~((uinteger)1 << (b)))
#define bit_set(a,b)	((a) |  ((uinteger)1 << (b)))

/* procedure parameter types for -A and -C++ */

#define F2C_proc_par_types 1
#ifdef __cplusplus
typedef HYPRE_Int /* Unknown procedure type */ (*U_fp)(...);
typedef shortint (*J_fp)(...);
typedef integer (*I_fp)(...);
typedef real (*R_fp)(...);
typedef doublereal (*D_fp)(...), (*E_fp)(...);
typedef /* Complex */ VOID (*C_fp)(...);
typedef /* Double Complex */ VOID (*Z_fp)(...);
typedef logical (*L_fp)(...);
typedef shortlogical (*K_fp)(...);
typedef /* Character */ VOID (*H_fp)(...);
typedef /* Subroutine */ HYPRE_Int (*S_fp)(...);
#else
typedef HYPRE_Int /* Unknown procedure type */ (*U_fp)(void);
typedef shortint (*J_fp)(void);
typedef integer (*I_fp)(void);
typedef real (*R_fp)(void);
typedef doublereal (*D_fp)(void), (*E_fp)(void);
typedef /* Complex */ VOID (*C_fp)(void);
typedef /* Double Complex */ VOID (*Z_fp)(void);
typedef logical (*L_fp)(void);
typedef shortlogical (*K_fp)(void);
typedef /* Character */ VOID (*H_fp)(void);
typedef /* Subroutine */ HYPRE_Int (*S_fp)(void);
#endif
/* E_fp is for real functions when -R is not specified */
typedef VOID C_f;	/* complex function */
typedef VOID H_f;	/* character function */
typedef VOID Z_f;	/* HYPRE_Real complex function */
typedef doublereal E_f;	/* real function with -R not specified */

/* undef any lower-case symbols that your C compiler predefines, e.g.: */

#ifndef Skip_f2c_Undefs
/* (No such symbols should be defined in a strict ANSI C compiler.
   We can avoid trouble with f2c-translated code by using
   gcc -ansi [-traditional].) */
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
