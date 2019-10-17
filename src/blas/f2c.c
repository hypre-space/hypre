/* Copyright (c) 1992-2008 The University of Tennessee.  All rights reserved.
 * See file COPYING in this directory for details. */

#ifdef __cplusplus
#define REGISTER 
#else
#define REGISTER register
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*-----------------------------------------------------------------------------
 * Contains functions found in the f2c library to avoid needing -lf2c
 *-----------------------------------------------------------------------------*/

#include "f2c.h"
#include "hypre_blas.h"
	
/* compare two strings */

integer s_cmp(char *a0,const char *b0, ftnlen la, ftnlen lb)
{
REGISTER unsigned char *a, *aend, *b, *bend;
a = (unsigned char *)a0;
b = (unsigned char *)b0;
aend = a + la;
bend = b + lb;

if(la <= lb)
	{
	while(a < aend)
		if(*a != *b)
			return( *a - *b );
		else
			{ ++a; ++b; }

	while(b < bend)
		if(*b != ' ')
			return( ' ' - *b );
		else	++b;
	}

else
	{
	while(b < bend)
		if(*a == *b)
			{ ++a; ++b; }
		else
			return( *a - *b );
	while(a < aend)
		if(*a != ' ')
			return(*a - ' ');
		else	++a;
	}
return(0);
}

/* assign strings:  a = b */

integer s_copy(char *a,const char *b, ftnlen la, ftnlen lb)
{
REGISTER char *aend, *bend;

aend = a + la;

if(la <= lb)
	while(a < aend)
		*a++ = *b++;

else
	{
		bend = (char*)b + lb;
	while(b < bend)
		*a++ = *b++;
	while(a < aend)
		*a++ = ' ';
	}
return(0);
}

integer s_cat(char *lp, char *rpp[], ftnlen rnp[], ftnlen *np, ftnlen ll)
{
ftnlen i, n, nc;
char *f__rp;

n = (integer)*np;
for(i = 0 ; i < n ; ++i)
	{
	nc = ll;
	if(rnp[i] < nc)
		nc = rnp[i];
	ll -= nc;
	f__rp = rpp[i];
	while(--nc >= 0)
		*lp++ = *f__rp++;
	}
while(--ll >= 0)
	*lp++ = ' ';
return 0;
}

#define log10e 0.43429448190325182765

#undef abs
#include "math.h"					 
doublereal d_lg10(doublereal *x)
{
return( log10e * log(*x) );
}

doublereal d_sign(doublereal *a, doublereal *b)
{
doublereal x;
x = (*a >= 0 ? *a : - *a);
return( *b >= 0 ? x : -x);
}

doublereal pow_di(doublereal *ap, integer *bp)
{
doublereal pow, x;
integer n;

pow = 1;
x = *ap;
n = *bp;

if(n != 0)
	{
	if(n < 0)
		{
		n = -n;
		x = 1/x;
		}
	for( ; ; )
		{
		if(n & 01)
			pow *= x;
		if(n >>= 1)
			x *= x;
		else
			break;
		}
	}
return(pow);
}

#undef abs
#include "math.h"
doublereal pow_dd(doublereal *ap, doublereal *bp)
{
return(pow(*ap, *bp) );
}

#ifdef __cplusplus
}
#endif
