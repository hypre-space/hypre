#include "f2c.h"
#include "hypre_blas.h"

/* assign strings:  a = b */

#ifdef KR_headers
VOID s_copy(a, b, la, lb) char *a, *b; ftnlen la, lb;
#else
void s_copy(char *a, char *b, ftnlen la, ftnlen lb)
#endif
{
register char *aend, *bend;

aend = a + la;

if(la <= lb)
	while(a < aend)
		*a++ = *b++;

else
	{
	bend = b + lb;
	while(b < bend)
		*a++ = *b++;
	while(a < aend)
		*a++ = ' ';
	}
}
