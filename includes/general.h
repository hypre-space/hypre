
#include <stdlib.h>

#ifdef HYPRE_DEBUG
#ifdef SOLARIS
#include <gmalloc.h>
#endif
#endif

/* Bring in CASC utilities */
#include "../utilities/memory.h"

/*--------------------------------------------------------------------------
 * Define max and min functions
 *--------------------------------------------------------------------------*/
 
#ifndef max
#define max(a,b)  (((a)<(b)) ? (b) : (a))
#endif
#ifndef min
#define min(a,b)  (((a)<(b)) ? (a) : (b))
#endif
 
#ifndef round
#define round(x)  ( ((x) < 0.0) ? ((int)(x - 0.5)) : ((int)(x + 0.5)) )
#endif
