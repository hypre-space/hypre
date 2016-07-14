
#include "_hypre_utilities.h"

/* This file will eventually contain functions needed to support
   a runtime decision of whether to use the assumed partition */


/* returns 1 if the assumed partition is in use */
HYPRE_Int HYPRE_AssumedPartitionCheck()
{
#ifdef HYPRE_NO_GLOBAL_PARTITION
   return 1;
#else
   return 0;
#endif

}

