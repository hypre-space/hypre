/* *****************************************************
 *
 *	File:  Hypre_ParCSRVectorBuilder_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_ParCSRVectorBuilder_DataMembers_
#define Hypre_ParCSRVectorBuilder_DataMembers_

/* jfp: this really belongs in a .h file... */
#ifndef array1int_
#define array1int_
typedef struct array1int{
	int upper[1];
	int lower[1];
	int *data;
}array1int;
#endif

#include "Hypre_ParCSRVector.h"
#include "Hypre_MPI_Com.h"

struct Hypre_ParCSRVectorBuilder_private_type
{
   Hypre_ParCSRVector newvec;
   /* ... the vector currently under construction */
   int vecgood;
   /* ... 1 if newvec is a valid vector, 0 if not, still under construction.
    GetConstructedObject will fail if vecgood=0, some set functions may
   fail if vecgood=1. */
   /* The following represent data provided in the construction process
      which need to be saved until the new vector build is finalized... */
   /* >>>> TO DO not needed????
   Hypre_MPI_Com com;
   int global_n;
   array1int partitioning;*/
}
;
#endif

