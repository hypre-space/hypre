/* *****************************************************
 *
 *	File:  Hypre_ParCSRVectorBuilder_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_ParCSRVectorBuilder_DataMembers_
#define Hypre_ParCSRVectorBuilder_DataMembers_

#include "Hypre_ParCSRVector_IOR.h"

struct Hypre_ParCSRVectorBuilder_private_type
{
   Hypre_ParCSRVector newvec;
   /* ... the vector currently under construction */
   int vecgood;
   /* ... 1 if newvec is a valid vector, 0 if not, still under construction.
    GetConstructedObject will fail if vecgood=0, some set functions may
   fail if vecgood=1. */
}
;
#endif

