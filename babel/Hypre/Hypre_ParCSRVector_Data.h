/* *****************************************************
 *
 *	File:  Hypre_ParCSRVector_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_ParCSRVector_DataMembers_
#define Hypre_ParCSRVector_DataMembers_

#include "HYPRE_IJ_mv.h"
#include "Hypre_MPI_Com_IOR.h"

struct Hypre_ParCSRVector_private_type
{
   HYPRE_IJVector *Hvec;
   /* We need to save constructor inputs to support a Clone function: */
   Hypre_MPI_Com comm;
}
;
#endif

