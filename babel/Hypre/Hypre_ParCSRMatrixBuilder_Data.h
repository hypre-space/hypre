/* *****************************************************
 *
 *	File:  Hypre_ParCSRMatrixBuilder_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_ParCSRMatrixBuilder_DataMembers_
#define Hypre_ParCSRMatrixBuilder_DataMembers_

#include "Hypre_ParCSRMatrix.h"

struct Hypre_ParCSRMatrixBuilder_private_type
{
   Hypre_ParCSRMatrix newmat;
   /* ... the matrix currently under construction */
   int matgood;
   /* ... 1 if newmat is a valid matrix, 0 if not, still under construction.
    GetConstructedObject will fail if matgood=0, some set functions may
   fail if matgood=1. */
}
;
#endif

