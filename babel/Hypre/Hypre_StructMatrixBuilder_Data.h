/* *****************************************************
 *
 *	File:  Hypre_StructMatrixBuilder_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_StructMatrixBuilder_DataMembers_
#define Hypre_StructMatrixBuilder_DataMembers_

#include "Hypre_StructMatrix_IOR.h"

struct Hypre_StructMatrixBuilder_private_type
{
   Hypre_StructMatrix newmat;
   /* ... the matrix currently under construction */
   int matgood;
   /* ... 1 if newmat is a valid matrix, 0 if not, still under construction */
}
;
#endif

