/* *****************************************************
 *
 *	File:  Hypre_StructMatrixBldr_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_StructMatrixBldr_DataMembers_
#define Hypre_StructMatrixBldr_DataMembers_

#include "Hypre_StructMatrix.h"

struct Hypre_StructMatrixBldr_private_type
{
   Hypre_StructMatrix newmat;
   /* ... the matrix currently under construction */
   int matgood;
   /* ... 1 if newmat is a valid matrix, 0 if not, still under construction */
};

#endif

