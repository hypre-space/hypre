/* *****************************************************
 *
 *	File:  Hypre_MapStructVector_DataMembers.h
 *
 *********************************************************/

#ifndef Hypre_MapStructVector_Data_H
#define Hypre_MapStructVector_Data_H

#ifdef __cplusplus
extern "C" { // }
#endif 

#include "Hypre_StructGrid_Stub.h"

struct Hypre_MapStructVector_private_type
{
   Hypre_StructGrid grid;
   int* num_ghost;
}
;
#ifdef __cplusplus 
} // end extern "C"
#endif

#endif

