/******************************************************************************
 *
 * Header for ZZZ_struct_pcg
 *
 *****************************************************************************/

#ifndef _ZZZ_STRUCT_PCG_HEADER
#define _ZZZ_STRUCT_PCG_HEADER

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void Matrix;
typedef void Vector;

typedef struct
{
  void             *pc_data;
  zzz_StructMatrix *A;
} ZZZ_PCGPrecondData;

/*--------------------------------------------------------------------------
 * Accessor functions for the ZZZ_PCGPrecondData structure
 *--------------------------------------------------------------------------*/

#define ZZZ_PCGPrecondDataPCData(precond_data) ((precond_data) -> pc_data)
#define ZZZ_PCGPrecondDataMatrix(precond_data) ((precond_data) -> A)

#endif

