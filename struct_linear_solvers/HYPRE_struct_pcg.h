/******************************************************************************
 *
 * Header for HYPRE_struct_pcg
 *
 *****************************************************************************/

#ifndef HYPRE_STRUCT_PCG_HEADER
#define HYPRE_STRUCT_PCG_HEADER

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void Matrix;
typedef void Vector;

typedef struct
{
  void             *pc_data;
  hypre_StructMatrix *A;
} HYPRE_PCGPrecondData;

/*--------------------------------------------------------------------------
 * Accessor functions for the HYPRE_PCGPrecondData structure
 *--------------------------------------------------------------------------*/

#define HYPRE_PCGPrecondDataPCData(precond_data) ((precond_data) -> pc_data)
#define HYPRE_PCGPrecondDataMatrix(precond_data) ((precond_data) -> A)

#endif

