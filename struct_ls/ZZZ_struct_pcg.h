/******************************************************************************
 *
 * Header for ZZZ_struct_pcg
 *
 *****************************************************************************/

#include "headers.h"
#include "smg.h"

#ifndef _ZZZ_STRUCT_PCG_HEADER
#define _ZZZ_STRUCT_PCG_HEADER

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef void Matrix;
typedef void Vector;

typedef struct
{
  zzz_SMGData      *smg_data;
  zzz_StructMatrix *A;
} ZZZ_PCGPrecondData;

/*--------------------------------------------------------------------------
 * Accessor functions for the ZZZ_PCG_PrecondData structure
 *--------------------------------------------------------------------------*/

#define ZZZ_PCGPrecondDataSMGData(precond_data) ((precond_data) -> smg_data)
#define ZZZ_PCGPrecondDataMatrix(precond_data)  ((precond_data) -> A)

#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

int ZZZ_Matvec( double alpha, Matrix *A, Vector *x, double  beta, Vector *y );
double ZZZ_InnerProd( Vector *x, Vector *y );
int ZZZ_CopyVector( Vector *x, Vector *y );
int ZZZ_InitVector( Vector *x, double value );
int ZZZ_ScaleVector( double alpha, Vector *x );
int ZZZ_Axpy( double alpha, Vector *x, Vector *y );
void ZZZ_PCGSMGSetup( Matrix *vA, int (*precond)(), void *precond_data, void *data );
void ZZZ_PCGSMGPrecondSetup( Matrix *vA, Vector *vb_l, Vector *vx_l, void *precond_vdata );
int ZZZ_PCGSMGPrecond( Vector *x, Vector *y, double dummy, void *precond_vdata );
void ZZZ_FreePCGSMGData( void *data );
