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
  void             *pc_data;
  zzz_StructMatrix *A;

} ZZZ_PCGPrecondData;

/*--------------------------------------------------------------------------
 * Accessor functions for the ZZZ_PCGPrecondData structure
 *--------------------------------------------------------------------------*/

#define ZZZ_PCGPrecondDataPCData(precond_data) ((precond_data) -> pc_data)
#define ZZZ_PCGPrecondDataMatrix(precond_data) ((precond_data) -> A)

#endif

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define        P(s) s
#else
# define P(s) ()
#endif
 
 
/* ZZZ_struct_pcg.c */
int ZZZ_Matvec P((double alpha , Matrix *A , Vector *x , double beta , Vector *y ));
double ZZZ_InnerProd P((Vector *x , Vector *y ));
int ZZZ_CopyVector P((Vector *x , Vector *y ));
int ZZZ_InitVector P((Vector *x , double value ));
int ZZZ_ScaleVector P((double alpha , Vector *x ));
int ZZZ_Axpy P((double alpha , Vector *x , Vector *y ));
void ZZZ_PCGSetup P((Matrix *vA , int (*ZZZ_PCGPrecond )(), void *precond_data , void *data ));
void ZZZ_PCGSMGPrecondSetup P((Matrix *vA , Vector *vb_l , Vector *vx_l , void *precond_vdata ));
int ZZZ_PCGSMGPrecond P((Vector *x , Vector *y , double dummy , void *precond_vdata ));
void ZZZ_FreePCGSMGData P((void *data ));
void ZZZ_PCGDiagScalePrecondSetup P((Matrix *vA , Vector *vb_l , Vector *vx_l , void *precond_vdata ));
int ZZZ_PCGDiagScalePrecond P((Vector *x , Vector *y , double dummy , void *precond_vdata ));
void ZZZ_FreePCGDiagScaleData P((void *data ));
 
#undef P
