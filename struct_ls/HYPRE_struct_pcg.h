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
  void               *pc_data;
  HYPRE_StructMatrix  A;

} HYPRE_PCGPrecondData;

/*--------------------------------------------------------------------------
 * Accessor functions for the HYPRE_PCGPrecondData structure
 *--------------------------------------------------------------------------*/

#define HYPRE_PCGPrecondDataPCData(precond_data) ((precond_data) -> pc_data)
#define HYPRE_PCGPrecondDataMatrix(precond_data) ((precond_data) -> A)

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_struct_pcg.c */
int HYPRE_Matvec P((double alpha , Matrix *A , Vector *x , double beta , Vector *y ));
double HYPRE_InnerProd P((Vector *x , Vector *y ));
int HYPRE_CopyVector P((Vector *x , Vector *y ));
int HYPRE_InitVector P((Vector *x , double value ));
int HYPRE_ScaleVector P((double alpha , Vector *x ));
int HYPRE_Axpy P((double alpha , Vector *x , Vector *y ));
void HYPRE_PCGSetup P((Matrix *vA , int (*HYPRE_PCGPrecond )(), void *precond_data , void *data ));
void HYPRE_PCGSMGPrecondSetup P((Matrix *vA , Vector *vb_l , Vector *vx_l , void *precond_vdata ));
int HYPRE_PCGSMGPrecond P((Vector *x , Vector *y , double dummy , void *precond_vdata ));
void HYPRE_FreePCGSMGData P((void *data ));
void HYPRE_PCGDiagScalePrecondSetup P((Matrix *vA , Vector *vb_l , Vector *vx_l , void *precond_vdata ));
int HYPRE_PCGDiagScalePrecond P((Vector *vx , Vector *vy , double dummy , void *precond_vdata ));
void HYPRE_FreePCGDiagScaleData P((void *data ));

#undef P

#endif

