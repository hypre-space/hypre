/******************************************************************************
 *
 * Structured Matrix and Vector Preconditioned conjugate gradient (Omin) 
 * functions
 *
 *****************************************************************************/

#include "headers.h"
#include "HYPRE_struct_pcg.h"
#include "HYPRE_pcg.h"
#include "smg.h"                  /* this shouldn't be here */

/*--------------------------------------------------------------------------
 * HYPRE_Matvec
 *--------------------------------------------------------------------------*/

int
HYPRE_Matvec( double  alpha,
              Matrix *A,
              Vector *x,
              double  beta,
              Vector *y     )
{
   return (hypre_StructMatvec( alpha,
                               (hypre_StructMatrix *) A,
                               (hypre_StructVector *) x,
                               beta,
                               (hypre_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InnerProd
 *--------------------------------------------------------------------------*/

double
HYPRE_InnerProd( Vector *x, 
                 Vector *y )
{
   return (hypre_StructInnerProd( (hypre_StructVector *) x,
                                  (hypre_StructVector *) y ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_CopyVector
 *--------------------------------------------------------------------------*/

int
HYPRE_CopyVector( Vector *x, 
                  Vector *y )
{
   return (hypre_StructCopy( (hypre_StructVector *) x,
                             (hypre_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitVector
 *--------------------------------------------------------------------------*/

int
HYPRE_InitVector( Vector *x,
                  double value )
{
   return (hypre_SetStructVectorConstantValues( (hypre_StructVector *) x,
                                                value ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ScaleVector
 *--------------------------------------------------------------------------*/

int
HYPRE_ScaleVector( double  alpha,
                   Vector *x     )
{
   return (hypre_StructScale( alpha, (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_Axpy
 *--------------------------------------------------------------------------*/

int
HYPRE_Axpy( double alpha,
            Vector *x,
            Vector *y )
{
   return (hypre_StructAxpy( alpha, (hypre_StructVector *) x,
                             (hypre_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetup
 *--------------------------------------------------------------------------*/

void
HYPRE_PCGSetup( Matrix   *vA,
                int     (*HYPRE_PCGPrecond)(),
                void     *precond_data,
                void     *data                )
{
   hypre_StructMatrix *A = vA;
   hypre_StructVector *p;
   hypre_StructVector *s;
   hypre_StructVector *r;
   HYPRE_PCGData      *pcg_data = data;
  
   HYPRE_PCGDataA(pcg_data) = A;
  
   p = hypre_NewStructVector(hypre_StructMatrixComm(A),
                             hypre_StructMatrixGrid(A));
   HYPRE_InitializeStructVector(p);
   HYPRE_AssembleStructVector(p);
   HYPRE_PCGDataP(pcg_data) = p;

   s = hypre_NewStructVector(hypre_StructMatrixComm(A),
                             hypre_StructMatrixGrid(A));
   HYPRE_InitializeStructVector(s);
   HYPRE_AssembleStructVector(s);
   HYPRE_PCGDataS(pcg_data) = s;

   r = hypre_NewStructVector(hypre_StructMatrixComm(A),
                             hypre_StructMatrixGrid(A));
   HYPRE_InitializeStructVector(r);
   HYPRE_AssembleStructVector(r);
   HYPRE_PCGDataR(pcg_data) = r;
  
   HYPRE_PCGDataPrecond(pcg_data)     = HYPRE_PCGPrecond;
   HYPRE_PCGDataPrecondData(pcg_data) = precond_data;
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSMGPrecondSetup
 *--------------------------------------------------------------------------*/

void
HYPRE_PCGSMGPrecondSetup( Matrix   *vA,
                          Vector   *vb_l,
                          Vector   *vx_l,
                          void     *precond_vdata )
{
   hypre_StructMatrix     *A = vA;
   hypre_StructVector     *b_l = vb_l;
   hypre_StructVector     *x_l = vx_l;
   hypre_SMGData          *smg_data;
   HYPRE_PCGPrecondData   *precond_data = precond_vdata;
  
   /*-----------------------------------------------------------
    * Setup SMG as preconditioner
    *-----------------------------------------------------------*/

   smg_data = hypre_SMGInitialize(hypre_StructMatrixComm(A));
   hypre_SMGSetMemoryUse(smg_data, 0);
   hypre_SMGSetMaxIter(smg_data, 1);
   hypre_SMGSetTol(smg_data, 0.0);
   hypre_SMGSetLogging(smg_data, 1);
   hypre_SMGSetNumPreRelax(smg_data, 1);
   hypre_SMGSetNumPostRelax(smg_data, 1);
   hypre_SMGSetup(smg_data, A, b_l, x_l);

   /*-----------------------------------------------------------
    * Load values into precond_data structure
    *-----------------------------------------------------------*/

   HYPRE_PCGPrecondDataPCData(precond_data) = smg_data;
   HYPRE_PCGPrecondDataMatrix(precond_data) = A;

}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSMGPrecond
 *--------------------------------------------------------------------------*/

int 
HYPRE_PCGSMGPrecond( Vector *x, 
                     Vector *y, 
                     double dummy, 
                     void *precond_vdata )
{
   HYPRE_PCGPrecondData *precond_data = precond_vdata;
   hypre_SMGData        *smg_data;
   hypre_StructMatrix   *A;
   hypre_StructVector   *b_l;
   hypre_StructVector   *x_l;

   int               ierr;
  
   smg_data = (hypre_SMGData *) HYPRE_PCGPrecondDataPCData(precond_data);
   A = HYPRE_PCGPrecondDataMatrix(precond_data);
  
   b_l = (smg_data -> b_l)[0];
   x_l = (smg_data -> x_l)[0];

   hypre_StructCopy( (hypre_StructVector *)x, x_l );
   hypre_StructCopy( (hypre_StructVector *)y, b_l );

   ierr = hypre_SMGSolve( (void *) smg_data, (hypre_StructMatrix *) A,
                          b_l, x_l );

   hypre_StructCopy( x_l, (hypre_StructVector *)x );

#if 0
   {
      int    num_iterations, ierr;
      double rel_norm;

      HYPRE_SMGGetNumIterations(smg_data, &num_iterations);
      HYPRE_SMGGetFinalRelativeResidualNorm(smg_data, &rel_norm);
      printf("Iterations = %d Final Rel Norm = %e\n", num_iterations,
             rel_norm);
   }
#endif

   return ierr;
  
}

/*--------------------------------------------------------------------------
 * HYPRE_FreePCGSMGData
 *--------------------------------------------------------------------------*/

void
HYPRE_FreePCGSMGData( void *data )
{
   HYPRE_PCGData        *pcg_data = data;
   HYPRE_PCGPrecondData *precond_data;
   hypre_SMGData        *smg_data;
  
   if (pcg_data)
   {
      hypre_FreeStructVector(HYPRE_PCGDataP(pcg_data));
      hypre_FreeStructVector(HYPRE_PCGDataS(pcg_data));
      hypre_FreeStructVector(HYPRE_PCGDataR(pcg_data));
      precond_data = HYPRE_PCGDataPrecondData(pcg_data);
      smg_data = HYPRE_PCGPrecondDataPCData(precond_data);
      hypre_SMGFinalize(smg_data);
      hypre_TFree(precond_data);
      hypre_TFree(pcg_data);
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGDiagScalePrecondSetup
 *--------------------------------------------------------------------------*/

void
HYPRE_PCGDiagScalePrecondSetup( Matrix   *vA,
                                Vector   *vb_l,
                                Vector   *vx_l,
                                void     *precond_vdata )
{
   hypre_StructMatrix    *A = vA;
   HYPRE_PCGPrecondData  *precond_data = precond_vdata;
  
   /*-----------------------------------------------------------
    * Load values into precond_data structure
    *-----------------------------------------------------------*/

   HYPRE_PCGPrecondDataPCData(precond_data) = NULL;
   HYPRE_PCGPrecondDataMatrix(precond_data) = A;

}

/*--------------------------------------------------------------------------
 * HYPRE_PCGDiagScalePrecond
 *--------------------------------------------------------------------------*/

int 
HYPRE_PCGDiagScalePrecond( Vector *vx, 
                           Vector *vy, 
                           double  dummy, 
                           void   *precond_vdata )
{
   HYPRE_PCGPrecondData *precond_data = precond_vdata;
   hypre_StructVector   *x = vx;
   hypre_StructVector   *y = vy;
   hypre_StructMatrix   *A;

   hypre_BoxArray       *boxes;
   hypre_Box            *box;

   hypre_Box            *A_data_box;
   hypre_Box            *x_data_box;
   hypre_Box            *y_data_box;
                     
   double               *Ap;
   double               *xp;
   double               *yp;
                       
   int                   Ai;
   int                   xi;
   int                   yi;
                     
   hypre_Index           index;
   hypre_IndexRef        start;
   hypre_Index           stride;
   hypre_Index           loop_size;
                     
   int                   i;
   int                   loopi, loopj, loopk;

   int                   ierr;
  
   A = HYPRE_PCGPrecondDataMatrix(precond_data);

   /* x = D^{-1} y */
   hypre_SetIndex(stride, 1, 1, 1);
   boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_ForBoxI(i, boxes)
      {
         box = hypre_BoxArrayBox(boxes, i);

         A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
         x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

         hypre_SetIndex(index, 0, 0, 0);
         Ap = hypre_StructMatrixExtractPointerByIndex(A, i, index);
         xp = hypre_StructVectorBoxData(x, i);
         yp = hypre_StructVectorBoxData(y, i);

         start  = hypre_BoxIMin(box);

         hypre_GetBoxSize(box, loop_size);
         hypre_BoxLoop3(loopi, loopj, loopk, loop_size,
                        A_data_box,  start,  stride,  Ai,
                        x_data_box,  start,  stride,  xi,
                        y_data_box,  start,  stride,  yi,
                        {
                           xp[xi] = yp[yi] / Ap[Ai];
                        });
      }

   return ierr;
}


/*--------------------------------------------------------------------------
 * HYPRE_FreePCGDiagScaleData
 *--------------------------------------------------------------------------*/

void
HYPRE_FreePCGDiagScaleData( void *data )
{
   HYPRE_PCGData        *pcg_data = data;
   HYPRE_PCGPrecondData *precond_data;
  
   if (pcg_data)
   {
      hypre_FreeStructVector(HYPRE_PCGDataP(pcg_data));
      hypre_FreeStructVector(HYPRE_PCGDataS(pcg_data));
      hypre_FreeStructVector(HYPRE_PCGDataR(pcg_data));
      precond_data = HYPRE_PCGDataPrecondData(pcg_data);
      hypre_TFree(precond_data);
      hypre_TFree(pcg_data);
   }
}
