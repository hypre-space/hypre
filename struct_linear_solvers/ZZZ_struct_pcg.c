/******************************************************************************
 *
 * Structured Matrix and Vector Preconditioned conjugate gradient (Omin) 
 * functions
 *
 *****************************************************************************/

#include "headers.h"
#include "ZZZ_struct_pcg.h"
#include "ZZZ_pcg.h"
#include "smg.h"                  /* this shouldn't be here */

/*--------------------------------------------------------------------------
 * ZZZ_Matvec
 *--------------------------------------------------------------------------*/

int
ZZZ_Matvec( double  alpha,
	    Matrix *A,
	    Vector *x,
	    double  beta,
	    Vector *y )
{
  return (zzz_StructMatvec( alpha,
			    (zzz_StructMatrix *) A,
			    (zzz_StructVector *) x,
			    beta,
			    (zzz_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_InnerProd
 *--------------------------------------------------------------------------*/

double
ZZZ_InnerProd( Vector *x, 
	       Vector *y )
{
  return (zzz_StructInnerProd( (zzz_StructVector *) x,
			       (zzz_StructVector *) y ) );
}


/*--------------------------------------------------------------------------
 * ZZZ_CopyVector
 *--------------------------------------------------------------------------*/

int
ZZZ_CopyVector( Vector *x, 
		Vector *y )
{
  return (zzz_StructCopy( (zzz_StructVector *)x, (zzz_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_InitVector
 *--------------------------------------------------------------------------*/

int
ZZZ_InitVector( Vector *x,
		double value )
{
  return (zzz_SetStructVectorConstantValues( (zzz_StructVector *) x,
					     value ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_ScaleVector
 *--------------------------------------------------------------------------*/

int
ZZZ_ScaleVector( double alpha,
		 Vector *x )
{
  return (zzz_StructScale( alpha, (zzz_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_Axpy
 *--------------------------------------------------------------------------*/

int
ZZZ_Axpy( double alpha,
	  Vector *x,
	  Vector *y )
{
  return (zzz_StructAxpy( alpha, (zzz_StructVector *) x,
			  (zzz_StructVector *) y ) );
}

/*--------------------------------------------------------------------------
 * ZZZ_PCGSetup
 *--------------------------------------------------------------------------*/

void
ZZZ_PCGSetup( Matrix   *vA,
              int     (*ZZZ_PCGPrecond)(),
              void     *precond_data,
              void     *data )
{
  zzz_StructMatrix *A = vA;
  zzz_StructVector *p;
  zzz_StructVector *s;
  zzz_StructVector *r;
  ZZZ_PCGData      *pcg_data = data;
  
  ZZZ_PCGDataA(pcg_data) = A;
  
  p = zzz_NewStructVector(zzz_StructMatrixComm(A),
			  zzz_StructMatrixGrid(A));
  ZZZ_InitializeStructVector(p);
  ZZZ_AssembleStructVector(p);
  ZZZ_PCGDataP(pcg_data) = p;

  s = zzz_NewStructVector(zzz_StructMatrixComm(A),
			  zzz_StructMatrixGrid(A));
  ZZZ_InitializeStructVector(s);
  ZZZ_AssembleStructVector(s);
  ZZZ_PCGDataS(pcg_data) = s;

  r = zzz_NewStructVector(zzz_StructMatrixComm(A),
			  zzz_StructMatrixGrid(A));
  ZZZ_InitializeStructVector(r);
  ZZZ_AssembleStructVector(r);
  ZZZ_PCGDataR(pcg_data) = r;
  
  ZZZ_PCGDataPrecond(pcg_data)     = ZZZ_PCGPrecond;
  ZZZ_PCGDataPrecondData(pcg_data) = precond_data;
}

/*--------------------------------------------------------------------------
 * ZZZ_PCGSMGPrecondSetup
 *--------------------------------------------------------------------------*/

void
ZZZ_PCGSMGPrecondSetup( Matrix   *vA,
			Vector   *vb_l,
			Vector   *vx_l,
			void     *precond_vdata )
{
  zzz_StructMatrix     *A = vA;
  zzz_StructVector     *b_l = vb_l;
  zzz_StructVector     *x_l = vx_l;
  zzz_SMGData          *smg_data;
  ZZZ_PCGPrecondData   *precond_data = precond_vdata;
  
   /*-----------------------------------------------------------
    * Setup SMG as preconditioner
    *-----------------------------------------------------------*/

   smg_data = zzz_SMGInitialize(zzz_StructMatrixComm(A));
   zzz_SMGSetMemoryUse(smg_data, 0);
   zzz_SMGSetMaxIter(smg_data, 1);
   zzz_SMGSetTol(smg_data, 0.0);
   zzz_SMGSetLogging(smg_data, 1);
   zzz_SMGSetNumPreRelax(smg_data, 1);
   zzz_SMGSetNumPostRelax(smg_data, 1);
   zzz_SMGSetup(smg_data, A, b_l, x_l);

   /*-----------------------------------------------------------
    * Load values into precond_data structure
    *-----------------------------------------------------------*/

   ZZZ_PCGPrecondDataPCData(precond_data) = smg_data;
   ZZZ_PCGPrecondDataMatrix(precond_data) = A;

}

/*--------------------------------------------------------------------------
 * ZZZ_PCGSMGPrecond
 *--------------------------------------------------------------------------*/

int 
ZZZ_PCGSMGPrecond( Vector *x, 
		Vector *y, 
		double dummy, 
		void *precond_vdata )
{
  ZZZ_PCGPrecondData *precond_data = precond_vdata;
  zzz_SMGData      *smg_data;
  zzz_StructMatrix *A;
  zzz_StructVector *b_l;
  zzz_StructVector *x_l;

  int               ierr;
  
  smg_data = (zzz_SMGData *) ZZZ_PCGPrecondDataPCData(precond_data);
  A = ZZZ_PCGPrecondDataMatrix(precond_data);
  
  b_l = (smg_data -> b_l)[0];
  x_l = (smg_data -> x_l)[0];

  zzz_StructCopy( (zzz_StructVector *)x, x_l );
  zzz_StructCopy( (zzz_StructVector *)y, b_l );

  ierr = zzz_SMGSolve( (void *) smg_data, (zzz_StructMatrix *) A,
		       b_l, x_l );

  zzz_StructCopy( x_l, (zzz_StructVector *)x );

#if 0
{
  int    num_iterations, ierr;
  double rel_norm;

  ZZZ_SMGGetNumIterations(smg_data, &num_iterations);
  ZZZ_SMGGetFinalRelativeResidualNorm(smg_data, &rel_norm);
  printf("Iterations = %d Final Rel Norm = %e\n", num_iterations,
	 rel_norm);
}
#endif

  return ierr;
  
}

/*--------------------------------------------------------------------------
 * ZZZ_FreePCGSMGData
 *--------------------------------------------------------------------------*/

void
ZZZ_FreePCGSMGData( void *data )
{
  ZZZ_PCGData        *pcg_data = data;
  ZZZ_PCGPrecondData *precond_data;
  zzz_SMGData      *smg_data;
  
  if (pcg_data)
    {
      zzz_FreeStructVector(ZZZ_PCGDataP(pcg_data));
      zzz_FreeStructVector(ZZZ_PCGDataS(pcg_data));
      zzz_FreeStructVector(ZZZ_PCGDataR(pcg_data));
      precond_data = ZZZ_PCGDataPrecondData(pcg_data);
      smg_data = ZZZ_PCGPrecondDataPCData(precond_data);
      zzz_SMGFinalize(smg_data);
      zzz_TFree(precond_data);
      zzz_TFree(pcg_data);
    }
}

/*--------------------------------------------------------------------------
 * ZZZ_PCGDiagScalePrecondSetup
 *--------------------------------------------------------------------------*/

void
ZZZ_PCGDiagScalePrecondSetup( Matrix   *vA,
                              Vector   *vb_l,
                              Vector   *vx_l,
                              void     *precond_vdata )
{
  zzz_StructMatrix    *A = vA;
  ZZZ_PCGPrecondData  *precond_data = precond_vdata;
  
   /*-----------------------------------------------------------
    * Load values into precond_data structure
    *-----------------------------------------------------------*/

   ZZZ_PCGPrecondDataPCData(precond_data) = NULL;
   ZZZ_PCGPrecondDataMatrix(precond_data) = A;

}

/*--------------------------------------------------------------------------
 * ZZZ_PCGDiagScalePrecond
 *--------------------------------------------------------------------------*/

int 
ZZZ_PCGDiagScalePrecond( Vector *vx, 
                         Vector *vy, 
                         double  dummy, 
                         void   *precond_vdata )
{
   ZZZ_PCGPrecondData *precond_data = precond_vdata;
   zzz_StructVector   *x = vx;
   zzz_StructVector   *y = vy;
   zzz_StructMatrix   *A;

   zzz_BoxArray       *boxes;
   zzz_Box            *box;

   zzz_Box            *A_data_box;
   zzz_Box            *x_data_box;
   zzz_Box            *y_data_box;
                     
   double             *Ap;
   double             *xp;
   double             *yp;
                     
   int                 Ai;
   int                 xi;
   int                 yi;
                     
   zzz_Index           index;
   zzz_IndexRef        start;
   zzz_Index           stride;
   zzz_Index           loop_size;
                     
   int                 i;
   int                 loopi, loopj, loopk;

   int               ierr;
  
   A = ZZZ_PCGPrecondDataMatrix(precond_data);

   /* x = D^{-1} y */
   zzz_SetIndex(stride, 1, 1, 1);
   boxes = zzz_StructGridBoxes(zzz_StructMatrixGrid(A));
   zzz_ForBoxI(i, boxes)
   {
      box = zzz_BoxArrayBox(boxes, i);

      A_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(A), i);
      x_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(x), i);
      y_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(y), i);

      zzz_SetIndex(index, 0, 0, 0);
      Ap = zzz_StructMatrixExtractPointerByIndex(A, i, index);
      xp = zzz_StructVectorBoxData(x, i);
      yp = zzz_StructVectorBoxData(y, i);

      start  = zzz_BoxIMin(box);

      zzz_GetBoxSize(box, loop_size);
      zzz_BoxLoop3(loopi, loopj, loopk, loop_size,
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
 * ZZZ_FreePCGDiagScaleData
 *--------------------------------------------------------------------------*/

void
ZZZ_FreePCGDiagScaleData( void *data )
{
  ZZZ_PCGData        *pcg_data = data;
  ZZZ_PCGPrecondData *precond_data;
  
  if (pcg_data)
  {
     zzz_FreeStructVector(ZZZ_PCGDataP(pcg_data));
     zzz_FreeStructVector(ZZZ_PCGDataS(pcg_data));
     zzz_FreeStructVector(ZZZ_PCGDataR(pcg_data));
     precond_data = ZZZ_PCGDataPrecondData(pcg_data);
     zzz_TFree(precond_data);
     zzz_TFree(pcg_data);
  }
}
