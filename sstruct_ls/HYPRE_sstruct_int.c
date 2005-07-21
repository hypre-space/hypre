#include "HYPRE_sstruct_int.h"
#include "sstruct_ls.h"
#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
#include "temp_multivector.h"

int 
hypre_StructVectorSetRandomValues( hypre_StructVector *vector,
                                   int seed )
{
   int    ierr = 0;

   hypre_Box          *v_data_box;
                    
   int                 vi;
   double             *vp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   int                 i;
   int                 loopi, loopj, loopk;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   srand( seed );

   hypre_SetIndex(unit_stride, 1, 1, 1);
 
   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   hypre_ForBoxI(i, boxes)
      {
         box      = hypre_BoxArrayBox(boxes, i);
         start = hypre_BoxIMin(box);

         v_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
         vp = hypre_StructVectorBoxData(vector, i);
 
         hypre_BoxGetSize(box, loop_size);

         hypre_BoxLoop1Begin(loop_size,
                             v_data_box, start, unit_stride, vi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,vi 
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop1For(loopi, loopj, loopk, vi)
            {
               vp[vi] = 2.0*rand()/RAND_MAX - 1.0;
            }
         hypre_BoxLoop1End(vi);
      }

   return ierr;
}

int 
hypre_SStructPVectorSetRandomValues( hypre_SStructPVector *pvector, int seed )
{
   int ierr = 0;
   int                 nvars = hypre_SStructPVectorNVars(pvector);
   hypre_StructVector *svector;
   int                 var;

   srand( seed );

   for (var = 0; var < nvars; var++)
   {
      svector = hypre_SStructPVectorSVector(pvector, var);
	  seed = rand();
      hypre_StructVectorSetRandomValues(svector, seed);
   }

   return ierr;
}

int 
hypre_SStructVectorSetRandomValues( hypre_SStructVector *vector, int seed )
{
   int ierr = 0;
   int                   nparts = hypre_SStructVectorNParts(vector);
   hypre_SStructPVector *pvector;
   int                   part;

   srand( seed );

   for (part = 0; part < nparts; part++)
   {
      pvector = hypre_SStructVectorPVector(vector, part);
	  seed = rand();
      hypre_SStructPVectorSetRandomValues(pvector, seed);
   }

   return ierr;
}

int
hypre_SStructSetRandomValues( void* v, int seed ) {

  return hypre_SStructVectorSetRandomValues( (hypre_SStructVector*)v, seed );
}

int
HYPRE_SStructSetupInterpreter( mv_InterfaceInterpreter *i )
{
  i->CreateVector = hypre_SStructKrylovCreateVector;
  i->DestroyVector = hypre_SStructKrylovDestroyVector; 
  i->InnerProd = hypre_SStructKrylovInnerProd; 
  i->CopyVector = hypre_SStructKrylovCopyVector;
  i->ClearVector = hypre_SStructKrylovClearVector;
  i->SetRandomValues = hypre_SStructSetRandomValues;
  i->ScaleVector = hypre_SStructKrylovScaleVector;
  i->Axpy = hypre_SStructKrylovAxpy;

  i->CreateMultiVector = mv_TempMultiVectorCreateFromSampleVector;
  i->CopyCreateMultiVector = mv_TempMultiVectorCreateCopy;
  i->DestroyMultiVector = mv_TempMultiVectorDestroy;

  i->Width = mv_TempMultiVectorWidth;
  i->Height = mv_TempMultiVectorHeight;
  i->SetMask = mv_TempMultiVectorSetMask;
  i->CopyMultiVector = mv_TempMultiVectorCopy;
  i->ClearMultiVector = mv_TempMultiVectorClear;
  i->SetRandomVectors = mv_TempMultiVectorSetRandom;
  i->MultiInnerProd = mv_TempMultiVectorByMultiVector;
  i->MultiInnerProdDiag = mv_TempMultiVectorByMultiVectorDiag;
  i->MultiVecMat = mv_TempMultiVectorByMatrix;
  i->MultiVecMatDiag = mv_TempMultiVectorByDiagonal;
  i->MultiAxpy = mv_TempMultiVectorAxpy;
  i->MultiXapy = mv_TempMultiVectorXapy;
  i->Eval = mv_TempMultiVectorEval;

  return 0;
}

int
HYPRE_SStructSetupMatvec(HYPRE_MatvecFunctions * mv)
{
  mv->MatvecCreate = hypre_SStructKrylovMatvecCreate;
  mv->Matvec = hypre_SStructKrylovMatvec; 
  mv->MatvecDestroy = hypre_SStructKrylovMatvecDestroy;

  mv->MatMultiVecCreate = NULL;
  mv->MatMultiVec = NULL;
  mv->MatMultiVecDestroy = NULL;

  return 0;
}
