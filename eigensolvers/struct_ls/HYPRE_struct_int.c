#include "HYPRE_struct_int.h"

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
hypre_StructSetRandomValues( void* v, int seed ) {

  return hypre_StructVectorSetRandomValues( (hypre_StructVector*)v, seed );
}

int
HYPRE_StructSetupInterpreter( HYPRE_InterfaceInterpreter *i )
{
  i->CAlloc = hypre_CAlloc;
  i->Free = hypre_StructKrylovFree;
  i->CommInfo = hypre_StructKrylovCommInfo;
  i->CreateVector = hypre_StructKrylovCreateVector;
  i->DestroyVector = hypre_StructKrylovDestroyVector; 
  i->MatvecCreate = hypre_StructKrylovMatvecCreate;
  i->Matvec = hypre_StructKrylovMatvec; 
  i->MatvecDestroy = hypre_StructKrylovMatvecDestroy;
  i->InnerProd = hypre_StructKrylovInnerProd; 
  i->CopyVector = hypre_StructKrylovCopyVector;
  i->ClearVector = hypre_StructKrylovClearVector;
  i->SetRandomValues = hypre_StructSetRandomValues;
  i->ScaleVector = hypre_StructKrylovScaleVector;
  i->Axpy = hypre_StructKrylovAxpy;
  i->PrintVector = NULL;
  i->ReadVector = NULL;

  return 0;
}

