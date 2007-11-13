/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/




#include "HYPRE_struct_int.h"
#include "temp_multivector.h"
#include "_hypre_struct_ls.h"

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
HYPRE_StructSetupInterpreter( mv_InterfaceInterpreter *i )
{
  i->CreateVector = hypre_StructKrylovCreateVector;
  i->DestroyVector = hypre_StructKrylovDestroyVector; 
  i->InnerProd = hypre_StructKrylovInnerProd; 
  i->CopyVector = hypre_StructKrylovCopyVector;
  i->ClearVector = hypre_StructKrylovClearVector;
  i->SetRandomValues = hypre_StructSetRandomValues;
  i->ScaleVector = hypre_StructKrylovScaleVector;
  i->Axpy = hypre_StructKrylovAxpy;

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
HYPRE_StructSetupMatvec(HYPRE_MatvecFunctions * mv)
{
  mv->MatvecCreate = hypre_StructKrylovMatvecCreate;
  mv->Matvec = hypre_StructKrylovMatvec;
  mv->MatvecDestroy = hypre_StructKrylovMatvecDestroy;

  mv->MatMultiVecCreate = NULL;
  mv->MatMultiVec = NULL;
  mv->MatMultiVecDestroy = NULL;

  return 0;
}
