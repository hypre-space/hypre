/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "sparse_msg.h"

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetupInterpOp
 *
 * Build interpolation operator P from base interpolation operator Q.
 *--------------------------------------------------------------------------*/

int
hypre_SparseMSGSetupInterpOp( hypre_StructMatrix *Q,
                              hypre_Index         findex,
                              hypre_Index         stride,
                              hypre_StructMatrix *P      )
{
   hypre_BoxArray        *compute_boxes;
   hypre_Box             *compute_box;
                        
   hypre_Box             *Q_data_box;
   hypre_Box             *P_data_box;
                        
   int                    Qi;
   int                    Pi;
                        
   double                *Qp0, *Qp1;
   double                *Pp0, *Pp1;
                        
   hypre_Index            loop_size;
   hypre_Index            start;
   hypre_IndexRef         startc;
   hypre_Index            stridec;
                        
   int                    i, loopi, loopj, loopk;

   int                    ierr = 0;

   /*----------------------------------------------------------
    * Initialize some things
    *----------------------------------------------------------*/

   hypre_SetIndex(stridec, 1, 1, 1);

   /*----------------------------------------------------------
    * Compute P
    *----------------------------------------------------------*/

   compute_boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(P));
   hypre_ForBoxI(i, compute_boxes)
      {
         compute_box = hypre_BoxArrayBox(compute_boxes, i);

         Q_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(Q), i);
         P_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), i);

         Qp0 = hypre_StructMatrixBoxData(Q, i, 0);
         Qp1 = hypre_StructMatrixBoxData(Q, i, 1);

         Pp0 = hypre_StructMatrixBoxData(P, i, 0);
         Pp1 = hypre_StructMatrixBoxData(P, i, 1);

         startc  = hypre_BoxIMin(compute_box);
         hypre_SparseMSGMapCoarseToFine(startc, findex, stride, start);

         hypre_GetStrideBoxSize(compute_box, stridec, loop_size);

         hypre_BoxLoop2Begin(loop_size,
                             Q_data_box, start,  stride,  Qi,
                             P_data_box, startc, stridec, Pi);
#define HYPRE_SMP_PRIVATE loopi,loopj,Qi,Pi
#include "hypre_smp_forloop.h"
         hypre_BoxLoop2For(loopi, loopj, loopk, Qi, Pi)
            {
               Pp0[Pi] = Qp0[Qi];
               Pp1[Pi] = Qp1[Qi];
            }
         hypre_BoxLoop2End(Qi, Pi);
      }

   hypre_AssembleStructMatrix(Q);

   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   return ierr;
}

