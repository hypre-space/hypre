/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
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
#include "smg.h"

/*--------------------------------------------------------------------------
 * hypre_SMGCreateInterpOp
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_SMGCreateInterpOp( hypre_StructMatrix *A,
                         hypre_StructGrid   *cgrid,
                         int                 cdir  )
{
   hypre_StructMatrix   *PT;

   hypre_StructStencil  *stencil;
   hypre_Index          *stencil_shape;
   int                   stencil_size;
   int                   stencil_dim;
                       
   int                   num_ghost[] = {1, 1, 1, 1, 1, 1};
                       
   int                   i;

   /* set up stencil */
   stencil_size = 2;
   stencil_dim = hypre_StructStencilDim(hypre_StructMatrixStencil(A));
   stencil_shape = hypre_CTAlloc(hypre_Index, stencil_size);
   for (i = 0; i < stencil_size; i++)
   {
      hypre_SetIndex(stencil_shape[i], 0, 0, 0);
   }
   hypre_IndexD(stencil_shape[0], cdir) = -1;
   hypre_IndexD(stencil_shape[1], cdir) =  1;
   stencil =
      hypre_StructStencilCreate(stencil_dim, stencil_size, stencil_shape);

   /* set up matrix */
   PT = hypre_StructMatrixCreate(hypre_StructMatrixComm(A), cgrid, stencil);
   hypre_StructMatrixSetNumGhost(PT, num_ghost);

   hypre_StructStencilDestroy(stencil);
 
   return PT;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetupInterpOp
 *
 *    This routine uses SMGRelax to set up the interpolation operator.
 *
 *    To illustrate how it proceeds, consider setting up the the {0, 0, -1}
 *    stencil coefficient of P^T.  This coefficient corresponds to the
 *    {0, 0, 1} coefficient of P.  Do one sweep of plane relaxation on the
 *    fine grid points for the system, A_mask x = b, with initial guess
 *    x_0 = all ones and right-hand-side b = all zeros.  The A_mask matrix
 *    contains all coefficients of A except for those in the same direction
 *    as {0, 0, -1}.
 *
 *    The relaxation data for the multigrid algorithm is passed in and used.
 *    When this routine returns, the only modified relaxation parameters
 *    are MaxIter, RegSpace and PreSpace info, the right-hand-side and
 *    solution info.
 *--------------------------------------------------------------------------*/

int
hypre_SMGSetupInterpOp( void               *relax_data,
                        hypre_StructMatrix *A,
                        hypre_StructVector *b,
                        hypre_StructVector *x,
                        hypre_StructMatrix *PT,
                        int                 cdir,
                        hypre_Index         cindex,
                        hypre_Index         findex,
                        hypre_Index         stride    )
{
   hypre_StructMatrix   *A_mask;

   hypre_StructStencil  *A_stencil;
   hypre_Index          *A_stencil_shape;
   int                   A_stencil_size;
   hypre_StructStencil  *PT_stencil;
   hypre_Index          *PT_stencil_shape;
   int                   PT_stencil_size;

   int                  *stencil_indices;
   int                   num_stencil_indices;

   hypre_StructGrid     *fgrid;

   hypre_StructStencil  *compute_pkg_stencil;
   hypre_Index          *compute_pkg_stencil_shape;
   int                   compute_pkg_stencil_size = 1;
   int                   compute_pkg_stencil_dim = 1;
   hypre_ComputePkg     *compute_pkg;
 
   hypre_BoxArrayArray  *send_boxes;
   hypre_BoxArrayArray  *recv_boxes;
   int                 **send_processes;
   int                 **recv_processes;
   hypre_BoxArrayArray  *indt_boxes;
   hypre_BoxArrayArray  *dept_boxes;
                     
   hypre_CommHandle     *comm_handle;
                     
   hypre_BoxArrayArray  *compute_box_aa;
   hypre_BoxArray       *compute_box_a;
   hypre_Box            *compute_box;
                     
   hypre_Box            *PT_data_box;
   hypre_Box            *x_data_box;
   double               *PTp;
   double               *xp;
   int                   PTi;
   int                   xi;

   hypre_Index           loop_size;
   hypre_Index           start;
   hypre_Index           startc;
   hypre_Index           stridec;
                      
   int                   si, sj, d;
   int                   compute_i, i, j;
   int                   loopi, loopj, loopk;
                        
   int                   ierr = 0;

   /*--------------------------------------------------------
    * Initialize some things
    *--------------------------------------------------------*/

   hypre_SetIndex(stridec, 1, 1, 1);

   fgrid = hypre_StructMatrixGrid(A);
   
   A_stencil = hypre_StructMatrixStencil(A);
   A_stencil_shape = hypre_StructStencilShape(A_stencil);
   A_stencil_size  = hypre_StructStencilSize(A_stencil);
   PT_stencil = hypre_StructMatrixStencil(PT);
   PT_stencil_shape = hypre_StructStencilShape(PT_stencil);
   PT_stencil_size  = hypre_StructStencilSize(PT_stencil);

   /* Set up relaxation parameters */
   hypre_SMGRelaxSetMaxIter(relax_data, 1);
   hypre_SMGRelaxSetNumPreSpaces(relax_data, 0);
   hypre_SMGRelaxSetNumRegSpaces(relax_data, 1);
   hypre_SMGRelaxSetRegSpaceRank(relax_data, 0, 1);

   compute_pkg_stencil_shape =
      hypre_CTAlloc(hypre_Index, compute_pkg_stencil_size);
   compute_pkg_stencil = hypre_StructStencilCreate(compute_pkg_stencil_dim,
                                                   compute_pkg_stencil_size,
                                                   compute_pkg_stencil_shape);

   for (si = 0; si < PT_stencil_size; si++)
   {
      /*-----------------------------------------------------
       * Compute A_mask matrix: This matrix contains all
       * stencil coefficients of A except for the coefficients
       * in the opposite direction of the current P stencil
       * coefficient being computed (same direction for P^T).
       *-----------------------------------------------------*/

      stencil_indices = hypre_TAlloc(int, A_stencil_size);
      num_stencil_indices = 0;
      for (sj = 0; sj < A_stencil_size; sj++)
      {
         if (hypre_IndexD(A_stencil_shape[sj],  cdir) !=
             hypre_IndexD(PT_stencil_shape[si], cdir)   )
         {
            stencil_indices[num_stencil_indices] = sj;
            num_stencil_indices++;
         }
      }
      A_mask =
         hypre_StructMatrixCreateMask(A, num_stencil_indices, stencil_indices);
      hypre_TFree(stencil_indices);

      /*-----------------------------------------------------
       * Do relaxation sweep to compute coefficients
       *-----------------------------------------------------*/

      hypre_StructVectorClearGhostValues(x);
      hypre_StructVectorSetConstantValues(x, 1.0);
      hypre_StructVectorSetConstantValues(b, 0.0);
      hypre_SMGRelaxSetNewMatrixStencil(relax_data, PT_stencil);
      hypre_SMGRelaxSetup(relax_data, A_mask, b, x);
      hypre_SMGRelax(relax_data, A_mask, b, x);

      /*-----------------------------------------------------
       * Free up A_mask matrix
       *-----------------------------------------------------*/

      hypre_StructMatrixDestroy(A_mask);

      /*-----------------------------------------------------
       * Set up compute package for communication of 
       * coefficients from fine to coarse across processor
       * boundaries.
       *-----------------------------------------------------*/

      hypre_CopyIndex(PT_stencil_shape[si], compute_pkg_stencil_shape[0]);
      hypre_CreateComputeInfo(fgrid, compute_pkg_stencil,
                              &send_boxes, &recv_boxes,
                              &send_processes, &recv_processes,
                              &indt_boxes, &dept_boxes);
 
      hypre_ProjectBoxArrayArray(send_boxes, findex, stride);
      hypre_ProjectBoxArrayArray(recv_boxes, findex, stride);
      hypre_ProjectBoxArrayArray(indt_boxes, cindex, stride);
      hypre_ProjectBoxArrayArray(dept_boxes, cindex, stride);
      hypre_ComputePkgCreate(send_boxes, recv_boxes,
                             stride, stride,
                             send_processes, recv_processes,
                             indt_boxes, dept_boxes,
                             stride, fgrid,
                             hypre_StructVectorDataSpace(x), 1,
                             &compute_pkg);

      /*-----------------------------------------------------
       * Copy coefficients from x into P^T
       *-----------------------------------------------------*/

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch(compute_i)
         {
            case 0:
            {
               xp = hypre_StructVectorData(x);
               hypre_InitializeIndtComputations(compute_pkg, xp, &comm_handle);
               compute_box_aa = hypre_ComputePkgIndtBoxes(compute_pkg);
            }
            break;

            case 1:
            {
               hypre_FinalizeIndtComputations(comm_handle);
               compute_box_aa = hypre_ComputePkgDeptBoxes(compute_pkg);
            }
            break;
         }

         hypre_ForBoxArrayI(i, compute_box_aa)
            {
               compute_box_a =
                  hypre_BoxArrayArrayBoxArray(compute_box_aa, i);

               x_data_box  =
                  hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
               PT_data_box =
                  hypre_BoxArrayBox(hypre_StructMatrixDataSpace(PT), i);
 
               xp  = hypre_StructVectorBoxData(x, i);
               PTp = hypre_StructMatrixBoxData(PT, i, si);

               hypre_ForBoxI(j, compute_box_a)
                  {
                     compute_box = hypre_BoxArrayBox(compute_box_a, j);

                     hypre_CopyIndex(hypre_BoxIMin(compute_box), start);
                     hypre_StructMapFineToCoarse(start, cindex, stride,
                                                 startc);

                     /* shift start index to appropriate F-point */
                     for (d = 0; d < 3; d++)
                     {
                        hypre_IndexD(start, d) +=
                           hypre_IndexD(PT_stencil_shape[si], d);
                     }

                     hypre_BoxGetStrideSize(compute_box, stride, loop_size);
                     hypre_BoxLoop2Begin(loop_size,
                                         x_data_box,  start,  stride,  xi,
                                         PT_data_box, startc, stridec, PTi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,xi,PTi
#include "hypre_box_smp_forloop.h"
                     hypre_BoxLoop2For(loopi, loopj, loopk, xi, PTi)
                        {
                           PTp[PTi] = xp[xi];
                        }
                     hypre_BoxLoop2End(xi, PTi);
                  }
            }
      }

      /*-----------------------------------------------------
       * Free up compute package info
       *-----------------------------------------------------*/

      hypre_ComputePkgDestroy(compute_pkg);
   }

   /* Tell SMGRelax that the stencil has changed */
   hypre_SMGRelaxSetNewMatrixStencil(relax_data, PT_stencil);

   hypre_StructStencilDestroy(compute_pkg_stencil);

   hypre_StructMatrixAssemble(PT);

   return ierr;
}

