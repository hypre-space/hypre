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
 * zzz_SMGNewInterpOp
 *--------------------------------------------------------------------------*/

zzz_StructMatrix *
zzz_SMGNewInterpOp( zzz_StructMatrix *A,
                    zzz_StructGrid   *cgrid,
                    int               cdir  )
{
   zzz_StructMatrix   *PT;

   zzz_StructStencil  *stencil;
   zzz_Index          *stencil_shape;
   int                 stencil_size;
   int                 stencil_dim;

   int                 num_ghost[] = {1, 1, 1, 1, 1, 1};

   int                 i;

   /* set up stencil */
   stencil_size = 2;
   stencil_dim = zzz_StructStencilDim(zzz_StructMatrixStencil(A));
   stencil_shape = zzz_CTAlloc(zzz_Index, stencil_size);
   for (i = 0; i < stencil_size; i++)
   {
      zzz_SetIndex(stencil_shape[i], 0, 0, 0);
   }
   zzz_IndexD(stencil_shape[0], cdir) = -1;
   zzz_IndexD(stencil_shape[1], cdir) =  1;
   stencil = zzz_NewStructStencil(stencil_dim, stencil_size, stencil_shape);

   /* set up matrix */
   PT = zzz_NewStructMatrix(zzz_StructMatrixComm(A), cgrid, stencil);
   zzz_SetStructMatrixNumGhost(PT, num_ghost);
   zzz_InitializeStructMatrix(PT);
 
   return PT;
}

/*--------------------------------------------------------------------------
 * zzz_SMGSetupInterpOp
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
zzz_SMGSetupInterpOp( void             *relax_data,
                      zzz_StructMatrix *A,
                      zzz_StructVector *b,
                      zzz_StructVector *x,
                      zzz_StructMatrix *PT,
                      int               cdir,
                      zzz_Index         cindex,
                      zzz_Index         cstride,
                      zzz_Index         findex,
                      zzz_Index         fstride    )
{
   zzz_StructMatrix   *A_mask;

   zzz_StructStencil  *A_stencil;
   zzz_Index          *A_stencil_shape;
   int                 A_stencil_size;
   zzz_StructStencil  *PT_stencil;
   zzz_Index          *PT_stencil_shape;
   int                 PT_stencil_size;

   int                *stencil_indices;
   int                 num_stencil_indices;

   zzz_StructGrid     *fgrid;

   zzz_StructStencil  *compute_pkg_stencil;
   zzz_Index          *compute_pkg_stencil_shape;
   int                 compute_pkg_stencil_size = 1;
   int                 compute_pkg_stencil_dim = 1;
   zzz_ComputePkg     *compute_pkg;
 
   zzz_BoxArrayArray  *send_boxes;
   zzz_BoxArrayArray  *recv_boxes;
   int               **send_box_ranks;
   int               **recv_box_ranks;
   zzz_BoxArrayArray  *indt_boxes;
   zzz_BoxArrayArray  *dept_boxes;
                     
   zzz_SBoxArrayArray *send_sboxes;
   zzz_SBoxArrayArray *recv_sboxes;
   zzz_SBoxArrayArray *indt_sboxes;
   zzz_SBoxArrayArray *dept_sboxes;

   zzz_CommHandle     *comm_handle;
                     
   zzz_SBoxArrayArray *compute_sbox_aa;
   zzz_SBoxArray      *compute_sbox_a;
   zzz_SBox           *compute_sbox;
                     
   zzz_Box            *PT_data_box;
   zzz_Box            *x_data_box;
   double             *PTp;
   double             *xp;
   int                 PTi;
   int                 xi;

   zzz_Index           loop_size;
   zzz_Index           start;
   zzz_Index           startc;
   zzz_IndexRef        stride;
   zzz_Index           stridec;
                      
   int                 si, sj, d;
   int                 compute_i, i, j;
   int                 loopi, loopj, loopk;
                      
   int                 ierr;

   /*--------------------------------------------------------
    * Initialize some things
    *--------------------------------------------------------*/

   zzz_SetIndex(stridec, 1, 1, 1);

   fgrid = zzz_StructMatrixGrid(A);
   
   A_stencil = zzz_StructMatrixStencil(A);
   A_stencil_shape = zzz_StructStencilShape(A_stencil);
   A_stencil_size  = zzz_StructStencilSize(A_stencil);
   PT_stencil = zzz_StructMatrixStencil(PT);
   PT_stencil_shape = zzz_StructStencilShape(PT_stencil);
   PT_stencil_size  = zzz_StructStencilSize(PT_stencil);

   /* Set up relaxation parameters */
   zzz_SMGRelaxSetMaxIter(relax_data, 1);
   zzz_SMGRelaxSetNumPreSpaces(relax_data, 0);
   zzz_SMGRelaxSetNumRegSpaces(relax_data, 1);
   zzz_SMGRelaxSetRegSpaceRank(relax_data, 0, 1);

   compute_pkg_stencil_shape =
      zzz_CTAlloc(zzz_Index, compute_pkg_stencil_size);
   compute_pkg_stencil = zzz_NewStructStencil(compute_pkg_stencil_dim,
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

      stencil_indices = zzz_TAlloc(int, A_stencil_size);
      num_stencil_indices = 0;
      for (sj = 0; sj < A_stencil_size; sj++)
      {
         if (zzz_IndexD(A_stencil_shape[sj],  cdir) !=
             zzz_IndexD(PT_stencil_shape[si], cdir)   )
         {
            stencil_indices[num_stencil_indices] = sj;
            num_stencil_indices++;
         }
      }
      A_mask =
         zzz_NewStructMatrixMask(A, num_stencil_indices, stencil_indices);
      zzz_TFree(stencil_indices);

      /*-----------------------------------------------------
       * Do relaxation sweep to compute coefficients
       *-----------------------------------------------------*/

      zzz_ClearStructVectorGhostValues(x);
      zzz_SetStructVectorConstantValues(x, 1.0);
      zzz_SetStructVectorConstantValues(b, 0.0);
      zzz_SMGRelaxSetNewMatrixStencil(relax_data, PT_stencil);
      zzz_SMGRelaxSetup(relax_data, A_mask, b, x);
      zzz_SMGRelax(relax_data, A_mask, b, x);

      /*-----------------------------------------------------
       * Free up A_mask matrix
       *-----------------------------------------------------*/

      zzz_FreeStructMatrixMask(A_mask);

      /*-----------------------------------------------------
       * Set up compute package for communication of 
       * coefficients from fine to coarse across processor
       * boundaries.
       *-----------------------------------------------------*/

      zzz_CopyIndex(PT_stencil_shape[si], compute_pkg_stencil_shape[0]);
      zzz_GetComputeInfo(&send_boxes, &recv_boxes,
                         &send_box_ranks, &recv_box_ranks,
                         &indt_boxes, &dept_boxes,
                         fgrid, compute_pkg_stencil);
 
      send_sboxes = zzz_ProjectBoxArrayArray(send_boxes, findex, fstride);
      recv_sboxes = zzz_ProjectBoxArrayArray(recv_boxes, findex, fstride);
      indt_sboxes = zzz_ProjectBoxArrayArray(indt_boxes, cindex, cstride);
      dept_sboxes = zzz_ProjectBoxArrayArray(dept_boxes, cindex, cstride);
      compute_pkg =
         zzz_NewComputePkg(send_sboxes, recv_sboxes,
                           send_box_ranks, recv_box_ranks,
                           indt_sboxes, dept_sboxes,
                           fgrid, zzz_StructVectorDataSpace(x), 1);

      zzz_FreeBoxArrayArray(send_boxes);
      zzz_FreeBoxArrayArray(recv_boxes);
      zzz_FreeBoxArrayArray(indt_boxes);
      zzz_FreeBoxArrayArray(dept_boxes);

      /*-----------------------------------------------------
       * Copy coefficients from x into P^T
       *-----------------------------------------------------*/

      for (compute_i = 0; compute_i < 2; compute_i++)
      {
         switch(compute_i)
         {
            case 0:
            {
               xp = zzz_StructVectorData(x);
               comm_handle = zzz_InitializeIndtComputations(compute_pkg, xp);
               compute_sbox_aa = zzz_ComputePkgIndtSBoxes(compute_pkg);
            }
            break;

            case 1:
            {
               zzz_FinalizeIndtComputations(comm_handle);
               compute_sbox_aa = zzz_ComputePkgDeptSBoxes(compute_pkg);
            }
            break;
         }

         zzz_ForSBoxArrayI(i, compute_sbox_aa)
         {
            compute_sbox_a = zzz_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

            x_data_box  = zzz_BoxArrayBox(zzz_StructVectorDataSpace(x), i);
            PT_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(PT), i);
 
            xp  = zzz_StructVectorBoxData(x, i);
            PTp = zzz_StructMatrixBoxData(PT, i, si);

            zzz_ForSBoxI(j, compute_sbox_a)
            {
               compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, j);

               zzz_CopyIndex(zzz_SBoxIMin(compute_sbox), start);
               zzz_SMGMapFineToCoarse(start, startc, cindex, cstride);

               /* shift start index to appropriate F-point */
               for (d = 0; d < 3; d++)
               {
                  zzz_IndexD(start, d) +=
                     zzz_IndexD(PT_stencil_shape[si], d);
               }

               stride = zzz_SBoxStride(compute_sbox);
 
               zzz_GetSBoxSize(compute_sbox, loop_size);
               zzz_BoxLoop2(loopi, loopj, loopk, loop_size,
                            x_data_box,  start,  stride,  xi,
                            PT_data_box, startc, stridec, PTi,
                            {
                               PTp[PTi] = xp[xi];
                            });
            }
         }
      }

      /*-----------------------------------------------------
       * Free up compute package info
       *-----------------------------------------------------*/

      zzz_FreeComputePkg(compute_pkg);
   }

   /* Tell SMGRelax that the stencil has changed */
   zzz_SMGRelaxSetNewMatrixStencil(relax_data, PT_stencil);

   zzz_FreeStructStencil(compute_pkg_stencil);

   zzz_AssembleStructMatrix(PT);

   return ierr;
}

