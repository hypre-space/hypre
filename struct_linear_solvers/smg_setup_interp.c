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
   zzz_Index         **stencil_shape;
   int                 stencil_size;

   int                 num_ghost[] = {1, 1, 1, 1, 1, 1};

   int                 i;

   /* set up stencil */
   stencil_size = 2;
   stencil_shape = zzz_CTAlloc(zzz_Index *, stencil_size);
   for (i = 0; i < stencil_size; i++)
   {
      stencil_shape[i] = zzz_NewIndex();
      zzz_SetIndex(stencil_shape[i], 0, 0, 0);
   }
   zzz_IndexD(stencil_shape[0], cdir) = -1;
   zzz_IndexD(stencil_shape[1], cdir) =  1;

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
 *    Two vectors are needed to do the above.  One of these vectors is
 *    passed in as argument `temp_vec', and a local vector `x' is set
 *    to point to it.  The second vector needed is the right-hand-side
 *    vector, `b'.  This vector is being allocated here until a more clever
 *    use of memory can be figured out.
 *--------------------------------------------------------------------------*/

int
zzz_SMGSetupInterpOp( zzz_StructMatrix *A,
                      zzz_StructMatrix *PT,
                      zzz_StructVector *temp_vec,
                      int               cdir,
                      zzz_Index        *cindex,
                      zzz_Index        *cstride,
                      zzz_Index        *findex,
                      zzz_Index        *fstride  )
{
   void               *relax_data;

   zzz_StructMatrix   *A_mask;
   zzz_StructVector   *x = temp_vec;
   zzz_StructVector   *b;
   int                 b_num_ghost[] = {0, 0, 0, 0, 0, 0};

   zzz_StructStencil  *A_stencil;
   zzz_Index         **A_stencil_shape;
   int                 A_stencil_size;
   zzz_StructStencil  *PT_stencil;
   zzz_Index         **PT_stencil_shape;
   int                 PT_stencil_size;

   int                *stencil_indices;
   int                 num_stencil_indices;

   zzz_StructGrid     *fgrid;
   zzz_SBoxArray      *fine_points;

   zzz_Box            *PT_data_box;
   zzz_Box            *x_data_box;
   double             *PTp;
   double             *xp;
   int                 PTi;
   int                 xi;

   zzz_SBox           *sbox;
   zzz_Box            *box;
   zzz_Index          *loop_index;
   zzz_Index          *loop_size;
   zzz_Index          *start;
   zzz_Index          *startc;
   zzz_Index          *stride;
   zzz_Index          *stridec;
                      
   int                 si, sj, i, d;
                      
   int                 ierr;

   /*--------------------------------------------------------
    * Initialize some things
    *--------------------------------------------------------*/

   loop_index = zzz_NewIndex();
   loop_size  = zzz_NewIndex();
   stridec = zzz_NewIndex();
   zzz_SetIndex(stridec, 1, 1, 1);

   fgrid = zzz_StructMatrixGrid(A);
   fine_points = zzz_ProjectBoxArray(zzz_StructGridBoxes(fgrid),
                                     findex, fstride);

   /* Set up right-hand-side vector shell */
   b = zzz_NewStructVector(zzz_StructMatrixComm(A), fgrid);
   zzz_SetStructVectorNumGhost(b, b_num_ghost);
   zzz_InitializeStructVector(b);
   zzz_AssembleStructVector(b);

   /* Set up relaxation parameters */
   relax_data = zzz_SMGRelaxInitialize(zzz_StructMatrixComm(A));
   zzz_SMGRelaxSetTol(relax_data, 0.0);
   zzz_SMGRelaxSetMaxIter(relax_data, 1);
   zzz_SMGRelaxSetNumSpaces(relax_data, 1);
   zzz_SMGRelaxSetSpace(relax_data, 0,
                        zzz_IndexD(findex, cdir), zzz_IndexD(fstride, cdir));

   A_stencil = zzz_StructMatrixStencil(A);
   A_stencil_shape = zzz_StructStencilShape(A_stencil);
   A_stencil_size  = zzz_StructStencilSize(A_stencil);
   PT_stencil = zzz_StructMatrixStencil(PT);
   PT_stencil_shape = zzz_StructStencilShape(PT_stencil);
   PT_stencil_size  = zzz_StructStencilSize(PT_stencil);

   for (si = 0; si < PT_stencil_size; si++)
   {
      /*-----------------------------------------------------
       * Compute A_mask matrix: This matrix contains all
       * stencil coefficients of A except for the coefficients
       * in the opposite direction of the current P stencil
       * coefficient being computed (same direction for P^T).
       *-----------------------------------------------------*/

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

      /*-----------------------------------------------------
       * Do relaxation sweep to compute coefficients
       *-----------------------------------------------------*/

      zzz_SMGRelaxSetup(relax_data, A_mask, b, x, b);
      zzz_StructScale(1.0, x);
      zzz_StructScale(0.0, b);
      zzz_SMGRelax(relax_data, b, x);

      /*-----------------------------------------------------
       * Copy coefficients from x into P^T
       *-----------------------------------------------------*/

      zzz_ForBoxI(i, fine_points)
      {
         sbox   = zzz_SBoxArraySBox(fine_points, i);
         start  = zzz_SBoxIMin(sbox);
         stride = zzz_SBoxStride(sbox);
 
         x_data_box  = zzz_BoxArrayBox(zzz_StructVectorDataSpace(x), i);
         PT_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(PT), i);
 
         xp  = zzz_StructVectorBoxData(x, i);
         PTp = zzz_StructMatrixBoxData(PT, i, si);

         /* zero out P^T coefficient data */
         startc = zzz_BoxIMin(PT_data_box);
         zzz_GetBoxSize(PT_data_box, loop_size);
         zzz_BoxLoop1(loop_index, loop_size,
                      PT_data_box, startc, stridec, PTi,
                      {
                         PTp[PTi] = 0.0;
                      });

         /* compute startc index */
         for (d = 0; d < 3; d++)
         {
            zzz_IndexD(startc, d) =
               zzz_IndexD(start, d) - zzz_IndexD(PT_stencil_shape[si], d);
         }
         zzz_SMGMapFineToCoarse(startc, startc, cindex, cstride);

         /* copy coefficients */
         zzz_GetBoxSize(box, loop_size);
         zzz_BoxLoop2(loop_index, loop_size,
                      x_data_box,  start,  stride,  xi,
                      PT_data_box, startc, stridec, PTi,
                      {
                         PTp[PTi] = xp[xi];
                      });
      }
   }

   zzz_SMGRelaxFinalize(relax_data);
   zzz_FreeStructVector(b);
   zzz_FreeSBoxArray(fine_points);
   zzz_FreeIndex(stridec);
   zzz_FreeIndex(loop_index);
   zzz_FreeIndex(loop_size);

   zzz_AssembleStructMatrix(PT);

   return ierr;
}

