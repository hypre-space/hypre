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
 * Routine for computing residuals in the SMG code
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_SMGResidualData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_StructMatrix  *A;
   zzz_StructVector  *x;
   zzz_StructVector  *b;
   zzz_StructVector  *r;
   zzz_ComputePkg    *compute_pkg;

} zzz_SMGResidualData;

/*--------------------------------------------------------------------------
 * zzz_SMGResidualInitialize
 *--------------------------------------------------------------------------*/

void *
zzz_SMGResidualInitialize( )
{
   zzz_SMGResidualData *residual_data;

   residual_data = ctalloc(zzz_SMGResidualData, 1);

   return (void *) residual_data;
}

/*--------------------------------------------------------------------------
 * zzz_SMGResidualSetup
 *--------------------------------------------------------------------------*/

int
zzz_SMGResidualSetup( void             *residual_vdata,
                      zzz_StructMatrix *A,
                      zzz_StructVector *x,
                      zzz_StructVector *b,
                      zzz_StructVector *r              )
{
   int ierr;

   zzz_SMGResidualData *residual_data = residual_vdata;

   zzz_StructGrid       *grid;
   zzz_StructStencil    *stencil;
                       
   zzz_BoxArrayArray    *send_boxes;
   zzz_BoxArrayArray    *recv_boxes;
   int                 **send_box_ranks;
   int                 **recv_box_ranks;
   zzz_BoxArrayArray    *indt_boxes;
   zzz_BoxArrayArray    *dept_boxes;
                       
   zzz_SBoxArrayArray    *send_sboxes;
   zzz_SBoxArrayArray    *recv_sboxes;
   zzz_SBoxArrayArray    *indt_sboxes;
   zzz_SBoxArrayArray    *dept_sboxes;
                       
   zzz_ComputePkg        *compute_pkg;

   /*----------------------------------------------------------
    * Set up the compute package
    *----------------------------------------------------------*/

   grid    = zzz_StructMatrixGrid(A);
   stencil = zzz_StructMatrixStencil(A);

   zzz_GetComputeInfo(&send_boxes, &recv_boxes,
                      &send_box_ranks, &recv_box_ranks,
                      &indt_boxes, &dept_boxes,
                      grid, stencil);

   send_sboxes = zzz_ProjectBoxArrayArray(send_boxes, base_index, base_stride);
   recv_sboxes = zzz_ProjectBoxArrayArray(recv_boxes, base_index, base_stride);
   indt_sboxes = zzz_ProjectBoxArrayArray(indt_boxes, base_index, base_stride);
   dept_sboxes = zzz_ProjectBoxArrayArray(dept_boxes, base_index, base_stride);

   zzz_FreeBoxArrayArray(send_boxes);
   zzz_FreeBoxArrayArray(recv_boxes);
   zzz_FreeBoxArrayArray(indt_boxes);
   zzz_FreeBoxArrayArray(dept_boxes);

   compute_pkg = zzz_NewComputePkg(send_sboxes, recv_sboxes,
                                   send_box_ranks, recv_box_ranks,
                                   indt_sboxes, dept_sboxes,
                                   grid, zzz_StructVectorDataSpace(x), 1);

   /*----------------------------------------------------------
    * Set up the residual data structure
    *----------------------------------------------------------*/

   (residual_data -> A)           = A;
   (residual_data -> x)           = x;
   (residual_data -> b)           = b;
   (residual_data -> r)           = r;
   (residual_data -> compute_pkg) = compute_pkg;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGResidual
 *--------------------------------------------------------------------------*/

int
zzz_SMGResidual( void             *residual_vdata,
                 zzz_StructMatrix *A,
                 zzz_StructVector *x,
                 zzz_StructVector *b,
                 zzz_StructVector *r              )
{
   int ierr;

   zzz_SMGResidualData *residual_data = residual_vdata;

   zzz_ComputePkg       *compute_pkg = (residual_data -> compute_pkg);

   zzz_CommHandle       *comm_handle;
                       
   zzz_SBoxArrayArray   *compute_sbox_aa;
   zzz_SBoxArray        *compute_sbox_a;
   zzz_SBox             *compute_sbox;
                       
   zzz_Box              *A_data_box;
   zzz_Box              *x_data_box;
   zzz_Box              *y_data_box;
                       
   int                   Ai;
   int                   xi, xoffset;
   int                   yi;
                       
   double               *Ap;
   double               *xp;
   double               *yp;
                       
   zzz_BoxArray         *boxes;
   zzz_Box              *box;
   zzz_Index            *index;
   zzz_Index            *start;
   zzz_Index            *stride;
   zzz_Index            *base_stride;
                       
   zzz_StructStencil    *stencil;
   zzz_Index           **stencil_shape;
   int                   stencil_size;

   double                temp;
   int                   compute_i, i, j, si;

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   index = zzz_NewIndex();

   base_stride = zzz_NewIndex();
   zzz_SetIndex(base_stride, 1, 1, 1);

   /*-----------------------------------------------------------------------
    * Compute residual r = b - Ax
    *-----------------------------------------------------------------------*/

   stencil       = zzz_StructMatrixStencil(A);
   stencil_shape = zzz_StructStencilShape(stencil);
   stencil_size  = zzz_StructStencilSize(stencil);

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch(compute_i)
      {
         case 0:
         {
            xp = zzz_StructVectorData(x);
            comm_handle = zzz_InitializeIndtComputations(compute_pkg, xp);
            compute_sbox_aa = zzz_ComputePkgIndtSBoxes(compute_pkg);

            /*----------------------------------------
             * Copy b into r
             *----------------------------------------*/

            boxes = zzz_StructGridBoxes(zzz_StructMatrixGrid(A));
            zzz_ForBoxI(i, boxes)
            {
               box   = zzz_BoxArrayBox(boxes, i);
               start = zzz_BoxIMin(box);

               b_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(b), i);
               r_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(r), i);

               bp = zzz_StructVectorBoxData(b, i);
               rp = zzz_StructVectorBoxData(r, i);

               zzz_BoxLoop2(box, index,
                            b_data_box, start, base_stride, bi,
                            r_data_box, start, base_stride, ri,
                            {
                               rp[ri] = bp[bi];
                            });
            }
         }
         break;

         case 1:
         {
            zzz_FinalizeIndtComputations(comm_handle);
            compute_sbox_aa = zzz_ComputePkgDeptSBoxes(compute_pkg);
         }
         break;
      }

      /*--------------------------------------------------------------------
       * Compute r -= A*x
       *--------------------------------------------------------------------*/

      zzz_ForSBoxArrayI(i, compute_sbox_aa)
      {
         compute_sbox_a = zzz_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

         A_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(A), i);
         x_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(x), i);
         r_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(r), i);

         rp = zzz_StructVectorBoxData(r, i);

         zzz_ForSBoxI(j, compute_sbox_a)
         {
            compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, j);

            box    = zzz_SBoxBox(compute_sbox);
            start  = zzz_SBoxIMin(compute_sbox);
            stride = zzz_SBoxStride(compute_sbox);

            for (si = 0; si < stencil_size; si++)
            {
               Ap = zzz_StructMatrixBoxData(A, i, si);
               xp = zzz_StructVectorBoxData(x, i) +
                  zzz_BoxOffsetDistance(x_data_box, stencil_shape[si]);

               zzz_BoxLoop3(box, index,
                            A_data_box, start, stride, Ai,
                            x_data_box, start, stride, xi,
                            r_data_box, start, stride, ri,
                            {
                               rp[ri] -= Ap[Ai] * xp[xi];
                            });
            }
         }
      }
   }
   
   /*-----------------------------------------------------------------------
    * Return
    *-----------------------------------------------------------------------*/

   zzz_FreeIndex(index);
   zzz_FreeIndex(base_stride);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGResidualFinalize
 *--------------------------------------------------------------------------*/

int
zzz_SMGResidualFinalize( void *residual_vdata )
{
   int ierr;

   zzz_SMGResidualData *residual_data = residual_vdata;

   if (residual_data)
   {
      zzz_FreeComputePkg(residual_data -> compute_pkg );
      tfree(residual_data);
   }

   return ierr;
}

