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
   zzz_Index         *base_index;
   zzz_Index         *base_stride;

   zzz_StructMatrix  *A;
   zzz_StructVector  *x;
   zzz_StructVector  *b;
   zzz_StructVector  *r;
   zzz_SBoxArray     *base_points;
   zzz_ComputePkg    *compute_pkg;

   int                time_index;
   int                flops;

} zzz_SMGResidualData;

/*--------------------------------------------------------------------------
 * zzz_SMGResidualInitialize
 *--------------------------------------------------------------------------*/

void *
zzz_SMGResidualInitialize( )
{
   zzz_SMGResidualData *residual_data;

   residual_data = zzz_CTAlloc(zzz_SMGResidualData, 1);

   (residual_data -> base_index)  = zzz_NewIndex();
   (residual_data -> base_stride) = zzz_NewIndex();
   (residual_data -> time_index)  = zzz_InitializeTiming("SMGResidual");

   /* set defaults */
   zzz_SetIndex((residual_data -> base_index), 0, 0, 0);
   zzz_SetIndex((residual_data -> base_stride), 1, 1, 1);

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

   zzz_SMGResidualData  *residual_data = residual_vdata;

   zzz_Index            *base_index  = (residual_data -> base_index);
   zzz_Index            *base_stride = (residual_data -> base_stride);

   zzz_StructGrid       *grid;
   zzz_StructStencil    *stencil;
                       
   zzz_BoxArrayArray    *send_boxes;
   zzz_BoxArrayArray    *recv_boxes;
   int                 **send_box_ranks;
   int                 **recv_box_ranks;
   zzz_BoxArrayArray    *indt_boxes;
   zzz_BoxArrayArray    *dept_boxes;
                       
   zzz_SBoxArrayArray   *send_sboxes;
   zzz_SBoxArrayArray   *recv_sboxes;
   zzz_SBoxArrayArray   *indt_sboxes;
   zzz_SBoxArrayArray   *dept_sboxes;
                       
   zzz_SBoxArray        *base_points;
   zzz_ComputePkg       *compute_pkg;

   /*----------------------------------------------------------
    * Set up base points and the compute package
    *----------------------------------------------------------*/

   grid    = zzz_StructMatrixGrid(A);
   stencil = zzz_StructMatrixStencil(A);

   base_points = zzz_ProjectBoxArray(zzz_StructGridBoxes(grid),
                                     base_index, base_stride);

   zzz_GetComputeInfo(&send_boxes, &recv_boxes,
                      &send_box_ranks, &recv_box_ranks,
                      &indt_boxes, &dept_boxes,
                      grid, stencil);

   send_sboxes = zzz_ConvertToSBoxArrayArray(send_boxes);
   recv_sboxes = zzz_ConvertToSBoxArrayArray(recv_boxes);
   indt_sboxes = zzz_ProjectBoxArrayArray(indt_boxes, base_index, base_stride);
   dept_sboxes = zzz_ProjectBoxArrayArray(dept_boxes, base_index, base_stride);

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
   (residual_data -> base_points) = base_points;
   (residual_data -> compute_pkg) = compute_pkg;

   /*-----------------------------------------------------
    * Compute flops
    *-----------------------------------------------------*/

   (residual_data -> flops) =
      (zzz_StructMatrixGlobalSize(A) + zzz_StructVectorGlobalSize(x)) /
      (zzz_IndexX(base_stride) *
       zzz_IndexY(base_stride) *
       zzz_IndexZ(base_stride)  );

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

   zzz_SMGResidualData  *residual_data = residual_vdata;

   zzz_Index            *base_stride = (residual_data -> base_stride);
   zzz_SBoxArray        *base_points = (residual_data -> base_points);
   zzz_ComputePkg       *compute_pkg = (residual_data -> compute_pkg);

   zzz_CommHandle       *comm_handle;
                       
   zzz_SBoxArrayArray   *compute_sbox_aa;
   zzz_SBoxArray        *compute_sbox_a;
   zzz_SBox             *compute_sbox;
                       
   zzz_Box              *A_data_box;
   zzz_Box              *x_data_box;
   zzz_Box              *b_data_box;
   zzz_Box              *r_data_box;
                       
   int                   Ai;
   int                   xi;
   int                   bi;
   int                   ri;
                       
   double               *Ap;
   double               *xp;
   double               *bp;
   double               *rp;
                       
   zzz_Index            *loop_index;
   zzz_Index            *loop_size;
   zzz_Index            *start;
                       
   zzz_StructStencil    *stencil;
   zzz_Index           **stencil_shape;
   int                   stencil_size;

   int                   compute_i, i, j, si;

   zzz_BeginTiming(residual_data -> time_index);

   /*-----------------------------------------------------------------------
    * Initialize some things
    *-----------------------------------------------------------------------*/

   loop_index = zzz_NewIndex();
   loop_size = zzz_NewIndex();

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

            compute_sbox_a = base_points;
            zzz_ForSBoxI(i, compute_sbox_a)
            {
               compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, i);
               start = zzz_SBoxIMin(compute_sbox);

               b_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(b), i);
               r_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(r), i);

               bp = zzz_StructVectorBoxData(b, i);
               rp = zzz_StructVectorBoxData(r, i);

               zzz_GetSBoxSize(compute_sbox, loop_size);
               zzz_BoxLoop2(loop_index, loop_size,
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

            start  = zzz_SBoxIMin(compute_sbox);

            for (si = 0; si < stencil_size; si++)
            {
               Ap = zzz_StructMatrixBoxData(A, i, si);
               xp = zzz_StructVectorBoxData(x, i) +
                  zzz_BoxOffsetDistance(x_data_box, stencil_shape[si]);

               zzz_GetSBoxSize(compute_sbox, loop_size);
               zzz_BoxLoop3(loop_index, loop_size,
                            A_data_box, start, base_stride, Ai,
                            x_data_box, start, base_stride, xi,
                            r_data_box, start, base_stride, ri,
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

   zzz_FreeIndex(loop_index);
   zzz_FreeIndex(loop_size);

   zzz_IncFLOPCount(residual_data -> flops);
   zzz_EndTiming(residual_data -> time_index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGResidualSetBase
 *--------------------------------------------------------------------------*/
 
int
zzz_SMGResidualSetBase( void      *residual_vdata,
                        zzz_Index *base_index,
                        zzz_Index *base_stride )
{
   zzz_SMGResidualData *residual_data = residual_vdata;
   int          d;
   int          ierr = 0;
 
   for (d = 0; d < 3; d++)
   {
      zzz_IndexD((residual_data -> base_index),  d)
         = zzz_IndexD(base_index,  d);
      zzz_IndexD((residual_data -> base_stride), d)
         = zzz_IndexD(base_stride, d);
   }
 
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
      zzz_FreeIndex(residual_data -> base_index);
      zzz_FreeIndex(residual_data -> base_stride);
      zzz_FreeSBoxArray(residual_data -> base_points);
      zzz_FreeComputePkg(residual_data -> compute_pkg );
      zzz_FinalizeTiming(residual_data -> time_index);
      zzz_TFree(residual_data);
   }

   return ierr;
}

