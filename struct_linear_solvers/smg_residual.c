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
#ifdef HYPRE_USE_PTHREADS
#include "box_pthreads.h"
#endif

/*--------------------------------------------------------------------------
 * hypre_SMGResidualData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   hypre_Index          base_index;
   hypre_Index          base_stride;

   hypre_StructMatrix  *A;
   hypre_StructVector  *x;
   hypre_StructVector  *b;
   hypre_StructVector  *r;
   hypre_SBoxArray     *base_points;
   hypre_ComputePkg    *compute_pkg;

   int                  time_index;
   int                  flops;

} hypre_SMGResidualData;

/*--------------------------------------------------------------------------
 * hypre_SMGResidualInitialize
 *--------------------------------------------------------------------------*/

void *
hypre_SMGResidualInitialize( )
{
   hypre_SMGResidualData *residual_data;

   residual_data = hypre_CTAlloc(hypre_SMGResidualData, 1);

   (residual_data -> time_index)  = hypre_InitializeTiming("SMGResidual");

   /* set defaults */
   hypre_SetIndex((residual_data -> base_index), 0, 0, 0);
   hypre_SetIndex((residual_data -> base_stride), 1, 1, 1);

   return (void *) residual_data;
}

/*--------------------------------------------------------------------------
 * hypre_SMGResidualSetup
 *--------------------------------------------------------------------------*/

int
hypre_SMGResidualSetup( void               *residual_vdata,
                        hypre_StructMatrix *A,
                        hypre_StructVector *x,
                        hypre_StructVector *b,
                        hypre_StructVector *r              )
{
   int ierr = 0;

   hypre_SMGResidualData  *residual_data = residual_vdata;

   hypre_IndexRef          base_index  = (residual_data -> base_index);
   hypre_IndexRef          base_stride = (residual_data -> base_stride);

   hypre_StructGrid       *grid;
   hypre_StructStencil    *stencil;
                       
   hypre_BoxArrayArray    *send_boxes;
   hypre_BoxArrayArray    *recv_boxes;
   int                   **send_processes;
   int                   **recv_processes;
   hypre_BoxArrayArray    *indt_boxes;
   hypre_BoxArrayArray    *dept_boxes;
                       
   hypre_SBoxArrayArray   *send_sboxes;
   hypre_SBoxArrayArray   *recv_sboxes;
   hypre_SBoxArrayArray   *indt_sboxes;
   hypre_SBoxArrayArray   *dept_sboxes;
                       
   hypre_SBoxArray        *base_points;
   hypre_ComputePkg       *compute_pkg;

   /*----------------------------------------------------------
    * Set up base points and the compute package
    *----------------------------------------------------------*/

   grid    = hypre_StructMatrixGrid(A);
   stencil = hypre_StructMatrixStencil(A);

   base_points = hypre_ProjectBoxArray(hypre_StructGridBoxes(grid),
                                       base_index, base_stride);

   hypre_GetComputeInfo(&send_boxes, &recv_boxes,
                        &send_processes, &recv_processes,
                        &indt_boxes, &dept_boxes,
                        grid, stencil);

   send_sboxes = hypre_ConvertToSBoxArrayArray(send_boxes);
   recv_sboxes = hypre_ConvertToSBoxArrayArray(recv_boxes);
   indt_sboxes = hypre_ProjectBoxArrayArray(indt_boxes,
                                            base_index, base_stride);
   dept_sboxes = hypre_ProjectBoxArrayArray(dept_boxes,
                                            base_index, base_stride);

   hypre_FreeBoxArrayArray(indt_boxes);
   hypre_FreeBoxArrayArray(dept_boxes);

   compute_pkg = hypre_NewComputePkg(send_sboxes, recv_sboxes,
                                     send_processes, recv_processes,
                                     indt_sboxes, dept_sboxes,
                                     grid, hypre_StructVectorDataSpace(x), 1);

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
      (hypre_StructMatrixGlobalSize(A) + hypre_StructVectorGlobalSize(x)) /
      (hypre_IndexX(base_stride) *
       hypre_IndexY(base_stride) *
       hypre_IndexZ(base_stride)  );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGResidual
 *--------------------------------------------------------------------------*/

int
hypre_SMGResidual( void               *residual_vdata,
                   hypre_StructMatrix *A,
                   hypre_StructVector *x,
                   hypre_StructVector *b,
                   hypre_StructVector *r              )
{
   int ierr = 0;

   hypre_SMGResidualData  *residual_data = residual_vdata;

   hypre_IndexRef          base_stride = (residual_data -> base_stride);
   hypre_SBoxArray        *base_points = (residual_data -> base_points);
   hypre_ComputePkg       *compute_pkg = (residual_data -> compute_pkg);

   hypre_CommHandle       *comm_handle;
                       
   hypre_SBoxArrayArray   *compute_sbox_aa;
   hypre_SBoxArray        *compute_sbox_a;
   hypre_SBox             *compute_sbox;
                       
   hypre_Box              *A_data_box;
   hypre_Box              *x_data_box;
   hypre_Box              *b_data_box;
   hypre_Box              *r_data_box;
                       
   int                     Ai;
   int                     xi;
   int                     bi;
   int                     ri;
                         
   double                 *Ap;
   double                 *xp;
   double                 *bp;
   double                 *rp;
                       
   hypre_Index             loop_size;
   hypre_IndexRef          start;
                       
   hypre_StructStencil    *stencil;
   hypre_Index            *stencil_shape;
   int                     stencil_size;

   int                     compute_i, i, j, si;
   int                     loopi, loopj, loopk;

   hypre_BeginTiming(residual_data -> time_index);

   /*-----------------------------------------------------------------------
    * Compute residual r = b - Ax
    *-----------------------------------------------------------------------*/

   stencil       = hypre_StructMatrixStencil(A);
   stencil_shape = hypre_StructStencilShape(stencil);
   stencil_size  = hypre_StructStencilSize(stencil);

   for (compute_i = 0; compute_i < 2; compute_i++)
   {
      switch(compute_i)
      {
         case 0:
         {
            xp = hypre_StructVectorData(x);
            comm_handle = hypre_InitializeIndtComputations(compute_pkg, xp);
            compute_sbox_aa = hypre_ComputePkgIndtSBoxes(compute_pkg);

            /*----------------------------------------
             * Copy b into r
             *----------------------------------------*/

            compute_sbox_a = base_points;
            hypre_ForSBoxI(i, compute_sbox_a)
               {
                  compute_sbox = hypre_SBoxArraySBox(compute_sbox_a, i);
                  start = hypre_SBoxIMin(compute_sbox);

                  b_data_box =
                     hypre_BoxArrayBox(hypre_StructVectorDataSpace(b), i);
                  r_data_box =
                     hypre_BoxArrayBox(hypre_StructVectorDataSpace(r), i);

                  bp = hypre_StructVectorBoxData(b, i);
                  rp = hypre_StructVectorBoxData(r, i);

                  hypre_GetSBoxSize(compute_sbox, loop_size);
                  hypre_BoxLoop2(loopi, loopj, loopk, loop_size,
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
            hypre_FinalizeIndtComputations(comm_handle);
            compute_sbox_aa = hypre_ComputePkgDeptSBoxes(compute_pkg);
         }
         break;
      }

      /*--------------------------------------------------------------------
       * Compute r -= A*x
       *--------------------------------------------------------------------*/

      hypre_ForSBoxArrayI(i, compute_sbox_aa)
         {
            compute_sbox_a = hypre_SBoxArrayArraySBoxArray(compute_sbox_aa, i);

            A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
            x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
            r_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(r), i);

            rp = hypre_StructVectorBoxData(r, i);

            hypre_ForSBoxI(j, compute_sbox_a)
               {
                  compute_sbox = hypre_SBoxArraySBox(compute_sbox_a, j);

                  start  = hypre_SBoxIMin(compute_sbox);

                  for (si = 0; si < stencil_size; si++)
                  {
                     Ap = hypre_StructMatrixBoxData(A, i, si);
                     xp = hypre_StructVectorBoxData(x, i) +
                        hypre_BoxOffsetDistance(x_data_box, stencil_shape[si]);

                     hypre_GetSBoxSize(compute_sbox, loop_size);
                     hypre_BoxLoop3(loopi, loopj, loopk, loop_size,
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

   hypre_IncFLOPCount(residual_data -> flops);
   hypre_EndTiming(residual_data -> time_index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGResidualSetBase
 *--------------------------------------------------------------------------*/
 
int
hypre_SMGResidualSetBase( void        *residual_vdata,
                          hypre_Index  base_index,
                          hypre_Index  base_stride )
{
   hypre_SMGResidualData *residual_data = residual_vdata;
   int                    d;
   int                    ierr = 0;
 
   for (d = 0; d < 3; d++)
   {
      hypre_IndexD((residual_data -> base_index),  d)
         = hypre_IndexD(base_index,  d);
      hypre_IndexD((residual_data -> base_stride), d)
         = hypre_IndexD(base_stride, d);
   }
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGResidualFinalize
 *--------------------------------------------------------------------------*/

int
hypre_SMGResidualFinalize( void *residual_vdata )
{
   int ierr = 0;

   hypre_SMGResidualData *residual_data = residual_vdata;

   if (residual_data)
   {
      hypre_FreeSBoxArray(residual_data -> base_points);
      hypre_FreeComputePkg(residual_data -> compute_pkg );
      hypre_FinalizeTiming(residual_data -> time_index);
      hypre_TFree(residual_data);
   }

   return ierr;
}

