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
 * zzz_SMGRelaxData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   double                tol;
   int                   max_iter;
                       
   int                   num_spaces;
   int                  *space_indices;
   int                  *space_strides;
                       
   int                   num_pre_spaces;
   int                  *pre_space_ranks;  /* Ranks of entries in `space_'
                                              arrays above.                */
                       
   zzz_StructMatrix     *A;
   zzz_StructVector     *x;
   zzz_StructVector     *b;
   zzz_StructVector     *temp_vec;

   /* data for 2D and 3D relaxation */
   zzz_StructMatrix     *A_sol;       /* Coefficients of A that make up
                                         the (sol)ve part of the relaxation */
   zzz_StructMatrix     *A_rem;       /* Coefficients of A (rem)aining:
                                         A_rem = A - A_sol                  */
   void                **residual_data;  /* Array of size `num_spaces' */
   void                **smg_data;       /* Array of size `num_spaces' */

   /* data for 1D relaxation */
   zzz_ComputePkg      **compute_pkgs;

   /* log info (always logged) */
   int                   num_iterations;

} zzz_SMGRelaxData;

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxInitialize
 *--------------------------------------------------------------------------*/

void *
zzz_SMGRelaxInitialize( )
{
   zzz_SMGRelaxData *relax_data;

   relax_data = zzz_CTAlloc(zzz_SMGRelaxData, 1);

   return (void *) relax_data;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetupDD
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetupDD( zzz_SMGRelaxData *smg_relax_data,
                     zzz_StructMatrix *A,
                     zzz_StructVector *x,
                     zzz_StructVector *b,
                     zzz_StructVector *tmp_vec        )
{
   int                   num_spaces    = (smg_relax_data -> num_spaces);
   int                  *space_indices = (smg_relax_data -> space_indices);
   int                  *space_strides = (smg_relax_data -> space_strides);

   zzz_StructStencil    *stencil       = zzz_StructMatrixStencil(A);     
   zzz_Index           **stencil_shape = zzz_StructStencilShape(stencil);
   int                   stencil_size  = zzz_StructStencilSize(stencil); 
   int                   stencil_dim   = zzz_StructStencilDim(stencil);
                       
   zzz_StructMatrix     *A_sol;
   zzz_StructMatrix     *A_rem;
   void                **residual_data;
   void                **smg_data;

   zzz_Index            *base_index;
   zzz_Index            *base_stride;

   int                   num_stencil_indices;
   int                  *stencil_indices;
                       
   int                   i;

   int                   ierr;

   /*----------------------------------------------------------
    * Set up data for 2D and 3D relaxations
    *----------------------------------------------------------*/

   base_index  = zzz_NewIndex();
   base_stride = zzz_NewIndex();
   zzz_SetIndex(base_index, 0, 0, 0);
   zzz_SetIndex(base_stride, 1, 1, 1);

   stencil_indices = zzz_TAlloc(int, stencil_size);

   /* set up A_sol matrix */
   num_stencil_indices = 0;
   for (i = 0; i < stencil_size; i++)
   {
      if (zzz_IndexD(stencil_shape[i], (stencil_dim - 1)) == 0)
      {
         stencil_indices[num_stencil_indices] = i;
         num_stencil_indices++;
      }
   }
   A_sol = zzz_NewStructMatrixMask(A, num_stencil_indices, stencil_indices);
   zzz_StructStencilDim(zzz_StructMatrixStencil(A_sol)) = stencil_dim - 1;

   /* set up A_rem matrix */
   num_stencil_indices = 0;
   for (i = 0; i < stencil_size; i++)
   {
      if (zzz_IndexD(stencil_shape[i], (stencil_dim - 1)) != 0)
      {
         stencil_indices[num_stencil_indices] = i;
         num_stencil_indices++;
      }
   }
   A_rem = zzz_NewStructMatrixMask(A, num_stencil_indices, stencil_indices);

   /* Set up residual_data and smg_data */
   residual_data = zzz_TAlloc(void *, num_spaces);
   smg_data      = zzz_TAlloc(void *, num_spaces);
   for (i = 0; i < num_spaces; i++)
   {
      zzz_IndexD(base_index,  (stencil_dim - 1)) = space_indices[i];
      zzz_IndexD(base_stride, (stencil_dim - 1)) = space_strides[i];

      residual_data[i] = zzz_SMGResidualInitialize();
      zzz_SMGResidualSetBaseSpace(residual_data[i],
                                  base_index, base_stride);
      zzz_SMGResidualSetup(residual_data[i], A_rem, x, b, temp_vec);

      smg_data[i] = zzz_SMGInitialize();
      zzz_SMGSetBaseSpace(smg_data[i], base_index, base_stride);
      zzz_SMGSetup(smg_data[i], A_sol, temp_vec, x);
   }

   zzz_TFree(stencil_indices);
   zzz_FreeIndex(base_index);
   zzz_FreeIndex(base_stride);

   (smg_relax_data -> A_sol)         = A_sol;
   (smg_relax_data -> A_rem)         = A_rem;
   (smg_relax_data -> residual_data) = residual_data;
   (smg_relax_data -> smg_data)      = smg_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetup1D
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetup1D( zzz_SMGRelaxData *smg_relax_data,
                     zzz_StructMatrix *A,
                     zzz_StructVector *x,
                     zzz_StructVector *b,
                     zzz_StructVector *tmp_vec         )
{
   int                   num_spaces    = (smg_relax_data -> num_spaces);
   int                  *space_indices = (smg_relax_data -> space_indices);
   int                  *space_strides = (smg_relax_data -> space_strides);

   zzz_StructGrid       *grid    = zzz_StructMatrixGrid(A);     
   zzz_StructStencil    *stencil = zzz_StructMatrixStencil(A);     

   zzz_ComputePkg      **compute_pkgs;

   zzz_Index            *base_index;
   zzz_Index            *base_stride;
                       
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

   int                   i;

   int                   ierr;

   /*----------------------------------------------------------
    * Set up data for 1D relaxations
    *----------------------------------------------------------*/

   base_index  = zzz_NewIndex();
   base_stride = zzz_NewIndex();
   zzz_SetIndex(base_index, 0, 0, 0);
   zzz_SetIndex(base_stride, 1, 1, 1);

   compute_pkgs  = zzz_TAlloc(zzz_ComputePkg *, num_spaces);
   for (i = 0; i < num_spaces; i++)
   {
      zzz_GetComputeInfo(&send_boxes, &recv_boxes,
                         &send_box_ranks, &recv_box_ranks,
                         &indt_boxes, &dept_boxes,
                         grid, stencil);
 
      zzz_IndexD(base_index,  0) = space_indices[i];
      zzz_IndexD(base_stride, 0) = space_strides[i];

      send_sboxes =
         zzz_ProjectBoxArrayArray(send_boxes, base_index, base_stride);
      recv_sboxes =
         zzz_ProjectBoxArrayArray(recv_boxes, base_index, base_stride);
      indt_sboxes =
         zzz_ProjectBoxArrayArray(indt_boxes, base_index, base_stride);
      dept_sboxes =
         zzz_ProjectBoxArrayArray(dept_boxes, base_index, base_stride);
 
      zzz_FreeBoxArrayArray(send_boxes);
      zzz_FreeBoxArrayArray(recv_boxes);
      zzz_FreeBoxArrayArray(indt_boxes);
      zzz_FreeBoxArrayArray(dept_boxes);
 
      compute_pkgs[i] =
         zzz_NewComputePkg(send_sboxes, recv_sboxes,
                           send_box_ranks, recv_box_ranks,
                           indt_sboxes, dept_sboxes,
                           grid, zzz_StructVectorDataSpace(x), 1);
   }

   zzz_FreeIndex(base_index);
   zzz_FreeIndex(base_stride);

   (smg_relax_data -> compute_pkgs) = compute_pkgs;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxSetup
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxSetup( void             *smg_relax_vdata,
                   zzz_StructMatrix *A,
                   zzz_StructVector *x,
                   zzz_StructVector *b,
                   zzz_StructVector *tmp_vec         )
{
   zzz_SMGRelaxData  *smg_relax_data = smg_relax_vdata;

   zzz_StructStencil *stencil        = zzz_StructMatrixStencil(A);     
   int                stencil_dim    = zzz_StructStencilDim(stencil);
                       
   int                ierr;

   (smg_relax_data -> A)        = A;
   (smg_relax_data -> x)        = x;
   (smg_relax_data -> b)        = b;
   (smg_relax_data -> temp_vec) = temp_vec;

   if (stencil_dim > 1)
      zzz_SMGRelaxSetupDD(smg_relax_data, A, x, b, temp_vec);
   else
      zzz_SMGRelaxSetup1D(smg_relax_data, A, x, b, temp_vec);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxDD
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxDD( zzz_SMGRelaxData *smg_relax_data,
                zzz_StructVector *x,
                zzz_StructVector *b,
                int               space_index    )
{
   zzz_StructVector   *temp_vec        = (smg_relax_data -> temp_vec);
   zzz_StructMatrix   *A_rem           = (smg_relax_data -> A_rem);
   void              **residual_data   = (smg_relax_data -> residual_data);
   void              **smg_data        = (smg_relax_data -> smg_data);

   int                 ierr;

   zzz_SMGResidual(residual_data[space_index], A_rem, x, b, temp_vec);
   zzz_SMGSolve(smg_data[space_index], temp_vec, x);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelax1D
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelax1D( zzz_SMGRelaxData *smg_relax_data,
                zzz_StructVector *x,
                zzz_StructVector *b,
                int               space_index    )
{
   zzz_ComputePkg    **compute_pkgs = (smg_relax_data -> compute_pkgs);
   zzz_ComputePkg     *compute_pkg  = compute_pkgs[space_index];

   zzz_CommHandle     *comm_handle;
                      
   zzz_SBoxArrayArray *compute_sbox_aa;
   zzz_SBoxArray      *compute_sbox_a;
   zzz_SBox           *compute_sbox;
                      
   zzz_Box            *box;
   zzz_Index          *index;
   zzz_Index          *start;
   zzz_Index          *stride;
                      
   zzz_Box            *A_data_box;
   zzz_Box            *x_data_box;
   zzz_Box            *b_data_box;
                      
   double             *Ap, *Awp, *Aep;
   double             *xp, *xwp, *xep;
   double             *yp;
                      
   int                 Ai;
   int                 xi;
   int                 bi;
                      
   int                 compute_i, i, j;
                      
   int                 ierr;

   /*-----------------------------------------------------------------------
    * Do relaxation
    *-----------------------------------------------------------------------*/

   index = zzz_NewIndex();

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

         A_data_box = zzz_BoxArrayBox(zzz_StructMatrixDataSpace(A), i);
         x_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(x), i);
         b_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(b), i);

         zzz_SetIndex(index, 0, 0, 0);
         Ap = zzz_StructMatrixExtractPointer(A, i, index);
         xp = zzz_StructVectorBoxData(x, i);
         bp = zzz_StructVectorBoxData(b, i);

         zzz_SetIndex(index, -1, 0, 0);
         Awp = zzz_StructMatrixExtractPointer(A, i, index);
         xwp = zzz_StructVectorBoxData(x, i) +
            zzz_BoxOffsetDistance(x_data_box, index);

         zzz_SetIndex(index,  1, 0, 0);
         Aep = zzz_StructMatrixExtractPointer(A, i, index);
         xep = zzz_StructVectorBoxData(x, i) +
            zzz_BoxOffsetDistance(x_data_box, index);

         zzz_ForSBoxI(j, compute_sbox_a)
         {
            compute_sbox = zzz_SBoxArraySBox(compute_sbox_a, j);

            box    = zzz_SBoxBox(compute_sbox);
            start  = zzz_SBoxIMin(compute_sbox);
            stride = zzz_SBoxStride(compute_sbox);

            zzz_BoxLoop3(box, index,
                         A_data_box, start, stride, Ai,
                         x_data_box, start, stride, xi,
                         b_data_box, start, stride, bi,
                         {
                            xp[xi] = (bp[bi] - (Awp[Ai]*xwp[xi] +
                                                Aep[Ai]*xep[xi]  )) / Ap[Ai];
                         });
         }
      }
   }
   
   zzz_FreeIndex(index);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelax
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelax( void             *smg_relax_vdata,
              zzz_StructVector *x,
              zzz_StructVector *b               )
{
   zzz_SMGRelaxData   *smg_relax_data = smg_relax_vdata;

   int                 num_spaces      = (smg_relax_data -> num_spaces);
   int                 num_pre_spaces  = (smg_relax_data -> num_pre_spaces);
   int                *pre_space_ranks = (smg_relax_data -> pre_space_ranks);

   int                 i, is;

   int                 ierr;

   /*----------------------------------------------------------
    * Do pre-relaxation
    *----------------------------------------------------------*/

   for (i = 0; i < num_pre_spaces; i++)
   {
      is = pre_space_ranks[i];
      if (stencil_dim > 1)
         zzz_SMGRelaxDD(smg_relax_data, x, b, is);
      else
         zzz_SMGRelax1D(smg_relax_data, x, b, is);
   }

   /*----------------------------------------------------------
    * Do regular relaxation iterations
    *----------------------------------------------------------*/

   for (i = 0; i < max_iter; i++)
   {
      for (is = 0; is < num_spaces; is++)
      {
         if (stencil_dim > 1)
            zzz_SMGRelaxDD(smg_relax_data, x, b, is);
         else
            zzz_SMGRelax1D(smg_relax_data, x, b, is);
      }

      (smg_relax_data -> num_iterations) = (i + 1);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMGRelaxFinalize
 *--------------------------------------------------------------------------*/

int
zzz_SMGRelaxFinalize( void *smg_relax_vdata )
{
   int ierr;

   zzz_SMGRelaxData *smg_relax_data = smg_relax_vdata;

   if (smg_relax_data)
   {
      zzz_FreeStructMatrixMask(smg_relax_data -> A_sol);
      zzz_FreeStructMatrixMask(smg_relax_data -> A_rem);
      zzz_SMGResidualFinalize(smg_relax_data -> residual_data);
      zzz_SMGFinalize(smg_relax_data -> smg_data);
      zzz_TFree(smg_relax_data);
   }

   return ierr;
}

