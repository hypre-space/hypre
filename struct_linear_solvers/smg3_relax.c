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
#include "smg3.h"

/*--------------------------------------------------------------------------
 * zzz_SMG3RelaxData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   zzz_StructMatrix    *A;
   zzz_StructVector    *x;
   zzz_StructVector    *b;
   zzz_StructVector    *temp_vec;

   zzz_StructMatrix    *A2;            /* 2D plane coefficients of A */
   zzz_StructMatrix    *A2_rem;        /* A2_rem = A - A2 */
   zzz_SMGResidualData *residual_data;
   zzz_SMG2Data        *smg2_data;

} zzz_SMG3RelaxData;

/*--------------------------------------------------------------------------
 * zzz_SMG3RelaxInitialize
 *--------------------------------------------------------------------------*/

void *
zzz_SMG3RelaxInitialize( )
{
   zzz_SMG3RelaxData *residual_data;

   residual_data = zzz_CTAlloc(zzz_SMG3RelaxData, 1);

   return (void *) residual_data;
}

/*--------------------------------------------------------------------------
 * zzz_SMG3RelaxSetup
 *--------------------------------------------------------------------------*/

int
zzz_SMG3RelaxSetup( void             *smg3_relax_vdata,
                    zzz_StructMatrix *A,
                    zzz_StructVector *x,
                    zzz_StructVector *b,
                    zzz_StructVector *tmp_vec          )
{
   zzz_SMG3RelaxData  *smg3_relax_data = smg3_relax_vdata;

   zzz_ResidualData   *residual_data = (smg3_relax_data -> residual_data);
   zzz_StructMatrix   *A2_rem          = (smg3_relax_data -> A2_rem);
   zzz_StructVector   *r             = (smg3_relax_data -> r);
   zzz_SMG2Data       *smg2_data     = (smg3_relax_data -> smg2_data);

   int ierr;

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
    * Set up data needed to compute residual
    *----------------------------------------------------------*/

   stencil = zzz_StructMatrixStencil(A);
   stencil_shape = zzz_StructStencilShape(stencil);
   stencil_size  = zzz_StructStencilSize(stencil);

   stencil_indices = zzz_TAlloc(int, stencil_size);

   /* set up A2 matrix */
   num_stencil_indices = 0;
   for (i = 0; i < stencil_size; i++)
   {
      if (zzz_IndexZ(stencil_shape[i]) == 0)
      {
         stencil_indices[num_stencil_indices] = i;
         num_stencil_indices++;
      }
   }
   A2 = zzz_NewStructMatrixMask(A, num_stencil_indices, stencil_indices);

   /* set up A2_rem matrix */
   num_stencil_indices = 0;
   for (i = 0; i < stencil_size; i++)
   {
      if (zzz_IndexZ(stencil_shape[i]) != 0)
      {
         stencil_indices[num_stencil_indices] = i;
         num_stencil_indices++;
      }
   }
   A2_rem = zzz_NewStructMatrixMask(A, num_stencil_indices, stencil_indices);

   zzz_TFree(stencil_indices);

   /* setup residual_data */
   residual_data = zzz_SMGResidualInitialize();
   zzz_SMGResidualSetup(residual_data, A2_rem, x, b, temp_vec);

   /* setup smg2_data */
   smg2_data = zzz_SMG3Initialize();
   zzz_SMG2Setup(smg2_data, A2, temp_vec, x);

   /*----------------------------------------------------------
    * Set up the residual data structure
    *----------------------------------------------------------*/

   (smg3_relax_data -> A)             = A;
   (smg3_relax_data -> x)             = x;
   (smg3_relax_data -> b)             = b;
   (smg3_relax_data -> temp_vec)      = temp_vec;
   (smg3_relax_data -> A2)            = A2;
   (smg3_relax_data -> A2_rem)        = A2_rem;
   (smg3_relax_data -> residual_data) = residual_data;
   (smg3_relax_data -> smg2_data)     = smg2_data;

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMG3Relax
 *--------------------------------------------------------------------------*/

int
zzz_SMG3Relax( void             *smg3_relax_vdata,
               zzz_StructMatrix *A,
               zzz_StructVector *x,
               zzz_StructVector *b           )
{
   zzz_SMG3RelaxData  *smg3_relax_data = smg3_relax_vdata;

   zzz_StructMatrix   *A2            = (smg3_relax_data -> A2);
   zzz_StructMatrix   *A2_rem        = (smg3_relax_data -> A2_rem);
   zzz_StructVector   *temp_vec      = (smg3_relax_data -> temp_vec);
   zzz_ResidualData   *residual_data = (smg3_relax_data -> residual_data);
   zzz_SMG2Data       *smg2_data     = (smg3_relax_data -> smg2_data);

   int ierr;

   /* Compute right-hand-side for plane solves */
   zzz_SMGResidual(residual_data, A2_rem, x, b, temp_vec);

   /* Call 2D SMG code */
   zzz_SMG2Solve(smg2_data, temp_vec, x);

   return ierr;
}

/*--------------------------------------------------------------------------
 * zzz_SMG3RelaxFinalize
 *--------------------------------------------------------------------------*/

int
zzz_SMG3RelaxFinalize( void *smg3_relax_vdata )
{
   int ierr;

   zzz_SMG3RelaxData *smg3_relax_data = smg3_relax_vdata;

   if (smg3_relax_data)
   {
      zzz_FreeStructMatrixMask(smg3_relax_data -> A2);
      zzz_FreeStructMatrixMask(smg3_relax_data -> A2_rem);
      zzz_SMGResidualFinalize(smg3_relax_data -> residual_data);
      zzz_SMG2Finalize(smg3_relax_data -> smg2_data);
      zzz_TFree(smg3_relax_data);
   }

   return ierr;
}

