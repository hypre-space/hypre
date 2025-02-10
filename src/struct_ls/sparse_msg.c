/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 *
 *****************************************************************************/

#include "_hypre_struct_ls.h"
#include "sparse_msg.h"

/*--------------------------------------------------------------------------
 * hypre_SparseMSGCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SparseMSGCreate( MPI_Comm  comm )
{
   hypre_SparseMSGData *smsg_data;

   smsg_data = hypre_CTAlloc(hypre_SparseMSGData,  1, HYPRE_MEMORY_HOST);

   (smsg_data -> comm)       = comm;
   (smsg_data -> time_index) = hypre_InitializeTiming("SparseMSG");

   /* set defaults */
   (smsg_data -> tol)              = 1.0e-06;
   (smsg_data -> max_iter)         = 200;
   (smsg_data -> rel_change)       = 0;
   (smsg_data -> zero_guess)       = 0;
   (smsg_data -> jump)             = 0;
   (smsg_data -> relax_type)       = 1;       /* weighted Jacobi */
   (smsg_data -> jacobi_weight)    = 0.0;
   (smsg_data -> usr_jacobi_weight) = 0;    /* no user Jacobi weight */
   (smsg_data -> num_pre_relax)    = 1;
   (smsg_data -> num_post_relax)   = 1;
   (smsg_data -> num_fine_relax)   = 1;
   (smsg_data -> logging)          = 0;
   (smsg_data -> print_level)      = 0;

   /* initialize */
   (smsg_data -> num_grids[0])     = 1;
   (smsg_data -> num_grids[1])     = 1;
   (smsg_data -> num_grids[2])     = 1;

   (smsg_data -> memory_location)  = hypre_HandleMemoryLocation(hypre_handle());

   return (void *) smsg_data;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGDestroy( void *smsg_vdata )
{
   HYPRE_Int ierr = 0;

   /* RDF */
#if 0
   hypre_SparseMSGData *smsg_data = smsg_vdata;

   HYPRE_Int fi, l;

   if (smsg_data)
   {
      if ((smsg_data -> logging) > 0)
      {
         hypre_TFree(smsg_data -> norms, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> rel_norms, HYPRE_MEMORY_HOST);
      }

      if ((smsg_data -> num_levels) > 1)
      {
         for (fi = 0; fi < (smsg_data -> num_all_grids); fi++)
         {
            hypre_PFMGRelaxDestroy(smsg_data -> relax_array[fi]);
            hypre_StructMatvecDestroy(smsg_data -> matvec_array[fi]);
            hypre_SemiRestrictDestroy(smsg_data -> restrictx_array[fi]);
            hypre_SemiRestrictDestroy(smsg_data -> restricty_array[fi]);
            hypre_SemiRestrictDestroy(smsg_data -> restrictz_array[fi]);
            hypre_SemiInterpDestroy(smsg_data -> interpx_array[fi]);
            hypre_SemiInterpDestroy(smsg_data -> interpy_array[fi]);
            hypre_SemiInterpDestroy(smsg_data -> interpz_array[fi]);
            hypre_StructMatrixDestroy(smsg_data -> A_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> b_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> x_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> t_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> r_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> visitx_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> visity_array[fi]);
            hypre_StructVectorDestroy(smsg_data -> visitz_array[fi]);
            hypre_StructGridDestroy(smsg_data -> grid_array[fi]);
         }

         for (l = 0; l < (smsg_data -> num_grids[0]) - 1; l++)
         {
            hypre_StructMatrixDestroy(smsg_data -> Px_array[l]);
            hypre_StructGridDestroy(smsg_data -> Px_grid_array[l]);
         }
         for (l = 0; l < (smsg_data -> num_grids[1]) - 1; l++)
         {
            hypre_StructMatrixDestroy(smsg_data -> Py_array[l]);
            hypre_StructGridDestroy(smsg_data -> Py_grid_array[l]);
         }
         for (l = 0; l < (smsg_data -> num_grids[2]) - 1; l++)
         {
            hypre_StructMatrixDestroy(smsg_data -> Pz_array[l]);
            hypre_StructGridDestroy(smsg_data -> Pz_grid_array[l]);
         }

         hypre_TFree(smsg_data -> data, HYPRE_MEMORY_HOST);

         hypre_TFree(smsg_data -> relax_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> matvec_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> restrictx_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> restricty_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> restrictz_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> interpx_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> interpy_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> interpz_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> A_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> Px_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> Py_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> Pz_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> RTx_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> RTy_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> RTz_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> b_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> x_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> t_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> r_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> grid_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> Px_grid_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> Py_grid_array, HYPRE_MEMORY_HOST);
         hypre_TFree(smsg_data -> Pz_grid_array, HYPRE_MEMORY_HOST);
      }

      hypre_FinalizeTiming(smsg_data -> time_index);
      hypre_TFree(smsg_data, HYPRE_MEMORY_HOST);
   }
#else
   HYPRE_UNUSED_VAR(smsg_vdata);
#endif
   /* RDF */

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGSetTol( void   *smsg_vdata,
                       HYPRE_Real  tol        )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;
   HYPRE_Int       ierr = 0;

   (smsg_data -> tol) = tol;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGSetMaxIter( void *smsg_vdata,
                           HYPRE_Int   max_iter   )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;
   HYPRE_Int       ierr = 0;

   (smsg_data -> max_iter) = max_iter;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetJump
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGSetJump(  void *smsg_vdata,
                         HYPRE_Int   jump       )

{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;
   HYPRE_Int            ierr = 0;

   (smsg_data -> jump) = jump;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGSetRelChange( void *smsg_vdata,
                             HYPRE_Int   rel_change )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;
   HYPRE_Int       ierr = 0;

   (smsg_data -> rel_change) = rel_change;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetZeroGuess
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGSetZeroGuess( void *smsg_vdata,
                             HYPRE_Int   zero_guess )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;
   HYPRE_Int       ierr = 0;

   (smsg_data -> zero_guess) = zero_guess;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGSetRelaxType( void *smsg_vdata,
                             HYPRE_Int   relax_type )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;
   HYPRE_Int       ierr = 0;

   (smsg_data -> relax_type) = relax_type;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetJacobiWeight
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SparseMSGSetJacobiWeight( void  *smsg_vdata,
                                HYPRE_Real weight )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;

   (smsg_data -> jacobi_weight)    = weight;
   (smsg_data -> usr_jacobi_weight) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetNumPreRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGSetNumPreRelax( void *smsg_vdata,
                               HYPRE_Int   num_pre_relax )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;
   HYPRE_Int       ierr = 0;

   (smsg_data -> num_pre_relax) = num_pre_relax;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetNumPostRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGSetNumPostRelax( void *smsg_vdata,
                                HYPRE_Int   num_post_relax )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;
   HYPRE_Int       ierr = 0;

   (smsg_data -> num_post_relax) = num_post_relax;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetNumFineRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGSetNumFineRelax( void *smsg_vdata,
                                HYPRE_Int   num_fine_relax )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;
   HYPRE_Int       ierr = 0;

   (smsg_data -> num_fine_relax) = num_fine_relax;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGSetLogging( void *smsg_vdata,
                           HYPRE_Int   logging    )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;
   HYPRE_Int       ierr = 0;

   (smsg_data -> logging) = logging;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGSetPrintLevel( void *smsg_vdata,
                              HYPRE_Int   print_level    )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;
   HYPRE_Int       ierr = 0;

   (smsg_data -> print_level) = print_level;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGGetNumIterations( void *smsg_vdata,
                                 HYPRE_Int  *num_iterations )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;
   HYPRE_Int       ierr = 0;

   *num_iterations = (smsg_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGPrintLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGPrintLogging( void *smsg_vdata,
                             HYPRE_Int   myid       )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;
   HYPRE_Int       ierr = 0;
   HYPRE_Int       i;
   HYPRE_Int       num_iterations  = (smsg_data -> num_iterations);
   HYPRE_Int       logging   = (smsg_data -> logging);
   HYPRE_Int     print_level = (smsg_data -> print_level);
   HYPRE_Real     *norms     = (smsg_data -> norms);
   HYPRE_Real     *rel_norms = (smsg_data -> rel_norms);

   if (myid == 0)
   {
      if (print_level > 0)
      {
         if (logging > 0)
         {
            for (i = 0; i < num_iterations; i++)
            {
               hypre_printf("Residual norm[%d] = %e   ", i, norms[i]);
               hypre_printf("Relative residual norm[%d] = %e\n", i, rel_norms[i]);
            }
         }
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SparseMSGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SparseMSGGetFinalRelativeResidualNorm( void   *smsg_vdata,
                                             HYPRE_Real *relative_residual_norm )
{
   hypre_SparseMSGData *smsg_data = (hypre_SparseMSGData *)smsg_vdata;

   HYPRE_Int       max_iter        = (smsg_data -> max_iter);
   HYPRE_Int       num_iterations  = (smsg_data -> num_iterations);
   HYPRE_Int       logging         = (smsg_data -> logging);
   HYPRE_Real     *rel_norms       = (smsg_data -> rel_norms);

   HYPRE_Int       ierr = 0;


   if (logging > 0)
   {
      if (max_iter == 0)
      {
         ierr = 1;
      }
      else if (num_iterations == max_iter)
      {
         *relative_residual_norm = rel_norms[num_iterations - 1];
      }
      else
      {
         *relative_residual_norm = rel_norms[num_iterations];
      }
   }

   return ierr;
}
