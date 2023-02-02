/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "sys_pfmg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SysPFMGCreate( MPI_Comm  comm )
{
   hypre_SysPFMGData *sys_pfmg_data;

   sys_pfmg_data = hypre_CTAlloc(hypre_SysPFMGData,  1, HYPRE_MEMORY_HOST);

   (sys_pfmg_data -> comm)       = comm;
   (sys_pfmg_data -> time_index) = hypre_InitializeTiming("SYS_PFMG");

   /* set defaults */
   (sys_pfmg_data -> tol)              = 1.0e-06;
   (sys_pfmg_data -> max_iter  )       = 200;
   (sys_pfmg_data -> rel_change)       = 0;
   (sys_pfmg_data -> zero_guess)       = 0;
   (sys_pfmg_data -> max_levels)       = 0;
   (sys_pfmg_data -> dxyz)[0]          = 0.0;
   (sys_pfmg_data -> dxyz)[1]          = 0.0;
   (sys_pfmg_data -> dxyz)[2]          = 0.0;
   (sys_pfmg_data -> relax_type)       = 1;       /* weighted Jacobi */
   (sys_pfmg_data -> jacobi_weight)    = 0.0;
   (sys_pfmg_data -> usr_jacobi_weight) = 0;
   (sys_pfmg_data -> num_pre_relax)    = 1;
   (sys_pfmg_data -> num_post_relax)   = 1;
   (sys_pfmg_data -> skip_relax)       = 1;
   (sys_pfmg_data -> logging)          = 0;
   (sys_pfmg_data -> print_level)      = 0;

   /* initialize */
   (sys_pfmg_data -> num_levels) = -1;

   return (void *) sys_pfmg_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGDestroy( void *sys_pfmg_vdata )
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   HYPRE_Int l;

   if (sys_pfmg_data)
   {
      if ((sys_pfmg_data -> logging) > 0)
      {
         hypre_TFree(sys_pfmg_data -> norms, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> rel_norms, HYPRE_MEMORY_HOST);
      }

      if ((sys_pfmg_data -> num_levels) > -1)
      {
         for (l = 0; l < (sys_pfmg_data -> num_levels); l++)
         {
            hypre_SysPFMGRelaxDestroy(sys_pfmg_data -> relax_data_l[l]);
            hypre_SStructPMatvecDestroy(sys_pfmg_data -> matvec_data_l[l]);
         }
         for (l = 0; l < ((sys_pfmg_data -> num_levels) - 1); l++)
         {
            hypre_SysSemiRestrictDestroy(sys_pfmg_data -> restrict_data_l[l]);
            hypre_SysSemiInterpDestroy(sys_pfmg_data -> interp_data_l[l]);
         }
         hypre_TFree(sys_pfmg_data -> relax_data_l, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> matvec_data_l, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> restrict_data_l, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> interp_data_l, HYPRE_MEMORY_HOST);

         hypre_SStructPVectorDestroy(sys_pfmg_data -> tx_l[0]);
         /*hypre_SStructPGridDestroy(sys_pfmg_data -> grid_l[0]);*/
         hypre_SStructPMatrixDestroy(sys_pfmg_data -> A_l[0]);
         hypre_SStructPVectorDestroy(sys_pfmg_data -> b_l[0]);
         hypre_SStructPVectorDestroy(sys_pfmg_data -> x_l[0]);
         for (l = 0; l < ((sys_pfmg_data -> num_levels) - 1); l++)
         {
            hypre_SStructPGridDestroy(sys_pfmg_data -> grid_l[l + 1]);
            hypre_SStructPGridDestroy(sys_pfmg_data -> P_grid_l[l + 1]);
            hypre_SStructPMatrixDestroy(sys_pfmg_data -> A_l[l + 1]);
            hypre_SStructPMatrixDestroy(sys_pfmg_data -> P_l[l]);
            hypre_SStructPVectorDestroy(sys_pfmg_data -> b_l[l + 1]);
            hypre_SStructPVectorDestroy(sys_pfmg_data -> x_l[l + 1]);
            hypre_SStructPVectorDestroy(sys_pfmg_data -> tx_l[l + 1]);
         }
         hypre_TFree(sys_pfmg_data -> data, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> cdir_l, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> active_l, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> grid_l, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> P_grid_l, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> A_l, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> P_l, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> RT_l, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> b_l, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> x_l, HYPRE_MEMORY_HOST);
         hypre_TFree(sys_pfmg_data -> tx_l, HYPRE_MEMORY_HOST);
      }

      hypre_FinalizeTiming(sys_pfmg_data -> time_index);
      hypre_TFree(sys_pfmg_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetTol( void   *sys_pfmg_vdata,
                     HYPRE_Real  tol       )
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> tol) = tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetMaxIter( void *sys_pfmg_vdata,
                         HYPRE_Int   max_iter  )
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> max_iter) = max_iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetRelChange( void *sys_pfmg_vdata,
                           HYPRE_Int   rel_change  )
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> rel_change) = rel_change;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetZeroGuess( void *sys_pfmg_vdata,
                           HYPRE_Int   zero_guess )
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> zero_guess) = zero_guess;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetRelaxType( void *sys_pfmg_vdata,
                           HYPRE_Int   relax_type )
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> relax_type) = relax_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SysPFMGSetJacobiWeight( void  *sys_pfmg_vdata,
                              HYPRE_Real weight )
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> jacobi_weight)    = weight;
   (sys_pfmg_data -> usr_jacobi_weight) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetNumPreRelax( void *sys_pfmg_vdata,
                             HYPRE_Int   num_pre_relax )
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> num_pre_relax) = num_pre_relax;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetNumPostRelax( void *sys_pfmg_vdata,
                              HYPRE_Int   num_post_relax )
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> num_post_relax) = num_post_relax;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetSkipRelax( void *sys_pfmg_vdata,
                           HYPRE_Int  skip_relax )
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> skip_relax) = skip_relax;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetDxyz( void   *sys_pfmg_vdata,
                      HYPRE_Real *dxyz       )
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> dxyz[0]) = dxyz[0];
   (sys_pfmg_data -> dxyz[1]) = dxyz[1];
   (sys_pfmg_data -> dxyz[2]) = dxyz[2];

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetLogging( void *sys_pfmg_vdata,
                         HYPRE_Int   logging)
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> logging) = logging;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGSetPrintLevel( void *sys_pfmg_vdata,
                            HYPRE_Int   print_level)
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   (sys_pfmg_data -> print_level) = print_level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGGetNumIterations( void *sys_pfmg_vdata,
                               HYPRE_Int  *num_iterations )
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   *num_iterations = (sys_pfmg_data -> num_iterations);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGPrintLogging( void *sys_pfmg_vdata,
                           HYPRE_Int   myid)
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;
   HYPRE_Int          i;
   HYPRE_Int          num_iterations  = (sys_pfmg_data -> num_iterations);
   HYPRE_Int          logging   = (sys_pfmg_data -> logging);
   HYPRE_Int          print_level   = (sys_pfmg_data -> print_level);
   HYPRE_Real        *norms     = (sys_pfmg_data -> norms);
   HYPRE_Real        *rel_norms = (sys_pfmg_data -> rel_norms);

   if (myid == 0)
   {
      if (print_level > 0 )
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

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SysPFMGGetFinalRelativeResidualNorm( void   *sys_pfmg_vdata,
                                           HYPRE_Real *relative_residual_norm )
{
   hypre_SysPFMGData *sys_pfmg_data = (hypre_SysPFMGData *)sys_pfmg_vdata;

   HYPRE_Int          max_iter        = (sys_pfmg_data -> max_iter);
   HYPRE_Int          num_iterations  = (sys_pfmg_data -> num_iterations);
   HYPRE_Int          logging         = (sys_pfmg_data -> logging);
   HYPRE_Real        *rel_norms       = (sys_pfmg_data -> rel_norms);

   if (logging > 0)
   {
      if (max_iter == 0)
      {
         hypre_error_in_arg(1);
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

   return hypre_error_flag;
}


