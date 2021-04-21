/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "ssamg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SSAMGCreate( hypre_MPI_Comm comm )
{
   hypre_SSAMGData   *ssamg_data;
   HYPRE_Int          d;

   ssamg_data = hypre_CTAlloc(hypre_SSAMGData, 1, HYPRE_MEMORY_HOST);

   (ssamg_data -> comm)       = comm;
   (ssamg_data -> time_index) = hypre_InitializeTiming("SSAMG");

   /* set defaults */
   (ssamg_data -> tol)              = 1.0e-06;
   (ssamg_data -> max_iter)         = 200;
   (ssamg_data -> rel_change)       = 0;
   (ssamg_data -> non_galerkin)     = 0;
   (ssamg_data -> zero_guess)       = 0;
   (ssamg_data -> max_levels)       = 0;
   (ssamg_data -> relax_type)       = 0;
   (ssamg_data -> skip_relax)       = 0;
   (ssamg_data -> usr_relax_weight) = 0.0;
   (ssamg_data -> num_pre_relax)    = 1;
   (ssamg_data -> num_post_relax)   = 1;
   (ssamg_data -> num_coarse_relax) = -1;
   (ssamg_data -> logging)          = 0;
   (ssamg_data -> print_level)      = 0;
   (ssamg_data -> print_freq)       = 1;

   /* initialize */
   (ssamg_data -> nparts)           = -1;
   (ssamg_data -> num_levels)       = -1;
   for (d = 0; d < 3; d++)
   {
      (ssamg_data -> dxyz[d])       = NULL;
   }

   return (void *) ssamg_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGDestroy( void *ssamg_vdata )
{
   hypre_SSAMGData   *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   HYPRE_Int          num_levels;
   HYPRE_Int          max_levels;
   HYPRE_Int          l, p;

   if (ssamg_data)
   {
      if (hypre_SSAMGDataLogging(ssamg_data) > 0)
      {
         hypre_TFree(ssamg_data -> norms, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> rel_norms, HYPRE_MEMORY_HOST);
      }

      if (hypre_SSAMGDataNumLevels(ssamg_data) > -1)
      {
         num_levels = hypre_SSAMGDataNumLevels(ssamg_data);
         max_levels = hypre_SSAMGDataMaxLevels(ssamg_data);

         hypre_SSAMGRelaxDestroy(ssamg_data -> relax_data_l[0]);
         hypre_SStructMatvecDestroy(ssamg_data -> matvec_data_l[0]);
         HYPRE_SStructVectorDestroy(ssamg_data -> b_l[0]);
         HYPRE_SStructVectorDestroy(ssamg_data -> x_l[0]);
         HYPRE_SStructVectorDestroy(ssamg_data -> tx_l[0]);
         HYPRE_SStructMatrixDestroy(ssamg_data -> A_l[0]);
         HYPRE_SStructGridDestroy(ssamg_data -> grid_l[0]);
         hypre_TFree(ssamg_data -> cdir_l[0], HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> active_l[0], HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> relax_weights[0], HYPRE_MEMORY_HOST);
         for (l = 1; l < num_levels; l++)
         {
            hypre_SSAMGRelaxDestroy(ssamg_data -> relax_data_l[l]);
            hypre_SStructMatvecDestroy(ssamg_data -> matvec_data_l[l]);
            HYPRE_SStructGridDestroy(ssamg_data -> grid_l[l]);
            HYPRE_SStructVectorDestroy(ssamg_data -> b_l[l]);
            HYPRE_SStructVectorDestroy(ssamg_data -> x_l[l]);
            HYPRE_SStructVectorDestroy(ssamg_data -> tx_l[l]);
            HYPRE_SStructMatrixDestroy(ssamg_data -> A_l[l]);
            HYPRE_SStructMatrixDestroy(ssamg_data -> P_l[l-1]);
            HYPRE_SStructMatrixDestroy(ssamg_data -> RT_l[l-1]);
            hypre_SStructMatvecDestroy(ssamg_data -> restrict_data_l[l-1]);
            hypre_SStructMatvecDestroy(ssamg_data -> interp_data_l[l-1]);
            hypre_TFree(ssamg_data -> cdir_l[l], HYPRE_MEMORY_HOST);
            hypre_TFree(ssamg_data -> active_l[l], HYPRE_MEMORY_HOST);
            hypre_TFree(ssamg_data -> relax_weights[l], HYPRE_MEMORY_HOST);
         }

         for (l = num_levels; l < max_levels; l++)
         {
            hypre_TFree(ssamg_data -> active_l[l], HYPRE_MEMORY_HOST);
            hypre_TFree(ssamg_data -> relax_weights[l], HYPRE_MEMORY_HOST);
         }

         hypre_TFree(ssamg_data -> b_l, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> x_l, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> tx_l, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> A_l, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> P_l, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> RT_l, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> grid_l, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> cdir_l, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> active_l, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> relax_weights, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> relax_data_l, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> matvec_data_l, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> restrict_data_l, HYPRE_MEMORY_HOST);
         hypre_TFree(ssamg_data -> interp_data_l, HYPRE_MEMORY_HOST);

         ssamg_data -> e_l = NULL;
         ssamg_data -> r_l = NULL;
      }

      for (p = 0; p < hypre_SSAMGDataNParts(ssamg_data); p++)
      {
         hypre_TFree(ssamg_data -> dxyz[p], HYPRE_MEMORY_HOST);
      }
      hypre_TFree(ssamg_data -> dxyz[0], HYPRE_MEMORY_HOST);
      hypre_TFree(ssamg_data -> dxyz[1], HYPRE_MEMORY_HOST);
      hypre_TFree(ssamg_data -> dxyz[2], HYPRE_MEMORY_HOST);

      hypre_FinalizeTiming(ssamg_data -> time_index);
      hypre_TFree(ssamg_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetTol( void       *ssamg_vdata,
                   HYPRE_Real  tol)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataTol(ssamg_data) = tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetMaxIter( void       *ssamg_vdata,
                       HYPRE_Int   max_iter)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataMaxIter(ssamg_data) = max_iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetMaxLevels( void       *ssamg_vdata,
                         HYPRE_Int   max_levels)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataMaxLevels(ssamg_data) = max_levels;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetRelChange( void       *ssamg_vdata,
                         HYPRE_Real  rel_change)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataRelChange(ssamg_data) = rel_change;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetZeroGuess( void       *ssamg_vdata,
                         HYPRE_Int   zero_guess)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataZeroGuess(ssamg_data) = zero_guess;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetNonGalerkinRAP( void      *ssamg_vdata,
                              HYPRE_Int  non_galerkin )
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   (ssamg_data -> non_galerkin) = non_galerkin;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetDxyz( void        *ssamg_vdata,
                    HYPRE_Int    nparts,
                    HYPRE_Real **dxyz       )
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   HYPRE_Int        part;

   for (part = 0; part < nparts; part++)
   {
      (ssamg_data -> dxyz[part][0]) = dxyz[part][0];
      (ssamg_data -> dxyz[part][1]) = dxyz[part][1];
      (ssamg_data -> dxyz[part][2]) = dxyz[part][2];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetRelaxType( void       *ssamg_vdata,
                         HYPRE_Int   relax_type)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataRelaxType(ssamg_data) = relax_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetSkipRelax( void       *ssamg_vdata,
                         HYPRE_Int   skip_relax)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataSkipRelax(ssamg_data) = skip_relax;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetRelaxWeight( void        *ssamg_vdata,
                           HYPRE_Real   usr_relax_weight)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataUsrRelaxWeight(ssamg_data) = usr_relax_weight;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetNumPreRelax( void       *ssamg_vdata,
                           HYPRE_Int   num_pre_relax)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataNumPreRelax(ssamg_data) = num_pre_relax;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetNumPosRelax( void       *ssamg_vdata,
                           HYPRE_Int   num_pos_relax)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataNumPosRelax(ssamg_data) = num_pos_relax;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetNumCoarseRelax( void       *ssamg_vdata,
                              HYPRE_Int   num_coarse_relax)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataNumCoarseRelax(ssamg_data) = num_coarse_relax;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetPrintLevel( void       *ssamg_vdata,
                          HYPRE_Int   print_level)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataPrintLevel(ssamg_data) = print_level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetPrintFreq( void       *ssamg_vdata,
                         HYPRE_Int   print_freq)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataPrintFreq(ssamg_data) = print_freq;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGSetLogging( void       *ssamg_vdata,
                       HYPRE_Int   logging)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataLogging(ssamg_data) = logging;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGPrintLogging( void *ssamg_vdata )
{
   hypre_SSAMGData   *ssamg_data     = (hypre_SSAMGData *) ssamg_vdata;
   MPI_Comm           comm           = (ssamg_data -> comm);
   HYPRE_Int          num_iterations = (ssamg_data -> num_iterations);
   HYPRE_Int          logging        = (ssamg_data -> logging);
   HYPRE_Int          print_level    = (ssamg_data -> print_level);
   HYPRE_Int          print_freq     = (ssamg_data -> print_freq);
   HYPRE_Real        *norms          = (ssamg_data -> norms);
   HYPRE_Real        *rel_norms      = (ssamg_data -> rel_norms);
   HYPRE_Int          myid, i;
   HYPRE_Real         convr = 1.0;
   HYPRE_Real         avg_convr;

   hypre_MPI_Comm_rank(comm, &myid);

   if (myid == 0)
   {
      if ((print_level > 0) && (logging > 1))
      {
         hypre_printf("Iters         ||r||_2   conv.rate  ||r||_2/||b||_2\n");
         hypre_printf("% 5d    %e    %f     %e\n", 0, norms[0], convr, rel_norms[0]);
         for (i = print_freq; i < num_iterations; i = (i + print_freq))
         {
            convr = norms[i] / norms[i-1];
            hypre_printf("% 5d    %e    %f     %e\n", i, norms[i], convr, rel_norms[i]);
         }

         if ((i != num_iterations - 1) && (num_iterations > 0))
         {
            i = num_iterations;
            convr = norms[i] / norms[i-1];
            hypre_printf("% 5d    %e    %f     %e\n", i, norms[i], convr, rel_norms[i]);
         }
      }

      if ((print_level > 1) && (rel_norms[0] > 0.))
      {
         avg_convr = pow((rel_norms[num_iterations]/rel_norms[0]),
                         (1.0/(HYPRE_Real) num_iterations));
         hypre_printf("\nAverage convergence factor = %f\n", avg_convr);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGGetNumIterations( void       *ssamg_vdata,
                             HYPRE_Int  *num_iterations)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   *num_iterations = hypre_SSAMGDataNumIterations(ssamg_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGGetFinalRelativeResidualNorm( void       *ssamg_vdata,
                                         HYPRE_Real *relative_residual_norm )
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   HYPRE_Int        max_iter        = hypre_SSAMGDataMaxIter(ssamg_data);
   HYPRE_Int        num_iterations  = hypre_SSAMGDataNumIterations(ssamg_data);
   HYPRE_Int        logging         = hypre_SSAMGDataLogging(ssamg_data);
   HYPRE_Real      *rel_norms       = hypre_SSAMGDataRelNorms(ssamg_data);

   if (logging > 0)
   {
      if (max_iter == 0)
      {
         hypre_error_in_arg(1);
      }
      else
      {
         *relative_residual_norm = rel_norms[num_iterations];
      }
   }

   return hypre_error_flag;
}
