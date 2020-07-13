/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "_hypre_sstruct_ls.h"
#include "ssamg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_SSAMGCreate( hypre_MPI_Comm comm )
{
   hypre_SSAMGData   *ssamg_data;

   ssamg_data = hypre_CTAlloc(hypre_SSAMGData, 1);

   (ssamg_data -> comm)       = comm;
   (ssamg_data -> time_index) = hypre_InitializeTiming("SSAMG");

   /* set defaults */
   (ssamg_data -> tol)              = 1.0e-06;
   (ssamg_data -> max_iter)         = 200;
   (ssamg_data -> rel_change)       = 0;
   (ssamg_data -> zero_guess)       = 0;
   (ssamg_data -> max_levels)       = 0;
   (ssamg_data -> relax_type)       = 0;
   (ssamg_data -> usr_relax_weight) = 0.0;
   (ssamg_data -> num_pre_relax)    = 1;
   (ssamg_data -> num_post_relax)   = 1;
   (ssamg_data -> num_coarse_relax) = -1;
   (ssamg_data -> logging)          = 0;
   (ssamg_data -> print_level)      = 0;

   /* initialize */
   (ssamg_data -> nparts)           = NULL;
   (ssamg_data -> num_levels)       = -1;

   return (void *) ssamg_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGDestroy( void *ssamg_vdata )
{
   hypre_SSAMGData   *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;
   HYPRE_Int         *nparts     = hypre_SSAMGDataNParts(ssamg_data);
   HYPRE_Int          num_levels = hypre_SSAMGDataNumLevels(ssamg_data);
   HYPRE_Int          logging    = hypre_SSAMGDataLogging(ssamg_data);
   HYPRE_Int          l, p;

   if (ssamg_data)
   {
      if (logging > 0)
      {
         hypre_TFree(ssamg_data -> norms);
         hypre_TFree(ssamg_data -> rel_norms);
      }

      if (num_levels > -1)
      {
         hypre_SSAMGRelaxDestroy(ssamg_data -> relax_data_l[0]);
         hypre_SStructMatvecDestroy(ssamg_data -> matvec_data_l[0]);
         HYPRE_SStructVectorDestroy(ssamg_data -> b_l[0]);
         HYPRE_SStructVectorDestroy(ssamg_data -> x_l[0]);
         HYPRE_SStructVectorDestroy(ssamg_data -> tx_l[0]);
         HYPRE_SStructMatrixDestroy(ssamg_data -> A_l[0]);
         HYPRE_SStructGridDestroy(ssamg_data -> grid_l[0]);
         hypre_TFree(ssamg_data -> cdir_l[0]);
         hypre_TFree(ssamg_data -> relax_weights[0]);
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
            hypre_TFree(ssamg_data -> cdir_l[l]);
            hypre_TFree(ssamg_data -> relax_weights[l]);
         }

         hypre_TFree(ssamg_data -> b_l);
         hypre_TFree(ssamg_data -> x_l);
         hypre_TFree(ssamg_data -> tx_l);
         hypre_TFree(ssamg_data -> A_l);
         hypre_TFree(ssamg_data -> P_l);
         hypre_TFree(ssamg_data -> RT_l);
         hypre_TFree(ssamg_data -> grid_l);
         hypre_TFree(ssamg_data -> cdir_l);
         hypre_TFree(ssamg_data -> relax_weights);
         hypre_TFree(ssamg_data -> relax_data_l);
         hypre_TFree(ssamg_data -> matvec_data_l);
         hypre_TFree(ssamg_data -> restrict_data_l);
         hypre_TFree(ssamg_data -> interp_data_l);

         ssamg_data -> e_l = NULL;
         ssamg_data -> r_l = NULL;
      }

      if (nparts != NULL)
      {
         for (p = 0; p < nparts[0]; p++)
         {
            hypre_TFree(ssamg_data -> dxyz[p]);
         }
         hypre_TFree(ssamg_data -> dxyz);
      }
      hypre_TFree(ssamg_data -> nparts);

      hypre_FinalizeTiming(ssamg_data -> time_index);
      hypre_TFree(ssamg_data);
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
   HYPRE_Real        *norms          = (ssamg_data -> norms);
   HYPRE_Real        *rel_norms      = (ssamg_data -> rel_norms);
   HYPRE_Int          myid, i;
   HYPRE_Real         convr = 1.0;

   hypre_MPI_Comm_rank(comm, &myid);

   if (myid == 0)
   {
      if ((print_level > 0) && (logging > 1))
      {
         hypre_printf("Iters         ||r||_2   conv.rate  ||r||_2/||b||_2\n");
         hypre_printf("% 5d    %e    %f     %e\n", 0, norms[0], convr, rel_norms[0]);
         for (i = print_level; i < num_iterations; i = (i + print_level))
         {
            convr = norms[i] / norms[i-1];
            hypre_printf("% 5d    %e    %f     %e\n", i, norms[i], convr, rel_norms[i]);
         }

         if ((i != num_iterations) && (num_iterations > 0))
         {
            i = (num_iterations - 1);
            hypre_printf("% 5d    %e    %f     %e\n", i, norms[i], convr, rel_norms[i]);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGPrintStats( void *ssamg_vdata )
{
   hypre_SSAMGData    *ssamg_data    = (hypre_SSAMGData *) ssamg_vdata;
   HYPRE_Int           num_levels    = hypre_SSAMGDataNumLevels(ssamg_data);
   MPI_Comm            comm          = hypre_SSAMGDataComm(ssamg_data);
   HYPRE_Int           print_level   = hypre_SSAMGDataPrintLevel(ssamg_data);
   HYPRE_Int           relax_type    = hypre_SSAMGDataRelaxType(ssamg_data);
   HYPRE_Int           num_pre_relax = hypre_SSAMGDataNumPreRelax(ssamg_data);
   HYPRE_Int           num_pos_relax = hypre_SSAMGDataNumPosRelax(ssamg_data);
   HYPRE_Int           num_crelax    = hypre_SSAMGDataNumCoarseRelax(ssamg_data);
   HYPRE_Int          *nparts        = hypre_SSAMGDataNParts(ssamg_data);
   HYPRE_Int         **cdir_l        = hypre_SSAMGDataCdir(ssamg_data);
   HYPRE_Real        **relax_weights = hypre_SSAMGDataRelaxWeights(ssamg_data);

   HYPRE_Int           myid, i, l, part;
   HYPRE_Int           chunk, chunk_size, chunk_last;
   HYPRE_Int           nparts_per_line = 8;
   HYPRE_Int           ndigits;

   hypre_MPI_Comm_rank(comm, &myid);

   if ((myid == 0) && (print_level > 0))
   {
      hypre_printf("\nSSAMG Setup Parameters:\n\n");

      /* Print coarsening direction */
      hypre_printf("Coarsening direction:\n\n");
      chunk_size = hypre_min(nparts[0], nparts_per_line);
      for (chunk = 0; chunk < nparts[0]; chunk += chunk_size)
      {
         ndigits = 4;
         hypre_printf("lev   ");
         chunk_last = hypre_min(chunk + chunk_size, nparts[0]);
         for (part = chunk; part < chunk_last; part++)
         {
            hypre_printf("pt. %d  ", part);
            ndigits += 7;
         }
         hypre_printf("\n");
         for (i = 0; i < ndigits; i++) hypre_printf("%s", "=");
         hypre_printf("\n");
         for (l = 0; l < (num_levels - 1); l++)
         {
            hypre_printf("%3d  ", l);
            for (part = chunk; part < chunk_last; part++)
            {
               hypre_printf("%6d ", cdir_l[l][part]);
            }
            hypre_printf("\n");
         }
	 hypre_printf("\n\n");
      }

      /* Print Relaxation factor */
      if (relax_type > 0)
      {
         hypre_printf("Relaxation factors:\n\n");
         chunk_size = hypre_min(nparts[0], nparts_per_line);
         for (chunk = 0; chunk < nparts[0]; chunk += chunk_size)
         {
            ndigits = 4;
            hypre_printf("lev   ");
            chunk_last = hypre_min(chunk + chunk_size, nparts[0]);
            for (part = chunk; part < chunk_last; part++)
            {
               hypre_printf("pt. %d  ", part);
               ndigits += 7;
            }
            hypre_printf("\n");
            for (i = 0; i < ndigits; i++) hypre_printf("%s", "=");
            hypre_printf("\n");
            for (l = 0; l < num_levels; l++)
            {
               hypre_printf("%3d  ", l);
               for (part = chunk; part < chunk_last; part++)
               {
                  hypre_printf("%6.2f ", relax_weights[l][part]);
               }
               hypre_printf("\n");
            }
	    hypre_printf("\n\n");
         }
      }

      /* SSAMG details */
      hypre_printf("Relaxation type: ");
      if (relax_type == 0)
      {
         hypre_printf("Jacobi\n");
      }
      else if (relax_type == 1)
      {
         hypre_printf("Weighted Jacobi\n");
      }
      else if (relax_type == 2)
      {
         hypre_printf("Red-Black Gauss-Seidel\n");
      }
      else
      {
         hypre_printf("Unknown - %d\n", relax_type);
      }
      hypre_printf("Number of pre-sweeps: %d\n", num_pre_relax);
      hypre_printf("Number of pos-sweeps: %d\n", num_pos_relax);
      hypre_printf("Number of coarse-sweeps: %d\n", num_crelax);
      hypre_printf("Number of levels: %d\n", num_levels);

      hypre_printf("\n\n");
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
      else if (num_iterations == max_iter)
      {
         *relative_residual_norm = rel_norms[num_iterations-1];
      }
      else
      {
         *relative_residual_norm = rel_norms[num_iterations];
      }
   }

   return hypre_error_flag;
}
