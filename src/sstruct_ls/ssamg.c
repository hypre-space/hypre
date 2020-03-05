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
   (ssamg_data -> relax_weight)     = 1.0;
   (ssamg_data -> num_pre_relax)    = 1;
   (ssamg_data -> num_post_relax)   = 1;
   (ssamg_data -> logging)          = 0;
   (ssamg_data -> print_level)      = 0;

   /* initialize */
   (ssamg_data -> nparts)           = -1;
   (ssamg_data -> num_levels)       = -1;

   return (void *) ssamg_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SSAMGDestroy( void *ssamg_vdata )
{
   hypre_SSAMGData   *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;
   HYPRE_Int          nparts     = hypre_SSAMGDataNparts(ssamg_data);
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
         HYPRE_SStructVectorDestroy(ssamg_data -> b_l[0]);
         HYPRE_SStructVectorDestroy(ssamg_data -> x_l[0]);
         HYPRE_SStructMatrixDestroy(ssamg_data -> A_l[0]);
         HYPRE_SStructGridDestroy(ssamg_data -> grid_l[0]);
         for (l = 1; l < num_levels; l++)
         {
            HYPRE_SStructVectorDestroy(ssamg_data -> b_l[l]);
            HYPRE_SStructVectorDestroy(ssamg_data -> x_l[l]);
            HYPRE_SStructVectorDestroy(ssamg_data -> r_l[l-1]);
            HYPRE_SStructMatrixDestroy(ssamg_data -> P_l[l-1]);
            HYPRE_SStructMatrixDestroy(ssamg_data -> RT_l[l-1]);
            HYPRE_SStructMatrixDestroy(ssamg_data -> A_l[l]);
            HYPRE_SStructGridDestroy(ssamg_data -> grid_l[l]);
         }

         hypre_TFree(ssamg_data -> b_l);
         hypre_TFree(ssamg_data -> x_l);
         hypre_TFree(ssamg_data -> r_l);
         hypre_TFree(ssamg_data -> A_l);
         hypre_TFree(ssamg_data -> P_l);
         hypre_TFree(ssamg_data -> RT_l);
         hypre_TFree(ssamg_data -> grid_l);
      }

      if (nparts > -1)
      {
         for (p = 0; p < nparts; p++)
         {
            hypre_TFree(ssamg_data -> dxyz[p]);
         }
         hypre_TFree(ssamg_data -> dxyz);
      }

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
                           HYPRE_Real   relax_weight)
{
   hypre_SSAMGData *ssamg_data = (hypre_SSAMGData *) ssamg_vdata;

   hypre_SSAMGDataRelaxWeight(ssamg_data) = relax_weight;

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
