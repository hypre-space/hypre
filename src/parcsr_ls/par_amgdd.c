/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParAMGDD functions
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGDDCreate
 *--------------------------------------------------------------------------*/

void *
hypre_BoomerAMGDDCreate( void )
{
   hypre_ParAMGDDData  *amgdd_data = hypre_CTAlloc(hypre_ParAMGDDData, 1, HYPRE_MEMORY_HOST);

   hypre_ParAMGDDDataAMG(amgdd_data) = (hypre_ParAMGData*) hypre_BoomerAMGCreate();

   hypre_ParAMGDDDataFACNumCycles(amgdd_data)   = 2;
   hypre_ParAMGDDDataFACCycleType(amgdd_data)   = 1;
   hypre_ParAMGDDDataFACRelaxType(amgdd_data)   = 3;
   hypre_ParAMGDDDataFACNumRelax(amgdd_data)    = 1;
   hypre_ParAMGDDDataFACRelaxWeight(amgdd_data) = 1.0;
   hypre_ParAMGDDDataPadding(amgdd_data)        = 1;
   hypre_ParAMGDDDataNumGhostLayers(amgdd_data) = 1;
   hypre_ParAMGDDDataCommPkg(amgdd_data)        = NULL;
   hypre_ParAMGDDDataCompGrid(amgdd_data)       = NULL;
   hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = hypre_BoomerAMGDD_FAC_CFL1Jacobi;

   return (void *) amgdd_data;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGDDDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGDDDestroy( void *data )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;
   hypre_ParAMGData    *amg_data;
   HYPRE_Int            num_levels;
   HYPRE_Int            i;

   if (amgdd_data)
   {
      amg_data   = hypre_ParAMGDDDataAMG(amgdd_data);
      num_levels = hypre_ParAMGDataNumLevels(amg_data);

      /* destroy amgdd composite grids and commpkg */
      if (hypre_ParAMGDDDataCompGrid(amgdd_data))
      {
         for (i = 0; i < num_levels; i++)
         {
            hypre_AMGDDCompGridDestroy(hypre_ParAMGDDDataCompGrid(amgdd_data)[i]);
         }
         hypre_TFree(hypre_ParAMGDDDataCompGrid(amgdd_data), HYPRE_MEMORY_HOST);
      }

      if (hypre_ParAMGDDDataCommPkg(amgdd_data))
      {
         hypre_AMGDDCommPkgDestroy(hypre_ParAMGDDDataCommPkg(amgdd_data));
      }

      /* destroy temporary vector */
      hypre_ParVectorDestroy(hypre_ParAMGDDDataZtemp(amgdd_data));

      /* destroy the underlying amg */
      hypre_BoomerAMGDestroy((void*) amg_data);

      hypre_TFree(amgdd_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Routines to set parameters
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BoomerAMGDDSetStartLevel( void     *data,
                                HYPRE_Int start_level )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDDDataStartLevel(amgdd_data) = start_level;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDGetStartLevel( void      *data,
                                HYPRE_Int *start_level )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *start_level = hypre_ParAMGDDDataStartLevel(amgdd_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDSetFACNumRelax( void     *data,
                                 HYPRE_Int fac_num_relax )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDDDataFACNumRelax(amgdd_data) = fac_num_relax;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDGetFACNumRelax( void      *data,
                                 HYPRE_Int *fac_num_relax )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *fac_num_relax = hypre_ParAMGDDDataFACNumRelax(amgdd_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDSetFACNumCycles( void     *data,
                                  HYPRE_Int fac_num_cycles )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDDDataFACNumCycles(amgdd_data) = fac_num_cycles;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDGetFACNumCycles( void      *data,
                                  HYPRE_Int *fac_num_cycles )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *fac_num_cycles = hypre_ParAMGDDDataFACNumCycles(amgdd_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDSetFACCycleType( void     *data,
                                  HYPRE_Int fac_cycle_type )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDDDataFACCycleType(amgdd_data) = fac_cycle_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDGetFACCycleType( void      *data,
                                  HYPRE_Int *fac_cycle_type )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *fac_cycle_type = hypre_ParAMGDDDataFACCycleType(amgdd_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDSetFACRelaxType( void     *data,
                                  HYPRE_Int fac_relax_type )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDDDataFACRelaxType(amgdd_data) = fac_relax_type;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDGetFACRelaxType( void      *data,
                                  HYPRE_Int *fac_relax_type )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *fac_relax_type = hypre_ParAMGDDDataFACRelaxType(amgdd_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDSetFACRelaxWeight( void       *data,
                                    HYPRE_Real  fac_relax_weight )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDDDataFACRelaxWeight(amgdd_data) = fac_relax_weight;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDGetFACRelaxWeight( void       *data,
                                    HYPRE_Real *fac_relax_weight )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *fac_relax_weight = hypre_ParAMGDDDataFACRelaxWeight(amgdd_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDSetPadding( void      *data,
                             HYPRE_Int  padding )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDDDataPadding(amgdd_data) = padding;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDGetPadding( void      *data,
                             HYPRE_Int *padding )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *padding = hypre_ParAMGDDDataPadding(amgdd_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDSetNumGhostLayers( void      *data,
                                    HYPRE_Int  num_ghost_layers )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   hypre_ParAMGDDDataNumGhostLayers(amgdd_data) = num_ghost_layers;

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDGetNumGhostLayers( void      *data,
                                    HYPRE_Int *num_ghost_layers )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *num_ghost_layers = hypre_ParAMGDDDataNumGhostLayers(amgdd_data);

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDDSetUserFACRelaxation( void *data,
                                       HYPRE_Int (*userFACRelaxation)( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int cycle_param ))
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   hypre_ParAMGDDDataUserFACRelaxation(amgdd_data) = userFACRelaxation;

   return 0;
}

HYPRE_Int
hypre_BoomerAMGDDGetAMG( void   *data,
                         void  **amg_solver )
{
   hypre_ParAMGDDData  *amgdd_data = (hypre_ParAMGDDData*) data;

   if (!amgdd_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *amg_solver = (void*) hypre_ParAMGDDDataAMG(amgdd_data);

   return hypre_error_flag;
}
