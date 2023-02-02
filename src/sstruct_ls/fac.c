/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fac.h"

/*--------------------------------------------------------------------------
 * hypre_FACCreate
 *--------------------------------------------------------------------------*/
void *
hypre_FACCreate( MPI_Comm  comm )
{
   hypre_FACData *fac_data;

   fac_data = hypre_CTAlloc(hypre_FACData,  1, HYPRE_MEMORY_HOST);

   (fac_data -> comm)       = comm;
   (fac_data -> time_index) = hypre_InitializeTiming("FAC");

   /* set defaults */
   (fac_data -> tol)              = 1.0e-06;
   (fac_data -> max_cycles)       = 200;
   (fac_data -> zero_guess)       = 0;
   (fac_data -> max_levels)       = 0;
   (fac_data -> relax_type)       = 2; /*  1 Jacobi; 2 Gauss-Seidel */
   (fac_data -> jacobi_weight)    = 0.0;
   (fac_data -> usr_jacobi_weight) = 0;
   (fac_data -> num_pre_smooth)   = 1;
   (fac_data -> num_post_smooth)  = 1;
   (fac_data -> csolver_type)     = 1;
   (fac_data -> logging)          = 0;

   return (void *) fac_data;
}

/*--------------------------------------------------------------------------
 * hypre_FACDestroy
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_FACDestroy2(void *fac_vdata)
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;

   HYPRE_Int level;
   HYPRE_Int ierr = 0;

   if (fac_data)
   {
      hypre_TFree((fac_data ->plevels), HYPRE_MEMORY_HOST);
      hypre_TFree((fac_data ->prefinements), HYPRE_MEMORY_HOST);

      HYPRE_SStructGraphDestroy(hypre_SStructMatrixGraph((fac_data -> A_rap)));
      HYPRE_SStructMatrixDestroy((fac_data -> A_rap));
      for (level = 0; level <= (fac_data -> max_levels); level++)
      {
         HYPRE_SStructMatrixDestroy( (fac_data -> A_level[level]) );
         HYPRE_SStructVectorDestroy( (fac_data -> x_level[level]) );
         HYPRE_SStructVectorDestroy( (fac_data -> b_level[level]) );
         HYPRE_SStructVectorDestroy( (fac_data -> r_level[level]) );
         HYPRE_SStructVectorDestroy( (fac_data -> e_level[level]) );
         hypre_SStructPVectorDestroy( (fac_data -> tx_level[level]) );

         HYPRE_SStructGraphDestroy( (fac_data -> graph_level[level]) );
         HYPRE_SStructGridDestroy(  (fac_data -> grid_level[level]) );

         hypre_SStructMatvecDestroy( (fac_data   -> matvec_data_level[level]) );
         hypre_SStructPMatvecDestroy((fac_data  -> pmatvec_data_level[level]) );

         hypre_SysPFMGRelaxDestroy( (fac_data -> relax_data_level[level]) );

         if (level > 0)
         {
            hypre_FacSemiRestrictDestroy2( (fac_data -> restrict_data_level[level]) );
         }

         if (level < (fac_data -> max_levels))
         {
            hypre_FacSemiInterpDestroy2( (fac_data -> interp_data_level[level]) );
         }
      }
      hypre_SStructMatvecDestroy( (fac_data -> matvec_data) );

      hypre_TFree(fac_data -> A_level, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_data -> x_level, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_data -> b_level, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_data -> r_level, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_data -> e_level, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_data -> tx_level, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_data -> relax_data_level, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_data -> restrict_data_level, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_data -> matvec_data_level, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_data -> pmatvec_data_level, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_data -> interp_data_level, HYPRE_MEMORY_HOST);

      hypre_TFree(fac_data -> grid_level, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_data -> graph_level, HYPRE_MEMORY_HOST);

      HYPRE_SStructVectorDestroy(fac_data -> tx);

      hypre_TFree(fac_data -> level_to_part, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_data -> part_to_level, HYPRE_MEMORY_HOST);
      hypre_TFree(fac_data -> refine_factors, HYPRE_MEMORY_HOST);

      if ( (fac_data -> csolver_type) == 1)
      {
         HYPRE_SStructPCGDestroy(fac_data -> csolver);
         HYPRE_SStructSysPFMGDestroy(fac_data -> cprecond);
      }
      else if ((fac_data -> csolver_type) == 2)
      {
         HYPRE_SStructSysPFMGDestroy(fac_data -> csolver);
      }

      if ((fac_data -> logging) > 0)
      {
         hypre_TFree(fac_data -> norms, HYPRE_MEMORY_HOST);
         hypre_TFree(fac_data -> rel_norms, HYPRE_MEMORY_HOST);
      }

      hypre_FinalizeTiming(fac_data -> time_index);

      hypre_TFree(fac_data, HYPRE_MEMORY_HOST);
   }

   return (ierr);
}

HYPRE_Int
hypre_FACSetTol( void   *fac_vdata,
                 HYPRE_Real  tol       )
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;
   HYPRE_Int          ierr = 0;

   (fac_data -> tol) = tol;

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_FACSetPLevels
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_FACSetPLevels( void *fac_vdata,
                     HYPRE_Int   nparts,
                     HYPRE_Int  *plevels)
{
   hypre_FACData *fac_data   = (hypre_FACData *)fac_vdata;
   HYPRE_Int     *fac_plevels;
   HYPRE_Int      ierr       = 0;
   HYPRE_Int      i;

   fac_plevels = hypre_CTAlloc(HYPRE_Int,  nparts, HYPRE_MEMORY_HOST);

   for (i = 0; i < nparts; i++)
   {
      fac_plevels[i] = plevels[i];
   }

   (fac_data -> plevels) =  fac_plevels;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetPRefinements
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_FACSetPRefinements( void         *fac_vdata,
                          HYPRE_Int     nparts,
                          hypre_Index  *prefinements )
{
   hypre_FACData *fac_data   = (hypre_FACData *)fac_vdata;
   hypre_Index   *fac_prefinements;
   HYPRE_Int      ierr       = 0;
   HYPRE_Int      i;

   fac_prefinements = hypre_TAlloc(hypre_Index,  nparts, HYPRE_MEMORY_HOST);

   for (i = 0; i < nparts; i++)
   {
      hypre_CopyIndex( prefinements[i], fac_prefinements[i] );
   }

   (fac_data -> prefinements) =  fac_prefinements;

   return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_FACSetMaxLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FACSetMaxLevels( void *fac_vdata,
                       HYPRE_Int   nparts )
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;
   HYPRE_Int          ierr = 0;

   (fac_data -> max_levels) = nparts - 1;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FACSetMaxIter( void *fac_vdata,
                     HYPRE_Int   max_iter  )
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;
   HYPRE_Int          ierr = 0;

   (fac_data -> max_cycles) = max_iter;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FACSetRelChange( void *fac_vdata,
                       HYPRE_Int   rel_change  )
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;
   HYPRE_Int          ierr = 0;

   (fac_data -> rel_change) = rel_change;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetZeroGuess
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FACSetZeroGuess( void *fac_vdata,
                       HYPRE_Int   zero_guess )
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;
   HYPRE_Int          ierr = 0;

   (fac_data -> zero_guess) = zero_guess;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FACSetRelaxType( void *fac_vdata,
                       HYPRE_Int   relax_type )
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;
   HYPRE_Int          ierr = 0;

   (fac_data -> relax_type) = relax_type;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetJacobiWeight
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_FACSetJacobiWeight( void  *fac_vdata,
                          HYPRE_Real weight )
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;

   (fac_data -> jacobi_weight)    = weight;
   (fac_data -> usr_jacobi_weight) = 1;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetNumPreRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FACSetNumPreSmooth( void *fac_vdata,
                          HYPRE_Int   num_pre_smooth )
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;
   HYPRE_Int          ierr = 0;

   (fac_data -> num_pre_smooth) = num_pre_smooth;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetNumPostRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FACSetNumPostSmooth( void *fac_vdata,
                           HYPRE_Int   num_post_smooth )
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;
   HYPRE_Int          ierr = 0;

   (fac_data -> num_post_smooth) = num_post_smooth;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetCoarseSolverType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FACSetCoarseSolverType( void *fac_vdata,
                              HYPRE_Int   csolver_type)
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;
   HYPRE_Int          ierr = 0;

   (fac_data -> csolver_type) = csolver_type;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FACSetLogging( void *fac_vdata,
                     HYPRE_Int   logging)
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;
   HYPRE_Int          ierr = 0;

   (fac_data -> logging) = logging;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SysFACGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FACGetNumIterations( void *fac_vdata,
                           HYPRE_Int  *num_iterations )
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;
   HYPRE_Int          ierr = 0;

   *num_iterations = (fac_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACPrintLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FACPrintLogging( void *fac_vdata,
                       HYPRE_Int   myid)
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;
   HYPRE_Int          ierr = 0;
   HYPRE_Int          i;
   HYPRE_Int          num_iterations  = (fac_data -> num_iterations);
   HYPRE_Int          logging   = (fac_data -> logging);
   HYPRE_Real        *norms     = (fac_data -> norms);
   HYPRE_Real        *rel_norms = (fac_data -> rel_norms);

   if (myid == 0)
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

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_FACGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FACGetFinalRelativeResidualNorm( void   *fac_vdata,
                                       HYPRE_Real *relative_residual_norm )
{
   hypre_FACData *fac_data = (hypre_FACData *)fac_vdata;

   HYPRE_Int          max_iter        = (fac_data -> max_cycles);
   HYPRE_Int          num_iterations  = (fac_data -> num_iterations);
   HYPRE_Int          logging         = (fac_data -> logging);
   HYPRE_Real        *rel_norms       = (fac_data -> rel_norms);

   HYPRE_Int          ierr = 0;


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

