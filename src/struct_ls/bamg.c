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

#include "_hypre_struct_ls.h"
#include "bamg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void *
hypre_BAMGCreate( MPI_Comm  comm )
{
   hypre_BAMGData *bamg_data;

   bamg_data = hypre_CTAlloc(hypre_BAMGData, 1);

   (bamg_data -> comm)       = comm;
   (bamg_data -> time_index) = hypre_InitializeTiming("BAMG");

   /* set defaults */
   (bamg_data -> tol)               = 1.0e-06;
   (bamg_data -> max_iter)          = 200;
   (bamg_data -> rel_change)        = 0;
   (bamg_data -> zero_guess)        = 0;
   (bamg_data -> max_levels)        = 0;
   (bamg_data -> dxyz)[0]           = 0.0;
   (bamg_data -> dxyz)[1]           = 0.0;
   (bamg_data -> dxyz)[2]           = 0.0;
   (bamg_data -> relax_type)        = 1;       /* 1 -> weighted Jacobi */
   (bamg_data -> jacobi_weight)     = 0.0;
   (bamg_data -> usr_jacobi_weight) = 0;       /* no user Jacobi weight */
   (bamg_data -> num_pre_relax)     = 1;
   (bamg_data -> num_post_relax)    = 1;
   (bamg_data -> skip_relax)        = 1;
   (bamg_data -> logging)           = 0;
   (bamg_data -> print_level)       = 0;

   /* initialize */
   (bamg_data -> num_levels) = -1;

   return (void *) bamg_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGDestroy( void *bamg_vdata )
{
   hypre_BAMGData *bamg_data = bamg_vdata;

   HYPRE_Int l;

   if (bamg_data)
   {
      if ((bamg_data -> logging) > 0)
      {
         hypre_TFree(bamg_data -> norms);
         hypre_TFree(bamg_data -> rel_norms);
      }

      if ((bamg_data -> num_levels) > -1)
      {
         for (l = 0; l < (bamg_data -> num_levels); l++)
         {
            if (bamg_data -> active_l[l])
            {
               hypre_BAMGRelaxDestroy(bamg_data -> relax_data_l[l]);
            }
            hypre_StructMatvecDestroy(bamg_data -> matvec_data_l[l]);
         }
         for (l = 0; l < ((bamg_data -> num_levels) - 1); l++)
         {
            hypre_SemiRestrictDestroy(bamg_data -> restrict_data_l[l]);
            hypre_SemiInterpDestroy(bamg_data -> interp_data_l[l]);
         }
         hypre_TFree(bamg_data -> relax_data_l);
         hypre_TFree(bamg_data -> matvec_data_l);
         hypre_TFree(bamg_data -> restrict_data_l);
         hypre_TFree(bamg_data -> interp_data_l);
 
         hypre_StructVectorDestroy(bamg_data -> tx_l[0]);
         hypre_StructGridDestroy(bamg_data -> grid_l[0]);
         hypre_StructMatrixDestroy(bamg_data -> A_l[0]);
         hypre_StructVectorDestroy(bamg_data -> b_l[0]);
         hypre_StructVectorDestroy(bamg_data -> x_l[0]);
         for (l = 0; l < ((bamg_data -> num_levels) - 1); l++)
         {
            hypre_StructGridDestroy(bamg_data -> grid_l[l+1]);
            hypre_StructGridDestroy(bamg_data -> P_grid_l[l+1]);
            hypre_StructMatrixDestroy(bamg_data -> A_l[l+1]);
            hypre_StructMatrixDestroy(bamg_data -> P_l[l]);
            hypre_StructVectorDestroy(bamg_data -> b_l[l+1]);
            hypre_StructVectorDestroy(bamg_data -> x_l[l+1]);
            hypre_StructVectorDestroy(bamg_data -> tx_l[l+1]);
         }
         hypre_SharedTFree(bamg_data -> data);
         hypre_TFree(bamg_data -> cdir_l);
         hypre_TFree(bamg_data -> active_l);
         hypre_TFree(bamg_data -> grid_l);
         hypre_TFree(bamg_data -> P_grid_l);
         hypre_TFree(bamg_data -> A_l);
         hypre_TFree(bamg_data -> P_l);
         hypre_TFree(bamg_data -> RT_l);
         hypre_TFree(bamg_data -> b_l);
         hypre_TFree(bamg_data -> x_l);
         hypre_TFree(bamg_data -> tx_l);
      }
 
      hypre_FinalizeTiming(bamg_data -> time_index);
      hypre_TFree(bamg_data);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGSetTol( void   *bamg_vdata,
                  HYPRE_Real  tol       )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   (bamg_data -> tol) = tol;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_BAMGGetTol( void   *bamg_vdata,
                  HYPRE_Real *tol       )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   *tol = (bamg_data -> tol);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGSetMaxIter( void *bamg_vdata,
                      HYPRE_Int   max_iter  )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   (bamg_data -> max_iter) = max_iter;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_BAMGGetMaxIter( void *bamg_vdata,
                      HYPRE_Int * max_iter  )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   *max_iter = (bamg_data -> max_iter);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGSetMaxLevels( void *bamg_vdata,
                        HYPRE_Int   max_levels  )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   (bamg_data -> max_levels) = max_levels;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_BAMGGetMaxLevels( void *bamg_vdata,
                        HYPRE_Int * max_levels  )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   *max_levels = (bamg_data -> max_levels);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGSetRelChange( void *bamg_vdata,
                        HYPRE_Int   rel_change  )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   (bamg_data -> rel_change) = rel_change;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_BAMGGetRelChange( void *bamg_vdata,
                        HYPRE_Int * rel_change  )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   *rel_change = (bamg_data -> rel_change);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_BAMGSetZeroGuess( void *bamg_vdata,
                        HYPRE_Int   zero_guess )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   (bamg_data -> zero_guess) = zero_guess;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_BAMGGetZeroGuess( void *bamg_vdata,
                        HYPRE_Int * zero_guess )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   *zero_guess = (bamg_data -> zero_guess);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGSetRelaxType( void *bamg_vdata,
                        HYPRE_Int   relax_type )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   (bamg_data -> relax_type) = relax_type;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_BAMGGetRelaxType( void *bamg_vdata,
                        HYPRE_Int * relax_type )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   *relax_type = (bamg_data -> relax_type);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_BAMGSetJacobiWeight( void  *bamg_vdata,
                           HYPRE_Real weight )
{
   hypre_BAMGData *bamg_data = bamg_vdata;

   (bamg_data -> jacobi_weight)    = weight;
   (bamg_data -> usr_jacobi_weight)= 1;
                                                                                                                                      
   return hypre_error_flag;
}

HYPRE_Int
hypre_BAMGGetJacobiWeight( void  *bamg_vdata,
                           HYPRE_Real *weight )
{
   hypre_BAMGData *bamg_data = bamg_vdata;

   *weight = (bamg_data -> jacobi_weight);

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGSetNumPreRelax( void *bamg_vdata,
                          HYPRE_Int   num_pre_relax )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   (bamg_data -> num_pre_relax) = num_pre_relax;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_BAMGGetNumPreRelax( void *bamg_vdata,
                          HYPRE_Int * num_pre_relax )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   *num_pre_relax = (bamg_data -> num_pre_relax);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGSetNumPostRelax( void *bamg_vdata,
                           HYPRE_Int   num_post_relax )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   (bamg_data -> num_post_relax) = num_post_relax;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_BAMGGetNumPostRelax( void *bamg_vdata,
                           HYPRE_Int * num_post_relax )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   *num_post_relax = (bamg_data -> num_post_relax);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGSetSkipRelax( void *bamg_vdata,
                        HYPRE_Int  skip_relax )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   (bamg_data -> skip_relax) = skip_relax;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_BAMGGetSkipRelax( void *bamg_vdata,
                        HYPRE_Int *skip_relax )
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   *skip_relax = (bamg_data -> skip_relax);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGSetDxyz( void   *bamg_vdata,
                   HYPRE_Real *dxyz       )
{
   hypre_BAMGData *bamg_data = bamg_vdata;

   (bamg_data -> dxyz[0]) = dxyz[0];
   (bamg_data -> dxyz[1]) = dxyz[1];
   (bamg_data -> dxyz[2]) = dxyz[2];
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGSetLogging( void *bamg_vdata,
                      HYPRE_Int   logging)
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   (bamg_data -> logging) = logging;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_BAMGGetLogging( void *bamg_vdata,
                      HYPRE_Int * logging)
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   *logging = (bamg_data -> logging);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGSetPrintLevel( void *bamg_vdata,
                         HYPRE_Int   print_level)
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   (bamg_data -> print_level) = print_level;
 
   return hypre_error_flag;
}

HYPRE_Int
hypre_BAMGGetPrintLevel( void *bamg_vdata,
                         HYPRE_Int * print_level)
{
   hypre_BAMGData *bamg_data = bamg_vdata;
 
   *print_level = (bamg_data -> print_level);
 
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGGetNumIterations( void *bamg_vdata,
                            HYPRE_Int  *num_iterations )
{
   hypre_BAMGData *bamg_data = bamg_vdata;

   *num_iterations = (bamg_data -> num_iterations);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGPrintLogging( void *bamg_vdata,
                        HYPRE_Int   myid)
{
   hypre_BAMGData *bamg_data = bamg_vdata;
   HYPRE_Int       i;
   HYPRE_Int       num_iterations  = (bamg_data -> num_iterations);
   HYPRE_Int       logging   = (bamg_data -> logging);
   HYPRE_Int    print_level  = (bamg_data -> print_level);
   HYPRE_Real     *norms     = (bamg_data -> norms);
   HYPRE_Real     *rel_norms = (bamg_data -> rel_norms);

   if (myid == 0)
   {
      if (print_level > 0)
      {
         if (logging > 0)
         {
            for (i = 0; i < num_iterations; i++)
            {
               hypre_printf("Residual norm[%d] = %e   ",i,norms[i]);
               hypre_printf("Relative residual norm[%d] = %e\n",i,rel_norms[i]);
            }
         }
      }
   }
  
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_BAMGGetFinalRelativeResidualNorm( void   *bamg_vdata,
                                        HYPRE_Real *relative_residual_norm )
{
   hypre_BAMGData *bamg_data = bamg_vdata;

   HYPRE_Int       max_iter        = (bamg_data -> max_iter);
   HYPRE_Int       num_iterations  = (bamg_data -> num_iterations);
   HYPRE_Int       logging         = (bamg_data -> logging);
   HYPRE_Real     *rel_norms       = (bamg_data -> rel_norms);
            
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


