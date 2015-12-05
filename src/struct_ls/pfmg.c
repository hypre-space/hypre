/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.13 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 * hypre_PFMGCreate
 *--------------------------------------------------------------------------*/

void *
hypre_PFMGCreate( MPI_Comm  comm )
{
   hypre_PFMGData *pfmg_data;

   pfmg_data = hypre_CTAlloc(hypre_PFMGData, 1);

   (pfmg_data -> comm)       = comm;
   (pfmg_data -> time_index) = hypre_InitializeTiming("PFMG");

   /* set defaults */
   (pfmg_data -> tol)              = 1.0e-06;
   (pfmg_data -> max_iter)         = 200;
   (pfmg_data -> rel_change)       = 0;
   (pfmg_data -> zero_guess)       = 0;
   (pfmg_data -> max_levels)       = 0;
   (pfmg_data -> dxyz)[0]          = 0.0;
   (pfmg_data -> dxyz)[1]          = 0.0;
   (pfmg_data -> dxyz)[2]          = 0.0;
   (pfmg_data -> relax_type)       = 1;       /* weighted Jacobi */
   (pfmg_data -> jacobi_weight)    = 0.0;  
   (pfmg_data -> usr_jacobi_weight)= 0;     /* no user Jacobi weight */
   (pfmg_data -> rap_type)         = 0;       
   (pfmg_data -> num_pre_relax)    = 1;
   (pfmg_data -> num_post_relax)   = 1;
   (pfmg_data -> skip_relax)       = 1;
   (pfmg_data -> logging)          = 0;
   (pfmg_data -> print_level)      = 0;

   /* initialize */
   (pfmg_data -> num_levels) = -1;

   return (void *) pfmg_data;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGDestroy( void *pfmg_vdata )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;

   HYPRE_Int l;
   HYPRE_Int ierr = 0;

   if (pfmg_data)
   {
      if ((pfmg_data -> logging) > 0)
      {
         hypre_TFree(pfmg_data -> norms);
         hypre_TFree(pfmg_data -> rel_norms);
      }

      if ((pfmg_data -> num_levels) > -1)
      {
         for (l = 0; l < (pfmg_data -> num_levels); l++)
         {
            if (pfmg_data -> active_l[l])
            {
            hypre_PFMGRelaxDestroy(pfmg_data -> relax_data_l[l]);
            }
            hypre_StructMatvecDestroy(pfmg_data -> matvec_data_l[l]);
         }
         for (l = 0; l < ((pfmg_data -> num_levels) - 1); l++)
         {
            hypre_SemiRestrictDestroy(pfmg_data -> restrict_data_l[l]);
            hypre_SemiInterpDestroy(pfmg_data -> interp_data_l[l]);
         }
         hypre_TFree(pfmg_data -> relax_data_l);
         hypre_TFree(pfmg_data -> matvec_data_l);
         hypre_TFree(pfmg_data -> restrict_data_l);
         hypre_TFree(pfmg_data -> interp_data_l);
 
         hypre_StructVectorDestroy(pfmg_data -> tx_l[0]);
         hypre_StructGridDestroy(pfmg_data -> grid_l[0]);
         hypre_StructMatrixDestroy(pfmg_data -> A_l[0]);
         hypre_StructVectorDestroy(pfmg_data -> b_l[0]);
         hypre_StructVectorDestroy(pfmg_data -> x_l[0]);
         for (l = 0; l < ((pfmg_data -> num_levels) - 1); l++)
         {
            hypre_StructGridDestroy(pfmg_data -> grid_l[l+1]);
            hypre_StructGridDestroy(pfmg_data -> P_grid_l[l+1]);
            hypre_StructMatrixDestroy(pfmg_data -> A_l[l+1]);
            hypre_StructMatrixDestroy(pfmg_data -> P_l[l]);
            hypre_StructVectorDestroy(pfmg_data -> b_l[l+1]);
            hypre_StructVectorDestroy(pfmg_data -> x_l[l+1]);
            hypre_StructVectorDestroy(pfmg_data -> tx_l[l+1]);
         }
         hypre_SharedTFree(pfmg_data -> data);
         hypre_TFree(pfmg_data -> cdir_l);
         hypre_TFree(pfmg_data -> active_l);
         hypre_TFree(pfmg_data -> grid_l);
         hypre_TFree(pfmg_data -> P_grid_l);
         hypre_TFree(pfmg_data -> A_l);
         hypre_TFree(pfmg_data -> P_l);
         hypre_TFree(pfmg_data -> RT_l);
         hypre_TFree(pfmg_data -> b_l);
         hypre_TFree(pfmg_data -> x_l);
         hypre_TFree(pfmg_data -> tx_l);
      }
 
      hypre_FinalizeTiming(pfmg_data -> time_index);
      hypre_TFree(pfmg_data);
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetTol, hypre_PFMGGetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetTol( void   *pfmg_vdata,
                  double  tol       )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   (pfmg_data -> tol) = tol;
 
   return ierr;
}

HYPRE_Int
hypre_PFMGGetTol( void   *pfmg_vdata,
                  double *tol       )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   *tol = (pfmg_data -> tol);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetMaxIter, hypre_PFMGGetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetMaxIter( void *pfmg_vdata,
                      HYPRE_Int   max_iter  )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   (pfmg_data -> max_iter) = max_iter;
 
   return ierr;
}

HYPRE_Int
hypre_PFMGGetMaxIter( void *pfmg_vdata,
                      HYPRE_Int * max_iter  )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   *max_iter = (pfmg_data -> max_iter);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetMaxLevels, hypre_PFMGGetMaxLevels
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetMaxLevels( void *pfmg_vdata,
                        HYPRE_Int   max_levels  )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   (pfmg_data -> max_levels) = max_levels;
 
   return ierr;
}

HYPRE_Int
hypre_PFMGGetMaxLevels( void *pfmg_vdata,
                        HYPRE_Int * max_levels  )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   *max_levels = (pfmg_data -> max_levels);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetRelChange, hypre_PFMGGetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetRelChange( void *pfmg_vdata,
                        HYPRE_Int   rel_change  )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   (pfmg_data -> rel_change) = rel_change;
 
   return ierr;
}

HYPRE_Int
hypre_PFMGGetRelChange( void *pfmg_vdata,
                        HYPRE_Int * rel_change  )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   *rel_change = (pfmg_data -> rel_change);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetZeroGuess, hypre_PFMGGetZeroGuess
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_PFMGSetZeroGuess( void *pfmg_vdata,
                        HYPRE_Int   zero_guess )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   (pfmg_data -> zero_guess) = zero_guess;
 
   return ierr;
}

HYPRE_Int
hypre_PFMGGetZeroGuess( void *pfmg_vdata,
                        HYPRE_Int * zero_guess )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   *zero_guess = (pfmg_data -> zero_guess);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetRelaxType, hypre_PFMGGetRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetRelaxType( void *pfmg_vdata,
                        HYPRE_Int   relax_type )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   (pfmg_data -> relax_type) = relax_type;
 
   return ierr;
}

HYPRE_Int
hypre_PFMGGetRelaxType( void *pfmg_vdata,
                        HYPRE_Int * relax_type )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   *relax_type = (pfmg_data -> relax_type);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetJacobiWeight, hypre_PFMGGetJacobiWeight
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_PFMGSetJacobiWeight( void  *pfmg_vdata,
                           double weight )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;

   (pfmg_data -> jacobi_weight)    = weight;
   (pfmg_data -> usr_jacobi_weight)= 1;
                                                                                                                                      
   return hypre_error_flag;
}

HYPRE_Int
hypre_PFMGGetJacobiWeight( void  *pfmg_vdata,
                           double *weight )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;

   *weight = (pfmg_data -> jacobi_weight);

   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_PFMGSetRAPType, hypre_PFMGGetRAPType
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetRAPType( void *pfmg_vdata,
                      HYPRE_Int   rap_type )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   (pfmg_data -> rap_type) = rap_type;
 
   return ierr;
}

HYPRE_Int
hypre_PFMGGetRAPType( void *pfmg_vdata,
                      HYPRE_Int * rap_type )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   *rap_type = (pfmg_data -> rap_type);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetNumPreRelax, hypre_PFMGGetNumPreRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetNumPreRelax( void *pfmg_vdata,
                          HYPRE_Int   num_pre_relax )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   (pfmg_data -> num_pre_relax) = num_pre_relax;
 
   return ierr;
}

HYPRE_Int
hypre_PFMGGetNumPreRelax( void *pfmg_vdata,
                          HYPRE_Int * num_pre_relax )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   *num_pre_relax = (pfmg_data -> num_pre_relax);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetNumPostRelax, hypre_PFMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetNumPostRelax( void *pfmg_vdata,
                           HYPRE_Int   num_post_relax )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   (pfmg_data -> num_post_relax) = num_post_relax;
 
   return ierr;
}

HYPRE_Int
hypre_PFMGGetNumPostRelax( void *pfmg_vdata,
                           HYPRE_Int * num_post_relax )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   *num_post_relax = (pfmg_data -> num_post_relax);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetNumSkipRelax, hypre_PFMGGetNumSkipRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetSkipRelax( void *pfmg_vdata,
                        HYPRE_Int  skip_relax )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   (pfmg_data -> skip_relax) = skip_relax;
 
   return ierr;
}

HYPRE_Int
hypre_PFMGGetSkipRelax( void *pfmg_vdata,
                        HYPRE_Int *skip_relax )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   *skip_relax = (pfmg_data -> skip_relax);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetDxyz
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetDxyz( void   *pfmg_vdata,
                   double *dxyz       )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;

   (pfmg_data -> dxyz[0]) = dxyz[0];
   (pfmg_data -> dxyz[1]) = dxyz[1];
   (pfmg_data -> dxyz[2]) = dxyz[2];
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetLogging, hypre_PFMGGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetLogging( void *pfmg_vdata,
                      HYPRE_Int   logging)
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   (pfmg_data -> logging) = logging;
 
   return ierr;
}

HYPRE_Int
hypre_PFMGGetLogging( void *pfmg_vdata,
                      HYPRE_Int * logging)
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   *logging = (pfmg_data -> logging);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGSetPrintLevel, hypre_PFMGGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGSetPrintLevel( void *pfmg_vdata,
                         HYPRE_Int   print_level)
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   (pfmg_data -> print_level) = print_level;
 
   return ierr;
}

HYPRE_Int
hypre_PFMGGetPrintLevel( void *pfmg_vdata,
                         HYPRE_Int * print_level)
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
 
   *print_level = (pfmg_data -> print_level);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGGetNumIterations( void *pfmg_vdata,
                            HYPRE_Int  *num_iterations )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;

   *num_iterations = (pfmg_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGPrintLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGPrintLogging( void *pfmg_vdata,
                        HYPRE_Int   myid)
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;
   HYPRE_Int       ierr = 0;
   HYPRE_Int       i;
   HYPRE_Int       num_iterations  = (pfmg_data -> num_iterations);
   HYPRE_Int       logging   = (pfmg_data -> logging);
   HYPRE_Int    print_level  = (pfmg_data -> print_level);
   double         *norms     = (pfmg_data -> norms);
   double         *rel_norms = (pfmg_data -> rel_norms);

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
  
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_PFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMGGetFinalRelativeResidualNorm( void   *pfmg_vdata,
                                        double *relative_residual_norm )
{
   hypre_PFMGData *pfmg_data = pfmg_vdata;

   HYPRE_Int       max_iter        = (pfmg_data -> max_iter);
   HYPRE_Int       num_iterations  = (pfmg_data -> num_iterations);
   HYPRE_Int       logging         = (pfmg_data -> logging);
   double         *rel_norms       = (pfmg_data -> rel_norms);
            
   HYPRE_Int       ierr = 0;

   
   if (logging > 0)
   {
      if (max_iter == 0)
      {
         ierr = 1;
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
   
   return ierr;
}


