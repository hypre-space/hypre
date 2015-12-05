/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 *
 *****************************************************************************/

#include "headers.h"
#include "smg.h"

/*--------------------------------------------------------------------------
 * hypre_SMGCreate
 *--------------------------------------------------------------------------*/

void *
hypre_SMGCreate( MPI_Comm  comm )
{
   hypre_SMGData *smg_data;

   smg_data = hypre_CTAlloc(hypre_SMGData, 1);

   (smg_data -> comm)        = comm;
   (smg_data -> time_index)  = hypre_InitializeTiming("SMG");

   /* set defaults */
   (smg_data -> memory_use) = 0;
   (smg_data -> tol)        = 1.0e-06;
   (smg_data -> max_iter)   = 200;
   (smg_data -> rel_change) = 0;
   (smg_data -> zero_guess) = 0;
   (smg_data -> max_levels) = 0;
   (smg_data -> num_pre_relax)  = 1;
   (smg_data -> num_post_relax) = 1;
   (smg_data -> cdir) = 2;
   hypre_SetIndex((smg_data -> base_index), 0, 0, 0);
   hypre_SetIndex((smg_data -> base_stride), 1, 1, 1);
   (smg_data -> logging) = 0;
   (smg_data -> print_level) = 0;

   /* initialize */
   (smg_data -> num_levels) = -1;

   return (void *) smg_data;
}

/*--------------------------------------------------------------------------
 * hypre_SMGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGDestroy( void *smg_vdata )
{
   hypre_SMGData *smg_data = smg_vdata;

   HYPRE_Int l;
   HYPRE_Int ierr = 0;

   if (smg_data)
   {
      if ((smg_data -> logging) > 0)
      {
         hypre_TFree(smg_data -> norms);
         hypre_TFree(smg_data -> rel_norms);
      }

      if ((smg_data -> num_levels) > -1)
      {
         for (l = 0; l < ((smg_data -> num_levels) - 1); l++)
         {
            hypre_SMGRelaxDestroy(smg_data -> relax_data_l[l]);
            hypre_SMGResidualDestroy(smg_data -> residual_data_l[l]);
            hypre_SemiRestrictDestroy(smg_data -> restrict_data_l[l]);
            hypre_SemiInterpDestroy(smg_data -> interp_data_l[l]);
         }
         hypre_SMGRelaxDestroy(smg_data -> relax_data_l[l]);
         if (l == 0)
         {
            hypre_SMGResidualDestroy(smg_data -> residual_data_l[l]);
         }
         hypre_TFree(smg_data -> relax_data_l);
         hypre_TFree(smg_data -> residual_data_l);
         hypre_TFree(smg_data -> restrict_data_l);
         hypre_TFree(smg_data -> interp_data_l);
 
         hypre_StructVectorDestroy(smg_data -> tb_l[0]);
         hypre_StructVectorDestroy(smg_data -> tx_l[0]);
         hypre_StructGridDestroy(smg_data -> grid_l[0]);
         hypre_StructMatrixDestroy(smg_data -> A_l[0]);
         hypre_StructVectorDestroy(smg_data -> b_l[0]);
         hypre_StructVectorDestroy(smg_data -> x_l[0]);
         for (l = 0; l < ((smg_data -> num_levels) - 1); l++)
         {
            hypre_StructGridDestroy(smg_data -> grid_l[l+1]);
            hypre_StructGridDestroy(smg_data -> PT_grid_l[l+1]);
            hypre_StructMatrixDestroy(smg_data -> A_l[l+1]);
            if (smg_data -> PT_l[l] == smg_data -> R_l[l])
            {
               hypre_StructMatrixDestroy(smg_data -> PT_l[l]);
            }
            else
            {
               hypre_StructMatrixDestroy(smg_data -> PT_l[l]);
               hypre_StructMatrixDestroy(smg_data -> R_l[l]);
            }
            hypre_StructVectorDestroy(smg_data -> b_l[l+1]);
            hypre_StructVectorDestroy(smg_data -> x_l[l+1]);
            hypre_StructVectorDestroy(smg_data -> tb_l[l+1]);
            hypre_StructVectorDestroy(smg_data -> tx_l[l+1]);
         }
         hypre_SharedTFree(smg_data -> data);
         hypre_TFree(smg_data -> grid_l);
         hypre_TFree(smg_data -> PT_grid_l);
         hypre_TFree(smg_data -> A_l);
         hypre_TFree(smg_data -> PT_l);
         hypre_TFree(smg_data -> R_l);
         hypre_TFree(smg_data -> b_l);
         hypre_TFree(smg_data -> x_l);
         hypre_TFree(smg_data -> tb_l);
         hypre_TFree(smg_data -> tx_l);
      }
 
      hypre_FinalizeTiming(smg_data -> time_index);
      hypre_TFree(smg_data);
   }

   return(ierr);
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetMemoryUse, hypre_SMGGetMemoryUse
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGSetMemoryUse( void *smg_vdata,
                       HYPRE_Int   memory_use )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   (smg_data -> memory_use) = memory_use;
 
   return ierr;
}

HYPRE_Int
hypre_SMGGetMemoryUse( void *smg_vdata,
                       HYPRE_Int * memory_use )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   *memory_use = (smg_data -> memory_use);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetTol, hypre_SMGGetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGSetTol( void   *smg_vdata,
                 double  tol       )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   (smg_data -> tol) = tol;
 
   return ierr;
}

HYPRE_Int
hypre_SMGGetTol( void   *smg_vdata,
                 double *tol       )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   *tol = (smg_data -> tol);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetMaxIter, hypre_SMGGetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGSetMaxIter( void *smg_vdata,
                     HYPRE_Int   max_iter  )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   (smg_data -> max_iter) = max_iter;
 
   return ierr;
}

HYPRE_Int
hypre_SMGGetMaxIter( void *smg_vdata,
                     HYPRE_Int * max_iter  )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   *max_iter = (smg_data -> max_iter);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetRelChange, hypre_SMGGetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGSetRelChange( void *smg_vdata,
                       HYPRE_Int   rel_change  )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   (smg_data -> rel_change) = rel_change;
 
   return ierr;
}

HYPRE_Int
hypre_SMGGetRelChange( void *smg_vdata,
                       HYPRE_Int * rel_change  )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   *rel_change = (smg_data -> rel_change);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetZeroGuess, hypre_SMGGetZeroGuess
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_SMGSetZeroGuess( void *smg_vdata,
                       HYPRE_Int   zero_guess )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   (smg_data -> zero_guess) = zero_guess;
 
   return ierr;
}

HYPRE_Int
hypre_SMGGetZeroGuess( void *smg_vdata,
                       HYPRE_Int * zero_guess )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   *zero_guess = (smg_data -> zero_guess);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetNumPreRelax, hypre_SMGGetNumPreRelax
 * Note that we require at least 1 pre-relax sweep. 
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGSetNumPreRelax( void *smg_vdata,
                         HYPRE_Int   num_pre_relax )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   (smg_data -> num_pre_relax) = hypre_max(num_pre_relax,1);
 
   return ierr;
}

HYPRE_Int
hypre_SMGGetNumPreRelax( void *smg_vdata,
                         HYPRE_Int * num_pre_relax )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   *num_pre_relax = (smg_data -> num_pre_relax);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetNumPostRelax, hypre_SMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGSetNumPostRelax( void *smg_vdata,
                          HYPRE_Int   num_post_relax )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   (smg_data -> num_post_relax) = num_post_relax;
 
   return ierr;
}

HYPRE_Int
hypre_SMGGetNumPostRelax( void *smg_vdata,
                          HYPRE_Int * num_post_relax )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   *num_post_relax = (smg_data -> num_post_relax);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetBase
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
hypre_SMGSetBase( void        *smg_vdata,
                  hypre_Index  base_index,
                  hypre_Index  base_stride )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      d;
   HYPRE_Int      ierr = 0;
 
   for (d = 0; d < 3; d++)
   {
      hypre_IndexD((smg_data -> base_index),  d) =
         hypre_IndexD(base_index,  d);
      hypre_IndexD((smg_data -> base_stride), d) =
         hypre_IndexD(base_stride, d);
   }
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetLogging, hypre_SMGGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGSetLogging( void *smg_vdata,
                     HYPRE_Int   logging)
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   (smg_data -> logging) = logging;
 
   return ierr;
}

HYPRE_Int
hypre_SMGGetLogging( void *smg_vdata,
                     HYPRE_Int * logging)
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   *logging = (smg_data -> logging);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetPrintLevel, hypre_SMGGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGSetPrintLevel( void *smg_vdata,
                        HYPRE_Int   print_level)
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   (smg_data -> print_level) = print_level;
 
   return ierr;
}

HYPRE_Int
hypre_SMGGetPrintLevel( void *smg_vdata,
                        HYPRE_Int * print_level)
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;
 
   *print_level = (smg_data -> print_level);
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGGetNumIterations( void *smg_vdata,
                           HYPRE_Int  *num_iterations )
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int      ierr = 0;

   *num_iterations = (smg_data -> num_iterations);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGPrintLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGPrintLogging( void *smg_vdata,
                       HYPRE_Int   myid)
{
   hypre_SMGData *smg_data = smg_vdata;
   HYPRE_Int    ierr = 0;
   HYPRE_Int    i;
   HYPRE_Int    num_iterations  = (smg_data -> num_iterations);
   HYPRE_Int    logging   = (smg_data -> logging);
   HYPRE_Int    print_level  = (smg_data -> print_level);
   double      *norms     = (smg_data -> norms);
   double      *rel_norms = (smg_data -> rel_norms);

   
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
 * hypre_SMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGGetFinalRelativeResidualNorm( void   *smg_vdata,
                                       double *relative_residual_norm )
{
   hypre_SMGData *smg_data = smg_vdata;

   HYPRE_Int      max_iter        = (smg_data -> max_iter);
   HYPRE_Int      num_iterations  = (smg_data -> num_iterations);
   HYPRE_Int      logging         = (smg_data -> logging);
   double        *rel_norms       = (smg_data -> rel_norms);

   HYPRE_Int      ierr = -1;

   
   if (logging > 0)
   {
      if (num_iterations == max_iter)
      {
         *relative_residual_norm = rel_norms[num_iterations-1];
      }
      else
      {
         *relative_residual_norm = rel_norms[num_iterations];
      }

      ierr = 0;
   }
   
   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SMGSetStructVectorConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGSetStructVectorConstantValues( hypre_StructVector *vector,
                                        double              values,
                                        hypre_BoxArray     *box_array,
                                        hypre_Index         stride    )
{
   HYPRE_Int    ierr = 0;

   hypre_Box          *v_data_box;

   HYPRE_Int           vi;
   double             *vp;

   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;

   HYPRE_Int           loopi, loopj, loopk;
   HYPRE_Int           i;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   hypre_ForBoxI(i, box_array)
      {
         box   = hypre_BoxArrayBox(box_array, i);
         start = hypre_BoxIMin(box);

         v_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
         vp = hypre_StructVectorBoxData(vector, i);

         hypre_BoxGetStrideSize(box, stride, loop_size);

         hypre_BoxLoop1Begin(loop_size,
                             v_data_box, start, stride, vi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,vi
#include "hypre_box_smp_forloop.h"
         hypre_BoxLoop1For(loopi, loopj, loopk, vi)
            {
               vp[vi] = values;
            }
         hypre_BoxLoop1End(vi);
      }

   return ierr;
}


