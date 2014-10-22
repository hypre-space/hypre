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
#include "sys_bamg.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  void *
hypre_SysBAMGCreate( MPI_Comm  comm )
{
  hypre_SysBAMGData *sys_bamg_data;

  sys_bamg_data = hypre_CTAlloc(hypre_SysBAMGData, 1);

  (sys_bamg_data -> comm)       = comm;
  (sys_bamg_data -> time_index) = hypre_InitializeTiming("SYS_BAMG");

  /* set defaults */
  (sys_bamg_data -> tol)              = 1.0e-06;
  (sys_bamg_data -> max_iter  )       = 200;
  (sys_bamg_data -> rel_change)       = 0;
  (sys_bamg_data -> zero_guess)       = 0;
  (sys_bamg_data -> max_levels)       = 0;
  (sys_bamg_data -> dxyz)[0]          = 0.0;
  (sys_bamg_data -> dxyz)[1]          = 0.0;
  (sys_bamg_data -> dxyz)[2]          = 0.0;
  (sys_bamg_data -> relax_type)       = 1;       /* weighted Jacobi */
  (sys_bamg_data -> jacobi_weight)    = 0.0;
  (sys_bamg_data -> usr_jacobi_weight)= 0;
  (sys_bamg_data -> num_pre_relax)     = 1;
  (sys_bamg_data -> num_post_relax)    = 1;

  (sys_bamg_data -> num_rtv)           = 10;
  (sys_bamg_data -> num_stv)           = 10;
  (sys_bamg_data -> num_relax_tv)      = 20;

  (sys_bamg_data -> skip_relax)       = 1;
  (sys_bamg_data -> logging)          = 0;
  (sys_bamg_data -> print_level)      = 0;

  /* initialize */
  (sys_bamg_data -> num_levels) = -1;

  return (void *) sys_bamg_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SysBAMGDestroy( void *sys_bamg_vdata )
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  HYPRE_Int l;

  if (sys_bamg_data)
  {
    if ((sys_bamg_data -> logging) > 0)
    {
      hypre_TFree(sys_bamg_data -> norms);
      hypre_TFree(sys_bamg_data -> rel_norms);
    }

    if ((sys_bamg_data -> num_levels) > -1)
    {
      for (l = 0; l < (sys_bamg_data -> num_levels); l++)
      {
        hypre_SysBAMGRelaxDestroy(sys_bamg_data -> relax_data_l[l]);
        hypre_SStructPMatvecDestroy(sys_bamg_data -> matvec_data_l[l]);
      }
      for (l = 0; l < ((sys_bamg_data -> num_levels) - 1); l++)
      {
        hypre_SysSemiRestrictDestroy(sys_bamg_data -> restrict_data_l[l]);
        hypre_SysSemiInterpDestroy(sys_bamg_data -> interp_data_l[l]);
      }
      hypre_TFree(sys_bamg_data -> relax_data_l);
      hypre_TFree(sys_bamg_data -> matvec_data_l);
      hypre_TFree(sys_bamg_data -> restrict_data_l);
      hypre_TFree(sys_bamg_data -> interp_data_l);

      hypre_SStructPVectorDestroy(sys_bamg_data -> tx_l[0]);
      hypre_SStructPMatrixDestroy(sys_bamg_data -> A_l[0]);
      hypre_SStructPVectorDestroy(sys_bamg_data -> b_l[0]);
      hypre_SStructPVectorDestroy(sys_bamg_data -> x_l[0]);
      for (l = 0; l < ((sys_bamg_data -> num_levels) - 1); l++)
      {
        hypre_SStructPGridDestroy(sys_bamg_data -> PGrid_l[l+1]);
        hypre_SStructPGridDestroy(sys_bamg_data -> P_PGrid_l[l+1]);
        hypre_SStructPMatrixDestroy(sys_bamg_data -> A_l[l+1]);
        hypre_SStructPMatrixDestroy(sys_bamg_data -> P_l[l]);
        hypre_SStructPVectorDestroy(sys_bamg_data -> b_l[l+1]);
        hypre_SStructPVectorDestroy(sys_bamg_data -> x_l[l+1]);
        hypre_SStructPVectorDestroy(sys_bamg_data -> tx_l[l+1]);
      }
      hypre_SharedTFree(sys_bamg_data -> data);
      hypre_TFree(sys_bamg_data -> cdir_l);
      hypre_TFree(sys_bamg_data -> active_l);
      hypre_TFree(sys_bamg_data -> PGrid_l);
      hypre_TFree(sys_bamg_data -> P_PGrid_l);
      hypre_TFree(sys_bamg_data -> A_l);
      hypre_TFree(sys_bamg_data -> P_l);
      hypre_TFree(sys_bamg_data -> RT_l);
      hypre_TFree(sys_bamg_data -> b_l);
      hypre_TFree(sys_bamg_data -> x_l);
      hypre_TFree(sys_bamg_data -> tx_l);
    }

    hypre_FinalizeTiming(sys_bamg_data -> time_index);
    hypre_TFree(sys_bamg_data);
  }

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGSetTol( void   *sys_bamg_vdata,
    HYPRE_Real  tol       )
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  (sys_bamg_data -> tol) = tol;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGSetMaxIter( void *sys_bamg_vdata,
    HYPRE_Int   max_iter  )
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  (sys_bamg_data -> max_iter) = max_iter;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGSetRelChange( void *sys_bamg_vdata,
    HYPRE_Int   rel_change  )
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  (sys_bamg_data -> rel_change) = rel_change;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGSetZeroGuess( void *sys_bamg_vdata,
    HYPRE_Int   zero_guess )
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  (sys_bamg_data -> zero_guess) = zero_guess;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGSetRelaxType( void *sys_bamg_vdata,
    HYPRE_Int   relax_type )
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  (sys_bamg_data -> relax_type) = relax_type;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
  HYPRE_Int
hypre_SysBAMGSetJacobiWeight( void  *sys_bamg_vdata,
    HYPRE_Real weight )
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  (sys_bamg_data -> jacobi_weight)    = weight;
  (sys_bamg_data -> usr_jacobi_weight)= 1;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGSetNumPreRelax( void *sys_bamg_vdata,
    HYPRE_Int   num_pre_relax )
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  (sys_bamg_data -> num_pre_relax) = num_pre_relax;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGSetNumPostRelax( void *sys_bamg_vdata,
    HYPRE_Int   num_post_relax )
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  (sys_bamg_data -> num_post_relax) = num_post_relax;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SysBAMGSetNumTv1( void *bamg_vdata, HYPRE_Int num_rtv )
{
  hypre_SysBAMGData *bamg_data = bamg_vdata;

  (bamg_data -> num_rtv) = num_rtv;

  return hypre_error_flag;
}

HYPRE_Int hypre_SysBAMGGetNumTv1( void *bamg_vdata, HYPRE_Int *num_rtv )
{
  hypre_SysBAMGData *bamg_data = bamg_vdata;

  *num_rtv = (bamg_data -> num_rtv);

  return hypre_error_flag;
}

HYPRE_Int hypre_SysBAMGSetNumStv( void *bamg_vdata, HYPRE_Int num_stv )
{
  hypre_SysBAMGData *bamg_data = bamg_vdata;

  (bamg_data -> num_stv) = num_stv;

  return hypre_error_flag;
}

HYPRE_Int hypre_SysBAMGGetNumStv( void *bamg_vdata, HYPRE_Int *num_stv )
{
  hypre_SysBAMGData *bamg_data = bamg_vdata;

  *num_stv = (bamg_data -> num_stv);

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGSetSkipRelax( void *sys_bamg_vdata,
    HYPRE_Int  skip_relax )
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  (sys_bamg_data -> skip_relax) = skip_relax;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGSetDxyz( void   *sys_bamg_vdata,
    HYPRE_Real *dxyz       )
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  (sys_bamg_data -> dxyz[0]) = dxyz[0];
  (sys_bamg_data -> dxyz[1]) = dxyz[1];
  (sys_bamg_data -> dxyz[2]) = dxyz[2];

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGSetLogging( void *sys_bamg_vdata,
    HYPRE_Int   logging)
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  (sys_bamg_data -> logging) = logging;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGSetPrintLevel( void *sys_bamg_vdata,
    HYPRE_Int   print_level)
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  (sys_bamg_data -> print_level) = print_level;

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGGetNumIterations( void *sys_bamg_vdata,
    HYPRE_Int  *num_iterations )
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  *num_iterations = (sys_bamg_data -> num_iterations);

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SysBAMGPrintLogging( void *sys_bamg_vdata,
    HYPRE_Int   myid)
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;
  HYPRE_Int          i;
  HYPRE_Int          num_iterations  = (sys_bamg_data -> num_iterations);
  HYPRE_Int          logging   = (sys_bamg_data -> logging);
  HYPRE_Int          print_level   = (sys_bamg_data -> print_level);
  HYPRE_Real        *norms     = (sys_bamg_data -> norms);
  HYPRE_Real        *rel_norms = (sys_bamg_data -> rel_norms);

  if (myid == 0)
  {
    if (print_level > 0 )
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
hypre_SysBAMGGetFinalRelativeResidualNorm( void   *sys_bamg_vdata,
    HYPRE_Real *relative_residual_norm )
{
  hypre_SysBAMGData *sys_bamg_data = sys_bamg_vdata;

  HYPRE_Int          max_iter        = (sys_bamg_data -> max_iter);
  HYPRE_Int          num_iterations  = (sys_bamg_data -> num_iterations);
  HYPRE_Int          logging         = (sys_bamg_data -> logging);
  HYPRE_Real        *rel_norms       = (sys_bamg_data -> rel_norms);

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


