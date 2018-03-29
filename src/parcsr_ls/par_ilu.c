/*BHEADER**********************************************************************
 * Copyright (c) 2015,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Incomplete LU factorization smoother
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_ilu.h"
#include <assert.h>

/* Create */
void *
hypre_ILUCreate()
{
  hypre_ParILUData  *ilu_data;

  ilu_data = hypre_CTAlloc(hypre_ParILUData,  1, HYPRE_MEMORY_HOST);

  /* general data */
  (ilu_data -> global_solver) = 0;
  (ilu_data -> matA) = NULL;
  (ilu_data -> matL) = NULL;
  (ilu_data -> matD) = NULL;
  (ilu_data -> matU) = NULL;
  (ilu_data -> droptol) = 1.0e-6;
  (ilu_data -> lfil) = 10;
  (ilu_data -> CF_marker_array) = NULL;

  (ilu_data -> F) = NULL;
  (ilu_data -> U) = NULL;
  (ilu_data -> Utemp) = NULL;
  (ilu_data -> Ftemp) = NULL;
  (ilu_data -> residual) = NULL;
  (ilu_data -> rel_res_norms) = NULL;

  (ilu_data -> num_iterations) = 0;

  (ilu_data -> max_iter) = 20;
  (ilu_data -> tol) = 1.0e-7;

  (ilu_data -> logging) = 0;
  (ilu_data -> print_level) = 0;

  (ilu_data -> l1_norms) = NULL;

  return (void *) ilu_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* Destroy */
HYPRE_Int
hypre_ILUDestroy( void *data )
{
  hypre_ParILUData * ilu_data = (hypre_ParILUData*) data;
 
  /* final residual vector */
  if((ilu_data -> residual))
  {
    hypre_ParVectorDestroy( (ilu_data -> residual) );
    (ilu_data -> residual) = NULL;
  }
  if((ilu_data -> rel_res_norms))
  {
    hypre_TFree( (ilu_data -> rel_res_norms) , HYPRE_MEMORY_HOST);
    (ilu_data -> rel_res_norms) = NULL;
  }
  /* temp vectors for solve phase */
  if((ilu_data -> Utemp))
  {
    hypre_ParVectorDestroy( (ilu_data -> Utemp) );
    (ilu_data -> Utemp) = NULL;
  }
  if((ilu_data -> Ftemp))
  {
    hypre_ParVectorDestroy( (ilu_data -> Ftemp) );
    (ilu_data -> Ftemp) = NULL;
  }
  /* l1_norms */
  if ((ilu_data -> l1_norms))
  {
    hypre_TFree((ilu_data -> l1_norms), HYPRE_MEMORY_HOST);
    (ilu_data -> l1_norms) = NULL;
  }

  /* Factors */
  if(ilu_data -> matL)
  {
    hypre_ParCSRMatrixDestroy((ilu_data -> matL));
    (ilu_data -> matL) = NULL;
  }
  if(ilu_data -> matU)
  {
    hypre_ParCSRMatrixDestroy((ilu_data -> matU));
    (ilu_data -> matU) = NULL;
  }
  if(ilu_data -> matD)
  {
    hypre_TFree((ilu_data -> matD), HYPRE_MEMORY_HOST);
    (ilu_data -> matD) = NULL;
  }    
  /* CF marker array */
  if((ilu_data -> CF_marker_array))
  {
    hypre_TFree((ilu_data -> CF_marker_array), HYPRE_MEMORY_HOST);
    (ilu_data -> CF_marker_array) = NULL;
  }

  /* ilu data */
  hypre_TFree(ilu_data, HYPRE_MEMORY_HOST);

  return hypre_error_flag;
}
