/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Member functions for hypre_AuxParVector class.
 *
 *****************************************************************************/

#include "_hypre_IJ_mv.h"
#include "aux_par_vector.h"

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AuxParVectorCreate( hypre_AuxParVector **aux_vector)
{
   hypre_AuxParVector  *vector;
   
   vector = hypre_CTAlloc(hypre_AuxParVector,  1, HYPRE_MEMORY_HOST);
  
   /* set defaults */
   hypre_AuxParVectorMaxOffProcElmts(vector) = 0;
   hypre_AuxParVectorCurrentNumElmts(vector) = 0;
   /* stash for setting or adding off processor values */
   hypre_AuxParVectorOffProcI(vector) = NULL;
   hypre_AuxParVectorOffProcData(vector) = NULL;


   *aux_vector = vector;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_AuxParVectorDestroy( hypre_AuxParVector *vector )
{
   HYPRE_Int ierr=0;

   if (vector)
   {
      if (hypre_AuxParVectorOffProcI(vector))
         hypre_TFree(hypre_AuxParVectorOffProcI(vector), HYPRE_MEMORY_HOST);
      if (hypre_AuxParVectorOffProcData(vector))
         hypre_TFree(hypre_AuxParVectorOffProcData(vector), HYPRE_MEMORY_HOST);
      hypre_TFree(vector, HYPRE_MEMORY_HOST);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_AuxParVectorInitialize( hypre_AuxParVector *vector )
{
   HYPRE_Int max_off_proc_elmts = hypre_AuxParVectorMaxOffProcElmts(vector);

   /* allocate stash for setting or adding off processor values */
   if (max_off_proc_elmts > 0)
   {
      hypre_AuxParVectorOffProcI(vector) = hypre_CTAlloc(HYPRE_BigInt, 
                                                         max_off_proc_elmts, HYPRE_MEMORY_HOST);
      hypre_AuxParVectorOffProcData(vector) = hypre_CTAlloc(HYPRE_Complex, 
                                                            max_off_proc_elmts, HYPRE_MEMORY_HOST);
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorSetMaxOffProcElmts
 *--------------------------------------------------------------------------*/

HYPRE_Int 
hypre_AuxParVectorSetMaxOffPRocElmts( hypre_AuxParVector *vector,
                                      HYPRE_Int max_off_proc_elmts )
{
   HYPRE_Int ierr = 0;
   hypre_AuxParVectorMaxOffProcElmts(vector) = max_off_proc_elmts;
   return ierr;
}

