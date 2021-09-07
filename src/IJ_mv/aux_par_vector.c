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

   vector = hypre_CTAlloc(hypre_AuxParVector, 1, HYPRE_MEMORY_HOST);

   /* set defaults */
   hypre_AuxParVectorMaxOffProcElmts(vector) = 0;
   hypre_AuxParVectorCurrentOffProcElmts(vector) = 0;
   /* stash for setting or adding off processor values */
   hypre_AuxParVectorOffProcI(vector) = NULL;
   hypre_AuxParVectorOffProcData(vector) = NULL;
   hypre_AuxParVectorMemoryLocation(vector) = HYPRE_MEMORY_HOST;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   hypre_AuxParVectorMaxStackElmts(vector) = 0;
   hypre_AuxParVectorCurrentStackElmts(vector) = 0;
   hypre_AuxParVectorStackI(vector) = NULL;
   hypre_AuxParVectorStackData(vector) = NULL;
   hypre_AuxParVectorStackSorA(vector) = NULL;
   hypre_AuxParVectorUsrOffProcElmts(vector) = -1;
   hypre_AuxParVectorInitAllocFactor(vector) = 1.5;
   hypre_AuxParVectorGrowFactor(vector) = 2.0;
#endif

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
      hypre_TFree(hypre_AuxParVectorOffProcI(vector),    HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_AuxParVectorOffProcData(vector), HYPRE_MEMORY_HOST);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      hypre_TFree(hypre_AuxParVectorStackI(vector),    hypre_AuxParVectorMemoryLocation(vector));
      hypre_TFree(hypre_AuxParVectorStackData(vector), hypre_AuxParVectorMemoryLocation(vector));
      hypre_TFree(hypre_AuxParVectorStackSorA(vector), hypre_AuxParVectorMemoryLocation(vector));
#endif

      hypre_TFree(vector, HYPRE_MEMORY_HOST);
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorInitialize_v2
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AuxParVectorInitialize_v2( hypre_AuxParVector *vector, HYPRE_MemoryLocation memory_location )
{
   hypre_AuxParVectorMemoryLocation(vector) = memory_location;

   if ( memory_location == HYPRE_MEMORY_HOST )
   {
      /* CPU assembly */
      /* allocate stash for setting or adding off processor values */
      HYPRE_Int max_off_proc_elmts = hypre_AuxParVectorMaxOffProcElmts(vector);
      if (max_off_proc_elmts > 0)
      {
         hypre_AuxParVectorOffProcI(vector)    = hypre_CTAlloc(HYPRE_BigInt,  max_off_proc_elmts, HYPRE_MEMORY_HOST);
         hypre_AuxParVectorOffProcData(vector) = hypre_CTAlloc(HYPRE_Complex, max_off_proc_elmts, HYPRE_MEMORY_HOST);
      }
   }

   return 0;
}
