/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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
hypre_AuxParVectorCreate( hypre_AuxParVector **aux_vector_ptr)
{
   hypre_AuxParVector  *aux_vector;

   aux_vector = hypre_CTAlloc(hypre_AuxParVector, 1, HYPRE_MEMORY_HOST);

   /* set defaults */
   hypre_AuxParVectorMaxOffProcElmts(aux_vector)     = 0;
   hypre_AuxParVectorCurrentOffProcElmts(aux_vector) = 0;

   /* stash for setting or adding off processor values */
   hypre_AuxParVectorOffProcI(aux_vector)            = NULL;
   hypre_AuxParVectorOffProcData(aux_vector)         = NULL;
   hypre_AuxParVectorMemoryLocation(aux_vector)      = HYPRE_MEMORY_HOST;

#if defined(HYPRE_USING_GPU)
   hypre_AuxParVectorMaxStackElmts(aux_vector)       = 0;
   hypre_AuxParVectorCurrentStackElmts(aux_vector)   = 0;
   hypre_AuxParVectorStackI(aux_vector)              = NULL;
   hypre_AuxParVectorStackVoff(aux_vector)           = NULL;
   hypre_AuxParVectorStackData(aux_vector)           = NULL;
   hypre_AuxParVectorStackSorA(aux_vector)           = NULL;
   hypre_AuxParVectorUsrOffProcElmts(aux_vector)     = -1;
   hypre_AuxParVectorInitAllocFactor(aux_vector)     = 1.5;
   hypre_AuxParVectorGrowFactor(aux_vector)          = 2.0;
#endif

   *aux_vector_ptr = aux_vector;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AuxParVectorDestroy( hypre_AuxParVector *aux_vector )
{
   if (aux_vector)
   {
      hypre_TFree(hypre_AuxParVectorOffProcI(aux_vector),    HYPRE_MEMORY_HOST);
      hypre_TFree(hypre_AuxParVectorOffProcData(aux_vector), HYPRE_MEMORY_HOST);

#if defined(HYPRE_USING_GPU)
      HYPRE_MemoryLocation  memory_location = hypre_AuxParVectorMemoryLocation(aux_vector);

      hypre_TFree(hypre_AuxParVectorStackI(aux_vector),    memory_location);
      hypre_TFree(hypre_AuxParVectorStackVoff(aux_vector), memory_location);
      hypre_TFree(hypre_AuxParVectorStackData(aux_vector), memory_location);
      hypre_TFree(hypre_AuxParVectorStackSorA(aux_vector), memory_location);
#endif

      hypre_TFree(aux_vector, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AuxParVectorInitialize_v2
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AuxParVectorInitialize_v2( hypre_AuxParVector   *aux_vector,
                                 HYPRE_MemoryLocation  memory_location )
{
   hypre_AuxParVectorMemoryLocation(aux_vector) = memory_location;

   if (memory_location == HYPRE_MEMORY_HOST)
   {
      /* CPU assembly */
      /* allocate stash for setting or adding off processor values */
      HYPRE_Int max_off_proc_elmts = hypre_AuxParVectorMaxOffProcElmts(aux_vector);
      if (max_off_proc_elmts > 0)
      {
         hypre_AuxParVectorOffProcI(aux_vector)    = hypre_CTAlloc(HYPRE_BigInt,  max_off_proc_elmts,
                                                                   HYPRE_MEMORY_HOST);
         hypre_AuxParVectorOffProcData(aux_vector) = hypre_CTAlloc(HYPRE_Complex, max_off_proc_elmts,
                                                                   HYPRE_MEMORY_HOST);
      }
   }

   return hypre_error_flag;
}
