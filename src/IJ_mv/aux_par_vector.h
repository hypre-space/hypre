/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for Auxiliary Parallel Vector data structures
 *
 * Note: this vector currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_AUX_PAR_VECTOR_HEADER
#define hypre_AUX_PAR_VECTOR_HEADER

/*--------------------------------------------------------------------------
 * Auxiliary Parallel Vector
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int            max_off_proc_elmts;      /* length of off processor stash for
                                                    SetValues and AddToValues*/
   HYPRE_Int            current_off_proc_elmts;  /* current no. of elements stored in stash */
   HYPRE_BigInt        *off_proc_i;              /* contains column indices */
   HYPRE_Complex       *off_proc_data;           /* contains corresponding data */

   HYPRE_MemoryLocation memory_location;

#if defined(HYPRE_USING_GPU)
   HYPRE_Int            max_stack_elmts;      /* length of stash for SetValues and AddToValues*/
   HYPRE_Int            current_stack_elmts;  /* current no. of elements stored in stash */
   HYPRE_BigInt        *stack_i;              /* contains row indices */
   HYPRE_BigInt        *stack_voff;           /* contains vector offsets for multivectors */
   HYPRE_Complex       *stack_data;           /* contains corresponding data */
   char                *stack_sora;
   HYPRE_Int            usr_off_proc_elmts;   /* the num of off-proc elements usr guided */
   HYPRE_Real           init_alloc_factor;
   HYPRE_Real           grow_factor;
#endif
} hypre_AuxParVector;

/*--------------------------------------------------------------------------
 * Accessor functions for the Parallel Vector structure
 *--------------------------------------------------------------------------*/

#define hypre_AuxParVectorMaxOffProcElmts(vector)      ((vector) -> max_off_proc_elmts)
#define hypre_AuxParVectorCurrentOffProcElmts(vector)  ((vector) -> current_off_proc_elmts)
#define hypre_AuxParVectorOffProcI(vector)             ((vector) -> off_proc_i)
#define hypre_AuxParVectorOffProcData(vector)          ((vector) -> off_proc_data)

#define hypre_AuxParVectorMemoryLocation(vector)       ((vector) -> memory_location)

#if defined(HYPRE_USING_GPU)
#define hypre_AuxParVectorMaxStackElmts(vector)        ((vector) -> max_stack_elmts)
#define hypre_AuxParVectorCurrentStackElmts(vector)    ((vector) -> current_stack_elmts)
#define hypre_AuxParVectorStackI(vector)               ((vector) -> stack_i)
#define hypre_AuxParVectorStackVoff(vector)            ((vector) -> stack_voff)
#define hypre_AuxParVectorStackData(vector)            ((vector) -> stack_data)
#define hypre_AuxParVectorStackSorA(vector)            ((vector) -> stack_sora)
#define hypre_AuxParVectorUsrOffProcElmts(vector)      ((vector) -> usr_off_proc_elmts)
#define hypre_AuxParVectorInitAllocFactor(vector)      ((vector) -> init_alloc_factor)
#define hypre_AuxParVectorGrowFactor(vector)           ((vector) -> grow_factor)
#endif

#endif /* #ifndef hypre_AUX_PAR_VECTOR_HEADER */
