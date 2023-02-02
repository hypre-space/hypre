/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructCycRedCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   *solver = ( (HYPRE_StructSolver) hypre_CyclicReductionCreate( comm ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructCycRedDestroy( HYPRE_StructSolver solver )
{
   return ( hypre_CyclicReductionDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructCycRedSetup( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return ( hypre_CyclicReductionSetup( (void *) solver,
                                        (hypre_StructMatrix *) A,
                                        (hypre_StructVector *) b,
                                        (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructCycRedSolve( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return ( hypre_CyclicReduction( (void *) solver,
                                   (hypre_StructMatrix *) A,
                                   (hypre_StructVector *) b,
                                   (hypre_StructVector *) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructCycRedSetTDim( HYPRE_StructSolver solver,
                           HYPRE_Int          tdim )
{
   return ( hypre_CyclicReductionSetCDir( (void *) solver, tdim ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructCycRedSetBase( HYPRE_StructSolver solver,
                           HYPRE_Int          ndim,
                           HYPRE_Int         *base_index,
                           HYPRE_Int         *base_stride )
{
   hypre_Index  new_base_index;
   hypre_Index  new_base_stride;

   HYPRE_Int    d;

   hypre_SetIndex(new_base_index, 0);
   hypre_SetIndex(new_base_stride, 1);
   for (d = 0; d < ndim; d++)
   {
      hypre_IndexD(new_base_index, d)  = base_index[d];
      hypre_IndexD(new_base_stride, d) = base_stride[d];
   }

   return ( hypre_CyclicReductionSetBase( (void *) solver,
                                          new_base_index, new_base_stride ) );
}

