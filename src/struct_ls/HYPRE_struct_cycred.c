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
   return( hypre_CyclicReductionDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructCycRedSetup( HYPRE_StructSolver solver,
                         HYPRE_StructMatrix A,
                         HYPRE_StructVector b,
                         HYPRE_StructVector x      )
{
   return( hypre_CyclicReductionSetup( (void *) solver,
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
   return( hypre_CyclicReduction( (void *) solver,
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
   return( hypre_CyclicReductionSetCDir( (void *) solver, tdim ) );
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

   return( hypre_CyclicReductionSetBase( (void *) solver,
                                         new_base_index, new_base_stride ) );
}

