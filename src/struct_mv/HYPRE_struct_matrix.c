/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.17 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * HYPRE_StructMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixCreate( MPI_Comm             comm,
                          HYPRE_StructGrid     grid,
                          HYPRE_StructStencil  stencil,
                          HYPRE_StructMatrix  *matrix )
{
   *matrix = hypre_StructMatrixCreate(comm, grid, stencil);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructMatrixDestroy( HYPRE_StructMatrix matrix )
{
   return( hypre_StructMatrixDestroy(matrix) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixInitialize( HYPRE_StructMatrix matrix )
{
   return ( hypre_StructMatrixInitialize(matrix) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructMatrixSetValues( HYPRE_StructMatrix  matrix,
                             HYPRE_Int          *grid_index,
                             HYPRE_Int           num_stencil_indices,
                             HYPRE_Int          *stencil_indices,
                             double             *values )
{
   hypre_Index  new_grid_index;
   HYPRE_Int    d;

   hypre_ClearIndex(new_grid_index);
   for (d = 0; d < hypre_StructGridDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructMatrixSetValues(matrix, new_grid_index,
                               num_stencil_indices, stencil_indices,
                               values, 0, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructMatrixGetValues( HYPRE_StructMatrix  matrix,
                             HYPRE_Int          *grid_index,
                             HYPRE_Int           num_stencil_indices,
                             HYPRE_Int          *stencil_indices,
                             double             *values )
{
   hypre_Index  new_grid_index;
   HYPRE_Int    d;

   hypre_ClearIndex(new_grid_index);
   for (d = 0; d < hypre_StructGridDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructMatrixSetValues(matrix, new_grid_index,
                               num_stencil_indices, stencil_indices,
                               values, -1, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructMatrixSetBoxValues( HYPRE_StructMatrix  matrix,
                                HYPRE_Int          *ilower,
                                HYPRE_Int          *iupper,
                                HYPRE_Int           num_stencil_indices,
                                HYPRE_Int          *stencil_indices,
                                double             *values )
{
   hypre_Index         new_ilower;
   hypre_Index         new_iupper;
   hypre_Box          *new_value_box;
   HYPRE_Int           d;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_BoxCreate();
   hypre_BoxSetExtents(new_value_box, new_ilower, new_iupper);

   hypre_StructMatrixSetBoxValues(matrix, new_value_box, new_value_box,
                                  num_stencil_indices, stencil_indices,
                                  values, 0, -1, 0);

   hypre_BoxDestroy(new_value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetBoxValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixGetBoxValues( HYPRE_StructMatrix  matrix,
                                HYPRE_Int          *ilower,
                                HYPRE_Int          *iupper,
                                HYPRE_Int           num_stencil_indices,
                                HYPRE_Int          *stencil_indices,
                                double             *values )
{
   hypre_Index         new_ilower;
   hypre_Index         new_iupper;
   hypre_Box          *new_value_box;
   HYPRE_Int           d;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_BoxCreate();
   hypre_BoxSetExtents(new_value_box, new_ilower, new_iupper);

   hypre_StructMatrixSetBoxValues(matrix, new_value_box, new_value_box,
                                  num_stencil_indices, stencil_indices,
                                  values, -1, -1, 0);

   hypre_BoxDestroy(new_value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructMatrixSetConstantValues( HYPRE_StructMatrix matrix,
                                     HYPRE_Int       num_stencil_indices,
                                     HYPRE_Int      *stencil_indices,
                                     double         *values )
{
   return hypre_StructMatrixSetConstantValues( matrix,
                                               num_stencil_indices,
                                               stencil_indices,
                                               values, 0 );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructMatrixAddToValues( HYPRE_StructMatrix  matrix,
                               HYPRE_Int          *grid_index,
                               HYPRE_Int           num_stencil_indices,
                               HYPRE_Int          *stencil_indices,
                               double             *values )
{
   hypre_Index         new_grid_index;
   HYPRE_Int           d;

   hypre_ClearIndex(new_grid_index);
   for (d = 0; d < hypre_StructGridDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructMatrixSetValues(matrix, new_grid_index,
                               num_stencil_indices, stencil_indices,
                               values, 1, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToBoxValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructMatrixAddToBoxValues( HYPRE_StructMatrix  matrix,
                                  HYPRE_Int          *ilower,
                                  HYPRE_Int          *iupper,
                                  HYPRE_Int           num_stencil_indices,
                                  HYPRE_Int          *stencil_indices,
                                  double             *values )
{
   hypre_Index         new_ilower;
   hypre_Index         new_iupper;
   hypre_Box          *new_value_box;
   HYPRE_Int           d;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_BoxCreate();
   hypre_BoxSetExtents(new_value_box, new_ilower, new_iupper);

   hypre_StructMatrixSetBoxValues(matrix, new_value_box, new_value_box,
                                  num_stencil_indices, stencil_indices,
                                  values, 1, -1, 0);

   hypre_BoxDestroy(new_value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructMatrixAddToConstantValues( HYPRE_StructMatrix matrix,
                                       HYPRE_Int       num_stencil_indices,
                                       HYPRE_Int      *stencil_indices,
                                       double         *values )
{
   return hypre_StructMatrixSetConstantValues( matrix,
                                               num_stencil_indices,
                                               stencil_indices,
                                               values, 1 );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAssemble
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructMatrixAssemble( HYPRE_StructMatrix matrix )
{
   return( hypre_StructMatrixAssemble(matrix) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetNumGhost
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_StructMatrixSetNumGhost( HYPRE_StructMatrix  matrix,
                               HYPRE_Int          *num_ghost )
{
   return ( hypre_StructMatrixSetNumGhost(matrix, num_ghost) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetGrid
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixGetGrid( HYPRE_StructMatrix matrix, HYPRE_StructGrid *grid )
{
   *grid = hypre_StructMatrixGrid(matrix);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetSymmetric
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_StructMatrixSetSymmetric( HYPRE_StructMatrix  matrix,
                                HYPRE_Int           symmetric )
{
   hypre_StructMatrixSymmetric(matrix) = symmetric;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetConstantEntries
 * Call this function to declare that certain stencil points are constant
 * throughout the mesh.
 * - nentries is the number of array entries
 * - Each HYPRE_Int entries[i] is an index into the shape array of the stencil of the
 * matrix.
 * In the present version, only three possibilites are recognized:
 * - no entries constant                 (constant_coefficient==0)
 * - all entries constant                (constant_coefficient==1)
 * - all but the diagonal entry constant (constant_coefficient==2)
 * If something else is attempted, this function will return a nonzero error.
 * In the present version, if this function is called more than once, only
 * the last call will take effect.
 *--------------------------------------------------------------------------*/

HYPRE_Int  HYPRE_StructMatrixSetConstantEntries( HYPRE_StructMatrix  matrix,
                                           HYPRE_Int           nentries,
                                           HYPRE_Int          *entries )
{
   return hypre_StructMatrixSetConstantEntries( matrix, nentries, entries );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixPrint( const char         *filename,
                         HYPRE_StructMatrix  matrix,
                         HYPRE_Int           all )
{
   return ( hypre_StructMatrixPrint(filename, matrix, all) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixMatvec( double alpha,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector x,
                          double beta,
                          HYPRE_StructVector y     )
{
   return ( hypre_StructMatvec( alpha, (hypre_StructMatrix *) A,
                                (hypre_StructVector *) x, beta,
                                (hypre_StructVector *) y) );
}
