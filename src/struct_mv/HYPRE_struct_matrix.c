/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_StructMatrix interface
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
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
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixDestroy( HYPRE_StructMatrix matrix )
{
   return ( hypre_StructMatrixDestroy(matrix) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* RDF: Need a good user interface for setting range/domain grids. Maybe a
 * GridSetExtents approach would be the best approach. */

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixSetRangeStride(HYPRE_StructMatrix matrix,
                                 HYPRE_Int         *range_stride)
{
   return ( hypre_StructMatrixSetRangeStride(matrix, range_stride) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixSetDomainStride(HYPRE_StructMatrix matrix,
                                  HYPRE_Int         *domain_stride)
{
   return ( hypre_StructMatrixSetDomainStride(matrix, domain_stride) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixInitialize( HYPRE_StructMatrix matrix )
{
   return ( hypre_StructMatrixInitialize(matrix) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixSetValues( HYPRE_StructMatrix  matrix,
                             HYPRE_Int          *grid_index,
                             HYPRE_Int           num_stencil_indices,
                             HYPRE_Int          *stencil_indices,
                             HYPRE_Complex      *values )
{
   hypre_Index  new_grid_index;
   HYPRE_Int    d;

   hypre_SetIndex(new_grid_index, 0);
   for (d = 0; d < hypre_StructGridNDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructMatrixSetValues(matrix, new_grid_index,
                               num_stencil_indices, stencil_indices,
                               values, 0, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixGetValues( HYPRE_StructMatrix  matrix,
                             HYPRE_Int          *grid_index,
                             HYPRE_Int           num_stencil_indices,
                             HYPRE_Int          *stencil_indices,
                             HYPRE_Complex      *values )
{
   hypre_Index  new_grid_index;
   HYPRE_Int    d;

   hypre_SetIndex(new_grid_index, 0);
   for (d = 0; d < hypre_StructGridNDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructMatrixSetValues(matrix, new_grid_index,
                               num_stencil_indices, stencil_indices,
                               values, -1, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixSetBoxValues( HYPRE_StructMatrix  matrix,
                                HYPRE_Int          *ilower,
                                HYPRE_Int          *iupper,
                                HYPRE_Int           num_stencil_indices,
                                HYPRE_Int          *stencil_indices,
                                HYPRE_Complex      *values )
{
   HYPRE_StructMatrixSetBoxValues2(matrix, ilower, iupper, num_stencil_indices,
                                   stencil_indices, ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixGetBoxValues( HYPRE_StructMatrix  matrix,
                                HYPRE_Int          *ilower,
                                HYPRE_Int          *iupper,
                                HYPRE_Int           num_stencil_indices,
                                HYPRE_Int          *stencil_indices,
                                HYPRE_Complex      *values )
{
   HYPRE_StructMatrixGetBoxValues2(matrix, ilower, iupper, num_stencil_indices,
                                   stencil_indices, ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixSetBoxValues2( HYPRE_StructMatrix  matrix,
                                 HYPRE_Int          *ilower,
                                 HYPRE_Int          *iupper,
                                 HYPRE_Int           num_stencil_indices,
                                 HYPRE_Int          *stencil_indices,
                                 HYPRE_Int          *vilower,
                                 HYPRE_Int          *viupper,
                                 HYPRE_Complex      *values )
{
   hypre_Box  *set_box, *value_box;
   HYPRE_Int   d;

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));
   value_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));

   for (d = 0; d < hypre_StructMatrixNDim(matrix); d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_StructMatrixSetBoxValues(matrix, set_box, value_box,
                                  num_stencil_indices, stencil_indices,
                                  values, 0, -1, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixGetBoxValues2( HYPRE_StructMatrix  matrix,
                                 HYPRE_Int          *ilower,
                                 HYPRE_Int          *iupper,
                                 HYPRE_Int           num_stencil_indices,
                                 HYPRE_Int          *stencil_indices,
                                 HYPRE_Int          *vilower,
                                 HYPRE_Int          *viupper,
                                 HYPRE_Complex      *values )
{
   hypre_Box  *set_box, *value_box;
   HYPRE_Int   d;

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));
   value_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));

   for (d = 0; d < hypre_StructMatrixNDim(matrix); d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_StructMatrixSetBoxValues(matrix, set_box, value_box,
                                  num_stencil_indices, stencil_indices,
                                  values, -1, -1, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixSetConstantValues( HYPRE_StructMatrix matrix,
                                     HYPRE_Int          num_stencil_indices,
                                     HYPRE_Int         *stencil_indices,
                                     HYPRE_Complex     *values )
{
   hypre_StructMatrixSetConstantValues(matrix, num_stencil_indices,
                                       stencil_indices, values, 0 );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixAddToValues( HYPRE_StructMatrix  matrix,
                               HYPRE_Int          *grid_index,
                               HYPRE_Int           num_stencil_indices,
                               HYPRE_Int          *stencil_indices,
                               HYPRE_Complex      *values )
{
   hypre_Index         new_grid_index;
   HYPRE_Int           d;

   hypre_SetIndex(new_grid_index, 0);
   for (d = 0; d < hypre_StructGridNDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructMatrixSetValues(matrix, new_grid_index,
                               num_stencil_indices, stencil_indices,
                               values, 1, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixAddToBoxValues( HYPRE_StructMatrix  matrix,
                                  HYPRE_Int          *ilower,
                                  HYPRE_Int          *iupper,
                                  HYPRE_Int           num_stencil_indices,
                                  HYPRE_Int          *stencil_indices,
                                  HYPRE_Complex      *values )
{
   HYPRE_StructMatrixAddToBoxValues2(matrix, ilower, iupper, num_stencil_indices,
                                     stencil_indices, ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixAddToBoxValues2( HYPRE_StructMatrix  matrix,
                                   HYPRE_Int          *ilower,
                                   HYPRE_Int          *iupper,
                                   HYPRE_Int           num_stencil_indices,
                                   HYPRE_Int          *stencil_indices,
                                   HYPRE_Int          *vilower,
                                   HYPRE_Int          *viupper,
                                   HYPRE_Complex      *values )
{
   hypre_Box  *set_box, *value_box;
   HYPRE_Int   d;

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));
   value_box = hypre_BoxCreate(hypre_StructMatrixNDim(matrix));

   for (d = 0; d < hypre_StructMatrixNDim(matrix); d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_StructMatrixSetBoxValues(matrix, set_box, value_box,
                                  num_stencil_indices, stencil_indices,
                                  values, 1, -1, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixAddToConstantValues( HYPRE_StructMatrix matrix,
                                       HYPRE_Int          num_stencil_indices,
                                       HYPRE_Int         *stencil_indices,
                                       HYPRE_Complex     *values )
{
   hypre_StructMatrixSetConstantValues(matrix, num_stencil_indices,
                                       stencil_indices, values, 1 );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixAssemble( HYPRE_StructMatrix matrix )
{
   return ( hypre_StructMatrixAssemble(matrix) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixSetSymmetric( HYPRE_StructMatrix  matrix,
                                HYPRE_Int           symmetric )
{
   hypre_StructMatrixSymmetric(matrix) = symmetric;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixSetConstantEntries( HYPRE_StructMatrix  matrix,
                                      HYPRE_Int           nentries,
                                      HYPRE_Int          *entries )
{
   hypre_StructMatrixSetConstantEntries(matrix, nentries, entries);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixSetTranspose( HYPRE_StructMatrix  matrix,
                                HYPRE_Int           transpose )
{
   HYPRE_Int resize;

   hypre_StructMatrixSetTranspose(matrix, transpose, &resize);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixSetNumGhost( HYPRE_StructMatrix  matrix,
                               HYPRE_Int          *num_ghost )
{
   HYPRE_Int resize;

   hypre_StructMatrixSetNumGhost(matrix, num_ghost, &resize);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixPrint( const char         *filename,
                         HYPRE_StructMatrix  matrix,
                         HYPRE_Int           all )
{
   return ( hypre_StructMatrixPrint(filename, matrix, all) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixRead( MPI_Comm             comm,
                        const char          *filename,
                        HYPRE_Int           *num_ghost,
                        HYPRE_StructMatrix  *matrix )
{
   if (!matrix)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   *matrix = (HYPRE_StructMatrix) hypre_StructMatrixRead(comm, filename, num_ghost);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixMatvec( HYPRE_Complex      alpha,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector x,
                          HYPRE_Complex      beta,
                          HYPRE_StructVector y     )
{
   return ( hypre_StructMatvec( alpha, (hypre_StructMatrix *) A,
                                (hypre_StructVector *) x, beta,
                                (hypre_StructVector *) y) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixMatvecT( HYPRE_Complex      alpha,
                           HYPRE_StructMatrix A,
                           HYPRE_StructVector x,
                           HYPRE_Complex      beta,
                           HYPRE_StructVector y     )
{
   return ( hypre_StructMatvecT( alpha, (hypre_StructMatrix *) A,
                                 (hypre_StructVector *) x, beta,
                                 (hypre_StructVector *) y) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixMatmat( HYPRE_StructMatrix  A,
                          HYPRE_Int           Atranspose,
                          HYPRE_StructMatrix  B,
                          HYPRE_Int           Btranspose,
                          HYPRE_StructMatrix *C )
{
   HYPRE_Int           nmatrices     = 2;
   hypre_StructMatrix *matrices[2]   = {A, B};
   HYPRE_Int           nterms        = 2;
   HYPRE_Int           terms[2]      = {0, 1};
   HYPRE_Int           transposes[2] = {Atranspose, Btranspose};
#if defined (HYPRE_USING_GPU)
   HYPRE_Int           kernel_type   = 1;
#else
   HYPRE_Int           kernel_type   = 0;
#endif

   if (A == B)
   {
      nmatrices = 1;
      terms[1] = 0;
   }
   hypre_StructMatmult(kernel_type, nmatrices, matrices, nterms, terms, transposes, C);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixGetGrid( HYPRE_StructMatrix matrix, HYPRE_StructGrid *grid )
{
   *grid = hypre_StructMatrixGrid(matrix);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructMatrixClearBoundary( HYPRE_StructMatrix matrix )
{
   return ( hypre_StructMatrixClearBoundary(matrix) );
}
