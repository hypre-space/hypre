/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_StructMatrix interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_StructMatrixCreate( MPI_Comm             comm,
                          HYPRE_StructGrid     grid,
                          HYPRE_StructStencil  stencil,
                          HYPRE_StructMatrix  *matrix )
{
   *matrix = hypre_StructMatrixCreate(comm, grid, stencil);

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructMatrixDestroy( HYPRE_StructMatrix matrix )
{
   return( hypre_StructMatrixDestroy(matrix) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixInitialize
 *--------------------------------------------------------------------------*/

int
HYPRE_StructMatrixInitialize( HYPRE_StructMatrix matrix )
{
   return ( hypre_StructMatrixInitialize(matrix) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructMatrixSetValues( HYPRE_StructMatrix  matrix,
                             int                *grid_index,
                             int                 num_stencil_indices,
                             int                *stencil_indices,
                             double             *values )
{
   hypre_Index  new_grid_index;
                
   int          d;
   int          ierr = 0;

   hypre_ClearIndex(new_grid_index);
   for (d = 0; d < hypre_StructGridDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_StructMatrixSetValues(matrix, new_grid_index,
                                      num_stencil_indices, stencil_indices,
                                      values, 0);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructMatrixSetBoxValues( HYPRE_StructMatrix  matrix,
                                int                *ilower,
                                int                *iupper,
                                int                 num_stencil_indices,
                                int                *stencil_indices,
                                double             *values )
{
   hypre_Index         new_ilower;
   hypre_Index         new_iupper;
   hypre_Box          *new_value_box;
                    
   int                 d;
   int                 ierr = 0;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_BoxCreate();
   hypre_BoxSetExtents(new_value_box, new_ilower, new_iupper);

   ierr = hypre_StructMatrixSetBoxValues(matrix, new_value_box,
                                         num_stencil_indices, stencil_indices,
                                         values, 0);

   hypre_BoxDestroy(new_value_box);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetBoxValues
 *--------------------------------------------------------------------------*/
                                                                                                         
int
HYPRE_StructMatrixGetBoxValues( HYPRE_StructMatrix  matrix,
                                int                *ilower,
                                int                *iupper,
                                int                 num_stencil_indices,
                                int                *stencil_indices,
                                double             *values )
{
   hypre_Index         new_ilower;
   hypre_Index         new_iupper;
   hypre_Box          *new_value_box;
                                                                                                         
   int                 d;
   int                 ierr = 0;
                                                                                                         
   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_BoxCreate();
   hypre_BoxSetExtents(new_value_box, new_ilower, new_iupper);
                                                                                                         
   ierr = hypre_StructMatrixSetBoxValues(matrix, new_value_box,
                                         num_stencil_indices, stencil_indices,
                                         values, -2);
                                                                                                         
   hypre_BoxDestroy(new_value_box);
                                                                                                         
   return (ierr);
}
                                                                                                         

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetConstantValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructMatrixSetConstantValues( HYPRE_StructMatrix matrix,
                                     int             num_stencil_indices,
                                     int            *stencil_indices,
                                     double         *values )
{
   return hypre_StructMatrixSetConstantValues( matrix,
                                               num_stencil_indices,
                                               stencil_indices,
                                               values,
                                               0 );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructMatrixAddToValues( HYPRE_StructMatrix  matrix,
                               int                *grid_index,
                               int                 num_stencil_indices,
                               int                *stencil_indices,
                               double             *values )
{
   hypre_Index         new_grid_index;

   int                 d;
   int                 ierr = 0;

   hypre_ClearIndex(new_grid_index);
   for (d = 0; d < hypre_StructGridDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_StructMatrixSetValues(matrix, new_grid_index,
                                      num_stencil_indices, stencil_indices,
                                      values, 1);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToBoxValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructMatrixAddToBoxValues( HYPRE_StructMatrix  matrix,
                                  int                *ilower,
                                  int                *iupper,
                                  int                 num_stencil_indices,
                                  int                *stencil_indices,
                                  double             *values )
{
   hypre_Index         new_ilower;
   hypre_Index         new_iupper;
   hypre_Box          *new_value_box;
                    
   int                 d;
   int                 ierr = 0;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim(hypre_StructMatrixGrid(matrix)); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_BoxCreate();
   hypre_BoxSetExtents(new_value_box, new_ilower, new_iupper);

   ierr = hypre_StructMatrixSetBoxValues(matrix, new_value_box,
                                         num_stencil_indices, stencil_indices,
                                         values, 1);

   hypre_BoxDestroy(new_value_box);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToConstantValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructMatrixAddToConstantValues( HYPRE_StructMatrix matrix,
                                       int             num_stencil_indices,
                                       int            *stencil_indices,
                                       double         *values )
{
   return hypre_StructMatrixSetConstantValues( matrix,
                                               num_stencil_indices,
                                               stencil_indices,
                                               values,
                                               1 );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructMatrixAssemble( HYPRE_StructMatrix matrix )
{
   return( hypre_StructMatrixAssemble(matrix) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetNumGhost
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructMatrixSetNumGhost( HYPRE_StructMatrix  matrix,
                               int                *num_ghost )
{
   return ( hypre_StructMatrixSetNumGhost(matrix, num_ghost) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetGrid
 *--------------------------------------------------------------------------*/

int
HYPRE_StructMatrixGetGrid( HYPRE_StructMatrix matrix, HYPRE_StructGrid *grid )
{
   int ierr = 0;

   *grid = hypre_StructMatrixGrid(matrix);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetSymmetric
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructMatrixSetSymmetric( HYPRE_StructMatrix  matrix,
                                int                 symmetric )
{
   int ierr  = 0;

   hypre_StructMatrixSymmetric(matrix) = symmetric;

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetConstantEntries
 * - nentries is the number of array entries
 * - Each int entries[i] is an index into the shape array of the stencil of the
 * matrix
 * In the present version, only three possibilites are recognized:
 * - no entries constant                 (constant_coefficient==0)
 * - all entries constant                (constant_coefficient==1)
 * - all but the diagonal entry constant (constant_coefficient==2)
 * If something else is attempted, this function will return a nonzero error.
 * In the present version, if this function is called more than once, only
 * the last call will take effect.
 *--------------------------------------------------------------------------*/

int  HYPRE_StructMatrixSetConstantEntries( HYPRE_StructMatrix  matrix,
                                           int                 nentries,
                                           int                *entries )
{
   return hypre_StructMatrixSetConstantEntries( matrix, nentries, entries );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixPrint
 *--------------------------------------------------------------------------*/

int
HYPRE_StructMatrixPrint( const char         *filename,
                         HYPRE_StructMatrix  matrix,
                         int                 all )
{
   return ( hypre_StructMatrixPrint(filename, matrix, all) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixMatvec
 *--------------------------------------------------------------------------*/

int
HYPRE_StructMatrixMatvec( double alpha,
                          HYPRE_StructMatrix A,
                          HYPRE_StructVector x,
                          double beta,
                          HYPRE_StructVector y     )
{
   return ( hypre_StructMatvec( alpha, (hypre_StructMatrix *) A,
                                (hypre_StructVector *) x, beta, (hypre_StructVector *) y) );
}

