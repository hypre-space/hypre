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
   *matrix = ( (HYPRE_StructMatrix)
               hypre_StructMatrixCreate( comm,
                                         (hypre_StructGrid *) grid,
                                         (hypre_StructStencil *) stencil ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructMatrixDestroy( HYPRE_StructMatrix matrix )
{
   return( hypre_StructMatrixDestroy( (hypre_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixInitialize
 *--------------------------------------------------------------------------*/

int
HYPRE_StructMatrixInitialize( HYPRE_StructMatrix matrix )
{
   return ( hypre_StructMatrixInitialize( (hypre_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructMatrixSetValues( HYPRE_StructMatrix  matrix,
                             int                *grid_index,
                             int                 num_stencil_indices,
                             int                *stencil_indices,
                             double             *values              )
{
   hypre_StructMatrix *new_matrix = (hypre_StructMatrix *) matrix;
   hypre_Index         new_grid_index;

   int                 d;
   int                 ierr = 0;

   hypre_ClearIndex(new_grid_index);
   for (d = 0;
        d < hypre_StructGridDim(hypre_StructMatrixGrid(new_matrix));
        d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_StructMatrixSetValues( new_matrix,
                                       new_grid_index,
                                       num_stencil_indices, stencil_indices,
                                       values );

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
                                double             *values              )
{
   hypre_StructMatrix *new_matrix = (hypre_StructMatrix *) matrix;
   hypre_Index         new_ilower;
   hypre_Index         new_iupper;
   hypre_Box          *new_value_box;
                    
   int                 d;
   int                 ierr = 0;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0;
        d < hypre_StructGridDim(hypre_StructMatrixGrid(new_matrix));
        d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_BoxCreate();
   hypre_BoxSetExtents(new_value_box, new_ilower, new_iupper);

   ierr = hypre_StructMatrixSetBoxValues( new_matrix,
                                          new_value_box,
                                          num_stencil_indices, stencil_indices,
                                          values );

   hypre_BoxDestroy(new_value_box);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructMatrixAssemble( HYPRE_StructMatrix matrix )
{
   return( hypre_StructMatrixAssemble( (hypre_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetNumGhost
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructMatrixSetNumGhost( HYPRE_StructMatrix  matrix,
                               int                *num_ghost )
{
   return ( hypre_StructMatrixSetNumGhost( (hypre_StructMatrix *) matrix,
                                           num_ghost) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetGrid
 *--------------------------------------------------------------------------*/

int
HYPRE_StructMatrixGetGrid( HYPRE_StructMatrix matrix, HYPRE_StructGrid *grid )
{
   int ierr = 0;

   *grid = ( (HYPRE_StructGrid)
             (hypre_StructMatrixGrid( (hypre_StructMatrix *) matrix) ) );

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

   hypre_StructMatrixSymmetric( (hypre_StructMatrix *) matrix ) = symmetric;

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixPrint
 *--------------------------------------------------------------------------*/

int
HYPRE_StructMatrixPrint( char               *filename,
                         HYPRE_StructMatrix  matrix,
                         int                 all )
{
   return ( hypre_StructMatrixPrint( filename,
                                     (hypre_StructMatrix *) matrix, all ) );
}

