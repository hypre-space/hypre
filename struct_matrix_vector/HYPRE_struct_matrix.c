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
 * HYPRE_NewStructMatrix
 *--------------------------------------------------------------------------*/

int
HYPRE_NewStructMatrix( MPI_Comm             comm,
                       HYPRE_StructGrid     grid,
                       HYPRE_StructStencil  stencil,
                       HYPRE_StructMatrix  *matrix )
{
   *matrix = ( (HYPRE_StructMatrix)
              hypre_NewStructMatrix( comm,
                                    (hypre_StructGrid *) grid,
                                    (hypre_StructStencil *) stencil ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_FreeStructMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_FreeStructMatrix( HYPRE_StructMatrix matrix )
{
   return( hypre_FreeStructMatrix( (hypre_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_InitializeStructMatrix
 *--------------------------------------------------------------------------*/

int
HYPRE_InitializeStructMatrix( HYPRE_StructMatrix matrix )
{
   return ( hypre_InitializeStructMatrix( (hypre_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructMatrixValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetStructMatrixValues( HYPRE_StructMatrix  matrix,
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

   ierr = hypre_SetStructMatrixValues( new_matrix,
                                       new_grid_index,
                                       num_stencil_indices, stencil_indices,
                                       values );

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructMatrixBoxValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_SetStructMatrixBoxValues( HYPRE_StructMatrix  matrix,
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
   new_value_box = hypre_NewBox();
   hypre_SetBoxExtents(new_value_box, new_ilower, new_iupper);

   ierr = hypre_SetStructMatrixBoxValues( new_matrix,
                                          new_value_box,
                                          num_stencil_indices, stencil_indices,
                                          values );

   hypre_FreeBox(new_value_box);

   return (ierr);
}

/*--------------------------------------------------------------------------
 * HYPRE_AssembleStructMatrix
 *--------------------------------------------------------------------------*/

int 
HYPRE_AssembleStructMatrix( HYPRE_StructMatrix matrix )
{
   return( hypre_AssembleStructMatrix( (hypre_StructMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructMatrixNumGhost
 *--------------------------------------------------------------------------*/
 
int
HYPRE_SetStructMatrixNumGhost( HYPRE_StructMatrix  matrix,
                               int                *num_ghost )
{
   return ( hypre_SetStructMatrixNumGhost( (hypre_StructMatrix *) matrix,
                                           num_ghost) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGrid
 *--------------------------------------------------------------------------*/

int
HYPRE_StructMatrixGrid( HYPRE_StructMatrix matrix, HYPRE_StructGrid *grid )
{
   int ierr = 0;

   *grid = ( (HYPRE_StructGrid)
            (hypre_StructMatrixGrid( (hypre_StructMatrix *) matrix) ) );

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructMatrixSymmetric
 *--------------------------------------------------------------------------*/
 
int
HYPRE_SetStructMatrixSymmetric( HYPRE_StructMatrix  matrix,
                                int                 symmetric )
{
   int ierr  = 0;

   hypre_StructMatrixSymmetric( (hypre_StructMatrix *) matrix ) = symmetric;

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_PrintStructMatrix
 *--------------------------------------------------------------------------*/

int
HYPRE_PrintStructMatrix( char               *filename,
                         HYPRE_StructMatrix  matrix,
                         int                 all )
{
   return ( hypre_PrintStructMatrix( filename,
                                     (hypre_StructMatrix *) matrix, all ) );
}

