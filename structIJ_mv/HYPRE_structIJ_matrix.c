/******************************************************************************
 *
 * HYPRE_StructIJMatrix interface
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * HYPRE_StructIJMatrixCreate
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructIJMatrixCreate( MPI_Comm              comm,
		            HYPRE_StructGrid      grid,
		            HYPRE_StructStencil   stencil,
                            HYPRE_StructIJMatrix *matrix)
{
   *matrix = ( (HYPRE_StructIJMatrix)
               hypre_StructIJMatrixCreate( comm,
                                           (hypre_StructGrid *) grid,
                                           (hypre_StructStencil *) stencil ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructIJMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructIJMatrixDestroy( HYPRE_StructIJMatrix matrix )
{
   return( hypre_StructIJMatrixDestroy( (hypre_StructIJMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructIJMatrixInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructIJMatrixInitialize( HYPRE_StructIJMatrix matrix )
{
   return ( hypre_StructIJMatrixInitialize( (hypre_StructIJMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructIJMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructIJMatrixAssemble( HYPRE_StructIJMatrix matrix )
{
   return( hypre_StructIJMatrixAssemble( (hypre_StructIJMatrix *) matrix ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructIJMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructIJMatrixSetBoxValues( HYPRE_StructIJMatrix  matrix,
			          int                  *lower_grid_index,
			          int                  *upper_grid_index,
                                  int                   num_stencil_indices,
                                  int                  *stencil_indices,
			          double               *coeffs )

{
   hypre_Index           new_lower_grid_index;
   hypre_Index           new_upper_grid_index;
   int                   d, dim;
   int                   ierr;

   dim = hypre_StructGridDim( hypre_StructIJMatrixGrid( 
                               (hypre_StructIJMatrix *) matrix) );

   for (d = 0; d < dim; d++)
   {
      hypre_IndexD(new_lower_grid_index, d) = lower_grid_index[d];
      hypre_IndexD(new_upper_grid_index, d) = upper_grid_index[d];
   }

   ierr = hypre_StructIJMatrixSetBoxValues( (hypre_StructIJMatrix *) matrix ,
                                            new_lower_grid_index,
                                            new_upper_grid_index,
                                            num_stencil_indices,
                                            stencil_indices,
                                            coeffs );

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructIJMatrixSetSymmetric
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructIJMatrixSetSymmetric( HYPRE_StructIJMatrix matrix, int symmetric )
{
   hypre_StructIJMatrixSymmetric( (hypre_StructIJMatrix *) matrix ) = symmetric;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructIJMatrixGetLocalStorage
 *--------------------------------------------------------------------------*/

void *
HYPRE_StructIJMatrixGetLocalStorage( HYPRE_StructIJMatrix matrix )
{
   return( HYPRE_IJMatrixGetLocalStorage( 
              hypre_StructIJMatrixIJMatrix( 
                 (hypre_StructIJMatrix *) matrix) ) );
}
