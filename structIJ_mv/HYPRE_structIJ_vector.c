/******************************************************************************
 *
 * HYPRE_StructIJVector interface
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * HYPRE_StructIJVectorCreate
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructIJVectorCreate( MPI_Comm              comm,
		            HYPRE_StructGrid      grid,
		            HYPRE_StructStencil   stencil,
                            HYPRE_StructIJVector *vector)
{
   *vector = (HYPRE_StructIJVector)
               hypre_StructIJVectorCreate( comm, 
                                           (hypre_StructGrid *) grid,
                                           (hypre_StructStencil *) stencil );
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructIJVectorDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructIJVectorDestroy( HYPRE_StructIJVector vector )
{
   return( hypre_StructIJVectorDestroy( (hypre_StructIJVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructIJVectorInitialize
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructIJVectorInitialize( HYPRE_StructIJVector vector )
{
   return ( hypre_StructIJVectorInitialize( (hypre_StructIJVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructIJVectorAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructIJVectorAssemble( HYPRE_StructIJVector vector )
{
   return( hypre_StructIJVectorAssemble( (hypre_StructIJVector *) vector ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructIJVectorSetBoxValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructIJVectorSetBoxValues( HYPRE_StructIJVector  vector,
			          int                  *lower_grid_index,
			          int                  *upper_grid_index,
			          double               *coeffs )

{
   hypre_Index           new_lower_grid_index;
   hypre_Index           new_upper_grid_index;
   int                   d, dim;
   int                   ierr;

   dim = hypre_StructGridDim( hypre_StructIJVectorGrid( 
                               (hypre_StructIJVector *) vector) );

   for (d = 0; d < dim; d++)
   {
      hypre_IndexD(new_lower_grid_index, d) = lower_grid_index[d];
      hypre_IndexD(new_upper_grid_index, d) = upper_grid_index[d];
   }

   ierr = hypre_StructIJVectorSetBoxValues( (hypre_StructIJVector *) vector ,
                                            new_lower_grid_index,
                                            new_upper_grid_index,
                                            coeffs );

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructIJVectorGetLocalStorage
 *--------------------------------------------------------------------------*/

void *
HYPRE_StructIJVectorGetLocalStorage( HYPRE_StructIJVector in_vector )
{
   hypre_StructIJVector *vector = (hypre_StructIJVector *) in_vector;

   return( HYPRE_IJVectorGetLocalStorage( 
                  hypre_StructIJVectorIJVector( vector ) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructIJVectorSetPartitioning
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructIJVectorSetPartitioning( HYPRE_StructIJVector vector,
                                     const int           *partitioning)
{
   return HYPRE_IJVectorSetPartitioning(
             hypre_StructIJVectorIJVector( (hypre_StructIJVector *) vector),
             partitioning );
}
