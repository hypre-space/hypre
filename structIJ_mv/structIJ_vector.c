/******************************************************************************
 *
 * Member functions for hypre_StructIJVector class.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_StructIJVectorCreate
 *
 * Note that we assume the global dimension of the vector = total number of
 * grid points.  
 *
 *--------------------------------------------------------------------------*/

hypre_StructIJVector *
hypre_StructIJVectorCreate( MPI_Comm             comm,
                            hypre_StructGrid    *grid,
                            hypre_StructStencil *stencil )
{
   int                  ierr;
   HYPRE_IJVector       ij_vector;
   hypre_StructIJVector *vector;

   vector = hypre_CTAlloc(hypre_StructIJVector, 1);  
   
   hypre_StructIJVectorComm(vector)      = comm;
   hypre_StructGridRef(grid, &hypre_StructIJVectorGrid(vector));
   hypre_StructIJVectorStencil(vector)   = hypre_StructStencilRef(stencil);
   hypre_StructIJVectorRefCount(vector)  = 1;

   ierr = HYPRE_IJVectorCreate (comm, &ij_vector,
                                hypre_StructGridGlobalSize(grid));

   hypre_StructIJVectorIJVector(vector) = ij_vector;

   return vector;
}

/*--------------------------------------------------------------------------
 * hypre_StructIJVectorDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_StructIJVectorDestroy( hypre_StructIJVector *vector )
{
   int  ierr;

   if (vector)
   {
      hypre_StructIJVectorRefCount(vector) --;
      if (hypre_StructIJVectorRefCount(vector) == 0)
      {
         ierr  =hypre_StructGridDestroy(hypre_StructIJVectorGrid(vector));
         ierr +=hypre_StructStencilDestroy(hypre_StructIJVectorStencil(vector));
         ierr +=HYPRE_IJVectorDestroy(hypre_StructIJVectorIJVector(vector));
         
         hypre_FreeStructGridToCoordTable( (hypre_StructGridToCoordTable *)
			     hypre_StructIJVectorTranslator(vector) );

         hypre_TFree(vector);
      }
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructIJVectorInitialize
 *--------------------------------------------------------------------------*/

int 
hypre_StructIJVectorInitialize( hypre_StructIJVector *vector )
{
   int           ierr;

   hypre_StructGrid    *grid     = hypre_StructIJVectorGrid(vector);
   hypre_StructStencil *stencil  = hypre_StructIJVectorStencil(vector);
   HYPRE_IJVector       ijvector = hypre_StructIJVectorIJVector(vector);

   hypre_StructGridToCoordTable       *grid_to_coord_table;

   ierr  = HYPRE_IJVectorSetLocalStorageType( ijvector, HYPRE_PARCSR );

   ierr += HYPRE_IJVectorInitialize( ijvector );

   grid_to_coord_table = hypre_NewStructGridToCoordTable(grid, stencil);
   hypre_StructIJVectorTranslator(vector) = (void *) grid_to_coord_table;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructIJVectorSetBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructIJVectorSetBoxValues( hypre_StructIJVector *vector,
                                  hypre_Index           lower_grid_index,
                                  hypre_Index           upper_grid_index,
                                  double               *coeffs )
{
   hypre_Index grid_index;
   int         ierr=0;
   int         i, j, k, coeffs_index;

   /* Insert coefficients one grid point at a time */

   for (k = hypre_IndexZ(lower_grid_index), coeffs_index = 0; 
        k <= hypre_IndexZ(upper_grid_index); 
        k++)

      for (j = hypre_IndexY(lower_grid_index); 
           j <= hypre_IndexY(upper_grid_index); 
           j++)

         for (i = hypre_IndexX(lower_grid_index); 
              i <= hypre_IndexX(upper_grid_index); 
              i++, coeffs_index++ )
         
         {
            hypre_SetIndex(grid_index, i, j, k);

            ierr += hypre_StructIJVectorSetValue( vector,
                                                  grid_index,
                                                  coeffs[coeffs_index] );

         }

   return ierr ;
}

/*--------------------------------------------------------------------------
 * hypre_StructIJVectorSetValue
 *--------------------------------------------------------------------------*/

/*
  Sets a value into a StructIJVector corresponding to a particular grid point.
*/

int 
hypre_StructIJVectorSetValue( hypre_StructIJVector *vector, 
                               hypre_Index          index,
                               double               value )
{
   int  ierr=0; 
   int  i;
   int  vec_index;
   int  val_index;

   hypre_StructGridToCoordTable       *grid_to_coord_table;
   hypre_StructGridToCoordTableEntry  *grid_to_coord_table_entry;

   grid_to_coord_table =
      (hypre_StructGridToCoordTable *) hypre_StructIJVectorTranslator(vector);

   grid_to_coord_table_entry =
      hypre_FindStructGridToCoordTableEntry( index, grid_to_coord_table );

   if (grid_to_coord_table_entry==NULL)
   {
      printf("Warning: Attempt to set value for point not in grid\n");
      printf("hypre_StructIJVectorSetValue call aborted for grid point\n");
      printf("  %d, %d, %d\n", hypre_IndexD(index,0), hypre_IndexD(index,1), 
                               hypre_IndexD(index,2) );
      return(0);
   }

   vec_index = hypre_MapStructGridToCoord( index, grid_to_coord_table_entry );
   val_index = 0; /* always want the 0th entry in our length-1 list of values */

   ierr =  HYPRE_IJVectorSetLocalComponents( 
                    hypre_StructIJVectorIJVector(vector),
                    1, &vec_index, &val_index, &value );

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructIJVectorAssemble
 *--------------------------------------------------------------------------*/

int 
hypre_StructIJVectorAssemble( hypre_StructIJVector *vector )

{
   return ( HYPRE_IJVectorAssemble( hypre_StructIJVectorIJVector(vector) ) );
}
