/******************************************************************************
 *
 * Member functions for hypre_StructIJMatrix class.
 *
 *****************************************************************************/

#include "headers.h"

#define hypre_AddStructStencilEltToStructGridpoint( stencil_shape, index, new_index ) \
hypre_IndexD(new_index, 0) = hypre_IndexD(index, 0) + stencil_shape[0];\
hypre_IndexD(new_index, 1) = hypre_IndexD(index, 1) + stencil_shape[1];\
hypre_IndexD(new_index, 2) = hypre_IndexD(index, 2) + stencil_shape[2];

/*--------------------------------------------------------------------------
 * hypre_StructIJMatrixCreate
 *
 * Note that we assume total number of eqns/unks = total number of
 * grid points.  This is not necessarily true, depending on how
 * boundary conditions are handled, or if we have a system of equations, 
 * etc.
 *
 *--------------------------------------------------------------------------*/

hypre_StructIJMatrix *
hypre_StructIJMatrixCreate( MPI_Comm             comm,
                            hypre_StructGrid    *grid,
                            hypre_StructStencil *stencil )
{
   int                  ierr;
   HYPRE_IJMatrix       ij_matrix;
   hypre_StructIJMatrix *matrix;

   matrix = hypre_CTAlloc(hypre_StructIJMatrix, 1);  
   
   hypre_StructIJMatrixComm(matrix)      = comm;
   hypre_StructGridRef(grid, &hypre_StructIJMatrixGrid(matrix));
   hypre_StructIJMatrixStencil(matrix)   = hypre_StructStencilRef(stencil);
   hypre_StructIJMatrixSymmetric(matrix) = 0;
   hypre_StructIJMatrixRefCount(matrix)  = 1;

   ierr = HYPRE_IJMatrixCreate (comm, &ij_matrix,
                                hypre_StructGridGlobalSize(grid),
                                hypre_StructGridGlobalSize(grid));

   hypre_StructIJMatrixIJMatrix(matrix) = ij_matrix;

   return matrix;
}

/*--------------------------------------------------------------------------
 * hypre_StructIJMatrixDestroy
 *--------------------------------------------------------------------------*/

int 
hypre_StructIJMatrixDestroy( hypre_StructIJMatrix *matrix )
{
   int ierr;

   if (matrix)
   {
      hypre_StructIJMatrixRefCount(matrix) --;
      if (hypre_StructIJMatrixRefCount(matrix) == 0)
      {
         ierr = hypre_StructStencilDestroy(hypre_StructIJMatrixStencil(matrix));
         ierr += hypre_StructGridDestroy(hypre_StructIJMatrixGrid(matrix));
         ierr += HYPRE_IJMatrixDestroy(hypre_StructIJMatrixIJMatrix(matrix));
         
         hypre_FreeStructGridToCoordTable( (hypre_StructGridToCoordTable *)
			     hypre_StructIJMatrixTranslator(matrix) );

         hypre_TFree(matrix);
      }
   }

   return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_StructIJMatrixInitialize
 *
 * Notes:
 *   - Assumes the local storage type is PARCSR.
 *   - Assumes the number of local rows is equal to the number of local
 *     grid points.
 *   - Assumes the number of nonzeros in every row (row_sizes[i]) is
 *     equal to the "size" of the stencil, if the matrix is nonsymmetric.
 *     If the matrix is symmetric we assume the number of nonzeros is
 *     2*size-1.
 *
 *--------------------------------------------------------------------------*/

int 
hypre_StructIJMatrixInitialize( hypre_StructIJMatrix *matrix )
{
   int  ierr, i, nrows, nnzs;
   int *row_sizes;

   hypre_StructGrid    *grid     = hypre_StructIJMatrixGrid(matrix);
   hypre_StructStencil *stencil  = hypre_StructIJMatrixStencil(matrix);
   HYPRE_IJMatrix       ijmatrix = hypre_StructIJMatrixIJMatrix(matrix);

   hypre_StructGridToCoordTable *grid_to_coord_table;

   ierr  = HYPRE_IJMatrixSetLocalStorageType( ijmatrix, HYPRE_PARCSR );

   nrows = hypre_StructGridLocalSize( grid ); 

   ierr += HYPRE_IJMatrixSetLocalSize( ijmatrix, nrows, nrows );

   nnzs = hypre_StructStencilSize( hypre_StructIJMatrixStencil( matrix ) );
   if (hypre_StructIJMatrixSymmetric( matrix ) )
      nnzs = 2*nnzs - 1;

   row_sizes = hypre_CTAlloc(int, nrows);
   for (i=0; i < nrows; i++)
      row_sizes[i] = nnzs;

   ierr += HYPRE_IJMatrixSetRowSizes ( ijmatrix, (const int *) row_sizes );

   ierr += HYPRE_IJMatrixInitialize( ijmatrix );

   grid_to_coord_table = hypre_NewStructGridToCoordTable(grid, stencil);
   hypre_StructIJMatrixTranslator(matrix) = (void *) grid_to_coord_table;

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_StructIJMatrixAssemble
 *--------------------------------------------------------------------------*/

int 
hypre_StructIJMatrixAssemble( hypre_StructIJMatrix *matrix )

{
   return ( HYPRE_IJMatrixAssemble( hypre_StructIJMatrixIJMatrix(matrix) ) );
}

/*--------------------------------------------------------------------------
 * hypre_StructIJMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

int 
hypre_StructIJMatrixSetBoxValues( hypre_StructIJMatrix *matrix,
                                  hypre_Index           lower_grid_index,
                                  hypre_Index           upper_grid_index,
                                  int                   num_stencil_indices,
                                  int                  *stencil_indices,
                                  double               *coeffs )
{
   int  ierr=0;
   int  i, j, k, coeffs_index;

   hypre_Index          grid_index;
   hypre_StructStencil *stencil;

   /* Insert coefficients one grid point at a time */

   for (k = hypre_IndexZ(lower_grid_index), coeffs_index = 0; 
        k <= hypre_IndexZ(upper_grid_index); 
        k++)

      for (j = hypre_IndexY(lower_grid_index); 
           j <= hypre_IndexY(upper_grid_index); 
           j++)

         for (i = hypre_IndexX(lower_grid_index); 
              i <= hypre_IndexX(upper_grid_index); 
              i++, coeffs_index += num_stencil_indices)
         
         {
            hypre_SetIndex(grid_index, i, j, k);

            ierr += hypre_StructIJMatrixSetValues( matrix,
                                                   grid_index,
                                                   num_stencil_indices,
                                                   stencil_indices,
                                                   &(coeffs[coeffs_index]));

         }

   return ierr ;
}


/*--------------------------------------------------------------------------
 * hypre_StructIJMatrixSetValues
 *--------------------------------------------------------------------------*/

/*
  Sets values into a StructIJMatrix corresponding to a particular grid point.

  The array stencil_indices[num_stencil_indices] tells which points in the
  stencil correspond to the input values, i.e., if j = stencil_indices[i],
  then coeffs[i] corresponds to the j'th point in the stencil, whose position
  relative to the center point is defined by shape[j].
*/

int 
hypre_StructIJMatrixSetValues( hypre_StructIJMatrix *matrix, 
                               hypre_Index           index,
                               int                   num_stencil_indices,
                               int                  *stencil_indices,
                               double               *coeffs )
{
   int      ierr=0; 
   int      i, row_coord, col_coord, num_coefs;
   int     *cols;
   double  *values;

   hypre_StructStencil *stencil;
   hypre_Index          new_index;


   hypre_StructGridToCoordTable      *grid_to_coord_table;
   hypre_StructGridToCoordTableEntry *grid_to_coord_table_entry;

   stencil = hypre_StructIJMatrixStencil(matrix);

   grid_to_coord_table =
      (hypre_StructGridToCoordTable *) hypre_StructIJMatrixTranslator(matrix);

   grid_to_coord_table_entry =
      hypre_FindStructGridToCoordTableEntry( index, grid_to_coord_table );

   if (grid_to_coord_table_entry==NULL)
   {
      printf("Warning: Attempt to set coeffs for point not in grid\n");
      printf("hypre_StructIJMatrixSetValues call aborted for grid point\n");
      printf("  %d, %d, %d\n", hypre_IndexD(index,0), hypre_IndexD(index,1), 
                               hypre_IndexD(index,2) );
      return(0);
   }

   row_coord = hypre_MapStructGridToCoord( index, grid_to_coord_table_entry );

   cols   = hypre_CTAlloc(int,    num_stencil_indices);
   values = hypre_CTAlloc(double, num_stencil_indices);
   num_coefs = 0;
   for (i = 0; i < num_stencil_indices; i++)
   {
      hypre_AddStructStencilEltToStructGridpoint( 
                     hypre_StructStencilShape(stencil)[stencil_indices[i]],
                     index, new_index );

      grid_to_coord_table_entry =
         hypre_FindStructGridToCoordTableEntry( new_index, grid_to_coord_table);

      if ( grid_to_coord_table_entry != NULL )
      {
         col_coord = hypre_MapStructGridToCoord( new_index,
                                         grid_to_coord_table_entry );
         cols[num_coefs] = col_coord;
         values[num_coefs] = coeffs[i];
         num_coefs++;
      }
   }

   ierr =  HYPRE_IJMatrixInsertRow( hypre_StructIJMatrixIJMatrix(matrix),
                                    num_coefs,
                                    row_coord,
                                    (const int *) cols,
                                    (const double *) values);
                      

   hypre_TFree(cols);
   hypre_TFree(values);

   return ierr;
}
