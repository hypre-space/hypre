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
 * HYPRE_StructVector interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_StructVectorCreate( MPI_Comm             comm,
                          HYPRE_StructGrid     grid,
                          HYPRE_StructVector  *vector )
{
   int ierr = 0;

   *vector = hypre_StructVectorCreate(comm, grid);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructVectorDestroy( HYPRE_StructVector struct_vector )
{
   return( hypre_StructVectorDestroy(struct_vector) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorInitialize
 *--------------------------------------------------------------------------*/

int
HYPRE_StructVectorInitialize( HYPRE_StructVector vector )
{
   return ( hypre_StructVectorInitialize(vector) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructVectorSetValues( HYPRE_StructVector  vector,
                             int                *grid_index,
                             double              values )
{
   hypre_Index  new_grid_index;
                
   int          d;
   int          ierr = 0;

   hypre_ClearIndex(new_grid_index);
   for (d = 0; d < hypre_StructGridDim(hypre_StructVectorGrid(vector)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_StructVectorSetValues(vector, new_grid_index, values, 0);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructVectorSetBoxValues( HYPRE_StructVector  vector,
                                int                *ilower,
                                int                *iupper,
                                double             *values )
{
   hypre_Index   new_ilower;
   hypre_Index   new_iupper;
   hypre_Box    *new_value_box;
                 
   int           d;
   int           ierr = 0;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim(hypre_StructVectorGrid(vector)); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_BoxCreate();
   hypre_BoxSetExtents(new_value_box, new_ilower, new_iupper);

   ierr = hypre_StructVectorSetBoxValues(vector, new_value_box, values, 0 );

   hypre_BoxDestroy(new_value_box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAddToValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructVectorAddToValues( HYPRE_StructVector  vector,
                               int                *grid_index,
                               double              values )
{
   hypre_Index  new_grid_index;
                
   int          d;
   int          ierr = 0;

   hypre_ClearIndex(new_grid_index);
   for (d = 0; d < hypre_StructGridDim(hypre_StructVectorGrid(vector)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_StructVectorSetValues(vector, new_grid_index, values, 1);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructVectorAddToBoxValues( HYPRE_StructVector  vector,
                                  int                *ilower,
                                  int                *iupper,
                                  double             *values )
{
   hypre_Index   new_ilower;
   hypre_Index   new_iupper;
   hypre_Box    *new_value_box;
                 
   int           d;
   int           ierr = 0;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim(hypre_StructVectorGrid(vector)); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_BoxCreate();
   hypre_BoxSetExtents(new_value_box, new_ilower, new_iupper);

   ierr = hypre_StructVectorSetBoxValues(vector, new_value_box, values, 1);

   hypre_BoxDestroy(new_value_box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructVectorGetValues( HYPRE_StructVector  vector,
                             int                *grid_index,
                             double             *values_ptr )
{
   hypre_Index  new_grid_index;
                
   int          d;
   int          ierr = 0;

   hypre_ClearIndex(new_grid_index);
   for (d = 0; d < hypre_StructGridDim(hypre_StructVectorGrid(vector)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_StructVectorGetValues(vector, new_grid_index, values_ptr);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructVectorGetBoxValues( HYPRE_StructVector  vector,
                                int                *ilower,
                                int                *iupper,
                                double             *values )
{
   hypre_Index   new_ilower;
   hypre_Index   new_iupper;
   hypre_Box    *new_value_box;
                 
   int           d;
   int           ierr = 0;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim(hypre_StructVectorGrid(vector)); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_BoxCreate();
   hypre_BoxSetExtents(new_value_box, new_ilower, new_iupper);

   ierr = hypre_StructVectorGetBoxValues(vector, new_value_box, values);

   hypre_BoxDestroy(new_value_box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAssemble
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructVectorAssemble( HYPRE_StructVector vector )
{
   return( hypre_StructVectorAssemble(vector) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorPrint
 *--------------------------------------------------------------------------*/

int
HYPRE_StructVectorPrint( char               *filename,
                         HYPRE_StructVector  vector,
                         int                 all )
{
   return ( hypre_StructVectorPrint(filename, vector, all) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetNumGhost
 *--------------------------------------------------------------------------*/
 
int
HYPRE_StructVectorSetNumGhost( HYPRE_StructVector  vector,
                               int                *num_ghost )
{
   return ( hypre_StructVectorSetNumGhost(vector, num_ghost) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

int
HYPRE_StructVectorSetConstantValues( HYPRE_StructVector  vector,
                                     double              values )
{
   return( hypre_StructVectorSetConstantValues(vector, values) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetMigrateCommPkg
 *--------------------------------------------------------------------------*/

int
HYPRE_StructVectorGetMigrateCommPkg( HYPRE_StructVector  from_vector,
                                     HYPRE_StructVector  to_vector,
                                     HYPRE_CommPkg      *comm_pkg )
{
   int ierr = 0;

   *comm_pkg = hypre_StructVectorGetMigrateCommPkg(from_vector, to_vector);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorMigrate
 *--------------------------------------------------------------------------*/

int 
HYPRE_StructVectorMigrate( HYPRE_CommPkg      comm_pkg,
                           HYPRE_StructVector from_vector,
                           HYPRE_StructVector to_vector )
{
   return( hypre_StructVectorMigrate( comm_pkg, from_vector, to_vector) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CommPkgDestroy
 *--------------------------------------------------------------------------*/

int
HYPRE_CommPkgDestroy( HYPRE_CommPkg comm_pkg )
{
   return ( hypre_CommPkgDestroy(comm_pkg) );
}


