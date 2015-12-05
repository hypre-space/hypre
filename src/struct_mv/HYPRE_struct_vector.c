/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * HYPRE_StructVector interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorCreate( MPI_Comm             comm,
                          HYPRE_StructGrid     grid,
                          HYPRE_StructVector  *vector )
{
   HYPRE_Int ierr = 0;

   *vector = hypre_StructVectorCreate(comm, grid);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructVectorDestroy( HYPRE_StructVector struct_vector )
{
   return( hypre_StructVectorDestroy(struct_vector) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorInitialize
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorInitialize( HYPRE_StructVector vector )
{
   return ( hypre_StructVectorInitialize(vector) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructVectorSetValues( HYPRE_StructVector  vector,
                             HYPRE_Int          *grid_index,
                             double              values )
{
   hypre_Index  new_grid_index;
                
   HYPRE_Int    d;
   HYPRE_Int    ierr = 0;

   hypre_ClearIndex(new_grid_index);
   for (d = 0; d < hypre_StructGridDim(hypre_StructVectorGrid(vector)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_StructVectorSetValues(vector, new_grid_index, &values, 0, -1, 0);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructVectorSetBoxValues( HYPRE_StructVector  vector,
                                HYPRE_Int          *ilower,
                                HYPRE_Int          *iupper,
                                double             *values )
{
   hypre_Index   new_ilower;
   hypre_Index   new_iupper;
   hypre_Box    *new_value_box;
                 
   HYPRE_Int     d;
   HYPRE_Int     ierr = 0;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim(hypre_StructVectorGrid(vector)); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_BoxCreate();
   hypre_BoxSetExtents(new_value_box, new_ilower, new_iupper);

   ierr = hypre_StructVectorSetBoxValues(vector, new_value_box, new_value_box,
                                         values, 0, -1, 0);

   hypre_BoxDestroy(new_value_box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAddToValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructVectorAddToValues( HYPRE_StructVector  vector,
                               HYPRE_Int          *grid_index,
                               double              values )
{
   hypre_Index  new_grid_index;
                
   HYPRE_Int    d;
   HYPRE_Int    ierr = 0;

   hypre_ClearIndex(new_grid_index);
   for (d = 0; d < hypre_StructGridDim(hypre_StructVectorGrid(vector)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_StructVectorSetValues(vector, new_grid_index, &values, 1, -1, 0);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructVectorAddToBoxValues( HYPRE_StructVector  vector,
                                  HYPRE_Int          *ilower,
                                  HYPRE_Int          *iupper,
                                  double             *values )
{
   hypre_Index   new_ilower;
   hypre_Index   new_iupper;
   hypre_Box    *new_value_box;
                 
   HYPRE_Int     d;
   HYPRE_Int     ierr = 0;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim(hypre_StructVectorGrid(vector)); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_BoxCreate();
   hypre_BoxSetExtents(new_value_box, new_ilower, new_iupper);

   ierr = hypre_StructVectorSetBoxValues(vector, new_value_box, new_value_box,
                                         values, 1, -1, 0);

   hypre_BoxDestroy(new_value_box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorScaleValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructVectorScaleValues( HYPRE_StructVector  vector,
                               double              factor )
{
   return hypre_StructVectorScaleValues( vector, factor );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructVectorGetValues( HYPRE_StructVector  vector,
                             HYPRE_Int          *grid_index,
                             double             *values )
{
   hypre_Index  new_grid_index;
                
   HYPRE_Int    d;
   HYPRE_Int    ierr = 0;

   hypre_ClearIndex(new_grid_index);
   for (d = 0; d < hypre_StructGridDim(hypre_StructVectorGrid(vector)); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   ierr = hypre_StructVectorSetValues(vector, new_grid_index, values, -1, -1, 0);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructVectorGetBoxValues( HYPRE_StructVector  vector,
                                HYPRE_Int          *ilower,
                                HYPRE_Int          *iupper,
                                double             *values )
{
   hypre_Index   new_ilower;
   hypre_Index   new_iupper;
   hypre_Box    *new_value_box;
                 
   HYPRE_Int     d;
   HYPRE_Int     ierr = 0;

   hypre_ClearIndex(new_ilower);
   hypre_ClearIndex(new_iupper);
   for (d = 0; d < hypre_StructGridDim(hypre_StructVectorGrid(vector)); d++)
   {
      hypre_IndexD(new_ilower, d) = ilower[d];
      hypre_IndexD(new_iupper, d) = iupper[d];
   }
   new_value_box = hypre_BoxCreate();
   hypre_BoxSetExtents(new_value_box, new_ilower, new_iupper);

   ierr = hypre_StructVectorSetBoxValues(vector, new_value_box, new_value_box,
                                         values, -1, -1, 0);

   hypre_BoxDestroy(new_value_box);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAssemble
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructVectorAssemble( HYPRE_StructVector vector )
{
   return( hypre_StructVectorAssemble(vector) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorPrint
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorPrint( const char         *filename,
                         HYPRE_StructVector  vector,
                         HYPRE_Int           all )
{
   return ( hypre_StructVectorPrint(filename, vector, all) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetNumGhost
 *--------------------------------------------------------------------------*/
 
HYPRE_Int
HYPRE_StructVectorSetNumGhost( HYPRE_StructVector  vector,
                               HYPRE_Int          *num_ghost )
{
   return ( hypre_StructVectorSetNumGhost(vector, num_ghost) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorCopy
 * copies data from x to y
 * y has its own data array, so this is a deep copy in that sense.
 * The grid and other size information are not copied - they are
 * assumed to be consistent already.
 *--------------------------------------------------------------------------*/
HYPRE_Int
HYPRE_StructVectorCopy( HYPRE_StructVector x, HYPRE_StructVector y )
{
   return( hypre_StructVectorCopy( x, y ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorSetConstantValues( HYPRE_StructVector  vector,
                                     double              values )
{
   return( hypre_StructVectorSetConstantValues(vector, values) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetMigrateCommPkg
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorGetMigrateCommPkg( HYPRE_StructVector  from_vector,
                                     HYPRE_StructVector  to_vector,
                                     HYPRE_CommPkg      *comm_pkg )
{
   HYPRE_Int ierr = 0;

   *comm_pkg = hypre_StructVectorGetMigrateCommPkg(from_vector, to_vector);

   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorMigrate
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_StructVectorMigrate( HYPRE_CommPkg      comm_pkg,
                           HYPRE_StructVector from_vector,
                           HYPRE_StructVector to_vector )
{
   return( hypre_StructVectorMigrate( comm_pkg, from_vector, to_vector) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CommPkgDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CommPkgDestroy( HYPRE_CommPkg comm_pkg )
{
   return ( hypre_CommPkgDestroy(comm_pkg) );
}


