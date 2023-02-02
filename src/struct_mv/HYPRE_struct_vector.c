/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_StructVector interface
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorCreate( MPI_Comm             comm,
                          HYPRE_StructGrid     grid,
                          HYPRE_StructVector  *vector )
{
   *vector = hypre_StructVectorCreate(comm, grid);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorDestroy( HYPRE_StructVector struct_vector )
{
   return ( hypre_StructVectorDestroy(struct_vector) );
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
                             HYPRE_Complex       values )
{
   hypre_Index  new_grid_index;

   HYPRE_Int    d;

   hypre_SetIndex(new_grid_index, 0);
   for (d = 0; d < hypre_StructVectorNDim(vector); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructVectorSetValues(vector, new_grid_index, &values, 0, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorSetBoxValues( HYPRE_StructVector  vector,
                                HYPRE_Int          *ilower,
                                HYPRE_Int          *iupper,
                                HYPRE_Complex      *values )
{
   HYPRE_StructVectorSetBoxValues2(vector, ilower, iupper, ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorSetBoxValues2( HYPRE_StructVector  vector,
                                 HYPRE_Int          *ilower,
                                 HYPRE_Int          *iupper,
                                 HYPRE_Int          *vilower,
                                 HYPRE_Int          *viupper,
                                 HYPRE_Complex      *values )
{
   hypre_Box  *set_box, *value_box;
   HYPRE_Int   d;

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));
   value_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));

   for (d = 0; d < hypre_StructVectorNDim(vector); d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_StructVectorSetBoxValues(vector, set_box, value_box, values, 0, -1, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAddToValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorAddToValues( HYPRE_StructVector  vector,
                               HYPRE_Int          *grid_index,
                               HYPRE_Complex       values )
{
   hypre_Index  new_grid_index;

   HYPRE_Int    d;

   hypre_SetIndex(new_grid_index, 0);
   for (d = 0; d < hypre_StructVectorNDim(vector); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructVectorSetValues(vector, new_grid_index, &values, 1, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorAddToBoxValues( HYPRE_StructVector  vector,
                                  HYPRE_Int          *ilower,
                                  HYPRE_Int          *iupper,
                                  HYPRE_Complex      *values )
{
   HYPRE_StructVectorAddToBoxValues2(vector, ilower, iupper, ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorAddToBoxValues2( HYPRE_StructVector  vector,
                                   HYPRE_Int          *ilower,
                                   HYPRE_Int          *iupper,
                                   HYPRE_Int          *vilower,
                                   HYPRE_Int          *viupper,
                                   HYPRE_Complex      *values )
{
   hypre_Box  *set_box, *value_box;
   HYPRE_Int   d;

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));
   value_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));

   for (d = 0; d < hypre_StructVectorNDim(vector); d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_StructVectorSetBoxValues(vector, set_box, value_box, values, 1, -1, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorScaleValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorScaleValues( HYPRE_StructVector  vector,
                               HYPRE_Complex       factor )
{
   return hypre_StructVectorScaleValues( vector, factor );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorGetValues( HYPRE_StructVector  vector,
                             HYPRE_Int          *grid_index,
                             HYPRE_Complex      *values )
{
   hypre_Index  new_grid_index;

   HYPRE_Int    d;

   hypre_SetIndex(new_grid_index, 0);
   for (d = 0; d < hypre_StructVectorNDim(vector); d++)
   {
      hypre_IndexD(new_grid_index, d) = grid_index[d];
   }

   hypre_StructVectorSetValues(vector, new_grid_index, values, -1, -1, 0);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorGetBoxValues( HYPRE_StructVector  vector,
                                HYPRE_Int          *ilower,
                                HYPRE_Int          *iupper,
                                HYPRE_Complex      *values )
{
   HYPRE_StructVectorGetBoxValues2(vector, ilower, iupper, ilower, iupper, values);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorGetBoxValues2( HYPRE_StructVector  vector,
                                 HYPRE_Int          *ilower,
                                 HYPRE_Int          *iupper,
                                 HYPRE_Int          *vilower,
                                 HYPRE_Int          *viupper,
                                 HYPRE_Complex      *values )
{
   hypre_Box          *set_box, *value_box;
   HYPRE_Int           d;

   /* This creates boxes with zeroed-out extents */
   set_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));
   value_box = hypre_BoxCreate(hypre_StructVectorNDim(vector));

   for (d = 0; d < hypre_StructVectorNDim(vector); d++)
   {
      hypre_BoxIMinD(set_box, d) = ilower[d];
      hypre_BoxIMaxD(set_box, d) = iupper[d];
      hypre_BoxIMinD(value_box, d) = vilower[d];
      hypre_BoxIMaxD(value_box, d) = viupper[d];
   }

   hypre_StructVectorSetBoxValues(vector, set_box, value_box, values, -1, -1, 0);

   hypre_BoxDestroy(set_box);
   hypre_BoxDestroy(value_box);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAssemble
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorAssemble( HYPRE_StructVector vector )
{
   return ( hypre_StructVectorAssemble(vector) );
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
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorRead( MPI_Comm             comm,
                        const char          *filename,
                        HYPRE_Int           *num_ghost,
                        HYPRE_StructVector  *vector )
{
   if (!vector)
   {
      hypre_error_in_arg(4);
      return hypre_error_flag;
   }

   *vector = (HYPRE_StructVector) hypre_StructVectorRead(comm, filename, num_ghost);

   return hypre_error_flag;
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
   return ( hypre_StructVectorCopy( x, y ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorSetConstantValues( HYPRE_StructVector  vector,
                                     HYPRE_Complex       values )
{
   return ( hypre_StructVectorSetConstantValues(vector, values) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetMigrateCommPkg
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorGetMigrateCommPkg( HYPRE_StructVector  from_vector,
                                     HYPRE_StructVector  to_vector,
                                     HYPRE_CommPkg      *comm_pkg )
{
   *comm_pkg = hypre_StructVectorGetMigrateCommPkg(from_vector, to_vector);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorMigrate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorMigrate( HYPRE_CommPkg      comm_pkg,
                           HYPRE_StructVector from_vector,
                           HYPRE_StructVector to_vector )
{
   return ( hypre_StructVectorMigrate( comm_pkg, from_vector, to_vector) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CommPkgDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CommPkgDestroy( HYPRE_CommPkg comm_pkg )
{
   return ( hypre_CommPkgDestroy(comm_pkg) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorClone
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_StructVectorClone( HYPRE_StructVector x,
                         HYPRE_StructVector *y_ptr )
{
   *y_ptr = hypre_StructVectorClone(x);

   return hypre_error_flag;
}
