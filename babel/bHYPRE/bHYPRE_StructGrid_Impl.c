/*
 * File:          bHYPRE_StructGrid_Impl.c
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:39 PST
 * Generated:     20050317 11:17:43 PST
 * Description:   Server-side implementation for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 1106
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructGrid" (version 1.0.0)
 * 
 * Define a structured grid class.
 * 
 */

#include "bHYPRE_StructGrid_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "mpi.h"
#include "HYPRE_struct_mv.h"
#include "utilities.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid__ctor"

void
impl_bHYPRE_StructGrid__ctor(
  /*in*/ bHYPRE_StructGrid self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct bHYPRE_StructGrid__data * data;
   data = hypre_CTAlloc( struct bHYPRE_StructGrid__data, 1 );
   data -> grid = NULL;
   data -> comm = MPI_COMM_NULL;
   bHYPRE_StructGrid__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid__dtor"

void
impl_bHYPRE_StructGrid__dtor(
  /*in*/ bHYPRE_StructGrid self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_StructGrid__data * data;
   HYPRE_StructGrid Hgrid;
   data = bHYPRE_StructGrid__get_data( self );
   Hgrid = data -> grid;
   ierr = HYPRE_StructGridDestroy( Hgrid );
   assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetCommunicator"

int32_t
impl_bHYPRE_StructGrid_SetCommunicator(
  /*in*/ bHYPRE_StructGrid self, /*in*/ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   /* This should be called before SetDimension */
   int ierr = 0;
   struct bHYPRE_StructGrid__data * data;
   data = bHYPRE_StructGrid__get_data( self );
   data -> comm = (MPI_Comm) mpi_comm;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetCommunicator) */
}

/*
 * Method:  SetDimension[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetDimension"

int32_t
impl_bHYPRE_StructGrid_SetDimension(
  /*in*/ bHYPRE_StructGrid self, /*in*/ int32_t dim)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.SetDimension) */
  /* Insert the implementation of the SetDimension method here... */
   /* SetCommunicator should be called before this function.
      In Hypre, the dimension is permanently set at creation,
      so HYPRE_StructGridCreate is called here .*/

   int ierr = 0;
   struct bHYPRE_StructGrid__data * data;
   HYPRE_StructGrid * Hgrid;
   data = bHYPRE_StructGrid__get_data( self );
   Hgrid = &(data -> grid);
   assert( *Hgrid==NULL );  /* grid shouldn't have already been created */

   ierr += HYPRE_StructGridCreate( data->comm, dim, Hgrid );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetDimension) */
}

/*
 * Method:  SetExtents[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetExtents"

int32_t
impl_bHYPRE_StructGrid_SetExtents(
  /*in*/ bHYPRE_StructGrid self, /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.SetExtents) */
  /* Insert the implementation of the SetExtents method here... */

   /* SetCommunicator and SetDimension should have been called before
      this function, Assemble afterwards. */

   int ierr = 0;
   struct bHYPRE_StructGrid__data * data;
   HYPRE_StructGrid Hgrid;
   data = bHYPRE_StructGrid__get_data( self );
   Hgrid = data -> grid;

   ierr += HYPRE_StructGridSetExtents( Hgrid, sidlArrayAddr1( ilower, 0 ),
                                       sidlArrayAddr1( iupper, 0 ) );
   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetExtents) */
}

/*
 * Method:  SetPeriodic[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetPeriodic"

int32_t
impl_bHYPRE_StructGrid_SetPeriodic(
  /*in*/ bHYPRE_StructGrid self, /*in*/ struct sidl_int__array* periodic)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.SetPeriodic) */
  /* Insert the implementation of the SetPeriodic method here... */

   int ierr = 0;
   struct bHYPRE_StructGrid__data * data;
   HYPRE_StructGrid Hgrid;
   data = bHYPRE_StructGrid__get_data( self );
   Hgrid = data -> grid;

   ierr += HYPRE_StructGridSetPeriodic( Hgrid, sidlArrayAddr1( periodic, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetPeriodic) */
}

/*
 * Method:  SetNumGhost[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetNumGhost"

int32_t
impl_bHYPRE_StructGrid_SetNumGhost(
  /*in*/ bHYPRE_StructGrid self, /*in*/ struct sidl_int__array* num_ghost)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.SetNumGhost) */
  /* Insert the implementation of the SetNumGhost method here... */

   int ierr = 0;
   struct bHYPRE_StructGrid__data * data;
   HYPRE_StructGrid Hgrid;
   data = bHYPRE_StructGrid__get_data( self );
   Hgrid = data -> grid;

   ierr += HYPRE_StructGridSetNumGhost( Hgrid, sidlArrayAddr1( num_ghost, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetNumGhost) */
}

/*
 * Method:  Assemble[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_Assemble"

int32_t
impl_bHYPRE_StructGrid_Assemble(
  /*in*/ bHYPRE_StructGrid self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.Assemble) */
  /* Insert the implementation of the Assemble method here... */

   /* Call everything else before Assemble: constructor, SetCommunicator,
      SetDimension, SetExtents, SetPeriodic (optional) in that order (normally) */

   int ierr = 0;
   struct bHYPRE_StructGrid__data * data;
   HYPRE_StructGrid Hgrid;
   data = bHYPRE_StructGrid__get_data( self );
   Hgrid = data -> grid;

   ierr += HYPRE_StructGridAssemble( Hgrid );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.Assemble) */
}
