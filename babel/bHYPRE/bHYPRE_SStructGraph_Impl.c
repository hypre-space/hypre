/*
 * File:          bHYPRE_SStructGraph_Impl.c
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructGraph" (version 1.0.0)
 * 
 * The semi-structured grid graph class.
 * 
 */

#include "bHYPRE_SStructGraph_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "mpi.h"
#include "HYPRE_sstruct_mv.h"
#include "sstruct_mv.h"
#include "utilities.h"
#include "bHYPRE_SStructGrid_Impl.h"
#include "bHYPRE_SStructStencil_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph__ctor"

void
impl_bHYPRE_SStructGraph__ctor(
  /*in*/ bHYPRE_SStructGraph self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph._ctor) */
  /* Insert the implementation of the constructor method here... */

   /*
     To make a graph:  call
     bHYPRE_SStructGraph__create
     bHYPRE_SStructGraph_SetCommGrid
     bHYPRE_SStructGraph_SetObjectType
     bHYPRE_SStructGraph_SetStencil
     bHYPRE_SStructGraph_Assemble
    */

   struct bHYPRE_SStructGraph__data * data;
   data = hypre_CTAlloc( struct bHYPRE_SStructGraph__data, 1 );
   data -> graph = NULL;
   bHYPRE_SStructGraph__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph__dtor"

void
impl_bHYPRE_SStructGraph__dtor(
  /*in*/ bHYPRE_SStructGraph self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_SStructGraph__data * data;
   HYPRE_SStructGraph Hgraph;
   data = bHYPRE_SStructGraph__get_data( self );
   Hgraph = data -> graph;
   ierr = HYPRE_SStructGraphDestroy( Hgraph );
   assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph._dtor) */
}

/*
 * Set the grid and communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_SetCommGrid"

int32_t
impl_bHYPRE_SStructGraph_SetCommGrid(
  /*in*/ bHYPRE_SStructGraph self, /*in*/ void* mpi_comm,
    /*in*/ bHYPRE_SStructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.SetCommGrid) */
  /* Insert the implementation of the SetCommGrid method here... */

   int ierr = 0;
   struct bHYPRE_SStructGraph__data * data;
   HYPRE_SStructGraph * Hgraph;
   struct bHYPRE_SStructGrid__data * data_grid;
   HYPRE_SStructGrid Hgrid;

   data = bHYPRE_SStructGraph__get_data( self );
   Hgraph = &(data -> graph);
   assert( *Hgraph==NULL );  /* graph shouldn't have already been created */

   data_grid = bHYPRE_SStructGrid__get_data( grid );
   Hgrid = data_grid -> grid;

   ierr += HYPRE_SStructGraphCreate( (MPI_Comm) mpi_comm, Hgrid, Hgraph );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.SetCommGrid) */
}

/*
 * Set the stencil for a variable on a structured part of the
 * grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_SetStencil"

int32_t
impl_bHYPRE_SStructGraph_SetStencil(
  /*in*/ bHYPRE_SStructGraph self, /*in*/ int32_t part, /*in*/ int32_t var,
    /*in*/ bHYPRE_SStructStencil stencil)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.SetStencil) */
  /* Insert the implementation of the SetStencil method here... */

   int ierr = 0;
   struct bHYPRE_SStructGraph__data * data;
   HYPRE_SStructGraph Hgraph;
   struct bHYPRE_SStructStencil__data * data_stencil;
   HYPRE_SStructStencil Hstencil;
   data = bHYPRE_SStructGraph__get_data( self );
   Hgraph = data -> graph;
   data_stencil = bHYPRE_SStructStencil__get_data( stencil );
   Hstencil = data_stencil -> stencil;

   ierr += HYPRE_SStructGraphSetStencil( Hgraph, part, var, Hstencil );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.SetStencil) */
}

/*
 * Add a non-stencil graph entry at a particular index.  This
 * graph entry is appended to the existing graph entries, and is
 * referenced as such.
 * 
 * NOTE: Users are required to set graph entries on all
 * processes that own the associated variables.  This means that
 * some data will be multiply defined.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_AddEntries"

int32_t
impl_bHYPRE_SStructGraph_AddEntries(
  /*in*/ bHYPRE_SStructGraph self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*in*/ int32_t to_part, /*in*/ struct sidl_int__array* to_index,
    /*in*/ int32_t to_var)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.AddEntries) */
  /* Insert the implementation of the AddEntries method here... */

   int ierr = 0;
   struct bHYPRE_SStructGraph__data * data;
   HYPRE_SStructGraph Hgraph;
   data = bHYPRE_SStructGraph__get_data( self );
   Hgraph = data -> graph;

   ierr += HYPRE_SStructGraphAddEntries
      ( Hgraph, part, sidlArrayAddr1( index, 0 ), var, to_part,
        sidlArrayAddr1( to_index, 0 ), to_var );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.AddEntries) */
}

/*
 * Method:  SetObjectType[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_SetObjectType"

int32_t
impl_bHYPRE_SStructGraph_SetObjectType(
  /*in*/ bHYPRE_SStructGraph self, /*in*/ int32_t type)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.SetObjectType) */
  /* Insert the implementation of the SetObjectType method here... */

   int ierr = 0;
   struct bHYPRE_SStructGraph__data * data;
   HYPRE_SStructGraph Hgraph;
   data = bHYPRE_SStructGraph__get_data( self );
   Hgraph = data -> graph;

   ierr += HYPRE_SStructGraphSetObjectType( Hgraph, type );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.SetObjectType) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_SetCommunicator"

int32_t
impl_bHYPRE_SStructGraph_SetCommunicator(
  /*in*/ bHYPRE_SStructGraph self, /*in*/ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1; /* corresponding HYPRE function isn't implemented */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_Initialize"

int32_t
impl_bHYPRE_SStructGraph_Initialize(
  /*in*/ bHYPRE_SStructGraph self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.Initialize) */
  /* Insert the implementation of the Initialize method here... */
   /* this function is not necessary for SStructGraph */

   return 0;
   
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.Initialize) */
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_Assemble"

int32_t
impl_bHYPRE_SStructGraph_Assemble(
  /*in*/ bHYPRE_SStructGraph self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.Assemble) */
  /* Insert the implementation of the Assemble method here... */

   int ierr = 0;
   struct bHYPRE_SStructGraph__data * data;
   HYPRE_SStructGraph Hgraph;
   data = bHYPRE_SStructGraph__get_data( self );
   Hgraph = data -> graph;

   ierr += HYPRE_SStructGraphAssemble( Hgraph );

   return ierr;
   
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.Assemble) */
}

/*
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a sidl.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_GetObject"

int32_t
impl_bHYPRE_SStructGraph_GetObject(
  /*in*/ bHYPRE_SStructGraph self, /*out*/ sidl_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.GetObject) */
  /* Insert the implementation of the GetObject method here... */
 
   bHYPRE_SStructGraph_addRef( self );
   *A = sidl_BaseInterface__cast( self );
   return( 0 );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.GetObject) */
}
