/*
 * File:          bHYPRE_SStructGraph_Impl.c
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side implementation for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.10
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
/*#include "mpi.h"*/
#include "HYPRE_sstruct_mv.h"
#include "sstruct_mv.h"
#include "utilities.h"
#include "bHYPRE_SStructGrid_Impl.h"
#include "bHYPRE_SStructStencil_Impl.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructGraph__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph._load) */
  /* Insert-Code-Here {bHYPRE.SStructGraph._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructGraph__ctor(
  /* in */ bHYPRE_SStructGraph self)
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

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructGraph__dtor(
  /* in */ bHYPRE_SStructGraph self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_SStructGraph__data * data;
   HYPRE_SStructGraph Hgraph;
   data = bHYPRE_SStructGraph__get_data( self );
   Hgraph = data -> graph;
   ierr = HYPRE_SStructGraphDestroy( Hgraph );
   hypre_assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph._dtor) */
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_SStructGraph
impl_bHYPRE_SStructGraph_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.Create) */
  /* Insert-Code-Here {bHYPRE.SStructGraph.Create} (Create method) */

   int ierr = 0;
   bHYPRE_SStructGraph graph;
   struct bHYPRE_SStructGraph__data * data;
   HYPRE_SStructGraph Hgraph;
   struct bHYPRE_SStructGrid__data * data_grid;
   HYPRE_SStructGrid Hgrid;
   MPI_Comm comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   graph = bHYPRE_SStructGraph__create();
   data = bHYPRE_SStructGraph__get_data( graph );

   data_grid = bHYPRE_SStructGrid__get_data( grid );
   Hgrid = data_grid -> grid;

   ierr += HYPRE_SStructGraphCreate( comm, Hgrid, &Hgraph );
   data->graph = Hgraph;

   return graph;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.Create) */
}

/*
 * Set the grid and communicator.
 * DEPRECATED, use Create:
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_SetCommGrid"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGraph_SetCommGrid(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.SetCommGrid) */
  /* Insert the implementation of the SetCommGrid method here... */

   /* DEPRECATED    use Create */

   int ierr = 0;
   struct bHYPRE_SStructGraph__data * data;
   HYPRE_SStructGraph * Hgraph;
   struct bHYPRE_SStructGrid__data * data_grid;
   HYPRE_SStructGrid Hgrid;
   MPI_Comm comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   data = bHYPRE_SStructGraph__get_data( self );
   Hgraph = &(data -> graph);
   hypre_assert( *Hgraph==NULL );  /* graph shouldn't have already been created */

   data_grid = bHYPRE_SStructGrid__get_data( grid );
   Hgrid = data_grid -> grid;

   ierr += HYPRE_SStructGraphCreate( comm, Hgrid, Hgraph );

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

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGraph_SetStencil(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ bHYPRE_SStructStencil stencil)
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

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGraph_AddEntries(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t to_part,
  /* in rarray[dim] */ int32_t* to_index,
  /* in */ int32_t to_var)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.AddEntries) */
  /* Insert the implementation of the AddEntries method here... */

   int ierr = 0;
   struct bHYPRE_SStructGraph__data * data;
   HYPRE_SStructGraph Hgraph;
   data = bHYPRE_SStructGraph__get_data( self );
   Hgraph = data -> graph;

   ierr += HYPRE_SStructGraphAddEntries
      ( Hgraph, part, index, var, to_part,
        to_index, to_var );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.AddEntries) */
}

/*
 * Method:  SetObjectType[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_SetObjectType"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGraph_SetObjectType(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ int32_t type)
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
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGraph_SetCommunicator(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ bHYPRE_MPICommunicator mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1; /* corresponding HYPRE function isn't implemented, and shouldn't be */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_Initialize"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGraph_Initialize(
  /* in */ bHYPRE_SStructGraph self)
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

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGraph_Assemble(
  /* in */ bHYPRE_SStructGraph self)
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
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGrid(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_SStructGrid__connect(url, _ex);
}
char * impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGrid(struct 
  bHYPRE_SStructGrid__object* obj) {
  return bHYPRE_SStructGrid__getURL(obj);
}
struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructStencil(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_SStructStencil__connect(url, _ex);
}
char * impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj) {
  return bHYPRE_SStructStencil__getURL(obj);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connect(url, _ex);
}
char * impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) {
  return bHYPRE_MPICommunicator__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_SStructGraph_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_ProblemDefinition__connect(url, _ex);
}
char * impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj) {
  return bHYPRE_ProblemDefinition__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_SStructGraph_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructGraph_fconnect_bHYPRE_SStructGraph(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_SStructGraph__connect(url, _ex);
}
char * impl_bHYPRE_SStructGraph_fgetURL_bHYPRE_SStructGraph(struct 
  bHYPRE_SStructGraph__object* obj) {
  return bHYPRE_SStructGraph__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructGraph_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_SStructGraph_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
