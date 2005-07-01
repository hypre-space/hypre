/*
 * File:          bHYPRE_SStructGraph_Impl.c
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
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

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGraph_SetCommGrid(
  /* in */ bHYPRE_SStructGraph self,
  /* in */ void* mpi_comm,
  /* in */ bHYPRE_SStructGrid grid)
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
  /* in */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ int32_t to_part,
  /* in */ struct sidl_int__array* to_index,
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
  /* in */ void* mpi_comm)
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

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGraph_GetObject(
  /* in */ bHYPRE_SStructGraph self,
  /* out */ sidl_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.GetObject) */
  /* Insert the implementation of the GetObject method here... */
 
   bHYPRE_SStructGraph_addRef( self );
   *A = sidl_BaseInterface__cast( self );
   return( 0 );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.GetObject) */
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
