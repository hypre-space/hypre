/*
 * File:          bHYPRE_StructGrid_Impl.c
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for bHYPRE.StructGrid
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
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructGrid__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._load) */
  /* Insert-Code-Here {bHYPRE.StructGrid._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructGrid__ctor(
  /* in */ bHYPRE_StructGrid self)
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

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructGrid__dtor(
  /* in */ bHYPRE_StructGrid self)
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

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructGrid_SetCommunicator(
  /* in */ bHYPRE_StructGrid self,
  /* in */ void* mpi_comm)
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

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructGrid_SetDimension(
  /* in */ bHYPRE_StructGrid self,
  /* in */ int32_t dim)
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

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructGrid_SetExtents(
  /* in */ bHYPRE_StructGrid self,
  /* in */ int32_t* ilower,
  /* in */ int32_t* iupper,
  /* in */ int32_t dim)
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

   /* for sidl arrays:
      ierr += HYPRE_StructGridSetExtents( Hgrid, ilower,
      iupper );
   */
   ierr += HYPRE_StructGridSetExtents( Hgrid, ilower, iupper );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetExtents) */
}

/*
 * Method:  SetPeriodic[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetPeriodic"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructGrid_SetPeriodic(
  /* in */ bHYPRE_StructGrid self,
  /* in */ int32_t* periodic,
  /* in */ int32_t dim)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.SetPeriodic) */
  /* Insert the implementation of the SetPeriodic method here... */

   int ierr = 0;
   struct bHYPRE_StructGrid__data * data;
   HYPRE_StructGrid Hgrid;
   data = bHYPRE_StructGrid__get_data( self );
   Hgrid = data -> grid;

   ierr += HYPRE_StructGridSetPeriodic( Hgrid, periodic );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetPeriodic) */
}

/*
 * Method:  SetNumGhost[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetNumGhost"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructGrid_SetNumGhost(
  /* in */ bHYPRE_StructGrid self,
  /* in */ int32_t* num_ghost,
  /* in */ int32_t dim2)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.SetNumGhost) */
  /* Insert the implementation of the SetNumGhost method here... */

   int ierr = 0;
   struct bHYPRE_StructGrid__data * data;
   HYPRE_StructGrid Hgrid;
   data = bHYPRE_StructGrid__get_data( self );
   Hgrid = data -> grid;

   /* for sidl arrays: ierr += HYPRE_StructGridSetNumGhost( Hgrid, num_ghost ); */
   ierr += HYPRE_StructGridSetNumGhost( Hgrid, num_ghost );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetNumGhost) */
}

/*
 * Method:  Assemble[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_Assemble"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructGrid_Assemble(
  /* in */ bHYPRE_StructGrid self)
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
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_StructGrid__connect(url, _ex);
}
char * impl_bHYPRE_StructGrid_fgetURL_bHYPRE_StructGrid(struct 
  bHYPRE_StructGrid__object* obj) {
  return bHYPRE_StructGrid__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_StructGrid_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_StructGrid_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_StructGrid_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
