/*
 * File:          bHYPRE_StructGrid_Impl.c
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructGrid" (version 1.0.0)
 * 
 * Define a structured grid class.
 */

#include "bHYPRE_StructGrid_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._includes) */
/* Put additional includes or other arbitrary code here... */



#include "hypre_babel_exception_handler.h"
/*#include "mpi.h"*/
#include "HYPRE_struct_mv.h"
#include "_hypre_utilities.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
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
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._load) */
  /* Insert-Code-Here {bHYPRE.StructGrid._load} (static class initializer method) */
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._load) */
  }
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
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._ctor) */
  /* Insert the implementation of the constructor method here... */

   /* User calls of __create are DEPRECATED.  Instead, call _Create, which
      also calls this function. */

   struct bHYPRE_StructGrid__data * data;
   data = hypre_CTAlloc( struct bHYPRE_StructGrid__data, 1 );
   data -> grid = NULL;
   data -> comm = MPI_COMM_NULL;
   bHYPRE_StructGrid__set_data( self, data );

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructGrid__ctor2(
  /* in */ bHYPRE_StructGrid self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._ctor2) */
    /* Insert-Code-Here {bHYPRE.StructGrid._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._ctor2) */
  }
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
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_StructGrid__data * data;
   HYPRE_StructGrid Hgrid;
   data = bHYPRE_StructGrid__get_data( self );
   Hgrid = data -> grid;
   ierr = HYPRE_StructGridDestroy( Hgrid );
   hypre_assert( ierr==0 );
   hypre_TFree( data );

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._dtor) */
  }
}

/*
 *  This function is the preferred way to create a Struct Grid. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_StructGrid
impl_bHYPRE_StructGrid_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.Create) */
  /* Insert-Code-Here {bHYPRE.StructGrid.Create} (Create method) */

   int ierr = 0;
   bHYPRE_StructGrid grid;
   struct bHYPRE_StructGrid__data * data;
   HYPRE_StructGrid Hgrid;

   grid = bHYPRE_StructGrid__create(_ex); SIDL_CHECK(*_ex);
   data = bHYPRE_StructGrid__get_data( grid );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   ierr += HYPRE_StructGridCreate( data->comm, dim, &Hgrid );
   hypre_assert( ierr==0 );
   data->grid = Hgrid;

   return grid;

   hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.Create) */
  }
}

/*
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructGrid_SetCommunicator(
  /* in */ bHYPRE_StructGrid self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   /* DEPRECATED   use _Create */

   /* This should be called before SetDimension */
   int ierr = 0;
   struct bHYPRE_StructGrid__data * data;
   data = bHYPRE_StructGrid__get_data( self );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructGrid_Destroy(
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.Destroy) */
    /* Insert-Code-Here {bHYPRE.StructGrid.Destroy} (Destroy method) */
     bHYPRE_StructGrid_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.Destroy) */
  }
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
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.SetDimension) */
  /* Insert the implementation of the SetDimension method here... */
   /* SetCommunicator should be called before this function.
      In Hypre, the dimension is permanently set at creation,
      so HYPRE_StructGridCreate is called here .*/

   /* DEPRECATED   use _Create */

   int ierr = 0;
   struct bHYPRE_StructGrid__data * data;
   HYPRE_StructGrid * Hgrid;
   data = bHYPRE_StructGrid__get_data( self );
   Hgrid = &(data -> grid);
   hypre_assert( *Hgrid==NULL );  /* grid shouldn't have already been created */

   ierr += HYPRE_StructGridCreate( data->comm, dim, Hgrid );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetDimension) */
  }
}

/*
 *  Define the lower and upper corners of a box of the grid.
 * "ilower" and "iupper" are arrays of size "dim", the number of spatial
 * dimensions. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetExtents"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructGrid_SetExtents(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
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
}

/*
 *  Set the periodicity for the grid.  Default is no periodicity.
 * 
 * The argument {\tt periodic} is an {\tt dim}-dimensional integer array that
 * contains the periodicity for each dimension.  A zero value for a dimension
 * means non-periodic, while a nonzero value means periodic and contains the
 * actual period.  For example, periodicity in the first and third dimensions
 * for a 10x11x12 grid is indicated by the array [10,0,12].
 * 
 * NOTE: Some of the solvers in hypre have power-of-two restrictions on the size
 * of the periodic dimensions.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetPeriodic"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructGrid_SetPeriodic(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim] */ int32_t* periodic,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
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
}

/*
 *  Set the number of ghost zones, separately on the lower and upper sides
 * for each dimension.
 * "num_ghost" is an array of size "dim2", twice the number of dimensions. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetNumGhost"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructGrid_SetNumGhost(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
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
}

/*
 *  final construction of the object before its use 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_Assemble"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructGrid_Assemble(
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
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
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructGrid_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructGrid_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructGrid_fconnect_bHYPRE_StructGrid(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) {
  return bHYPRE_StructGrid__connectI(url, ar, _ex);
}
struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructGrid_fcast_bHYPRE_StructGrid(void* bi, sidl_BaseInterface* 
  _ex) {
  return bHYPRE_StructGrid__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_StructGrid_fconnect_sidl_BaseClass(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_StructGrid_fcast_sidl_BaseClass(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_BaseInterface(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructGrid_fcast_sidl_BaseInterface(void* bi, sidl_BaseInterface* 
  _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_StructGrid_fconnect_sidl_ClassInfo(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_StructGrid_fcast_sidl_ClassInfo(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructGrid_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructGrid_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}

