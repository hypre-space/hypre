/*
 * File:          bHYPRE_SStructGrid_Impl.c
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructGrid" (version 1.0.0)
 * 
 * The semi-structured grid class.
 */

#include "bHYPRE_SStructGrid_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._includes) */
/* Put additional includes or other arbitrary code here... */



#include "hypre_babel_exception_handler.h"
/*#include "mpi.h"*/
#include "HYPRE_sstruct_mv.h"
#include "_hypre_sstruct_mv.h"
#include "_hypre_utilities.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructGrid__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._load) */
  /* Insert-Code-Here {bHYPRE.SStructGrid._load} (static class initializer method) */
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructGrid__ctor(
  /* in */ bHYPRE_SStructGrid self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct bHYPRE_SStructGrid__data * data;
   data = hypre_CTAlloc( struct bHYPRE_SStructGrid__data, 1 );
   data -> grid = NULL;
   data -> comm = MPI_COMM_NULL;
   bHYPRE_SStructGrid__set_data( self, data );

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructGrid__ctor2(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._ctor2) */
    /* Insert-Code-Here {bHYPRE.SStructGrid._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructGrid__dtor(
  /* in */ bHYPRE_SStructGrid self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_SStructGrid__data * data;
   HYPRE_SStructGrid Hgrid;
   data = bHYPRE_SStructGrid__get_data( self );
   Hgrid = data -> grid;
   ierr = HYPRE_SStructGridDestroy( Hgrid );
   hypre_assert( ierr==0 );
   hypre_TFree( data );

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid._dtor) */
  }
}

/*
 *  This function is the preferred way to create a SStruct Grid. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_SStructGrid
impl_bHYPRE_SStructGrid_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t ndim,
  /* in */ int32_t nparts,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.Create) */
  /* Insert-Code-Here {bHYPRE.SStructGrid.Create} (Create method) */

   int ierr = 0;
   bHYPRE_SStructGrid grid;
   struct bHYPRE_SStructGrid__data * data;
   HYPRE_SStructGrid Hgrid;
   MPI_Comm comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   grid = bHYPRE_SStructGrid__create(_ex); SIDL_CHECK(*_ex);
   data = bHYPRE_SStructGrid__get_data( grid );

   ierr += HYPRE_SStructGridCreate( comm, ndim, nparts, &Hgrid );
   data->grid = Hgrid;
   data->comm = comm;

   return grid;

   hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.Create) */
  }
}

/*
 * Method:  SetNumDimParts[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_SetNumDimParts"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGrid_SetNumDimParts(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t ndim,
  /* in */ int32_t nparts,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.SetNumDimParts) */
  /* Insert the implementation of the SetNumDimParts method here... */

   /* DEPRECATED   use _Create */

   int ierr = 0;
   struct bHYPRE_SStructGrid__data * data;
   HYPRE_SStructGrid * Hgrid;
   data = bHYPRE_SStructGrid__get_data( self );
   Hgrid = &(data -> grid);
   hypre_assert( *Hgrid==NULL );  /* grid shouldn't have already been created */

   ierr += HYPRE_SStructGridCreate( data->comm, ndim, nparts, Hgrid );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.SetNumDimParts) */
  }
}

/*
 * Method:  SetCommunicator[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGrid_SetCommunicator(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   /* DEPRECATED   use _Create */

   int ierr = 0;
   struct bHYPRE_SStructGrid__data * data;
   data = bHYPRE_SStructGrid__get_data( self );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructGrid_Destroy(
  /* in */ bHYPRE_SStructGrid self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.Destroy) */
    /* Insert-Code-Here {bHYPRE.SStructGrid.Destroy} (Destroy method) */
     bHYPRE_SStructGrid_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.Destroy) */
  }
}

/*
 * Set the extents for a box on a structured part of the grid.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_SetExtents"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGrid_SetExtents(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.SetExtents) */
  /* Insert the implementation of the SetExtents method here... */

   int ierr = 0;
   struct bHYPRE_SStructGrid__data * data;
   HYPRE_SStructGrid Hgrid;
   data = bHYPRE_SStructGrid__get_data( self );
   Hgrid = data -> grid;

   ierr += HYPRE_SStructGridSetExtents( Hgrid, part,
                                        ilower,
                                        iupper );
   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.SetExtents) */
  }
}

/*
 * Describe the variables that live on a structured part of the
 * grid.  Input: part number, variable number, total number of
 * variables on that part (needed for memory allocation),
 * variable type.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_SetVariable"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGrid_SetVariable(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t nvars,
  /* in */ enum bHYPRE_SStructVariable__enum vartype,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.SetVariable) */
  /* Insert the implementation of the SetVariable method here... */

   /* note: the relevent enums are defined in bHYPRE_SStructVariable_IOR.h
      (derived from Interfaces.idl) and HYPRE_sstruct_mv.h .  They should
      be equivalent, and were when I last checked. */
   /* also note: there would be a SetVariables in the sidl file, except
      that babeldoesn't suport an array of enums */

   int ierr = 0;
   struct bHYPRE_SStructGrid__data * data;
   HYPRE_SStructGrid Hgrid;
   data = bHYPRE_SStructGrid__get_data( self );
   Hgrid = data -> grid;

   ierr += HYPRE_SStructGridSetVariable( Hgrid, part, var, nvars,
                                         (HYPRE_SStructVariable) vartype );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.SetVariable) */
  }
}

/*
 * Describe additional variables that live at a particular
 * index.  These variables are appended to the array of
 * variables set in {\tt SetVariables}, and are referenced as
 * such.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_AddVariable"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGrid_AddVariable(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ enum bHYPRE_SStructVariable__enum vartype,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.AddVariable) */
  /* Insert the implementation of the AddVariable method here... */

   int ierr = 0;
   struct bHYPRE_SStructGrid__data * data;
   HYPRE_SStructGrid Hgrid;
   data = bHYPRE_SStructGrid__get_data( self );
   Hgrid = data -> grid;

   ierr += HYPRE_SStructGridAddVariable( Hgrid, part, 
                                         index,
                                         var,
                                         (HYPRE_SStructVariable) vartype );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.AddVariable) */
  }
}

/*
 * Describe how regions just outside of a part relate to other
 * parts.  This is done a box at a time.
 * 
 * The indexes {\tt ilower} and {\tt iupper} map directly to the
 * indexes {\tt nbor\_ilower} and {\tt nbor\_iupper}.  Although,
 * it is required that indexes increase from {\tt ilower} to
 * {\tt iupper}, indexes may increase and/or decrease from {\tt
 * nbor\_ilower} to {\tt nbor\_iupper}.
 * 
 * The {\tt index\_map} describes the mapping of indexes 0, 1,
 * and 2 on part {\tt part} to the corresponding indexes on part
 * {\tt nbor\_part}.  For example, triple (1, 2, 0) means that
 * indexes 0, 1, and 2 on part {\tt part} map to indexes 1, 2,
 * and 0 on part {\tt nbor\_part}, respectively.
 * 
 * NOTE: All parts related to each other via this routine must
 * have an identical list of variables and variable types.  For
 * example, if part 0 has only two variables on it, a cell
 * centered variable and a node centered variable, and we
 * declare part 1 to be a neighbor of part 0, then part 1 must
 * also have only two variables on it, and they must be of type
 * cell and node.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_SetNeighborBox"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGrid_SetNeighborBox(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t nbor_part,
  /* in rarray[dim] */ int32_t* nbor_ilower,
  /* in rarray[dim] */ int32_t* nbor_iupper,
  /* in rarray[dim] */ int32_t* index_map,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
{
  int  index_dir[3] = {1, 1, 1};
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.SetNeighborBox) */
  /* Insert the implementation of the SetNeighborBox method here... */

   int ierr = 0;
   struct bHYPRE_SStructGrid__data * data;
   HYPRE_SStructGrid Hgrid;
   data = bHYPRE_SStructGrid__get_data( self );
   Hgrid = data -> grid;

   ierr += HYPRE_SStructGridSetNeighborPart
      ( Hgrid, part,
        ilower,
        iupper,
        nbor_part,
        nbor_ilower,
        nbor_iupper,
        index_map,
        index_dir );

      return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.SetNeighborBox) */
  }
}

/*
 * Add an unstructured part to the grid.  The variables in the
 * unstructured part of the grid are referenced by a global rank
 * between 0 and the total number of unstructured variables
 * minus one.  Each process owns some unique consecutive range
 * of variables, defined by {\tt ilower} and {\tt iupper}.
 * 
 * NOTE: This is just a placeholder.  This part of the interface
 * is not finished.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_AddUnstructuredPart"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGrid_AddUnstructuredPart(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.AddUnstructuredPart) */
  /* Insert the implementation of the AddUnstructuredPart method here... */

#if 0
   /* the function HYPRE_SStructGridAddUnstructuredPart hasn't been implemented yet */
   int ierr = 0;
   struct bHYPRE_SStructGrid__data * data;
   HYPRE_SStructGrid Hgrid;
   data = bHYPRE_SStructGrid__get_data( self );
   Hgrid = data -> grid;

   ierr += HYPRE_SStructGridAddUnstructuredPart( Hgrid, ilower, iupper );

   return ierr;
#endif

   return 1;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.AddUnstructuredPart) */
  }
}

/*
 * (Optional) Set periodic for a particular part.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_SetPeriodic"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGrid_SetPeriodic(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* periodic,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.SetPeriodic) */
  /* Insert the implementation of the SetPeriodic method here... */

   int ierr = 0;
   struct bHYPRE_SStructGrid__data * data;
   HYPRE_SStructGrid Hgrid;
   data = bHYPRE_SStructGrid__get_data( self );
   Hgrid = data -> grid;

   ierr += HYPRE_SStructGridSetPeriodic( Hgrid, part,
                                         periodic );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.SetPeriodic) */
  }
}

/*
 * Setting ghost in the sgrids.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_SetNumGhost"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGrid_SetNumGhost(
  /* in */ bHYPRE_SStructGrid self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.SetNumGhost) */
  /* Insert the implementation of the SetNumGhost method here... */

   int ierr = 0;
   struct bHYPRE_SStructGrid__data * data;
   HYPRE_SStructGrid Hgrid;
   data = bHYPRE_SStructGrid__get_data( self );
   Hgrid = data -> grid;

   ierr += HYPRE_SStructGridSetNumGhost( Hgrid, num_ghost );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.SetNumGhost) */
  }
}

/*
 *  final construction of the object before its use 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_Assemble"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructGrid_Assemble(
  /* in */ bHYPRE_SStructGrid self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.Assemble) */
  /* Insert the implementation of the Assemble method here... */

   int ierr = 0;
   struct bHYPRE_SStructGrid__data * data;
   HYPRE_SStructGrid Hgrid;
   data = bHYPRE_SStructGrid__get_data( self );
   Hgrid = data -> grid;

   ierr += HYPRE_SStructGridAssemble( Hgrid );

   return ierr;

    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.Assemble) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructGrid_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructGrid_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGrid_fconnect_bHYPRE_SStructGrid(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_SStructGrid__connectI(url, ar, _ex);
}
struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructGrid_fcast_bHYPRE_SStructGrid(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_SStructGrid__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_SStructGrid_fconnect_sidl_BaseClass(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_SStructGrid_fcast_sidl_BaseClass(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructGrid_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructGrid_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_SStructGrid_fconnect_sidl_ClassInfo(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_SStructGrid_fcast_sidl_ClassInfo(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructGrid_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructGrid_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}

