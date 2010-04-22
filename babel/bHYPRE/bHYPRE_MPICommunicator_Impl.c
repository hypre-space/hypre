/*
 * File:          bHYPRE_MPICommunicator_Impl.c
 * Symbol:        bHYPRE.MPICommunicator-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.MPICommunicator
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.MPICommunicator" (version 1.0.0)
 * 
 * MPICommunicator class
 * - two general Create functions: use CreateC if called from C code,
 * CreateF if called from Fortran code.
 * - Create_MPICommWorld will create a MPICommunicator to represent
 * MPI_Comm_World, and can be called from any language.
 */

#include "bHYPRE_MPICommunicator_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator._includes) */
/* Insert-Code-Here {bHYPRE.MPICommunicator._includes} (includes and arbitrary code) */


#include <stddef.h>
#include "hypre_babel_exception_handler.h"

/* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_MPICommunicator__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_MPICommunicator__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator._load) */
  /* Insert-Code-Here {bHYPRE.MPICommunicator._load} (static class initializer method) */
    /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_MPICommunicator__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_MPICommunicator__ctor(
  /* in */ bHYPRE_MPICommunicator self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator._ctor) */
  /* Insert-Code-Here {bHYPRE.MPICommunicator._ctor} (constructor method) */

   struct bHYPRE_MPICommunicator__data * data;
   data = hypre_CTAlloc( struct bHYPRE_MPICommunicator__data, 1 );

   data->mpi_comm = MPI_COMM_NULL;
   bHYPRE_MPICommunicator__set_data( self, data );

    /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_MPICommunicator__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_MPICommunicator__ctor2(
  /* in */ bHYPRE_MPICommunicator self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator._ctor2) */
    /* Insert-Code-Here {bHYPRE.MPICommunicator._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_MPICommunicator__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_MPICommunicator__dtor(
  /* in */ bHYPRE_MPICommunicator self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator._dtor) */
  /* Insert-Code-Here {bHYPRE.MPICommunicator._dtor} (destructor method) */

   struct bHYPRE_MPICommunicator__data * data;
   data = bHYPRE_MPICommunicator__get_data( self );
   hypre_TFree( data );

    /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator._dtor) */
  }
}

/*
 *  Create an MPICommunicator object from C code. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_MPICommunicator_CreateC"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_MPICommunicator
impl_bHYPRE_MPICommunicator_CreateC(
  /* in */ void* mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator.CreateC) */
      /* Insert-Code-Here {bHYPRE.MPICommunicator.CreateC} (CreateC method) */

      MPI_Comm mpicomm;
      bHYPRE_MPICommunicator bmpicomm = bHYPRE_MPICommunicator__create(_ex); SIDL_CHECK(*_ex);
      struct bHYPRE_MPICommunicator__data * data;

      if ( mpi_comm )
      {
         mpicomm = *( (MPI_Comm *) mpi_comm );
         data = bHYPRE_MPICommunicator__get_data(bmpicomm);
         data -> mpi_comm = mpicomm;
      }
      /* If mpi_comm is NULL, the default, MPI_COMM_NULL, is fine */

      return bmpicomm;

      hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator.CreateC) */
  }
}

/*
 *  Create an MPICommunicator object from Fortran code. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_MPICommunicator_CreateF"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_MPICommunicator
impl_bHYPRE_MPICommunicator_CreateF(
  /* in */ void* mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator.CreateF) */
  /* Insert-Code-Here {bHYPRE.MPICommunicator.CreateF} (CreateF method) */

   ptrdiff_t mpi_int = (ptrdiff_t) mpi_comm;   /* void* to integer of same length */
   /* not used MPI_Fint mpi_Fint = mpi_int;*/    /* pointer-length integer to MPI handle-length */
   MPI_Comm mpicomm_C = MPI_Comm_f2c( mpi_int );
   /* ... convert the MPI communicator from Fortran form (an integer handle)
      to C form (MPI_Comm).
      This function exists in current versions of LAM and MPICH, and is part of the
      MPI 2 standard; if someone is using an MPI library without it, we'll have to
      consider writing our own. */
   bHYPRE_MPICommunicator bmpicomm = bHYPRE_MPICommunicator__create(_ex); SIDL_CHECK(*_ex);
   struct bHYPRE_MPICommunicator__data * data =
      bHYPRE_MPICommunicator__get_data(bmpicomm);

   data -> mpi_comm = mpicomm_C;

   return bmpicomm;

   hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator.CreateF) */
  }
}

/*
 *  Create an MPICommunicator object which represents MPI_Comm_World. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_MPICommunicator_Create_MPICommWorld"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_MPICommunicator
impl_bHYPRE_MPICommunicator_Create_MPICommWorld(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator.Create_MPICommWorld) */
    /* Insert-Code-Here {bHYPRE.MPICommunicator.Create_MPICommWorld} (Create_MPICommWorld method) */

     MPI_Comm mpicomm = MPI_COMM_WORLD;
     bHYPRE_MPICommunicator bmpicomm = bHYPRE_MPICommunicator__create(_ex); SIDL_CHECK(*_ex);
     struct bHYPRE_MPICommunicator__data * data =
        bHYPRE_MPICommunicator__get_data(bmpicomm);

     data -> mpi_comm = mpicomm;

     return bmpicomm;

     hypre_babel_exception_no_return(_ex);
    /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator.Create_MPICommWorld) */
  }
}

/*
 *  Init and Finalize are to help debug MPI interfaces;
 * you should normally use the MPI library more directly:
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_MPICommunicator_Init"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_MPICommunicator_Init(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator.Init) */
    /* Insert-Code-Here {bHYPRE.MPICommunicator.Init} (Init method) */
     
     int argc = 1;
     char* argv1="no-args";
     char** argv = &argv1;

     MPI_Init(&argc, &argv);

     return;

    /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator.Init) */
  }
}

/*
 * Method:  Finalize[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_MPICommunicator_Finalize"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_MPICommunicator_Finalize(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator.Finalize) */
    /* Insert-Code-Here {bHYPRE.MPICommunicator.Finalize} (Finalize method) */

     MPI_Finalize();

     return;

    /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator.Finalize) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_MPICommunicator_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_MPICommunicator_Destroy(
  /* in */ bHYPRE_MPICommunicator self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator.Destroy) */
    /* Insert-Code-Here {bHYPRE.MPICommunicator.Destroy} (Destroy method) */
     bHYPRE_MPICommunicator_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator.Destroy) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_MPICommunicator_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_MPICommunicator_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_MPICommunicator_fcast_sidl_BaseClass(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_MPICommunicator_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_MPICommunicator_fcast_sidl_ClassInfo(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_MPICommunicator_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}

