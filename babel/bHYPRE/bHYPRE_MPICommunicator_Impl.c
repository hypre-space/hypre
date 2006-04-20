/*
 * File:          bHYPRE_MPICommunicator_Impl.c
 * Symbol:        bHYPRE.MPICommunicator-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for bHYPRE.MPICommunicator
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.MPICommunicator" (version 1.0.0)
 * 
 * MPICommunicator class
 *  two Create functions: use CreateC if called from C code,
 *  CreateF if called from Fortran code
 * 
 * 
 */

#include "bHYPRE_MPICommunicator_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator._includes) */
/* Insert-Code-Here {bHYPRE.MPICommunicator._includes} (includes and arbitrary code) */

#include <stddef.h>
extern MPI_Comm MPI_Comm_f2c(ptrdiff_t mpi_int);

/* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator._includes) */

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
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator._load) */
  /* Insert-Code-Here {bHYPRE.MPICommunicator._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator._load) */
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
  /* in */ bHYPRE_MPICommunicator self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator._ctor) */
  /* Insert-Code-Here {bHYPRE.MPICommunicator._ctor} (constructor method) */

   struct bHYPRE_MPICommunicator__data * data;
   data = hypre_CTAlloc( struct bHYPRE_MPICommunicator__data, 1 );

   data->mpi_comm = MPI_COMM_NULL;
   bHYPRE_MPICommunicator__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator._ctor) */
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
  /* in */ bHYPRE_MPICommunicator self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator._dtor) */
  /* Insert-Code-Here {bHYPRE.MPICommunicator._dtor} (destructor method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator._dtor) */
}

/*
 * Method:  CreateC[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_MPICommunicator_CreateC"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_MPICommunicator
impl_bHYPRE_MPICommunicator_CreateC(
  /* in */ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator.CreateC) */
  /* Insert-Code-Here {bHYPRE.MPICommunicator.CreateC} (CreateC method) */

   MPI_Comm mpicomm = *( (MPI_Comm *) mpi_comm );
   bHYPRE_MPICommunicator bmpicomm = bHYPRE_MPICommunicator__create();
   struct bHYPRE_MPICommunicator__data * data =
      bHYPRE_MPICommunicator__get_data(bmpicomm);

   data -> mpi_comm = mpicomm;

   return bmpicomm;

  /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator.CreateC) */
}

/*
 * Method:  CreateF[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_MPICommunicator_CreateF"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_MPICommunicator
impl_bHYPRE_MPICommunicator_CreateF(
  /* in */ void* mpi_comm)
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
   bHYPRE_MPICommunicator bmpicomm = bHYPRE_MPICommunicator__create();
   struct bHYPRE_MPICommunicator__data * data =
      bHYPRE_MPICommunicator__get_data(bmpicomm);

   data -> mpi_comm = mpicomm_C;

   return bmpicomm;

  /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator.CreateF) */
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_MPICommunicator_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connect(url, _ex);
}
char * impl_bHYPRE_MPICommunicator_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) {
  return bHYPRE_MPICommunicator__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_MPICommunicator_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_MPICommunicator_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_MPICommunicator_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
