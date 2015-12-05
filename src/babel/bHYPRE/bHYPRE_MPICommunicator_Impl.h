/*
 * File:          bHYPRE_MPICommunicator_Impl.h
 * Symbol:        bHYPRE.MPICommunicator-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.MPICommunicator
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_bHYPRE_MPICommunicator_Impl_h
#define included_bHYPRE_MPICommunicator_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator._includes) */
/* Insert-Code-Here {bHYPRE.MPICommunicator._includes} (include files) */

#include "_hypre_utilities.h"

/* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator._includes) */

/*
 * Private data for class bHYPRE.MPICommunicator
 */

struct bHYPRE_MPICommunicator__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.MPICommunicator._data) */
  /* Insert-Code-Here {bHYPRE.MPICommunicator._data} (private data members) */

   MPI_Comm mpi_comm;

  /* DO-NOT-DELETE splicer.end(bHYPRE.MPICommunicator._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_MPICommunicator__data*
bHYPRE_MPICommunicator__get_data(
  bHYPRE_MPICommunicator);

extern void
bHYPRE_MPICommunicator__set_data(
  bHYPRE_MPICommunicator,
  struct bHYPRE_MPICommunicator__data*);

extern
void
impl_bHYPRE_MPICommunicator__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_MPICommunicator__ctor(
  /* in */ bHYPRE_MPICommunicator self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_MPICommunicator__ctor2(
  /* in */ bHYPRE_MPICommunicator self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_MPICommunicator__dtor(
  /* in */ bHYPRE_MPICommunicator self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
bHYPRE_MPICommunicator
impl_bHYPRE_MPICommunicator_CreateC(
  /* in */ void* mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
bHYPRE_MPICommunicator
impl_bHYPRE_MPICommunicator_CreateF(
  /* in */ void* mpi_comm,
  /* out */ sidl_BaseInterface *_ex);

extern
bHYPRE_MPICommunicator
impl_bHYPRE_MPICommunicator_Create_MPICommWorld(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_MPICommunicator_Init(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_MPICommunicator_Finalize(
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_MPICommunicator_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_MPICommunicator_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_MPICommunicator_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_MPICommunicator_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_MPICommunicator_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_MPICommunicator_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern
void
impl_bHYPRE_MPICommunicator_Destroy(
  /* in */ bHYPRE_MPICommunicator self,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_MPICommunicator_fconnect_bHYPRE_MPICommunicator(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_MPICommunicator_fcast_bHYPRE_MPICommunicator(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_MPICommunicator_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_MPICommunicator_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_MPICommunicator_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_MPICommunicator_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);

#ifdef __cplusplus
}
#endif
#endif
