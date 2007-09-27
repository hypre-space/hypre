/*
 * File:          bHYPRE_ErrorHandler_Impl.c
 * Symbol:        bHYPRE.ErrorHandler-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for bHYPRE.ErrorHandler
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.ErrorHandler" (version 1.0.0)
 * 
 * ErrorHandler class is an interface to the hypre error handling system.
 * Its methods help interpret the error flag ierr returned by hypre functions.
 */

#include "bHYPRE_ErrorHandler_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.ErrorHandler._includes) */
/* Insert-Code-Here {bHYPRE.ErrorHandler._includes} (includes and arbitrary code) */

#include "_hypre_utilities.h"
#include "HYPRE_utilities.h"

/* DO-NOT-DELETE splicer.end(bHYPRE.ErrorHandler._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ErrorHandler__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ErrorHandler__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.ErrorHandler._load) */
    /* Insert-Code-Here {bHYPRE.ErrorHandler._load} (static class initializer method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.ErrorHandler._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ErrorHandler__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ErrorHandler__ctor(
  /* in */ bHYPRE_ErrorHandler self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.ErrorHandler._ctor) */
    /* Insert-Code-Here {bHYPRE.ErrorHandler._ctor} (constructor method) */
    /*
     * // boilerplate constructor
     * struct bHYPRE_ErrorHandler__data *dptr = (struct bHYPRE_ErrorHandler__data*)malloc(sizeof(struct bHYPRE_ErrorHandler__data));
     * if (dptr) {
     *   memset(dptr, 0, sizeof(struct bHYPRE_ErrorHandler__data));
     *   // initialize elements of dptr here
     * }
     * bHYPRE_ErrorHandler__set_data(self, dptr);
     */

    /* DO-NOT-DELETE splicer.end(bHYPRE.ErrorHandler._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ErrorHandler__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ErrorHandler__ctor2(
  /* in */ bHYPRE_ErrorHandler self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.ErrorHandler._ctor2) */
    /* Insert-Code-Here {bHYPRE.ErrorHandler._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.ErrorHandler._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ErrorHandler__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ErrorHandler__dtor(
  /* in */ bHYPRE_ErrorHandler self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.ErrorHandler._dtor) */
    /* Insert-Code-Here {bHYPRE.ErrorHandler._dtor} (destructor method) */
    /*
     * // boilerplate destructor
     * struct bHYPRE_ErrorHandler__data *dptr = bHYPRE_ErrorHandler__get_data(self);
     * if (dptr) {
     *   // free contained in dtor before next line
     *   free(dptr);
     *   bHYPRE_ErrorHandler__set_data(self, NULL);
     * }
     */

    /* DO-NOT-DELETE splicer.end(bHYPRE.ErrorHandler._dtor) */
  }
}

/*
 * The Check method will return nonzero when the error flag ierr
 * includes an error of type error\_code; and zero otherwise.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ErrorHandler_Check"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_ErrorHandler_Check(
  /* in */ int32_t ierr,
  /* in */ enum bHYPRE_ErrorCode__enum error_code,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.ErrorHandler.Check) */
    /* Insert-Code-Here {bHYPRE.ErrorHandler.Check} (Check method) */

     return HYPRE_CheckError( ierr, (int)error_code );

    /* DO-NOT-DELETE splicer.end(bHYPRE.ErrorHandler.Check) */
  }
}

/*
 * included in the error flag ierr.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_ErrorHandler_Describe"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_ErrorHandler_Describe(
  /* in */ int32_t ierr,
  /* out */ char** message,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.ErrorHandler.Describe) */
    /* Insert-Code-Here {bHYPRE.ErrorHandler.Describe} (Describe method) */

     int i;
     char * msg = hypre_CTAlloc( char, 128 );
     char * msg2 = msg;
     HYPRE_DescribeError( ierr, msg2 );
     *message = msg2;

     return;

    /* DO-NOT-DELETE splicer.end(bHYPRE.ErrorHandler.Describe) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_ErrorHandler__object* 
  impl_bHYPRE_ErrorHandler_fconnect_bHYPRE_ErrorHandler(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_ErrorHandler__connectI(url, ar, _ex);
}
struct bHYPRE_ErrorHandler__object* 
  impl_bHYPRE_ErrorHandler_fcast_bHYPRE_ErrorHandler(void* bi, 
  sidl_BaseInterface* _ex) {
  return bHYPRE_ErrorHandler__cast(bi, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_ErrorHandler_fconnect_sidl_BaseClass(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_bHYPRE_ErrorHandler_fcast_sidl_BaseClass(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_ErrorHandler_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_ErrorHandler_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_ErrorHandler_fconnect_sidl_ClassInfo(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_bHYPRE_ErrorHandler_fcast_sidl_ClassInfo(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_ErrorHandler_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_ErrorHandler_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}

