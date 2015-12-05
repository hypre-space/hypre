/*
 * File:          bHYPRE_ErrorHandler.h
 * Symbol:        bHYPRE.ErrorHandler-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.ErrorHandler
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_ErrorHandler_h
#define included_bHYPRE_ErrorHandler_h

/**
 * Symbol "bHYPRE.ErrorHandler" (version 1.0.0)
 * 
 * ErrorHandler class is an interface to the hypre error handling system.
 * Its methods help interpret the error flag ierr returned by hypre functions.
 */
struct bHYPRE_ErrorHandler__object;
struct bHYPRE_ErrorHandler__array;
typedef struct bHYPRE_ErrorHandler__object* bHYPRE_ErrorHandler;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_ErrorCode_h
#include "bHYPRE_ErrorCode.h"
#endif
#ifndef included_sidl_BaseException_h
#include "sidl_BaseException.h"
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
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif

#ifndef included_sidl_rmi_Call_h
#include "sidl_rmi_Call.h"
#endif
#ifndef included_sidl_rmi_Return_h
#include "sidl_rmi_Return.h"
#endif
#ifdef SIDL_C_HAS_INLINE
#ifndef included_bHYPRE_ErrorHandler_IOR_h
#include "bHYPRE_ErrorHandler_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct bHYPRE_ErrorHandler__object*
bHYPRE_ErrorHandler__create(sidl_BaseInterface* _ex);

/**
 * RMI constructor function for the class.
 */
bHYPRE_ErrorHandler
bHYPRE_ErrorHandler__createRemote(const char * url, sidl_BaseInterface *_ex);

/**
 * Wraps up the private data struct pointer (struct bHYPRE\_ErrorHandler\_\_data) passed in rather than running the constructor.
 */
bHYPRE_ErrorHandler
bHYPRE_ErrorHandler__wrapObj(void * data, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_ErrorHandler
bHYPRE_ErrorHandler__connect(const char *, sidl_BaseInterface *_ex);

/**
 * The Check method will return nonzero when the error flag ierr
 * includes an error of type error\_code; and zero otherwise.
 */
int32_t
bHYPRE_ErrorHandler_Check(
  /* in */ int32_t ierr,
  /* in */ enum bHYPRE_ErrorCode__enum error_code,
  /* out */ sidl_BaseInterface *_ex);

/**
 * The Describe method will return a string describing the errors
 * included in the error flag ierr.
 */
void
bHYPRE_ErrorHandler_Describe(
  /* in */ int32_t ierr,
  /* out */ char** message,
  /* out */ sidl_BaseInterface *_ex);

SIDL_C_INLINE_DECL
void
bHYPRE_ErrorHandler_addRef(
  /* in */ bHYPRE_ErrorHandler self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_addRef)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_ErrorHandler_deleteRef(
  /* in */ bHYPRE_ErrorHandler self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_deleteRef)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_ErrorHandler_isSame(
  /* in */ bHYPRE_ErrorHandler self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_ErrorHandler_isType(
  /* in */ bHYPRE_ErrorHandler self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_isType)(
    self,
    name,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_ClassInfo
bHYPRE_ErrorHandler_getClassInfo(
  /* in */ bHYPRE_ErrorHandler self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_getClassInfo)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_ErrorHandler__object*
bHYPRE_ErrorHandler__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_ErrorHandler__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_ErrorHandler__exec(
  /* in */ bHYPRE_ErrorHandler self,
  /* in */ const char* methodName,
  /* in */ sidl_rmi_Call inArgs,
  /* in */ sidl_rmi_Return outArgs,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f__exec)(
    self,
    methodName,
    inArgs,
    outArgs,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * Get the URL of the Implementation of this object (for RMI)
 */
SIDL_C_INLINE_DECL
char*
bHYPRE_ErrorHandler__getURL(
  /* in */ bHYPRE_ErrorHandler self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f__getURL)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * On a remote object, addrefs the remote instance.
 */
SIDL_C_INLINE_DECL
void
bHYPRE_ErrorHandler__raddRef(
  /* in */ bHYPRE_ErrorHandler self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f__raddRef)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * TRUE if this object is remote, false if local
 */
SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_ErrorHandler__isRemote(
  /* in */ bHYPRE_ErrorHandler self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f__isRemote)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * TRUE if this object is remote, false if local
 */
sidl_bool
bHYPRE_ErrorHandler__isLocal(
  /* in */ bHYPRE_ErrorHandler self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_ErrorHandler__array*
bHYPRE_ErrorHandler__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_ErrorHandler__array*
bHYPRE_ErrorHandler__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_ErrorHandler__array*
bHYPRE_ErrorHandler__array_create1d(int32_t len);

struct bHYPRE_ErrorHandler__array*
bHYPRE_ErrorHandler__array_create1dInit(
  int32_t len, 
  bHYPRE_ErrorHandler* data);

struct bHYPRE_ErrorHandler__array*
bHYPRE_ErrorHandler__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_ErrorHandler__array*
bHYPRE_ErrorHandler__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_ErrorHandler__array*
bHYPRE_ErrorHandler__array_borrow(
  bHYPRE_ErrorHandler* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_ErrorHandler__array*
bHYPRE_ErrorHandler__array_smartCopy(
  struct bHYPRE_ErrorHandler__array *array);

void
bHYPRE_ErrorHandler__array_addRef(
  struct bHYPRE_ErrorHandler__array* array);

void
bHYPRE_ErrorHandler__array_deleteRef(
  struct bHYPRE_ErrorHandler__array* array);

bHYPRE_ErrorHandler
bHYPRE_ErrorHandler__array_get1(
  const struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1);

bHYPRE_ErrorHandler
bHYPRE_ErrorHandler__array_get2(
  const struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_ErrorHandler
bHYPRE_ErrorHandler__array_get3(
  const struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_ErrorHandler
bHYPRE_ErrorHandler__array_get4(
  const struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_ErrorHandler
bHYPRE_ErrorHandler__array_get5(
  const struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_ErrorHandler
bHYPRE_ErrorHandler__array_get6(
  const struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_ErrorHandler
bHYPRE_ErrorHandler__array_get7(
  const struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_ErrorHandler
bHYPRE_ErrorHandler__array_get(
  const struct bHYPRE_ErrorHandler__array* array,
  const int32_t indices[]);

void
bHYPRE_ErrorHandler__array_set1(
  struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1,
  bHYPRE_ErrorHandler const value);

void
bHYPRE_ErrorHandler__array_set2(
  struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_ErrorHandler const value);

void
bHYPRE_ErrorHandler__array_set3(
  struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_ErrorHandler const value);

void
bHYPRE_ErrorHandler__array_set4(
  struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_ErrorHandler const value);

void
bHYPRE_ErrorHandler__array_set5(
  struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_ErrorHandler const value);

void
bHYPRE_ErrorHandler__array_set6(
  struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_ErrorHandler const value);

void
bHYPRE_ErrorHandler__array_set7(
  struct bHYPRE_ErrorHandler__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_ErrorHandler const value);

void
bHYPRE_ErrorHandler__array_set(
  struct bHYPRE_ErrorHandler__array* array,
  const int32_t indices[],
  bHYPRE_ErrorHandler const value);

int32_t
bHYPRE_ErrorHandler__array_dimen(
  const struct bHYPRE_ErrorHandler__array* array);

int32_t
bHYPRE_ErrorHandler__array_lower(
  const struct bHYPRE_ErrorHandler__array* array,
  const int32_t ind);

int32_t
bHYPRE_ErrorHandler__array_upper(
  const struct bHYPRE_ErrorHandler__array* array,
  const int32_t ind);

int32_t
bHYPRE_ErrorHandler__array_length(
  const struct bHYPRE_ErrorHandler__array* array,
  const int32_t ind);

int32_t
bHYPRE_ErrorHandler__array_stride(
  const struct bHYPRE_ErrorHandler__array* array,
  const int32_t ind);

int
bHYPRE_ErrorHandler__array_isColumnOrder(
  const struct bHYPRE_ErrorHandler__array* array);

int
bHYPRE_ErrorHandler__array_isRowOrder(
  const struct bHYPRE_ErrorHandler__array* array);

struct bHYPRE_ErrorHandler__array*
bHYPRE_ErrorHandler__array_slice(
  struct bHYPRE_ErrorHandler__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_ErrorHandler__array_copy(
  const struct bHYPRE_ErrorHandler__array* src,
  struct bHYPRE_ErrorHandler__array* dest);

struct bHYPRE_ErrorHandler__array*
bHYPRE_ErrorHandler__array_ensure(
  struct bHYPRE_ErrorHandler__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_ErrorHandler__connectI

#pragma weak bHYPRE_ErrorHandler__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_ErrorHandler__object*
bHYPRE_ErrorHandler__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_ErrorHandler__object*
bHYPRE_ErrorHandler__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
