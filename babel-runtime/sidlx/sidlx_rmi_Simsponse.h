/*
 * File:          sidlx_rmi_Simsponse.h
 * Symbol:        sidlx.rmi.Simsponse-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for sidlx.rmi.Simsponse
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_sidlx_rmi_Simsponse_h
#define included_sidlx_rmi_Simsponse_h

/**
 * Symbol "sidlx.rmi.Simsponse" (version 0.1)
 * 
 * implementation of Response using the Simhandle Protocol (written by Jim)
 */
struct sidlx_rmi_Simsponse__object;
struct sidlx_rmi_Simsponse__array;
typedef struct sidlx_rmi_Simsponse__object* sidlx_rmi_Simsponse;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
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
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#ifndef included_sidlx_rmi_Socket_h
#include "sidlx_rmi_Socket.h"
#endif

#ifndef included_sidl_rmi_Call_h
#include "sidl_rmi_Call.h"
#endif
#ifndef included_sidl_rmi_Return_h
#include "sidl_rmi_Return.h"
#endif
#ifdef SIDL_C_HAS_INLINE
#ifndef included_sidlx_rmi_Simsponse_IOR_h
#include "sidlx_rmi_Simsponse_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct sidlx_rmi_Simsponse__object*
sidlx_rmi_Simsponse__create(sidl_BaseInterface* _ex);

/**
 * RMI constructor function for the class.
 */
sidlx_rmi_Simsponse
sidlx_rmi_Simsponse__createRemote(const char * url, sidl_BaseInterface *_ex);

/**
 * Wraps up the private data struct pointer (struct sidlx_rmi_Simsponse__data) passed in rather than running the constructor.
 */
sidlx_rmi_Simsponse
sidlx_rmi_Simsponse__wrapObj(void * data, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.(addrefs)
 */
sidlx_rmi_Simsponse
sidlx_rmi_Simsponse__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Method:  init[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_init(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* methodName,
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_init)(
    self,
    methodName,
    objectid,
    sock,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  test[]
 */
SIDL_C_INLINE_DECL
sidl_bool
sidlx_rmi_Simsponse_test(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ int32_t secs,
  /* in */ int32_t usecs,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_test)(
    self,
    secs,
    usecs,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  pullData[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_pullData(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_pullData)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  getMethodName[]
 */
SIDL_C_INLINE_DECL
char*
sidlx_rmi_Simsponse_getMethodName(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_getMethodName)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  getObjectID[]
 */
SIDL_C_INLINE_DECL
char*
sidlx_rmi_Simsponse_getObjectID(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_getObjectID)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>sidl</code> have an intrinsic reference count.
 * Objects continue to exist as long as the reference count is
 * positive. Clients should call this method whenever they
 * create another ongoing reference to an object or interface.
 * </p>
 * <p>
 * This does not have a return value because there is no language
 * independent type that can refer to an interface or a
 * class.
 * </p>
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_addRef(
  /* in */ sidlx_rmi_Simsponse self,
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


/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_deleteRef(
  /* in */ sidlx_rmi_Simsponse self,
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


/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_C_INLINE_DECL
sidl_bool
sidlx_rmi_Simsponse_isSame(
  /* in */ sidlx_rmi_Simsponse self,
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


/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_C_INLINE_DECL
sidl_bool
sidlx_rmi_Simsponse_isType(
  /* in */ sidlx_rmi_Simsponse self,
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


/**
 * Return the meta-data about the class implementing this interface.
 */
SIDL_C_INLINE_DECL
sidl_ClassInfo
sidlx_rmi_Simsponse_getClassInfo(
  /* in */ sidlx_rmi_Simsponse self,
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
 *  
 * May return a communication exception or an execption thrown
 * from the remote server.  If it returns null, then it's safe
 * to unpack arguments
 */
SIDL_C_INLINE_DECL
sidl_BaseException
sidlx_rmi_Simsponse_getExceptionThrown(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_getExceptionThrown)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackBool[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackBool(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ sidl_bool* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackBool)(
    self,
    key,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackChar[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackChar(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ char* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackChar)(
    self,
    key,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackInt[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackInt(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackInt)(
    self,
    key,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackLong[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackLong(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ int64_t* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackLong)(
    self,
    key,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackOpaque[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackOpaque(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ void** value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackOpaque)(
    self,
    key,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackFloat[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackFloat(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ float* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackFloat)(
    self,
    key,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackDouble[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackDouble(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackDouble)(
    self,
    key,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackFcomplex[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackFcomplex(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ struct sidl_fcomplex* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackFcomplex)(
    self,
    key,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackDcomplex[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackDcomplex(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ struct sidl_dcomplex* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackDcomplex)(
    self,
    key,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackString[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackString(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ char** value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackString)(
    self,
    key,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackSerializable[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackSerializable(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ sidl_io_Serializable* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackSerializable)(
    self,
    key,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 *  unpack arrays of values 
 * It is possible to ensure an array is
 * in a certain order by passing in ordering and dimension
 * requirements.  ordering should represent a value in the
 * sidl_array_ordering enumeration in sidlArray.h If either
 * argument is 0, it means there is no restriction on that
 * aspect.  The rarray flag should be set if the array being
 * passed in is actually an rarray.  The semantics are slightly
 * different for rarrays.  The passed in array MUST be reused,
 * even if the array has changed bounds.
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackBoolArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<bool> */ struct sidl_bool__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackBoolArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackCharArray[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackCharArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<char> */ struct sidl_char__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackCharArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackIntArray[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackIntArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<int> */ struct sidl_int__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackIntArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackLongArray[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackLongArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<long> */ struct sidl_long__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackLongArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackOpaqueArray[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackOpaqueArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<opaque> */ struct sidl_opaque__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackOpaqueArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackFloatArray[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackFloatArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<float> */ struct sidl_float__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackFloatArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackDoubleArray[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackDoubleArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<double> */ struct sidl_double__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackDoubleArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackFcomplexArray[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackFcomplexArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<fcomplex> */ struct sidl_fcomplex__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackFcomplexArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackDcomplexArray[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackDcomplexArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<dcomplex> */ struct sidl_dcomplex__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackDcomplexArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackStringArray[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackStringArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<string> */ struct sidl_string__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackStringArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackGenericArray[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackGenericArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<> */ struct sidl__array** value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackGenericArray)(
    self,
    key,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  unpackSerializableArray[]
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse_unpackSerializableArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<sidl.io.Serializable> */ struct sidl_io_Serializable__array** 
    value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_unpackSerializableArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Cast method for interface and class type conversions.
 */
struct sidlx_rmi_Simsponse__object*
sidlx_rmi_Simsponse__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
sidlx_rmi_Simsponse__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
sidlx_rmi_Simsponse__exec(
  /* in */ sidlx_rmi_Simsponse self,
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
sidlx_rmi_Simsponse__getURL(
  /* in */ sidlx_rmi_Simsponse self,
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
sidlx_rmi_Simsponse__raddRef(
  /* in */ sidlx_rmi_Simsponse self,
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
sidlx_rmi_Simsponse__isRemote(
  /* in */ sidlx_rmi_Simsponse self,
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
sidlx_rmi_Simsponse__isLocal(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);
/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in column-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_rmi_Simsponse__array*
sidlx_rmi_Simsponse__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Create a contiguous array of the given dimension with specified
 * index bounds in row-major order. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_rmi_Simsponse__array*
sidlx_rmi_Simsponse__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Create a contiguous one-dimensional array with a lower index
 * of 0 and an upper index of len-1. This array
 * owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_rmi_Simsponse__array*
sidlx_rmi_Simsponse__array_create1d(int32_t len);

/**
 * Create a dense one-dimensional vector with a lower
 * index of 0 and an upper index of len-1. The initial data for this
 * new array is copied from data. This will increment the reference
 * count of each non-NULL object/interface reference in data.
 * 
 * This array owns and manages its data.
 */
struct sidlx_rmi_Simsponse__array*
sidlx_rmi_Simsponse__array_create1dInit(
  int32_t len, 
  sidlx_rmi_Simsponse* data);

/**
 * Create a contiguous two-dimensional array in column-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_rmi_Simsponse__array*
sidlx_rmi_Simsponse__array_create2dCol(int32_t m, int32_t n);

/**
 * Create a contiguous two-dimensional array in row-major
 * order with a lower index of (0,0) and an upper index of
 * (m-1,n-1). This array owns and manages its data.
 * This function initializes the contents of the array to
 * NULL.
 */
struct sidlx_rmi_Simsponse__array*
sidlx_rmi_Simsponse__array_create2dRow(int32_t m, int32_t n);

/**
 * Create an array that uses data (memory) from another
 * source. The initial contents are determined by the
 * data being borrowed.
 * Any time an element in the borrowed array is replaced
 * via a set call, deleteRef will be called on the
 * value being replaced if it is not NULL.
 */
struct sidlx_rmi_Simsponse__array*
sidlx_rmi_Simsponse__array_borrow(
  sidlx_rmi_Simsponse* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

/**
 * If array is borrowed, allocate a new self-sufficient
 * array and copy the borrowed array into the new array;
 * otherwise, increment the reference count and return
 * the array passed in. Use this whenever you want to
 * make a copy of a method argument because arrays
 * passed into methods aren't guaranteed to exist after
 * the method call.
 */
struct sidlx_rmi_Simsponse__array*
sidlx_rmi_Simsponse__array_smartCopy(
  struct sidlx_rmi_Simsponse__array *array);

/**
 * Increment the array's internal reference count by one.
 */
void
sidlx_rmi_Simsponse__array_addRef(
  struct sidlx_rmi_Simsponse__array* array);

/**
 * Decrement the array's internal reference count by one.
 * If the reference count goes to zero, destroy the array.
 * If the array isn't borrowed, this releases all the
 * object references held by the array.
 */
void
sidlx_rmi_Simsponse__array_deleteRef(
  struct sidlx_rmi_Simsponse__array* array);

/**
 * Retrieve element i1 of a(n) 1-dimensional array.
 */
sidlx_rmi_Simsponse
sidlx_rmi_Simsponse__array_get1(
  const struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1);

/**
 * Retrieve element (i1,i2) of a(n) 2-dimensional array.
 */
sidlx_rmi_Simsponse
sidlx_rmi_Simsponse__array_get2(
  const struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1,
  const int32_t i2);

/**
 * Retrieve element (i1,i2,i3) of a(n) 3-dimensional array.
 */
sidlx_rmi_Simsponse
sidlx_rmi_Simsponse__array_get3(
  const struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

/**
 * Retrieve element (i1,i2,i3,i4) of a(n) 4-dimensional array.
 */
sidlx_rmi_Simsponse
sidlx_rmi_Simsponse__array_get4(
  const struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

/**
 * Retrieve element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array.
 */
sidlx_rmi_Simsponse
sidlx_rmi_Simsponse__array_get5(
  const struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array.
 */
sidlx_rmi_Simsponse
sidlx_rmi_Simsponse__array_get6(
  const struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

/**
 * Retrieve element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array.
 */
sidlx_rmi_Simsponse
sidlx_rmi_Simsponse__array_get7(
  const struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

/**
 * Retrieve element indices of an n-dimensional array.
 * indices is assumed to have the right number of elements
 * for the dimension of array.
 */
sidlx_rmi_Simsponse
sidlx_rmi_Simsponse__array_get(
  const struct sidlx_rmi_Simsponse__array* array,
  const int32_t indices[]);

/**
 * Set element i1 of a(n) 1-dimensional array to value.
 */
void
sidlx_rmi_Simsponse__array_set1(
  struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1,
  sidlx_rmi_Simsponse const value);

/**
 * Set element (i1,i2) of a(n) 2-dimensional array to value.
 */
void
sidlx_rmi_Simsponse__array_set2(
  struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1,
  const int32_t i2,
  sidlx_rmi_Simsponse const value);

/**
 * Set element (i1,i2,i3) of a(n) 3-dimensional array to value.
 */
void
sidlx_rmi_Simsponse__array_set3(
  struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  sidlx_rmi_Simsponse const value);

/**
 * Set element (i1,i2,i3,i4) of a(n) 4-dimensional array to value.
 */
void
sidlx_rmi_Simsponse__array_set4(
  struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  sidlx_rmi_Simsponse const value);

/**
 * Set element (i1,i2,i3,i4,i5) of a(n) 5-dimensional array to value.
 */
void
sidlx_rmi_Simsponse__array_set5(
  struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  sidlx_rmi_Simsponse const value);

/**
 * Set element (i1,i2,i3,i4,i5,i6) of a(n) 6-dimensional array to value.
 */
void
sidlx_rmi_Simsponse__array_set6(
  struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  sidlx_rmi_Simsponse const value);

/**
 * Set element (i1,i2,i3,i4,i5,i6,i7) of a(n) 7-dimensional array to value.
 */
void
sidlx_rmi_Simsponse__array_set7(
  struct sidlx_rmi_Simsponse__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  sidlx_rmi_Simsponse const value);

/**
 * Set element indices of an n-dimensional array to value.indices is assumed to have the right number of elements
 * for the dimension of array.
 */
void
sidlx_rmi_Simsponse__array_set(
  struct sidlx_rmi_Simsponse__array* array,
  const int32_t indices[],
  sidlx_rmi_Simsponse const value);

/**
 * Return the dimension of array. If the array pointer is
 * NULL, zero is returned.
 */
int32_t
sidlx_rmi_Simsponse__array_dimen(
  const struct sidlx_rmi_Simsponse__array* array);

/**
 * Return the lower bound of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidlx_rmi_Simsponse__array_lower(
  const struct sidlx_rmi_Simsponse__array* array,
  const int32_t ind);

/**
 * Return the upper bound of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidlx_rmi_Simsponse__array_upper(
  const struct sidlx_rmi_Simsponse__array* array,
  const int32_t ind);

/**
 * Return the length of dimension ind.
 * If ind is not a valid dimension, -1 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidlx_rmi_Simsponse__array_length(
  const struct sidlx_rmi_Simsponse__array* array,
  const int32_t ind);

/**
 * Return the stride of dimension ind.
 * If ind is not a valid dimension, 0 is returned.
 * The valid range for ind is from 0 to dimen-1.
 */
int32_t
sidlx_rmi_Simsponse__array_stride(
  const struct sidlx_rmi_Simsponse__array* array,
  const int32_t ind);

/**
 * Return a true value iff the array is a contiguous
 * column-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
sidlx_rmi_Simsponse__array_isColumnOrder(
  const struct sidlx_rmi_Simsponse__array* array);

/**
 * Return a true value iff the array is a contiguous
 * row-major ordered array. A NULL array argument
 * causes 0 to be returned.
 */
int
sidlx_rmi_Simsponse__array_isRowOrder(
  const struct sidlx_rmi_Simsponse__array* array);

/**
 * Create a sub-array of another array. This resulting
 * array shares data with the original array. The new
 * array can be of the same dimension or potentially
 * less assuming the original array has dimension
 * greater than 1.  If you are removing dimension,
 * indicate the dimensions to remove by setting
 * numElem[i] to zero for any dimension i wthat should
 * go away in the new array.  The meaning of each
 * argument is covered below.
 * 
 * src       the array to be created will be a subset
 *           of this array. If this argument is NULL,
 *           NULL will be returned. The array returned
 *           borrows data from src, so modifying src or
 *           the returned array will modify both
 *           arrays.
 * 
 * dimen     this argument must be greater than zero
 *           and less than or equal to the dimension of
 *           src. An illegal value will cause a NULL
 *           return value.
 * 
 * numElem   this specifies how many elements from src
 *           should be taken in each dimension. A zero
 *           entry indicates that the dimension should
 *           not appear in the new array.  This
 *           argument should be an array with an entry
 *           for each dimension of src.  Passing NULL
 *           here will cause NULL to be returned.  If
 *           srcStart[i] + numElem[i]*srcStride[i] is
 *           greater than upper[i] for src or if
 *           srcStart[i] + numElem[i]*srcStride[i] is
 *           less than lower[i] for src, NULL will be
 *           returned.
 * 
 * srcStart  this array holds the coordinates of the
 *           first element of the new array. If this
 *           argument is NULL, the first element of src
 *           will be the first element of the new
 *           array. If non-NULL, this argument should
 *           be an array with an entry for each
 *           dimension of src.  If srcStart[i] is less
 *           than lower[i] for the array src, NULL will
 *           be returned.
 * 
 * srcStride this array lets you specify the stride
 *           between elements in each dimension of
 *           src. This stride is relative to the
 *           coordinate system of the src array. If
 *           this argument is NULL, the stride is taken
 *           to be one in each dimension.  If non-NULL,
 *           this argument should be an array with an
 *           entry for each dimension of src.
 * 
 * newLower  this argument is like lower in a create
 *           method. It sets the coordinates for the
 *           first element in the new array.  If this
 *           argument is NULL, the values indicated by
 *           srcStart will be used. If non-NULL, this
 *           should be an array with dimen elements.
 */
struct sidlx_rmi_Simsponse__array*
sidlx_rmi_Simsponse__array_slice(
  struct sidlx_rmi_Simsponse__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

/**
 * Copy the contents of one array (src) to a second array
 * (dest). For the copy to take place, both arrays must
 * exist and be of the same dimension. This method will
 * not modify dest's size, index bounds, or stride; only
 * the array element values of dest may be changed by
 * this function. No part of src is ever changed by copy.
 * 
 * On exit, dest[i][j][k]... = src[i][j][k]... for all
 * indices i,j,k...  that are in both arrays. If dest and
 * src have no indices in common, nothing is copied. For
 * example, if src is a 1-d array with elements 0-5 and
 * dest is a 1-d array with elements 2-3, this function
 * will make the following assignments:
 *   dest[2] = src[2],
 *   dest[3] = src[3].
 * The function copied the elements that both arrays have
 * in common.  If dest had elements 4-10, this function
 * will make the following assignments:
 *   dest[4] = src[4],
 *   dest[5] = src[5].
 */
void
sidlx_rmi_Simsponse__array_copy(
  const struct sidlx_rmi_Simsponse__array* src,
  struct sidlx_rmi_Simsponse__array* dest);

/**
 * If necessary, convert a general matrix into a matrix
 * with the required properties. This checks the
 * dimension and ordering of the matrix.  If both these
 * match, it simply returns a new reference to the
 * existing matrix. If the dimension of the incoming
 * array doesn't match, it returns NULL. If the ordering
 * of the incoming array doesn't match the specification,
 * a new array is created with the desired ordering and
 * the content of the incoming array is copied to the new
 * array.
 * 
 * The ordering parameter should be one of the constants
 * defined in enum sidl_array_ordering
 * (e.g. sidl_general_order, sidl_column_major_order, or
 * sidl_row_major_order). If you specify
 * sidl_general_order, this routine will only check the
 * dimension because any matrix is sidl_general_order.
 * 
 * The caller assumes ownership of the returned reference
 * unless it's NULL.
 */
struct sidlx_rmi_Simsponse__array*
sidlx_rmi_Simsponse__array_ensure(
  struct sidlx_rmi_Simsponse__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak sidlx_rmi_Simsponse__connectI

#pragma weak sidlx_rmi_Simsponse__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct sidlx_rmi_Simsponse__object*
sidlx_rmi_Simsponse__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct sidlx_rmi_Simsponse__object*
sidlx_rmi_Simsponse__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
