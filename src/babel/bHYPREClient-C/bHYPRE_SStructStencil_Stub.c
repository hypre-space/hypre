/*
 * File:          bHYPRE_SStructStencil_Stub.c
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_SStructStencil.h"
#include "bHYPRE_SStructStencil_IOR.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#include "sidl_Exception.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#include <stddef.h>
#include <string.h>
#include "sidl_BaseInterface_IOR.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "sidl_Loader.h"
#endif

#define LANG_SPECIFIC_INIT()
/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

/*
 * Hold pointer to IOR functions.
 */

static const struct bHYPRE_SStructStencil__external *_externals = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct bHYPRE_SStructStencil__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _externals = bHYPRE_SStructStencil__externals();
#else
  _externals = (struct 
    bHYPRE_SStructStencil__external*)sidl_dynamicLoadIOR(
    "bHYPRE.SStructStencil","bHYPRE_SStructStencil__externals") ;
  sidl_checkIORVersion("bHYPRE.SStructStencil", _externals->d_ior_major_version,
    _externals->d_ior_minor_version, 0, 10);
#endif
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())

/*
 * Hold pointer to static entry point vector
 */

static const struct bHYPRE_SStructStencil__sepv *_sepv = NULL;
/*
 * Return pointer to static functions.
 */

#define _getSEPV() (_sepv ? _sepv : (_sepv = (*(_getExternals()->getStaticEPV))()))
/*
 * Reset point to static functions.
 */

#define _resetSEPV() (_sepv = (*(_getExternals()->getStaticEPV))())

/*
 * Constructor function for the class.
 */

bHYPRE_SStructStencil
bHYPRE_SStructStencil__create(sidl_BaseInterface* _ex)
{
  return (*(_getExternals()->createObject))(NULL,_ex);
}

/**
 * Wraps up the private data struct pointer (struct bHYPRE_SStructStencil__data) passed in rather than running the constructor.
 */
bHYPRE_SStructStencil
bHYPRE_SStructStencil__wrapObj(void* data, sidl_BaseInterface* _ex)
{
  return (*(_getExternals()->createObject))(data,_ex);
}

static bHYPRE_SStructStencil bHYPRE_SStructStencil__remoteCreate(const char* 
  url, sidl_BaseInterface *_ex);
/*
 * RMI constructor function for the class.
 */

bHYPRE_SStructStencil
bHYPRE_SStructStencil__createRemote(const char* url, sidl_BaseInterface *_ex)
{
  return bHYPRE_SStructStencil__remoteCreate(url, _ex);
}

static struct bHYPRE_SStructStencil__object* 
  bHYPRE_SStructStencil__remoteConnect(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
static struct bHYPRE_SStructStencil__object* 
  bHYPRE_SStructStencil__IHConnect(struct sidl_rmi_InstanceHandle__object* 
  instance, sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

bHYPRE_SStructStencil
bHYPRE_SStructStencil__connect(const char* url, sidl_BaseInterface *_ex)
{
  return bHYPRE_SStructStencil__remoteConnect(url, TRUE, _ex);
}

/*
 *  This function is the preferred way to create a SStruct Stencil. 
 */

bHYPRE_SStructStencil
bHYPRE_SStructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex)
{
  return (_getSEPV()->f_Create)(
    ndim,
    size,
    _ex);
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

SIDL_C_INLINE_DEFN
void
bHYPRE_SStructStencil_Destroy(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f_Destroy)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Set the number of spatial dimensions and stencil entries.
 * DEPRECATED, use Create:
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructStencil_SetNumDimSize(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t ndim,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_SetNumDimSize)(
    self,
    ndim,
    size,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Set a stencil entry.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructStencil_SetEntry(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t entry,
  /* in rarray[dim] */ int32_t* offset,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  int32_t offset_lower[1], offset_upper[1], offset_stride[1]; 
  struct sidl_int__array offset_real;
  struct sidl_int__array*offset_tmp = &offset_real;
  offset_upper[0] = dim-1;
  sidl_int__array_init(offset, offset_tmp, 1, offset_lower, offset_upper,
    offset_stride);
  return (*self->d_epv->f_SetEntry)(
    self,
    entry,
    offset_tmp,
    var,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
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

SIDL_C_INLINE_DEFN
void
bHYPRE_SStructStencil_addRef(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f_addRef)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

SIDL_C_INLINE_DEFN
void
bHYPRE_SStructStencil_deleteRef(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f_deleteRef)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

SIDL_C_INLINE_DEFN
sidl_bool
bHYPRE_SStructStencil_isSame(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

SIDL_C_INLINE_DEFN
sidl_bool
bHYPRE_SStructStencil_isType(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_isType)(
    self,
    name,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Return the meta-data about the class implementing this interface.
 */

SIDL_C_INLINE_DEFN
sidl_ClassInfo
bHYPRE_SStructStencil_getClassInfo(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_getClassInfo)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_SStructStencil
bHYPRE_SStructStencil__cast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  bHYPRE_SStructStencil cast = NULL;

  if(!connect_loaded) {
    connect_loaded = 1;
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.SStructStencil",
      (void*)bHYPRE_SStructStencil__IHConnect,_ex);SIDL_CHECK(*_ex);
  }
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_SStructStencil) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.SStructStencil", _ex); SIDL_CHECK(*_ex);
  }

  EXIT:
  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_SStructStencil__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface* _ex)
{
  void* cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (*base->d_epv->f__cast)(base->d_object, type, _ex); SIDL_CHECK(*_ex);
  }

  EXIT:
  return cast;
}
/*
 * Select and execute a method by name
 */

SIDL_C_INLINE_DEFN
void
bHYPRE_SStructStencil__exec(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ const char* methodName,
  /* in */ sidl_rmi_Call inArgs,
  /* in */ sidl_rmi_Return outArgs,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f__exec)(
    self,
    methodName,
    inArgs,
    outArgs,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */


/*
 * Get the URL of the Implementation of this object (for RMI)
 */

SIDL_C_INLINE_DEFN
char*
bHYPRE_SStructStencil__getURL(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f__getURL)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */


/*
 * On a remote object, addrefs the remote instance.
 */

SIDL_C_INLINE_DEFN
void
bHYPRE_SStructStencil__raddRef(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f__raddRef)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */


/*
 * Method to enable/disable static hooks execution.
 */

void
bHYPRE_SStructStencil__set_hooks_static(
  int32_t on,
  struct sidl_BaseInterface__object **_ex)
{
  (_getSEPV()->f__set_hooks_static)(
  on, _ex);
  _resetSEPV();
}

/*
 * Method to set whether or not method hooks should be invoked.
 */

SIDL_C_INLINE_DEFN
void
bHYPRE_SStructStencil__set_hooks(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ sidl_bool on,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f__set_hooks)(
    self,
    on,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * TRUE if this object is remote, false if local
 */

SIDL_C_INLINE_DEFN
sidl_bool
bHYPRE_SStructStencil__isRemote(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f__isRemote)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */


/*
 * TRUE if this object is remote, false if local
 */

sidl_bool
bHYPRE_SStructStencil__isLocal(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex)
{
  return !bHYPRE_SStructStencil__isRemote(self,_ex);
}

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructStencil__array*)sidl_interface__array_createCol(dimen, lower,
    upper);
}

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructStencil__array*)sidl_interface__array_createRow(dimen, lower,
    upper);
}

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_create1d(int32_t len)
{
  return (struct 
    bHYPRE_SStructStencil__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructStencil* data)
{
  return (struct 
    bHYPRE_SStructStencil__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_SStructStencil__array*)sidl_interface__array_create2dCol(m, n);
}

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_SStructStencil__array*)sidl_interface__array_create2dRow(m, n);
}

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_borrow(
  bHYPRE_SStructStencil* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct bHYPRE_SStructStencil__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_smartCopy(
  struct bHYPRE_SStructStencil__array *array)
{
  return (struct bHYPRE_SStructStencil__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_SStructStencil__array_addRef(
  struct bHYPRE_SStructStencil__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_SStructStencil__array_deleteRef(
  struct bHYPRE_SStructStencil__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get1(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1)
{
  return (bHYPRE_SStructStencil)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get2(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_SStructStencil)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get3(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_SStructStencil)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get4(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_SStructStencil)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get5(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_SStructStencil)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get6(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_SStructStencil)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get7(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_SStructStencil)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_SStructStencil
bHYPRE_SStructStencil__array_get(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t indices[])
{
  return (bHYPRE_SStructStencil)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_SStructStencil__array_set1(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  bHYPRE_SStructStencil const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructStencil__array_set2(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructStencil const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructStencil__array_set3(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructStencil const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructStencil__array_set4(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructStencil const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructStencil__array_set5(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructStencil const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructStencil__array_set6(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructStencil const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructStencil__array_set7(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructStencil const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructStencil__array_set(
  struct bHYPRE_SStructStencil__array* array,
  const int32_t indices[],
  bHYPRE_SStructStencil const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_SStructStencil__array_dimen(
  const struct bHYPRE_SStructStencil__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_SStructStencil__array_lower(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructStencil__array_upper(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructStencil__array_length(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructStencil__array_stride(
  const struct bHYPRE_SStructStencil__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_SStructStencil__array_isColumnOrder(
  const struct bHYPRE_SStructStencil__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_SStructStencil__array_isRowOrder(
  const struct bHYPRE_SStructStencil__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_SStructStencil__array_copy(
  const struct bHYPRE_SStructStencil__array* src,
  struct bHYPRE_SStructStencil__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_slice(
  struct bHYPRE_SStructStencil__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_SStructStencil__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_SStructStencil__array*
bHYPRE_SStructStencil__array_ensure(
  struct bHYPRE_SStructStencil__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_SStructStencil__array*)
    sidl_interface__array_ensure((struct sidl_interface__array *)src, dimen,
      ordering);
}

#include <stdlib.h>
#include <string.h>
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_ProtocolFactory_h
#include "sidl_rmi_ProtocolFactory.h"
#endif
#ifndef included_sidl_rmi_InstanceRegistry_h
#include "sidl_rmi_InstanceRegistry.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_Invocation_h
#include "sidl_rmi_Invocation.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidl_rmi_ServerRegistry_h
#include "sidl_rmi_ServerRegistry.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#include "sidl_Exception.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t bHYPRE_SStructStencil__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE_SStructStencil__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE_SStructStencil__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE_SStructStencil__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/* Static variables to hold version of IOR */
static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 10;

/* Static variables for managing EPV initialization. */
static int s_remote_initialized = 0;

static struct bHYPRE_SStructStencil__epv s_rem_epv__bhypre_sstructstencil;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_bHYPRE_SStructStencil__cast(
  struct bHYPRE_SStructStencil__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidl.BaseClass");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = self;
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "bHYPRE.SStructStencil");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = self;
      return cast;
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.BaseInterface");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
      return cast;
    }
  }
  if ((*self->d_epv->f_isType)(self,name, _ex)) {
    void* (*func)(struct sidl_rmi_InstanceHandle__object*,
      struct sidl_BaseInterface__object**) = 
      (void* (*)(struct sidl_rmi_InstanceHandle__object*,
        struct sidl_BaseInterface__object**)) 
      sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
    cast =  (*func)(((struct bHYPRE_SStructStencil__remote*)self->d_data)->d_ih,
      _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE_SStructStencil__delete(
  struct bHYPRE_SStructStencil__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_bHYPRE_SStructStencil__getURL(
  struct bHYPRE_SStructStencil__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    bHYPRE_SStructStencil__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_bHYPRE_SStructStencil__raddRef(
  struct bHYPRE_SStructStencil__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    bHYPRE_SStructStencil__remote*)self->d_data)->d_ih;
  sidl_rmi_Response _rsvp = NULL;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "addRef", _ex ); SIDL_CHECK(*_ex);
  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex);SIDL_CHECK(*_ex);
  /* Check for exceptions */
  netex = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);
  if(netex != NULL) {
    sidl_BaseInterface throwaway_exception = NULL;
    *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(netex,
      &throwaway_exception);
    return;
  }

  /* cleanup and return */
  EXIT:
  if(_inv) { sidl_rmi_Invocation_deleteRef(_inv,&_throwaway); }
  if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp,&_throwaway); }
  return;
}

/* REMOTE ISREMOTE: returns true if this object is Remote (it is). */
static sidl_bool
remote_bHYPRE_SStructStencil__isRemote(
    struct bHYPRE_SStructStencil__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_bHYPRE_SStructStencil__set_hooks(
  /* in */ struct bHYPRE_SStructStencil__object* self ,
  /* in */ sidl_bool on,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_SStructStencil__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructStencil._set_hooks.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_bHYPRE_SStructStencil__exec(
  struct bHYPRE_SStructStencil__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:Destroy */
static void
remote_bHYPRE_SStructStencil_Destroy(
  /* in */ struct bHYPRE_SStructStencil__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_SStructStencil__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Destroy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructStencil.Destroy.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:SetNumDimSize */
static int32_t
remote_bHYPRE_SStructStencil_SetNumDimSize(
  /* in */ struct bHYPRE_SStructStencil__object* self ,
  /* in */ int32_t ndim,
  /* in */ int32_t size,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_SStructStencil__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetNumDimSize", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "ndim", ndim, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "size", size, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructStencil.SetNumDimSize.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetEntry */
static int32_t
remote_bHYPRE_SStructStencil_SetEntry(
  /* in */ struct bHYPRE_SStructStencil__object* self ,
  /* in */ int32_t entry,
  /* in rarray[dim] */ struct sidl_int__array* offset,
  /* in */ int32_t var,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_SStructStencil__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetEntry", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "entry", entry, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "offset", offset,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructStencil.SetEntry.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:addRef */
static void
remote_bHYPRE_SStructStencil_addRef(
  /* in */ struct bHYPRE_SStructStencil__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_SStructStencil__remote* r_obj = (struct 
      bHYPRE_SStructStencil__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_bHYPRE_SStructStencil_deleteRef(
  /* in */ struct bHYPRE_SStructStencil__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_SStructStencil__remote* r_obj = (struct 
      bHYPRE_SStructStencil__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount--;
    if(r_obj->d_refcount == 0) {
      sidl_rmi_InstanceHandle_deleteRef(r_obj->d_ih, _ex);
      free(r_obj);
      free(self);
    }
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:isSame */
static sidl_bool
remote_bHYPRE_SStructStencil_isSame(
  /* in */ struct bHYPRE_SStructStencil__object* self ,
  /* in */ struct sidl_BaseInterface__object* iobj,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_SStructStencil__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isSame", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(iobj){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)iobj,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "iobj", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "iobj", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructStencil.isSame.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_bHYPRE_SStructStencil_isType(
  /* in */ struct bHYPRE_SStructStencil__object* self ,
  /* in */ const char* name,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_SStructStencil__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructStencil.isType.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:getClassInfo */
static struct sidl_ClassInfo__object*
remote_bHYPRE_SStructStencil_getClassInfo(
  /* in */ struct bHYPRE_SStructStencil__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char*_retval_str = NULL;
    struct sidl_ClassInfo__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_SStructStencil__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructStencil.getClassInfo.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_ClassInfo__connectI(_retval_str, FALSE,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void bHYPRE_SStructStencil__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE_SStructStencil__epv* epv = &s_rem_epv__bhypre_sstructstencil;
  struct sidl_BaseClass__epv*        e0  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*    e1  = &s_rem_epv__sidl_baseinterface;

  epv->f__cast              = remote_bHYPRE_SStructStencil__cast;
  epv->f__delete            = remote_bHYPRE_SStructStencil__delete;
  epv->f__exec              = remote_bHYPRE_SStructStencil__exec;
  epv->f__getURL            = remote_bHYPRE_SStructStencil__getURL;
  epv->f__raddRef           = remote_bHYPRE_SStructStencil__raddRef;
  epv->f__isRemote          = remote_bHYPRE_SStructStencil__isRemote;
  epv->f__set_hooks         = remote_bHYPRE_SStructStencil__set_hooks;
  epv->f__ctor              = NULL;
  epv->f__ctor2             = NULL;
  epv->f__dtor              = NULL;
  epv->f_Destroy            = remote_bHYPRE_SStructStencil_Destroy;
  epv->f_SetNumDimSize      = remote_bHYPRE_SStructStencil_SetNumDimSize;
  epv->f_SetEntry           = remote_bHYPRE_SStructStencil_SetEntry;
  epv->f_addRef             = remote_bHYPRE_SStructStencil_addRef;
  epv->f_deleteRef          = remote_bHYPRE_SStructStencil_deleteRef;
  epv->f_isSame             = remote_bHYPRE_SStructStencil_isSame;
  epv->f_isType             = remote_bHYPRE_SStructStencil_isType;
  epv->f_getClassInfo       = remote_bHYPRE_SStructStencil_getClassInfo;

  e0->f__cast        = (void* (*)(struct sidl_BaseClass__object*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__delete;
  e0->f__getURL      = (char* (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__getURL;
  e0->f__raddRef     = (void (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e0->f__isRemote    = (sidl_bool (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e0->f__set_hooks   = (void (*)(struct sidl_BaseClass__object*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e0->f__exec        = (void (*)(struct sidl_BaseClass__object*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_addRef       = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e0->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e1->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e1->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e1->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e1->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e1->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e1->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e1->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e1->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_SStructStencil__object* self;

  struct bHYPRE_SStructStencil__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_SStructStencil__remote* r_obj;
  sidl_rmi_InstanceHandle instance = NULL;
  char* objectID = NULL;
  objectID = NULL;
  *_ex = NULL;
  if(url == NULL) {return NULL;}
  objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
  if(objectID) {
    sidl_BaseInterface bi = 
      (sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
      objectID, _ex); SIDL_CHECK(*_ex);
    return bHYPRE_SStructStencil__rmicast(bi,_ex);SIDL_CHECK(*_ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar,
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_SStructStencil__object*) malloc(
      sizeof(struct bHYPRE_SStructStencil__object));

  r_obj =
    (struct bHYPRE_SStructStencil__remote*) malloc(
      sizeof(struct bHYPRE_SStructStencil__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                 self;
  s1 =                                 &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructStencil__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_sstructstencil;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  return NULL;
}
/* Create an instance that uses an already existing  */
/* InstanceHandle to connect to an existing remote object. */
static struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_SStructStencil__object* self;

  struct bHYPRE_SStructStencil__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_SStructStencil__remote* r_obj;
  self =
    (struct bHYPRE_SStructStencil__object*) malloc(
      sizeof(struct bHYPRE_SStructStencil__object));

  r_obj =
    (struct bHYPRE_SStructStencil__remote*) malloc(
      sizeof(struct bHYPRE_SStructStencil__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                 self;
  s1 =                                 &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructStencil__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_sstructstencil;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}
/* REMOTE: generate remote instance given URL string. */
static struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__remoteCreate(const char *url, sidl_BaseInterface *_ex)
{
  sidl_BaseInterface _throwaway_exception = NULL;
  struct bHYPRE_SStructStencil__object* self;

  struct bHYPRE_SStructStencil__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_SStructStencil__remote* r_obj;
  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_createInstance(url, "bHYPRE.SStructStencil",
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_SStructStencil__object*) malloc(
      sizeof(struct bHYPRE_SStructStencil__object));

  r_obj =
    (struct bHYPRE_SStructStencil__remote*) malloc(
      sizeof(struct bHYPRE_SStructStencil__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                 self;
  s1 =                                 &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructStencil__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_sstructstencil;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  if(instance) { sidl_rmi_InstanceHandle_deleteRef(instance,
    &_throwaway_exception); }
  return NULL;
}
/*
 * Cast method for interface and class type conversions.
 */

struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct bHYPRE_SStructStencil__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.SStructStencil",
      (void*)bHYPRE_SStructStencil__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct bHYPRE_SStructStencil__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.SStructStencil", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct bHYPRE_SStructStencil__object*
bHYPRE_SStructStencil__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return bHYPRE_SStructStencil__remoteConnect(url, ar, _ex);
}

