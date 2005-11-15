/*
 * File:          bHYPRE_StructStencil_Stub.c
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "bHYPRE_StructStencil.h"
#include "bHYPRE_StructStencil_IOR.h"
#ifndef included_sidl_interface_IOR_h
#include "sidl_interface_IOR.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
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

/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

/*
 * Hold pointer to IOR functions.
 */

static const struct bHYPRE_StructStencil__external *_externals = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct bHYPRE_StructStencil__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _externals = bHYPRE_StructStencil__externals();
#else
  _externals = (struct 
    bHYPRE_StructStencil__external*)sidl_dynamicLoadIOR("bHYPRE.StructStencil",
    "bHYPRE_StructStencil__externals") ;
#endif
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())

/*
 * Hold pointer to static entry point vector
 */

static const struct bHYPRE_StructStencil__sepv *_sepv = NULL;
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

bHYPRE_StructStencil
bHYPRE_StructStencil__create()
{
  return (*(_getExternals()->createObject))();
}

static bHYPRE_StructStencil bHYPRE_StructStencil__remote(const char* url,
  sidl_BaseInterface *_ex);
/*
 * RMI constructor function for the class.
 */

bHYPRE_StructStencil
bHYPRE_StructStencil__createRemote(const char* url, sidl_BaseInterface *_ex)
{
  return bHYPRE_StructStencil__remote(url, _ex);
}

static struct bHYPRE_StructStencil__object* 
  bHYPRE_StructStencil__remoteConnect(const char* url, sidl_BaseInterface *_ex);
static struct bHYPRE_StructStencil__object* 
  bHYPRE_StructStencil__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

bHYPRE_StructStencil
bHYPRE_StructStencil__connect(const char* url, sidl_BaseInterface *_ex)
{
  return bHYPRE_StructStencil__remoteConnect(url, _ex);
}

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

void
bHYPRE_StructStencil_addRef(
  /* in */ bHYPRE_StructStencil self)
{
  (*self->d_epv->f_addRef)(
    self);
}

/*
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>sidl</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */

void
bHYPRE_StructStencil_deleteRef(
  /* in */ bHYPRE_StructStencil self)
{
  (*self->d_epv->f_deleteRef)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
bHYPRE_StructStencil_isSame(
  /* in */ bHYPRE_StructStencil self,
  /* in */ sidl_BaseInterface iobj)
{
  return (*self->d_epv->f_isSame)(
    self,
    iobj);
}

/*
 * Check whether the object can support the specified interface or
 * class.  If the <code>sidl</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteRef</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */

sidl_BaseInterface
bHYPRE_StructStencil_queryInt(
  /* in */ bHYPRE_StructStencil self,
  /* in */ const char* name)
{
  return (*self->d_epv->f_queryInt)(
    self,
    name);
}

/*
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>sidl</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */

sidl_bool
bHYPRE_StructStencil_isType(
  /* in */ bHYPRE_StructStencil self,
  /* in */ const char* name)
{
  return (*self->d_epv->f_isType)(
    self,
    name);
}

/*
 * Return the meta-data about the class implementing this interface.
 */

sidl_ClassInfo
bHYPRE_StructStencil_getClassInfo(
  /* in */ bHYPRE_StructStencil self)
{
  return (*self->d_epv->f_getClassInfo)(
    self);
}

/*
 * Method:  Create[]
 */

bHYPRE_StructStencil
bHYPRE_StructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size)
{
  return (_getSEPV()->f_Create)(
    ndim,
    size);
}

/*
 * Method:  SetDimension[]
 */

int32_t
bHYPRE_StructStencil_SetDimension(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t dim)
{
  return (*self->d_epv->f_SetDimension)(
    self,
    dim);
}

/*
 * Method:  SetSize[]
 */

int32_t
bHYPRE_StructStencil_SetSize(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t size)
{
  return (*self->d_epv->f_SetSize)(
    self,
    size);
}

/*
 * Method:  SetElement[]
 */

int32_t
bHYPRE_StructStencil_SetElement(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t index,
  /* in */ int32_t* offset,
  /* in */ int32_t dim)
{
  int32_t offset_lower[1], offset_upper[1], offset_stride[1]; 
  struct sidl_int__array offset_real;
  struct sidl_int__array*offset_tmp = &offset_real;
  offset_upper[0] = dim-1;
  sidl_int__array_init(offset, offset_tmp, 1, offset_lower, offset_upper,
    offset_stride);
  return (*self->d_epv->f_SetElement)(
    self,
    index,
    offset_tmp);
}

void
bHYPRE_StructStencil_Create__sexec(
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t ndim;
  int32_t size;
  bHYPRE_StructStencil _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;

  /* unpack in and inout argments */

  sidl_io_Deserializer_unpackInt( inArgs, "ndim", &ndim, _ex2);

  sidl_io_Deserializer_unpackInt( inArgs, "size", &size, _ex2);

  /* make the call */
  _retval = (_getSEPV()->f_Create)(
    ndim,
    size);

  /* pack return value */
  /* pack out and inout argments */

}

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_StructStencil
bHYPRE_StructStencil__cast(
  void* obj)
{
  bHYPRE_StructStencil cast = NULL;

  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.StructStencil",
      (void*)bHYPRE_StructStencil__IHConnect);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_StructStencil) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.StructStencil");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_StructStencil__cast2(
  void* obj,
  const char* type)
{
  void* cast = NULL;

  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (*base->d_epv->f__cast)(base->d_object, type);
  }

  return cast;
}
/*
 * Select and execute a method by name
 */

void
bHYPRE_StructStencil__exec(
  /* in */ bHYPRE_StructStencil self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs)
{
  (*self->d_epv->f__exec)(
  self,
  methodName,
  inArgs,
  outArgs);
}

struct bHYPRE_StructStencil__smethod {
  const char *d_name;
  void (*d_func)(struct sidl_io_Deserializer__object *,
    struct sidl_io_Serializer__object *);
};

void
bHYPRE_StructStencil__sexec(
        const char* methodName,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs ) { 
  static const struct bHYPRE_StructStencil__smethod s_methods[] = {
    { "Create", bHYPRE_StructStencil_Create__sexec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct bHYPRE_StructStencil__smethod);
  if (methodName) {
    /* Use binary search to locate method */
    while (l < u) {
      i = (l + u) >> 1;
      if (!(cmp=strcmp(methodName, s_methods[i].d_name))) {
        (s_methods[i].d_func)(inArgs, outArgs);
        return;
      }
      else if (cmp < 0) u = i;
      else l = i + 1;
    }
  }
  /* TODO: add code for method not found */
}
/*
 * Get the URL of the Implementation of this object (for RMI)
 */

char*
bHYPRE_StructStencil__getURL(
  /* in */ bHYPRE_StructStencil self)
{
  return (*self->d_epv->f__getURL)(
  self);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_StructStencil__array*)sidl_interface__array_createCol(dimen, lower,
    upper);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_StructStencil__array*)sidl_interface__array_createRow(dimen, lower,
    upper);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create1d(int32_t len)
{
  return (struct 
    bHYPRE_StructStencil__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create1dInit(
  int32_t len, 
  bHYPRE_StructStencil* data)
{
  return (struct 
    bHYPRE_StructStencil__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_StructStencil__array*)sidl_interface__array_create2dCol(m, n);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_StructStencil__array*)sidl_interface__array_create2dRow(m, n);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_borrow(
  bHYPRE_StructStencil* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct bHYPRE_StructStencil__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_smartCopy(
  struct bHYPRE_StructStencil__array *array)
{
  return (struct bHYPRE_StructStencil__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_StructStencil__array_addRef(
  struct bHYPRE_StructStencil__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_StructStencil__array_deleteRef(
  struct bHYPRE_StructStencil__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get1(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get2(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get3(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get4(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get5(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get6(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get7(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_StructStencil
bHYPRE_StructStencil__array_get(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t indices[])
{
  return (bHYPRE_StructStencil)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_StructStencil__array_set1(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set2(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set3(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set4(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set5(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set6(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set7(
  struct bHYPRE_StructStencil__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructStencil__array_set(
  struct bHYPRE_StructStencil__array* array,
  const int32_t indices[],
  bHYPRE_StructStencil const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_StructStencil__array_dimen(
  const struct bHYPRE_StructStencil__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_StructStencil__array_lower(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_StructStencil__array_upper(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_StructStencil__array_length(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_StructStencil__array_stride(
  const struct bHYPRE_StructStencil__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_StructStencil__array_isColumnOrder(
  const struct bHYPRE_StructStencil__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_StructStencil__array_isRowOrder(
  const struct bHYPRE_StructStencil__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_StructStencil__array_copy(
  const struct bHYPRE_StructStencil__array* src,
  struct bHYPRE_StructStencil__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_slice(
  struct bHYPRE_StructStencil__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_StructStencil__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_StructStencil__array*
bHYPRE_StructStencil__array_ensure(
  struct bHYPRE_StructStencil__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_StructStencil__array*)
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
#include "sidl_rmi_ProtocolFactory.h"
#include "sidl_rmi_Invocation.h"
#include "sidl_rmi_Response.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t bHYPRE_StructStencil__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE_StructStencil__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE_StructStencil__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE_StructStencil__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/* Static variables to hold version of IOR */
static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 9;

/* Static variables for managing EPV initialization. */
static int s_remote_initialized = 0;

static struct bHYPRE_StructStencil__epv s_rem_epv__bhypre_structstencil;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;

/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_bHYPRE_StructStencil__cast(
struct bHYPRE_StructStencil__object* self,
const char* name)
{
  void* cast = NULL;

  struct bHYPRE_StructStencil__object* s0;
  struct sidl_BaseClass__object* s1;
   s0 =                               self;
   s1 =                               &s0->d_sidl_baseclass;

  if (!strcmp(name, "bHYPRE.StructStencil")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "sidl.BaseClass")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s1->d_sidl_baseinterface;
  }
  else if(bHYPRE_StructStencil_isType(self, name)) {
    void* (*func)(sidl_rmi_InstanceHandle) = 
      (void* (*)(sidl_rmi_InstanceHandle)) 
      sidl_rmi_ConnectRegistry_getConnect(name);
    cast =  (*func)((sidl_rmi_InstanceHandle)self->d_data);
  }

  return cast;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE_StructStencil__delete(
  struct bHYPRE_StructStencil__object* self)
{
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_bHYPRE_StructStencil__getURL(
  struct bHYPRE_StructStencil__object* self)
{
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getURL(conn, &_ex);
  }
  return NULL;
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_bHYPRE_StructStencil__exec(
  struct bHYPRE_StructStencil__object* self,
  const char* methodName,
  sidl_io_Deserializer inArgs,
  sidl_io_Serializer outArgs)
{
}

/* REMOTE METHOD STUB:addRef */
static void
remote_bHYPRE_StructStencil_addRef(
  /* in */ struct bHYPRE_StructStencil__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_bHYPRE_StructStencil_deleteRef(
  /* in */ struct bHYPRE_StructStencil__object* self /* TLD */)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "deleteRef", _ex2 );
  sidl_rmi_Response _rsvp = NULL;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
}

/* REMOTE METHOD STUB:isSame */
static sidl_bool
remote_bHYPRE_StructStencil_isSame(
  /* in */ struct bHYPRE_StructStencil__object* self /* TLD */,
  /* in */ struct sidl_BaseInterface__object* iobj)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:queryInt */
static struct sidl_BaseInterface__object*
remote_bHYPRE_StructStencil_queryInt(
  /* in */ struct bHYPRE_StructStencil__object* self /* TLD */,
  /* in */ const char* name)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_bHYPRE_StructStencil_isType(
  /* in */ struct bHYPRE_StructStencil__object* self /* TLD */,
  /* in */ const char* name)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "isType", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  sidl_bool _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:getClassInfo */
static struct sidl_ClassInfo__object*
remote_bHYPRE_StructStencil_getClassInfo(
  /* in */ struct bHYPRE_StructStencil__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:SetDimension */
static int32_t
remote_bHYPRE_StructStencil_SetDimension(
  /* in */ struct bHYPRE_StructStencil__object* self /* TLD */,
  /* in */ int32_t dim)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetDimension", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "dim", dim, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetSize */
static int32_t
remote_bHYPRE_StructStencil_SetSize(
  /* in */ struct bHYPRE_StructStencil__object* self /* TLD */,
  /* in */ int32_t size)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetSize", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "size", size, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetElement */
static int32_t
remote_bHYPRE_StructStencil_SetElement(
  /* in */ struct bHYPRE_StructStencil__object* self /* TLD */,
  /* in */ int32_t index,
  /* in */ struct sidl_int__array* offset)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetElement", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "index", index, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void bHYPRE_StructStencil__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE_StructStencil__epv* epv = &s_rem_epv__bhypre_structstencil;
  struct sidl_BaseClass__epv*       e0  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*   e1  = &s_rem_epv__sidl_baseinterface;

  epv->f__cast             = remote_bHYPRE_StructStencil__cast;
  epv->f__delete           = remote_bHYPRE_StructStencil__delete;
  epv->f__exec             = remote_bHYPRE_StructStencil__exec;
  epv->f__getURL           = remote_bHYPRE_StructStencil__getURL;
  epv->f__ctor             = NULL;
  epv->f__dtor             = NULL;
  epv->f_addRef            = remote_bHYPRE_StructStencil_addRef;
  epv->f_deleteRef         = remote_bHYPRE_StructStencil_deleteRef;
  epv->f_isSame            = remote_bHYPRE_StructStencil_isSame;
  epv->f_queryInt          = remote_bHYPRE_StructStencil_queryInt;
  epv->f_isType            = remote_bHYPRE_StructStencil_isType;
  epv->f_getClassInfo      = remote_bHYPRE_StructStencil_getClassInfo;
  epv->f_SetDimension      = remote_bHYPRE_StructStencil_SetDimension;
  epv->f_SetSize           = remote_bHYPRE_StructStencil_SetSize;
  epv->f_SetElement        = remote_bHYPRE_StructStencil_SetElement;

  e0->f__cast        = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete      = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e0->f__exec        = (void (*)(struct sidl_BaseClass__object*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e0->f_addRef       = (void (*)(struct sidl_BaseClass__object*)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt     = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete      = (void (*)(void*)) epv->f__delete;
  e1->f__exec        = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e1->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__remoteConnect(const char *url, sidl_BaseInterface *_ex)
{
  struct bHYPRE_StructStencil__object* self;

  struct bHYPRE_StructStencil__object* s0;
  struct sidl_BaseClass__object* s1;

  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_connectInstance(url, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_StructStencil__object*) malloc(
      sizeof(struct bHYPRE_StructStencil__object));

   s0 =                               self;
   s1 =                               &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_StructStencil__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__bhypre_structstencil;

  self->d_data = (void*) instance;
  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_StructStencil__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;


  return self;
}
/* Create an instance that uses an already existing InstanceHandel to connect 
  to an existing remote object. */
static struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_StructStencil__object* self;

  struct bHYPRE_StructStencil__object* s0;
  struct sidl_BaseClass__object* s1;

  self =
    (struct bHYPRE_StructStencil__object*) malloc(
      sizeof(struct bHYPRE_StructStencil__object));

   s0 =                               self;
   s1 =                               &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_StructStencil__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__bhypre_structstencil;

  self->d_data = (void*) instance;

  sidl_rmi_InstanceHandle_addRef(instance);
  return self;
}
/* REMOTE: generate remote instance given URL string. */
static struct bHYPRE_StructStencil__object*
bHYPRE_StructStencil__remote(const char *url, sidl_BaseInterface *_ex)
{
  struct bHYPRE_StructStencil__object* self;

  struct bHYPRE_StructStencil__object* s0;
  struct sidl_BaseClass__object* s1;

  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_createInstance(url, "bHYPRE.StructStencil", _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_StructStencil__object*) malloc(
      sizeof(struct bHYPRE_StructStencil__object));

   s0 =                               self;
   s1 =                               &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_StructStencil__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__bhypre_structstencil;

  self->d_data = (void*) instance;

  return self;
}
