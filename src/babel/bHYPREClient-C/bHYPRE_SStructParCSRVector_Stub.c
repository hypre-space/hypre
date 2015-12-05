/*
 * File:          bHYPRE_SStructParCSRVector_Stub.c
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_SStructParCSRVector.h"
#include "bHYPRE_SStructParCSRVector_IOR.h"
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

static const struct bHYPRE_SStructParCSRVector__external *_externals = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct bHYPRE_SStructParCSRVector__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _externals = bHYPRE_SStructParCSRVector__externals();
#else
  _externals = (struct 
    bHYPRE_SStructParCSRVector__external*)sidl_dynamicLoadIOR(
    "bHYPRE.SStructParCSRVector","bHYPRE_SStructParCSRVector__externals") ;
  sidl_checkIORVersion("bHYPRE.SStructParCSRVector",
    _externals->d_ior_major_version, _externals->d_ior_minor_version, 0, 10);
#endif
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())

/*
 * Hold pointer to static entry point vector
 */

static const struct bHYPRE_SStructParCSRVector__sepv *_sepv = NULL;
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

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__create(sidl_BaseInterface* _ex)
{
  return (*(_getExternals()->createObject))(NULL,_ex);
}

/**
 * Wraps up the private data struct pointer (struct bHYPRE_SStructParCSRVector__data) passed in rather than running the constructor.
 */
bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__wrapObj(void* data, sidl_BaseInterface* _ex)
{
  return (*(_getExternals()->createObject))(data,_ex);
}

static bHYPRE_SStructParCSRVector 
  bHYPRE_SStructParCSRVector__remoteCreate(const char* url,
  sidl_BaseInterface *_ex);
/*
 * RMI constructor function for the class.
 */

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__createRemote(const char* url,
  sidl_BaseInterface *_ex)
{
  return bHYPRE_SStructParCSRVector__remoteCreate(url, _ex);
}

static struct bHYPRE_SStructParCSRVector__object* 
  bHYPRE_SStructParCSRVector__remoteConnect(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
static struct bHYPRE_SStructParCSRVector__object* 
  bHYPRE_SStructParCSRVector__IHConnect(struct sidl_rmi_InstanceHandle__object* 
  instance, sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__connect(const char* url, sidl_BaseInterface *_ex)
{
  return bHYPRE_SStructParCSRVector__remoteConnect(url, TRUE, _ex);
}

/*
 *  This function is the preferred way to create a SStruct ParCSR Vector. 
 */

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex)
{
  return (_getSEPV()->f_Create)(
    mpi_comm,
    grid,
    _ex);
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

SIDL_C_INLINE_DEFN
void
bHYPRE_SStructParCSRVector_addRef(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_deleteRef(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_isSame(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_isType(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_getClassInfo(
  /* in */ bHYPRE_SStructParCSRVector self,
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
 * Set the vector grid.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_SetGrid(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_SetGrid)(
    self,
    grid,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_SetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  int32_t index_lower[1], index_upper[1], index_stride[1]; 
  struct sidl_int__array index_real;
  struct sidl_int__array*index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  return (*self->d_epv->f_SetValues)(
    self,
    part,
    index_tmp,
    var,
    value,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */

int32_t
bHYPRE_SStructParCSRVector_SetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1]; 
  struct sidl_int__array ilower_real;
  struct sidl_int__array*ilower_tmp = &ilower_real;
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1]; 
  struct sidl_int__array iupper_real;
  struct sidl_int__array*iupper_tmp = &iupper_real;
  int32_t values_lower[1], values_upper[1], values_stride[1]; 
  struct sidl_double__array values_real;
  struct sidl_double__array*values_tmp = &values_real;
  ilower_upper[0] = dim-1;
  sidl_int__array_init(ilower, ilower_tmp, 1, ilower_lower, ilower_upper,
    ilower_stride);
  iupper_upper[0] = dim-1;
  sidl_int__array_init(iupper, iupper_tmp, 1, iupper_lower, iupper_upper,
    iupper_stride);
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  return (*self->d_epv->f_SetBoxValues)(
    self,
    part,
    ilower_tmp,
    iupper_tmp,
    var,
    values_tmp,
    _ex);
}

/*
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_AddToValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  int32_t index_lower[1], index_upper[1], index_stride[1]; 
  struct sidl_int__array index_real;
  struct sidl_int__array*index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  return (*self->d_epv->f_AddToValues)(
    self,
    part,
    index_tmp,
    var,
    value,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */

int32_t
bHYPRE_SStructParCSRVector_AddToBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1]; 
  struct sidl_int__array ilower_real;
  struct sidl_int__array*ilower_tmp = &ilower_real;
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1]; 
  struct sidl_int__array iupper_real;
  struct sidl_int__array*iupper_tmp = &iupper_real;
  int32_t values_lower[1], values_upper[1], values_stride[1]; 
  struct sidl_double__array values_real;
  struct sidl_double__array*values_tmp = &values_real;
  ilower_upper[0] = dim-1;
  sidl_int__array_init(ilower, ilower_tmp, 1, ilower_lower, ilower_upper,
    ilower_stride);
  iupper_upper[0] = dim-1;
  sidl_int__array_init(iupper, iupper_tmp, 1, iupper_lower, iupper_upper,
    iupper_stride);
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  return (*self->d_epv->f_AddToBoxValues)(
    self,
    part,
    ilower_tmp,
    iupper_tmp,
    var,
    values_tmp,
    _ex);
}

/*
 * Gather vector data before calling {\tt GetValues}.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_Gather(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_Gather)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Get vector coefficients index by index.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_GetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  int32_t index_lower[1], index_upper[1], index_stride[1]; 
  struct sidl_int__array index_real;
  struct sidl_int__array*index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  return (*self->d_epv->f_GetValues)(
    self,
    part,
    index_tmp,
    var,
    value,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Get vector coefficients a box at a time.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */

int32_t
bHYPRE_SStructParCSRVector_GetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* inout rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1]; 
  struct sidl_int__array ilower_real;
  struct sidl_int__array*ilower_tmp = &ilower_real;
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1]; 
  struct sidl_int__array iupper_real;
  struct sidl_int__array*iupper_tmp = &iupper_real;
  int32_t values_lower[1], values_upper[1], values_stride[1]; 
  struct sidl_double__array values_real;
  struct sidl_double__array*values_tmp = &values_real;
  ilower_upper[0] = dim-1;
  sidl_int__array_init(ilower, ilower_tmp, 1, ilower_lower, ilower_upper,
    ilower_stride);
  iupper_upper[0] = dim-1;
  sidl_int__array_init(iupper, iupper_tmp, 1, iupper_lower, iupper_upper,
    iupper_stride);
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  return (*self->d_epv->f_GetBoxValues)(
    self,
    part,
    ilower_tmp,
    iupper_tmp,
    var,
    &values_tmp,
    _ex);
}

/*
 * Set the vector to be complex.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_SetComplex(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_SetComplex)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_Print(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ const char* filename,
  /* in */ int32_t all,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_Print)(
    self,
    filename,
    all,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * A semi-structured matrix or vector contains a Struct or IJ matrix
 * or vector.  GetObject returns it.
 * The returned type is a sidl.BaseInterface.
 * A cast must be used on the returned object to convert it into a known type.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_GetObject(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface* A,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_GetObject)(
    self,
    A,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_SetCommunicator(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

SIDL_C_INLINE_DEFN
void
bHYPRE_SStructParCSRVector_Destroy(
  /* in */ bHYPRE_SStructParCSRVector self,
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
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_Initialize(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_Initialize)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_Assemble(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_Assemble)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Set {\tt self} to 0.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_Clear(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_Clear)(
    self,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Copy data from x into {\tt self}.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_Copy(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_Copy)(
    self,
    x,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Create an {\tt x} compatible with {\tt self}.
 * The new vector's data is not specified.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_Clone(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_Clone)(
    self,
    x,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Scale {\tt self} by {\tt a}.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_Scale(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ double a,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_Scale)(
    self,
    a,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_Dot(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_Dot)(
    self,
    x,
    d,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Add {\tt a}{\tt x} to {\tt self}.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructParCSRVector_Axpy(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_Axpy)(
    self,
    a,
    x,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__cast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  bHYPRE_SStructParCSRVector cast = NULL;

  if(!connect_loaded) {
    connect_loaded = 1;
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.SStructParCSRVector",
      (void*)bHYPRE_SStructParCSRVector__IHConnect,_ex);SIDL_CHECK(*_ex);
  }
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_SStructParCSRVector) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.SStructParCSRVector", _ex); SIDL_CHECK(*_ex);
  }

  EXIT:
  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_SStructParCSRVector__cast2(
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
bHYPRE_SStructParCSRVector__exec(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector__getURL(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector__raddRef(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector__set_hooks_static(
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
bHYPRE_SStructParCSRVector__set_hooks(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector__isRemote(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector__isLocal(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
{
  return !bHYPRE_SStructParCSRVector__isRemote(self,_ex);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_createCol(dimen,
    lower, upper);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_createRow(dimen,
    lower, upper);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create1d(int32_t len)
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructParCSRVector* data)
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_create2dCol(m, n);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_create2dRow(m, n);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_borrow(
  bHYPRE_SStructParCSRVector* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct 
    bHYPRE_SStructParCSRVector__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_smartCopy(
  struct bHYPRE_SStructParCSRVector__array *array)
{
  return (struct bHYPRE_SStructParCSRVector__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_SStructParCSRVector__array_addRef(
  struct bHYPRE_SStructParCSRVector__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_SStructParCSRVector__array_deleteRef(
  struct bHYPRE_SStructParCSRVector__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get1(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get2(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get3(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get4(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get5(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get6(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get7(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t indices[])
{
  return (bHYPRE_SStructParCSRVector)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_SStructParCSRVector__array_set1(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set2(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set3(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set4(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set5(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set6(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set7(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRVector__array_set(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t indices[],
  bHYPRE_SStructParCSRVector const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_SStructParCSRVector__array_dimen(
  const struct bHYPRE_SStructParCSRVector__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_SStructParCSRVector__array_lower(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructParCSRVector__array_upper(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructParCSRVector__array_length(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructParCSRVector__array_stride(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_SStructParCSRVector__array_isColumnOrder(
  const struct bHYPRE_SStructParCSRVector__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_SStructParCSRVector__array_isRowOrder(
  const struct bHYPRE_SStructParCSRVector__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_SStructParCSRVector__array_copy(
  const struct bHYPRE_SStructParCSRVector__array* src,
  struct bHYPRE_SStructParCSRVector__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_slice(
  struct bHYPRE_SStructParCSRVector__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_SStructParCSRVector__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_ensure(
  struct bHYPRE_SStructParCSRVector__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_SStructParCSRVector__array*)
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
static struct sidl_recursive_mutex_t bHYPRE_SStructParCSRVector__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE_SStructParCSRVector__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE_SStructParCSRVector__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE_SStructParCSRVector__mutex )==EDEADLOCK) */
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

static struct bHYPRE_SStructParCSRVector__epv 
  s_rem_epv__bhypre_sstructparcsrvector;

static struct bHYPRE_MatrixVectorView__epv s_rem_epv__bhypre_matrixvectorview;

static struct bHYPRE_ProblemDefinition__epv s_rem_epv__bhypre_problemdefinition;

static struct bHYPRE_SStructMatrixVectorView__epv 
  s_rem_epv__bhypre_sstructmatrixvectorview;

static struct bHYPRE_SStructVectorView__epv s_rem_epv__bhypre_sstructvectorview;

static struct bHYPRE_Vector__epv s_rem_epv__bhypre_vector;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_bHYPRE_SStructParCSRVector__cast(
  struct bHYPRE_SStructParCSRVector__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2,
    cmp3;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "bHYPRE.SStructVectorView");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_bhypre_sstructvectorview);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "bHYPRE.SStructMatrixVectorView");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_bhypre_sstructmatrixvectorview);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "bHYPRE.ProblemDefinition");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_bhypre_problemdefinition);
        return cast;
      }
      else if (cmp2 < 0) {
        cmp3 = strcmp(name, "bHYPRE.MatrixVectorView");
        if (!cmp3) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = &((*self).d_bhypre_matrixvectorview);
          return cast;
        }
      }
    }
    else if (cmp1 > 0) {
      cmp2 = strcmp(name, "bHYPRE.SStructParCSRVector");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = self;
        return cast;
      }
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.BaseClass");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = self;
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "bHYPRE.Vector");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_bhypre_vector);
        return cast;
      }
    }
    else if (cmp1 > 0) {
      cmp2 = strcmp(name, "sidl.BaseInterface");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
        return cast;
      }
    }
  }
  if ((*self->d_epv->f_isType)(self,name, _ex)) {
    void* (*func)(struct sidl_rmi_InstanceHandle__object*,
      struct sidl_BaseInterface__object**) = 
      (void* (*)(struct sidl_rmi_InstanceHandle__object*,
        struct sidl_BaseInterface__object**)) 
      sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
    cast =  (*func)(((struct 
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih, _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE_SStructParCSRVector__delete(
  struct bHYPRE_SStructParCSRVector__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_bHYPRE_SStructParCSRVector__getURL(
  struct bHYPRE_SStructParCSRVector__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_bHYPRE_SStructParCSRVector__raddRef(
  struct bHYPRE_SStructParCSRVector__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
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
remote_bHYPRE_SStructParCSRVector__isRemote(
    struct bHYPRE_SStructParCSRVector__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_bHYPRE_SStructParCSRVector__set_hooks(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector._set_hooks.", &throwaway_exception);
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
static void remote_bHYPRE_SStructParCSRVector__exec(
  struct bHYPRE_SStructParCSRVector__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:addRef */
static void
remote_bHYPRE_SStructParCSRVector_addRef(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_SStructParCSRVector__remote* r_obj = (struct 
      bHYPRE_SStructParCSRVector__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_bHYPRE_SStructParCSRVector_deleteRef(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_SStructParCSRVector__remote* r_obj = (struct 
      bHYPRE_SStructParCSRVector__remote*)self->d_data;
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
remote_bHYPRE_SStructParCSRVector_isSame(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.isSame.", &throwaway_exception);
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
remote_bHYPRE_SStructParCSRVector_isType(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.isType.", &throwaway_exception);
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
remote_bHYPRE_SStructParCSRVector_getClassInfo(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.getClassInfo.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetGrid */
static int32_t
remote_bHYPRE_SStructParCSRVector_SetGrid(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ struct bHYPRE_SStructGrid__object* grid,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetGrid", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(grid){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)grid,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "grid", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "grid", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.SetGrid.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetValues */
static int32_t
remote_bHYPRE_SStructParCSRVector_SetValues(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ double value,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "index", index,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDouble( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.SetValues.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetBoxValues */
static int32_t
remote_bHYPRE_SStructParCSRVector_SetBoxValues(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ struct sidl_double__array* values,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetBoxValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "ilower", ilower,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "iupper", iupper,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.SetBoxValues.", &throwaway_exception);
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

/* REMOTE METHOD STUB:AddToValues */
static int32_t
remote_bHYPRE_SStructParCSRVector_AddToValues(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ double value,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "AddToValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "index", index,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDouble( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.AddToValues.", &throwaway_exception);
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

/* REMOTE METHOD STUB:AddToBoxValues */
static int32_t
remote_bHYPRE_SStructParCSRVector_AddToBoxValues(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ struct sidl_double__array* values,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "AddToBoxValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "ilower", ilower,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "iupper", iupper,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.AddToBoxValues.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Gather */
static int32_t
remote_bHYPRE_SStructParCSRVector_Gather(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Gather", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Gather.", &throwaway_exception);
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

/* REMOTE METHOD STUB:GetValues */
static int32_t
remote_bHYPRE_SStructParCSRVector_GetValues(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* out */ double* value,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "GetValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "index", index,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.GetValues.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDouble( _rsvp, "value", value,
      _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:GetBoxValues */
static int32_t
remote_bHYPRE_SStructParCSRVector_GetBoxValues(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* inout rarray[nvalues] */ struct sidl_double__array** values,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "GetBoxValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "ilower", ilower,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "iupper", iupper,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", *values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.GetBoxValues.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDoubleArray( _rsvp, "values", values,
      sidl_column_major_order,1,TRUE, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetComplex */
static int32_t
remote_bHYPRE_SStructParCSRVector_SetComplex(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetComplex", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.SetComplex.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Print */
static int32_t
remote_bHYPRE_SStructParCSRVector_Print(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ const char* filename,
  /* in */ int32_t all,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Print", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "filename", filename,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "all", all, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Print.", &throwaway_exception);
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

/* REMOTE METHOD STUB:GetObject */
static int32_t
remote_bHYPRE_SStructParCSRVector_GetObject(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* out */ struct sidl_BaseInterface__object** A,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char* A_str= NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "GetObject", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.GetObject.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackString( _rsvp, "A", &A_str, _ex);SIDL_CHECK(*_ex);
    *A = sidl_BaseInterface__connectI(A_str, FALSE, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetCommunicator */
static int32_t
remote_bHYPRE_SStructParCSRVector_SetCommunicator(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetCommunicator", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(mpi_comm){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)mpi_comm,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "mpi_comm", _url,
        _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "mpi_comm", NULL,
        _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.SetCommunicator.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Destroy */
static void
remote_bHYPRE_SStructParCSRVector_Destroy(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Destroy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Destroy.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Initialize */
static int32_t
remote_bHYPRE_SStructParCSRVector_Initialize(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Initialize", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Initialize.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Assemble */
static int32_t
remote_bHYPRE_SStructParCSRVector_Assemble(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Assemble", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Assemble.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Clear */
static int32_t
remote_bHYPRE_SStructParCSRVector_Clear(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Clear", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Clear.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Copy */
static int32_t
remote_bHYPRE_SStructParCSRVector_Copy(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ struct bHYPRE_Vector__object* x,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Copy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(x){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)x,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "x", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "x", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Copy.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Clone */
static int32_t
remote_bHYPRE_SStructParCSRVector_Clone(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* out */ struct bHYPRE_Vector__object** x,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char* x_str= NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Clone", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Clone.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackString( _rsvp, "x", &x_str, _ex);SIDL_CHECK(*_ex);
    *x = bHYPRE_Vector__connectI(x_str, FALSE, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Scale */
static int32_t
remote_bHYPRE_SStructParCSRVector_Scale(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ double a,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Scale", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packDouble( _inv, "a", a, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Scale.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Dot */
static int32_t
remote_bHYPRE_SStructParCSRVector_Dot(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ struct bHYPRE_Vector__object* x,
  /* out */ double* d,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Dot", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(x){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)x,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "x", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "x", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Dot.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
      _ex);SIDL_CHECK(*_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDouble( _rsvp, "d", d, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Axpy */
static int32_t
remote_bHYPRE_SStructParCSRVector_Axpy(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ double a,
  /* in */ struct bHYPRE_Vector__object* x,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Axpy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packDouble( _inv, "a", a, _ex);SIDL_CHECK(*_ex);
    if(x){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)x,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "x", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "x", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Axpy.", &throwaway_exception);
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

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void bHYPRE_SStructParCSRVector__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE_SStructParCSRVector__epv*     epv = 
    &s_rem_epv__bhypre_sstructparcsrvector;
  struct bHYPRE_MatrixVectorView__epv*        e0  = 
    &s_rem_epv__bhypre_matrixvectorview;
  struct bHYPRE_ProblemDefinition__epv*       e1  = 
    &s_rem_epv__bhypre_problemdefinition;
  struct bHYPRE_SStructMatrixVectorView__epv* e2  = 
    &s_rem_epv__bhypre_sstructmatrixvectorview;
  struct bHYPRE_SStructVectorView__epv*       e3  = 
    &s_rem_epv__bhypre_sstructvectorview;
  struct bHYPRE_Vector__epv*                  e4  = &s_rem_epv__bhypre_vector;
  struct sidl_BaseClass__epv*                 e5  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*             e6  = 
    &s_rem_epv__sidl_baseinterface;

  epv->f__cast                = remote_bHYPRE_SStructParCSRVector__cast;
  epv->f__delete              = remote_bHYPRE_SStructParCSRVector__delete;
  epv->f__exec                = remote_bHYPRE_SStructParCSRVector__exec;
  epv->f__getURL              = remote_bHYPRE_SStructParCSRVector__getURL;
  epv->f__raddRef             = remote_bHYPRE_SStructParCSRVector__raddRef;
  epv->f__isRemote            = remote_bHYPRE_SStructParCSRVector__isRemote;
  epv->f__set_hooks           = remote_bHYPRE_SStructParCSRVector__set_hooks;
  epv->f__ctor                = NULL;
  epv->f__ctor2               = NULL;
  epv->f__dtor                = NULL;
  epv->f_addRef               = remote_bHYPRE_SStructParCSRVector_addRef;
  epv->f_deleteRef            = remote_bHYPRE_SStructParCSRVector_deleteRef;
  epv->f_isSame               = remote_bHYPRE_SStructParCSRVector_isSame;
  epv->f_isType               = remote_bHYPRE_SStructParCSRVector_isType;
  epv->f_getClassInfo         = remote_bHYPRE_SStructParCSRVector_getClassInfo;
  epv->f_SetGrid              = remote_bHYPRE_SStructParCSRVector_SetGrid;
  epv->f_SetValues            = remote_bHYPRE_SStructParCSRVector_SetValues;
  epv->f_SetBoxValues         = remote_bHYPRE_SStructParCSRVector_SetBoxValues;
  epv->f_AddToValues          = remote_bHYPRE_SStructParCSRVector_AddToValues;
  epv->f_AddToBoxValues       = 
    remote_bHYPRE_SStructParCSRVector_AddToBoxValues;
  epv->f_Gather               = remote_bHYPRE_SStructParCSRVector_Gather;
  epv->f_GetValues            = remote_bHYPRE_SStructParCSRVector_GetValues;
  epv->f_GetBoxValues         = remote_bHYPRE_SStructParCSRVector_GetBoxValues;
  epv->f_SetComplex           = remote_bHYPRE_SStructParCSRVector_SetComplex;
  epv->f_Print                = remote_bHYPRE_SStructParCSRVector_Print;
  epv->f_GetObject            = remote_bHYPRE_SStructParCSRVector_GetObject;
  epv->f_SetCommunicator      = 
    remote_bHYPRE_SStructParCSRVector_SetCommunicator;
  epv->f_Destroy              = remote_bHYPRE_SStructParCSRVector_Destroy;
  epv->f_Initialize           = remote_bHYPRE_SStructParCSRVector_Initialize;
  epv->f_Assemble             = remote_bHYPRE_SStructParCSRVector_Assemble;
  epv->f_Clear                = remote_bHYPRE_SStructParCSRVector_Clear;
  epv->f_Copy                 = remote_bHYPRE_SStructParCSRVector_Copy;
  epv->f_Clone                = remote_bHYPRE_SStructParCSRVector_Clone;
  epv->f_Scale                = remote_bHYPRE_SStructParCSRVector_Scale;
  epv->f_Dot                  = remote_bHYPRE_SStructParCSRVector_Dot;
  epv->f_Axpy                 = remote_bHYPRE_SStructParCSRVector_Axpy;

  e0->f__cast           = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e0->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e0->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e0->f__isRemote       = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e0->f__set_hooks      = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e0->f__exec           = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e0->f_Destroy         = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e0->f_Initialize      = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e0->f_addRef          = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e0->f_isType          = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e1->f__cast           = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e1->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e1->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e1->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e1->f__isRemote       = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e1->f__set_hooks      = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e1->f__exec           = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e1->f_Destroy         = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e1->f_Initialize      = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e1->f_Assemble        = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e1->f_addRef          = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e1->f_deleteRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e1->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType          = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e2->f__cast           = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e2->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e2->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e2->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e2->f__isRemote       = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e2->f__set_hooks      = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e2->f__exec           = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_GetObject       = (int32_t (*)(void*,
    struct sidl_BaseInterface__object**,
    struct sidl_BaseInterface__object **)) epv->f_GetObject;
  e2->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e2->f_Destroy         = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e2->f_Initialize      = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e2->f_Assemble        = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e2->f_addRef          = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e2->f_deleteRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e2->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e2->f_isType          = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e2->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e3->f__cast           = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e3->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e3->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e3->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e3->f__isRemote       = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e3->f__set_hooks      = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e3->f__exec           = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e3->f_SetGrid         = (int32_t (*)(void*,struct bHYPRE_SStructGrid__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetGrid;
  e3->f_SetValues       = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    int32_t,double,struct sidl_BaseInterface__object **)) epv->f_SetValues;
  e3->f_SetBoxValues    = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,int32_t,struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetBoxValues;
  e3->f_AddToValues     = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    int32_t,double,struct sidl_BaseInterface__object **)) epv->f_AddToValues;
  e3->f_AddToBoxValues  = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,int32_t,struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_AddToBoxValues;
  e3->f_Gather          = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Gather;
  e3->f_GetValues       = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    int32_t,double*,struct sidl_BaseInterface__object **)) epv->f_GetValues;
  e3->f_GetBoxValues    = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,int32_t,struct sidl_double__array**,
    struct sidl_BaseInterface__object **)) epv->f_GetBoxValues;
  e3->f_SetComplex      = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_SetComplex;
  e3->f_Print           = (int32_t (*)(void*,const char*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_Print;
  e3->f_GetObject       = (int32_t (*)(void*,
    struct sidl_BaseInterface__object**,
    struct sidl_BaseInterface__object **)) epv->f_GetObject;
  e3->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e3->f_Destroy         = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e3->f_Initialize      = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e3->f_Assemble        = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e3->f_addRef          = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e3->f_deleteRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e3->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e3->f_isType          = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e3->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e4->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e4->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e4->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e4->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e4->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e4->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e4->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e4->f_Clear        = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Clear;
  e4->f_Copy         = (int32_t (*)(void*,struct bHYPRE_Vector__object*,
    struct sidl_BaseInterface__object **)) epv->f_Copy;
  e4->f_Clone        = (int32_t (*)(void*,struct bHYPRE_Vector__object**,
    struct sidl_BaseInterface__object **)) epv->f_Clone;
  e4->f_Scale        = (int32_t (*)(void*,double,
    struct sidl_BaseInterface__object **)) epv->f_Scale;
  e4->f_Dot          = (int32_t (*)(void*,struct bHYPRE_Vector__object*,double*,
    struct sidl_BaseInterface__object **)) epv->f_Dot;
  e4->f_Axpy         = (int32_t (*)(void*,double,struct bHYPRE_Vector__object*,
    struct sidl_BaseInterface__object **)) epv->f_Axpy;
  e4->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e4->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e4->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e4->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e4->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e5->f__cast        = (void* (*)(struct sidl_BaseClass__object*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e5->f__delete      = (void (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__delete;
  e5->f__getURL      = (char* (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__getURL;
  e5->f__raddRef     = (void (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e5->f__isRemote    = (sidl_bool (*)(struct sidl_BaseClass__object*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e5->f__set_hooks   = (void (*)(struct sidl_BaseClass__object*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e5->f__exec        = (void (*)(struct sidl_BaseClass__object*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e5->f_addRef       = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e5->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e5->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e5->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e5->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e6->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e6->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e6->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e6->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e6->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e6->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e6->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e6->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e6->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e6->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e6->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e6->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_SStructParCSRVector__object* self;

  struct bHYPRE_SStructParCSRVector__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_SStructParCSRVector__remote* r_obj;
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
    return bHYPRE_SStructParCSRVector__rmicast(bi,_ex);SIDL_CHECK(*_ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar,
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_SStructParCSRVector__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__object));

  r_obj =
    (struct bHYPRE_SStructParCSRVector__remote*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                      self;
  s1 =                                      &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructParCSRVector__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_sstructmatrixvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstructmatrixvectorview;
  s0->d_bhypre_sstructmatrixvectorview.d_object = (void*) self;

  s0->d_bhypre_sstructvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstructvectorview;
  s0->d_bhypre_sstructvectorview.d_object = (void*) self;

  s0->d_bhypre_vector.d_epv    = &s_rem_epv__bhypre_vector;
  s0->d_bhypre_vector.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_sstructparcsrvector;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  return NULL;
}
/* Create an instance that uses an already existing  */
/* InstanceHandle to connect to an existing remote object. */
static struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_SStructParCSRVector__object* self;

  struct bHYPRE_SStructParCSRVector__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_SStructParCSRVector__remote* r_obj;
  self =
    (struct bHYPRE_SStructParCSRVector__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__object));

  r_obj =
    (struct bHYPRE_SStructParCSRVector__remote*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                      self;
  s1 =                                      &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructParCSRVector__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_sstructmatrixvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstructmatrixvectorview;
  s0->d_bhypre_sstructmatrixvectorview.d_object = (void*) self;

  s0->d_bhypre_sstructvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstructvectorview;
  s0->d_bhypre_sstructvectorview.d_object = (void*) self;

  s0->d_bhypre_vector.d_epv    = &s_rem_epv__bhypre_vector;
  s0->d_bhypre_vector.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_sstructparcsrvector;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}
/* REMOTE: generate remote instance given URL string. */
static struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__remoteCreate(const char *url,
  sidl_BaseInterface *_ex)
{
  sidl_BaseInterface _throwaway_exception = NULL;
  struct bHYPRE_SStructParCSRVector__object* self;

  struct bHYPRE_SStructParCSRVector__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_SStructParCSRVector__remote* r_obj;
  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_createInstance(url, "bHYPRE.SStructParCSRVector",
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_SStructParCSRVector__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__object));

  r_obj =
    (struct bHYPRE_SStructParCSRVector__remote*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                      self;
  s1 =                                      &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructParCSRVector__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_sstructmatrixvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstructmatrixvectorview;
  s0->d_bhypre_sstructmatrixvectorview.d_object = (void*) self;

  s0->d_bhypre_sstructvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstructvectorview;
  s0->d_bhypre_sstructvectorview.d_object = (void*) self;

  s0->d_bhypre_vector.d_epv    = &s_rem_epv__bhypre_vector;
  s0->d_bhypre_vector.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_sstructparcsrvector;

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

struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct bHYPRE_SStructParCSRVector__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.SStructParCSRVector",
      (void*)bHYPRE_SStructParCSRVector__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct bHYPRE_SStructParCSRVector__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.SStructParCSRVector", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return bHYPRE_SStructParCSRVector__remoteConnect(url, ar, _ex);
}

