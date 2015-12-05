/*
 * File:          bHYPRE_StructMatrixView_Stub.c
 * Symbol:        bHYPRE.StructMatrixView-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.StructMatrixView
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_StructMatrixView.h"
#include "bHYPRE_StructMatrixView_IOR.h"
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

static struct bHYPRE_StructMatrixView__object* 
  bHYPRE_StructMatrixView__remoteConnect(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
static struct bHYPRE_StructMatrixView__object* 
  bHYPRE_StructMatrixView__IHConnect(struct sidl_rmi_InstanceHandle__object* 
  instance, sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__connect(const char* url, sidl_BaseInterface *_ex)
{
  return bHYPRE_StructMatrixView__remoteConnect(url, TRUE, _ex);
}

/*
 *  Set the grid on which vectors are defined.  This and the stencil
 * determine the matrix structure. 
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_StructMatrixView_SetGrid(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ bHYPRE_StructGrid grid,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_SetGrid)(
    self->d_object,
    grid,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 *  Set the stencil. This and the grid determine the matrix structure. 
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_StructMatrixView_SetStencil(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ bHYPRE_StructStencil stencil,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_SetStencil)(
    self->d_object,
    stencil,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 *  Set matrix values at grid point, given by "index".
 * You can supply values for one or more positions in the stencil.
 * "index" is an array of size "dim"; and "stencil_indices" and "values"
 * are arrays of size "num_stencil_indices".
 */

int32_t
bHYPRE_StructMatrixView_SetValues(
  /* in */ bHYPRE_StructMatrixView self,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */ double* values,
  /* out */ sidl_BaseInterface *_ex)
{
  int32_t index_lower[1], index_upper[1], index_stride[1]; 
  struct sidl_int__array index_real;
  struct sidl_int__array*index_tmp = &index_real;
  int32_t stencil_indices_lower[1], stencil_indices_upper[1],
    stencil_indices_stride[1]; 
  struct sidl_int__array stencil_indices_real;
  struct sidl_int__array*stencil_indices_tmp = &stencil_indices_real;
  int32_t values_lower[1], values_upper[1], values_stride[1]; 
  struct sidl_double__array values_real;
  struct sidl_double__array*values_tmp = &values_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  stencil_indices_upper[0] = num_stencil_indices-1;
  sidl_int__array_init(stencil_indices, stencil_indices_tmp, 1,
    stencil_indices_lower, stencil_indices_upper, stencil_indices_stride);
  values_upper[0] = num_stencil_indices-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  return (*self->d_epv->f_SetValues)(
    self->d_object,
    index_tmp,
    stencil_indices_tmp,
    values_tmp,
    _ex);
}

/*
 *  Set matrix values throughout a box in the grid, specified by its lower
 * and upper corners.  You can supply these values for one or more positions
 * in the stencil.  Thus the total number of matrix values you supply,
 * "nvalues", is num_stencil_indices x box_size, where box_size is the
 * number of grid points in the box.  The values array should be organized
 * so all values for a given box point are together (i.e., the stencil
 * index is the most rapidly varying).
 * "ilower" and "iupper" are arrays of size "dim", "stencil_indices" is an
 * array of size "num_stencil_indices", and "values" is an array of size
 * "nvalues". 
 */

int32_t
bHYPRE_StructMatrixView_SetBoxValues(
  /* in */ bHYPRE_StructMatrixView self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
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
  int32_t stencil_indices_lower[1], stencil_indices_upper[1],
    stencil_indices_stride[1]; 
  struct sidl_int__array stencil_indices_real;
  struct sidl_int__array*stencil_indices_tmp = &stencil_indices_real;
  int32_t values_lower[1], values_upper[1], values_stride[1]; 
  struct sidl_double__array values_real;
  struct sidl_double__array*values_tmp = &values_real;
  ilower_upper[0] = dim-1;
  sidl_int__array_init(ilower, ilower_tmp, 1, ilower_lower, ilower_upper,
    ilower_stride);
  iupper_upper[0] = dim-1;
  sidl_int__array_init(iupper, iupper_tmp, 1, iupper_lower, iupper_upper,
    iupper_stride);
  stencil_indices_upper[0] = num_stencil_indices-1;
  sidl_int__array_init(stencil_indices, stencil_indices_tmp, 1,
    stencil_indices_lower, stencil_indices_upper, stencil_indices_stride);
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  return (*self->d_epv->f_SetBoxValues)(
    self->d_object,
    ilower_tmp,
    iupper_tmp,
    stencil_indices_tmp,
    values_tmp,
    _ex);
}

/*
 *  Set the number of ghost zones, separately on the lower and upper sides
 * for each dimension.
 * "num_ghost" is an array of size "dim2", twice the number of dimensions
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_StructMatrixView_SetNumGhost(
  /* in */ bHYPRE_StructMatrixView self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  int32_t num_ghost_lower[1], num_ghost_upper[1], num_ghost_stride[1]; 
  struct sidl_int__array num_ghost_real;
  struct sidl_int__array*num_ghost_tmp = &num_ghost_real;
  num_ghost_upper[0] = dim2-1;
  sidl_int__array_init(num_ghost, num_ghost_tmp, 1, num_ghost_lower,
    num_ghost_upper, num_ghost_stride);
  return (*self->d_epv->f_SetNumGhost)(
    self->d_object,
    num_ghost_tmp,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 *  Call SetSymmetric with symmetric=1 to turn on symmetric matrix storage if
 * available. 
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_StructMatrixView_SetSymmetric(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ int32_t symmetric,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_SetSymmetric)(
    self->d_object,
    symmetric,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 *  State which stencil entries are constant over the grid.
 * Supported options are: (i) none (the default),
 * (ii) all (stencil_constant_points should include all stencil points)
 * (iii) all entries but the diagonal. 
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_StructMatrixView_SetConstantEntries(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ int32_t num_stencil_constant_points,
  /* in rarray[num_stencil_constant_points] */ int32_t* stencil_constant_points,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  int32_t stencil_constant_points_lower[1], stencil_constant_points_upper[1],
    stencil_constant_points_stride[1]; 
  struct sidl_int__array stencil_constant_points_real;
  struct sidl_int__array*stencil_constant_points_tmp = 
    &stencil_constant_points_real;
  stencil_constant_points_upper[0] = num_stencil_constant_points-1;
  sidl_int__array_init(stencil_constant_points, stencil_constant_points_tmp, 1,
    stencil_constant_points_lower, stencil_constant_points_upper,
    stencil_constant_points_stride);
  return (*self->d_epv->f_SetConstantEntries)(
    self->d_object,
    stencil_constant_points_tmp,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 *  Provide values for matrix coefficients which are constant throughout
 * the grid, one value for each stencil point.
 * "stencil_indices" and "values" is each an array of length
 * "num_stencil_indices" 
 */

int32_t
bHYPRE_StructMatrixView_SetConstantValues(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */ double* values,
  /* out */ sidl_BaseInterface *_ex)
{
  int32_t stencil_indices_lower[1], stencil_indices_upper[1],
    stencil_indices_stride[1]; 
  struct sidl_int__array stencil_indices_real;
  struct sidl_int__array*stencil_indices_tmp = &stencil_indices_real;
  int32_t values_lower[1], values_upper[1], values_stride[1]; 
  struct sidl_double__array values_real;
  struct sidl_double__array*values_tmp = &values_real;
  stencil_indices_upper[0] = num_stencil_indices-1;
  sidl_int__array_init(stencil_indices, stencil_indices_tmp, 1,
    stencil_indices_lower, stencil_indices_upper, stencil_indices_stride);
  values_upper[0] = num_stencil_indices-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  return (*self->d_epv->f_SetConstantValues)(
    self->d_object,
    stencil_indices_tmp,
    values_tmp,
    _ex);
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_StructMatrixView_SetCommunicator(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_SetCommunicator)(
    self->d_object,
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
bHYPRE_StructMatrixView_Destroy(
  /* in */ bHYPRE_StructMatrixView self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f_Destroy)(
    self->d_object,
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
bHYPRE_StructMatrixView_Initialize(
  /* in */ bHYPRE_StructMatrixView self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_Initialize)(
    self->d_object,
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
bHYPRE_StructMatrixView_Assemble(
  /* in */ bHYPRE_StructMatrixView self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_Assemble)(
    self->d_object,
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
bHYPRE_StructMatrixView_addRef(
  /* in */ bHYPRE_StructMatrixView self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f_addRef)(
    self->d_object,
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
bHYPRE_StructMatrixView_deleteRef(
  /* in */ bHYPRE_StructMatrixView self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f_deleteRef)(
    self->d_object,
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
bHYPRE_StructMatrixView_isSame(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_isSame)(
    self->d_object,
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
bHYPRE_StructMatrixView_isType(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_isType)(
    self->d_object,
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
bHYPRE_StructMatrixView_getClassInfo(
  /* in */ bHYPRE_StructMatrixView self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_getClassInfo)(
    self->d_object,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__cast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  bHYPRE_StructMatrixView cast = NULL;

  if(!connect_loaded) {
    connect_loaded = 1;
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.StructMatrixView",
      (void*)bHYPRE_StructMatrixView__IHConnect,_ex);SIDL_CHECK(*_ex);
  }
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_StructMatrixView) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.StructMatrixView", _ex); SIDL_CHECK(*_ex);
  }

  EXIT:
  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_StructMatrixView__cast2(
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
bHYPRE_StructMatrixView__exec(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ const char* methodName,
  /* in */ sidl_rmi_Call inArgs,
  /* in */ sidl_rmi_Return outArgs,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f__exec)(
    self->d_object,
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
bHYPRE_StructMatrixView__getURL(
  /* in */ bHYPRE_StructMatrixView self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f__getURL)(
    self->d_object,
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
bHYPRE_StructMatrixView__raddRef(
  /* in */ bHYPRE_StructMatrixView self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f__raddRef)(
    self->d_object,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */


/*
 * Method to set whether or not method hooks should be invoked.
 */

SIDL_C_INLINE_DEFN
void
bHYPRE_StructMatrixView__set_hooks(
  /* in */ bHYPRE_StructMatrixView self,
  /* in */ sidl_bool on,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  (*self->d_epv->f__set_hooks)(
    self->d_object,
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
bHYPRE_StructMatrixView__isRemote(
  /* in */ bHYPRE_StructMatrixView self,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f__isRemote)(
    self->d_object,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */


/*
 * TRUE if this object is remote, false if local
 */

sidl_bool
bHYPRE_StructMatrixView__isLocal(
  /* in */ bHYPRE_StructMatrixView self,
  /* out */ sidl_BaseInterface *_ex)
{
  return !bHYPRE_StructMatrixView__isRemote(self,_ex);
}

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_StructMatrixView__array*)sidl_interface__array_createCol(dimen,
    lower, upper);
}

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_StructMatrixView__array*)sidl_interface__array_createRow(dimen,
    lower, upper);
}

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_create1d(int32_t len)
{
  return (struct 
    bHYPRE_StructMatrixView__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_create1dInit(
  int32_t len, 
  bHYPRE_StructMatrixView* data)
{
  return (struct 
    bHYPRE_StructMatrixView__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_StructMatrixView__array*)sidl_interface__array_create2dCol(m, n);
}

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_StructMatrixView__array*)sidl_interface__array_create2dRow(m, n);
}

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_borrow(
  bHYPRE_StructMatrixView* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct bHYPRE_StructMatrixView__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_smartCopy(
  struct bHYPRE_StructMatrixView__array *array)
{
  return (struct bHYPRE_StructMatrixView__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_StructMatrixView__array_addRef(
  struct bHYPRE_StructMatrixView__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_StructMatrixView__array_deleteRef(
  struct bHYPRE_StructMatrixView__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get1(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1)
{
  return (bHYPRE_StructMatrixView)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get2(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_StructMatrixView)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get3(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_StructMatrixView)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get4(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_StructMatrixView)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get5(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_StructMatrixView)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get6(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_StructMatrixView)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get7(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_StructMatrixView)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_StructMatrixView
bHYPRE_StructMatrixView__array_get(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t indices[])
{
  return (bHYPRE_StructMatrixView)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_StructMatrixView__array_set1(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  bHYPRE_StructMatrixView const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructMatrixView__array_set2(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructMatrixView const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructMatrixView__array_set3(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructMatrixView const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructMatrixView__array_set4(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructMatrixView const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructMatrixView__array_set5(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructMatrixView const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructMatrixView__array_set6(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructMatrixView const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructMatrixView__array_set7(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructMatrixView const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_StructMatrixView__array_set(
  struct bHYPRE_StructMatrixView__array* array,
  const int32_t indices[],
  bHYPRE_StructMatrixView const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_StructMatrixView__array_dimen(
  const struct bHYPRE_StructMatrixView__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_StructMatrixView__array_lower(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_StructMatrixView__array_upper(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_StructMatrixView__array_length(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_StructMatrixView__array_stride(
  const struct bHYPRE_StructMatrixView__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_StructMatrixView__array_isColumnOrder(
  const struct bHYPRE_StructMatrixView__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_StructMatrixView__array_isRowOrder(
  const struct bHYPRE_StructMatrixView__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_StructMatrixView__array_copy(
  const struct bHYPRE_StructMatrixView__array* src,
  struct bHYPRE_StructMatrixView__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_slice(
  struct bHYPRE_StructMatrixView__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_StructMatrixView__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_StructMatrixView__array*
bHYPRE_StructMatrixView__array_ensure(
  struct bHYPRE_StructMatrixView__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_StructMatrixView__array*)
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
static struct sidl_recursive_mutex_t bHYPRE__StructMatrixView__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE__StructMatrixView__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE__StructMatrixView__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE__StructMatrixView__mutex )==EDEADLOCK) */
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

static struct bHYPRE__StructMatrixView__epv s_rem_epv__bhypre__structmatrixview;

static struct bHYPRE_MatrixVectorView__epv s_rem_epv__bhypre_matrixvectorview;

static struct bHYPRE_ProblemDefinition__epv s_rem_epv__bhypre_problemdefinition;

static struct bHYPRE_StructMatrixView__epv s_rem_epv__bhypre_structmatrixview;

static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_bHYPRE__StructMatrixView__cast(
  struct bHYPRE__StructMatrixView__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "bHYPRE.StructMatrixView");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_bhypre_structmatrixview);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "bHYPRE.ProblemDefinition");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_bhypre_problemdefinition);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "bHYPRE.MatrixVectorView");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_bhypre_matrixvectorview);
        return cast;
      }
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.BaseInterface");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_baseinterface);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "bHYPRE._StructMatrixView");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = self;
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih, _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE__StructMatrixView__delete(
  struct bHYPRE__StructMatrixView__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_bHYPRE__StructMatrixView__getURL(
  struct bHYPRE__StructMatrixView__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_bHYPRE__StructMatrixView__raddRef(
  struct bHYPRE__StructMatrixView__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
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
remote_bHYPRE__StructMatrixView__isRemote(
    struct bHYPRE__StructMatrixView__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_bHYPRE__StructMatrixView__set_hooks(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView._set_hooks.", &throwaway_exception);
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
static void remote_bHYPRE__StructMatrixView__exec(
  struct bHYPRE__StructMatrixView__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:SetGrid */
static int32_t
remote_bHYPRE__StructMatrixView_SetGrid(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
  /* in */ struct bHYPRE_StructGrid__object* grid,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.SetGrid.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetStencil */
static int32_t
remote_bHYPRE__StructMatrixView_SetStencil(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
  /* in */ struct bHYPRE_StructStencil__object* stencil,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetStencil", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(stencil){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)stencil,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "stencil", _url,
        _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "stencil", NULL,
        _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.SetStencil.", &throwaway_exception);
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
remote_bHYPRE__StructMatrixView_SetValues(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in rarray[num_stencil_indices] */ struct sidl_int__array* stencil_indices,
  /* in rarray[num_stencil_indices] */ struct sidl_double__array* values,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "index", index,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "stencil_indices", stencil_indices,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.SetValues.", &throwaway_exception);
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
remote_bHYPRE__StructMatrixView_SetBoxValues(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in rarray[num_stencil_indices] */ struct sidl_int__array* stencil_indices,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetBoxValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "ilower", ilower,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "iupper", iupper,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "stencil_indices", stencil_indices,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.SetBoxValues.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetNumGhost */
static int32_t
remote_bHYPRE__StructMatrixView_SetNumGhost(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
  /* in rarray[dim2] */ struct sidl_int__array* num_ghost,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetNumGhost", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "num_ghost", num_ghost,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.SetNumGhost.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetSymmetric */
static int32_t
remote_bHYPRE__StructMatrixView_SetSymmetric(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
  /* in */ int32_t symmetric,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetSymmetric", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "symmetric", symmetric,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.SetSymmetric.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetConstantEntries */
static int32_t
remote_bHYPRE__StructMatrixView_SetConstantEntries(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
  /* in rarray[num_stencil_constant_points] */ struct sidl_int__array* 
    stencil_constant_points,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetConstantEntries", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "stencil_constant_points",
      stencil_constant_points,sidl_column_major_order,1,0,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.SetConstantEntries.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetConstantValues */
static int32_t
remote_bHYPRE__StructMatrixView_SetConstantValues(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
  /* in rarray[num_stencil_indices] */ struct sidl_int__array* stencil_indices,
  /* in rarray[num_stencil_indices] */ struct sidl_double__array* values,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetConstantValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "stencil_indices", stencil_indices,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.SetConstantValues.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetCommunicator */
static int32_t
remote_bHYPRE__StructMatrixView_SetCommunicator(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.SetCommunicator.", &throwaway_exception);
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
remote_bHYPRE__StructMatrixView_Destroy(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Destroy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.Destroy.", &throwaway_exception);
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
remote_bHYPRE__StructMatrixView_Initialize(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Initialize", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.Initialize.", &throwaway_exception);
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
remote_bHYPRE__StructMatrixView_Assemble(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Assemble", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.Assemble.", &throwaway_exception);
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
remote_bHYPRE__StructMatrixView_addRef(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE__StructMatrixView__remote* r_obj = (struct 
      bHYPRE__StructMatrixView__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_bHYPRE__StructMatrixView_deleteRef(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE__StructMatrixView__remote* r_obj = (struct 
      bHYPRE__StructMatrixView__remote*)self->d_data;
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
remote_bHYPRE__StructMatrixView_isSame(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.isSame.", &throwaway_exception);
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
remote_bHYPRE__StructMatrixView_isType(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.isType.", &throwaway_exception);
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
remote_bHYPRE__StructMatrixView_getClassInfo(
  /* in */ struct bHYPRE__StructMatrixView__object* self ,
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
      bHYPRE__StructMatrixView__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._StructMatrixView.getClassInfo.", &throwaway_exception);
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
static void bHYPRE__StructMatrixView__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE__StructMatrixView__epv* epv = 
    &s_rem_epv__bhypre__structmatrixview;
  struct bHYPRE_MatrixVectorView__epv*  e0  = 
    &s_rem_epv__bhypre_matrixvectorview;
  struct bHYPRE_ProblemDefinition__epv* e1  = 
    &s_rem_epv__bhypre_problemdefinition;
  struct bHYPRE_StructMatrixView__epv*  e2  = 
    &s_rem_epv__bhypre_structmatrixview;
  struct sidl_BaseInterface__epv*       e3  = &s_rem_epv__sidl_baseinterface;

  epv->f__cast                   = remote_bHYPRE__StructMatrixView__cast;
  epv->f__delete                 = remote_bHYPRE__StructMatrixView__delete;
  epv->f__exec                   = remote_bHYPRE__StructMatrixView__exec;
  epv->f__getURL                 = remote_bHYPRE__StructMatrixView__getURL;
  epv->f__raddRef                = remote_bHYPRE__StructMatrixView__raddRef;
  epv->f__isRemote               = remote_bHYPRE__StructMatrixView__isRemote;
  epv->f__set_hooks              = remote_bHYPRE__StructMatrixView__set_hooks;
  epv->f__ctor                   = NULL;
  epv->f__ctor2                  = NULL;
  epv->f__dtor                   = NULL;
  epv->f_SetGrid                 = remote_bHYPRE__StructMatrixView_SetGrid;
  epv->f_SetStencil              = remote_bHYPRE__StructMatrixView_SetStencil;
  epv->f_SetValues               = remote_bHYPRE__StructMatrixView_SetValues;
  epv->f_SetBoxValues            = remote_bHYPRE__StructMatrixView_SetBoxValues;
  epv->f_SetNumGhost             = remote_bHYPRE__StructMatrixView_SetNumGhost;
  epv->f_SetSymmetric            = remote_bHYPRE__StructMatrixView_SetSymmetric;
  epv->f_SetConstantEntries      = 
    remote_bHYPRE__StructMatrixView_SetConstantEntries;
  epv->f_SetConstantValues       = 
    remote_bHYPRE__StructMatrixView_SetConstantValues;
  epv->f_SetCommunicator         = 
    remote_bHYPRE__StructMatrixView_SetCommunicator;
  epv->f_Destroy                 = remote_bHYPRE__StructMatrixView_Destroy;
  epv->f_Initialize              = remote_bHYPRE__StructMatrixView_Initialize;
  epv->f_Assemble                = remote_bHYPRE__StructMatrixView_Assemble;
  epv->f_addRef                  = remote_bHYPRE__StructMatrixView_addRef;
  epv->f_deleteRef               = remote_bHYPRE__StructMatrixView_deleteRef;
  epv->f_isSame                  = remote_bHYPRE__StructMatrixView_isSame;
  epv->f_isType                  = remote_bHYPRE__StructMatrixView_isType;
  epv->f_getClassInfo            = remote_bHYPRE__StructMatrixView_getClassInfo;

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

  e2->f__cast              = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e2->f__delete            = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__delete;
  e2->f__getURL            = (char* (*)(void*,
    sidl_BaseInterface*)) epv->f__getURL;
  e2->f__raddRef           = (void (*)(void*,
    sidl_BaseInterface*)) epv->f__raddRef;
  e2->f__isRemote          = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e2->f__set_hooks         = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e2->f__exec              = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_SetGrid            = (int32_t (*)(void*,
    struct bHYPRE_StructGrid__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetGrid;
  e2->f_SetStencil         = (int32_t (*)(void*,
    struct bHYPRE_StructStencil__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetStencil;
  e2->f_SetValues          = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetValues;
  e2->f_SetBoxValues       = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_int__array*,struct sidl_int__array*,struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetBoxValues;
  e2->f_SetNumGhost        = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetNumGhost;
  e2->f_SetSymmetric       = (int32_t (*)(void*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_SetSymmetric;
  e2->f_SetConstantEntries = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetConstantEntries;
  e2->f_SetConstantValues  = (int32_t (*)(void*,struct sidl_int__array*,
    struct sidl_double__array*,
    struct sidl_BaseInterface__object **)) epv->f_SetConstantValues;
  e2->f_SetCommunicator    = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
  e2->f_Destroy            = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Destroy;
  e2->f_Initialize         = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Initialize;
  e2->f_Assemble           = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_Assemble;
  e2->f_addRef             = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e2->f_deleteRef          = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e2->f_isSame             = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e2->f_isType             = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e2->f_getClassInfo       = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e3->f__cast        = (void* (*)(void*,const char*,
    sidl_BaseInterface*)) epv->f__cast;
  e3->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e3->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e3->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e3->f__isRemote    = (sidl_bool (*)(void*,
    sidl_BaseInterface*)) epv->f__isRemote;
  e3->f__set_hooks   = (void (*)(void*,int32_t,
    sidl_BaseInterface*)) epv->f__set_hooks;
  e3->f__exec        = (void (*)(void*,const char*,
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e3->f_addRef       = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e3->f_deleteRef    = (void (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e3->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e3->f_isType       = (sidl_bool (*)(void*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_isType;
  e3->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct bHYPRE_StructMatrixView__object*
bHYPRE_StructMatrixView__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE__StructMatrixView__object* self;

  struct bHYPRE__StructMatrixView__object* s0;

  struct bHYPRE__StructMatrixView__remote* r_obj;
  sidl_rmi_InstanceHandle instance = NULL;
  char* objectID = NULL;
  objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
  if(objectID) {
    sidl_BaseInterface bi = 
      (sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
      objectID, _ex);
    if(ar) {
      sidl_BaseInterface_addRef(bi, _ex);
    }
    return bHYPRE_StructMatrixView__rmicast(bi, _ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE__StructMatrixView__object*) malloc(
      sizeof(struct bHYPRE__StructMatrixView__object));

  r_obj =
    (struct bHYPRE__StructMatrixView__remote*) malloc(
      sizeof(struct bHYPRE__StructMatrixView__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                    self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE__StructMatrixView__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_structmatrixview.d_epv    = &s_rem_epv__bhypre_structmatrixview;
  s0->d_bhypre_structmatrixview.d_object = (void*) self;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre__structmatrixview;

  self->d_data = (void*) r_obj;

  return bHYPRE_StructMatrixView__rmicast(self, _ex);
}
/* Create an instance that uses an already existing  */
/* InstanceHandel to connect to an existing remote object. */
static struct bHYPRE_StructMatrixView__object*
bHYPRE_StructMatrixView__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE__StructMatrixView__object* self;

  struct bHYPRE__StructMatrixView__object* s0;

  struct bHYPRE__StructMatrixView__remote* r_obj;
  self =
    (struct bHYPRE__StructMatrixView__object*) malloc(
      sizeof(struct bHYPRE__StructMatrixView__object));

  r_obj =
    (struct bHYPRE__StructMatrixView__remote*) malloc(
      sizeof(struct bHYPRE__StructMatrixView__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                    self;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE__StructMatrixView__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_structmatrixview.d_epv    = &s_rem_epv__bhypre_structmatrixview;
  s0->d_bhypre_structmatrixview.d_object = (void*) self;

  s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s0->d_sidl_baseinterface.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre__structmatrixview;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance, _ex);
  return bHYPRE_StructMatrixView__rmicast(self, _ex);
}
/*
 * Cast method for interface and class type conversions.
 */

struct bHYPRE_StructMatrixView__object*
bHYPRE_StructMatrixView__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct bHYPRE_StructMatrixView__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.StructMatrixView",
      (void*)bHYPRE_StructMatrixView__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct bHYPRE_StructMatrixView__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.StructMatrixView", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct bHYPRE_StructMatrixView__object*
bHYPRE_StructMatrixView__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return bHYPRE_StructMatrixView__remoteConnect(url, ar, _ex);
}

