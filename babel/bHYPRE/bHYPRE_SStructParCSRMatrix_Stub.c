/*
 * File:          bHYPRE_SStructParCSRMatrix_Stub.c
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Client-side glue code for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "bHYPRE_SStructParCSRMatrix.h"
#include "bHYPRE_SStructParCSRMatrix_IOR.h"
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

static const struct bHYPRE_SStructParCSRMatrix__external *_externals = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct bHYPRE_SStructParCSRMatrix__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _externals = bHYPRE_SStructParCSRMatrix__externals();
#else
  _externals = (struct 
    bHYPRE_SStructParCSRMatrix__external*)sidl_dynamicLoadIOR(
    "bHYPRE.SStructParCSRMatrix","bHYPRE_SStructParCSRMatrix__externals") ;
#endif
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())

/*
 * Hold pointer to static entry point vector
 */

static const struct bHYPRE_SStructParCSRMatrix__sepv *_sepv = NULL;
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

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__create()
{
  return (*(_getExternals()->createObject))();
}

static bHYPRE_SStructParCSRMatrix bHYPRE_SStructParCSRMatrix__remote(const 
  char* url, sidl_BaseInterface *_ex);
/*
 * RMI constructor function for the class.
 */

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__createRemote(const char* url,
  sidl_BaseInterface *_ex)
{
  return bHYPRE_SStructParCSRMatrix__remote(url, _ex);
}

static struct bHYPRE_SStructParCSRMatrix__object* 
  bHYPRE_SStructParCSRMatrix__remoteConnect(const char* url,
  sidl_BaseInterface *_ex);
static struct bHYPRE_SStructParCSRMatrix__object* 
  bHYPRE_SStructParCSRMatrix__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__connect(const char* url, sidl_BaseInterface *_ex)
{
  return bHYPRE_SStructParCSRMatrix__remoteConnect(url, _ex);
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
bHYPRE_SStructParCSRMatrix_addRef(
  /* in */ bHYPRE_SStructParCSRMatrix self)
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
bHYPRE_SStructParCSRMatrix_deleteRef(
  /* in */ bHYPRE_SStructParCSRMatrix self)
{
  (*self->d_epv->f_deleteRef)(
    self);
}

/*
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */

sidl_bool
bHYPRE_SStructParCSRMatrix_isSame(
  /* in */ bHYPRE_SStructParCSRMatrix self,
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
bHYPRE_SStructParCSRMatrix_queryInt(
  /* in */ bHYPRE_SStructParCSRMatrix self,
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
bHYPRE_SStructParCSRMatrix_isType(
  /* in */ bHYPRE_SStructParCSRMatrix self,
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
bHYPRE_SStructParCSRMatrix_getClassInfo(
  /* in */ bHYPRE_SStructParCSRMatrix self)
{
  return (*self->d_epv->f_getClassInfo)(
    self);
}

/*
 * Method:  Create[]
 */

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGraph graph)
{
  return (_getSEPV()->f_Create)(
    mpi_comm,
    graph);
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetCommunicator(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_MPICommunicator mpi_comm)
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm);
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_Initialize(
  /* in */ bHYPRE_SStructParCSRMatrix self)
{
  return (*self->d_epv->f_Initialize)(
    self);
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_Assemble(
  /* in */ bHYPRE_SStructParCSRMatrix self)
{
  return (*self->d_epv->f_Assemble)(
    self);
}

/*
 *  A semi-structured matrix or vector contains a Struct or IJ matrix
 *  or vector.  GetObject returns it.
 * The returned type is a sidl.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_GetObject(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface* A)
{
  return (*self->d_epv->f_GetObject)(
    self,
    A);
}

/*
 * Set the matrix graph.
 * DEPRECATED     Use Create
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetGraph(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_SStructGraph graph)
{
  return (*self->d_epv->f_SetGraph)(
    self,
    graph);
}

/*
 * Set matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in */ int32_t* entries,
  /* in */ double* values)
{
  int32_t index_lower[1], index_upper[1], index_stride[1]; 
  struct sidl_int__array index_real;
  struct sidl_int__array*index_tmp = &index_real;
  int32_t entries_lower[1], entries_upper[1], entries_stride[1]; 
  struct sidl_int__array entries_real;
  struct sidl_int__array*entries_tmp = &entries_real;
  int32_t values_lower[1], values_upper[1], values_stride[1]; 
  struct sidl_double__array values_real;
  struct sidl_double__array*values_tmp = &values_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  entries_upper[0] = nentries-1;
  sidl_int__array_init(entries, entries_tmp, 1, entries_lower, entries_upper,
    entries_stride);
  values_upper[0] = nentries-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  return (*self->d_epv->f_SetValues)(
    self,
    part,
    index_tmp,
    var,
    entries_tmp,
    values_tmp);
}

/*
 * Set matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetBoxValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t* ilower,
  /* in */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in */ int32_t* entries,
  /* in */ double* values,
  /* in */ int32_t nvalues)
{
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1]; 
  struct sidl_int__array ilower_real;
  struct sidl_int__array*ilower_tmp = &ilower_real;
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1]; 
  struct sidl_int__array iupper_real;
  struct sidl_int__array*iupper_tmp = &iupper_real;
  int32_t entries_lower[1], entries_upper[1], entries_stride[1]; 
  struct sidl_int__array entries_real;
  struct sidl_int__array*entries_tmp = &entries_real;
  int32_t values_lower[1], values_upper[1], values_stride[1]; 
  struct sidl_double__array values_real;
  struct sidl_double__array*values_tmp = &values_real;
  ilower_upper[0] = dim-1;
  sidl_int__array_init(ilower, ilower_tmp, 1, ilower_lower, ilower_upper,
    ilower_stride);
  iupper_upper[0] = dim-1;
  sidl_int__array_init(iupper, iupper_tmp, 1, iupper_lower, iupper_upper,
    iupper_stride);
  entries_upper[0] = nentries-1;
  sidl_int__array_init(entries, entries_tmp, 1, entries_lower, entries_upper,
    entries_stride);
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  return (*self->d_epv->f_SetBoxValues)(
    self,
    part,
    ilower_tmp,
    iupper_tmp,
    var,
    entries_tmp,
    values_tmp);
}

/*
 * Add to matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_AddToValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in */ int32_t* entries,
  /* in */ double* values)
{
  int32_t index_lower[1], index_upper[1], index_stride[1]; 
  struct sidl_int__array index_real;
  struct sidl_int__array*index_tmp = &index_real;
  int32_t entries_lower[1], entries_upper[1], entries_stride[1]; 
  struct sidl_int__array entries_real;
  struct sidl_int__array*entries_tmp = &entries_real;
  int32_t values_lower[1], values_upper[1], values_stride[1]; 
  struct sidl_double__array values_real;
  struct sidl_double__array*values_tmp = &values_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  entries_upper[0] = nentries-1;
  sidl_int__array_init(entries, entries_tmp, 1, entries_lower, entries_upper,
    entries_stride);
  values_upper[0] = nentries-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  return (*self->d_epv->f_AddToValues)(
    self,
    part,
    index_tmp,
    var,
    entries_tmp,
    values_tmp);
}

/*
 * Add to matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of stencil
 * type.  Also, they must all represent couplings to the same
 * variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_AddToBoxValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t* ilower,
  /* in */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in */ int32_t* entries,
  /* in */ double* values,
  /* in */ int32_t nvalues)
{
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1]; 
  struct sidl_int__array ilower_real;
  struct sidl_int__array*ilower_tmp = &ilower_real;
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1]; 
  struct sidl_int__array iupper_real;
  struct sidl_int__array*iupper_tmp = &iupper_real;
  int32_t entries_lower[1], entries_upper[1], entries_stride[1]; 
  struct sidl_int__array entries_real;
  struct sidl_int__array*entries_tmp = &entries_real;
  int32_t values_lower[1], values_upper[1], values_stride[1]; 
  struct sidl_double__array values_real;
  struct sidl_double__array*values_tmp = &values_real;
  ilower_upper[0] = dim-1;
  sidl_int__array_init(ilower, ilower_tmp, 1, ilower_lower, ilower_upper,
    ilower_stride);
  iupper_upper[0] = dim-1;
  sidl_int__array_init(iupper, iupper_tmp, 1, iupper_lower, iupper_upper,
    iupper_stride);
  entries_upper[0] = nentries-1;
  sidl_int__array_init(entries, entries_tmp, 1, entries_lower, entries_upper,
    entries_stride);
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  return (*self->d_epv->f_AddToBoxValues)(
    self,
    part,
    ilower_tmp,
    iupper_tmp,
    var,
    entries_tmp,
    values_tmp);
}

/*
 * Define symmetry properties for the stencil entries in the
 * matrix.  The boolean argument {\tt symmetric} is applied to
 * stencil entries on part {\tt part} that couple variable {\tt
 * var} to variable {\tt to\_var}.  A value of -1 may be used
 * for {\tt part}, {\tt var}, or {\tt to\_var} to specify
 * ``all''.  For example, if {\tt part} and {\tt to\_var} are
 * set to -1, then the boolean is applied to stencil entries on
 * all parts that couple variable {\tt var} to all other
 * variables.
 * 
 * By default, matrices are assumed to be nonsymmetric.
 * Significant storage savings can be made if the matrix is
 * symmetric.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetSymmetric(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t to_var,
  /* in */ int32_t symmetric)
{
  return (*self->d_epv->f_SetSymmetric)(
    self,
    part,
    var,
    to_var,
    symmetric);
}

/*
 * Define symmetry properties for all non-stencil matrix
 * entries.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetNSSymmetric(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t symmetric)
{
  return (*self->d_epv->f_SetNSSymmetric)(
    self,
    symmetric);
}

/*
 * Set the matrix to be complex.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetComplex(
  /* in */ bHYPRE_SStructParCSRMatrix self)
{
  return (*self->d_epv->f_SetComplex)(
    self);
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_Print(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* filename,
  /* in */ int32_t all)
{
  return (*self->d_epv->f_Print)(
    self,
    filename,
    all);
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetIntParameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value)
{
  return (*self->d_epv->f_SetIntParameter)(
    self,
    name,
    value);
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetDoubleParameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ double value)
{
  return (*self->d_epv->f_SetDoubleParameter)(
    self,
    name,
    value);
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetStringParameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ const char* value)
{
  return (*self->d_epv->f_SetStringParameter)(
    self,
    name,
    value);
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ int32_t* value,
  /* in */ int32_t nvalues)
{
  int32_t value_lower[1], value_upper[1], value_stride[1]; 
  struct sidl_int__array value_real;
  struct sidl_int__array*value_tmp = &value_real;
  value_upper[0] = nvalues-1;
  sidl_int__array_init(value, value_tmp, 1, value_lower, value_upper,
    value_stride);
  return (*self->d_epv->f_SetIntArray1Parameter)(
    self,
    name,
    value_tmp);
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value)
{
  return (*self->d_epv->f_SetIntArray2Parameter)(
    self,
    name,
    value);
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ double* value,
  /* in */ int32_t nvalues)
{
  int32_t value_lower[1], value_upper[1], value_stride[1]; 
  struct sidl_double__array value_real;
  struct sidl_double__array*value_tmp = &value_real;
  value_upper[0] = nvalues-1;
  sidl_double__array_init(value, value_tmp, 1, value_lower, value_upper,
    value_stride);
  return (*self->d_epv->f_SetDoubleArray1Parameter)(
    self,
    name,
    value_tmp);
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value)
{
  return (*self->d_epv->f_SetDoubleArray2Parameter)(
    self,
    name,
    value);
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_GetIntValue(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value)
{
  return (*self->d_epv->f_GetIntValue)(
    self,
    name,
    value);
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_GetDoubleValue(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* out */ double* value)
{
  return (*self->d_epv->f_GetDoubleValue)(
    self,
    name,
    value);
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_Setup(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x)
{
  return (*self->d_epv->f_Setup)(
    self,
    b,
    x);
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_Apply(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x)
{
  return (*self->d_epv->f_Apply)(
    self,
    b,
    x);
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 * 
 */

int32_t
bHYPRE_SStructParCSRMatrix_ApplyAdjoint(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x)
{
  return (*self->d_epv->f_ApplyAdjoint)(
    self,
    b,
    x);
}

void
bHYPRE_SStructParCSRMatrix_Create__sexec(
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  bHYPRE_MPICommunicator mpi_comm;
  bHYPRE_SStructGraph graph;
  bHYPRE_SStructParCSRMatrix _retval;

  /* unpack in and inout argments */

  /* make the call */
  _retval = (_getSEPV()->f_Create)(
    mpi_comm,
    graph);

  /* pack return value */
  /* pack out and inout argments */

}

/*
 * Cast method for interface and class type conversions.
 */

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__cast(
  void* obj)
{
  bHYPRE_SStructParCSRMatrix cast = NULL;

  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.SStructParCSRMatrix",
      (void*)bHYPRE_SStructParCSRMatrix__IHConnect);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_SStructParCSRMatrix) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.SStructParCSRMatrix");
  }

  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_SStructParCSRMatrix__cast2(
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
bHYPRE_SStructParCSRMatrix__exec(
  /* in */ bHYPRE_SStructParCSRMatrix self,
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

struct bHYPRE_SStructParCSRMatrix__smethod {
  const char *d_name;
  void (*d_func)(struct sidl_io_Deserializer__object *,
    struct sidl_io_Serializer__object *);
};

void
bHYPRE_SStructParCSRMatrix__sexec(
        const char* methodName,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs ) { 
  static const struct bHYPRE_SStructParCSRMatrix__smethod s_methods[] = {
    { "Create", bHYPRE_SStructParCSRMatrix_Create__sexec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct bHYPRE_SStructParCSRMatrix__smethod);
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
bHYPRE_SStructParCSRMatrix__getURL(
  /* in */ bHYPRE_SStructParCSRMatrix self)
{
  return (*self->d_epv->f__getURL)(
  self);
}

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructParCSRMatrix__array*)sidl_interface__array_createCol(dimen,
    lower, upper);
}

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructParCSRMatrix__array*)sidl_interface__array_createRow(dimen,
    lower, upper);
}

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_create1d(int32_t len)
{
  return (struct 
    bHYPRE_SStructParCSRMatrix__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructParCSRMatrix* data)
{
  return (struct 
    bHYPRE_SStructParCSRMatrix__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_create2dCol(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_SStructParCSRMatrix__array*)sidl_interface__array_create2dCol(m, n);
}

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_create2dRow(int32_t m, int32_t n)
{
  return (struct 
    bHYPRE_SStructParCSRMatrix__array*)sidl_interface__array_create2dRow(m, n);
}

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_borrow(
  bHYPRE_SStructParCSRMatrix* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct 
    bHYPRE_SStructParCSRMatrix__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_smartCopy(
  struct bHYPRE_SStructParCSRMatrix__array *array)
{
  return (struct bHYPRE_SStructParCSRMatrix__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_SStructParCSRMatrix__array_addRef(
  struct bHYPRE_SStructParCSRMatrix__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_SStructParCSRMatrix__array_deleteRef(
  struct bHYPRE_SStructParCSRMatrix__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get1(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1)
{
  return (bHYPRE_SStructParCSRMatrix)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get2(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_SStructParCSRMatrix)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get3(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_SStructParCSRMatrix)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get4(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_SStructParCSRMatrix)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get5(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_SStructParCSRMatrix)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get6(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_SStructParCSRMatrix)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get7(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_SStructParCSRMatrix)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_SStructParCSRMatrix
bHYPRE_SStructParCSRMatrix__array_get(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t indices[])
{
  return (bHYPRE_SStructParCSRMatrix)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_SStructParCSRMatrix__array_set1(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  bHYPRE_SStructParCSRMatrix const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRMatrix__array_set2(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructParCSRMatrix const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRMatrix__array_set3(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructParCSRMatrix const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRMatrix__array_set4(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructParCSRMatrix const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRMatrix__array_set5(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructParCSRMatrix const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRMatrix__array_set6(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructParCSRMatrix const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRMatrix__array_set7(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructParCSRMatrix const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructParCSRMatrix__array_set(
  struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t indices[],
  bHYPRE_SStructParCSRMatrix const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_SStructParCSRMatrix__array_dimen(
  const struct bHYPRE_SStructParCSRMatrix__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_SStructParCSRMatrix__array_lower(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructParCSRMatrix__array_upper(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructParCSRMatrix__array_length(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructParCSRMatrix__array_stride(
  const struct bHYPRE_SStructParCSRMatrix__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_SStructParCSRMatrix__array_isColumnOrder(
  const struct bHYPRE_SStructParCSRMatrix__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_SStructParCSRMatrix__array_isRowOrder(
  const struct bHYPRE_SStructParCSRMatrix__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_SStructParCSRMatrix__array_copy(
  const struct bHYPRE_SStructParCSRMatrix__array* src,
  struct bHYPRE_SStructParCSRMatrix__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_slice(
  struct bHYPRE_SStructParCSRMatrix__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_SStructParCSRMatrix__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_SStructParCSRMatrix__array*
bHYPRE_SStructParCSRMatrix__array_ensure(
  struct bHYPRE_SStructParCSRMatrix__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_SStructParCSRMatrix__array*)
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
static struct sidl_recursive_mutex_t bHYPRE_SStructParCSRMatrix__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE_SStructParCSRMatrix__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE_SStructParCSRMatrix__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE_SStructParCSRMatrix__mutex )==EDEADLOCK) */
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

static struct bHYPRE_SStructParCSRMatrix__epv 
  s_rem_epv__bhypre_sstructparcsrmatrix;

static struct bHYPRE_MatrixVectorView__epv s_rem_epv__bhypre_matrixvectorview;

static struct bHYPRE_Operator__epv s_rem_epv__bhypre_operator;

static struct bHYPRE_ProblemDefinition__epv s_rem_epv__bhypre_problemdefinition;

static struct bHYPRE_SStructMatrixView__epv s_rem_epv__bhypre_sstructmatrixview;

static struct bHYPRE_SStruct_MatrixVectorView__epv 
  s_rem_epv__bhypre_sstruct_matrixvectorview;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;

/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_bHYPRE_SStructParCSRMatrix__cast(
struct bHYPRE_SStructParCSRMatrix__object* self,
const char* name)
{
  void* cast = NULL;

  struct bHYPRE_SStructParCSRMatrix__object* s0;
  struct sidl_BaseClass__object* s1;
   s0 =                                     self;
   s1 =                                     &s0->d_sidl_baseclass;

  if (!strcmp(name, "bHYPRE.SStructParCSRMatrix")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "bHYPRE.MatrixVectorView")) {
    cast = (void*) &s0->d_bhypre_matrixvectorview;
  } else if (!strcmp(name, "bHYPRE.Operator")) {
    cast = (void*) &s0->d_bhypre_operator;
  } else if (!strcmp(name, "bHYPRE.ProblemDefinition")) {
    cast = (void*) &s0->d_bhypre_problemdefinition;
  } else if (!strcmp(name, "bHYPRE.SStructMatrixView")) {
    cast = (void*) &s0->d_bhypre_sstructmatrixview;
  } else if (!strcmp(name, "bHYPRE.SStruct_MatrixVectorView")) {
    cast = (void*) &s0->d_bhypre_sstruct_matrixvectorview;
  } else if (!strcmp(name, "sidl.BaseClass")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s1->d_sidl_baseinterface;
  }
  else if(bHYPRE_SStructParCSRMatrix_isType(self, name)) {
    void* (*func)(sidl_rmi_InstanceHandle) = 
      (void* (*)(sidl_rmi_InstanceHandle)) 
      sidl_rmi_ConnectRegistry_getConnect(name);
    cast =  (*func)((sidl_rmi_InstanceHandle)self->d_data);
  }

  return cast;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE_SStructParCSRMatrix__delete(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_bHYPRE_SStructParCSRMatrix__getURL(
  struct bHYPRE_SStructParCSRMatrix__object* self)
{
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getURL(conn, &_ex);
  }
  return NULL;
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_bHYPRE_SStructParCSRMatrix__exec(
  struct bHYPRE_SStructParCSRMatrix__object* self,
  const char* methodName,
  sidl_io_Deserializer inArgs,
  sidl_io_Serializer outArgs)
{
}

/* REMOTE METHOD STUB:addRef */
static void
remote_bHYPRE_SStructParCSRMatrix_addRef(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_bHYPRE_SStructParCSRMatrix_deleteRef(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */)
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
remote_bHYPRE_SStructParCSRMatrix_isSame(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ struct sidl_BaseInterface__object* iobj)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:queryInt */
static struct sidl_BaseInterface__object*
remote_bHYPRE_SStructParCSRMatrix_queryInt(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ const char* name)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_bHYPRE_SStructParCSRMatrix_isType(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
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
remote_bHYPRE_SStructParCSRMatrix_getClassInfo(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */)
{
  /* FIXME  need to think through all of these special cases */
  return 0;
}

/* REMOTE METHOD STUB:SetCommunicator */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetCommunicator(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetCommunicator", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "mpi_comm",
    bHYPRE_MPICommunicator__getURL(mpi_comm), _ex2);

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

/* REMOTE METHOD STUB:Initialize */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_Initialize(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "Initialize", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

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

/* REMOTE METHOD STUB:Assemble */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_Assemble(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "Assemble", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

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

/* REMOTE METHOD STUB:GetObject */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_GetObject(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* out */ struct sidl_BaseInterface__object** A)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "GetObject", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackString( _rsvp, "A", A, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:SetGraph */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetGraph(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ struct bHYPRE_SStructGraph__object* graph)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetGraph", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "graph",
    bHYPRE_SStructGraph__getURL(graph), _ex2);

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

/* REMOTE METHOD STUB:SetValues */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetValues(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ int32_t part,
  /* in */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ struct sidl_int__array* entries,
  /* in */ struct sidl_double__array* values)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetValues", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "part", part, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "var", var, _ex2);

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

/* REMOTE METHOD STUB:SetBoxValues */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetBoxValues(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ int32_t part,
  /* in */ struct sidl_int__array* ilower,
  /* in */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in */ struct sidl_int__array* entries,
  /* in */ struct sidl_double__array* values)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetBoxValues", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "part", part, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "var", var, _ex2);

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

/* REMOTE METHOD STUB:AddToValues */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_AddToValues(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ int32_t part,
  /* in */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ struct sidl_int__array* entries,
  /* in */ struct sidl_double__array* values)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "AddToValues", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "part", part, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "var", var, _ex2);

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

/* REMOTE METHOD STUB:AddToBoxValues */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_AddToBoxValues(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ int32_t part,
  /* in */ struct sidl_int__array* ilower,
  /* in */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in */ struct sidl_int__array* entries,
  /* in */ struct sidl_double__array* values)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "AddToBoxValues", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "part", part, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "var", var, _ex2);

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

/* REMOTE METHOD STUB:SetSymmetric */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetSymmetric(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t to_var,
  /* in */ int32_t symmetric)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetSymmetric", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "part", part, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "var", var, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "to_var", to_var, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "symmetric", symmetric, _ex2);

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

/* REMOTE METHOD STUB:SetNSSymmetric */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetNSSymmetric(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ int32_t symmetric)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetNSSymmetric", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packInt( _inv, "symmetric", symmetric, _ex2);

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

/* REMOTE METHOD STUB:SetComplex */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetComplex(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetComplex", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

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

/* REMOTE METHOD STUB:Print */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_Print(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ const char* filename,
  /* in */ int32_t all)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "Print", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "filename", filename, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "all", all, _ex2);

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

/* REMOTE METHOD STUB:SetIntParameter */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetIntParameter(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ int32_t value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetIntParameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);
  sidl_rmi_Invocation_packInt( _inv, "value", value, _ex2);

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

/* REMOTE METHOD STUB:SetDoubleParameter */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetDoubleParameter(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ double value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetDoubleParameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);
  sidl_rmi_Invocation_packDouble( _inv, "value", value, _ex2);

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

/* REMOTE METHOD STUB:SetStringParameter */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetStringParameter(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ const char* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetStringParameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);
  sidl_rmi_Invocation_packString( _inv, "value", value, _ex2);

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

/* REMOTE METHOD STUB:SetIntArray1Parameter */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetIntArray1Parameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

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

/* REMOTE METHOD STUB:SetIntArray2Parameter */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ struct sidl_int__array* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetIntArray2Parameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

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

/* REMOTE METHOD STUB:SetDoubleArray1Parameter */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetDoubleArray1Parameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

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

/* REMOTE METHOD STUB:SetDoubleArray2Parameter */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ const char* name,
  /* in */ struct sidl_double__array* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "SetDoubleArray2Parameter", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

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

/* REMOTE METHOD STUB:GetIntValue */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_GetIntValue(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ const char* name,
  /* out */ int32_t* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "GetIntValue", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackInt( _rsvp, "value", value, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:GetDoubleValue */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_GetDoubleValue(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ const char* name,
  /* out */ double* value)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "GetDoubleValue", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */
  sidl_rmi_Invocation_packString( _inv, "name", name, _ex2);

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackDouble( _rsvp, "value", value, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:Setup */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_Setup(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ struct bHYPRE_Vector__object* b,
  /* in */ struct bHYPRE_Vector__object* x)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "Setup", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

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

/* REMOTE METHOD STUB:Apply */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_Apply(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ struct bHYPRE_Vector__object* b,
  /* inout */ struct bHYPRE_Vector__object** x)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "Apply", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackString( _rsvp, "x", x, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE METHOD STUB:ApplyAdjoint */
static int32_t
remote_bHYPRE_SStructParCSRMatrix_ApplyAdjoint(
  /* in */ struct bHYPRE_SStructParCSRMatrix__object* self /* TLD */,
  /* in */ struct bHYPRE_Vector__object* b,
  /* inout */ struct bHYPRE_Vector__object** x)
{
  sidl_BaseInterface _ex = NULL;
  sidl_BaseInterface *_ex2 =&_ex;
  /* initialize a new invocation */
  sidl_rmi_InstanceHandle _conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
    "ApplyAdjoint", _ex2 );
  sidl_rmi_Response _rsvp = NULL;
  int32_t _retval;

  /* pack in and inout arguments */

  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex2);

  /* extract return value */
  sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex2);

  /* unpack out and inout arguments */
  sidl_rmi_Response_unpackString( _rsvp, "x", x, _ex2);

  /* cleanup and return */
  sidl_rmi_Response_done(_rsvp, _ex2);
  sidl_rmi_Invocation_deleteRef(_inv);
  sidl_rmi_Response_deleteRef(_rsvp);
  return _retval;
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void bHYPRE_SStructParCSRMatrix__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE_SStructParCSRMatrix__epv*      epv = 
    &s_rem_epv__bhypre_sstructparcsrmatrix;
  struct bHYPRE_MatrixVectorView__epv*         e0  = 
    &s_rem_epv__bhypre_matrixvectorview;
  struct bHYPRE_Operator__epv*                 e1  = 
    &s_rem_epv__bhypre_operator;
  struct bHYPRE_ProblemDefinition__epv*        e2  = 
    &s_rem_epv__bhypre_problemdefinition;
  struct bHYPRE_SStructMatrixView__epv*        e3  = 
    &s_rem_epv__bhypre_sstructmatrixview;
  struct bHYPRE_SStruct_MatrixVectorView__epv* e4  = 
    &s_rem_epv__bhypre_sstruct_matrixvectorview;
  struct sidl_BaseClass__epv*                  e5  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*              e6  = 
    &s_rem_epv__sidl_baseinterface;

  epv->f__cast                         = 
    remote_bHYPRE_SStructParCSRMatrix__cast;
  epv->f__delete                       = 
    remote_bHYPRE_SStructParCSRMatrix__delete;
  epv->f__exec                         = 
    remote_bHYPRE_SStructParCSRMatrix__exec;
  epv->f__getURL                       = 
    remote_bHYPRE_SStructParCSRMatrix__getURL;
  epv->f__ctor                         = NULL;
  epv->f__dtor                         = NULL;
  epv->f_addRef                        = 
    remote_bHYPRE_SStructParCSRMatrix_addRef;
  epv->f_deleteRef                     = 
    remote_bHYPRE_SStructParCSRMatrix_deleteRef;
  epv->f_isSame                        = 
    remote_bHYPRE_SStructParCSRMatrix_isSame;
  epv->f_queryInt                      = 
    remote_bHYPRE_SStructParCSRMatrix_queryInt;
  epv->f_isType                        = 
    remote_bHYPRE_SStructParCSRMatrix_isType;
  epv->f_getClassInfo                  = 
    remote_bHYPRE_SStructParCSRMatrix_getClassInfo;
  epv->f_SetCommunicator               = 
    remote_bHYPRE_SStructParCSRMatrix_SetCommunicator;
  epv->f_Initialize                    = 
    remote_bHYPRE_SStructParCSRMatrix_Initialize;
  epv->f_Assemble                      = 
    remote_bHYPRE_SStructParCSRMatrix_Assemble;
  epv->f_GetObject                     = 
    remote_bHYPRE_SStructParCSRMatrix_GetObject;
  epv->f_SetGraph                      = 
    remote_bHYPRE_SStructParCSRMatrix_SetGraph;
  epv->f_SetValues                     = 
    remote_bHYPRE_SStructParCSRMatrix_SetValues;
  epv->f_SetBoxValues                  = 
    remote_bHYPRE_SStructParCSRMatrix_SetBoxValues;
  epv->f_AddToValues                   = 
    remote_bHYPRE_SStructParCSRMatrix_AddToValues;
  epv->f_AddToBoxValues                = 
    remote_bHYPRE_SStructParCSRMatrix_AddToBoxValues;
  epv->f_SetSymmetric                  = 
    remote_bHYPRE_SStructParCSRMatrix_SetSymmetric;
  epv->f_SetNSSymmetric                = 
    remote_bHYPRE_SStructParCSRMatrix_SetNSSymmetric;
  epv->f_SetComplex                    = 
    remote_bHYPRE_SStructParCSRMatrix_SetComplex;
  epv->f_Print                         = 
    remote_bHYPRE_SStructParCSRMatrix_Print;
  epv->f_SetIntParameter               = 
    remote_bHYPRE_SStructParCSRMatrix_SetIntParameter;
  epv->f_SetDoubleParameter            = 
    remote_bHYPRE_SStructParCSRMatrix_SetDoubleParameter;
  epv->f_SetStringParameter            = 
    remote_bHYPRE_SStructParCSRMatrix_SetStringParameter;
  epv->f_SetIntArray1Parameter         = 
    remote_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter;
  epv->f_SetIntArray2Parameter         = 
    remote_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter;
  epv->f_SetDoubleArray1Parameter      = 
    remote_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter;
  epv->f_SetDoubleArray2Parameter      = 
    remote_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter;
  epv->f_GetIntValue                   = 
    remote_bHYPRE_SStructParCSRMatrix_GetIntValue;
  epv->f_GetDoubleValue                = 
    remote_bHYPRE_SStructParCSRMatrix_GetDoubleValue;
  epv->f_Setup                         = 
    remote_bHYPRE_SStructParCSRMatrix_Setup;
  epv->f_Apply                         = 
    remote_bHYPRE_SStructParCSRMatrix_Apply;
  epv->f_ApplyAdjoint                  = 
    remote_bHYPRE_SStructParCSRMatrix_ApplyAdjoint;

  e0->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e0->f__delete         = (void (*)(void*)) epv->f__delete;
  e0->f__exec           = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e0->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e0->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e0->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt        = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e0->f_isType          = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e0->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e0->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e0->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;

  e1->f__cast                    = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete                  = (void (*)(void*)) epv->f__delete;
  e1->f__exec                    = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e1->f_addRef                   = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef                = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame                   = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(void*,const char*)) epv->f_queryInt;
  e1->f_isType                   = (sidl_bool (*)(void*,
    const char*)) epv->f_isType;
  e1->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e1->f_SetCommunicator          = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e1->f_SetIntParameter          = (int32_t (*)(void*,const char*,
    int32_t)) epv->f_SetIntParameter;
  e1->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,
    double)) epv->f_SetDoubleParameter;
  e1->f_SetStringParameter       = (int32_t (*)(void*,const char*,
    const char*)) epv->f_SetStringParameter;
  e1->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray1Parameter;
  e1->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,
    struct sidl_int__array*)) epv->f_SetIntArray2Parameter;
  e1->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray1Parameter;
  e1->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,
    struct sidl_double__array*)) epv->f_SetDoubleArray2Parameter;
  e1->f_GetIntValue              = (int32_t (*)(void*,const char*,
    int32_t*)) epv->f_GetIntValue;
  e1->f_GetDoubleValue           = (int32_t (*)(void*,const char*,
    double*)) epv->f_GetDoubleValue;
  e1->f_Setup                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object*)) epv->f_Setup;
  e1->f_Apply                    = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,struct bHYPRE_Vector__object**)) epv->f_Apply;
  e1->f_ApplyAdjoint             = (int32_t (*)(void*,
    struct bHYPRE_Vector__object*,
    struct bHYPRE_Vector__object**)) epv->f_ApplyAdjoint;

  e2->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete         = (void (*)(void*)) epv->f__delete;
  e2->f__exec           = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e2->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt        = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType          = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e2->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e2->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e2->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;

  e3->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete         = (void (*)(void*)) epv->f__delete;
  e3->f__exec           = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e3->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt        = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType          = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e3->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e3->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e3->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e3->f_GetObject       = (int32_t (*)(void*,
    struct sidl_BaseInterface__object**)) epv->f_GetObject;
  e3->f_SetGraph        = (int32_t (*)(void*,
    struct bHYPRE_SStructGraph__object*)) epv->f_SetGraph;
  e3->f_SetValues       = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    int32_t,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_SetValues;
  e3->f_SetBoxValues    = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,int32_t,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_SetBoxValues;
  e3->f_AddToValues     = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    int32_t,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_AddToValues;
  e3->f_AddToBoxValues  = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,int32_t,struct sidl_int__array*,
    struct sidl_double__array*)) epv->f_AddToBoxValues;
  e3->f_SetSymmetric    = (int32_t (*)(void*,int32_t,int32_t,int32_t,
    int32_t)) epv->f_SetSymmetric;
  e3->f_SetNSSymmetric  = (int32_t (*)(void*,int32_t)) epv->f_SetNSSymmetric;
  e3->f_SetComplex      = (int32_t (*)(void*)) epv->f_SetComplex;
  e3->f_Print           = (int32_t (*)(void*,const char*,int32_t)) epv->f_Print;

  e4->f__cast           = (void* (*)(void*,const char*)) epv->f__cast;
  e4->f__delete         = (void (*)(void*)) epv->f__delete;
  e4->f__exec           = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e4->f_addRef          = (void (*)(void*)) epv->f_addRef;
  e4->f_deleteRef       = (void (*)(void*)) epv->f_deleteRef;
  e4->f_isSame          = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e4->f_queryInt        = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e4->f_isType          = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e4->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e4->f_SetCommunicator = (int32_t (*)(void*,
    struct bHYPRE_MPICommunicator__object*)) epv->f_SetCommunicator;
  e4->f_Initialize      = (int32_t (*)(void*)) epv->f_Initialize;
  e4->f_Assemble        = (int32_t (*)(void*)) epv->f_Assemble;
  e4->f_GetObject       = (int32_t (*)(void*,
    struct sidl_BaseInterface__object**)) epv->f_GetObject;

  e5->f__cast        = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e5->f__delete      = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e5->f__exec        = (void (*)(struct sidl_BaseClass__object*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e5->f_addRef       = (void (*)(struct sidl_BaseClass__object*)) epv->f_addRef;
  e5->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e5->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e5->f_queryInt     = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e5->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e5->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e6->f__cast        = (void* (*)(void*,const char*)) epv->f__cast;
  e6->f__delete      = (void (*)(void*)) epv->f__delete;
  e6->f__exec        = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e6->f_addRef       = (void (*)(void*)) epv->f_addRef;
  e6->f_deleteRef    = (void (*)(void*)) epv->f_deleteRef;
  e6->f_isSame       = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e6->f_queryInt     = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e6->f_isType       = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e6->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct bHYPRE_SStructParCSRMatrix__object*
bHYPRE_SStructParCSRMatrix__remoteConnect(const char *url,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_SStructParCSRMatrix__object* self;

  struct bHYPRE_SStructParCSRMatrix__object* s0;
  struct sidl_BaseClass__object* s1;

  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_connectInstance(url, _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_SStructParCSRMatrix__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRMatrix__object));

   s0 =                                     self;
   s1 =                                     &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructParCSRMatrix__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_operator.d_epv    = &s_rem_epv__bhypre_operator;
  s0->d_bhypre_operator.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_sstructmatrixview.d_epv    = 
    &s_rem_epv__bhypre_sstructmatrixview;
  s0->d_bhypre_sstructmatrixview.d_object = (void*) self;

  s0->d_bhypre_sstruct_matrixvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstruct_matrixvectorview;
  s0->d_bhypre_sstruct_matrixvectorview.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__bhypre_sstructparcsrmatrix;

  self->d_data = (void*) instance;
  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructParCSRMatrix__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;


  return self;
}
/* Create an instance that uses an already existing InstanceHandel to connect 
  to an existing remote object. */
static struct bHYPRE_SStructParCSRMatrix__object*
bHYPRE_SStructParCSRMatrix__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_SStructParCSRMatrix__object* self;

  struct bHYPRE_SStructParCSRMatrix__object* s0;
  struct sidl_BaseClass__object* s1;

  self =
    (struct bHYPRE_SStructParCSRMatrix__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRMatrix__object));

   s0 =                                     self;
   s1 =                                     &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructParCSRMatrix__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_operator.d_epv    = &s_rem_epv__bhypre_operator;
  s0->d_bhypre_operator.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_sstructmatrixview.d_epv    = 
    &s_rem_epv__bhypre_sstructmatrixview;
  s0->d_bhypre_sstructmatrixview.d_object = (void*) self;

  s0->d_bhypre_sstruct_matrixvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstruct_matrixvectorview;
  s0->d_bhypre_sstruct_matrixvectorview.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__bhypre_sstructparcsrmatrix;

  self->d_data = (void*) instance;

  sidl_rmi_InstanceHandle_addRef(instance);
  return self;
}
/* REMOTE: generate remote instance given URL string. */
static struct bHYPRE_SStructParCSRMatrix__object*
bHYPRE_SStructParCSRMatrix__remote(const char *url, sidl_BaseInterface *_ex)
{
  struct bHYPRE_SStructParCSRMatrix__object* self;

  struct bHYPRE_SStructParCSRMatrix__object* s0;
  struct sidl_BaseClass__object* s1;

  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_createInstance(url, "bHYPRE.SStructParCSRMatrix",
    _ex );
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_SStructParCSRMatrix__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRMatrix__object));

   s0 =                                     self;
   s1 =                                     &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructParCSRMatrix__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) instance;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_operator.d_epv    = &s_rem_epv__bhypre_operator;
  s0->d_bhypre_operator.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_sstructmatrixview.d_epv    = 
    &s_rem_epv__bhypre_sstructmatrixview;
  s0->d_bhypre_sstructmatrixview.d_object = (void*) self;

  s0->d_bhypre_sstruct_matrixvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstruct_matrixvectorview;
  s0->d_bhypre_sstruct_matrixvectorview.d_object = (void*) self;

  s0->d_data = (void*) instance;
  s0->d_epv  = &s_rem_epv__bhypre_sstructparcsrmatrix;

  self->d_data = (void*) instance;

  return self;
}
