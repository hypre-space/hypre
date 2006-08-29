/*
 * File:          bHYPRE_SStructGrid_Stub.c
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_SStructGrid.h"
#include "bHYPRE_SStructGrid_IOR.h"
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

static const struct bHYPRE_SStructGrid__external *_externals = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct bHYPRE_SStructGrid__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _externals = bHYPRE_SStructGrid__externals();
#else
  _externals = (struct 
    bHYPRE_SStructGrid__external*)sidl_dynamicLoadIOR("bHYPRE.SStructGrid",
    "bHYPRE_SStructGrid__externals") ;
  sidl_checkIORVersion("bHYPRE.SStructGrid", _externals->d_ior_major_version,
    _externals->d_ior_minor_version, 0, 10);
#endif
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())

/*
 * Hold pointer to static entry point vector
 */

static const struct bHYPRE_SStructGrid__sepv *_sepv = NULL;
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

bHYPRE_SStructGrid
bHYPRE_SStructGrid__create(sidl_BaseInterface* _ex)
{
  return (*(_getExternals()->createObject))(NULL,_ex);
}

/**
 * Wraps up the private data struct pointer (struct bHYPRE_SStructGrid__data) passed in rather than running the constructor.
 */
bHYPRE_SStructGrid
bHYPRE_SStructGrid__wrapObj(void* data, sidl_BaseInterface* _ex)
{
  return (*(_getExternals()->createObject))(data,_ex);
}

static bHYPRE_SStructGrid bHYPRE_SStructGrid__remoteCreate(const char* url,
  sidl_BaseInterface *_ex);
/*
 * RMI constructor function for the class.
 */

bHYPRE_SStructGrid
bHYPRE_SStructGrid__createRemote(const char* url, sidl_BaseInterface *_ex)
{
  return bHYPRE_SStructGrid__remoteCreate(url, _ex);
}

static struct bHYPRE_SStructGrid__object* 
  bHYPRE_SStructGrid__remoteConnect(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
static struct bHYPRE_SStructGrid__object* bHYPRE_SStructGrid__IHConnect(struct 
  sidl_rmi_InstanceHandle__object* instance, sidl_BaseInterface *_ex);
/*
 * RMI connector function for the class.
 */

bHYPRE_SStructGrid
bHYPRE_SStructGrid__connect(const char* url, sidl_BaseInterface *_ex)
{
  return bHYPRE_SStructGrid__remoteConnect(url, TRUE, _ex);
}

/*
 * Set the number of dimensions {\tt ndim} and the number of
 * structured parts {\tt nparts}.
 */

bHYPRE_SStructGrid
bHYPRE_SStructGrid_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t ndim,
  /* in */ int32_t nparts,
  /* out */ sidl_BaseInterface *_ex)
{
  return (_getSEPV()->f_Create)(
    mpi_comm,
    ndim,
    nparts,
    _ex);
}

/*
 * Method:  SetNumDimParts[]
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructGrid_SetNumDimParts(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t ndim,
  /* in */ int32_t nparts,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_SetNumDimParts)(
    self,
    ndim,
    nparts,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Method:  SetCommunicator[]
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructGrid_SetCommunicator(
  /* in */ bHYPRE_SStructGrid self,
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
 * Set the extents for a box on a structured part of the grid.
 */

int32_t
bHYPRE_SStructGrid_SetExtents(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
{
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1]; 
  struct sidl_int__array ilower_real;
  struct sidl_int__array*ilower_tmp = &ilower_real;
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1]; 
  struct sidl_int__array iupper_real;
  struct sidl_int__array*iupper_tmp = &iupper_real;
  ilower_upper[0] = dim-1;
  sidl_int__array_init(ilower, ilower_tmp, 1, ilower_lower, ilower_upper,
    ilower_stride);
  iupper_upper[0] = dim-1;
  sidl_int__array_init(iupper, iupper_tmp, 1, iupper_lower, iupper_upper,
    iupper_stride);
  return (*self->d_epv->f_SetExtents)(
    self,
    part,
    ilower_tmp,
    iupper_tmp,
    _ex);
}

/*
 * Describe the variables that live on a structured part of the
 * grid.  Input: part number, variable number, total number of
 * variables on that part (needed for memory allocation),
 * variable type.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructGrid_SetVariable(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t nvars,
  /* in */ enum bHYPRE_SStructVariable__enum vartype,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_SetVariable)(
    self,
    part,
    var,
    nvars,
    vartype,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Describe additional variables that live at a particular
 * index.  These variables are appended to the array of
 * variables set in {\tt SetVariables}, and are referenced as
 * such.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructGrid_AddVariable(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ enum bHYPRE_SStructVariable__enum vartype,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  int32_t index_lower[1], index_upper[1], index_stride[1]; 
  struct sidl_int__array index_real;
  struct sidl_int__array*index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  return (*self->d_epv->f_AddVariable)(
    self,
    part,
    index_tmp,
    var,
    vartype,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Describe how regions just outside of a part relate to other
 * parts.  This is done a box at a time.
 * 
 * The indexes {\tt ilower} and {\tt iupper} map directly to the
 * indexes {\tt nbor\_ilower} and {\tt nbor\_iupper}.  Although,
 * it is required that indexes increase from {\tt ilower} to
 * {\tt iupper}, indexes may increase and/or decrease from {\tt
 * nbor\_ilower} to {\tt nbor\_iupper}.
 * 
 * The {\tt index\_map} describes the mapping of indexes 0, 1,
 * and 2 on part {\tt part} to the corresponding indexes on part
 * {\tt nbor\_part}.  For example, triple (1, 2, 0) means that
 * indexes 0, 1, and 2 on part {\tt part} map to indexes 1, 2,
 * and 0 on part {\tt nbor\_part}, respectively.
 * 
 * NOTE: All parts related to each other via this routine must
 * have an identical list of variables and variable types.  For
 * example, if part 0 has only two variables on it, a cell
 * centered variable and a node centered variable, and we
 * declare part 1 to be a neighbor of part 0, then part 1 must
 * also have only two variables on it, and they must be of type
 * cell and node.
 */

int32_t
bHYPRE_SStructGrid_SetNeighborBox(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t nbor_part,
  /* in rarray[dim] */ int32_t* nbor_ilower,
  /* in rarray[dim] */ int32_t* nbor_iupper,
  /* in rarray[dim] */ int32_t* index_map,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
{
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1]; 
  struct sidl_int__array ilower_real;
  struct sidl_int__array*ilower_tmp = &ilower_real;
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1]; 
  struct sidl_int__array iupper_real;
  struct sidl_int__array*iupper_tmp = &iupper_real;
  int32_t nbor_ilower_lower[1], nbor_ilower_upper[1], nbor_ilower_stride[1]; 
  struct sidl_int__array nbor_ilower_real;
  struct sidl_int__array*nbor_ilower_tmp = &nbor_ilower_real;
  int32_t nbor_iupper_lower[1], nbor_iupper_upper[1], nbor_iupper_stride[1]; 
  struct sidl_int__array nbor_iupper_real;
  struct sidl_int__array*nbor_iupper_tmp = &nbor_iupper_real;
  int32_t index_map_lower[1], index_map_upper[1], index_map_stride[1]; 
  struct sidl_int__array index_map_real;
  struct sidl_int__array*index_map_tmp = &index_map_real;
  ilower_upper[0] = dim-1;
  sidl_int__array_init(ilower, ilower_tmp, 1, ilower_lower, ilower_upper,
    ilower_stride);
  iupper_upper[0] = dim-1;
  sidl_int__array_init(iupper, iupper_tmp, 1, iupper_lower, iupper_upper,
    iupper_stride);
  nbor_ilower_upper[0] = dim-1;
  sidl_int__array_init(nbor_ilower, nbor_ilower_tmp, 1, nbor_ilower_lower,
    nbor_ilower_upper, nbor_ilower_stride);
  nbor_iupper_upper[0] = dim-1;
  sidl_int__array_init(nbor_iupper, nbor_iupper_tmp, 1, nbor_iupper_lower,
    nbor_iupper_upper, nbor_iupper_stride);
  index_map_upper[0] = dim-1;
  sidl_int__array_init(index_map, index_map_tmp, 1, index_map_lower,
    index_map_upper, index_map_stride);
  return (*self->d_epv->f_SetNeighborBox)(
    self,
    part,
    ilower_tmp,
    iupper_tmp,
    nbor_part,
    nbor_ilower_tmp,
    nbor_iupper_tmp,
    index_map_tmp,
    _ex);
}

/*
 * Add an unstructured part to the grid.  The variables in the
 * unstructured part of the grid are referenced by a global rank
 * between 0 and the total number of unstructured variables
 * minus one.  Each process owns some unique consecutive range
 * of variables, defined by {\tt ilower} and {\tt iupper}.
 * 
 * NOTE: This is just a placeholder.  This part of the interface
 * is not finished.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructGrid_AddUnstructuredPart(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  return (*self->d_epv->f_AddUnstructuredPart)(
    self,
    ilower,
    iupper,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * (Optional) Set periodic for a particular part.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructGrid_SetPeriodic(
  /* in */ bHYPRE_SStructGrid self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* periodic,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
#if SIDL_C_INLINE_REPEAT_DEFN
{
  int32_t periodic_lower[1], periodic_upper[1], periodic_stride[1]; 
  struct sidl_int__array periodic_real;
  struct sidl_int__array*periodic_tmp = &periodic_real;
  periodic_upper[0] = dim-1;
  sidl_int__array_init(periodic, periodic_tmp, 1, periodic_lower,
    periodic_upper, periodic_stride);
  return (*self->d_epv->f_SetPeriodic)(
    self,
    part,
    periodic_tmp,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Setting ghost in the sgrids.
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructGrid_SetNumGhost(
  /* in */ bHYPRE_SStructGrid self,
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
    self,
    num_ghost_tmp,
    _ex);
}
#else /* ISO C 1999 inline semantics */
;
#endif /* SIDL_C_INLINE_REPEAT_DEFN */

/*
 * Method:  Assemble[]
 */

SIDL_C_INLINE_DEFN
int32_t
bHYPRE_SStructGrid_Assemble(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid_addRef(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid_deleteRef(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid_isSame(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid_isType(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid_getClassInfo(
  /* in */ bHYPRE_SStructGrid self,
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

bHYPRE_SStructGrid
bHYPRE_SStructGrid__cast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  bHYPRE_SStructGrid cast = NULL;

  if(!connect_loaded) {
    connect_loaded = 1;
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.SStructGrid",
      (void*)bHYPRE_SStructGrid__IHConnect,_ex);SIDL_CHECK(*_ex);
  }
  if (obj != NULL) {
    sidl_BaseInterface base = (sidl_BaseInterface) obj;
    cast = (bHYPRE_SStructGrid) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.SStructGrid", _ex); SIDL_CHECK(*_ex);
  }

  EXIT:
  return cast;
}

/*
 * String cast method for interface and class type conversions.
 */

void*
bHYPRE_SStructGrid__cast2(
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
bHYPRE_SStructGrid__exec(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid__getURL(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid__raddRef(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid__set_hooks_static(
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
bHYPRE_SStructGrid__set_hooks(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid__isRemote(
  /* in */ bHYPRE_SStructGrid self,
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
bHYPRE_SStructGrid__isLocal(
  /* in */ bHYPRE_SStructGrid self,
  /* out */ sidl_BaseInterface *_ex)
{
  return !bHYPRE_SStructGrid__isRemote(self,_ex);
}

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructGrid__array*)sidl_interface__array_createCol(dimen, lower,
    upper);
}

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[])
{
  return (struct 
    bHYPRE_SStructGrid__array*)sidl_interface__array_createRow(dimen, lower,
    upper);
}

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create1d(int32_t len)
{
  return (struct bHYPRE_SStructGrid__array*)sidl_interface__array_create1d(len);
}

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructGrid* data)
{
  return (struct 
    bHYPRE_SStructGrid__array*)sidl_interface__array_create1dInit(len,
    (struct sidl_BaseInterface__object **)data);
}

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create2dCol(int32_t m, int32_t n)
{
  return (struct bHYPRE_SStructGrid__array*)sidl_interface__array_create2dCol(m,
    n);
}

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_create2dRow(int32_t m, int32_t n)
{
  return (struct bHYPRE_SStructGrid__array*)sidl_interface__array_create2dRow(m,
    n);
}

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_borrow(
  bHYPRE_SStructGrid* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[])
{
  return (struct bHYPRE_SStructGrid__array*)sidl_interface__array_borrow(
    (struct sidl_BaseInterface__object **)
    firstElement, dimen, lower, upper, stride);
}

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_smartCopy(
  struct bHYPRE_SStructGrid__array *array)
{
  return (struct bHYPRE_SStructGrid__array*)
    sidl_interface__array_smartCopy((struct sidl_interface__array *)array);
}

void
bHYPRE_SStructGrid__array_addRef(
  struct bHYPRE_SStructGrid__array* array)
{
  sidl_interface__array_addRef((struct sidl_interface__array *)array);
}

void
bHYPRE_SStructGrid__array_deleteRef(
  struct bHYPRE_SStructGrid__array* array)
{
  sidl_interface__array_deleteRef((struct sidl_interface__array *)array);
}

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get1(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1)
{
  return (bHYPRE_SStructGrid)
    sidl_interface__array_get1((const struct sidl_interface__array *)array
    , i1);
}

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get2(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2)
{
  return (bHYPRE_SStructGrid)
    sidl_interface__array_get2((const struct sidl_interface__array *)array
    , i1, i2);
}

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get3(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3)
{
  return (bHYPRE_SStructGrid)
    sidl_interface__array_get3((const struct sidl_interface__array *)array
    , i1, i2, i3);
}

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get4(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4)
{
  return (bHYPRE_SStructGrid)
    sidl_interface__array_get4((const struct sidl_interface__array *)array
    , i1, i2, i3, i4);
}

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get5(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5)
{
  return (bHYPRE_SStructGrid)
    sidl_interface__array_get5((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5);
}

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get6(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6)
{
  return (bHYPRE_SStructGrid)
    sidl_interface__array_get6((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6);
}

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get7(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7)
{
  return (bHYPRE_SStructGrid)
    sidl_interface__array_get7((const struct sidl_interface__array *)array
    , i1, i2, i3, i4, i5, i6, i7);
}

bHYPRE_SStructGrid
bHYPRE_SStructGrid__array_get(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t indices[])
{
  return (bHYPRE_SStructGrid)
    sidl_interface__array_get((const struct sidl_interface__array *)array,
      indices);
}

void
bHYPRE_SStructGrid__array_set1(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  bHYPRE_SStructGrid const value)
{
  sidl_interface__array_set1((struct sidl_interface__array *)array
  , i1, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructGrid__array_set2(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructGrid const value)
{
  sidl_interface__array_set2((struct sidl_interface__array *)array
  , i1, i2, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructGrid__array_set3(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructGrid const value)
{
  sidl_interface__array_set3((struct sidl_interface__array *)array
  , i1, i2, i3, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructGrid__array_set4(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructGrid const value)
{
  sidl_interface__array_set4((struct sidl_interface__array *)array
  , i1, i2, i3, i4, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructGrid__array_set5(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructGrid const value)
{
  sidl_interface__array_set5((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructGrid__array_set6(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructGrid const value)
{
  sidl_interface__array_set6((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructGrid__array_set7(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructGrid const value)
{
  sidl_interface__array_set7((struct sidl_interface__array *)array
  , i1, i2, i3, i4, i5, i6, i7, (struct sidl_BaseInterface__object *)value);
}

void
bHYPRE_SStructGrid__array_set(
  struct bHYPRE_SStructGrid__array* array,
  const int32_t indices[],
  bHYPRE_SStructGrid const value)
{
  sidl_interface__array_set((struct sidl_interface__array *)array, indices,
    (struct sidl_BaseInterface__object *)value);
}

int32_t
bHYPRE_SStructGrid__array_dimen(
  const struct bHYPRE_SStructGrid__array* array)
{
  return sidl_interface__array_dimen((struct sidl_interface__array *)array);
}

int32_t
bHYPRE_SStructGrid__array_lower(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t ind)
{
  return sidl_interface__array_lower((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructGrid__array_upper(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t ind)
{
  return sidl_interface__array_upper((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructGrid__array_length(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t ind)
{
  return sidl_interface__array_length((struct sidl_interface__array *)array,
    ind);
}

int32_t
bHYPRE_SStructGrid__array_stride(
  const struct bHYPRE_SStructGrid__array* array,
  const int32_t ind)
{
  return sidl_interface__array_stride((struct sidl_interface__array *)array,
    ind);
}

int
bHYPRE_SStructGrid__array_isColumnOrder(
  const struct bHYPRE_SStructGrid__array* array)
{
  return sidl_interface__array_isColumnOrder((struct sidl_interface__array 
    *)array);
}

int
bHYPRE_SStructGrid__array_isRowOrder(
  const struct bHYPRE_SStructGrid__array* array)
{
  return sidl_interface__array_isRowOrder((struct sidl_interface__array 
    *)array);
}

void
bHYPRE_SStructGrid__array_copy(
  const struct bHYPRE_SStructGrid__array* src,
  struct bHYPRE_SStructGrid__array* dest)
{
  sidl_interface__array_copy((const struct sidl_interface__array *)src,
                             (struct sidl_interface__array *)dest);
}

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_slice(
  struct bHYPRE_SStructGrid__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart)
{
  return (struct bHYPRE_SStructGrid__array*)
    sidl_interface__array_slice((struct sidl_interface__array *)src,
                                dimen, numElem, srcStart, srcStride, newStart);
}

struct bHYPRE_SStructGrid__array*
bHYPRE_SStructGrid__array_ensure(
  struct bHYPRE_SStructGrid__array* src,
  int32_t dimen,
  int     ordering)
{
  return (struct bHYPRE_SStructGrid__array*)
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
static struct sidl_recursive_mutex_t bHYPRE_SStructGrid__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE_SStructGrid__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE_SStructGrid__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE_SStructGrid__mutex )==EDEADLOCK) */
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

static struct bHYPRE_SStructGrid__epv s_rem_epv__bhypre_sstructgrid;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_bHYPRE_SStructGrid__cast(
  struct bHYPRE_SStructGrid__object* self,
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
    cmp1 = strcmp(name, "bHYPRE.SStructGrid");
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
    cast =  (*func)(((struct bHYPRE_SStructGrid__remote*)self->d_data)->d_ih,
      _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE_SStructGrid__delete(
  struct bHYPRE_SStructGrid__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_bHYPRE_SStructGrid__getURL(
  struct bHYPRE_SStructGrid__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_bHYPRE_SStructGrid__raddRef(
  struct bHYPRE_SStructGrid__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
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
remote_bHYPRE_SStructGrid__isRemote(
    struct bHYPRE_SStructGrid__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_bHYPRE_SStructGrid__set_hooks(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid._set_hooks.", &throwaway_exception);
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
static void remote_bHYPRE_SStructGrid__exec(
  struct bHYPRE_SStructGrid__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:SetNumDimParts */
static int32_t
remote_bHYPRE_SStructGrid_SetNumDimParts(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
  /* in */ int32_t ndim,
  /* in */ int32_t nparts,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetNumDimParts", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "ndim", ndim, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "nparts", nparts, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid.SetNumDimParts.", &throwaway_exception);
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
remote_bHYPRE_SStructGrid_SetCommunicator(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid.SetCommunicator.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetExtents */
static int32_t
remote_bHYPRE_SStructGrid_SetExtents(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetExtents", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "ilower", ilower,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "iupper", iupper,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid.SetExtents.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetVariable */
static int32_t
remote_bHYPRE_SStructGrid_SetVariable(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t nvars,
  /* in */ enum bHYPRE_SStructVariable__enum vartype,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetVariable", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "nvars", nvars, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "vartype", vartype,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid.SetVariable.", &throwaway_exception);
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

/* REMOTE METHOD STUB:AddVariable */
static int32_t
remote_bHYPRE_SStructGrid_AddVariable(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
  /* in */ enum bHYPRE_SStructVariable__enum vartype,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "AddVariable", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "index", index,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "vartype", vartype,
      _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid.AddVariable.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetNeighborBox */
static int32_t
remote_bHYPRE_SStructGrid_SetNeighborBox(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t nbor_part,
  /* in rarray[dim] */ struct sidl_int__array* nbor_ilower,
  /* in rarray[dim] */ struct sidl_int__array* nbor_iupper,
  /* in rarray[dim] */ struct sidl_int__array* index_map,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetNeighborBox", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "ilower", ilower,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "iupper", iupper,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "nbor_part", nbor_part,
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "nbor_ilower", nbor_ilower,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "nbor_iupper", nbor_iupper,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "index_map", index_map,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid.SetNeighborBox.", &throwaway_exception);
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

/* REMOTE METHOD STUB:AddUnstructuredPart */
static int32_t
remote_bHYPRE_SStructGrid_AddUnstructuredPart(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
  /* in */ int32_t ilower,
  /* in */ int32_t iupper,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "AddUnstructuredPart", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "ilower", ilower, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "iupper", iupper, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid.AddUnstructuredPart.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetPeriodic */
static int32_t
remote_bHYPRE_SStructGrid_SetPeriodic(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* periodic,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "SetPeriodic", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "periodic", periodic,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid.SetPeriodic.", &throwaway_exception);
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
remote_bHYPRE_SStructGrid_SetNumGhost(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid.SetNumGhost.", &throwaway_exception);
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
remote_bHYPRE_SStructGrid_Assemble(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "Assemble", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid.Assemble.", &throwaway_exception);
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
remote_bHYPRE_SStructGrid_addRef(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_SStructGrid__remote* r_obj = (struct 
      bHYPRE_SStructGrid__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_bHYPRE_SStructGrid_deleteRef(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_SStructGrid__remote* r_obj = (struct 
      bHYPRE_SStructGrid__remote*)self->d_data;
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
remote_bHYPRE_SStructGrid_isSame(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid.isSame.", &throwaway_exception);
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
remote_bHYPRE_SStructGrid_isType(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid.isType.", &throwaway_exception);
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
remote_bHYPRE_SStructGrid_getClassInfo(
  /* in */ struct bHYPRE_SStructGrid__object* self ,
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
      bHYPRE_SStructGrid__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructGrid.getClassInfo.", &throwaway_exception);
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
static void bHYPRE_SStructGrid__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE_SStructGrid__epv* epv = &s_rem_epv__bhypre_sstructgrid;
  struct sidl_BaseClass__epv*     e0  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv* e1  = &s_rem_epv__sidl_baseinterface;

  epv->f__cast                    = remote_bHYPRE_SStructGrid__cast;
  epv->f__delete                  = remote_bHYPRE_SStructGrid__delete;
  epv->f__exec                    = remote_bHYPRE_SStructGrid__exec;
  epv->f__getURL                  = remote_bHYPRE_SStructGrid__getURL;
  epv->f__raddRef                 = remote_bHYPRE_SStructGrid__raddRef;
  epv->f__isRemote                = remote_bHYPRE_SStructGrid__isRemote;
  epv->f__set_hooks               = remote_bHYPRE_SStructGrid__set_hooks;
  epv->f__ctor                    = NULL;
  epv->f__ctor2                   = NULL;
  epv->f__dtor                    = NULL;
  epv->f_SetNumDimParts           = remote_bHYPRE_SStructGrid_SetNumDimParts;
  epv->f_SetCommunicator          = remote_bHYPRE_SStructGrid_SetCommunicator;
  epv->f_SetExtents               = remote_bHYPRE_SStructGrid_SetExtents;
  epv->f_SetVariable              = remote_bHYPRE_SStructGrid_SetVariable;
  epv->f_AddVariable              = remote_bHYPRE_SStructGrid_AddVariable;
  epv->f_SetNeighborBox           = remote_bHYPRE_SStructGrid_SetNeighborBox;
  epv->f_AddUnstructuredPart      = 
    remote_bHYPRE_SStructGrid_AddUnstructuredPart;
  epv->f_SetPeriodic              = remote_bHYPRE_SStructGrid_SetPeriodic;
  epv->f_SetNumGhost              = remote_bHYPRE_SStructGrid_SetNumGhost;
  epv->f_Assemble                 = remote_bHYPRE_SStructGrid_Assemble;
  epv->f_addRef                   = remote_bHYPRE_SStructGrid_addRef;
  epv->f_deleteRef                = remote_bHYPRE_SStructGrid_deleteRef;
  epv->f_isSame                   = remote_bHYPRE_SStructGrid_isSame;
  epv->f_isType                   = remote_bHYPRE_SStructGrid_isType;
  epv->f_getClassInfo             = remote_bHYPRE_SStructGrid_getClassInfo;

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
static struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__remoteConnect(const char *url, sidl_bool ar,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_SStructGrid__object* self;

  struct bHYPRE_SStructGrid__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_SStructGrid__remote* r_obj;
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
    return bHYPRE_SStructGrid__rmicast(bi,_ex);SIDL_CHECK(*_ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar,
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_SStructGrid__object*) malloc(
      sizeof(struct bHYPRE_SStructGrid__object));

  r_obj =
    (struct bHYPRE_SStructGrid__remote*) malloc(
      sizeof(struct bHYPRE_SStructGrid__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                              self;
  s1 =                              &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructGrid__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_sstructgrid;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  return NULL;
}
/* Create an instance that uses an already existing  */
/* InstanceHandle to connect to an existing remote object. */
static struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__IHConnect(sidl_rmi_InstanceHandle instance,
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_SStructGrid__object* self;

  struct bHYPRE_SStructGrid__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_SStructGrid__remote* r_obj;
  self =
    (struct bHYPRE_SStructGrid__object*) malloc(
      sizeof(struct bHYPRE_SStructGrid__object));

  r_obj =
    (struct bHYPRE_SStructGrid__remote*) malloc(
      sizeof(struct bHYPRE_SStructGrid__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                              self;
  s1 =                              &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructGrid__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_sstructgrid;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}
/* REMOTE: generate remote instance given URL string. */
static struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__remoteCreate(const char *url, sidl_BaseInterface *_ex)
{
  sidl_BaseInterface _throwaway_exception = NULL;
  struct bHYPRE_SStructGrid__object* self;

  struct bHYPRE_SStructGrid__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_SStructGrid__remote* r_obj;
  sidl_rmi_InstanceHandle instance = 
    sidl_rmi_ProtocolFactory_createInstance(url, "bHYPRE.SStructGrid",
    _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_SStructGrid__object*) malloc(
      sizeof(struct bHYPRE_SStructGrid__object));

  r_obj =
    (struct bHYPRE_SStructGrid__remote*) malloc(
      sizeof(struct bHYPRE_SStructGrid__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                              self;
  s1 =                              &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructGrid__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_sstructgrid;

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

struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct bHYPRE_SStructGrid__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.SStructGrid",
      (void*)bHYPRE_SStructGrid__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct bHYPRE_SStructGrid__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.SStructGrid", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct bHYPRE_SStructGrid__object*
bHYPRE_SStructGrid__connectI(const char* url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex)
{
  return bHYPRE_SStructGrid__remoteConnect(url, ar, _ex);
}

