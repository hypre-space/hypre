/*
 * File:          bHYPRE_StructGrid.h
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side glue code for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructGrid_h
#define included_bHYPRE_StructGrid_h

/**
 * Symbol "bHYPRE.StructGrid" (version 1.0.0)
 * 
 * Define a structured grid class.
 */
struct bHYPRE_StructGrid__object;
struct bHYPRE_StructGrid__array;
typedef struct bHYPRE_StructGrid__object* bHYPRE_StructGrid;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
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
#ifndef included_bHYPRE_StructGrid_IOR_h
#include "bHYPRE_StructGrid_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct bHYPRE_StructGrid__object*
bHYPRE_StructGrid__create(sidl_BaseInterface* _ex);

/**
 * RMI constructor function for the class.
 */
bHYPRE_StructGrid
bHYPRE_StructGrid__createRemote(const char * url, sidl_BaseInterface *_ex);

/**
 * Wraps up the private data struct pointer (struct bHYPRE\_StructGrid\_\_data) passed in rather than running the constructor.
 */
bHYPRE_StructGrid
bHYPRE_StructGrid__wrapObj(void * data, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_StructGrid
bHYPRE_StructGrid__connect(const char *, sidl_BaseInterface *_ex);

/**
 *  This function is the preferred way to create a Struct Grid. 
 */
bHYPRE_StructGrid
bHYPRE_StructGrid_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Set the MPI Communicator.
 * DEPRECATED, use Create:
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructGrid_SetCommunicator(
  /* in */ bHYPRE_StructGrid self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */
SIDL_C_INLINE_DECL
void
bHYPRE_StructGrid_Destroy(
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  (*self->d_epv->f_Destroy)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Method:  SetDimension[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructGrid_SetDimension(
  /* in */ bHYPRE_StructGrid self,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_SetDimension)(
    self,
    dim,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 *  Define the lower and upper corners of a box of the grid.
 * "ilower" and "iupper" are arrays of size "dim", the number of spatial
 * dimensions. 
 */
int32_t
bHYPRE_StructGrid_SetExtents(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

/**
 *  Set the periodicity for the grid.  Default is no periodicity.
 * 
 * The argument {\tt periodic} is an {\tt dim}-dimensional integer array that
 * contains the periodicity for each dimension.  A zero value for a dimension
 * means non-periodic, while a nonzero value means periodic and contains the
 * actual period.  For example, periodicity in the first and third dimensions
 * for a 10x11x12 grid is indicated by the array [10,0,12].
 * 
 * NOTE: Some of the solvers in hypre have power-of-two restrictions on the size
 * of the periodic dimensions.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructGrid_SetPeriodic(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim] */ int32_t* periodic,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  int32_t periodic_lower[1], periodic_upper[1], periodic_stride[1]; 
  struct sidl_int__array periodic_real;
  struct sidl_int__array*periodic_tmp = &periodic_real;
  periodic_upper[0] = dim-1;
  sidl_int__array_init(periodic, periodic_tmp, 1, periodic_lower, 
    periodic_upper, periodic_stride);
  _result = (*self->d_epv->f_SetPeriodic)(
    self,
    periodic_tmp,
    _ex);
#ifdef SIDL_DEBUG_REFCOUNT
  sidl__array_deleteRef((struct sidl__array*)periodic_tmp);
#endif /* SIDL_DEBUG_REFCOUNT */
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 *  Set the number of ghost zones, separately on the lower and upper sides
 * for each dimension.
 * "num\_ghost" is an array of size "dim2", twice the number of dimensions. 
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructGrid_SetNumGhost(
  /* in */ bHYPRE_StructGrid self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  int32_t num_ghost_lower[1], num_ghost_upper[1], num_ghost_stride[1]; 
  struct sidl_int__array num_ghost_real;
  struct sidl_int__array*num_ghost_tmp = &num_ghost_real;
  num_ghost_upper[0] = dim2-1;
  sidl_int__array_init(num_ghost, num_ghost_tmp, 1, num_ghost_lower, 
    num_ghost_upper, num_ghost_stride);
  _result = (*self->d_epv->f_SetNumGhost)(
    self,
    num_ghost_tmp,
    _ex);
#ifdef SIDL_DEBUG_REFCOUNT
  sidl__array_deleteRef((struct sidl__array*)num_ghost_tmp);
#endif /* SIDL_DEBUG_REFCOUNT */
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 *  final construction of the object before its use 
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructGrid_Assemble(
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t _result;
  _result = (*self->d_epv->f_Assemble)(
    self,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_StructGrid_addRef(
  /* in */ bHYPRE_StructGrid self,
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
bHYPRE_StructGrid_deleteRef(
  /* in */ bHYPRE_StructGrid self,
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
bHYPRE_StructGrid_isSame(
  /* in */ bHYPRE_StructGrid self,
  /* in */ sidl_BaseInterface iobj,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  sidl_bool _result;
  _result = (*self->d_epv->f_isSame)(
    self,
    iobj,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_bool
bHYPRE_StructGrid_isType(
  /* in */ bHYPRE_StructGrid self,
  /* in */ const char* name,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  sidl_bool _result;
  _result = (*self->d_epv->f_isType)(
    self,
    name,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
sidl_ClassInfo
bHYPRE_StructGrid_getClassInfo(
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  sidl_ClassInfo _result;
  _result = (*self->d_epv->f_getClassInfo)(
    self,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructGrid__object*
bHYPRE_StructGrid__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructGrid__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_StructGrid__exec(
  /* in */ bHYPRE_StructGrid self,
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
bHYPRE_StructGrid__getURL(
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  char* _result;
  _result = (*self->d_epv->f__getURL)(
    self,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * On a remote object, addrefs the remote instance.
 */
SIDL_C_INLINE_DECL
void
bHYPRE_StructGrid__raddRef(
  /* in */ bHYPRE_StructGrid self,
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
bHYPRE_StructGrid__isRemote(
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  sidl_bool _result;
  _result = (*self->d_epv->f__isRemote)(
    self,
    _ex);
  return _result;
}
#else
;
#endif /* SIDL_C_HAS_INLINE */

/**
 * TRUE if this object is remote, false if local
 */
sidl_bool
bHYPRE_StructGrid__isLocal(
  /* in */ bHYPRE_StructGrid self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_create1d(int32_t len);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_create1dInit(
  int32_t len, 
  bHYPRE_StructGrid* data);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_borrow(
  bHYPRE_StructGrid* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_smartCopy(
  struct bHYPRE_StructGrid__array *array);

void
bHYPRE_StructGrid__array_addRef(
  struct bHYPRE_StructGrid__array* array);

void
bHYPRE_StructGrid__array_deleteRef(
  struct bHYPRE_StructGrid__array* array);

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get1(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1);

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get2(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get3(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get4(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get5(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get6(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get7(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_StructGrid
bHYPRE_StructGrid__array_get(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t indices[]);

void
bHYPRE_StructGrid__array_set1(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  bHYPRE_StructGrid const value);

void
bHYPRE_StructGrid__array_set2(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructGrid const value);

void
bHYPRE_StructGrid__array_set3(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructGrid const value);

void
bHYPRE_StructGrid__array_set4(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructGrid const value);

void
bHYPRE_StructGrid__array_set5(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructGrid const value);

void
bHYPRE_StructGrid__array_set6(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructGrid const value);

void
bHYPRE_StructGrid__array_set7(
  struct bHYPRE_StructGrid__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructGrid const value);

void
bHYPRE_StructGrid__array_set(
  struct bHYPRE_StructGrid__array* array,
  const int32_t indices[],
  bHYPRE_StructGrid const value);

int32_t
bHYPRE_StructGrid__array_dimen(
  const struct bHYPRE_StructGrid__array* array);

int32_t
bHYPRE_StructGrid__array_lower(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructGrid__array_upper(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructGrid__array_length(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructGrid__array_stride(
  const struct bHYPRE_StructGrid__array* array,
  const int32_t ind);

int
bHYPRE_StructGrid__array_isColumnOrder(
  const struct bHYPRE_StructGrid__array* array);

int
bHYPRE_StructGrid__array_isRowOrder(
  const struct bHYPRE_StructGrid__array* array);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_slice(
  struct bHYPRE_StructGrid__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_StructGrid__array_copy(
  const struct bHYPRE_StructGrid__array* src,
  struct bHYPRE_StructGrid__array* dest);

struct bHYPRE_StructGrid__array*
bHYPRE_StructGrid__array_ensure(
  struct bHYPRE_StructGrid__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_StructGrid__connectI

#pragma weak bHYPRE_StructGrid__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructGrid__object*
bHYPRE_StructGrid__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_StructGrid__object*
bHYPRE_StructGrid__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
