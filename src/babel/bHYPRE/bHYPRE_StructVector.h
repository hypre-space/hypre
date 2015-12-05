/*
 * File:          bHYPRE_StructVector.h
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_StructVector_h
#define included_bHYPRE_StructVector_h

/**
 * Symbol "bHYPRE.StructVector" (version 1.0.0)
 */
struct bHYPRE_StructVector__object;
struct bHYPRE_StructVector__array;
typedef struct bHYPRE_StructVector__object* bHYPRE_StructVector;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_StructGrid_h
#include "bHYPRE_StructGrid.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
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
#ifndef included_bHYPRE_StructVector_IOR_h
#include "bHYPRE_StructVector_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct bHYPRE_StructVector__object*
bHYPRE_StructVector__create(sidl_BaseInterface* _ex);

/**
 * RMI constructor function for the class.
 */
bHYPRE_StructVector
bHYPRE_StructVector__createRemote(const char * url, sidl_BaseInterface *_ex);

/**
 * Wraps up the private data struct pointer (struct bHYPRE_StructVector__data) passed in rather than running the constructor.
 */
bHYPRE_StructVector
bHYPRE_StructVector__wrapObj(void * data, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_StructVector
bHYPRE_StructVector__connect(const char *, sidl_BaseInterface *_ex);

/**
 *  This function is the preferred way to create a Struct Vector. 
 */
bHYPRE_StructVector
bHYPRE_StructVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

SIDL_C_INLINE_DECL
void
bHYPRE_StructVector_addRef(
  /* in */ bHYPRE_StructVector self,
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
bHYPRE_StructVector_deleteRef(
  /* in */ bHYPRE_StructVector self,
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
bHYPRE_StructVector_isSame(
  /* in */ bHYPRE_StructVector self,
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
bHYPRE_StructVector_isType(
  /* in */ bHYPRE_StructVector self,
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
bHYPRE_StructVector_getClassInfo(
  /* in */ bHYPRE_StructVector self,
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
 *  Set the grid on which vectors are defined. 
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVector_SetGrid(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_StructGrid grid,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetGrid)(
    self,
    grid,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 *  Set the number of ghost zones, separately on the lower and upper sides
 * for each dimension.
 * "num_ghost" is an array of size "dim2", twice the number of dimensions. 
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVector_SetNumGhost(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
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
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 *  Set the value of a single vector coefficient, given by "grid_index".
 * "grid_index" is an array of size "dim", where dim is the number
 * of dimensions. 
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVector_SetValue(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim] */ int32_t* grid_index,
  /* in */ int32_t dim,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  int32_t grid_index_lower[1], grid_index_upper[1], grid_index_stride[1]; 
  struct sidl_int__array grid_index_real;
  struct sidl_int__array*grid_index_tmp = &grid_index_real;
  grid_index_upper[0] = dim-1;
  sidl_int__array_init(grid_index, grid_index_tmp, 1, grid_index_lower,
    grid_index_upper, grid_index_stride);
  return (*self->d_epv->f_SetValue)(
    self,
    grid_index_tmp,
    value,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 *  Set the values of all vector coefficient for grid points in a box.
 * The box is defined by its lower and upper corners in the grid.
 * "ilower" and "iupper" are arrays of size "dim", where dim is the
 * number of dimensions.  The "values" array has size "nvalues", which
 * is the number of grid points in the box. 
 */
int32_t
bHYPRE_StructVector_SetBoxValues(
  /* in */ bHYPRE_StructVector self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVector_SetCommunicator(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetCommunicator)(
    self,
    mpi_comm,
    _ex);
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
bHYPRE_StructVector_Destroy(
  /* in */ bHYPRE_StructVector self,
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
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVector_Initialize(
  /* in */ bHYPRE_StructVector self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Initialize)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVector_Assemble(
  /* in */ bHYPRE_StructVector self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Assemble)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set {\tt self} to 0.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVector_Clear(
  /* in */ bHYPRE_StructVector self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Clear)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Copy data from x into {\tt self}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVector_Copy(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Copy)(
    self,
    x,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Create an {\tt x} compatible with {\tt self}.
 * The new vector's data is not specified.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVector_Clone(
  /* in */ bHYPRE_StructVector self,
  /* out */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Clone)(
    self,
    x,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Scale {\tt self} by {\tt a}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVector_Scale(
  /* in */ bHYPRE_StructVector self,
  /* in */ double a,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Scale)(
    self,
    a,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVector_Dot(
  /* in */ bHYPRE_StructVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Dot)(
    self,
    x,
    d,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Add {\tt a}{\tt x} to {\tt self}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_StructVector_Axpy(
  /* in */ bHYPRE_StructVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Axpy)(
    self,
    a,
    x,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructVector__object*
bHYPRE_StructVector__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_StructVector__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_StructVector__exec(
  /* in */ bHYPRE_StructVector self,
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
bHYPRE_StructVector__getURL(
  /* in */ bHYPRE_StructVector self,
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
bHYPRE_StructVector__raddRef(
  /* in */ bHYPRE_StructVector self,
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
bHYPRE_StructVector__isRemote(
  /* in */ bHYPRE_StructVector self,
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
bHYPRE_StructVector__isLocal(
  /* in */ bHYPRE_StructVector self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_create1d(int32_t len);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_create1dInit(
  int32_t len, 
  bHYPRE_StructVector* data);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_borrow(
  bHYPRE_StructVector* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_smartCopy(
  struct bHYPRE_StructVector__array *array);

void
bHYPRE_StructVector__array_addRef(
  struct bHYPRE_StructVector__array* array);

void
bHYPRE_StructVector__array_deleteRef(
  struct bHYPRE_StructVector__array* array);

bHYPRE_StructVector
bHYPRE_StructVector__array_get1(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1);

bHYPRE_StructVector
bHYPRE_StructVector__array_get2(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_StructVector
bHYPRE_StructVector__array_get3(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_StructVector
bHYPRE_StructVector__array_get4(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_StructVector
bHYPRE_StructVector__array_get5(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_StructVector
bHYPRE_StructVector__array_get6(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_StructVector
bHYPRE_StructVector__array_get7(
  const struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_StructVector
bHYPRE_StructVector__array_get(
  const struct bHYPRE_StructVector__array* array,
  const int32_t indices[]);

void
bHYPRE_StructVector__array_set1(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set2(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set3(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set4(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set5(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set6(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set7(
  struct bHYPRE_StructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_StructVector const value);

void
bHYPRE_StructVector__array_set(
  struct bHYPRE_StructVector__array* array,
  const int32_t indices[],
  bHYPRE_StructVector const value);

int32_t
bHYPRE_StructVector__array_dimen(
  const struct bHYPRE_StructVector__array* array);

int32_t
bHYPRE_StructVector__array_lower(
  const struct bHYPRE_StructVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructVector__array_upper(
  const struct bHYPRE_StructVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructVector__array_length(
  const struct bHYPRE_StructVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_StructVector__array_stride(
  const struct bHYPRE_StructVector__array* array,
  const int32_t ind);

int
bHYPRE_StructVector__array_isColumnOrder(
  const struct bHYPRE_StructVector__array* array);

int
bHYPRE_StructVector__array_isRowOrder(
  const struct bHYPRE_StructVector__array* array);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_slice(
  struct bHYPRE_StructVector__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_StructVector__array_copy(
  const struct bHYPRE_StructVector__array* src,
  struct bHYPRE_StructVector__array* dest);

struct bHYPRE_StructVector__array*
bHYPRE_StructVector__array_ensure(
  struct bHYPRE_StructVector__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_StructVector__connectI

#pragma weak bHYPRE_StructVector__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_StructVector__object*
bHYPRE_StructVector__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_StructVector__object*
bHYPRE_StructVector__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
