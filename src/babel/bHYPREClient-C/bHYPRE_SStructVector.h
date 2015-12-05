/*
 * File:          bHYPRE_SStructVector.h
 * Symbol:        bHYPRE.SStructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.SStructVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructVector_h
#define included_bHYPRE_SStructVector_h

/**
 * Symbol "bHYPRE.SStructVector" (version 1.0.0)
 * 
 * The semi-structured grid vector class.
 * 
 * Objects of this type can be cast to SStructVectorView or Vector
 * objects using the {\tt \_\_cast} methods.
 */
struct bHYPRE_SStructVector__object;
struct bHYPRE_SStructVector__array;
typedef struct bHYPRE_SStructVector__object* bHYPRE_SStructVector;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_SStructGrid_h
#include "bHYPRE_SStructGrid.h"
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
#ifndef included_bHYPRE_SStructVector_IOR_h
#include "bHYPRE_SStructVector_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct bHYPRE_SStructVector__object*
bHYPRE_SStructVector__create(sidl_BaseInterface* _ex);

/**
 * RMI constructor function for the class.
 */
bHYPRE_SStructVector
bHYPRE_SStructVector__createRemote(const char * url, sidl_BaseInterface *_ex);

/**
 * Wraps up the private data struct pointer (struct bHYPRE\_SStructVector\_\_data) passed in rather than running the constructor.
 */
bHYPRE_SStructVector
bHYPRE_SStructVector__wrapObj(void * data, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_SStructVector
bHYPRE_SStructVector__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Method:  Create[]
 */
bHYPRE_SStructVector
bHYPRE_SStructVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Method:  SetObjectType[]
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVector_SetObjectType(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t type,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetObjectType)(
    self,
    type,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


SIDL_C_INLINE_DECL
void
bHYPRE_SStructVector_addRef(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_deleteRef(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_isSame(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_isType(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_getClassInfo(
  /* in */ bHYPRE_SStructVector self,
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
 * Set the vector grid.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVector_SetGrid(
  /* in */ bHYPRE_SStructVector self,
  /* in */ bHYPRE_SStructGrid grid,
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
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVector_SetValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
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
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
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
bHYPRE_SStructVector_SetBoxValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

/**
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
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVector_AddToValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
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
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
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
bHYPRE_SStructVector_AddToBoxValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Gather vector data before calling {\tt GetValues}.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVector_Gather(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Gather)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Get vector coefficients index by index.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVector_GetValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
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
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
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
bHYPRE_SStructVector_GetBoxValues(
  /* in */ bHYPRE_SStructVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* inout rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex);

/**
 * Set the vector to be complex.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVector_SetComplex(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_SetComplex)(
    self,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVector_Print(
  /* in */ bHYPRE_SStructVector self,
  /* in */ const char* filename,
  /* in */ int32_t all,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_Print)(
    self,
    filename,
    all,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * A semi-structured matrix or vector contains a Struct or IJ matrix
 * or vector.  GetObject returns it.
 * The returned type is a sidl.BaseInterface.
 * A cast must be used on the returned object to convert it into a known type.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVector_GetObject(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface* A,
  /* out */ sidl_BaseInterface *_ex)
#ifdef SIDL_C_HAS_INLINE
{
  return (*self->d_epv->f_GetObject)(
    self,
    A,
    _ex);
}
#else
;
#endif /* SIDL_C_HAS_INLINE */


/**
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructVector_SetCommunicator(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_Destroy(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_Initialize(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_Assemble(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_Clear(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_Copy(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_Clone(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_Scale(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_Dot(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector_Axpy(
  /* in */ bHYPRE_SStructVector self,
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
struct bHYPRE_SStructVector__object*
bHYPRE_SStructVector__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructVector__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_SStructVector__exec(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector__getURL(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector__raddRef(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector__isRemote(
  /* in */ bHYPRE_SStructVector self,
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
bHYPRE_SStructVector__isLocal(
  /* in */ bHYPRE_SStructVector self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_create1d(int32_t len);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructVector* data);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_borrow(
  bHYPRE_SStructVector* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_smartCopy(
  struct bHYPRE_SStructVector__array *array);

void
bHYPRE_SStructVector__array_addRef(
  struct bHYPRE_SStructVector__array* array);

void
bHYPRE_SStructVector__array_deleteRef(
  struct bHYPRE_SStructVector__array* array);

bHYPRE_SStructVector
bHYPRE_SStructVector__array_get1(
  const struct bHYPRE_SStructVector__array* array,
  const int32_t i1);

bHYPRE_SStructVector
bHYPRE_SStructVector__array_get2(
  const struct bHYPRE_SStructVector__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructVector
bHYPRE_SStructVector__array_get3(
  const struct bHYPRE_SStructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructVector
bHYPRE_SStructVector__array_get4(
  const struct bHYPRE_SStructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructVector
bHYPRE_SStructVector__array_get5(
  const struct bHYPRE_SStructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructVector
bHYPRE_SStructVector__array_get6(
  const struct bHYPRE_SStructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructVector
bHYPRE_SStructVector__array_get7(
  const struct bHYPRE_SStructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructVector
bHYPRE_SStructVector__array_get(
  const struct bHYPRE_SStructVector__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructVector__array_set1(
  struct bHYPRE_SStructVector__array* array,
  const int32_t i1,
  bHYPRE_SStructVector const value);

void
bHYPRE_SStructVector__array_set2(
  struct bHYPRE_SStructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructVector const value);

void
bHYPRE_SStructVector__array_set3(
  struct bHYPRE_SStructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructVector const value);

void
bHYPRE_SStructVector__array_set4(
  struct bHYPRE_SStructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructVector const value);

void
bHYPRE_SStructVector__array_set5(
  struct bHYPRE_SStructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructVector const value);

void
bHYPRE_SStructVector__array_set6(
  struct bHYPRE_SStructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructVector const value);

void
bHYPRE_SStructVector__array_set7(
  struct bHYPRE_SStructVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_SStructVector const value);

void
bHYPRE_SStructVector__array_set(
  struct bHYPRE_SStructVector__array* array,
  const int32_t indices[],
  bHYPRE_SStructVector const value);

int32_t
bHYPRE_SStructVector__array_dimen(
  const struct bHYPRE_SStructVector__array* array);

int32_t
bHYPRE_SStructVector__array_lower(
  const struct bHYPRE_SStructVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructVector__array_upper(
  const struct bHYPRE_SStructVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructVector__array_length(
  const struct bHYPRE_SStructVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructVector__array_stride(
  const struct bHYPRE_SStructVector__array* array,
  const int32_t ind);

int
bHYPRE_SStructVector__array_isColumnOrder(
  const struct bHYPRE_SStructVector__array* array);

int
bHYPRE_SStructVector__array_isRowOrder(
  const struct bHYPRE_SStructVector__array* array);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_slice(
  struct bHYPRE_SStructVector__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructVector__array_copy(
  const struct bHYPRE_SStructVector__array* src,
  struct bHYPRE_SStructVector__array* dest);

struct bHYPRE_SStructVector__array*
bHYPRE_SStructVector__array_ensure(
  struct bHYPRE_SStructVector__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_SStructVector__connectI

#pragma weak bHYPRE_SStructVector__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructVector__object*
bHYPRE_SStructVector__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructVector__object*
bHYPRE_SStructVector__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
