/*
 * File:          bHYPRE_SStructParCSRVector.h
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Client-side glue code for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_bHYPRE_SStructParCSRVector_h
#define included_bHYPRE_SStructParCSRVector_h

/**
 * Symbol "bHYPRE.SStructParCSRVector" (version 1.0.0)
 * 
 * The SStructParCSR vector class.
 * 
 * Objects of this type can be cast to SStructVectorView or Vector
 * objects using the {\tt \_\_cast} methods.
 */
struct bHYPRE_SStructParCSRVector__object;
struct bHYPRE_SStructParCSRVector__array;
typedef struct bHYPRE_SStructParCSRVector__object* bHYPRE_SStructParCSRVector;

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
#ifndef included_bHYPRE_SStructParCSRVector_IOR_h
#include "bHYPRE_SStructParCSRVector_IOR.h"
#endif
#endif /* SIDL_C_HAS_INLINE */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__create(sidl_BaseInterface* _ex);

/**
 * RMI constructor function for the class.
 */
bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__createRemote(const char * url,
  sidl_BaseInterface *_ex);

/**
 * Wraps up the private data struct pointer (struct bHYPRE_SStructParCSRVector__data) passed in rather than running the constructor.
 */
bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__wrapObj(void * data, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.(addrefs)
 */
bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__connect(const char *, sidl_BaseInterface *_ex);

/**
 * Method:  Create[]
 */
bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex);

SIDL_C_INLINE_DECL
void
bHYPRE_SStructParCSRVector_addRef(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_deleteRef(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_isSame(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_isType(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_getClassInfo(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_SetGrid(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_SetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_SetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_AddToValues(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_AddToBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_Gather(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_GetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_GetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_SetComplex(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_Print(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_GetObject(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_SetCommunicator(
  /* in */ bHYPRE_SStructParCSRVector self,
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
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 */
SIDL_C_INLINE_DECL
int32_t
bHYPRE_SStructParCSRVector_Initialize(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_Assemble(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_Clear(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_Copy(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_Clone(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_Scale(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_Dot(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector_Axpy(
  /* in */ bHYPRE_SStructParCSRVector self,
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
struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__cast(
  void* obj,
  sidl_BaseInterface* _ex);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_SStructParCSRVector__cast2(
  void* obj,
  const char* type,
  sidl_BaseInterface *_ex);

/**
 * Select and execute a method by name
 */
SIDL_C_INLINE_DECL
void
bHYPRE_SStructParCSRVector__exec(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector__getURL(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector__raddRef(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector__isRemote(
  /* in */ bHYPRE_SStructParCSRVector self,
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
bHYPRE_SStructParCSRVector__isLocal(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex);
struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create1d(int32_t len);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create1dInit(
  int32_t len, 
  bHYPRE_SStructParCSRVector* data);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_borrow(
  bHYPRE_SStructParCSRVector* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_smartCopy(
  struct bHYPRE_SStructParCSRVector__array *array);

void
bHYPRE_SStructParCSRVector__array_addRef(
  struct bHYPRE_SStructParCSRVector__array* array);

void
bHYPRE_SStructParCSRVector__array_deleteRef(
  struct bHYPRE_SStructParCSRVector__array* array);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get1(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get2(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get3(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get4(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get5(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get6(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get7(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_SStructParCSRVector
bHYPRE_SStructParCSRVector__array_get(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t indices[]);

void
bHYPRE_SStructParCSRVector__array_set1(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set2(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set3(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set4(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set5(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set6(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_SStructParCSRVector const value);

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
  bHYPRE_SStructParCSRVector const value);

void
bHYPRE_SStructParCSRVector__array_set(
  struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t indices[],
  bHYPRE_SStructParCSRVector const value);

int32_t
bHYPRE_SStructParCSRVector__array_dimen(
  const struct bHYPRE_SStructParCSRVector__array* array);

int32_t
bHYPRE_SStructParCSRVector__array_lower(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructParCSRVector__array_upper(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructParCSRVector__array_length(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_SStructParCSRVector__array_stride(
  const struct bHYPRE_SStructParCSRVector__array* array,
  const int32_t ind);

int
bHYPRE_SStructParCSRVector__array_isColumnOrder(
  const struct bHYPRE_SStructParCSRVector__array* array);

int
bHYPRE_SStructParCSRVector__array_isRowOrder(
  const struct bHYPRE_SStructParCSRVector__array* array);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_slice(
  struct bHYPRE_SStructParCSRVector__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_SStructParCSRVector__array_copy(
  const struct bHYPRE_SStructParCSRVector__array* src,
  struct bHYPRE_SStructParCSRVector__array* dest);

struct bHYPRE_SStructParCSRVector__array*
bHYPRE_SStructParCSRVector__array_ensure(
  struct bHYPRE_SStructParCSRVector__array* src,
  int32_t dimen,
  int     ordering);


#pragma weak bHYPRE_SStructParCSRVector__connectI

#pragma weak bHYPRE_SStructParCSRVector__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
