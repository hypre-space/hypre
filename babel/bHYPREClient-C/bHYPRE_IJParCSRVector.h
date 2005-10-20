/*
 * File:          bHYPRE_IJParCSRVector.h
 * Symbol:        bHYPRE.IJParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Client-side glue code for bHYPRE.IJParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#ifndef included_bHYPRE_IJParCSRVector_h
#define included_bHYPRE_IJParCSRVector_h

/**
 * Symbol "bHYPRE.IJParCSRVector" (version 1.0.0)
 * 
 * The IJParCSR vector class.
 * 
 * Objects of this type can be cast to IJVectorView or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */
struct bHYPRE_IJParCSRVector__object;
struct bHYPRE_IJParCSRVector__array;
typedef struct bHYPRE_IJParCSRVector__object* bHYPRE_IJParCSRVector;

/*
 * Includes for all header dependencies.
 */

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_h
#include "bHYPRE_MPICommunicator.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif

#ifndef included_sidl_io_Serializer_h
#include "sidl_io_Serializer.h"
#endif
#ifndef included_sidl_io_Deserializer_h
#include "sidl_io_Deserializer.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
struct bHYPRE_IJParCSRVector__object*
bHYPRE_IJParCSRVector__create(void);

/**
 * RMI constructor function for the class.
 */
bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__createRemote(const char *, sidl_BaseInterface *_ex);

/**
 * RMI connector function for the class.
 */
bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__connect(const char *, sidl_BaseInterface *_ex);
void
bHYPRE_IJParCSRVector_addRef(
  /* in */ bHYPRE_IJParCSRVector self);

void
bHYPRE_IJParCSRVector_deleteRef(
  /* in */ bHYPRE_IJParCSRVector self);

sidl_bool
bHYPRE_IJParCSRVector_isSame(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ sidl_BaseInterface iobj);

sidl_BaseInterface
bHYPRE_IJParCSRVector_queryInt(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ const char* name);

sidl_bool
bHYPRE_IJParCSRVector_isType(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ const char* name);

sidl_ClassInfo
bHYPRE_IJParCSRVector_getClassInfo(
  /* in */ bHYPRE_IJParCSRVector self);

/**
 * Method:  Create[]
 */
bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper);

/**
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */
int32_t
bHYPRE_IJParCSRVector_SetCommunicator(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Initialize(
  /* in */ bHYPRE_IJParCSRVector self);

/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Assemble(
  /* in */ bHYPRE_IJParCSRVector self);

/**
 * Set the local range for a vector object.  Each process owns
 * some unique consecutive range of vector unknowns, indicated
 * by the global indices {\tt jlower} and {\tt jupper}.  The
 * data is required to be such that the value of {\tt jlower} on
 * any process $p$ be exactly one more than the value of {\tt
 * jupper} on process $p-1$.  Note that the first index of the
 * global vector may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * Collective.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_SetLocalRange(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper);

/**
 * Sets values in vector.  The arrays {\tt values} and {\tt
 * indices} are of dimension {\tt nvalues} and contain the
 * vector values to be set and the corresponding global vector
 * indices, respectively.  Erases any previous values at the
 * specified locations and replaces them with new ones.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_SetValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t nvalues,
  /* in rarray[nvalues] */ int32_t* indices,
  /* in rarray[nvalues] */ double* values);

/**
 * Adds to values in vector.  Usage details are analogous to
 * {\tt SetValues}.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_AddToValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t nvalues,
  /* in rarray[nvalues] */ int32_t* indices,
  /* in rarray[nvalues] */ double* values);

/**
 * Returns range of the part of the vector owned by this
 * processor.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_GetLocalRange(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper);

/**
 * Gets values in vector.  Usage details are analogous to {\tt
 * SetValues}.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_GetValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t nvalues,
  /* in rarray[nvalues] */ int32_t* indices,
  /* inout rarray[nvalues] */ double* values);

/**
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Print(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ const char* filename);

/**
 * Read the vector from file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Read(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ const char* filename,
  /* in */ bHYPRE_MPICommunicator comm);

/**
 * Set {\tt self} to 0.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Clear(
  /* in */ bHYPRE_IJParCSRVector self);

/**
 * Copy x into {\tt self}.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Copy(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ bHYPRE_Vector x);

/**
 * Create an {\tt x} compatible with {\tt self}.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Clone(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ bHYPRE_Vector* x);

/**
 * Scale {\tt self} by {\tt a}.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Scale(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ double a);

/**
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Dot(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d);

/**
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Axpy(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x);

/**
 * Cast method for interface and class type conversions.
 */
struct bHYPRE_IJParCSRVector__object*
bHYPRE_IJParCSRVector__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_IJParCSRVector__cast2(
  void* obj,
  const char* type);

/**
 * Select and execute a method by name
 */
void
bHYPRE_IJParCSRVector__exec(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * static Exec method for reflexity.
 */
void
bHYPRE_IJParCSRVector__sexec(
  /* in */ const char* methodName,
  /* in */ sidl_io_Deserializer inArgs,
  /* in */ sidl_io_Serializer outArgs);
/**
 * Get the URL of the Implementation of this object (for RMI)
 */
char*
bHYPRE_IJParCSRVector__getURL(
  /* in */ bHYPRE_IJParCSRVector self);
struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_createCol(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_createRow(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_create1d(int32_t len);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_create1dInit(
  int32_t len, 
  bHYPRE_IJParCSRVector* data);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_borrow(
  bHYPRE_IJParCSRVector* firstElement,
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[],
  const int32_t stride[]);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_smartCopy(
  struct bHYPRE_IJParCSRVector__array *array);

void
bHYPRE_IJParCSRVector__array_addRef(
  struct bHYPRE_IJParCSRVector__array* array);

void
bHYPRE_IJParCSRVector__array_deleteRef(
  struct bHYPRE_IJParCSRVector__array* array);

bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__array_get1(
  const struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1);

bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__array_get2(
  const struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2);

bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__array_get3(
  const struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3);

bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__array_get4(
  const struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4);

bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__array_get5(
  const struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5);

bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__array_get6(
  const struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6);

bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__array_get7(
  const struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7);

bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__array_get(
  const struct bHYPRE_IJParCSRVector__array* array,
  const int32_t indices[]);

void
bHYPRE_IJParCSRVector__array_set1(
  struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1,
  bHYPRE_IJParCSRVector const value);

void
bHYPRE_IJParCSRVector__array_set2(
  struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  bHYPRE_IJParCSRVector const value);

void
bHYPRE_IJParCSRVector__array_set3(
  struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  bHYPRE_IJParCSRVector const value);

void
bHYPRE_IJParCSRVector__array_set4(
  struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  bHYPRE_IJParCSRVector const value);

void
bHYPRE_IJParCSRVector__array_set5(
  struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  bHYPRE_IJParCSRVector const value);

void
bHYPRE_IJParCSRVector__array_set6(
  struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  bHYPRE_IJParCSRVector const value);

void
bHYPRE_IJParCSRVector__array_set7(
  struct bHYPRE_IJParCSRVector__array* array,
  const int32_t i1,
  const int32_t i2,
  const int32_t i3,
  const int32_t i4,
  const int32_t i5,
  const int32_t i6,
  const int32_t i7,
  bHYPRE_IJParCSRVector const value);

void
bHYPRE_IJParCSRVector__array_set(
  struct bHYPRE_IJParCSRVector__array* array,
  const int32_t indices[],
  bHYPRE_IJParCSRVector const value);

int32_t
bHYPRE_IJParCSRVector__array_dimen(
  const struct bHYPRE_IJParCSRVector__array* array);

int32_t
bHYPRE_IJParCSRVector__array_lower(
  const struct bHYPRE_IJParCSRVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJParCSRVector__array_upper(
  const struct bHYPRE_IJParCSRVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJParCSRVector__array_length(
  const struct bHYPRE_IJParCSRVector__array* array,
  const int32_t ind);

int32_t
bHYPRE_IJParCSRVector__array_stride(
  const struct bHYPRE_IJParCSRVector__array* array,
  const int32_t ind);

int
bHYPRE_IJParCSRVector__array_isColumnOrder(
  const struct bHYPRE_IJParCSRVector__array* array);

int
bHYPRE_IJParCSRVector__array_isRowOrder(
  const struct bHYPRE_IJParCSRVector__array* array);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_slice(
  struct bHYPRE_IJParCSRVector__array* src,
  int32_t        dimen,
  const int32_t  numElem[],
  const int32_t  *srcStart,
  const int32_t  *srcStride,
  const int32_t  *newStart);

void
bHYPRE_IJParCSRVector__array_copy(
  const struct bHYPRE_IJParCSRVector__array* src,
  struct bHYPRE_IJParCSRVector__array* dest);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_ensure(
  struct bHYPRE_IJParCSRVector__array* src,
  int32_t dimen,
  int     ordering);

#ifdef __cplusplus
}
#endif
#endif
