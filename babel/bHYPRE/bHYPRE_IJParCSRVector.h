/*
 * File:          bHYPRE_IJParCSRVector.h
 * Symbol:        bHYPRE.IJParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:29 PST
 * Description:   Client-side glue code for bHYPRE.IJParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 815
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_IJParCSRVector_h
#define included_bHYPRE_IJParCSRVector_h

/**
 * Symbol "bHYPRE.IJParCSRVector" (version 1.0.0)
 * 
 * The IJParCSR vector class.
 * 
 * Objects of this type can be cast to IJBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */
struct bHYPRE_IJParCSRVector__object;
struct bHYPRE_IJParCSRVector__array;
typedef struct bHYPRE_IJParCSRVector__object* bHYPRE_IJParCSRVector;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif
#ifndef included_SIDL_ClassInfo_h
#include "SIDL_ClassInfo.h"
#endif
#ifndef included_bHYPRE_Vector_h
#include "bHYPRE_Vector.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Constructor function for the class.
 */
bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__create(void);

void
bHYPRE_IJParCSRVector_addRef(
  bHYPRE_IJParCSRVector self);

void
bHYPRE_IJParCSRVector_deleteRef(
  bHYPRE_IJParCSRVector self);

SIDL_bool
bHYPRE_IJParCSRVector_isSame(
  bHYPRE_IJParCSRVector self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_IJParCSRVector_queryInt(
  bHYPRE_IJParCSRVector self,
  const char* name);

SIDL_bool
bHYPRE_IJParCSRVector_isType(
  bHYPRE_IJParCSRVector self,
  const char* name);

SIDL_ClassInfo
bHYPRE_IJParCSRVector_getClassInfo(
  bHYPRE_IJParCSRVector self);

/**
 * Set {\tt self} to 0.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Clear(
  bHYPRE_IJParCSRVector self);

/**
 * Copy x into {\tt self}.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Copy(
  bHYPRE_IJParCSRVector self,
  bHYPRE_Vector x);

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
  bHYPRE_IJParCSRVector self,
  bHYPRE_Vector* x);

/**
 * Scale {\tt self} by {\tt a}.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Scale(
  bHYPRE_IJParCSRVector self,
  double a);

/**
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Dot(
  bHYPRE_IJParCSRVector self,
  bHYPRE_Vector x,
  double* d);

/**
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Axpy(
  bHYPRE_IJParCSRVector self,
  double a,
  bHYPRE_Vector x);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_SetCommunicator(
  bHYPRE_IJParCSRVector self,
  void* mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Initialize(
  bHYPRE_IJParCSRVector self);

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
  bHYPRE_IJParCSRVector self);

/**
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_GetObject(
  bHYPRE_IJParCSRVector self,
  SIDL_BaseInterface* A);

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
  bHYPRE_IJParCSRVector self,
  int32_t jlower,
  int32_t jupper);

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
  bHYPRE_IJParCSRVector self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values);

/**
 * Adds to values in vector.  Usage details are analogous to
 * {\tt SetValues}.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_AddToValues(
  bHYPRE_IJParCSRVector self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values);

/**
 * Returns range of the part of the vector owned by this
 * processor.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_GetLocalRange(
  bHYPRE_IJParCSRVector self,
  int32_t* jlower,
  int32_t* jupper);

/**
 * Gets values in vector.  Usage details are analogous to {\tt
 * SetValues}.
 * 
 * Not collective.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_GetValues(
  bHYPRE_IJParCSRVector self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array** values);

/**
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Print(
  bHYPRE_IJParCSRVector self,
  const char* filename);

/**
 * Read the vector from file.  This is mainly for debugging
 * purposes.
 * 
 */
int32_t
bHYPRE_IJParCSRVector_Read(
  bHYPRE_IJParCSRVector self,
  const char* filename,
  void* comm);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_IJParCSRVector__cast2(
  void* obj,
  const char* type);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_createCol(int32_t        dimen,
                                       const int32_t lower[],
                                       const int32_t upper[]);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_createRow(int32_t        dimen,
                                       const int32_t lower[],
                                       const int32_t upper[]);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_create1d(int32_t len);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_borrow(bHYPRE_IJParCSRVector*firstElement,
                                    int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_smartCopy(struct bHYPRE_IJParCSRVector__array 
  *array);

void
bHYPRE_IJParCSRVector__array_addRef(struct bHYPRE_IJParCSRVector__array* array);

void
bHYPRE_IJParCSRVector__array_deleteRef(struct bHYPRE_IJParCSRVector__array* 
  array);

bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__array_get1(const struct bHYPRE_IJParCSRVector__array* 
  array,
                                  const int32_t i1);

bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__array_get2(const struct bHYPRE_IJParCSRVector__array* 
  array,
                                  const int32_t i1,
                                  const int32_t i2);

bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__array_get3(const struct bHYPRE_IJParCSRVector__array* 
  array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  const int32_t i3);

bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__array_get4(const struct bHYPRE_IJParCSRVector__array* 
  array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  const int32_t i3,
                                  const int32_t i4);

bHYPRE_IJParCSRVector
bHYPRE_IJParCSRVector__array_get(const struct bHYPRE_IJParCSRVector__array* 
  array,
                                 const int32_t indices[]);

void
bHYPRE_IJParCSRVector__array_set1(struct bHYPRE_IJParCSRVector__array* array,
                                  const int32_t i1,
                                  bHYPRE_IJParCSRVector const value);

void
bHYPRE_IJParCSRVector__array_set2(struct bHYPRE_IJParCSRVector__array* array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  bHYPRE_IJParCSRVector const value);

void
bHYPRE_IJParCSRVector__array_set3(struct bHYPRE_IJParCSRVector__array* array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  const int32_t i3,
                                  bHYPRE_IJParCSRVector const value);

void
bHYPRE_IJParCSRVector__array_set4(struct bHYPRE_IJParCSRVector__array* array,
                                  const int32_t i1,
                                  const int32_t i2,
                                  const int32_t i3,
                                  const int32_t i4,
                                  bHYPRE_IJParCSRVector const value);

void
bHYPRE_IJParCSRVector__array_set(struct bHYPRE_IJParCSRVector__array* array,
                                 const int32_t indices[],
                                 bHYPRE_IJParCSRVector const value);

int32_t
bHYPRE_IJParCSRVector__array_dimen(const struct bHYPRE_IJParCSRVector__array* 
  array);

int32_t
bHYPRE_IJParCSRVector__array_lower(const struct bHYPRE_IJParCSRVector__array* 
  array,
                                   const int32_t ind);

int32_t
bHYPRE_IJParCSRVector__array_upper(const struct bHYPRE_IJParCSRVector__array* 
  array,
                                   const int32_t ind);

int32_t
bHYPRE_IJParCSRVector__array_stride(const struct bHYPRE_IJParCSRVector__array* 
  array,
                                    const int32_t ind);

int
bHYPRE_IJParCSRVector__array_isColumnOrder(const struct 
  bHYPRE_IJParCSRVector__array* array);

int
bHYPRE_IJParCSRVector__array_isRowOrder(const struct 
  bHYPRE_IJParCSRVector__array* array);

void
bHYPRE_IJParCSRVector__array_slice(const struct bHYPRE_IJParCSRVector__array* 
  src,
                                         int32_t        dimen,
                                         const int32_t  numElem[],
                                         const int32_t  *srcStart,
                                         const int32_t  *srcStride,
                                         const int32_t  *newStart);

void
bHYPRE_IJParCSRVector__array_copy(const struct bHYPRE_IJParCSRVector__array* 
  src,
                                        struct bHYPRE_IJParCSRVector__array* 
  dest);

struct bHYPRE_IJParCSRVector__array*
bHYPRE_IJParCSRVector__array_ensure(struct bHYPRE_IJParCSRVector__array* src,
                                    int32_t dimen,
                                    int     ordering);

#ifdef __cplusplus
}
#endif
#endif
