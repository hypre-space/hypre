/*
 * File:          bHYPRE_ProblemDefinition.h
 * Symbol:        bHYPRE.ProblemDefinition-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:33 PST
 * Generated:     20030401 14:47:40 PST
 * Description:   Client-side glue code for bHYPRE.ProblemDefinition
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.8.2
 * source-line   = 42
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_bHYPRE_ProblemDefinition_h
#define included_bHYPRE_ProblemDefinition_h

/**
 * Symbol "bHYPRE.ProblemDefinition" (version 1.0.0)
 * 
 * The purpose of a ProblemDefinition is to:
 * 
 * \begin{itemize}
 * \item provide a particular view of how to define a problem
 * \item construct and return a {\it problem object}
 * \end{itemize}
 * 
 * A {\it problem object} is an intentionally vague term that
 * corresponds to any useful object used to define a problem.
 * Prime examples are:
 * 
 * \begin{itemize}
 * \item a LinearOperator object, i.e., something with a matvec
 * \item a MatrixAccess object, i.e., something with a getrow
 * \item a Vector, i.e., something with a dot, axpy, ...
 * \end{itemize}
 * 
 * Note that {\tt Initialize} and {\tt Assemble} are reserved here
 * for defining problem objects through a particular interface.
 * 
 */
struct bHYPRE_ProblemDefinition__object;
struct bHYPRE_ProblemDefinition__array;
typedef struct bHYPRE_ProblemDefinition__object* bHYPRE_ProblemDefinition;

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

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_ProblemDefinition_addRef(
  bHYPRE_ProblemDefinition self);

void
bHYPRE_ProblemDefinition_deleteRef(
  bHYPRE_ProblemDefinition self);

SIDL_bool
bHYPRE_ProblemDefinition_isSame(
  bHYPRE_ProblemDefinition self,
  SIDL_BaseInterface iobj);

SIDL_BaseInterface
bHYPRE_ProblemDefinition_queryInt(
  bHYPRE_ProblemDefinition self,
  const char* name);

SIDL_bool
bHYPRE_ProblemDefinition_isType(
  bHYPRE_ProblemDefinition self,
  const char* name);

SIDL_ClassInfo
bHYPRE_ProblemDefinition_getClassInfo(
  bHYPRE_ProblemDefinition self);

/**
 * Set the MPI Communicator.
 * 
 */
int32_t
bHYPRE_ProblemDefinition_SetCommunicator(
  bHYPRE_ProblemDefinition self,
  void* mpi_comm);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */
int32_t
bHYPRE_ProblemDefinition_Initialize(
  bHYPRE_ProblemDefinition self);

/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */
int32_t
bHYPRE_ProblemDefinition_Assemble(
  bHYPRE_ProblemDefinition self);

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
bHYPRE_ProblemDefinition_GetObject(
  bHYPRE_ProblemDefinition self,
  SIDL_BaseInterface* A);

/**
 * Cast method for interface and class type conversions.
 */
bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
bHYPRE_ProblemDefinition__cast2(
  void* obj,
  const char* type);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_createCol(int32_t        dimen,
                                          const int32_t lower[],
                                          const int32_t upper[]);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_createRow(int32_t        dimen,
                                          const int32_t lower[],
                                          const int32_t upper[]);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_create1d(int32_t len);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_create2dCol(int32_t m, int32_t n);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_create2dRow(int32_t m, int32_t n);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_borrow(bHYPRE_ProblemDefinition*firstElement,
                                       int32_t       dimen,
const int32_t lower[],
const int32_t upper[],
const int32_t stride[]);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_smartCopy(struct 
  bHYPRE_ProblemDefinition__array *array);

void
bHYPRE_ProblemDefinition__array_addRef(struct bHYPRE_ProblemDefinition__array* 
  array);

void
bHYPRE_ProblemDefinition__array_deleteRef(struct 
  bHYPRE_ProblemDefinition__array* array);

bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__array_get1(const struct 
  bHYPRE_ProblemDefinition__array* array,
                                     const int32_t i1);

bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__array_get2(const struct 
  bHYPRE_ProblemDefinition__array* array,
                                     const int32_t i1,
                                     const int32_t i2);

bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__array_get3(const struct 
  bHYPRE_ProblemDefinition__array* array,
                                     const int32_t i1,
                                     const int32_t i2,
                                     const int32_t i3);

bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__array_get4(const struct 
  bHYPRE_ProblemDefinition__array* array,
                                     const int32_t i1,
                                     const int32_t i2,
                                     const int32_t i3,
                                     const int32_t i4);

bHYPRE_ProblemDefinition
bHYPRE_ProblemDefinition__array_get(const struct 
  bHYPRE_ProblemDefinition__array* array,
                                    const int32_t indices[]);

void
bHYPRE_ProblemDefinition__array_set1(struct bHYPRE_ProblemDefinition__array* 
  array,
                                     const int32_t i1,
                                     bHYPRE_ProblemDefinition const value);

void
bHYPRE_ProblemDefinition__array_set2(struct bHYPRE_ProblemDefinition__array* 
  array,
                                     const int32_t i1,
                                     const int32_t i2,
                                     bHYPRE_ProblemDefinition const value);

void
bHYPRE_ProblemDefinition__array_set3(struct bHYPRE_ProblemDefinition__array* 
  array,
                                     const int32_t i1,
                                     const int32_t i2,
                                     const int32_t i3,
                                     bHYPRE_ProblemDefinition const value);

void
bHYPRE_ProblemDefinition__array_set4(struct bHYPRE_ProblemDefinition__array* 
  array,
                                     const int32_t i1,
                                     const int32_t i2,
                                     const int32_t i3,
                                     const int32_t i4,
                                     bHYPRE_ProblemDefinition const value);

void
bHYPRE_ProblemDefinition__array_set(struct bHYPRE_ProblemDefinition__array* 
  array,
                                    const int32_t indices[],
                                    bHYPRE_ProblemDefinition const value);

int32_t
bHYPRE_ProblemDefinition__array_dimen(const struct 
  bHYPRE_ProblemDefinition__array* array);

int32_t
bHYPRE_ProblemDefinition__array_lower(const struct 
  bHYPRE_ProblemDefinition__array* array,
                                      const int32_t ind);

int32_t
bHYPRE_ProblemDefinition__array_upper(const struct 
  bHYPRE_ProblemDefinition__array* array,
                                      const int32_t ind);

int32_t
bHYPRE_ProblemDefinition__array_stride(const struct 
  bHYPRE_ProblemDefinition__array* array,
                                       const int32_t ind);

int
bHYPRE_ProblemDefinition__array_isColumnOrder(const struct 
  bHYPRE_ProblemDefinition__array* array);

int
bHYPRE_ProblemDefinition__array_isRowOrder(const struct 
  bHYPRE_ProblemDefinition__array* array);

void
bHYPRE_ProblemDefinition__array_slice(const struct 
  bHYPRE_ProblemDefinition__array* src,
                                            int32_t        dimen,
                                            const int32_t  numElem[],
                                            const int32_t  *srcStart,
                                            const int32_t  *srcStride,
                                            const int32_t  *newStart);

void
bHYPRE_ProblemDefinition__array_copy(const struct 
  bHYPRE_ProblemDefinition__array* src,
                                           struct 
  bHYPRE_ProblemDefinition__array* dest);

struct bHYPRE_ProblemDefinition__array*
bHYPRE_ProblemDefinition__array_ensure(struct bHYPRE_ProblemDefinition__array* 
  src,
                                       int32_t dimen,
                                       int     ordering);

#ifdef __cplusplus
}
#endif
#endif
