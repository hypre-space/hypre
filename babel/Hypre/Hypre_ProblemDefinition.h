/*
 * File:          Hypre_ProblemDefinition.h
 * Symbol:        Hypre.ProblemDefinition-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.6.3
 * SIDL Created:  20021001 09:48:43 PDT
 * Generated:     20021001 09:48:51 PDT
 * Description:   Client-side glue code for Hypre.ProblemDefinition
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_ProblemDefinition_h
#define included_Hypre_ProblemDefinition_h

/**
 * Symbol "Hypre.ProblemDefinition" (version 0.1.5)
 * 
 * <p>The purpose of a ProblemDefinition is to:</p>
 * <ul>
 * <li>present the user with a particular view of how to define
 *     a problem</li>
 * <li>construct and return a "problem object"</li>
 * </ul>
 * 
 * <p>A "problem object" is an intentionally vague term that corresponds
 * to any useful object used to define a problem.  Prime examples are:</p>
 * <ul>
 * <li>a LinearOperator object, i.e., something with a matvec</li>
 * <li>a MatrixAccess object, i.e., something with a getrow</li>
 * <li>a Vector, i.e., something with a dot, axpy, ...</li>
 * </ul>
 * 
 * <p>Note that the terms "Initialize" and "Assemble" are reserved here
 * for defining problem objects through a particular user interface.</p>
 */
struct Hypre_ProblemDefinition__object;
struct Hypre_ProblemDefinition__array;
typedef struct Hypre_ProblemDefinition__object* Hypre_ProblemDefinition;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * <p>
 * Add one to the intrinsic reference count in the underlying object.
 * Object in <code>SIDL</code> have an intrinsic reference count.
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
Hypre_ProblemDefinition_addReference(
  Hypre_ProblemDefinition self);

/**
 * Method:  SetCommunicator
 */
int32_t
Hypre_ProblemDefinition_SetCommunicator(
  Hypre_ProblemDefinition self,
  void* mpi_comm);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
Hypre_ProblemDefinition_isInstanceOf(
  Hypre_ProblemDefinition self,
  const char* name);

/**
 * The problem definition interface is a "builder" that creates an object
 * that contains the problem definition information, e.g. a matrix. To
 * perform subsequent operations with that object, it must be returned from
 * the problem definition object. "GetObject" performs this function.
 * <note>At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface. QueryInterface or Cast must
 * be used on the returned object to convert it into a known type.</note>
 * 
 * 
 */
int32_t
Hypre_ProblemDefinition_GetObject(
  Hypre_ProblemDefinition self,
  SIDL_BaseInterface* A);

/**
 * Finalize the construction of an object before using, either for
 * the first time or on subsequent uses. "Initialize" and "Assemble"
 * always appear in a matched set, with Initialize preceding Assemble. Values
 * can only be set in between a call to Initialize and Assemble.
 * 
 * 
 */
int32_t
Hypre_ProblemDefinition_Assemble(
  Hypre_ProblemDefinition self);

/**
 * Check whether the object can support the specified interface or
 * class.  If the <code>SIDL</code> type name in <code>name</code>
 * is supported, then a reference to that object is returned with the
 * reference count incremented.  The callee will be responsible for
 * calling <code>deleteReference</code> on the returned object.  If
 * the specified type is not supported, then a null reference is
 * returned.
 */
SIDL_BaseInterface
Hypre_ProblemDefinition_queryInterface(
  Hypre_ProblemDefinition self,
  const char* name);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */
int32_t
Hypre_ProblemDefinition_Initialize(
  Hypre_ProblemDefinition self);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
Hypre_ProblemDefinition_isSame(
  Hypre_ProblemDefinition self,
  SIDL_BaseInterface iobj);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
Hypre_ProblemDefinition_deleteReference(
  Hypre_ProblemDefinition self);

/**
 * Cast method for interface and class type conversions.
 */
Hypre_ProblemDefinition
Hypre_ProblemDefinition__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
Hypre_ProblemDefinition__cast2(
  void* obj,
  const char* type);

/**
 * Constructor for a new array.
 */
struct Hypre_ProblemDefinition__array*
Hypre_ProblemDefinition__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Constructor to borrow array data.
 */
struct Hypre_ProblemDefinition__array*
Hypre_ProblemDefinition__array_borrow(
  struct Hypre_ProblemDefinition__object** firstElement,
  int32_t                                  dimen,
  const int32_t                            lower[],
  const int32_t                            upper[],
  const int32_t                            stride[]);

/**
 * Destructor for the array.
 */
void
Hypre_ProblemDefinition__array_destroy(
  struct Hypre_ProblemDefinition__array* array);

/**
 * Return the array dimension.
 */
int32_t
Hypre_ProblemDefinition__array_dimen(const struct 
  Hypre_ProblemDefinition__array *array);

/**
 * Return the lower bounds of the array.
 */
int32_t
Hypre_ProblemDefinition__array_lower(const struct 
  Hypre_ProblemDefinition__array *array, int32_t ind);

/**
 * Return the upper bounds of the array.
 */
int32_t
Hypre_ProblemDefinition__array_upper(const struct 
  Hypre_ProblemDefinition__array *array, int32_t ind);

/**
 * Return an array element (int[] indices).
 */
struct Hypre_ProblemDefinition__object*
Hypre_ProblemDefinition__array_get(
  const struct Hypre_ProblemDefinition__array* array,
  const int32_t                                indices[]);

/**
 * Return an array element (integer indices).
 */
struct Hypre_ProblemDefinition__object*
Hypre_ProblemDefinition__array_get4(
  const struct Hypre_ProblemDefinition__array* array,
  int32_t                                      i1,
  int32_t                                      i2,
  int32_t                                      i3,
  int32_t                                      i4);

/**
 * Set an array element (int[] indices).
 */
void
Hypre_ProblemDefinition__array_set(
  struct Hypre_ProblemDefinition__array*  array,
  const int32_t                           indices[],
  struct Hypre_ProblemDefinition__object* value);

/**
 * Set an array element (integer indices).
 */
void
Hypre_ProblemDefinition__array_set4(
  struct Hypre_ProblemDefinition__array*  array,
  int32_t                                 i1,
  int32_t                                 i2,
  int32_t                                 i3,
  int32_t                                 i4,
  struct Hypre_ProblemDefinition__object* value);

/*
 * Macros to simplify access to the array.
 */

#define Hypre_ProblemDefinition__array_get1(a,i1) \
  Hypre_ProblemDefinition__array_get4(a,i1,0,0,0)

#define Hypre_ProblemDefinition__array_get2(a,i1,i2) \
  Hypre_ProblemDefinition__array_get4(a,i1,i2,0,0)

#define Hypre_ProblemDefinition__array_get3(a,i1,i2,i3) \
  Hypre_ProblemDefinition__array_get4(a,i1,i2,i3,0)

#define Hypre_ProblemDefinition__array_set1(a,i1,v) \
  Hypre_ProblemDefinition__array_set4(a,i1,0,0,0,v)

#define Hypre_ProblemDefinition__array_set2(a,i1,i2,v) \
  Hypre_ProblemDefinition__array_set4(a,i1,i2,0,0,v)

#define Hypre_ProblemDefinition__array_set3(a,i1,i2,i3,v) \
  Hypre_ProblemDefinition__array_set4(a,i1,i2,i3,0,v)

#ifdef __cplusplus
}
#endif
#endif
