/*
 * File:          Hypre_CoefficientAccess.h
 * Symbol:        Hypre.CoefficientAccess-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.6.3
 * SIDL Created:  20020904 10:05:21 PDT
 * Generated:     20020904 10:05:27 PDT
 * Description:   Client-side glue code for Hypre.CoefficientAccess
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_CoefficientAccess_h
#define included_Hypre_CoefficientAccess_h

/**
 * Symbol "Hypre.CoefficientAccess" (version 0.1.5)
 * 
 * The GetRow method will allocate space for its two output arrays on
 * the first call.  The space will be reused on subsequent calls.
 * Thus the user must not delete them, yet must not depend on the
 * data from GetRow to persist beyond the next GetRow call.
 */
struct Hypre_CoefficientAccess__object;
struct Hypre_CoefficientAccess__array;
typedef struct Hypre_CoefficientAccess__object* Hypre_CoefficientAccess;

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
Hypre_CoefficientAccess_addReference(
  Hypre_CoefficientAccess self);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
Hypre_CoefficientAccess_isInstanceOf(
  Hypre_CoefficientAccess self,
  const char* name);

/**
 * Method:  GetRow
 */
int32_t
Hypre_CoefficientAccess_GetRow(
  Hypre_CoefficientAccess self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values);

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
Hypre_CoefficientAccess_queryInterface(
  Hypre_CoefficientAccess self,
  const char* name);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
Hypre_CoefficientAccess_isSame(
  Hypre_CoefficientAccess self,
  SIDL_BaseInterface iobj);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
Hypre_CoefficientAccess_deleteReference(
  Hypre_CoefficientAccess self);

/**
 * Cast method for interface and class type conversions.
 */
Hypre_CoefficientAccess
Hypre_CoefficientAccess__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
Hypre_CoefficientAccess__cast2(
  void* obj,
  const char* type);

/**
 * Constructor for a new array.
 */
struct Hypre_CoefficientAccess__array*
Hypre_CoefficientAccess__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Constructor to borrow array data.
 */
struct Hypre_CoefficientAccess__array*
Hypre_CoefficientAccess__array_borrow(
  struct Hypre_CoefficientAccess__object** firstElement,
  int32_t                                  dimen,
  const int32_t                            lower[],
  const int32_t                            upper[],
  const int32_t                            stride[]);

/**
 * Destructor for the array.
 */
void
Hypre_CoefficientAccess__array_destroy(
  struct Hypre_CoefficientAccess__array* array);

/**
 * Return the array dimension.
 */
int32_t
Hypre_CoefficientAccess__array_dimen(const struct 
  Hypre_CoefficientAccess__array *array);

/**
 * Return the lower bounds of the array.
 */
int32_t
Hypre_CoefficientAccess__array_lower(const struct 
  Hypre_CoefficientAccess__array *array, int32_t ind);

/**
 * Return the upper bounds of the array.
 */
int32_t
Hypre_CoefficientAccess__array_upper(const struct 
  Hypre_CoefficientAccess__array *array, int32_t ind);

/**
 * Return an array element (int[] indices).
 */
struct Hypre_CoefficientAccess__object*
Hypre_CoefficientAccess__array_get(
  const struct Hypre_CoefficientAccess__array* array,
  const int32_t                                indices[]);

/**
 * Return an array element (integer indices).
 */
struct Hypre_CoefficientAccess__object*
Hypre_CoefficientAccess__array_get4(
  const struct Hypre_CoefficientAccess__array* array,
  int32_t                                      i1,
  int32_t                                      i2,
  int32_t                                      i3,
  int32_t                                      i4);

/**
 * Set an array element (int[] indices).
 */
void
Hypre_CoefficientAccess__array_set(
  struct Hypre_CoefficientAccess__array*  array,
  const int32_t                           indices[],
  struct Hypre_CoefficientAccess__object* value);

/**
 * Set an array element (integer indices).
 */
void
Hypre_CoefficientAccess__array_set4(
  struct Hypre_CoefficientAccess__array*  array,
  int32_t                                 i1,
  int32_t                                 i2,
  int32_t                                 i3,
  int32_t                                 i4,
  struct Hypre_CoefficientAccess__object* value);

/*
 * Macros to simplify access to the array.
 */

#define Hypre_CoefficientAccess__array_get1(a,i1) \
  Hypre_CoefficientAccess__array_get4(a,i1,0,0,0)

#define Hypre_CoefficientAccess__array_get2(a,i1,i2) \
  Hypre_CoefficientAccess__array_get4(a,i1,i2,0,0)

#define Hypre_CoefficientAccess__array_get3(a,i1,i2,i3) \
  Hypre_CoefficientAccess__array_get4(a,i1,i2,i3,0)

#define Hypre_CoefficientAccess__array_set1(a,i1,v) \
  Hypre_CoefficientAccess__array_set4(a,i1,0,0,0,v)

#define Hypre_CoefficientAccess__array_set2(a,i1,i2,v) \
  Hypre_CoefficientAccess__array_set4(a,i1,i2,0,0,v)

#define Hypre_CoefficientAccess__array_set3(a,i1,i2,i3,v) \
  Hypre_CoefficientAccess__array_set4(a,i1,i2,i3,0,v)

#ifdef __cplusplus
}
#endif
#endif
