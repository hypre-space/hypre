/*
 * File:          Hypre_Operator.h
 * Symbol:        Hypre.Operator-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.6.3
 * SIDL Created:  20020522 13:59:35 PDT
 * Generated:     20020522 13:59:43 PDT
 * Description:   Client-side glue code for Hypre.Operator
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_Operator_h
#define included_Hypre_Operator_h

/**
 * Symbol "Hypre.Operator" (version 0.1.5)
 * 
 * An Operator is anything that maps one Vector to another.
 * The terms "Setup" and "Apply" are reserved for Operators.
 */
struct Hypre_Operator__object;
struct Hypre_Operator__array;
typedef struct Hypre_Operator__object* Hypre_Operator;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_Vector_h
#include "Hypre_Vector.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Method:  Apply
 */
int32_t
Hypre_Operator_Apply(
  Hypre_Operator self,
  Hypre_Vector x,
  Hypre_Vector* y);

/**
 * Method:  Setup
 */
int32_t
Hypre_Operator_Setup(
  Hypre_Operator self);

/**
 * Method:  SetIntArrayParameter
 */
int32_t
Hypre_Operator_SetIntArrayParameter(
  Hypre_Operator self,
  const char* name,
  struct SIDL_int__array* value);

/**
 * Method:  SetIntParameter
 */
int32_t
Hypre_Operator_SetIntParameter(
  Hypre_Operator self,
  const char* name,
  int32_t value);

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
Hypre_Operator_addReference(
  Hypre_Operator self);

/**
 * Method:  SetCommunicator
 */
int32_t
Hypre_Operator_SetCommunicator(
  Hypre_Operator self,
  void* comm);

/**
 * Method:  SetStringParameter
 */
int32_t
Hypre_Operator_SetStringParameter(
  Hypre_Operator self,
  const char* name,
  const char* value);

/**
 * Method:  SetDoubleParameter
 */
int32_t
Hypre_Operator_SetDoubleParameter(
  Hypre_Operator self,
  const char* name,
  double value);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
Hypre_Operator_isInstanceOf(
  Hypre_Operator self,
  const char* name);

/**
 * Method:  SetDoubleArrayParameter
 */
int32_t
Hypre_Operator_SetDoubleArrayParameter(
  Hypre_Operator self,
  const char* name,
  struct SIDL_double__array* value);

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
Hypre_Operator_queryInterface(
  Hypre_Operator self,
  const char* name);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
Hypre_Operator_isSame(
  Hypre_Operator self,
  SIDL_BaseInterface iobj);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
Hypre_Operator_deleteReference(
  Hypre_Operator self);

/**
 * Cast method for interface and class type conversions.
 */
Hypre_Operator
Hypre_Operator__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
Hypre_Operator__cast2(
  void* obj,
  const char* type);

/**
 * Constructor for a new array.
 */
struct Hypre_Operator__array*
Hypre_Operator__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Constructor to borrow array data.
 */
struct Hypre_Operator__array*
Hypre_Operator__array_borrow(
  struct Hypre_Operator__object** firstElement,
  int32_t                         dimen,
  const int32_t                   lower[],
  const int32_t                   upper[],
  const int32_t                   stride[]);

/**
 * Destructor for the array.
 */
void
Hypre_Operator__array_destroy(
  struct Hypre_Operator__array* array);

/**
 * Return the array dimension.
 */
int32_t
Hypre_Operator__array_dimen(const struct Hypre_Operator__array *array);

/**
 * Return the lower bounds of the array.
 */
int32_t
Hypre_Operator__array_lower(const struct Hypre_Operator__array *array,
  int32_t ind);

/**
 * Return the upper bounds of the array.
 */
int32_t
Hypre_Operator__array_upper(const struct Hypre_Operator__array *array,
  int32_t ind);

/**
 * Return an array element (int[] indices).
 */
struct Hypre_Operator__object*
Hypre_Operator__array_get(
  const struct Hypre_Operator__array* array,
  const int32_t                       indices[]);

/**
 * Return an array element (integer indices).
 */
struct Hypre_Operator__object*
Hypre_Operator__array_get4(
  const struct Hypre_Operator__array* array,
  int32_t                             i1,
  int32_t                             i2,
  int32_t                             i3,
  int32_t                             i4);

/**
 * Set an array element (int[] indices).
 */
void
Hypre_Operator__array_set(
  struct Hypre_Operator__array*  array,
  const int32_t                  indices[],
  struct Hypre_Operator__object* value);

/**
 * Set an array element (integer indices).
 */
void
Hypre_Operator__array_set4(
  struct Hypre_Operator__array*  array,
  int32_t                        i1,
  int32_t                        i2,
  int32_t                        i3,
  int32_t                        i4,
  struct Hypre_Operator__object* value);

/*
 * Macros to simplify access to the array.
 */

#define Hypre_Operator__array_get1(a,i1) \
  Hypre_Operator__array_get4(a,i1,0,0,0)

#define Hypre_Operator__array_get2(a,i1,i2) \
  Hypre_Operator__array_get4(a,i1,i2,0,0)

#define Hypre_Operator__array_get3(a,i1,i2,i3) \
  Hypre_Operator__array_get4(a,i1,i2,i3,0)

#define Hypre_Operator__array_set1(a,i1,v) \
  Hypre_Operator__array_set4(a,i1,0,0,0,v)

#define Hypre_Operator__array_set2(a,i1,i2,v) \
  Hypre_Operator__array_set4(a,i1,i2,0,0,v)

#define Hypre_Operator__array_set3(a,i1,i2,i3,v) \
  Hypre_Operator__array_set4(a,i1,i2,i3,0,v)

#ifdef __cplusplus
}
#endif
#endif
