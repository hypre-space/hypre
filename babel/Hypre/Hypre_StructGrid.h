/*
 * File:          Hypre_StructGrid.h
 * Symbol:        Hypre.StructGrid-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20021001 09:48:43 PDT
 * Generated:     20021001 09:48:49 PDT
 * Description:   Client-side glue code for Hypre.StructGrid
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_StructGrid_h
#define included_Hypre_StructGrid_h

/**
 * Symbol "Hypre.StructGrid" (version 0.1.5)
 * 
 * Define a structured grid class.
 */
struct Hypre_StructGrid__object;
struct Hypre_StructGrid__array;
typedef struct Hypre_StructGrid__object* Hypre_StructGrid;

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
 * Constructor function for the class.
 */
Hypre_StructGrid
Hypre_StructGrid__create(void);

/**
 * Method:  SetPeriodic
 */
int32_t
Hypre_StructGrid_SetPeriodic(
  Hypre_StructGrid self,
  struct SIDL_int__array* periodic);

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
Hypre_StructGrid_addReference(
  Hypre_StructGrid self);

/**
 * Method:  SetExtents
 */
int32_t
Hypre_StructGrid_SetExtents(
  Hypre_StructGrid self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper);

/**
 * Method:  SetCommunicator
 */
int32_t
Hypre_StructGrid_SetCommunicator(
  Hypre_StructGrid self,
  void* MPI_comm);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
Hypre_StructGrid_isInstanceOf(
  Hypre_StructGrid self,
  const char* name);

/**
 * Method:  SetDimension
 */
int32_t
Hypre_StructGrid_SetDimension(
  Hypre_StructGrid self,
  int32_t dim);

/**
 * Method:  Assemble
 */
int32_t
Hypre_StructGrid_Assemble(
  Hypre_StructGrid self);

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
Hypre_StructGrid_queryInterface(
  Hypre_StructGrid self,
  const char* name);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
Hypre_StructGrid_deleteReference(
  Hypre_StructGrid self);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
Hypre_StructGrid_isSame(
  Hypre_StructGrid self,
  SIDL_BaseInterface iobj);

/**
 * Cast method for interface and class type conversions.
 */
Hypre_StructGrid
Hypre_StructGrid__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
Hypre_StructGrid__cast2(
  void* obj,
  const char* type);

/**
 * Constructor for a new array.
 */
struct Hypre_StructGrid__array*
Hypre_StructGrid__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Constructor to borrow array data.
 */
struct Hypre_StructGrid__array*
Hypre_StructGrid__array_borrow(
  struct Hypre_StructGrid__object** firstElement,
  int32_t                           dimen,
  const int32_t                     lower[],
  const int32_t                     upper[],
  const int32_t                     stride[]);

/**
 * Destructor for the array.
 */
void
Hypre_StructGrid__array_destroy(
  struct Hypre_StructGrid__array* array);

/**
 * Return the array dimension.
 */
int32_t
Hypre_StructGrid__array_dimen(const struct Hypre_StructGrid__array *array);

/**
 * Return the lower bounds of the array.
 */
int32_t
Hypre_StructGrid__array_lower(const struct Hypre_StructGrid__array *array,
  int32_t ind);

/**
 * Return the upper bounds of the array.
 */
int32_t
Hypre_StructGrid__array_upper(const struct Hypre_StructGrid__array *array,
  int32_t ind);

/**
 * Return an array element (int[] indices).
 */
struct Hypre_StructGrid__object*
Hypre_StructGrid__array_get(
  const struct Hypre_StructGrid__array* array,
  const int32_t                         indices[]);

/**
 * Return an array element (integer indices).
 */
struct Hypre_StructGrid__object*
Hypre_StructGrid__array_get4(
  const struct Hypre_StructGrid__array* array,
  int32_t                               i1,
  int32_t                               i2,
  int32_t                               i3,
  int32_t                               i4);

/**
 * Set an array element (int[] indices).
 */
void
Hypre_StructGrid__array_set(
  struct Hypre_StructGrid__array*  array,
  const int32_t                    indices[],
  struct Hypre_StructGrid__object* value);

/**
 * Set an array element (integer indices).
 */
void
Hypre_StructGrid__array_set4(
  struct Hypre_StructGrid__array*  array,
  int32_t                          i1,
  int32_t                          i2,
  int32_t                          i3,
  int32_t                          i4,
  struct Hypre_StructGrid__object* value);

/*
 * Macros to simplify access to the array.
 */

#define Hypre_StructGrid__array_get1(a,i1) \
  Hypre_StructGrid__array_get4(a,i1,0,0,0)

#define Hypre_StructGrid__array_get2(a,i1,i2) \
  Hypre_StructGrid__array_get4(a,i1,i2,0,0)

#define Hypre_StructGrid__array_get3(a,i1,i2,i3) \
  Hypre_StructGrid__array_get4(a,i1,i2,i3,0)

#define Hypre_StructGrid__array_set1(a,i1,v) \
  Hypre_StructGrid__array_set4(a,i1,0,0,0,v)

#define Hypre_StructGrid__array_set2(a,i1,i2,v) \
  Hypre_StructGrid__array_set4(a,i1,i2,0,0,v)

#define Hypre_StructGrid__array_set3(a,i1,i2,i3,v) \
  Hypre_StructGrid__array_set4(a,i1,i2,i3,0,v)

#ifdef __cplusplus
}
#endif
#endif
