/*
 * File:          Hypre_StructuredGridBuildVector.h
 * Symbol:        Hypre.StructuredGridBuildVector-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:31 PDT
 * Description:   Client-side glue code for Hypre.StructuredGridBuildVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_Hypre_StructuredGridBuildVector_h
#define included_Hypre_StructuredGridBuildVector_h

/**
 * Symbol "Hypre.StructuredGridBuildVector" (version 0.1.5)
 */
struct Hypre_StructuredGridBuildVector__object;
struct Hypre_StructuredGridBuildVector__array;
typedef struct Hypre_StructuredGridBuildVector__object* 
  Hypre_StructuredGridBuildVector;

/*
 * Includes for all header dependencies.
 */

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifndef included_Hypre_StructGrid_h
#include "Hypre_StructGrid.h"
#endif
#ifndef included_Hypre_StructStencil_h
#include "Hypre_StructStencil.h"
#endif
#ifndef included_SIDL_BaseInterface_h
#include "SIDL_BaseInterface.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Method:  SetValue
 */
int32_t
Hypre_StructuredGridBuildVector_SetValue(
  Hypre_StructuredGridBuildVector self,
  struct SIDL_int__array* grid_index,
  double value);

/**
 * Method:  SetBoxValues
 */
int32_t
Hypre_StructuredGridBuildVector_SetBoxValues(
  Hypre_StructuredGridBuildVector self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  struct SIDL_double__array* values);

/**
 * Method:  SetStencil
 */
int32_t
Hypre_StructuredGridBuildVector_SetStencil(
  Hypre_StructuredGridBuildVector self,
  Hypre_StructStencil stencil);

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
Hypre_StructuredGridBuildVector_addReference(
  Hypre_StructuredGridBuildVector self);

/**
 * Method:  SetGrid
 */
int32_t
Hypre_StructuredGridBuildVector_SetGrid(
  Hypre_StructuredGridBuildVector self,
  Hypre_StructGrid grid);

/**
 * Method:  SetCommunicator
 */
int32_t
Hypre_StructuredGridBuildVector_SetCommunicator(
  Hypre_StructuredGridBuildVector self,
  void* mpi_comm);

/**
 * Return whether this object is an instance of the specified type.
 * The string name must be the <code>SIDL</code> type name.  This
 * routine will return <code>true</code> if and only if a cast to
 * the string type name would succeed.
 */
SIDL_bool
Hypre_StructuredGridBuildVector_isInstanceOf(
  Hypre_StructuredGridBuildVector self,
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
Hypre_StructuredGridBuildVector_GetObject(
  Hypre_StructuredGridBuildVector self,
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
Hypre_StructuredGridBuildVector_Assemble(
  Hypre_StructuredGridBuildVector self);

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
Hypre_StructuredGridBuildVector_queryInterface(
  Hypre_StructuredGridBuildVector self,
  const char* name);

/**
 * Decrease by one the intrinsic reference count in the underlying
 * object, and delete the object if the reference is non-positive.
 * Objects in <code>SIDL</code> have an intrinsic reference count.
 * Clients should call this method whenever they remove a
 * reference to an object or interface.
 */
void
Hypre_StructuredGridBuildVector_deleteReference(
  Hypre_StructuredGridBuildVector self);

/**
 * Return true if and only if <code>obj</code> refers to the same
 * object as this object.
 */
SIDL_bool
Hypre_StructuredGridBuildVector_isSame(
  Hypre_StructuredGridBuildVector self,
  SIDL_BaseInterface iobj);

/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */
int32_t
Hypre_StructuredGridBuildVector_Initialize(
  Hypre_StructuredGridBuildVector self);

/**
 * Cast method for interface and class type conversions.
 */
Hypre_StructuredGridBuildVector
Hypre_StructuredGridBuildVector__cast(
  void* obj);

/**
 * String cast method for interface and class type conversions.
 */
void*
Hypre_StructuredGridBuildVector__cast2(
  void* obj,
  const char* type);

/**
 * Constructor for a new array.
 */
struct Hypre_StructuredGridBuildVector__array*
Hypre_StructuredGridBuildVector__array_create(
  int32_t       dimen,
  const int32_t lower[],
  const int32_t upper[]);

/**
 * Constructor to borrow array data.
 */
struct Hypre_StructuredGridBuildVector__array*
Hypre_StructuredGridBuildVector__array_borrow(
  struct Hypre_StructuredGridBuildVector__object** firstElement,
  int32_t                                          dimen,
  const int32_t                                    lower[],
  const int32_t                                    upper[],
  const int32_t                                    stride[]);

/**
 * Destructor for the array.
 */
void
Hypre_StructuredGridBuildVector__array_destroy(
  struct Hypre_StructuredGridBuildVector__array* array);

/**
 * Return the array dimension.
 */
int32_t
Hypre_StructuredGridBuildVector__array_dimen(const struct 
  Hypre_StructuredGridBuildVector__array *array);

/**
 * Return the lower bounds of the array.
 */
int32_t
Hypre_StructuredGridBuildVector__array_lower(const struct 
  Hypre_StructuredGridBuildVector__array *array, int32_t ind);

/**
 * Return the upper bounds of the array.
 */
int32_t
Hypre_StructuredGridBuildVector__array_upper(const struct 
  Hypre_StructuredGridBuildVector__array *array, int32_t ind);

/**
 * Return an array element (int[] indices).
 */
struct Hypre_StructuredGridBuildVector__object*
Hypre_StructuredGridBuildVector__array_get(
  const struct Hypre_StructuredGridBuildVector__array* array,
  const int32_t                                        indices[]);

/**
 * Return an array element (integer indices).
 */
struct Hypre_StructuredGridBuildVector__object*
Hypre_StructuredGridBuildVector__array_get4(
  const struct Hypre_StructuredGridBuildVector__array* array,
  int32_t                                              i1,
  int32_t                                              i2,
  int32_t                                              i3,
  int32_t                                              i4);

/**
 * Set an array element (int[] indices).
 */
void
Hypre_StructuredGridBuildVector__array_set(
  struct Hypre_StructuredGridBuildVector__array*  array,
  const int32_t                                   indices[],
  struct Hypre_StructuredGridBuildVector__object* value);

/**
 * Set an array element (integer indices).
 */
void
Hypre_StructuredGridBuildVector__array_set4(
  struct Hypre_StructuredGridBuildVector__array*  array,
  int32_t                                         i1,
  int32_t                                         i2,
  int32_t                                         i3,
  int32_t                                         i4,
  struct Hypre_StructuredGridBuildVector__object* value);

/*
 * Macros to simplify access to the array.
 */

#define Hypre_StructuredGridBuildVector__array_get1(a,i1) \
  Hypre_StructuredGridBuildVector__array_get4(a,i1,0,0,0)

#define Hypre_StructuredGridBuildVector__array_get2(a,i1,i2) \
  Hypre_StructuredGridBuildVector__array_get4(a,i1,i2,0,0)

#define Hypre_StructuredGridBuildVector__array_get3(a,i1,i2,i3) \
  Hypre_StructuredGridBuildVector__array_get4(a,i1,i2,i3,0)

#define Hypre_StructuredGridBuildVector__array_set1(a,i1,v) \
  Hypre_StructuredGridBuildVector__array_set4(a,i1,0,0,0,v)

#define Hypre_StructuredGridBuildVector__array_set2(a,i1,i2,v) \
  Hypre_StructuredGridBuildVector__array_set4(a,i1,i2,0,0,v)

#define Hypre_StructuredGridBuildVector__array_set3(a,i1,i2,i3,v) \
  Hypre_StructuredGridBuildVector__array_set4(a,i1,i2,i3,0,v)

#ifdef __cplusplus
}
#endif
#endif
