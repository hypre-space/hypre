/*
 * File:          Hypre_StructToIJVector_Impl.c
 * Symbol:        Hypre.StructToIJVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20021001 09:48:43 PDT
 * Generated:     20021001 09:48:53 PDT
 * Description:   Server-side implementation for Hypre.StructToIJVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.StructToIJVector" (version 0.1.5)
 */

#include "Hypre_StructToIJVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJVector__ctor"

void
impl_Hypre_StructToIJVector__ctor(
  Hypre_StructToIJVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJVector__dtor"

void
impl_Hypre_StructToIJVector__dtor(
  Hypre_StructToIJVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector._dtor) */
}

/*
 * Finalize the construction of an object before using, either for
 * the first time or on subsequent uses. "Initialize" and "Assemble"
 * always appear in a matched set, with Initialize preceding Assemble. Values
 * can only be set in between a call to Initialize and Assemble.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJVector_Assemble"

int32_t
impl_Hypre_StructToIJVector_Assemble(
  Hypre_StructToIJVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector.Assemble) */
}

/*
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

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJVector_GetObject"

int32_t
impl_Hypre_StructToIJVector_GetObject(
  Hypre_StructToIJVector self,
  SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector.GetObject) */
  /* Insert the implementation of the GetObject method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector.GetObject) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJVector_Initialize"

int32_t
impl_Hypre_StructToIJVector_Initialize(
  Hypre_StructToIJVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector.Initialize) */
}

/*
 * Method:  SetBoxValues
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJVector_SetBoxValues"

int32_t
impl_Hypre_StructToIJVector_SetBoxValues(
  Hypre_StructToIJVector self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector.SetBoxValues) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJVector_SetCommunicator"

int32_t
impl_Hypre_StructToIJVector_SetCommunicator(
  Hypre_StructToIJVector self,
  void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector.SetCommunicator) */
}

/*
 * Method:  SetGrid
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJVector_SetGrid"

int32_t
impl_Hypre_StructToIJVector_SetGrid(
  Hypre_StructToIJVector self,
  Hypre_StructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector.SetGrid) */
}

/*
 * Method:  SetIJVector
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJVector_SetIJVector"

int32_t
impl_Hypre_StructToIJVector_SetIJVector(
  Hypre_StructToIJVector self,
  Hypre_IJBuildVector I)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector.SetIJVector) */
  /* Insert the implementation of the SetIJVector method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector.SetIJVector) */
}

/*
 * Method:  SetStencil
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJVector_SetStencil"

int32_t
impl_Hypre_StructToIJVector_SetStencil(
  Hypre_StructToIJVector self,
  Hypre_StructStencil stencil)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector.SetStencil) */
  /* Insert the implementation of the SetStencil method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector.SetStencil) */
}

/*
 * Method:  SetValue
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJVector_SetValue"

int32_t
impl_Hypre_StructToIJVector_SetValue(
  Hypre_StructToIJVector self,
  struct SIDL_int__array* grid_index,
  double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJVector.SetValue) */
  /* Insert the implementation of the SetValue method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJVector.SetValue) */
}
