/*
 * File:          Hypre_StructVector_Impl.c
 * Symbol:        Hypre.StructVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:18 PST
 * Description:   Server-side implementation for Hypre.StructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.StructVector" (version 0.1.5)
 */

#include "Hypre_StructVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.StructVector._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.StructVector._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector__ctor"

void
impl_Hypre_StructVector__ctor(
  Hypre_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector__dtor"

void
impl_Hypre_StructVector__dtor(
  Hypre_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector._dtor) */
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
#define __FUNC__ "impl_Hypre_StructVector_Assemble"

int32_t
impl_Hypre_StructVector_Assemble(
  Hypre_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.Assemble) */
}

/*
 * y <- a*x + y
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector_Axpy"

int32_t
impl_Hypre_StructVector_Axpy(
  Hypre_StructVector self,
  double a,
  Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.Axpy) */
}

/*
 * y <- 0 (where y=self)
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector_Clear"

int32_t
impl_Hypre_StructVector_Clear(
  Hypre_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.Clear) */
  /* Insert the implementation of the Clear method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.Clear) */
}

/*
 * create an x compatible with y
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector_Clone"

int32_t
impl_Hypre_StructVector_Clone(
  Hypre_StructVector self,
  Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.Clone) */
  /* Insert the implementation of the Clone method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.Clone) */
}

/*
 * y <- x 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector_Copy"

int32_t
impl_Hypre_StructVector_Copy(
  Hypre_StructVector self,
  Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.Copy) */
  /* Insert the implementation of the Copy method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.Copy) */
}

/*
 * d <- (y,x)
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector_Dot"

int32_t
impl_Hypre_StructVector_Dot(
  Hypre_StructVector self,
  Hypre_Vector x,
  double* d)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.Dot) */
  /* Insert the implementation of the Dot method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.Dot) */
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
#define __FUNC__ "impl_Hypre_StructVector_GetObject"

int32_t
impl_Hypre_StructVector_GetObject(
  Hypre_StructVector self,
  SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.GetObject) */
  /* Insert the implementation of the GetObject method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.GetObject) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector_Initialize"

int32_t
impl_Hypre_StructVector_Initialize(
  Hypre_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.Initialize) */
}

/*
 * y <- a*y 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector_Scale"

int32_t
impl_Hypre_StructVector_Scale(
  Hypre_StructVector self,
  double a)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.Scale) */
  /* Insert the implementation of the Scale method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.Scale) */
}

/*
 * Method:  SetBoxValues
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector_SetBoxValues"

int32_t
impl_Hypre_StructVector_SetBoxValues(
  Hypre_StructVector self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.SetBoxValues) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector_SetCommunicator"

int32_t
impl_Hypre_StructVector_SetCommunicator(
  Hypre_StructVector self,
  void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.SetCommunicator) */
}

/*
 * Method:  SetGrid
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector_SetGrid"

int32_t
impl_Hypre_StructVector_SetGrid(
  Hypre_StructVector self,
  Hypre_StructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.SetGrid) */
}

/*
 * Method:  SetStencil
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector_SetStencil"

int32_t
impl_Hypre_StructVector_SetStencil(
  Hypre_StructVector self,
  Hypre_StructStencil stencil)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.SetStencil) */
  /* Insert the implementation of the SetStencil method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.SetStencil) */
}

/*
 * Method:  SetValue
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructVector_SetValue"

int32_t
impl_Hypre_StructVector_SetValue(
  Hypre_StructVector self,
  struct SIDL_int__array* grid_index,
  double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructVector.SetValue) */
  /* Insert the implementation of the SetValue method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructVector.SetValue) */
}
