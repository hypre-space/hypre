/*
 * File:          Hypre_StructToIJMatrix_Impl.c
 * Symbol:        Hypre.StructToIJMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:34 PDT
 * Description:   Server-side implementation for Hypre.StructToIJMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.StructToIJMatrix" (version 0.1.5)
 * 
 * This class implements the StructuredGrid user interface, but builds
 * an unstructured matrix behind the curtain.  It does this by using
 * an IJBuildMatrix (e.g., ParCSRMatrix, PETScMatrix, ...)
 * specified by the user with an extra method ...
 */

#include "Hypre_StructToIJMatrix_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJMatrix__ctor"

void
impl_Hypre_StructToIJMatrix__ctor(
  Hypre_StructToIJMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJMatrix__dtor"

void
impl_Hypre_StructToIJMatrix__dtor(
  Hypre_StructToIJMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix._dtor) */
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
#define __FUNC__ "impl_Hypre_StructToIJMatrix_Assemble"

int32_t
impl_Hypre_StructToIJMatrix_Assemble(
  Hypre_StructToIJMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix.Assemble) */
  /* Insert the implementation of the Assemble method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix.Assemble) */
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
#define __FUNC__ "impl_Hypre_StructToIJMatrix_GetObject"

int32_t
impl_Hypre_StructToIJMatrix_GetObject(
  Hypre_StructToIJMatrix self,
  SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix.GetObject) */
  /* Insert the implementation of the GetObject method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix.GetObject) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJMatrix_Initialize"

int32_t
impl_Hypre_StructToIJMatrix_Initialize(
  Hypre_StructToIJMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix.Initialize) */
  /* Insert the implementation of the Initialize method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix.Initialize) */
}

/*
 * Method:  SetBoxValues
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJMatrix_SetBoxValues"

int32_t
impl_Hypre_StructToIJMatrix_SetBoxValues(
  Hypre_StructToIJMatrix self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper,
  int32_t num_stencil_indices,
  struct SIDL_int__array* stencil_indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix.SetBoxValues) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJMatrix_SetCommunicator"

int32_t
impl_Hypre_StructToIJMatrix_SetCommunicator(
  Hypre_StructToIJMatrix self,
  void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix.SetCommunicator) */
}

/*
 * Method:  SetGrid
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJMatrix_SetGrid"

int32_t
impl_Hypre_StructToIJMatrix_SetGrid(
  Hypre_StructToIJMatrix self,
  Hypre_StructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix.SetGrid) */
}

/*
 * Method:  SetIJMatrix
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJMatrix_SetIJMatrix"

int32_t
impl_Hypre_StructToIJMatrix_SetIJMatrix(
  Hypre_StructToIJMatrix self,
  Hypre_IJBuildMatrix I)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix.SetIJMatrix) */
  /* Insert the implementation of the SetIJMatrix method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix.SetIJMatrix) */
}

/*
 * Method:  SetNumGhost
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJMatrix_SetNumGhost"

int32_t
impl_Hypre_StructToIJMatrix_SetNumGhost(
  Hypre_StructToIJMatrix self,
  struct SIDL_int__array* num_ghost)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix.SetNumGhost) */
  /* Insert the implementation of the SetNumGhost method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix.SetNumGhost) */
}

/*
 * Method:  SetStencil
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJMatrix_SetStencil"

int32_t
impl_Hypre_StructToIJMatrix_SetStencil(
  Hypre_StructToIJMatrix self,
  Hypre_StructStencil stencil)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix.SetStencil) */
  /* Insert the implementation of the SetStencil method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix.SetStencil) */
}

/*
 * Method:  SetSymmetric
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJMatrix_SetSymmetric"

int32_t
impl_Hypre_StructToIJMatrix_SetSymmetric(
  Hypre_StructToIJMatrix self,
  int32_t symmetric)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix.SetSymmetric) */
  /* Insert the implementation of the SetSymmetric method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix.SetSymmetric) */
}

/*
 * Method:  SetValues
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructToIJMatrix_SetValues"

int32_t
impl_Hypre_StructToIJMatrix_SetValues(
  Hypre_StructToIJMatrix self,
  struct SIDL_int__array* index,
  int32_t num_stencil_indices,
  struct SIDL_int__array* stencil_indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructToIJMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructToIJMatrix.SetValues) */
}
