/*
 * File:          Hypre_StructMatrix_Impl.c
 * Symbol:        Hypre.StructMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.7.4
 * SIDL Created:  20021217 16:01:16 PST
 * Generated:     20021217 16:01:25 PST
 * Description:   Server-side implementation for Hypre.StructMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.7.4
 * source-line   = 426
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.StructMatrix" (version 0.1.5)
 * 
 * A single class that implements both a build interface and an operator
 * interface. It returns itself for <code>GetConstructedObject</code>.
 */

#include "Hypre_StructMatrix_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.StructMatrix._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix__ctor"

void
impl_Hypre_StructMatrix__ctor(
  Hypre_StructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix__dtor"

void
impl_Hypre_StructMatrix__dtor(
  Hypre_StructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix._dtor) */
}

/*
 * Method:  SetCommunicator[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_SetCommunicator"

int32_t
impl_Hypre_StructMatrix_SetCommunicator(
  Hypre_StructMatrix self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_Initialize"

int32_t
impl_Hypre_StructMatrix_Initialize(
  Hypre_StructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.Initialize) */
  /* Insert the implementation of the Initialize method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.Initialize) */
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
#define __FUNC__ "impl_Hypre_StructMatrix_Assemble"

int32_t
impl_Hypre_StructMatrix_Assemble(
  Hypre_StructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.Assemble) */
  /* Insert the implementation of the Assemble method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.Assemble) */
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
#define __FUNC__ "impl_Hypre_StructMatrix_GetObject"

int32_t
impl_Hypre_StructMatrix_GetObject(
  Hypre_StructMatrix self, SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.GetObject) */
  /* Insert the implementation of the GetObject method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.GetObject) */
}

/*
 * Method:  SetGrid[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_SetGrid"

int32_t
impl_Hypre_StructMatrix_SetGrid(
  Hypre_StructMatrix self, Hypre_StructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.SetGrid) */
}

/*
 * Method:  SetStencil[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_SetStencil"

int32_t
impl_Hypre_StructMatrix_SetStencil(
  Hypre_StructMatrix self, Hypre_StructStencil stencil)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.SetStencil) */
  /* Insert the implementation of the SetStencil method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.SetStencil) */
}

/*
 * Method:  SetValues[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_SetValues"

int32_t
impl_Hypre_StructMatrix_SetValues(
  Hypre_StructMatrix self, struct SIDL_int__array* index,
    int32_t num_stencil_indices, struct SIDL_int__array* stencil_indices,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.SetValues) */
}

/*
 * Method:  SetBoxValues[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_SetBoxValues"

int32_t
impl_Hypre_StructMatrix_SetBoxValues(
  Hypre_StructMatrix self, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t num_stencil_indices,
    struct SIDL_int__array* stencil_indices, struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.SetBoxValues) */
}

/*
 * Method:  SetNumGhost[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_SetNumGhost"

int32_t
impl_Hypre_StructMatrix_SetNumGhost(
  Hypre_StructMatrix self, struct SIDL_int__array* num_ghost)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.SetNumGhost) */
  /* Insert the implementation of the SetNumGhost method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.SetNumGhost) */
}

/*
 * Method:  SetSymmetric[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_SetSymmetric"

int32_t
impl_Hypre_StructMatrix_SetSymmetric(
  Hypre_StructMatrix self, int32_t symmetric)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.SetSymmetric) */
  /* Insert the implementation of the SetSymmetric method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.SetSymmetric) */
}

/*
 * Method:  GetDoubleValue[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_GetDoubleValue"

int32_t
impl_Hypre_StructMatrix_GetDoubleValue(
  Hypre_StructMatrix self, const char* name, double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.GetDoubleValue) */
}

/*
 * Method:  GetIntValue[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_GetIntValue"

int32_t
impl_Hypre_StructMatrix_GetIntValue(
  Hypre_StructMatrix self, const char* name, int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.GetIntValue) */
}

/*
 * Method:  SetDoubleParameter[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_SetDoubleParameter"

int32_t
impl_Hypre_StructMatrix_SetDoubleParameter(
  Hypre_StructMatrix self, const char* name, double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.SetDoubleParameter) */
}

/*
 * Method:  SetIntParameter[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_SetIntParameter"

int32_t
impl_Hypre_StructMatrix_SetIntParameter(
  Hypre_StructMatrix self, const char* name, int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.SetIntParameter) */
}

/*
 * Method:  SetStringParameter[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_SetStringParameter"

int32_t
impl_Hypre_StructMatrix_SetStringParameter(
  Hypre_StructMatrix self, const char* name, const char* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.SetStringParameter) */
}

/*
 * Method:  SetIntArrayParameter[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_SetIntArrayParameter"

int32_t
impl_Hypre_StructMatrix_SetIntArrayParameter(
  Hypre_StructMatrix self, const char* name, struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.SetIntArrayParameter) */
  /* Insert the implementation of the SetIntArrayParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.SetIntArrayParameter) */
}

/*
 * Method:  SetDoubleArrayParameter[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_SetDoubleArrayParameter"

int32_t
impl_Hypre_StructMatrix_SetDoubleArrayParameter(
  Hypre_StructMatrix self, const char* name, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.SetDoubleArrayParameter) */
  /* Insert the implementation of the SetDoubleArrayParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.SetDoubleArrayParameter) */
}

/*
 * Method:  Setup[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_Setup"

int32_t
impl_Hypre_StructMatrix_Setup(
  Hypre_StructMatrix self, Hypre_Vector b, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.Setup) */
  /* Insert the implementation of the Setup method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.Setup) */
}

/*
 * Method:  Apply[]
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructMatrix_Apply"

int32_t
impl_Hypre_StructMatrix_Apply(
  Hypre_StructMatrix self, Hypre_Vector b, Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.Apply) */
  /* Insert the implementation of the Apply method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.Apply) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
  /* DO-NOT-DELETE splicer.begin(Hypre.StructMatrix.SetParameter) */
  /* Insert the implementation of the SetParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructMatrix.SetParameter) */
/* ================== END UNREFERENCED METHOD(S) ================= */
