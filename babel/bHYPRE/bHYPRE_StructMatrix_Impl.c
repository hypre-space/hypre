/*
 * File:          bHYPRE_StructMatrix_Impl.c
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:31 PST
 * Description:   Server-side implementation for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1124
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructMatrix" (version 1.0.0)
 * 
 * A single class that implements both a build interface and an
 * operator interface. It returns itself for GetConstructedObject.
 * 
 */

#include "bHYPRE_StructMatrix_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix__ctor"

void
impl_bHYPRE_StructMatrix__ctor(
  bHYPRE_StructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix__dtor"

void
impl_bHYPRE_StructMatrix__dtor(
  bHYPRE_StructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetCommunicator"

int32_t
impl_bHYPRE_StructMatrix_SetCommunicator(
  bHYPRE_StructMatrix self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_Initialize"

int32_t
impl_bHYPRE_StructMatrix_Initialize(
  bHYPRE_StructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.Initialize) */
  /* Insert the implementation of the Initialize method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.Initialize) */
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_Assemble"

int32_t
impl_bHYPRE_StructMatrix_Assemble(
  bHYPRE_StructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.Assemble) */
  /* Insert the implementation of the Assemble method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.Assemble) */
}

/*
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

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_GetObject"

int32_t
impl_bHYPRE_StructMatrix_GetObject(
  bHYPRE_StructMatrix self, SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.GetObject) */
  /* Insert the implementation of the GetObject method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.GetObject) */
}

/*
 * Method:  SetGrid[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetGrid"

int32_t
impl_bHYPRE_StructMatrix_SetGrid(
  bHYPRE_StructMatrix self, bHYPRE_StructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetGrid) */
}

/*
 * Method:  SetStencil[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetStencil"

int32_t
impl_bHYPRE_StructMatrix_SetStencil(
  bHYPRE_StructMatrix self, bHYPRE_StructStencil stencil)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetStencil) */
  /* Insert the implementation of the SetStencil method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetStencil) */
}

/*
 * Method:  SetValues[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetValues"

int32_t
impl_bHYPRE_StructMatrix_SetValues(
  bHYPRE_StructMatrix self, struct SIDL_int__array* index,
    int32_t num_stencil_indices, struct SIDL_int__array* stencil_indices,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetValues) */
}

/*
 * Method:  SetBoxValues[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetBoxValues"

int32_t
impl_bHYPRE_StructMatrix_SetBoxValues(
  bHYPRE_StructMatrix self, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t num_stencil_indices,
    struct SIDL_int__array* stencil_indices, struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetBoxValues) */
}

/*
 * Method:  SetNumGhost[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetNumGhost"

int32_t
impl_bHYPRE_StructMatrix_SetNumGhost(
  bHYPRE_StructMatrix self, struct SIDL_int__array* num_ghost)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetNumGhost) */
  /* Insert the implementation of the SetNumGhost method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetNumGhost) */
}

/*
 * Method:  SetSymmetric[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetSymmetric"

int32_t
impl_bHYPRE_StructMatrix_SetSymmetric(
  bHYPRE_StructMatrix self, int32_t symmetric)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetSymmetric) */
  /* Insert the implementation of the SetSymmetric method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetSymmetric) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetIntParameter"

int32_t
impl_bHYPRE_StructMatrix_SetIntParameter(
  bHYPRE_StructMatrix self, const char* name, int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetDoubleParameter"

int32_t
impl_bHYPRE_StructMatrix_SetDoubleParameter(
  bHYPRE_StructMatrix self, const char* name, double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetStringParameter"

int32_t
impl_bHYPRE_StructMatrix_SetStringParameter(
  bHYPRE_StructMatrix self, const char* name, const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetIntArray1Parameter"

int32_t
impl_bHYPRE_StructMatrix_SetIntArray1Parameter(
  bHYPRE_StructMatrix self, const char* name, struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetIntArray2Parameter"

int32_t
impl_bHYPRE_StructMatrix_SetIntArray2Parameter(
  bHYPRE_StructMatrix self, const char* name, struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetDoubleArray1Parameter"

int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray1Parameter(
  bHYPRE_StructMatrix self, const char* name, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetDoubleArray1Parameter) 
    */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetDoubleArray2Parameter"

int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray2Parameter(
  bHYPRE_StructMatrix self, const char* name, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetDoubleArray2Parameter) 
    */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_GetIntValue"

int32_t
impl_bHYPRE_StructMatrix_GetIntValue(
  bHYPRE_StructMatrix self, const char* name, int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_GetDoubleValue"

int32_t
impl_bHYPRE_StructMatrix_GetDoubleValue(
  bHYPRE_StructMatrix self, const char* name, double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_Setup"

int32_t
impl_bHYPRE_StructMatrix_Setup(
  bHYPRE_StructMatrix self, bHYPRE_Vector b, bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.Setup) */
  /* Insert the implementation of the Setup method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_Apply"

int32_t
impl_bHYPRE_StructMatrix_Apply(
  bHYPRE_StructMatrix self, bHYPRE_Vector b, bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.Apply) */
  /* Insert the implementation of the Apply method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.Apply) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
/* ================== END UNREFERENCED METHOD(S) ================= */
