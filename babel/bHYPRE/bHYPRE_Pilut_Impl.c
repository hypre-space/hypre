/*
 * File:          bHYPRE_Pilut_Impl.c
 * Symbol:        bHYPRE.Pilut-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050208 15:29:05 PST
 * Generated:     20050208 15:29:08 PST
 * Description:   Server-side implementation for bHYPRE.Pilut
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 1227
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.Pilut" (version 1.0.0)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 */

#include "bHYPRE_Pilut_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut__ctor"

void
impl_bHYPRE_Pilut__ctor(
  /*in*/ bHYPRE_Pilut self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut__dtor"

void
impl_bHYPRE_Pilut__dtor(
  /*in*/ bHYPRE_Pilut self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetCommunicator"

int32_t
impl_bHYPRE_Pilut_SetCommunicator(
  /*in*/ bHYPRE_Pilut self, /*in*/ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetIntParameter"

int32_t
impl_bHYPRE_Pilut_SetIntParameter(
  /*in*/ bHYPRE_Pilut self, /*in*/ const char* name, /*in*/ int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetDoubleParameter"

int32_t
impl_bHYPRE_Pilut_SetDoubleParameter(
  /*in*/ bHYPRE_Pilut self, /*in*/ const char* name, /*in*/ double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetStringParameter"

int32_t
impl_bHYPRE_Pilut_SetStringParameter(
  /*in*/ bHYPRE_Pilut self, /*in*/ const char* name, /*in*/ const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetIntArray1Parameter"

int32_t
impl_bHYPRE_Pilut_SetIntArray1Parameter(
  /*in*/ bHYPRE_Pilut self, /*in*/ const char* name,
    /*in*/ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetIntArray2Parameter"

int32_t
impl_bHYPRE_Pilut_SetIntArray2Parameter(
  /*in*/ bHYPRE_Pilut self, /*in*/ const char* name,
    /*in*/ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetDoubleArray1Parameter"

int32_t
impl_bHYPRE_Pilut_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_Pilut self, /*in*/ const char* name,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetDoubleArray2Parameter"

int32_t
impl_bHYPRE_Pilut_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_Pilut self, /*in*/ const char* name,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_GetIntValue"

int32_t
impl_bHYPRE_Pilut_GetIntValue(
  /*in*/ bHYPRE_Pilut self, /*in*/ const char* name, /*out*/ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_GetDoubleValue"

int32_t
impl_bHYPRE_Pilut_GetDoubleValue(
  /*in*/ bHYPRE_Pilut self, /*in*/ const char* name, /*out*/ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_Setup"

int32_t
impl_bHYPRE_Pilut_Setup(
  /*in*/ bHYPRE_Pilut self, /*in*/ bHYPRE_Vector b, /*in*/ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.Setup) */
  /* Insert the implementation of the Setup method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_Apply"

int32_t
impl_bHYPRE_Pilut_Apply(
  /*in*/ bHYPRE_Pilut self, /*in*/ bHYPRE_Vector b, /*inout*/ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.Apply) */
  /* Insert the implementation of the Apply method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.Apply) */
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetOperator"

int32_t
impl_bHYPRE_Pilut_SetOperator(
  /*in*/ bHYPRE_Pilut self, /*in*/ bHYPRE_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetTolerance"

int32_t
impl_bHYPRE_Pilut_SetTolerance(
  /*in*/ bHYPRE_Pilut self, /*in*/ double tolerance)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetTolerance) */
}

/*
 * (Optional) Set maximum number of iterations.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetMaxIterations"

int32_t
impl_bHYPRE_Pilut_SetMaxIterations(
  /*in*/ bHYPRE_Pilut self, /*in*/ int32_t max_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetMaxIterations) */
}

/*
 * (Optional) Set the {\it logging level}, specifying the degree
 * of additional informational data to be accumulated.  Does
 * nothing by default (level = 0).  Other levels (if any) are
 * implementation-specific.  Must be called before {\tt Setup}
 * and {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetLogging"

int32_t
impl_bHYPRE_Pilut_SetLogging(
  /*in*/ bHYPRE_Pilut self, /*in*/ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetLogging) */
}

/*
 * (Optional) Set the {\it print level}, specifying the degree
 * of informational data to be printed either to the screen or
 * to a file.  Does nothing by default (level=0).  Other levels
 * (if any) are implementation-specific.  Must be called before
 * {\tt Setup} and {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_SetPrintLevel"

int32_t
impl_bHYPRE_Pilut_SetPrintLevel(
  /*in*/ bHYPRE_Pilut self, /*in*/ int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.SetPrintLevel) */
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_GetNumIterations"

int32_t
impl_bHYPRE_Pilut_GetNumIterations(
  /*in*/ bHYPRE_Pilut self, /*out*/ int32_t* num_iterations)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.GetNumIterations) */
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_Pilut_GetRelResidualNorm"

int32_t
impl_bHYPRE_Pilut_GetRelResidualNorm(
  /*in*/ bHYPRE_Pilut self, /*out*/ double* norm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.Pilut.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.Pilut.GetRelResidualNorm) */
}
