/*
 * File:          Hypre_Pilut_Impl.c
 * Symbol:        Hypre.Pilut-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.Pilut
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1242
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.Pilut" (version 0.1.7)
 * 
 * Objects of this type can be cast to Solver objects using the
 * {\tt \_\_cast} methods.
 * 
 * RDF: Documentation goes here.
 * 
 */

#include "Hypre_Pilut_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.Pilut._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.Pilut._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut__ctor"

void
impl_Hypre_Pilut__ctor(
  Hypre_Pilut self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut__dtor"

void
impl_Hypre_Pilut__dtor(
  Hypre_Pilut self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_SetCommunicator"

int32_t
impl_Hypre_Pilut_SetCommunicator(
  Hypre_Pilut self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_SetIntParameter"

int32_t
impl_Hypre_Pilut_SetIntParameter(
  Hypre_Pilut self, const char* name, int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_SetDoubleParameter"

int32_t
impl_Hypre_Pilut_SetDoubleParameter(
  Hypre_Pilut self, const char* name, double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_SetStringParameter"

int32_t
impl_Hypre_Pilut_SetStringParameter(
  Hypre_Pilut self, const char* name, const char* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.SetStringParameter) */
}

/*
 * Set the int array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_SetIntArrayParameter"

int32_t
impl_Hypre_Pilut_SetIntArrayParameter(
  Hypre_Pilut self, const char* name, struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.SetIntArrayParameter) */
  /* Insert the implementation of the SetIntArrayParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.SetIntArrayParameter) */
}

/*
 * Set the double array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_SetDoubleArrayParameter"

int32_t
impl_Hypre_Pilut_SetDoubleArrayParameter(
  Hypre_Pilut self, const char* name, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.SetDoubleArrayParameter) */
  /* Insert the implementation of the SetDoubleArrayParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.SetDoubleArrayParameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_GetIntValue"

int32_t
impl_Hypre_Pilut_GetIntValue(
  Hypre_Pilut self, const char* name, int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_GetDoubleValue"

int32_t
impl_Hypre_Pilut_GetDoubleValue(
  Hypre_Pilut self, const char* name, double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_Setup"

int32_t
impl_Hypre_Pilut_Setup(
  Hypre_Pilut self, Hypre_Vector b, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.Setup) */
  /* Insert the implementation of the Setup method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_Apply"

int32_t
impl_Hypre_Pilut_Apply(
  Hypre_Pilut self, Hypre_Vector b, Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.Apply) */
  /* Insert the implementation of the Apply method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.Apply) */
}

/*
 * Set the operator for the linear system being solved.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_SetOperator"

int32_t
impl_Hypre_Pilut_SetOperator(
  Hypre_Pilut self, Hypre_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.SetOperator) */
}

/*
 * (Optional) Set the convergence tolerance.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_SetTolerance"

int32_t
impl_Hypre_Pilut_SetTolerance(
  Hypre_Pilut self, double tolerance)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.SetTolerance) */
  /* Insert the implementation of the SetTolerance method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.SetTolerance) */
}

/*
 * (Optional) Set maximum number of iterations.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_SetMaxIterations"

int32_t
impl_Hypre_Pilut_SetMaxIterations(
  Hypre_Pilut self, int32_t max_iterations)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.SetMaxIterations) */
  /* Insert the implementation of the SetMaxIterations method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.SetMaxIterations) */
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
#define __FUNC__ "impl_Hypre_Pilut_SetLogging"

int32_t
impl_Hypre_Pilut_SetLogging(
  Hypre_Pilut self, int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.SetLogging) */
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
#define __FUNC__ "impl_Hypre_Pilut_SetPrintLevel"

int32_t
impl_Hypre_Pilut_SetPrintLevel(
  Hypre_Pilut self, int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.SetPrintLevel) */
}

/*
 * (Optional) Return the number of iterations taken.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_GetNumIterations"

int32_t
impl_Hypre_Pilut_GetNumIterations(
  Hypre_Pilut self, int32_t* num_iterations)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.GetNumIterations) */
  /* Insert the implementation of the GetNumIterations method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.GetNumIterations) */
}

/*
 * (Optional) Return the norm of the relative residual.
 * 
 * RDF: New
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_Pilut_GetRelResidualNorm"

int32_t
impl_Hypre_Pilut_GetRelResidualNorm(
  Hypre_Pilut self, double* norm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.Pilut.GetRelResidualNorm) */
  /* Insert the implementation of the GetRelResidualNorm method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.Pilut.GetRelResidualNorm) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
/* ================== END UNREFERENCED METHOD(S) ================= */
