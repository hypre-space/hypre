/*
 * File:          Hypre_GMRES_Impl.c
 * Symbol:        Hypre.GMRES-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020711 16:38:24 PDT
 * Generated:     20020711 16:38:32 PDT
 * Description:   Server-side implementation for Hypre.GMRES
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.GMRES" (version 0.1.5)
 */

#include "Hypre_GMRES_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.GMRES._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.GMRES._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES__ctor"

void
impl_Hypre_GMRES__ctor(
  Hypre_GMRES self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES__dtor"

void
impl_Hypre_GMRES__dtor(
  Hypre_GMRES self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES._dtor) */
}

/*
 * Method:  Apply
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_Apply"

int32_t
impl_Hypre_GMRES_Apply(
  Hypre_GMRES self,
  Hypre_Vector x,
  Hypre_Vector* y)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.Apply) */
  /* Insert the implementation of the Apply method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.Apply) */
}

/*
 * Method:  GetDoubleValue
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_GetDoubleValue"

int32_t
impl_Hypre_GMRES_GetDoubleValue(
  Hypre_GMRES self,
  const char* name,
  double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.GetDoubleValue) */
}

/*
 * Method:  GetIntValue
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_GetIntValue"

int32_t
impl_Hypre_GMRES_GetIntValue(
  Hypre_GMRES self,
  const char* name,
  int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.GetIntValue) */
}

/*
 * Method:  GetPreconditionedResidual
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_GetPreconditionedResidual"

int32_t
impl_Hypre_GMRES_GetPreconditionedResidual(
  Hypre_GMRES self,
  Hypre_Vector* r)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.GetPreconditionedResidual) */
  /* Insert the implementation of the GetPreconditionedResidual method here... 
    */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.GetPreconditionedResidual) */
}

/*
 * Method:  GetResidual
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_GetResidual"

int32_t
impl_Hypre_GMRES_GetResidual(
  Hypre_GMRES self,
  Hypre_Vector* r)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.GetResidual) */
  /* Insert the implementation of the GetResidual method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.GetResidual) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetCommunicator"

int32_t
impl_Hypre_GMRES_SetCommunicator(
  Hypre_GMRES self,
  void* comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetCommunicator) */
}

/*
 * Method:  SetDoubleArrayParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetDoubleArrayParameter"

int32_t
impl_Hypre_GMRES_SetDoubleArrayParameter(
  Hypre_GMRES self,
  const char* name,
  struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetDoubleArrayParameter) */
  /* Insert the implementation of the SetDoubleArrayParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetDoubleArrayParameter) */
}

/*
 * Method:  SetDoubleParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetDoubleParameter"

int32_t
impl_Hypre_GMRES_SetDoubleParameter(
  Hypre_GMRES self,
  const char* name,
  double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetDoubleParameter) */
}

/*
 * Method:  SetIntArrayParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetIntArrayParameter"

int32_t
impl_Hypre_GMRES_SetIntArrayParameter(
  Hypre_GMRES self,
  const char* name,
  struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetIntArrayParameter) */
  /* Insert the implementation of the SetIntArrayParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetIntArrayParameter) */
}

/*
 * Method:  SetIntParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetIntParameter"

int32_t
impl_Hypre_GMRES_SetIntParameter(
  Hypre_GMRES self,
  const char* name,
  int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetIntParameter) */
}

/*
 * Method:  SetLogging
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetLogging"

int32_t
impl_Hypre_GMRES_SetLogging(
  Hypre_GMRES self,
  int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetLogging) */
  /* Insert the implementation of the SetLogging method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetLogging) */
}

/*
 * Method:  SetOperator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetOperator"

int32_t
impl_Hypre_GMRES_SetOperator(
  Hypre_GMRES self,
  Hypre_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetOperator) */
}

/*
 * Method:  SetPreconditioner
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetPreconditioner"

int32_t
impl_Hypre_GMRES_SetPreconditioner(
  Hypre_GMRES self,
  Hypre_Solver s)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetPreconditioner) */
  /* Insert the implementation of the SetPreconditioner method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetPreconditioner) */
}

/*
 * Method:  SetPrintLevel
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetPrintLevel"

int32_t
impl_Hypre_GMRES_SetPrintLevel(
  Hypre_GMRES self,
  int32_t level)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetPrintLevel) */
  /* Insert the implementation of the SetPrintLevel method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetPrintLevel) */
}

/*
 * Method:  SetStringParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetStringParameter"

int32_t
impl_Hypre_GMRES_SetStringParameter(
  Hypre_GMRES self,
  const char* name,
  const char* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetStringParameter) */
}

/*
 * Method:  Setup
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_Setup"

int32_t
impl_Hypre_GMRES_Setup(
  Hypre_GMRES self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.Setup) */
  /* Insert the implementation of the Setup method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.Setup) */
}
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetParameter) */
  /* Insert the implementation of the SetParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetParameter) */
