/*
 * File:          Hypre_GMRES_Impl.c
 * Symbol:        Hypre.GMRES-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:18 PST
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
 * Method:  SetParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_GMRES_SetParameter"

int32_t
impl_Hypre_GMRES_SetParameter(
  Hypre_GMRES self,
  const char* name,
  double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.GMRES.SetParameter) */
  /* Insert the implementation of the SetParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.GMRES.SetParameter) */
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
