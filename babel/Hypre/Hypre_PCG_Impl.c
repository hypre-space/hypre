/*
 * File:          Hypre_PCG_Impl.c
 * Symbol:        Hypre.PCG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:19 PST
 * Description:   Server-side implementation for Hypre.PCG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.PCG" (version 0.1.5)
 */

#include "Hypre_PCG_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.PCG._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.PCG._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG__ctor"

void
impl_Hypre_PCG__ctor(
  Hypre_PCG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.PCG._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG__dtor"

void
impl_Hypre_PCG__dtor(
  Hypre_PCG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.PCG._dtor) */
}

/*
 * Method:  Apply
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_Apply"

int32_t
impl_Hypre_PCG_Apply(
  Hypre_PCG self,
  Hypre_Vector x,
  Hypre_Vector* y)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.Apply) */
  /* Insert the implementation of the Apply method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.Apply) */
}

/*
 * Method:  GetPreconditionedResidual
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_GetPreconditionedResidual"

int32_t
impl_Hypre_PCG_GetPreconditionedResidual(
  Hypre_PCG self,
  Hypre_Vector* r)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.GetPreconditionedResidual) */
  /* Insert the implementation of the GetPreconditionedResidual method here... 
    */
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.GetPreconditionedResidual) */
}

/*
 * Method:  GetResidual
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_GetResidual"

int32_t
impl_Hypre_PCG_GetResidual(
  Hypre_PCG self,
  Hypre_Vector* r)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.GetResidual) */
  /* Insert the implementation of the GetResidual method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.GetResidual) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetCommunicator"

int32_t
impl_Hypre_PCG_SetCommunicator(
  Hypre_PCG self,
  void* comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetCommunicator) */
}

/*
 * Method:  SetOperator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetOperator"

int32_t
impl_Hypre_PCG_SetOperator(
  Hypre_PCG self,
  Hypre_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetOperator) */
}

/*
 * Method:  SetParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetParameter"

int32_t
impl_Hypre_PCG_SetParameter(
  Hypre_PCG self,
  const char* name,
  double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetParameter) */
  /* Insert the implementation of the SetParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetParameter) */
}

/*
 * Method:  SetPreconditioner
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_SetPreconditioner"

int32_t
impl_Hypre_PCG_SetPreconditioner(
  Hypre_PCG self,
  Hypre_Solver s)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.SetPreconditioner) */
  /* Insert the implementation of the SetPreconditioner method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.SetPreconditioner) */
}

/*
 * Method:  Setup
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_PCG_Setup"

int32_t
impl_Hypre_PCG_Setup(
  Hypre_PCG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.PCG.Setup) */
  /* Insert the implementation of the Setup method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.PCG.Setup) */
}
