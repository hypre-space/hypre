/*
 * File:          Hypre_ParAMG_Impl.c
 * Symbol:        Hypre.ParAMG-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:18 PST
 * Description:   Server-side implementation for Hypre.ParAMG
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.ParAMG" (version 0.1.5)
 */

#include "Hypre_ParAMG_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.ParAMG._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.ParAMG._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG__ctor"

void
impl_Hypre_ParAMG__ctor(
  Hypre_ParAMG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG__dtor"

void
impl_Hypre_ParAMG__dtor(
  Hypre_ParAMG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG._dtor) */
}

/*
 * Method:  Apply
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_Apply"

int32_t
impl_Hypre_ParAMG_Apply(
  Hypre_ParAMG self,
  Hypre_Vector x,
  Hypre_Vector* y)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.Apply) */
  /* Insert the implementation of the Apply method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.Apply) */
}

/*
 * Method:  GetResidual
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_GetResidual"

int32_t
impl_Hypre_ParAMG_GetResidual(
  Hypre_ParAMG self,
  Hypre_Vector* r)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.GetResidual) */
  /* Insert the implementation of the GetResidual method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.GetResidual) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_SetCommunicator"

int32_t
impl_Hypre_ParAMG_SetCommunicator(
  Hypre_ParAMG self,
  void* comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.SetCommunicator) */
}

/*
 * Method:  SetOperator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_SetOperator"

int32_t
impl_Hypre_ParAMG_SetOperator(
  Hypre_ParAMG self,
  Hypre_Operator A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.SetOperator) */
  /* Insert the implementation of the SetOperator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.SetOperator) */
}

/*
 * Method:  SetParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_SetParameter"

int32_t
impl_Hypre_ParAMG_SetParameter(
  Hypre_ParAMG self,
  const char* name,
  double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.SetParameter) */
  /* Insert the implementation of the SetParameter method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.SetParameter) */
}

/*
 * Method:  Setup
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParAMG_Setup"

int32_t
impl_Hypre_ParAMG_Setup(
  Hypre_ParAMG self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParAMG.Setup) */
  /* Insert the implementation of the Setup method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.ParAMG.Setup) */
}
