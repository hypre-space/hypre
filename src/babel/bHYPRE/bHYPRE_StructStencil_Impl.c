/*
 * File:          bHYPRE_StructStencil_Impl.c
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.2
 * SIDL Created:  20030401 14:47:20 PST
 * Generated:     20030401 14:47:32 PST
 * Description:   Server-side implementation for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.2
 * source-line   = 1088
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructStencil" (version 1.0.0)
 * 
 * Define a structured stencil for a structured problem
 * description.  More than one implementation is not envisioned,
 * thus the decision has been made to make this a class rather than
 * an interface.
 * 
 */

#include "bHYPRE_StructStencil_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil__ctor"

void
impl_bHYPRE_StructStencil__ctor(
  bHYPRE_StructStencil self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil__dtor"

void
impl_bHYPRE_StructStencil__dtor(
  bHYPRE_StructStencil self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._dtor) */
}

/*
 * Method:  SetDimension[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_SetDimension"

int32_t
impl_bHYPRE_StructStencil_SetDimension(
  bHYPRE_StructStencil self, int32_t dim)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil.SetDimension) */
  /* Insert the implementation of the SetDimension method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil.SetDimension) */
}

/*
 * Method:  SetSize[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_SetSize"

int32_t
impl_bHYPRE_StructStencil_SetSize(
  bHYPRE_StructStencil self, int32_t size)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil.SetSize) */
  /* Insert the implementation of the SetSize method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil.SetSize) */
}

/*
 * Method:  SetElement[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_SetElement"

int32_t
impl_bHYPRE_StructStencil_SetElement(
  bHYPRE_StructStencil self, int32_t index, struct SIDL_int__array* offset)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil.SetElement) */
  /* Insert the implementation of the SetElement method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil.SetElement) */
}
