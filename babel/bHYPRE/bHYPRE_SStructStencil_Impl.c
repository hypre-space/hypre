/*
 * File:          bHYPRE_SStructStencil_Impl.c
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:31 PST
 * Description:   Server-side implementation for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1001
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructStencil" (version 1.0.0)
 * 
 * The semi-structured grid stencil class.
 * 
 */

#include "bHYPRE_SStructStencil_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil__ctor"

void
impl_bHYPRE_SStructStencil__ctor(
  bHYPRE_SStructStencil self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil__dtor"

void
impl_bHYPRE_SStructStencil__dtor(
  bHYPRE_SStructStencil self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._dtor) */
}

/*
 * Set the number of spatial dimensions and stencil entries.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil_SetNumDimSize"

int32_t
impl_bHYPRE_SStructStencil_SetNumDimSize(
  bHYPRE_SStructStencil self, int32_t ndim, int32_t size)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil.SetNumDimSize) */
  /* Insert the implementation of the SetNumDimSize method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil.SetNumDimSize) */
}

/*
 * Set a stencil entry.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil_SetEntry"

int32_t
impl_bHYPRE_SStructStencil_SetEntry(
  bHYPRE_SStructStencil self, int32_t entry, struct SIDL_int__array* offset,
    int32_t var)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil.SetEntry) */
  /* Insert the implementation of the SetEntry method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil.SetEntry) */
}
