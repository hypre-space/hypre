/*
 * File:          bHYPRE_SStructStencil_Impl.c
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:39 PST
 * Generated:     20050317 11:17:43 PST
 * Description:   Server-side implementation for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 1006
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
  /*in*/ bHYPRE_SStructStencil self)
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
  /*in*/ bHYPRE_SStructStencil self)
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
  /*in*/ bHYPRE_SStructStencil self, /*in*/ int32_t ndim, /*in*/ int32_t size)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil.SetNumDimSize) */
  /* Insert the implementation of the SetNumDimSize method here... */
   return 1;
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
  /*in*/ bHYPRE_SStructStencil self, /*in*/ int32_t entry,
    /*in*/ struct sidl_int__array* offset, /*in*/ int32_t var)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil.SetEntry) */
  /* Insert the implementation of the SetEntry method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil.SetEntry) */
}
