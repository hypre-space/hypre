/*
 * File:          Hypre_SStructStencil_Impl.c
 * Symbol:        Hypre.SStructStencil-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1011
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.SStructStencil" (version 0.1.7)
 * 
 * The semi-structured grid stencil class.
 * 
 */

#include "Hypre_SStructStencil_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.SStructStencil._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructStencil._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructStencil__ctor"

void
impl_Hypre_SStructStencil__ctor(
  Hypre_SStructStencil self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructStencil._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructStencil._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructStencil__dtor"

void
impl_Hypre_SStructStencil__dtor(
  Hypre_SStructStencil self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructStencil._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructStencil._dtor) */
}

/*
 * Set the number of spatial dimensions and stencil entries.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructStencil_SetNumDimSize"

int32_t
impl_Hypre_SStructStencil_SetNumDimSize(
  Hypre_SStructStencil self, int32_t ndim, int32_t size)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructStencil.SetNumDimSize) */
  /* Insert the implementation of the SetNumDimSize method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructStencil.SetNumDimSize) */
}

/*
 * Set a stencil entry.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructStencil_SetEntry"

int32_t
impl_Hypre_SStructStencil_SetEntry(
  Hypre_SStructStencil self, int32_t entry, struct SIDL_int__array* offset,
    int32_t var)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructStencil.SetEntry) */
  /* Insert the implementation of the SetEntry method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructStencil.SetEntry) */
}
