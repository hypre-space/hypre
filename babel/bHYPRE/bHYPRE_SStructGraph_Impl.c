/*
 * File:          bHYPRE_SStructGraph_Impl.c
 * Symbol:        bHYPRE.SStructGraph-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:30 PST
 * Description:   Server-side implementation for bHYPRE.SStructGraph
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1022
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructGraph" (version 1.0.0)
 * 
 * The semi-structured grid graph class.
 * 
 */

#include "bHYPRE_SStructGraph_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph__ctor"

void
impl_bHYPRE_SStructGraph__ctor(
  bHYPRE_SStructGraph self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph__dtor"

void
impl_bHYPRE_SStructGraph__dtor(
  bHYPRE_SStructGraph self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph._dtor) */
}

/*
 * Set the grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_SetGrid"

int32_t
impl_bHYPRE_SStructGraph_SetGrid(
  bHYPRE_SStructGraph self, bHYPRE_SStructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.SetGrid) */
}

/*
 * Set the stencil for a variable on a structured part of the
 * grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_SetStencil"

int32_t
impl_bHYPRE_SStructGraph_SetStencil(
  bHYPRE_SStructGraph self, int32_t part, int32_t var,
    bHYPRE_SStructStencil stencil)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.SetStencil) */
  /* Insert the implementation of the SetStencil method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.SetStencil) */
}

/*
 * Add a non-stencil graph entry at a particular index.  This
 * graph entry is appended to the existing graph entries, and is
 * referenced as such.
 * 
 * NOTE: Users are required to set graph entries on all
 * processes that own the associated variables.  This means that
 * some data will be multiply defined.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGraph_AddEntries"

int32_t
impl_bHYPRE_SStructGraph_AddEntries(
  bHYPRE_SStructGraph self, int32_t part, struct SIDL_int__array* index,
    int32_t var, int32_t to_part, struct SIDL_int__array* to_index,
    int32_t to_var)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGraph.AddEntries) */
  /* Insert the implementation of the AddEntries method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGraph.AddEntries) */
}
