/*
 * File:          Hypre_SStructGraph_Impl.c
 * Symbol:        Hypre.SStructGraph-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:12 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructGraph
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1032
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.SStructGraph" (version 0.1.7)
 * 
 * The semi-structured grid graph class.
 * 
 */

#include "Hypre_SStructGraph_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.SStructGraph._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructGraph._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGraph__ctor"

void
impl_Hypre_SStructGraph__ctor(
  Hypre_SStructGraph self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGraph._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGraph._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGraph__dtor"

void
impl_Hypre_SStructGraph__dtor(
  Hypre_SStructGraph self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGraph._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGraph._dtor) */
}

/*
 * Set the grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGraph_SetGrid"

int32_t
impl_Hypre_SStructGraph_SetGrid(
  Hypre_SStructGraph self, Hypre_SStructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGraph.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGraph.SetGrid) */
}

/*
 * Set the stencil for a variable on a structured part of the
 * grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGraph_SetStencil"

int32_t
impl_Hypre_SStructGraph_SetStencil(
  Hypre_SStructGraph self, int32_t part, int32_t var,
    Hypre_SStructStencil stencil)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGraph.SetStencil) */
  /* Insert the implementation of the SetStencil method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGraph.SetStencil) */
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
#define __FUNC__ "impl_Hypre_SStructGraph_AddEntries"

int32_t
impl_Hypre_SStructGraph_AddEntries(
  Hypre_SStructGraph self, int32_t part, struct SIDL_int__array* index,
    int32_t var, int32_t to_part, struct SIDL_int__array* to_index,
    int32_t to_var)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGraph.AddEntries) */
  /* Insert the implementation of the AddEntries method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGraph.AddEntries) */
}
