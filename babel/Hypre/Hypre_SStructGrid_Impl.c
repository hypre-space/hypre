/*
 * File:          Hypre_SStructGrid_Impl.c
 * Symbol:        Hypre.SStructGrid-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.SStructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 914
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.SStructGrid" (version 0.1.7)
 * 
 * The semi-structured grid class.
 * 
 */

#include "Hypre_SStructGrid_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.SStructGrid._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.SStructGrid._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGrid__ctor"

void
impl_Hypre_SStructGrid__ctor(
  Hypre_SStructGrid self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGrid._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGrid._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGrid__dtor"

void
impl_Hypre_SStructGrid__dtor(
  Hypre_SStructGrid self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGrid._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGrid._dtor) */
}

/*
 * Set the number of dimensions {\tt ndim} and the number of
 * structured parts {\tt nparts}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGrid_SetNumDimParts"

int32_t
impl_Hypre_SStructGrid_SetNumDimParts(
  Hypre_SStructGrid self, int32_t ndim, int32_t nparts)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGrid.SetNumDimParts) */
  /* Insert the implementation of the SetNumDimParts method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGrid.SetNumDimParts) */
}

/*
 * Set the extents for a box on a structured part of the grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGrid_SetExtents"

int32_t
impl_Hypre_SStructGrid_SetExtents(
  Hypre_SStructGrid self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGrid.SetExtents) */
  /* Insert the implementation of the SetExtents method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGrid.SetExtents) */
}

/*
 * Describe the variables that live on a structured part of the
 * grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGrid_SetVariable"

int32_t
impl_Hypre_SStructGrid_SetVariable(
  Hypre_SStructGrid self, int32_t part, int32_t var,
    enum Hypre_SStructVariable__enum vartype)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGrid.SetVariable) */
  /* Insert the implementation of the SetVariable method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGrid.SetVariable) */
}

/*
 * Describe additional variables that live at a particular
 * index.  These variables are appended to the array of
 * variables set in {\tt SetVariables}, and are referenced as
 * such.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGrid_AddVariable"

int32_t
impl_Hypre_SStructGrid_AddVariable(
  Hypre_SStructGrid self, int32_t part, struct SIDL_int__array* index,
    int32_t var, enum Hypre_SStructVariable__enum vartype)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGrid.AddVariable) */
  /* Insert the implementation of the AddVariable method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGrid.AddVariable) */
}

/*
 * Describe how regions just outside of a part relate to other
 * parts.  This is done a box at a time.
 * 
 * The indexes {\tt ilower} and {\tt iupper} map directly to the
 * indexes {\tt nbor\_ilower} and {\tt nbor\_iupper}.  Although,
 * it is required that indexes increase from {\tt ilower} to
 * {\tt iupper}, indexes may increase and/or decrease from {\tt
 * nbor\_ilower} to {\tt nbor\_iupper}.
 * 
 * The {\tt index\_map} describes the mapping of indexes 0, 1,
 * and 2 on part {\tt part} to the corresponding indexes on part
 * {\tt nbor\_part}.  For example, triple (1, 2, 0) means that
 * indexes 0, 1, and 2 on part {\tt part} map to indexes 1, 2,
 * and 0 on part {\tt nbor\_part}, respectively.
 * 
 * NOTE: All parts related to each other via this routine must
 * have an identical list of variables and variable types.  For
 * example, if part 0 has only two variables on it, a cell
 * centered variable and a node centered variable, and we
 * declare part 1 to be a neighbor of part 0, then part 1 must
 * also have only two variables on it, and they must be of type
 * cell and node.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGrid_SetNeighborBox"

int32_t
impl_Hypre_SStructGrid_SetNeighborBox(
  Hypre_SStructGrid self, int32_t part, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper, int32_t nbor_part,
    struct SIDL_int__array* nbor_ilower, struct SIDL_int__array* nbor_iupper,
    struct SIDL_int__array* index_map)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGrid.SetNeighborBox) */
  /* Insert the implementation of the SetNeighborBox method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGrid.SetNeighborBox) */
}

/*
 * Add an unstructured part to the grid.  The variables in the
 * unstructured part of the grid are referenced by a global rank
 * between 0 and the total number of unstructured variables
 * minus one.  Each process owns some unique consecutive range
 * of variables, defined by {\tt ilower} and {\tt iupper}.
 * 
 * NOTE: This is just a placeholder.  This part of the interface
 * is not finished.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGrid_AddUnstructuredPart"

int32_t
impl_Hypre_SStructGrid_AddUnstructuredPart(
  Hypre_SStructGrid self, int32_t ilower, int32_t iupper)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGrid.AddUnstructuredPart) */
  /* Insert the implementation of the AddUnstructuredPart method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGrid.AddUnstructuredPart) */
}

/*
 * (Optional) Set periodic for a particular part.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGrid_SetPeriodic"

int32_t
impl_Hypre_SStructGrid_SetPeriodic(
  Hypre_SStructGrid self, int32_t part, struct SIDL_int__array* periodic)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGrid.SetPeriodic) */
  /* Insert the implementation of the SetPeriodic method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGrid.SetPeriodic) */
}

/*
 * Setting ghost in the sgrids.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_SStructGrid_SetNumGhost"

int32_t
impl_Hypre_SStructGrid_SetNumGhost(
  Hypre_SStructGrid self, struct SIDL_int__array* num_ghost)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.SStructGrid.SetNumGhost) */
  /* Insert the implementation of the SetNumGhost method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.SStructGrid.SetNumGhost) */
}
