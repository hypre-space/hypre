/*
 * File:          bHYPRE_SStructGrid_Impl.c
 * Symbol:        bHYPRE.SStructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:39 PST
 * Generated:     20050317 11:17:43 PST
 * Description:   Server-side implementation for bHYPRE.SStructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 909
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructGrid" (version 1.0.0)
 * 
 * The semi-structured grid class.
 * 
 */

#include "bHYPRE_SStructGrid_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid__ctor"

void
impl_bHYPRE_SStructGrid__ctor(
  /*in*/ bHYPRE_SStructGrid self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid__dtor"

void
impl_bHYPRE_SStructGrid__dtor(
  /*in*/ bHYPRE_SStructGrid self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid._dtor) */
}

/*
 * Set the number of dimensions {\tt ndim} and the number of
 * structured parts {\tt nparts}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_SetNumDimParts"

int32_t
impl_bHYPRE_SStructGrid_SetNumDimParts(
  /*in*/ bHYPRE_SStructGrid self, /*in*/ int32_t ndim, /*in*/ int32_t nparts)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.SetNumDimParts) */
  /* Insert the implementation of the SetNumDimParts method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.SetNumDimParts) */
}

/*
 * Set the extents for a box on a structured part of the grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_SetExtents"

int32_t
impl_bHYPRE_SStructGrid_SetExtents(
  /*in*/ bHYPRE_SStructGrid self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.SetExtents) */
  /* Insert the implementation of the SetExtents method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.SetExtents) */
}

/*
 * Describe the variables that live on a structured part of the
 * grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_SetVariable"

int32_t
impl_bHYPRE_SStructGrid_SetVariable(
  /*in*/ bHYPRE_SStructGrid self, /*in*/ int32_t part, /*in*/ int32_t var,
    /*in*/ enum bHYPRE_SStructVariable__enum vartype)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.SetVariable) */
  /* Insert the implementation of the SetVariable method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.SetVariable) */
}

/*
 * Describe additional variables that live at a particular
 * index.  These variables are appended to the array of
 * variables set in {\tt SetVariables}, and are referenced as
 * such.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_AddVariable"

int32_t
impl_bHYPRE_SStructGrid_AddVariable(
  /*in*/ bHYPRE_SStructGrid self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*in*/ enum bHYPRE_SStructVariable__enum vartype)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.AddVariable) */
  /* Insert the implementation of the AddVariable method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.AddVariable) */
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
#define __FUNC__ "impl_bHYPRE_SStructGrid_SetNeighborBox"

int32_t
impl_bHYPRE_SStructGrid_SetNeighborBox(
  /*in*/ bHYPRE_SStructGrid self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t nbor_part,
    /*in*/ struct sidl_int__array* nbor_ilower,
    /*in*/ struct sidl_int__array* nbor_iupper,
    /*in*/ struct sidl_int__array* index_map)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.SetNeighborBox) */
  /* Insert the implementation of the SetNeighborBox method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.SetNeighborBox) */
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
#define __FUNC__ "impl_bHYPRE_SStructGrid_AddUnstructuredPart"

int32_t
impl_bHYPRE_SStructGrid_AddUnstructuredPart(
  /*in*/ bHYPRE_SStructGrid self, /*in*/ int32_t ilower, /*in*/ int32_t iupper)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.AddUnstructuredPart) */
  /* Insert the implementation of the AddUnstructuredPart method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.AddUnstructuredPart) */
}

/*
 * (Optional) Set periodic for a particular part.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_SetPeriodic"

int32_t
impl_bHYPRE_SStructGrid_SetPeriodic(
  /*in*/ bHYPRE_SStructGrid self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* periodic)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.SetPeriodic) */
  /* Insert the implementation of the SetPeriodic method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.SetPeriodic) */
}

/*
 * Setting ghost in the sgrids.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructGrid_SetNumGhost"

int32_t
impl_bHYPRE_SStructGrid_SetNumGhost(
  /*in*/ bHYPRE_SStructGrid self, /*in*/ struct sidl_int__array* num_ghost)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructGrid.SetNumGhost) */
  /* Insert the implementation of the SetNumGhost method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructGrid.SetNumGhost) */
}
