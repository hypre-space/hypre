/*
 * File:          Hypre_StructGrid_Impl.c
 * Symbol:        Hypre.StructGrid-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.3
 * SIDL Created:  20020522 13:59:35 PDT
 * Generated:     20020522 13:59:44 PDT
 * Description:   Server-side implementation for Hypre.StructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.StructGrid" (version 0.1.5)
 * 
 * Define a structured grid class.
 */

#include "Hypre_StructGrid_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.StructGrid._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.StructGrid._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructGrid__ctor"

void
impl_Hypre_StructGrid__ctor(
  Hypre_StructGrid self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructGrid._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructGrid._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructGrid__dtor"

void
impl_Hypre_StructGrid__dtor(
  Hypre_StructGrid self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructGrid._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructGrid._dtor) */
}

/*
 * Method:  Assemble
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructGrid_Assemble"

int32_t
impl_Hypre_StructGrid_Assemble(
  Hypre_StructGrid self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructGrid.Assemble) */
  /* Insert the implementation of the Assemble method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructGrid.Assemble) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructGrid_SetCommunicator"

int32_t
impl_Hypre_StructGrid_SetCommunicator(
  Hypre_StructGrid self,
  void* MPI_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructGrid.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructGrid.SetCommunicator) */
}

/*
 * Method:  SetDimension
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructGrid_SetDimension"

int32_t
impl_Hypre_StructGrid_SetDimension(
  Hypre_StructGrid self,
  int32_t dim)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructGrid.SetDimension) */
  /* Insert the implementation of the SetDimension method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructGrid.SetDimension) */
}

/*
 * Method:  SetExtents
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructGrid_SetExtents"

int32_t
impl_Hypre_StructGrid_SetExtents(
  Hypre_StructGrid self,
  struct SIDL_int__array* ilower,
  struct SIDL_int__array* iupper)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructGrid.SetExtents) */
  /* Insert the implementation of the SetExtents method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructGrid.SetExtents) */
}

/*
 * Method:  SetPeriodic
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_StructGrid_SetPeriodic"

int32_t
impl_Hypre_StructGrid_SetPeriodic(
  Hypre_StructGrid self,
  struct SIDL_int__array* periodic)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.StructGrid.SetPeriodic) */
  /* Insert the implementation of the SetPeriodic method here... */
  /* DO-NOT-DELETE splicer.end(Hypre.StructGrid.SetPeriodic) */
}
