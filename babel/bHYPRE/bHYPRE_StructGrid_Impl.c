/*
 * File:          bHYPRE_StructGrid_Impl.c
 * Symbol:        bHYPRE.StructGrid-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030320 16:52:19 PST
 * Generated:     20030320 16:52:31 PST
 * Description:   Server-side implementation for bHYPRE.StructGrid
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 1101
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructGrid" (version 1.0.0)
 * 
 * Define a structured grid class.
 * 
 */

#include "bHYPRE_StructGrid_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid__ctor"

void
impl_bHYPRE_StructGrid__ctor(
  bHYPRE_StructGrid self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid__dtor"

void
impl_bHYPRE_StructGrid__dtor(
  bHYPRE_StructGrid self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetCommunicator"

int32_t
impl_bHYPRE_StructGrid_SetCommunicator(
  bHYPRE_StructGrid self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetCommunicator) */
}

/*
 * Method:  SetDimension[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetDimension"

int32_t
impl_bHYPRE_StructGrid_SetDimension(
  bHYPRE_StructGrid self, int32_t dim)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.SetDimension) */
  /* Insert the implementation of the SetDimension method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetDimension) */
}

/*
 * Method:  SetExtents[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetExtents"

int32_t
impl_bHYPRE_StructGrid_SetExtents(
  bHYPRE_StructGrid self, struct SIDL_int__array* ilower,
    struct SIDL_int__array* iupper)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.SetExtents) */
  /* Insert the implementation of the SetExtents method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetExtents) */
}

/*
 * Method:  SetPeriodic[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_SetPeriodic"

int32_t
impl_bHYPRE_StructGrid_SetPeriodic(
  bHYPRE_StructGrid self, struct SIDL_int__array* periodic)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.SetPeriodic) */
  /* Insert the implementation of the SetPeriodic method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.SetPeriodic) */
}

/*
 * Method:  Assemble[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructGrid_Assemble"

int32_t
impl_bHYPRE_StructGrid_Assemble(
  bHYPRE_StructGrid self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructGrid.Assemble) */
  /* Insert the implementation of the Assemble method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructGrid.Assemble) */
}
