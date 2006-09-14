/*
 * File:          bHYPRE_SStructParCSRVector_Impl.c
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructParCSRVector" (version 1.0.0)
 * 
 * The SStructParCSR vector class.
 * 
 * Objects of this type can be cast to SStructVectorView or Vector
 * objects using the {\tt \_\_cast} methods.
 */

#include "bHYPRE_SStructParCSRVector_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._includes) */
/* Put additional includes or other arbitrary code here... */

/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include <assert.h>
#include "hypre_babel_exception_handler.h"
/*#include "mpi.h"*/
#include "sstruct_mv.h"
#include "bHYPRE_SStructGrid_Impl.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructParCSRVector__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._load) */
  /* Insert-Code-Here {bHYPRE.SStructParCSRVector._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructParCSRVector__ctor(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._ctor) */
  /* Insert the implementation of the constructor method here... */

  /* How to make a vector: first call _Create (not __create),
     then Initialize, then SetValues (or SetBoxValues, etc.), then Assemble.
  */

   struct bHYPRE_SStructParCSRVector__data * data;
   data = hypre_CTAlloc( struct bHYPRE_SStructParCSRVector__data, 1 );
   data -> vec = NULL;
   data -> comm = MPI_COMM_NULL;
   bHYPRE_SStructParCSRVector__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructParCSRVector__ctor2(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._ctor2) */
    /* Insert-Code-Here {bHYPRE.SStructParCSRVector._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructParCSRVector__dtor(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector vec;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   vec = data -> vec;
   ierr += HYPRE_SStructVectorDestroy( vec );
   hypre_assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._dtor) */
  }
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_SStructParCSRVector
impl_bHYPRE_SStructParCSRVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Create) */
  /* Insert-Code-Here {bHYPRE.SStructParCSRVector.Create} (Create method) */

   int ierr = 0;
   bHYPRE_SStructParCSRVector vec;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hvec;
   struct bHYPRE_SStructGrid__data * gdata;
   HYPRE_SStructGrid Hgrid;
   MPI_Comm comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   vec = bHYPRE_SStructParCSRVector__create(_ex); SIDL_CHECK(*_ex);
   data = bHYPRE_SStructParCSRVector__get_data( vec );

   gdata = bHYPRE_SStructGrid__get_data( grid );
   Hgrid = gdata->grid;

   ierr += HYPRE_SStructVectorCreate( comm, Hgrid, &Hvec );
   data->vec = Hvec;
   data -> comm = comm;

   return( vec );

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Create) */
  }
}

/*
 * Set the vector grid.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetGrid"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_SetGrid(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_SStructGrid grid,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
   /* N.B. This is the only grid-setting function defined in the interface.
    So this is the only place to call HYPRE_SStructVectorCreate, which requires a grid.
    Note that SetGrid cannot be called twice on the same vector.  The grid cannot be changed.

    SetCommunicator should have been called before the time SetGrid is called.
    Initialize, value-setting functions, and Assemble should be called afterwards.
   */

   /* DEPRECATED   use _Create */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   HYPRE_SStructGrid Hgrid;
   MPI_Comm comm;
   struct bHYPRE_SStructGrid__data * gdata;

   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data->vec;
   hypre_assert( Hy==NULL ); /* shouldn't have already been created */
   comm = data->comm;
   gdata = bHYPRE_SStructGrid__get_data( grid );
   Hgrid = gdata->grid;

   ierr += HYPRE_SStructVectorCreate( comm, Hgrid, &Hy );
   data->vec = Hy;

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetGrid) */
  }
}

/*
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_SetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetValues) */
  /* Insert the implementation of the SetValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_SStructVectorSetValues
      ( Hy, part, index, var,
        &value );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetValues) */
  }
}

/*
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetBoxValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_SetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_SStructVectorSetBoxValues
      ( Hy, part, ilower, iupper,
        var, values );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetBoxValues) */
  }
}

/*
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_AddToValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_AddToValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_SStructVectorAddToValues
      ( Hy, part, index, var,
        &value );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.AddToValues) */
  }
}

/*
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_AddToBoxValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_AddToBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.AddToBoxValues) */
  /* Insert the implementation of the AddToBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_SStructVectorAddToBoxValues
      ( Hy, part, ilower, iupper,
        var, values );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.AddToBoxValues) */
  }
}

/*
 * Gather vector data before calling {\tt GetValues}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Gather"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_Gather(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Gather) */
  /* Insert the implementation of the Gather method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr = HYPRE_SStructVectorGather( Hy );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Gather) */
  }
}

/*
 * Get vector coefficients index by index.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_GetValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_GetValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.GetValues) */
  /* Insert the implementation of the GetValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_SStructVectorGetValues
      ( Hy, part, index, var,
        value );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.GetValues) */
  }
}

/*
 * Get vector coefficients a box at a time.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_GetBoxValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_GetBoxValues(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* inout rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.GetBoxValues) */
  /* Insert the implementation of the GetBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_SStructVectorGetBoxValues
      ( Hy, part, ilower, iupper,
        var, values );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.GetBoxValues) */
  }
}

/*
 * Set the vector to be complex.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetComplex"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_SetComplex(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetComplex) */
  /* Insert the implementation of the SetComplex method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetComplex) */
  }
}

/*
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Print"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_Print(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ const char* filename,
  /* in */ int32_t all,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Print) */
  /* Insert the implementation of the Print method here... */

   int ierr=0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hv;

   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hv = data -> vec;

   /* If this doesn't work well, try geting a ParCSR vector with GetObject,
      then ParVectorPrint...*/
   ierr += HYPRE_SStructVectorPrint( filename, Hv, all );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Print) */
  }
}

/*
 * A semi-structured matrix or vector contains a Struct or IJ matrix
 * or vector.  GetObject returns it.
 * The returned type is a sidl.BaseInterface.
 * A cast must be used on the returned object to convert it into a known type.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_GetObject"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_GetObject(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface* A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.GetObject) */
  /* Insert the implementation of the GetObject method here... */

   int ierr=0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hx;
   bHYPRE_IJParCSRVector px;
   struct bHYPRE_IJParCSRVector__data * p_data;
   HYPRE_ParVector Hpx;
   HYPRE_IJVector Hijx;
   hypre_SStructGrid      * grid;
   int  ilower, iupper;

   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hx = data -> vec;
   grid =  hypre_SStructVectorGrid( Hx );

   px = bHYPRE_IJParCSRVector__create(_ex); SIDL_CHECK(*_ex);
   p_data = bHYPRE_IJParCSRVector__get_data( px );
   ierr += HYPRE_SStructVectorGetObject( Hx, (void **) (&Hpx) );

   /* The purpose of the following block is to convert the HYPRE_ParVector Hpx
      to a HYPRE_IJVector Hijx ... */

   ilower = hypre_ParVectorFirstIndex( (hypre_ParVector *) Hpx );
   iupper = ilower - 1 +
      hypre_VectorSize( hypre_ParVectorLocalVector( (hypre_ParVector *) Hpx ) );

   ierr += HYPRE_IJVectorCreate( data->comm, ilower, iupper, &Hijx );
   ierr += HYPRE_IJVectorSetObjectType( Hijx, HYPRE_PARCSR );
   hypre_IJVectorObject( (hypre_IJVector *) Hijx ) = HYPRE_ParVectorCloneShallow( Hpx );
   ierr += HYPRE_IJVectorInitialize( Hijx );

   /* Now that we have made Hijx from Hpx, load it into p_data, the data of px.
      Then px is ready for output, as A.*/

   p_data -> ij_b = Hijx;
   p_data -> comm = data->comm;

   *A = sidl_BaseInterface__cast( px, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRVector_deleteRef( px, _ex ); SIDL_CHECK(*_ex);

   return( ierr );

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.GetObject) */
  }
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_SetCommunicator(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   /* DEPRECATED   use _Create */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructParCSRVector_Destroy(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Destroy) */
    /* Insert-Code-Here {bHYPRE.SStructParCSRVector.Destroy} (Destroy method) */
     bHYPRE_SStructParCSRVector_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Destroy) */
  }
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Initialize"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_Initialize(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   HYPRE_SStructVectorSetObjectType( Hy, HYPRE_PARCSR );
   ierr = HYPRE_SStructVectorInitialize( Hy );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Initialize) */
  }
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Assemble"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_Assemble(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr = HYPRE_SStructVectorAssemble( Hy );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Assemble) */
  }
}

/*
 * Set {\tt self} to 0.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Clear"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_Clear(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Clear) */
  /* Insert the implementation of the Clear method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   HYPRE_ParVector py;

   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   HYPRE_SStructVectorGetObject( Hy, (void **) &py );

   ierr += HYPRE_ParVectorSetConstantValues( py, 0.0);

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Clear) */
  }
}

/*
 * Copy data from x into {\tt self}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Copy"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_Copy(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Copy) */
  /* Insert the implementation of the Copy method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   struct bHYPRE_SStructParCSRVector__data * datax;
   bHYPRE_SStructParCSRVector bSSx;
   HYPRE_SStructVector Hx;
   HYPRE_SStructVector Hself;
   HYPRE_ParVector px;
   HYPRE_ParVector pself;

   bSSx = (bHYPRE_SStructParCSRVector) bHYPRE_Vector__cast2( x, "bHYPRE.SStructParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   hypre_assert( bSSx!=NULL );

   data = bHYPRE_SStructParCSRVector__get_data( self );
   datax = bHYPRE_SStructParCSRVector__get_data( bSSx );
   Hself = data->vec;
   Hx = datax->vec;
   HYPRE_SStructVectorGetObject( Hself, (void **) &pself );
   HYPRE_SStructVectorGetObject( Hx, (void **) &px );

   ierr += HYPRE_ParVectorCopy( px, pself );

   bHYPRE_SStructParCSRVector_deleteRef( bSSx, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Copy) */
  }
}

/*
 * Create an {\tt x} compatible with {\tt self}.
 * The new vector's data is not specified.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Clone"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_Clone(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* out */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Clone) */
  /* Insert the implementation of the Clone method here... */
   /* Note that Initialize is called on the new vector, but not Assemble. */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data, * data_x;
   bHYPRE_SStructVectorView bHYPRE_x;
   bHYPRE_SStructParCSRVector bHYPREP_x;
   HYPRE_SStructVector Hself, Hx;
   HYPRE_SStructGrid grid;
   MPI_Comm comm;
   int my_id;

   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hself = data->vec;
   comm = data->comm;
   MPI_Comm_rank(comm, &my_id );

   bHYPREP_x = bHYPRE_SStructParCSRVector__create(_ex); SIDL_CHECK(*_ex);
   bHYPRE_x = bHYPRE_SStructVectorView__cast( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   data_x = bHYPRE_SStructParCSRVector__get_data( bHYPREP_x );
   data_x->comm = comm;

   grid = hypre_SStructVectorGrid(Hself);
   ierr += HYPRE_SStructVectorCreate( comm, grid, &Hx );
   ierr += HYPRE_SStructVectorInitialize( Hx );
   data_x -> vec = Hx;

   ierr += HYPRE_SStructVectorSetObjectType(
      Hx,
      hypre_SStructVectorObjectType((hypre_SStructVector *)Hself) );
   ierr += bHYPRE_SStructVectorView_Initialize( bHYPRE_x, _ex ); SIDL_CHECK(*_ex);

   *x = bHYPRE_Vector__cast( bHYPRE_x, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_SStructVectorView_deleteRef( bHYPRE_x, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_SStructParCSRVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   return( ierr );

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Clone) */
  }
}

/*
 * Scale {\tt self} by {\tt a}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Scale"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_Scale(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ double a,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Scale) */
  /* Insert the implementation of the Scale method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   HYPRE_ParVector py;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;
   HYPRE_SStructVectorGetObject( Hy, (void **) &py );

   ierr += HYPRE_ParVectorScale( a, py );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Scale) */
  }
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Dot"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_Dot(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Dot) */
  /* Insert the implementation of the Dot method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   struct bHYPRE_SStructParCSRVector__data * datax;
   bHYPRE_SStructParCSRVector bSSx;
   HYPRE_SStructVector Hself;
   HYPRE_SStructVector Hx;
   HYPRE_ParVector pself;
   HYPRE_ParVector px;

   bSSx = (bHYPRE_SStructParCSRVector) bHYPRE_Vector__cast2( x, "bHYPRE.SStructParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   hypre_assert( bSSx!=NULL );

   data = bHYPRE_SStructParCSRVector__get_data( self );
   datax = bHYPRE_SStructParCSRVector__get_data( bSSx );
   Hself = data->vec;
   Hx = datax->vec;
   HYPRE_SStructVectorGetObject( Hself, (void **) &pself );
   HYPRE_SStructVectorGetObject( Hx, (void **) &px );

   ierr += HYPRE_ParVectorInnerProd( pself, px, d );

   bHYPRE_SStructParCSRVector_deleteRef( bSSx, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Dot) */
  }
}

/*
 * Add {\tt a}{\tt x} to {\tt self}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Axpy"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRVector_Axpy(
  /* in */ bHYPRE_SStructParCSRVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   struct bHYPRE_SStructParCSRVector__data * datax;
   bHYPRE_SStructParCSRVector bSSx;
   HYPRE_SStructVector Hself;
   HYPRE_SStructVector Hx;
   HYPRE_ParVector pself;
   HYPRE_ParVector px;

   bSSx = (bHYPRE_SStructParCSRVector) bHYPRE_Vector__cast2( x, "bHYPRE.SStructParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   hypre_assert( bSSx!=NULL );

   data = bHYPRE_SStructParCSRVector__get_data( self );
   datax = bHYPRE_SStructParCSRVector__get_data( bSSx );
   Hself = data->vec;
   Hx = datax->vec;
   HYPRE_SStructVectorGetObject( Hself, (void **) &pself );
   HYPRE_SStructVectorGetObject( Hx, (void **) &px );

   ierr += HYPRE_ParVectorAxpy( a, px, pself );

   bHYPRE_SStructParCSRVector_deleteRef( bSSx, _ex ); SIDL_CHECK(*_ex);

   return ierr;

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Axpy) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MPICommunicator(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_MatrixVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MatrixVectorView__connectI(url, ar, _ex);
}
struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MatrixVectorView__cast(bi, _ex);
}
struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_ProblemDefinition(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_ProblemDefinition__connectI(url, ar, _ex);
}
struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_ProblemDefinition__cast(bi, _ex);
}
struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_SStructGrid__connectI(url, ar, _ex);
}
struct bHYPRE_SStructGrid__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructGrid(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_SStructGrid__cast(bi, _ex);
}
struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructMatrixVectorView(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_SStructMatrixVectorView__connectI(url, ar, _ex);
}
struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructMatrixVectorView(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_SStructMatrixVectorView__cast(bi, _ex);
}
struct bHYPRE_SStructParCSRVector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructParCSRVector(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_SStructParCSRVector__connectI(url, ar, _ex);
}
struct bHYPRE_SStructParCSRVector__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructParCSRVector(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_SStructParCSRVector__cast(bi, _ex);
}
struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_SStructVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_SStructVectorView__connectI(url, ar, _ex);
}
struct bHYPRE_SStructVectorView__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_SStructVectorView(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_SStructVectorView__cast(bi, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRVector_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRVector_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
