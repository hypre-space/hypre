/*
 * File:          bHYPRE_SStructParCSRMatrix_Impl.c
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.SStructParCSRMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructParCSRMatrix" (version 1.0.0)
 * 
 * The SStructParCSR matrix class.
 * 
 * Objects of this type can be cast to SStructMatrixView or
 * Operator objects using the {\tt \_\_cast} methods.
 */

#include "bHYPRE_SStructParCSRMatrix_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix._includes) */
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
#include "bHYPRE_SStructParCSRVector_Impl.h"
#include "bHYPRE_SStructGraph_Impl.h"
#include "bHYPRE_IJParCSRMatrix_Impl.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructParCSRMatrix__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix._load) */
  /* Insert-Code-Here {bHYPRE.SStructParCSRMatrix._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructParCSRMatrix__ctor(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */

   /* To build a SStructParCSRMatrix via Babel: first call _Create (not __create),
      then any optional parameter set functions
      (e.g. SetSymmetric) then Initialize, then value set functions (such as
      SetValues or SetBoxValues), and finally Assemble (Setup is equivalent to Assemble).
    */

   struct bHYPRE_SStructParCSRMatrix__data * data;
   data = hypre_CTAlloc( struct bHYPRE_SStructParCSRMatrix__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> matrix = NULL;
   bHYPRE_SStructParCSRMatrix__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructParCSRMatrix__ctor2(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix._ctor2) */
    /* Insert-Code-Here {bHYPRE.SStructParCSRMatrix._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructParCSRMatrix__dtor(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix matrix;
   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   matrix = data -> matrix;
   if ( matrix ) ierr += HYPRE_SStructMatrixDestroy( matrix );
   hypre_assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix._dtor) */
  }
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_SStructParCSRMatrix
impl_bHYPRE_SStructParCSRMatrix_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_SStructGraph graph,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.Create) */
  /* Insert-Code-Here {bHYPRE.SStructParCSRMatrix.Create} (Create method) */

   int ierr = 0;
   bHYPRE_SStructParCSRMatrix mat;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix Hmat;
   struct bHYPRE_SStructGraph__data * gdata;
   HYPRE_SStructGraph Hgraph;
   MPI_Comm comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   mat = bHYPRE_SStructParCSRMatrix__create(_ex); SIDL_CHECK(*_ex);
   data = bHYPRE_SStructParCSRMatrix__get_data( mat );
   Hmat = data->matrix;

   gdata = bHYPRE_SStructGraph__get_data( graph );
   Hgraph = gdata->graph;

   ierr += HYPRE_SStructMatrixCreate( comm, Hgraph, &Hmat );
   data->matrix = Hmat;
   data->comm = comm;

   return( mat );

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.Create) */
  }
}

/*
 * Set the matrix graph.
 * DEPRECATED     Use Create
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetGraph"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetGraph(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_SStructGraph graph,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetGraph) */
  /* Insert the implementation of the SetGraph method here... */

   /* To create a matrix one needs a graph and communicator.
      We assume SetCommunicator will be called first.
      So SetGraph can simply call HYPRE_StructMatrixCreate.
      It is an error to call this function
      if HYPRE_StructMatrixCreate has already been called for this matrix.
   */


   /* DEPRECATED   use _Create */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;
   HYPRE_SStructGraph Hgraph;
   MPI_Comm comm;
   struct bHYPRE_SStructGraph__data * gdata;

   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data->matrix;
   hypre_assert( HA==NULL ); /* shouldn't have already been created */
   comm = data->comm;

   gdata = bHYPRE_SStructGraph__get_data( graph );
   Hgraph = gdata->graph;

   ierr += HYPRE_SStructMatrixCreate( comm, Hgraph, &HA );
   data->matrix = HA;

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetGraph) */
  }
}

/*
 * Set matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nentries] */ double* values,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;
   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_SStructMatrixSetValues
      ( HA, part, index, var, nentries,
        entries, values );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetValues) */
  }
}

/*
 * Set matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetBoxValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetBoxValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;
   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_SStructMatrixSetBoxValues
      ( HA, part, ilower, iupper,
        var, nentries, entries, values );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetBoxValues) */
  }
}

/*
 * Add to matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_AddToValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nentries] */ double* values,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;
   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_SStructMatrixAddToValues
      ( HA, part, index, var, nentries,
        entries, values );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.AddToValues) */
  }
}

/*
 * Add to matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of stencil
 * type.  Also, they must all represent couplings to the same
 * variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_AddToBoxValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToBoxValues(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* in */ int32_t nentries,
  /* in rarray[nentries] */ int32_t* entries,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.AddToBoxValues) */
  /* Insert the implementation of the AddToBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;
   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_SStructMatrixAddToBoxValues
      ( HA, part, ilower, iupper,
        var, nentries, entries, values );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.AddToBoxValues) */
  }
}

/*
 * Define symmetry properties for the stencil entries in the
 * matrix.  The boolean argument {\tt symmetric} is applied to
 * stencil entries on part {\tt part} that couple variable {\tt
 * var} to variable {\tt to\_var}.  A value of -1 may be used
 * for {\tt part}, {\tt var}, or {\tt to\_var} to specify
 * ``all''.  For example, if {\tt part} and {\tt to\_var} are
 * set to -1, then the boolean is applied to stencil entries on
 * all parts that couple variable {\tt var} to all other
 * variables.
 * 
 * By default, matrices are assumed to be nonsymmetric.
 * Significant storage savings can be made if the matrix is
 * symmetric.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetSymmetric"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetSymmetric(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t part,
  /* in */ int32_t var,
  /* in */ int32_t to_var,
  /* in */ int32_t symmetric,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetSymmetric) */
  /* Insert the implementation of the SetSymmetric method here... */

   int ierr=0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;

   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_SStructMatrixSetSymmetric( HA, part, var, to_var, symmetric );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetSymmetric) */
  }
}

/*
 * Define symmetry properties for all non-stencil matrix
 * entries.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetNSSymmetric"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetNSSymmetric(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ int32_t symmetric,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetNSSymmetric) */
  /* Insert the implementation of the SetNSSymmetric method here... */

   int ierr=0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;

   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_SStructMatrixSetNSSymmetric( HA, symmetric );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetNSSymmetric) */
  }
}

/*
 * Set the matrix to be complex.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetComplex"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetComplex(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetComplex) */
  /* Insert the implementation of the SetComplex method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetComplex) */
  }
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Print"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_Print(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* filename,
  /* in */ int32_t all,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.Print) */
  /* Insert the implementation of the Print method here... */

   int ierr=0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;

   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_SStructMatrixPrint( filename, HA, all );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.Print) */
  }
}

/*
 * A semi-structured matrix or vector contains a Struct or IJ matrix
 * or vector.  GetObject returns it.
 * The returned type is a sidl.BaseInterface.
 * A cast must be used on the returned object to convert it into a known type.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_GetObject"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_GetObject(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface* A,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.GetObject) */
  /* Insert the implementation of the GetObject method here... */
 
   /* bHYPRE_SStructMatrix_addRef( self );*/
   /* *A = sidl_BaseInterface__cast( self );*/
   /* the matrix needs to be made into a struct or parcsr matrix for solver use,
    parcsr here (struct in the case of SStructMatrix) */

   int ierr=0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;
   bHYPRE_IJParCSRMatrix pA;
   struct bHYPRE_IJParCSRMatrix__data * p_data;
   HYPRE_IJMatrix ijA;
   int ilower, iupper, jlower, jupper;

   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;
   ierr += HYPRE_SStructMatrixGetObject2( HA, (void **) (&ijA) );
   /* ...Be careful about this HYPRE_IJMatrix ijA.  There are now two pointers
    to the same HYPRE_IJMatrix, ijA and something inside HA.  They don't know
    about each other, and if you use one to destroy it once you mustn't use the
    other to destroy it again. It would be better to use reference counting for
    IJ matrices, as is done for SStruct matrices.  My solution for here involves
    an owns_matrix flag, see below. */

   HYPRE_IJMatrixGetLocalRange( ijA, &ilower, &iupper, &jlower, &jupper );

   pA = bHYPRE_IJParCSRMatrix__create(_ex); SIDL_CHECK(*_ex);
   p_data = bHYPRE_IJParCSRMatrix__get_data( pA );
   p_data->ij_A = ijA;
   p_data->owns_matrix = 0;  /* the matrix still belongs to "self", not to pA. */
   p_data->comm = data -> comm;
   /* The grid and stencil slots of p_data haven't been set, but they shouldn't
      be needed- they are just used for creation of the HYPRE_StructMatrix object.
    */
   *A = sidl_BaseInterface__cast( pA, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_IJParCSRMatrix_deleteRef( pA, _ex ); SIDL_CHECK(*_ex);

   return( ierr );

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.GetObject) */
  }
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetCommunicator(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   /* DEPRECATED   use _Create */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetCommunicator) */
  }
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Initialize"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_Initialize(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.Initialize) */
  /* Insert the implementation of the Initialize method here... */

   int ierr=0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;

   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   HYPRE_SStructMatrixSetObjectType( HA, HYPRE_PARCSR );
   ierr = HYPRE_SStructMatrixInitialize( HA );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.Initialize) */
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
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Assemble"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_Assemble(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.Assemble) */
  /* Insert the implementation of the Assemble method here... */

   int ierr=0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;

   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   ierr = HYPRE_SStructMatrixAssemble( HA );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.Assemble) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntParameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleParameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetDoubleParameter) 
    */
  /* Insert the implementation of the SetDoubleParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetStringParameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetStringParameter) 
    */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_GetIntValue(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_GetDoubleValue(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_Setup(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.Setup) */
  /* Insert the implementation of the Setup method here... */

   int ierr=0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;

   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   ierr = HYPRE_SStructMatrixAssemble( HA );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_Apply(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.Apply) */
  /* Insert the implementation of the Apply method here... */

   /* Apply means to multiply by a vector, x = A*b .  Here, we call
    * the HYPRE Matvec function which performs x = alpha*A*b + beta*x (we set
    * alpha=1 and beta=0).  */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   struct bHYPRE_SStructParCSRVector__data * data_x, * data_b;
   bHYPRE_SStructParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_SStructMatrix HA;
   HYPRE_SStructVector Hx, Hb;
   HYPRE_ParCSRMatrix pA;
   HYPRE_ParVector pb;
   HYPRE_ParVector px;

   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   /* A bHYPRE_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   bHYPREP_b = (bHYPRE_SStructParCSRVector) bHYPRE_Vector__cast2(b, "bHYPRE.SStructParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   hypre_assert( bHYPREP_b!=NULL );

   bHYPREP_x = (bHYPRE_SStructParCSRVector) bHYPRE_Vector__cast2( *x, "bHYPRE.SStructParCSRVector", _ex );
   SIDL_CHECK(*_ex);
   hypre_assert( bHYPREP_x!=NULL );

   data_x = bHYPRE_SStructParCSRVector__get_data( bHYPREP_x );
   Hx = data_x -> vec;
   data_b = bHYPRE_SStructParCSRVector__get_data( bHYPREP_b );
   Hb = data_b -> vec;

   HYPRE_SStructMatrixGetObject( HA, (void **) &pA);
   HYPRE_SStructVectorGetObject( Hb, (void **) &pb);
   HYPRE_SStructVectorGetObject( Hx, (void **) &px);

   ierr += HYPRE_ParCSRMatrixMatvec( 1.0, pA, pb, 0.0, px );

   bHYPRE_SStructVector_deleteRef( bHYPREP_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_SStructVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   return( ierr );

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructParCSRMatrix_ApplyAdjoint(
  /* in */ bHYPRE_SStructParCSRMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.SStructParCSRMatrix.ApplyAdjoint} (ApplyAdjoint method) */

   return 1; /* not implemented */

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_MPICommunicator(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_MatrixVectorView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MatrixVectorView__connectI(url, ar, _ex);
}
struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MatrixVectorView__cast(bi, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_ProblemDefinition(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_ProblemDefinition__connectI(url, ar, _ex);
}
struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_ProblemDefinition__cast(bi, _ex);
}
struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructGraph(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_SStructGraph__connectI(url, ar, _ex);
}
struct bHYPRE_SStructGraph__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructGraph(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_SStructGraph__cast(bi, _ex);
}
struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructMatrixVectorView(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_SStructMatrixVectorView__connectI(url, ar, _ex);
}
struct bHYPRE_SStructMatrixVectorView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructMatrixVectorView(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_SStructMatrixVectorView__cast(bi, _ex);
}
struct bHYPRE_SStructMatrixView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructMatrixView(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_SStructMatrixView__connectI(url, ar, _ex);
}
struct bHYPRE_SStructMatrixView__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructMatrixView(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_SStructMatrixView__cast(bi, _ex);
}
struct bHYPRE_SStructParCSRMatrix__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_SStructParCSRMatrix(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_SStructParCSRMatrix__connectI(url, ar, _ex);
}
struct bHYPRE_SStructParCSRMatrix__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_SStructParCSRMatrix(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_SStructParCSRMatrix__cast(bi, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_bHYPRE_Vector(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRMatrix_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructParCSRMatrix_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
