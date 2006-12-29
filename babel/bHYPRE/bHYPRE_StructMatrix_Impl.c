/*
 * File:          bHYPRE_StructMatrix_Impl.c
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructMatrix" (version 1.0.0)
 * 
 * A single class that implements both a view interface and an
 * operator interface.
 * A StructMatrix is a matrix on a structured grid.
 * One function unique to a StructMatrix is SetConstantEntries.
 * This declares that matrix entries corresponding to certain stencil points
 * (supplied as stencil element indices) will be constant throughout the grid.
 */

#include "bHYPRE_StructMatrix_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix._includes) */
/* Put additional includes or other arbitrary code here... */


#include <assert.h>
#include "hypre_babel_exception_handler.h"
/*#include "mpi.h"*/
#include "_hypre_struct_mv.h"
#include "bHYPRE_StructVector_Impl.h"
#include "bHYPRE_StructGrid_Impl.h"
#include "bHYPRE_StructStencil_Impl.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructMatrix__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix._load) */
  /* Insert-Code-Here {bHYPRE.StructMatrix._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructMatrix__ctor(
  /* in */ bHYPRE_StructMatrix self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */

   /* To build a StructMatrix via Babel: first call Create.
      (User calls of __create are DEPRECATED.)
      Then call any optional parameter set functions
      (e.g. SetSymmetric) then Initialize, then value set functions (presently
      SetValues or SetBoxValues), and finally Assemble (Setup is equivalent to Assemble).
    */

   struct bHYPRE_StructMatrix__data * data;
   data = hypre_CTAlloc( struct bHYPRE_StructMatrix__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> grid = NULL;
   data -> stencil = NULL;
   data -> matrix = NULL;
   bHYPRE_StructMatrix__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructMatrix__ctor2(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix._ctor2) */
    /* Insert-Code-Here {bHYPRE.StructMatrix._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructMatrix__dtor(
  /* in */ bHYPRE_StructMatrix self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix matrix;
   data = bHYPRE_StructMatrix__get_data( self );
   matrix = data -> matrix;
   if ( matrix ) ierr += HYPRE_StructMatrixDestroy( matrix );
   hypre_assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix._dtor) */
  }
}

/*
 *  This function is the preferred way to create a Struct Matrix. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_StructMatrix
impl_bHYPRE_StructMatrix_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ bHYPRE_StructGrid grid,
  /* in */ bHYPRE_StructStencil stencil,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.Create) */
  /* Insert-Code-Here {bHYPRE.StructMatrix.Create} (Create method) */

   int ierr = 0;
   bHYPRE_StructMatrix mat;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix Hmat;
   struct bHYPRE_StructGrid__data * gdata;
   HYPRE_StructGrid Hgrid;
   struct bHYPRE_StructStencil__data * sdata;
   HYPRE_StructStencil Hstencil;
   MPI_Comm comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   mat = bHYPRE_StructMatrix__create(_ex); SIDL_CHECK(*_ex);
   data = bHYPRE_StructMatrix__get_data( mat );

   gdata = bHYPRE_StructGrid__get_data( grid );
   Hgrid = gdata->grid;

   sdata = bHYPRE_StructStencil__get_data( stencil );
   Hstencil = sdata->stencil;

   ierr += HYPRE_StructMatrixCreate( comm, Hgrid, Hstencil, &Hmat );
   data->matrix = Hmat;
   data->comm = comm;

   return( mat );

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.Create) */
  }
}

/*
 *  Set the grid on which vectors are defined.  This and the stencil
 * determine the matrix structure. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetGrid"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetGrid(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_StructGrid grid,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */

   /* DEPRECATED   call Create */

   /* To create a matrix one needs a grid, stencil, and communicator.
      We assume SetCommunicator will be called first or can be changed.
      SetGrid and SetStencil both check for whether the other one has been called.
      If both have been called, we have enough information to call
      HYPRE_StructMatrixCreate, so we do so.  It is an error to call this function
      if HYPRE_StructMatrixCreate has already been called for this matrix.
   */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;
   HYPRE_StructGrid Hgrid;
   HYPRE_StructStencil Hstencil;
   MPI_Comm comm;
   struct bHYPRE_StructGrid__data * gdata;

   data = bHYPRE_StructMatrix__get_data( self );
   HA = data->matrix;
   hypre_assert( HA==NULL ); /* shouldn't have already been created */
   comm = data->comm;
   Hstencil = data->stencil;

   gdata = bHYPRE_StructGrid__get_data( grid );
   Hgrid = gdata->grid;
   data->grid = Hgrid;

   if ( Hstencil != NULL )
   {
      ierr += HYPRE_StructMatrixCreate( comm, Hgrid, Hstencil, &HA );
      data->matrix = HA;
   }

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetGrid) */
  }
}

/*
 *  Set the stencil. This and the grid determine the matrix structure. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetStencil"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetStencil(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_StructStencil stencil,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetStencil) */
  /* Insert the implementation of the SetStencil method here... */

   /* DEPRECATED   call Create */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;
   HYPRE_StructGrid Hgrid;
   HYPRE_StructStencil Hstencil;
   MPI_Comm comm;
   struct bHYPRE_StructStencil__data * sdata;

   data = bHYPRE_StructMatrix__get_data( self );
   HA = data->matrix;
   hypre_assert( HA==NULL ); /* shouldn't have already been created */
   comm = data->comm;
   Hgrid = data->grid;

   sdata = bHYPRE_StructStencil__get_data( stencil );
   Hstencil = sdata->stencil;
   data->stencil = Hstencil;

   if ( Hgrid != NULL )
   {
      ierr += HYPRE_StructMatrixCreate( comm, Hgrid, Hstencil, &HA );
      data->matrix = HA;
   }

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetStencil) */
  }
}

/*
 *  Set matrix values at grid point, given by "index".
 * You can supply values for one or more positions in the stencil.
 * "index" is an array of size "dim"; and "stencil_indices" and "values"
 * are arrays of size "num_stencil_indices".
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in rarray[dim] */ int32_t* index,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */ double* values,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;
   data = bHYPRE_StructMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_StructMatrixSetValues
      ( HA, index, num_stencil_indices,
        stencil_indices, values );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetValues) */
  }
}

/*
 *  Set matrix values throughout a box in the grid, specified by its lower
 * and upper corners.  You can supply these values for one or more positions
 * in the stencil.  Thus the total number of matrix values you supply,
 * "nvalues", is num_stencil_indices x box_size, where box_size is the
 * number of grid points in the box.  The values array should be organized
 * so all values for a given box point are together (i.e., the stencil
 * index is the most rapidly varying).
 * "ilower" and "iupper" are arrays of size "dim", "stencil_indices" is an
 * array of size "num_stencil_indices", and "values" is an array of size
 * "nvalues". 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetBoxValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetBoxValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in rarray[dim] */ int32_t* ilower,
  /* in rarray[dim] */ int32_t* iupper,
  /* in */ int32_t dim,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[nvalues] */ double* values,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;
   data = bHYPRE_StructMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_StructMatrixSetBoxValues
      ( HA, ilower, iupper,
        num_stencil_indices, stencil_indices,
        values );

   return ierr;


  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetBoxValues) */
  }
}

/*
 *  Set the number of ghost zones, separately on the lower and upper sides
 * for each dimension.
 * "num_ghost" is an array of size "dim2", twice the number of dimensions
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetNumGhost"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetNumGhost(
  /* in */ bHYPRE_StructMatrix self,
  /* in rarray[dim2] */ int32_t* num_ghost,
  /* in */ int32_t dim2,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetNumGhost) */
  /* Insert the implementation of the SetNumGhost method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;

   data = bHYPRE_StructMatrix__get_data( self );
   HA = data->matrix;

   ierr += HYPRE_StructMatrixSetNumGhost( HA, num_ghost );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetNumGhost) */
  }
}

/*
 *  Call SetSymmetric with symmetric=1 to turn on symmetric matrix storage if
 * available. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetSymmetric"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetSymmetric(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ int32_t symmetric,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetSymmetric) */
  /* Insert the implementation of the SetSymmetric method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;

   data = bHYPRE_StructMatrix__get_data( self );
   HA = data->matrix;

   ierr += HYPRE_StructMatrixSetSymmetric( HA, symmetric );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetSymmetric) */
  }
}

/*
 *  State which stencil entries are constant over the grid.
 * Supported options are: (i) none (the default),
 * (ii) all (stencil_constant_points should include all stencil points)
 * (iii) all entries but the diagonal. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetConstantEntries"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetConstantEntries(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ int32_t num_stencil_constant_points,
  /* in rarray[num_stencil_constant_points] */ int32_t* stencil_constant_points,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetConstantEntries) */
  /* Insert the implementation of the SetConstantEntries method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;

   data = bHYPRE_StructMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_StructMatrixSetConstantEntries
      ( HA, num_stencil_constant_points, stencil_constant_points );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetConstantEntries) */
  }
}

/*
 *  Provide values for matrix coefficients which are constant throughout
 * the grid, one value for each stencil point.
 * "stencil_indices" and "values" is each an array of length
 * "num_stencil_indices" 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetConstantValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetConstantValues(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ int32_t num_stencil_indices,
  /* in rarray[num_stencil_indices] */ int32_t* stencil_indices,
  /* in rarray[num_stencil_indices] */ double* values,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetConstantValues) */
  /* Insert the implementation of the SetConstantValues method here... */


   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;
   data = bHYPRE_StructMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_StructMatrixSetConstantValues(
      HA, num_stencil_indices,
      stencil_indices, values );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetConstantValues) */
  }
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetCommunicator(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   /* DEPRECATED   call Create */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   data = bHYPRE_StructMatrix__get_data( self );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetCommunicator) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructMatrix_Destroy(
  /* in */ bHYPRE_StructMatrix self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.Destroy) */
    /* Insert-Code-Here {bHYPRE.StructMatrix.Destroy} (Destroy method) */
     bHYPRE_StructMatrix_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.Destroy) */
  }
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_Initialize"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_Initialize(
  /* in */ bHYPRE_StructMatrix self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.Initialize) */
  /* Insert the implementation of the Initialize method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;

   data = bHYPRE_StructMatrix__get_data( self );

   HA = data -> matrix;

   ierr = HYPRE_StructMatrixInitialize( HA );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.Initialize) */
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
#define __FUNC__ "impl_bHYPRE_StructMatrix_Assemble"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_Assemble(
  /* in */ bHYPRE_StructMatrix self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.Assemble) */
  /* Insert the implementation of the Assemble method here... */

   int ierr=0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;

   data = bHYPRE_StructMatrix__get_data( self );

   HA = data -> matrix;

   ierr += HYPRE_StructMatrixAssemble( HA );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.Assemble) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetIntParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetIntParameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetIntParameter) */
  }
}

/*
 * Set the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetDoubleParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetDoubleParameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetDoubleParameter) */
  }
}

/*
 * Set the string parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetStringParameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetStringParameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetStringParameter) */
  }
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetIntArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetIntArray1Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ int32_t* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetIntArray1Parameter) */
  }
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetIntArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetIntArray2Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in array<int,2,column-major> */ struct sidl_int__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetIntArray2Parameter) */
  }
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetDoubleArray1Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray1Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in rarray[nvalues] */ double* value,
  /* in */ int32_t nvalues,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetDoubleArray1Parameter) 
    */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetDoubleArray1Parameter) */
  }
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetDoubleArray2Parameter"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray2Parameter(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* in array<double,2,column-major> */ struct sidl_double__array* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetDoubleArray2Parameter) 
    */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetDoubleArray2Parameter) */
  }
}

/*
 * Set the int parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_GetIntValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_GetIntValue(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;

   data = bHYPRE_StructMatrix__get_data( self );
   HA = data -> matrix;

   if ( strcmp(name,"ConstantCoefficient")==0 )
   {
      *value = hypre_StructMatrixConstantCoefficient( (hypre_StructMatrix *)HA );
   }
   else
   {
      ierr = 1;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.GetIntValue) */
  }
}

/*
 * Get the double parameter associated with {\tt name}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_GetDoubleValue"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_GetDoubleValue(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ const char* name,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.GetDoubleValue) */
  }
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_Setup"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_Setup(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* in */ bHYPRE_Vector x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.Setup) */
  /* Insert the implementation of the Setup method here... */

   int ierr=0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;

   data = bHYPRE_StructMatrix__get_data( self );
   HA = data -> matrix;

   ierr = HYPRE_StructMatrixAssemble( HA );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.Setup) */
  }
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_Apply"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_Apply(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.Apply) */
  /* Insert the implementation of the Apply method here... */

   /* Apply means to multiply by a vector, x = A*b .  Here, we call
    * the HYPRE Matvec function which performs x = a*A*b + b*x (we set
    * a=1 and b=0).  */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   struct bHYPRE_StructVector__data * data_x, * data_b;
   bHYPRE_StructVector bHYPREP_b, bHYPREP_x;
   HYPRE_StructMatrix HA;
   HYPRE_StructVector Hx, Hb;

   data = bHYPRE_StructMatrix__get_data( self );
   HA = data -> matrix;

   /* A bHYPRE_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   bHYPREP_b = (bHYPRE_StructVector) bHYPRE_Vector__cast2( b, "bHYPRE.StructVector", _ex ); SIDL_CHECK(*_ex);
   if ( bHYPREP_b==NULL ) hypre_assert( "Unrecognized vector type."==(char *)b );

   bHYPREP_x = (bHYPRE_StructVector) bHYPRE_Vector__cast2( *x, "bHYPRE.StructVector", _ex ); SIDL_CHECK(*_ex);
   if ( bHYPREP_x==NULL ) hypre_assert( "Unrecognized vector type."==(char *)x );

   data_x = bHYPRE_StructVector__get_data( bHYPREP_x );
   Hx = data_x -> vec;
   data_b = bHYPRE_StructVector__get_data( bHYPREP_b );
   Hb = data_b -> vec;

   ierr += HYPRE_StructMatrixMatvec( 1.0, HA, Hb, 0.0, Hx );

   bHYPRE_StructVector_deleteRef( bHYPREP_b, _ex ); SIDL_CHECK(*_ex);
   bHYPRE_StructVector_deleteRef( bHYPREP_x, _ex ); SIDL_CHECK(*_ex);

   return( ierr );

   hypre_babel_exception_return_error(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.Apply) */
  }
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_ApplyAdjoint"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructMatrix_ApplyAdjoint(
  /* in */ bHYPRE_StructMatrix self,
  /* in */ bHYPRE_Vector b,
  /* inout */ bHYPRE_Vector* x,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.ApplyAdjoint) */
  /* Insert-Code-Here {bHYPRE.StructMatrix.ApplyAdjoint} (ApplyAdjoint method) */

   return 1; /* not implemented */

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.ApplyAdjoint) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_MPICommunicator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connectI(url, ar, _ex);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_MPICommunicator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MPICommunicator__cast(bi, _ex);
}
struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_MatrixVectorView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_MatrixVectorView__connectI(url, ar, _ex);
}
struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_MatrixVectorView(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_MatrixVectorView__cast(bi, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Operator(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_Operator__connectI(url, ar, _ex);
}
struct bHYPRE_Operator__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_Operator(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Operator__cast(bi, _ex);
}
struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_ProblemDefinition(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_ProblemDefinition__connectI(url, ar, _ex);
}
struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_ProblemDefinition(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_ProblemDefinition__cast(bi, _ex);
}
struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructGrid(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_StructGrid__connectI(url, ar, _ex);
}
struct bHYPRE_StructGrid__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructGrid(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_StructGrid__cast(bi, _ex);
}
struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrix(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_StructMatrix__connectI(url, ar, _ex);
}
struct bHYPRE_StructMatrix__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructMatrix(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_StructMatrix__cast(bi, _ex);
}
struct bHYPRE_StructMatrixView__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructMatrixView(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_StructMatrixView__connectI(url, ar, _ex);
}
struct bHYPRE_StructMatrixView__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructMatrixView(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_StructMatrixView__cast(bi, _ex);
}
struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_StructStencil(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_StructStencil__connectI(url, ar, _ex);
}
struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_StructStencil(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_StructStencil__cast(bi, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructMatrix_fconnect_bHYPRE_Vector(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connectI(url, ar, _ex);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_StructMatrix_fcast_bHYPRE_Vector(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_Vector__cast(bi, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_StructMatrix_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructMatrix_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructMatrix_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructMatrix_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructMatrix_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
