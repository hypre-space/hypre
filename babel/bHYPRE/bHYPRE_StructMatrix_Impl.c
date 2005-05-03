/*
 * File:          bHYPRE_StructMatrix_Impl.c
 * Symbol:        bHYPRE.StructMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.StructMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructMatrix" (version 1.0.0)
 * 
 * A single class that implements both a build interface and an
 * operator interface. It returns itself for GetConstructedObject.
 * A StructMatrix is a matrix on a structured grid.
 * One function unique to a StructMatrix is SetConstantEntries.
 * This declares that matrix entries corresponding to certain stencil points
 * (supplied as stencil element indices) will be constant throughout the grid.
 * 
 */

#include "bHYPRE_StructMatrix_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "mpi.h"
#include "struct_mv.h"
#include "bHYPRE_StructVector_Impl.h"
#include "bHYPRE_StructGrid_Impl.h"
#include "bHYPRE_StructStencil_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix__ctor"

void
impl_bHYPRE_StructMatrix__ctor(
  /*in*/ bHYPRE_StructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */

   /* To build a StructMatrix via Babel: first call the constructor,
      then SetCommunicator, then SetGrid and SetStencil (which internally call
      HYPRE_StructMatrixCreate), then any optional parameter set functions
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

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix__dtor"

void
impl_bHYPRE_StructMatrix__dtor(
  /*in*/ bHYPRE_StructMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix matrix;
   data = bHYPRE_StructMatrix__get_data( self );
   matrix = data -> matrix;
   ierr += HYPRE_StructMatrixDestroy( matrix );
   assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetCommunicator"

int32_t
impl_bHYPRE_StructMatrix_SetCommunicator(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   data = bHYPRE_StructMatrix__get_data( self );
   data -> comm = (MPI_Comm) mpi_comm;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetIntParameter"

int32_t
impl_bHYPRE_StructMatrix_SetIntParameter(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ const char* name,
    /*in*/ int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetDoubleParameter"

int32_t
impl_bHYPRE_StructMatrix_SetDoubleParameter(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ const char* name, /*in*/ double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetStringParameter"

int32_t
impl_bHYPRE_StructMatrix_SetStringParameter(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ const char* name,
    /*in*/ const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetIntArray1Parameter"

int32_t
impl_bHYPRE_StructMatrix_SetIntArray1Parameter(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetIntArray2Parameter"

int32_t
impl_bHYPRE_StructMatrix_SetIntArray2Parameter(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetDoubleArray1Parameter"

int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetDoubleArray1Parameter) 
    */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetDoubleArray2Parameter"

int32_t
impl_bHYPRE_StructMatrix_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetDoubleArray2Parameter) 
    */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_GetIntValue"

int32_t
impl_bHYPRE_StructMatrix_GetIntValue(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ const char* name,
    /*out*/ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_GetDoubleValue"

int32_t
impl_bHYPRE_StructMatrix_GetDoubleValue(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ const char* name,
    /*out*/ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_Setup"

int32_t
impl_bHYPRE_StructMatrix_Setup(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ bHYPRE_Vector b,
    /*in*/ bHYPRE_Vector x)
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

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_Apply"

int32_t
impl_bHYPRE_StructMatrix_Apply(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ bHYPRE_Vector b,
    /*inout*/ bHYPRE_Vector* x)
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
   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.StructVector" ) )
   {
      bHYPREP_b = bHYPRE_StructVector__cast( b );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)b );
   }

   if ( bHYPRE_Vector_queryInt( *x, "bHYPRE.StructVector" ) )
   {
      bHYPREP_x = bHYPRE_StructVector__cast( *x );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
   }

   data_x = bHYPRE_StructVector__get_data( bHYPREP_x );
   Hx = data_x -> vec;
   data_b = bHYPRE_StructVector__get_data( bHYPREP_b );
   Hb = data_b -> vec;

   ierr += HYPRE_StructMatrixMatvec( 1.0, HA, Hb, 0.0, Hx );

   bHYPRE_StructVector_deleteRef( bHYPREP_b ); /* ref was created by queryInt */
   bHYPRE_StructVector_deleteRef( bHYPREP_x ); /* ref was created by queryInt */

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.Apply) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_Initialize"

int32_t
impl_bHYPRE_StructMatrix_Initialize(
  /*in*/ bHYPRE_StructMatrix self)
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

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_Assemble"

int32_t
impl_bHYPRE_StructMatrix_Assemble(
  /*in*/ bHYPRE_StructMatrix self)
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

/*
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a sidl.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_GetObject"

int32_t
impl_bHYPRE_StructMatrix_GetObject(
  /*in*/ bHYPRE_StructMatrix self, /*out*/ sidl_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.GetObject) */
  /* Insert the implementation of the GetObject method here... */
 
   bHYPRE_StructMatrix_addRef( self );
   *A = sidl_BaseInterface__cast( self );
   return( 0 );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.GetObject) */
}

/*
 * Method:  SetGrid[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetGrid"

int32_t
impl_bHYPRE_StructMatrix_SetGrid(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ bHYPRE_StructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */

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
   assert( HA==NULL ); /* shouldn't have already been created */
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

/*
 * Method:  SetStencil[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetStencil"

int32_t
impl_bHYPRE_StructMatrix_SetStencil(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ bHYPRE_StructStencil stencil)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetStencil) */
  /* Insert the implementation of the SetStencil method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;
   HYPRE_StructGrid Hgrid;
   HYPRE_StructStencil Hstencil;
   MPI_Comm comm;
   struct bHYPRE_StructStencil__data * sdata;

   data = bHYPRE_StructMatrix__get_data( self );
   HA = data->matrix;
   assert( HA==NULL ); /* shouldn't have already been created */
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

/*
 * Method:  SetValues[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetValues"

int32_t
impl_bHYPRE_StructMatrix_SetValues(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ struct sidl_int__array* index,
    /*in*/ int32_t num_stencil_indices,
    /*in*/ struct sidl_int__array* stencil_indices,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;
   data = bHYPRE_StructMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_StructMatrixSetValues
      ( HA, sidlArrayAddr1( index, 0 ), num_stencil_indices,
        sidlArrayAddr1( stencil_indices, 0 ), sidlArrayAddr1( values, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetValues) */
}

/*
 * Method:  SetBoxValues[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetBoxValues"

int32_t
impl_bHYPRE_StructMatrix_SetBoxValues(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t num_stencil_indices,
    /*in*/ struct sidl_int__array* stencil_indices,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;
   data = bHYPRE_StructMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_StructMatrixSetBoxValues
      ( HA, sidlArrayAddr1( ilower, 0 ), sidlArrayAddr1( iupper, 0 ),
        num_stencil_indices, sidlArrayAddr1( stencil_indices, 0 ),
        sidlArrayAddr1( values, 0 ) );

   return ierr;


  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetBoxValues) */
}

/*
 * Method:  SetNumGhost[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetNumGhost"

int32_t
impl_bHYPRE_StructMatrix_SetNumGhost(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ struct sidl_int__array* num_ghost)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetNumGhost) */
  /* Insert the implementation of the SetNumGhost method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;

   data = bHYPRE_StructMatrix__get_data( self );
   HA = data->matrix;

   ierr += HYPRE_StructMatrixSetNumGhost( HA, sidlArrayAddr1( num_ghost, 0 ) );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetNumGhost) */
}

/*
 * Method:  SetSymmetric[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetSymmetric"

int32_t
impl_bHYPRE_StructMatrix_SetSymmetric(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ int32_t symmetric)
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

/*
 * Method:  SetConstantEntries[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetConstantEntries"

int32_t
impl_bHYPRE_StructMatrix_SetConstantEntries(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ int32_t num_stencil_constant_points,
    /*in*/ struct sidl_int__array* stencil_constant_points)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructMatrix.SetConstantEntries) */
  /* Insert the implementation of the SetConstantEntries method here... */

   int ierr = 0;
   struct bHYPRE_StructMatrix__data * data;
   HYPRE_StructMatrix HA;

   data = bHYPRE_StructMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_StructMatrixSetConstantEntries
      ( HA, num_stencil_constant_points, sidlArrayAddr1( stencil_constant_points, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetConstantEntries) */
}

/*
 * Method:  SetConstantValues[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructMatrix_SetConstantValues"

int32_t
impl_bHYPRE_StructMatrix_SetConstantValues(
  /*in*/ bHYPRE_StructMatrix self, /*in*/ int32_t num_stencil_indices,
    /*in*/ struct sidl_int__array* stencil_indices,
    /*in*/ struct sidl_double__array* values)
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
      sidlArrayAddr1( stencil_indices, 0 ), sidlArrayAddr1( values, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructMatrix.SetConstantValues) */
}
