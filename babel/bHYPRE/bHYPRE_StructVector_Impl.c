/*
 * File:          bHYPRE_StructVector_Impl.c
 * Symbol:        bHYPRE.StructVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:37 PST
 * Generated:     20050225 15:45:40 PST
 * Description:   Server-side implementation for bHYPRE.StructVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 1129
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.StructVector" (version 1.0.0)
 */

#include "bHYPRE_StructVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "mpi.h"
#include "struct_mv.h"
#include "bHYPRE_StructGrid_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector__ctor"

void
impl_bHYPRE_StructVector__ctor(
  /*in*/ bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._ctor) */
  /* Insert the implementation of the constructor method here... */

  /* How to make a vector: first call the constructor.
     Then SetCommunicator, then SetGrid (which calls HYPRE_StructVectorCreate),
     then Initialize, then SetValue (or SetBoxValues, etc.), then Assemble.
     Or you can call Clone.
  */

   struct bHYPRE_StructVector__data * data;
   data = hypre_CTAlloc( struct bHYPRE_StructVector__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> vec = NULL;
   bHYPRE_StructVector__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector__dtor"

void
impl_bHYPRE_StructVector__dtor(
  /*in*/ bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector vec;
   data = bHYPRE_StructVector__get_data( self );
   vec = data -> vec;
   ierr += HYPRE_StructVectorDestroy( vec );
   assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector._dtor) */
}

/*
 * Set {\tt self} to 0.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Clear"

int32_t
impl_bHYPRE_StructVector_Clear(
  /*in*/ bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Clear) */
  /* Insert the implementation of the Clear method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector vec;
   data = bHYPRE_StructVector__get_data( self );
   vec = data -> vec;
   ierr += HYPRE_StructVectorSetConstantValues( vec, 0 );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Clear) */
}

/*
 * Copy x into {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Copy"

int32_t
impl_bHYPRE_StructVector_Copy(
  /*in*/ bHYPRE_StructVector self, /*in*/ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Copy) */
  /* Insert the implementation of the Copy method here... */

   /* Copy the contents of x onto self.
      Self has its own data array, so this is a deep copy in that sense.
      The grid and other size information are not copied -
      assumed to be consistent already. */

      int ierr = 0;
      struct bHYPRE_StructVector__data * data_x;
      struct bHYPRE_StructVector__data * data;
      bHYPRE_StructVector xx;
      HYPRE_StructVector vec_x;
      HYPRE_StructVector vec;

      /* A bHYPRE_Vector is just an interface, we have no knowledge of its
       * contents.  Check whether it's something we know how to handle.
       * If not, die. */
      if ( bHYPRE_Vector_queryInt(x, "bHYPRE.StructVector" ) )
      {
         xx = bHYPRE_StructVector__cast( x );
      }
      else
      {
         assert( "Unrecognized vector type."==(char *)x );
      }

      data_x = bHYPRE_StructVector__get_data( xx );
      data   = bHYPRE_StructVector__get_data( self );
      vec_x = data_x -> vec;
      vec   = data -> vec;
      ierr += HYPRE_StructVectorCopy( vec_x, vec );

      return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Copy) */
}

/*
 * Create an {\tt x} compatible with {\tt self}.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Clone"

int32_t
impl_bHYPRE_StructVector_Clone(
  /*in*/ bHYPRE_StructVector self, /*out*/ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Clone) */
  /* Insert the implementation of the Clone method here... */

   /* creates and returns a copy of self */
   /* This is a deep copy in terms of the data array and num_ghost,
      shallow in terms of the grid */

   int ierr = 0;
   int my_id;
   struct bHYPRE_StructVector__data * data, * data_x;
   bHYPRE_StructBuildVector bHYPRE_x;
   bHYPRE_StructVector bHYPREP_x;
   HYPRE_StructVector yy, xx;
   HYPRE_StructGrid grid;
   int * num_ghost;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_id );

   bHYPREP_x = bHYPRE_StructVector__create();
   bHYPRE_x = bHYPRE_StructBuildVector__cast( bHYPREP_x );

   data = bHYPRE_StructVector__get_data( self );
   data_x = bHYPRE_StructVector__get_data( bHYPREP_x );

   data_x->comm = data->comm;

   yy = data->vec;

   grid = hypre_StructVectorGrid(yy);
   ierr += HYPRE_StructVectorCreate( data_x->comm, grid, &xx );
   ierr += HYPRE_StructVectorInitialize( xx );
   data_x -> vec = xx;

   num_ghost = hypre_StructVectorNumGhost(yy);
   ierr += HYPRE_StructVectorSetNumGhost( xx, num_ghost );

   /* Copy data in y to x... */
   HYPRE_StructVectorCopy( yy, xx );

   ierr += bHYPRE_StructBuildVector_Initialize( bHYPRE_x );

   *x = bHYPRE_Vector__cast( bHYPRE_x );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Clone) */
}

/*
 * Scale {\tt self} by {\tt a}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Scale"

int32_t
impl_bHYPRE_StructVector_Scale(
  /*in*/ bHYPRE_StructVector self, /*in*/ double a)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Scale) */
  /* Insert the implementation of the Scale method here... */
   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector vec;
   data = bHYPRE_StructVector__get_data( self );
   vec = data -> vec;
   ierr += HYPRE_StructVectorScaleValues( vec, a );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Scale) */
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Dot"

int32_t
impl_bHYPRE_StructVector_Dot(
  /*in*/ bHYPRE_StructVector self, /*in*/ bHYPRE_Vector x, /*out*/ double* d)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Dot) */
  /* Insert the implementation of the Dot method here... */
   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector vec;
   hypre_StructVector *hx;
   hypre_StructVector *hy;
   bHYPRE_StructVector xx;

   data = bHYPRE_StructVector__get_data( self );
   vec = data -> vec;
   hy = (hypre_StructVector *) vec;

   /* A bHYPRE_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( bHYPRE_Vector_queryInt(x, "bHYPRE.StructVector" ) )
   {
      xx = bHYPRE_StructVector__cast( x );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
   }
   data = bHYPRE_StructVector__get_data( xx );
   vec = data -> vec;
   hx = (hypre_StructVector *) vec;

   *d = hypre_StructInnerProd(  hx, hy );
   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Dot) */
}

/*
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Axpy"

int32_t
impl_bHYPRE_StructVector_Axpy(
  /*in*/ bHYPRE_StructVector self, /*in*/ double a, /*in*/ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data, * data_x;
   bHYPRE_StructVector bHYPREP_x;
   HYPRE_StructVector Hy, Hx;

   data = bHYPRE_StructVector__get_data( self );
   Hy = data -> vec;

   /* A bHYPRE_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( bHYPRE_Vector_queryInt(x, "bHYPRE.StructVector" ) )
   {
      bHYPREP_x = bHYPRE_StructVector__cast( x );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
   }

   data_x = bHYPRE_StructVector__get_data( bHYPREP_x );
   bHYPRE_StructVector_deleteRef( bHYPREP_x );
   Hx = data_x->vec;

   ierr += hypre_StructAxpy( a, (hypre_StructVector *) Hx,
                             (hypre_StructVector *) Hy );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Axpy) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetCommunicator"

int32_t
impl_bHYPRE_StructVector_SetCommunicator(
  /*in*/ bHYPRE_StructVector self, /*in*/ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   data = bHYPRE_StructVector__get_data( self );
   data -> comm = (MPI_Comm) mpi_comm;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_Initialize"

int32_t
impl_bHYPRE_StructVector_Initialize(
  /*in*/ bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector Hy;
   data = bHYPRE_StructVector__get_data( self );
   Hy = data -> vec;
   ierr = HYPRE_StructVectorInitialize( Hy );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Initialize) */
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
#define __FUNC__ "impl_bHYPRE_StructVector_Assemble"

int32_t
impl_bHYPRE_StructVector_Assemble(
  /*in*/ bHYPRE_StructVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector Hy;
   data = bHYPRE_StructVector__get_data( self );
   Hy = data -> vec;

   ierr = HYPRE_StructVectorAssemble( Hy );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.Assemble) */
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
#define __FUNC__ "impl_bHYPRE_StructVector_GetObject"

int32_t
impl_bHYPRE_StructVector_GetObject(
  /*in*/ bHYPRE_StructVector self, /*out*/ sidl_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.GetObject) */
  /* Insert the implementation of the GetObject method here... */

   bHYPRE_StructVector_addRef( self );
   *A = sidl_BaseInterface__cast( self );
   return( 0 );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.GetObject) */
}

/*
 * Method:  SetGrid[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetGrid"

int32_t
impl_bHYPRE_StructVector_SetGrid(
  /*in*/ bHYPRE_StructVector self, /*in*/ bHYPRE_StructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
   /* N.B. This is the only grid-setting function defined in the interface.
    So this is the only place to call HYPRE_StructVectorCreate, which requires a grid.
    Note that SetGrid cannot be called twice on the same vector.  The grid cannot be changed.

    SetCommunicator should have been called before the time SetGrid is called.
    Initialize, value-setting functions, and Assemble should be called afterwards.
   */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector Hy;
   HYPRE_StructGrid Hgrid;
   MPI_Comm comm;
   struct bHYPRE_StructGrid__data * gdata;

   data = bHYPRE_StructVector__get_data( self );
   Hy = data->vec;
   assert( Hy==NULL ); /* shouldn't have already been created */
   comm = data->comm;
   gdata = bHYPRE_StructGrid__get_data( grid );
   Hgrid = gdata->grid;

   ierr += HYPRE_StructVectorCreate( comm, Hgrid, &Hy );
   data->vec = Hy;

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetGrid) */
}

/*
 * Method:  SetNumGhost[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetNumGhost"

int32_t
impl_bHYPRE_StructVector_SetNumGhost(
  /*in*/ bHYPRE_StructVector self, /*in*/ struct sidl_int__array* num_ghost)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetNumGhost) */
  /* Insert the implementation of the SetNumGhost method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector Hy;
   data = bHYPRE_StructVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_StructVectorSetNumGhost( Hy, sidlArrayAddr1( num_ghost, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetNumGhost) */
}

/*
 * Method:  SetValue[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetValue"

int32_t
impl_bHYPRE_StructVector_SetValue(
  /*in*/ bHYPRE_StructVector self, /*in*/ struct sidl_int__array* grid_index,
    /*in*/ double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetValue) */
  /* Insert the implementation of the SetValue method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector Hy;
   data = bHYPRE_StructVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_StructVectorSetValues( Hy, sidlArrayAddr1( grid_index, 0 ), value );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetValue) */
}

/*
 * Method:  SetBoxValues[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructVector_SetBoxValues"

int32_t
impl_bHYPRE_StructVector_SetBoxValues(
  /*in*/ bHYPRE_StructVector self, /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructVector.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_StructVector__data * data;
   HYPRE_StructVector Hy;
   data = bHYPRE_StructVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_StructVectorSetBoxValues
      ( Hy, sidlArrayAddr1( ilower, 0 ), sidlArrayAddr1( iupper, 0 ),
        sidlArrayAddr1( values, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructVector.SetBoxValues) */
}
