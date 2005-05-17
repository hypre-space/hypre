/*
 * File:          bHYPRE_SStructParCSRVector_Impl.c
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.SStructParCSRVector
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
 * Symbol "bHYPRE.SStructParCSRVector" (version 1.0.0)
 * 
 * The SStructParCSR vector class.
 * 
 * Objects of this type can be cast to SStructBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

#include "bHYPRE_SStructParCSRVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "mpi.h"
#include "sstruct_mv.h"
#include "bHYPRE_SStructGrid_Impl.h"
#include "bHYPRE_IJParCSRVector_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector__ctor"

void
impl_bHYPRE_SStructParCSRVector__ctor(
  /*in*/ bHYPRE_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._ctor) */
  /* Insert the implementation of the constructor method here... */

  /* How to make a vector: first call the constructor.
     Then SetCommunicator, then SetGrid (which calls HYPRE_SStructParCSRVectorCreate),
     then Initialize, then SetValues (or SetBoxValues, etc.), then Assemble.
  */

   struct bHYPRE_SStructParCSRVector__data * data;
   data = hypre_CTAlloc( struct bHYPRE_SStructParCSRVector__data, 1 );
   data -> vec = NULL;
   data -> comm = MPI_COMM_NULL;
   bHYPRE_SStructParCSRVector__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector__dtor"

void
impl_bHYPRE_SStructParCSRVector__dtor(
  /*in*/ bHYPRE_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector vec;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   vec = data -> vec;
   ierr += HYPRE_SStructVectorDestroy( vec );
   assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector._dtor) */
}

/*
 * Set {\tt self} to 0.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Clear"

int32_t
impl_bHYPRE_SStructParCSRVector_Clear(
  /*in*/ bHYPRE_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Clear) */
  /* Insert the implementation of the Clear method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Clear) */
}

/*
 * Copy x into {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Copy"

int32_t
impl_bHYPRE_SStructParCSRVector_Copy(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Copy) */
  /* Insert the implementation of the Copy method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Copy) */
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
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Clone"

int32_t
impl_bHYPRE_SStructParCSRVector_Clone(
  /*in*/ bHYPRE_SStructParCSRVector self, /*out*/ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Clone) */
  /* Insert the implementation of the Clone method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Clone) */
}

/*
 * Scale {\tt self} by {\tt a}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Scale"

int32_t
impl_bHYPRE_SStructParCSRVector_Scale(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ double a)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Scale) */
  /* Insert the implementation of the Scale method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Scale) */
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Dot"

int32_t
impl_bHYPRE_SStructParCSRVector_Dot(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ bHYPRE_Vector x,
    /*out*/ double* d)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Dot) */
  /* Insert the implementation of the Dot method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Dot) */
}

/*
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Axpy"

int32_t
impl_bHYPRE_SStructParCSRVector_Axpy(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ double a,
    /*in*/ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Axpy) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetCommunicator"

int32_t
impl_bHYPRE_SStructParCSRVector_SetCommunicator(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   data -> comm = (MPI_Comm) mpi_comm;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Initialize"

int32_t
impl_bHYPRE_SStructParCSRVector_Initialize(
  /*in*/ bHYPRE_SStructParCSRVector self)
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

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Assemble"

int32_t
impl_bHYPRE_SStructParCSRVector_Assemble(
  /*in*/ bHYPRE_SStructParCSRVector self)
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
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_GetObject"

int32_t
impl_bHYPRE_SStructParCSRVector_GetObject(
  /*in*/ bHYPRE_SStructParCSRVector self, /*out*/ sidl_BaseInterface* A)
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

   px = bHYPRE_IJParCSRVector__create();
   p_data = bHYPRE_IJParCSRVector__get_data( px );
   ierr += HYPRE_SStructVectorGetObject( Hx, (void **) (&Hpx) );

   /* The purpose of the following block is to convert the HYPRE_ParVector Hpx
      to a HYPRE_IJVector Hijx ... */

   ilower = hypre_ParVectorFirstIndex( (hypre_ParVector *) Hpx );
   iupper = hypre_VectorSize( hypre_ParVectorLocalVector( (hypre_ParVector *) Hpx ) );

   ierr += HYPRE_IJVectorCreate( data->comm, ilower, iupper, &Hijx );
   ierr += HYPRE_IJVectorSetObjectType( Hijx, HYPRE_PARCSR );
   hypre_IJVectorObject( (hypre_IJVector *) Hijx ) = HYPRE_ParVectorCloneShallow( Hpx );
   ierr += HYPRE_IJVectorInitialize( Hijx );

   /* Now that we have made Hijx from Hpx, load it into p_data, the data of px.
      Then px is ready for output, as A.*/

   p_data -> ij_b = Hijx;
   p_data -> comm = data->comm;

   *A = sidl_BaseInterface__cast( px );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.GetObject) */
}

/*
 * Set the vector grid.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetGrid"

int32_t
impl_bHYPRE_SStructParCSRVector_SetGrid(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ bHYPRE_SStructGrid grid)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetGrid) */
  /* Insert the implementation of the SetGrid method here... */
   /* N.B. This is the only grid-setting function defined in the interface.
    So this is the only place to call HYPRE_SStructVectorCreate, which requires a grid.
    Note that SetGrid cannot be called twice on the same vector.  The grid cannot be changed.

    SetCommunicator should have been called before the time SetGrid is called.
    Initialize, value-setting functions, and Assemble should be called afterwards.
   */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   HYPRE_SStructGrid Hgrid;
   MPI_Comm comm;
   struct bHYPRE_SStructGrid__data * gdata;

   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data->vec;
   assert( Hy==NULL ); /* shouldn't have already been created */
   comm = data->comm;
   gdata = bHYPRE_SStructGrid__get_data( grid );
   Hgrid = gdata->grid;

   ierr += HYPRE_SStructVectorCreate( comm, Hgrid, &Hy );
   data->vec = Hy;

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetGrid) */
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
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetValues"

int32_t
impl_bHYPRE_SStructParCSRVector_SetValues(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetValues) */
  /* Insert the implementation of the SetValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_SStructVectorSetValues
      ( Hy, part, sidlArrayAddr1( index, 0 ), var,
        sidlArrayAddr1( values, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetValues) */
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
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetBoxValues"

int32_t
impl_bHYPRE_SStructParCSRVector_SetBoxValues(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t var,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_SStructVectorSetBoxValues
      ( Hy, part, sidlArrayAddr1( ilower, 0 ), sidlArrayAddr1( iupper, 0 ),
        var, sidlArrayAddr1( values, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetBoxValues) */
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
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_AddToValues"

int32_t
impl_bHYPRE_SStructParCSRVector_AddToValues(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_SStructVectorAddToValues
      ( Hy, part, sidlArrayAddr1( index, 0 ), var,
        sidlArrayAddr1( values, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.AddToValues) */
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
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_AddToBoxValues"

int32_t
impl_bHYPRE_SStructParCSRVector_AddToBoxValues(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t var,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.AddToBoxValues) */
  /* Insert the implementation of the AddToBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_SStructVectorAddToBoxValues
      ( Hy, part, sidlArrayAddr1( ilower, 0 ), sidlArrayAddr1( iupper, 0 ),
        var, sidlArrayAddr1( values, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.AddToBoxValues) */
}

/*
 * Gather vector data before calling {\tt GetValues}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Gather"

int32_t
impl_bHYPRE_SStructParCSRVector_Gather(
  /*in*/ bHYPRE_SStructParCSRVector self)
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

/*
 * Get vector coefficients index by index.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_GetValues"

int32_t
impl_bHYPRE_SStructParCSRVector_GetValues(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*out*/ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.GetValues) */
  /* Insert the implementation of the GetValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_SStructVectorGetValues
      ( Hy, part, sidlArrayAddr1( index, 0 ), var,
        value );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.GetValues) */
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
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_GetBoxValues"

int32_t
impl_bHYPRE_SStructParCSRVector_GetBoxValues(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t var,
    /*inout*/ struct sidl_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.GetBoxValues) */
  /* Insert the implementation of the GetBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRVector__data * data;
   HYPRE_SStructVector Hy;
   data = bHYPRE_SStructParCSRVector__get_data( self );
   Hy = data -> vec;

   ierr += HYPRE_SStructVectorGetBoxValues
      ( Hy, part, sidlArrayAddr1( ilower, 0 ), sidlArrayAddr1( iupper, 0 ),
        var, sidlArrayAddr1( *values, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.GetBoxValues) */
}

/*
 * Set the vector to be complex.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_SetComplex"

int32_t
impl_bHYPRE_SStructParCSRVector_SetComplex(
  /*in*/ bHYPRE_SStructParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.SetComplex) */
  /* Insert the implementation of the SetComplex method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.SetComplex) */
}

/*
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRVector_Print"

int32_t
impl_bHYPRE_SStructParCSRVector_Print(
  /*in*/ bHYPRE_SStructParCSRVector self, /*in*/ const char* filename,
    /*in*/ int32_t all)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRVector.Print) */
  /* Insert the implementation of the Print method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRVector.Print) */
}
