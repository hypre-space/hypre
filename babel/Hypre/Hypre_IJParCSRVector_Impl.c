/*
 * File:          Hypre_IJParCSRVector_Impl.c
 * Symbol:        Hypre.IJParCSRVector-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.IJParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 825
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.IJParCSRVector" (version 0.1.7)
 * 
 * The IJParCSR vector class.
 * 
 * Objects of this type can be cast to IJBuildVector or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

#include "Hypre_IJParCSRVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "parcsr_mv.h"
#include "Hypre_IJBuildVector.h"
/* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector__ctor"

void
impl_Hypre_IJParCSRVector__ctor(
  Hypre_IJParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct Hypre_IJParCSRVector__data * data;
   data = hypre_CTAlloc( struct Hypre_IJParCSRVector__data, 1 );
   data -> comm = (MPI_Comm)NULL;
   data -> ij_b = NULL;
   Hypre_IJParCSRVector__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector__dtor"

void
impl_Hypre_IJParCSRVector__dtor(
  Hypre_IJParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct Hypre_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;
   ierr = HYPRE_IJVectorDestroy( ij_b );
   assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_SetCommunicator"

int32_t
impl_Hypre_IJParCSRVector_SetCommunicator(
  Hypre_IJParCSRVector self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   int ierr = 0;
   struct Hypre_IJParCSRVector__data * data;
   data = Hypre_IJParCSRVector__get_data( self );
   data -> comm = (MPI_Comm) mpi_comm;
   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_Initialize"

int32_t
impl_Hypre_IJParCSRVector_Initialize(
  Hypre_IJParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */

   int ierr = 0;
   struct Hypre_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;
   ierr = HYPRE_IJVectorInitialize( ij_b );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.Initialize) */
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
#define __FUNC__ "impl_Hypre_IJParCSRVector_Assemble"

int32_t
impl_Hypre_IJParCSRVector_Assemble(
  Hypre_IJParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */

   int ierr = 0;
   struct Hypre_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;

   ierr = HYPRE_IJVectorAssemble( ij_b );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.Assemble) */
}

/*
 * The problem definition interface is a {\it builder} that
 * creates an object that contains the problem definition
 * information, e.g. a matrix. To perform subsequent operations
 * with that object, it must be returned from the problem
 * definition object. {\tt GetObject} performs this function.
 * At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface.
 * QueryInterface or Cast must be used on the returned object to
 * convert it into a known type.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_GetObject"

int32_t
impl_Hypre_IJParCSRVector_GetObject(
  Hypre_IJParCSRVector self, SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.GetObject) */
  /* Insert the implementation of the GetObject method here... */

   Hypre_IJParCSRVector_addRef( self );
   *A = SIDL_BaseInterface__cast( self );
   return( 0 );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.GetObject) */
}

/*
 * Set the local range for a vector object.  Each process owns
 * some unique consecutive range of vector unknowns, indicated
 * by the global indices {\tt jlower} and {\tt jupper}.  The
 * data is required to be such that the value of {\tt jlower} on
 * any process $p$ be exactly one more than the value of {\tt
 * jupper} on process $p-1$.  Note that the first index of the
 * global vector may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * Collective.
 * 
 * RDF: Changed name from 'Create' (x)
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_SetLocalRange"

int32_t
impl_Hypre_IJParCSRVector_SetLocalRange(
  Hypre_IJParCSRVector self, int32_t jlower, int32_t jupper)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.SetLocalRange) */
  /* Insert the implementation of the SetLocalRange method here... */

   int ierr = 0;
   struct Hypre_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;
   /* SetCommunicator should be called before Create */
   assert( data->comm != (MPI_Comm)NULL );

   ierr = HYPRE_IJVectorCreate( data->comm, jlower, jupper, &ij_b );
   ierr += HYPRE_IJVectorSetObjectType( ij_b, HYPRE_PARCSR );
   data -> ij_b = ij_b;

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.SetLocalRange) */
}

/*
 * Sets values in vector.  The arrays {\tt values} and {\tt
 * indices} are of dimension {\tt nvalues} and contain the
 * vector values to be set and the corresponding global vector
 * indices, respectively.  Erases any previous values at the
 * specified locations and replaces them with new ones.
 * 
 * Not collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_SetValues"

int32_t
impl_Hypre_IJParCSRVector_SetValues(
  Hypre_IJParCSRVector self, int32_t nvalues, struct SIDL_int__array* indices,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.SetValues) */
  /* Insert the implementation of the SetValues method here... */

   int ierr = 0;
   struct Hypre_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;

   ierr = HYPRE_IJVectorSetValues( ij_b, nvalues,
                                   SIDLArrayAddr1( indices, 0 ),
                                   SIDLArrayAddr1( values, 0 ) );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.SetValues) */
}

/*
 * Adds to values in vector.  Usage details are analogous to
 * {\tt SetValues}.
 * 
 * Not collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_AddToValues"

int32_t
impl_Hypre_IJParCSRVector_AddToValues(
  Hypre_IJParCSRVector self, int32_t nvalues, struct SIDL_int__array* indices,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */

   int ierr = 0;
   struct Hypre_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_IJParCSRVector__get_data( self );
   ij_b = data->ij_b;

   ierr = HYPRE_IJVectorAddToValues( ij_b, nvalues,
                                     SIDLArrayAddr1( indices, 0 ),
                                     SIDLArrayAddr1( values, 0 ) );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.AddToValues) */
}

/*
 * Returns range of the part of the vector owned by this
 * processor.
 * 
 * RDF: New (x)
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_GetLocalRange"

int32_t
impl_Hypre_IJParCSRVector_GetLocalRange(
  Hypre_IJParCSRVector self, int32_t* jlower, int32_t* jupper)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.GetLocalRange) */
  /* Insert the implementation of the GetLocalRange method here... */

   int ierr = 0;
   struct Hypre_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;

   ierr = HYPRE_IJVectorGetLocalRange( ij_b, jlower, jupper );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.GetLocalRange) */
}

/*
 * Gets values in vector.  Usage details are analogous to {\tt
 * SetValues}.
 * 
 * Not collective.
 * 
 * RDF: New (x)
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_GetValues"

int32_t
impl_Hypre_IJParCSRVector_GetValues(
  Hypre_IJParCSRVector self, int32_t nvalues, struct SIDL_int__array* indices,
    struct SIDL_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.GetValues) */
  /* Insert the implementation of the GetValues method here... */

   int ierr = 0;
   struct Hypre_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;

   ierr = HYPRE_IJVectorGetValues( ij_b, nvalues,
                                   SIDLArrayAddr1( indices, 0 ),
                                   SIDLArrayAddr1( *values, 0 ) );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.GetValues) */
}

/*
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_Print"

int32_t
impl_Hypre_IJParCSRVector_Print(
  Hypre_IJParCSRVector self, const char* filename)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.Print) */
  /* Insert the implementation of the Print method here... */

   int ierr = 0;
   struct Hypre_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_IJParCSRVector__get_data( self );
   ij_b = data->ij_b;

   ierr = HYPRE_IJVectorPrint( ij_b, filename );

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.Print) */
}

/*
 * Read the vector from file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_Read"

int32_t
impl_Hypre_IJParCSRVector_Read(
  Hypre_IJParCSRVector self, const char* filename, void* comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.Read) */
  /* Insert the implementation of the Read method here... */

   int ierr = 0;
   struct Hypre_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_IJParCSRVector__get_data( self );
   ij_b = data->ij_b;

   /* HYPRE_IJVectorRead will make a new one */
   ierr = HYPRE_IJVectorDestroy( ij_b );

   ierr = HYPRE_IJVectorRead( filename, data->comm,
                              HYPRE_PARCSR, &ij_b );
   data->ij_b = ij_b;
   Hypre_IJParCSRVector__set_data( self, data );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.Read) */
}

/*
 * Set {\tt self} to 0.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_Clear"

int32_t
impl_Hypre_IJParCSRVector_Clear(
  Hypre_IJParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.Clear) */
  /* Insert the implementation of the Clear method here... */

   int ierr = 0;
   void * object;
   struct Hypre_IJParCSRVector__data * data;
   HYPRE_ParVector xx;
   HYPRE_IJVector ij_x;
   data = Hypre_IJParCSRVector__get_data( self );
   ij_x = data -> ij_b;

   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;
   ierr += HYPRE_ParVectorSetConstantValues( xx, 0 );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.Clear) */
}

/*
 * Copy x into {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_Copy"

int32_t
impl_Hypre_IJParCSRVector_Copy(
  Hypre_IJParCSRVector self, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.Copy) */
  /* Insert the implementation of the Copy method here... */

   /* Copy the contents of x onto self.  This is a deep copy,
    * ultimately done by hypre_SeqVectorCopy.  */
   int ierr = 0;
   int type[1]; /* type[0] produces silly error messages on Sun */
   void * objectx, * objecty;
   struct Hypre_IJParCSRVector__data * data_y, * data_x;
   HYPRE_IJVector ij_y, ij_x;
   Hypre_IJParCSRVector HypreP_x;
   HYPRE_ParVector yy, xx;
   
   /* A Hypre_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( Hypre_Vector_queryInt(x, "Hypre.IJParCSRVector" ) )
   {
      HypreP_x = Hypre_IJParCSRVector__cast( x );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
   }

   data_y = Hypre_IJParCSRVector__get_data( self );
   data_x = Hypre_IJParCSRVector__get_data( HypreP_x );

   data_y->comm = data_x->comm;

   ij_x = data_x -> ij_b;
   ij_y = data_y -> ij_b;

   ierr += HYPRE_IJVectorGetObjectType( ij_y, type );
   /* ... don't know how to deal with other types */
   assert( *type == HYPRE_PARCSR );
   ierr += HYPRE_IJVectorGetObject( ij_y, &objecty );
   yy = (HYPRE_ParVector) objecty;

   ierr += HYPRE_IJVectorGetObjectType( ij_x, type );
   /* ... don't know how to deal with other types */
   assert( *type == HYPRE_PARCSR );
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   ierr += HYPRE_ParVectorCopy( xx, yy );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.Copy) */
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
#define __FUNC__ "impl_Hypre_IJParCSRVector_Clone"

int32_t
impl_Hypre_IJParCSRVector_Clone(
  Hypre_IJParCSRVector self, Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.Clone) */
  /* Insert the implementation of the Clone method here... */

   int ierr = 0;
   int type[1];  /* type[0] produces silly error messages on Sun */
   int * partitioning, jlower, jupper, my_id;
   void * objectx, * objecty;
   struct Hypre_IJParCSRVector__data * data_y, * data_x;
   HYPRE_IJVector ij_y, ij_x;
   Hypre_IJBuildVector Hypre_ij_x;
   Hypre_IJParCSRVector HypreP_x;
   HYPRE_ParVector yy, xx;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_id );

   HypreP_x = Hypre_IJParCSRVector__create();
   Hypre_ij_x = Hypre_IJBuildVector__cast( HypreP_x );

   data_y = Hypre_IJParCSRVector__get_data( self );
   data_x = Hypre_IJParCSRVector__get_data( HypreP_x );

   data_x->comm = data_y->comm;

   ij_y = data_y -> ij_b;
   ierr = HYPRE_IJVectorGetLocalRange( ij_y, &jlower, &jupper );

   ij_x = data_x->ij_b;
   ierr += HYPRE_IJVectorCreate( data_x->comm, jlower, jupper, &ij_x );
   ierr += HYPRE_IJVectorSetObjectType( ij_x, HYPRE_PARCSR );
   ierr += HYPRE_IJVectorInitialize( ij_x );
   data_x->ij_b = ij_x;

   ierr += HYPRE_IJVectorGetObjectType( ij_y, type );
   /* ... don't know how to deal with other types */
   assert( *type == HYPRE_PARCSR );
   ierr += HYPRE_IJVectorGetObject( ij_y, &objecty );
   yy = (HYPRE_ParVector) objecty;

   ierr += HYPRE_IJVectorGetObjectType( ij_x, type );
   /* ... don't know how to deal with other types */
   assert( *type == HYPRE_PARCSR );
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   /* Copy data in y to x... */
   HYPRE_ParVectorCopy( yy, xx );

   ierr += Hypre_IJBuildVector_Initialize( Hypre_ij_x );

   *x = Hypre_Vector__cast( Hypre_ij_x );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.Clone) */
}

/*
 * Scale {\self} by {\tt a}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_Scale"

int32_t
impl_Hypre_IJParCSRVector_Scale(
  Hypre_IJParCSRVector self, double a)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.Scale) */
  /* Insert the implementation of the Scale method here... */

   int ierr = 0;
   void * object;
   struct Hypre_IJParCSRVector__data * data;
   HYPRE_ParVector xx;
   HYPRE_IJVector ij_x;
   data = Hypre_IJParCSRVector__get_data( self );
   ij_x = data -> ij_b;

   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;
   ierr += HYPRE_ParVectorScale( a, xx );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.Scale) */
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_Dot"

int32_t
impl_Hypre_IJParCSRVector_Dot(
  Hypre_IJParCSRVector self, Hypre_Vector x, double* d)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.Dot) */
  /* Insert the implementation of the Dot method here... */

   int ierr = 0;
   void * object;
   struct Hypre_IJParCSRVector__data * data;
   Hypre_IJParCSRVector HypreP_x;
   HYPRE_ParVector xx, yy;
   HYPRE_IJVector ij_x, ij_y;

   /* A Hypre_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( Hypre_Vector_queryInt(x, "Hypre.IJParCSRVector" ) )
   {
      HypreP_x = Hypre_IJParCSRVector__cast( x );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
   }

   data = Hypre_IJParCSRVector__get_data( self );
   ij_y = data -> ij_b;
   data = Hypre_IJParCSRVector__get_data( HypreP_x );
   Hypre_IJParCSRVector_deleteRef( HypreP_x );
   ij_x = data -> ij_b;

   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;
   ierr += HYPRE_IJVectorGetObject( ij_y, &object );
   yy = (HYPRE_ParVector) object;

   ierr += HYPRE_ParVectorInnerProd( xx, yy, d );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.Dot) */
}

/*
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRVector_Axpy"

int32_t
impl_Hypre_IJParCSRVector_Axpy(
  Hypre_IJParCSRVector self, double a, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */

   int ierr = 0;
   int type[1];
   void * object;
   struct Hypre_IJParCSRVector__data * data, * data_x;
   Hypre_IJParCSRVector HypreP_x;
   HYPRE_IJVector ij_y, ij_x;
   HYPRE_ParVector yy, xx;
   data = Hypre_IJParCSRVector__get_data( self );
   ij_y = data -> ij_b;

   ierr += HYPRE_IJVectorGetObjectType( ij_y, type );
   /* ... don't know how to deal with other types */
   assert( *type == HYPRE_PARCSR );
   ierr += HYPRE_IJVectorGetObject( ij_y, &object );
   yy = (HYPRE_ParVector) object;

   /* A Hypre_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( Hypre_Vector_queryInt(x, "Hypre.IJParCSRVector" ) )
   {
      HypreP_x = Hypre_IJParCSRVector__cast( x );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
   }

   data_x = Hypre_IJParCSRVector__get_data( HypreP_x );
   Hypre_IJParCSRVector_deleteRef( HypreP_x );
   ij_x = data_x->ij_b;
   ierr += HYPRE_IJVectorGetObjectType( ij_x, type );
   /* ... don't know how to deal with other types */
   assert( *type == HYPRE_PARCSR );
   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;

   ierr += hypre_ParVectorAxpy( a, (hypre_ParVector *) xx,
                                (hypre_ParVector *) yy );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRVector.Axpy) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
/* ================== END UNREFERENCED METHOD(S) ================= */
