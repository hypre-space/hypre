/*
 * File:          Hypre_ParCSRVector_Impl.c
 * Symbol:        Hypre.ParCSRVector-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:19 PST
 * Description:   Server-side implementation for Hypre.ParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.ParCSRVector" (version 0.1.5)
 *
 * This class implements build and operator interfaces for vectors.
 * Thus its <code>GetConstructedObject</code> method returns itself.
 */

#include "Hypre_ParCSRVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector._includes) */
/* Put additional includes or other arbitrary code here... */
/*  >>> TO DO: anything to do with components */
#include <assert.h>
#include "parcsr_mv.h"
#include "Hypre_IJBuildVector.h"
#include "IJ_mv.h"
/* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector__ctor"

void
impl_Hypre_ParCSRVector__ctor(
  Hypre_ParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector._ctor) */
  /* Insert the implementation of the constructor method here... */
   struct Hypre_ParCSRVector__data * data;
   data = hypre_CTAlloc( struct Hypre_ParCSRVector__data, 1 );
   data -> comm = NULL;
   data -> ij_b = NULL;
   Hypre_ParCSRVector__set_data( self, data );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector__dtor"

void
impl_Hypre_ParCSRVector__dtor(
  Hypre_ParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector._dtor) */
  /* Insert the implementation of the destructor method here... */
   int ierr = 0;
   struct Hypre_ParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_ParCSRVector__get_data( self );
   ij_b = data -> ij_b;
   ierr = HYPRE_IJVectorDestroy( ij_b );
   assert( ierr==0 );
   hypre_TFree( data );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector._dtor) */
}

/*
 * Method:  AddToLocalComponentsInBlock
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_AddToLocalComponentsInBlock"

int32_t
impl_Hypre_ParCSRVector_AddToLocalComponentsInBlock(
  Hypre_ParCSRVector self,
  int32_t glob_vec_index_start,
  int32_t glob_vec_index_stop,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE 
    splicer.begin(Hypre.ParCSRVector.AddToLocalComponentsInBlock) */
  /* Insert the implementation of the AddToLocalComponentsInBlock method 
     here... */
/* >>> not implemented <<< */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.AddToLocalComponentsInBlock) 
    */
}

/*
 * Adds to values in vector.  Usage details are analogous to
 * \Ref{SetValues}.
 * 
 * Not collective.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_AddToValues"

int32_t
impl_Hypre_ParCSRVector_AddToValues(
  Hypre_ParCSRVector self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */
   int ierr = 0;
   struct Hypre_ParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_ParCSRVector__get_data( self );
   ij_b = data->ij_b;

   ierr = HYPRE_IJVectorAddToValues( ij_b, nvalues,
                                     SIDLArrayAddr1( indices, 0 ),
                                     SIDLArrayAddr1( values, 0 ) );
   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.AddToValues) */
}

/*
 * Method:  AddtoLocalComponents
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_AddtoLocalComponents"

int32_t
impl_Hypre_ParCSRVector_AddtoLocalComponents(
  Hypre_ParCSRVector self,
  int32_t num_values,
  struct SIDL_int__array* glob_vec_indices,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.AddtoLocalComponents) */
  /* Insert the implementation of the AddtoLocalComponents method here... */
/* >>> not implemented <<< */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.AddtoLocalComponents) */
}

/*
 * Finalize the construction of an object before using, either for
 * the first time or on subsequent uses. "Initialize" and "Assemble"
 * always appear in a matched set, with Initialize preceding Assemble. Values
 * can only be set in between a call to Initialize and Assemble.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Assemble"

int32_t
impl_Hypre_ParCSRVector_Assemble(
  Hypre_ParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */
   int ierr = 0;
   struct Hypre_ParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_ParCSRVector__get_data( self );
   ij_b = data -> ij_b;

   printf( "impl_Hypre_ParCSRVectorAssemble ij_b=%i\n", ij_b );
   ierr = HYPRE_IJVectorAssemble( ij_b );
   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Assemble) */
}

/*
 * y <- a*x + y
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Axpy"

int32_t
impl_Hypre_ParCSRVector_Axpy(
  Hypre_ParCSRVector self,
  double a,
  Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */
   int ierr = 0;
   int * type;
   void * object;
   struct Hypre_ParCSRVector__data * data, * data_x;
   Hypre_ParCSRVector HypreP_x;
   HYPRE_IJVector ij_y, ij_x;
   HYPRE_ParVector yy, xx;
   data = Hypre_ParCSRVector__get_data( self );
   ij_y = data -> ij_b;

   ierr += HYPRE_IJVectorGetObjectType( ij_y, type );
   assert( *type == HYPRE_PARCSR );  /* ... don't know how to deal with other types */
   ierr += HYPRE_IJVectorGetObject( ij_y, &object );
   yy = (HYPRE_ParVector) object;

   /*  A Hypre_Vector is just an interface, we have no knowledge of its contents.
       Check whether it's something we know how to handle.  If not, die. */
   HypreP_x = Hypre_Vector__cast2
      ( Hypre_Vector_queryInterface( x, "Hypre.ParCSRVector"),
        "Hypre.ParCSRVector" );
   assert( HypreP_x!=NULL );
   /* ... Without the cast, we could also have done (I think)
      assert( Hypre_Vector_isInstanceOf( x, "ParCSRVector" ) );
   */
   data_x = Hypre_ParCSRVector__get_data( HypreP_x );
   ij_x = data_x->ij_b;
   ierr += HYPRE_IJVectorGetObjectType( ij_x, type );
   assert( *type == HYPRE_PARCSR );  /* ... don't know how to deal with other types */
   /* ... don't know how to deal with other types */
   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;

   printf( "impl_Hypre_ParCSRVector_Axpy\n" );

   ierr += hypre_ParVectorAxpy( a, (hypre_ParVector *) xx, (hypre_ParVector *) yy );

   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Axpy) */
}

/*
 * y <- 0 (where y=self)
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Clear"

int32_t
impl_Hypre_ParCSRVector_Clear(
  Hypre_ParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Clear) */
  /* Insert the implementation of the Clear method here... */
   int ierr = 0;
   void * object;
   struct Hypre_ParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   HYPRE_ParVector xx;
   HYPRE_IJVector ij_x;
   data = Hypre_ParCSRVector__get_data( self );
   ij_x = data -> ij_b;

   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;
   ierr += HYPRE_ParVectorSetConstantValues( xx, 0 );
   printf( "impl_Hypre_ParCSRVector_Clear\n" );

   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Clear) */
}

/*
 * create an x compatible with y
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Clone"

int32_t
impl_Hypre_ParCSRVector_Clone(
  Hypre_ParCSRVector self,
  Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Clone) */
  /* Insert the implementation of the Clone method here... */
   /* Set x to a clone of self. */
   int ierr = 0;
   int * type, * partitioning, jlower, jupper, my_id;
   void * object;
   struct Hypre_ParCSRVector__data * data_y, * data_x;
   HYPRE_IJVector ij_y, ij_x;
   hypre_IJVector * hypre_ij_y;
   Hypre_IJBuildVector Hypre_ij_x;
   Hypre_ParCSRVector HypreP_x;
   HYPRE_ParVector yy, xx;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_id );

   HypreP_x = Hypre_ParCSRVector__create();  /* I assume this does an addReference - not checked */
   Hypre_ij_x = (Hypre_IJBuildVector) Hypre_ParCSRVector__cast2( HypreP_x, "Hypre.IJBuildVector" );
   Hypre_IJBuildVector_addReference( Hypre_ij_x );

   data_y = Hypre_ParCSRVector__get_data( self );
   data_x = Hypre_ParCSRVector__get_data( HypreP_x );

   data_x->comm = data_y->comm;

   ij_y = data_y -> ij_b;
   hypre_ij_y = (hypre_IJVector *) ij_y;
   partitioning = hypre_IJVectorPartitioning( hypre_ij_y );
   jlower = partitioning[ my_id ];
   jupper = partitioning[ my_id+1 ];

   ij_x = data_x->ij_b;
   ierr = HYPRE_IJVectorCreate( *(data_x->comm), jlower, jupper, &ij_x );
   ierr += HYPRE_IJVectorSetObjectType( ij_x, HYPRE_PARCSR );
   data_x->ij_b = ij_x;

   /* Copy data in y to x... */
   HYPRE_ParVectorCopy( yy, xx );

   ierr += Hypre_IJBuildVector_Initialize( Hypre_ij_x );

   *x = (Hypre_Vector) Hypre_IJBuildVector__cast2( Hypre_ij_x, "Hypre.Vector" );
   Hypre_IJBuildVector_deleteReference( Hypre_ij_x );
   /* We are returning x with a positive reference count in the form of
      HypreP_x, a Hypre_ParCSRVector */

   return( ierr );


  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Clone) */
}

/*
 * y <- x 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Copy"

int32_t
impl_Hypre_ParCSRVector_Copy(
  Hypre_ParCSRVector self,
  Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Copy) */
  /* Insert the implementation of the Copy method here... */
   /* Copy the contents of x onto self.
      This is a deep copy, ultimately done by hypre_SeqVectorCopy.
   */
   int ierr = 0;
   int * type;
   void * object;
   struct Hypre_ParCSRVector__data * data_y, * data_x;
   HYPRE_IJVector ij_y, ij_x;
   Hypre_ParCSRVector HypreP_x;
   HYPRE_ParVector yy, xx;
   
   /*  A Hypre_Vector is just an interface, we have no knowledge of its contents.
       Check whether it's something we know how to handle.  If not, die. */
   HypreP_x = Hypre_Vector__cast2
      ( Hypre_Vector_queryInterface( x, "Hypre.ParCSRVector"),
        "Hypre.ParCSRVector" );
   assert( HypreP_x!=NULL );
   /* ... Without the cast, we could also have done (I think)
      assert( Hypre_Vector_isInstanceOf( x, "ParCSRVector" ) );
   */

   data_y = Hypre_ParCSRVector__get_data( self );
   data_x = Hypre_ParCSRVector__get_data( HypreP_x );

   data_y->comm = data_x->comm;

   ij_x = data_x -> ij_b;
   ij_y = data_y -> ij_b;

   ierr += HYPRE_IJVectorGetObjectType( ij_y, type );
   assert( *type == HYPRE_PARCSR );  /* ... don't know how to deal with other types */
   ierr += HYPRE_IJVectorGetObject( ij_y, &object );
   yy = (HYPRE_ParVector) object;

   ij_x = data_x->ij_b;
   ierr += HYPRE_IJVectorGetObjectType( ij_x, type );
   assert( *type == HYPRE_PARCSR );  /* ... don't know how to deal with other types */
   /* ... don't know how to deal with other types */
   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;

   ierr += HYPRE_ParVectorCopy( xx, yy );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Copy) */
}

/*
 * Create a vector object.  Each process owns some unique consecutive
 * range of vector unknowns, indicated by the global indices {\tt
 * jlower} and {\tt jupper}.  The data is required to be such that the
 * value of {\tt jlower} on any process $p$ be exactly one more than
 * the value of {\tt jupper} on process $p-1$.  Note that the first
 * index of the global vector may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * Collective.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Create"

int32_t
impl_Hypre_ParCSRVector_Create(
  Hypre_ParCSRVector self,
  void* comm,
  int32_t jlower,
  int32_t jupper)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Create) */
  /* Insert the implementation of the Create method here... */
   int ierr = 0;
   struct Hypre_ParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_ParCSRVector__get_data( self );
   ij_b = data -> ij_b;
   assert( data->comm != NULL ); /* SetCommunicator should be called before Create */

   ierr = HYPRE_IJVectorCreate( *(data->comm), jlower, jupper, &ij_b );
   ierr += HYPRE_IJVectorSetObjectType( ij_b, HYPRE_PARCSR );
   data -> ij_b = ij_b;

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Create) */
}

/*
 * d <- (y,x)
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Dot"

int32_t
impl_Hypre_ParCSRVector_Dot(
  Hypre_ParCSRVector self,
  Hypre_Vector x,
  double* d)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Dot) */
  /* Insert the implementation of the Dot method here... */
   int ierr = 0;
   void * object;
   struct Hypre_ParCSRVector__data * data;
   Hypre_ParCSRVector HypreP_x;
   HYPRE_ParVector xx, yy;
   HYPRE_IJVector ij_x, ij_y;

   /*  A Hypre_Vector is just an interface, we have no knowledge of its contents.
       Check whether it's something we know how to handle.  If not, die. */
   HypreP_x = Hypre_Vector__cast2
      ( Hypre_Vector_queryInterface( x, "Hypre.ParCSRVector"),
        "Hypre.ParCSRVector" );
   assert( HypreP_x!=NULL );

   data = Hypre_ParCSRVector__get_data( self );
   ij_y = data -> ij_b;
   data = Hypre_ParCSRVector__get_data( HypreP_x );
   ij_x = data -> ij_b;

   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;
   ierr += HYPRE_IJVectorGetObject( ij_y, &object );
   yy = (HYPRE_ParVector) object;

   ierr += HYPRE_ParVectorInnerProd( xx, yy, d );
   printf( "impl_Hypre_ParCSRVector_Dot\n" );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Dot) */
}

/*
 * The problem definition interface is a "builder" that creates an object
 * that contains the problem definition information, e.g. a matrix. To
 * perform subsequent operations with that object, it must be returned from
 * the problem definition object. "GetObject" performs this function.
 * <note>At compile time, the type of the returned object is unknown.
 * Thus, the returned type is a SIDL.BaseInterface. QueryInterface or Cast must
 * be used on the returned object to convert it into a known type.</note>
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_GetObject"

int32_t
impl_Hypre_ParCSRVector_GetObject(
  Hypre_ParCSRVector self,
  SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.GetObject) */
  /* Insert the implementation of the GetObject method here... */
   Hypre_ParCSRMatrix_addReference( self );
   *A = (SIDL_BaseInterface) SIDL_BaseInterface__cast( self );
   return( 0 );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.GetObject) */
}

/*
 * Method:  GetRow
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_GetRow"

int32_t
impl_Hypre_ParCSRVector_GetRow(
  Hypre_ParCSRVector self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.GetRow) */
  /* Insert the implementation of the GetRow method here... */
   /* The standard vector is a column vector, so GetRow simply returns one value.
      Thus we ignore the size and col_ind argumens, and simply set
      values[0][0] = vector[row]
   */
   int ierr = 0;
   struct Hypre_ParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_ParCSRVector__get_data( self );
   ij_b = data -> ij_b;

   ierr += HYPRE_IJVectorGetValues( ij_b, 1, &row, SIDLArrayAddr1( values[0], 0 ) );
   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.GetRow) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Initialize"

int32_t
impl_Hypre_ParCSRVector_Initialize(
  Hypre_ParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */
   int ierr = 0;
   struct Hypre_ParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_ParCSRVector__get_data( self );
   ij_b = data -> ij_b;
   ierr = HYPRE_IJVectorInitialize( ij_b );
   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Initialize) */
}

/*
 * Print the vector to file.  This is mainly for debugging purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Print"

int32_t
impl_Hypre_ParCSRVector_Print(
  Hypre_ParCSRVector self,
  const char* filename)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Print) */
  /* Insert the implementation of the Print method here... */
   int ierr = 0;
   struct Hypre_ParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_ParCSRVector__get_data( self );
   ij_b = data->ij_b;

   printf("impl_Hypre_ParCSRVector_Print\n");
   ierr = HYPRE_IJVectorPrint( ij_b, filename );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Print) */
}

/*
 * Read the vector from file.  This is mainly for debugging purposes.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Read"

int32_t
impl_Hypre_ParCSRVector_Read(
  Hypre_ParCSRVector self,
  const char* filename,
  void* comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Read) */
  /* Insert the implementation of the Read method here... */
   int ierr = 0;
   struct Hypre_ParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_ParCSRVector__get_data( self );
   ij_b = data->ij_b;

   ierr = HYPRE_IJVectorRead( filename, *(data->comm),
                              HYPRE_PARCSR, &ij_b );
   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Read) */
}

/*
 * y <- a*y 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_Scale"

int32_t
impl_Hypre_ParCSRVector_Scale(
  Hypre_ParCSRVector self,
  double a)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.Scale) */
  /* Insert the implementation of the Scale method here... */
   int ierr = 0;
   void * object;
   struct Hypre_ParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   HYPRE_ParVector xx;
   HYPRE_IJVector ij_x;
   data = Hypre_ParCSRVector__get_data( self );
   ij_x = data -> ij_b;

   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;
   ierr += HYPRE_ParVectorScale( a, xx );
   printf( "impl_Hypre_ParCSRVector_Scale\n" );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.Scale) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_SetCommunicator"

int32_t
impl_Hypre_ParCSRVector_SetCommunicator(
  Hypre_ParCSRVector self,
  void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   int ierr = 0;
   struct Hypre_ParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_ParCSRVector__get_data( self );
   ij_b = data -> ij_b;
   data -> comm = (MPI_Comm *) mpi_comm;
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.SetCommunicator) */
}

/*
 * Method:  SetGlobalSize
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_SetGlobalSize"

int32_t
impl_Hypre_ParCSRVector_SetGlobalSize(
  Hypre_ParCSRVector self,
  int32_t n)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.SetGlobalSize) */
  /* Insert the implementation of the SetGlobalSize method here... */
   /* We have no use fot the global size, hence do nothing with it. */
   return 0;
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.SetGlobalSize) */
}

/*
 * Method:  SetLocalComponents
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_SetLocalComponents"

int32_t
impl_Hypre_ParCSRVector_SetLocalComponents(
  Hypre_ParCSRVector self,
  int32_t num_values,
  struct SIDL_int__array* glob_vec_indices,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.SetLocalComponents) */
  /* Insert the implementation of the SetLocalComponents method here... */
/* >>> not implemented <<< */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.SetLocalComponents) */
}

/*
 * Method:  SetLocalComponentsInBlock
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_SetLocalComponentsInBlock"

int32_t
impl_Hypre_ParCSRVector_SetLocalComponentsInBlock(
  Hypre_ParCSRVector self,
  int32_t glob_vec_index_start,
  int32_t glob_vec_index_stop,
  struct SIDL_int__array* value_indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.SetLocalComponentsInBlock) 
    */
  /* Insert the implementation of the SetLocalComponentsInBlock method here... 
    */
/* >>> not implemented <<< */
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.SetLocalComponentsInBlock) */
}

/*
 * Method:  SetPartitioning
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_SetPartitioning"

int32_t
impl_Hypre_ParCSRVector_SetPartitioning(
  Hypre_ParCSRVector self,
  struct SIDL_int__array* partitioning)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.SetPartitioning) */
  /* Insert the implementation of the SetPartitioning method here... */
   /* We have no need for partitioning in this form (it is provided by the row
      bound arguments of  calls of Hypre_IJBuildVector_Create).
      Hence nothing is done here. */
   return 0;
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.SetPartitioning) */
}

/*
 * Sets values in vector.  The arrays {\tt values} and {\tt indices}
 * are of dimension {\tt nvalues} and contain the vector values to be
 * set and the corresponding global vector indices, respectively.
 * Erases any previous values at the specified locations and replaces
 * them with new ones.
 * 
 * Not collective.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRVector_SetValues"

int32_t
impl_Hypre_ParCSRVector_SetValues(
  Hypre_ParCSRVector self,
  int32_t nvalues,
  struct SIDL_int__array* indices,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRVector.SetValues) */
  /* Insert the implementation of the SetValues method here... */
   int ierr = 0;
   struct Hypre_ParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = Hypre_ParCSRVector__get_data( self );
   ij_b = data -> ij_b;

   ierr = HYPRE_IJVectorSetValues( ij_b, nvalues,
                                   SIDLArrayAddr1( indices, 0 ),
                                   SIDLArrayAddr1( values, 0 ) );
   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRVector.SetValues) */
}
