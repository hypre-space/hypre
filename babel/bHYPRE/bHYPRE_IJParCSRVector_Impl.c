/*
 * File:          bHYPRE_IJParCSRVector_Impl.c
 * Symbol:        bHYPRE.IJParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for bHYPRE.IJParCSRVector
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.IJParCSRVector" (version 1.0.0)
 * 
 * The IJParCSR vector class.
 * 
 * Objects of this type can be cast to IJVectorView or Vector
 * objects using the {\tt \_\_cast} methods.
 * 
 */

#include "bHYPRE_IJParCSRVector_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "parcsr_mv.h"
#include "bHYPRE_IJVectorView.h"
#include "bHYPRE_MPICommunicator_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_IJParCSRVector__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector._load) */
  /* Insert-Code-Here {bHYPRE.IJParCSRVector._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_IJParCSRVector__ctor(
  /* in */ bHYPRE_IJParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector._ctor) */
  /* Insert the implementation of the constructor method here... */

   /* Note: User calls of__create are DEPRECATED.
      Use Create(), which also calls this function */

   struct bHYPRE_IJParCSRVector__data * data;
   data = hypre_CTAlloc( struct bHYPRE_IJParCSRVector__data, 1 );
   data -> comm = MPI_COMM_NULL;
   data -> ij_b = NULL;
   bHYPRE_IJParCSRVector__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_IJParCSRVector__dtor(
  /* in */ bHYPRE_IJParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;
   ierr = HYPRE_IJVectorDestroy( ij_b );
   hypre_assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector._dtor) */
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_IJParCSRVector
impl_bHYPRE_IJParCSRVector_Create(
  /* in */ bHYPRE_MPICommunicator mpi_comm,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.Create) */
  /* Insert-Code-Here {bHYPRE.IJParCSRVector.Create} (Create method) */

   int ierr = 0;
   HYPRE_IJVector dummy;
   HYPRE_IJVector * Hvec = &dummy;
   struct bHYPRE_IJParCSRVector__data * data;

   bHYPRE_IJParCSRVector vec = bHYPRE_IJParCSRVector__create();
   data = bHYPRE_IJParCSRVector__get_data( vec );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm,
   ierr += HYPRE_IJVectorCreate( data->comm, jlower, jupper, Hvec );
   hypre_assert( ierr == 0 );
   ierr += HYPRE_IJVectorSetObjectType( *Hvec, HYPRE_PARCSR );
   data -> ij_b = *Hvec;

   return vec;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.Create) */
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_SetCommunicator"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_SetCommunicator(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ bHYPRE_MPICommunicator mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   /* DEPRECATED  Use Create */

   int ierr = 0;
   struct bHYPRE_IJParCSRVector__data * data;
   data = bHYPRE_IJParCSRVector__get_data( self );
   data->comm = bHYPRE_MPICommunicator__get_data(mpi_comm)->mpi_comm;
   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.SetCommunicator) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_Initialize"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_Initialize(
  /* in */ bHYPRE_IJParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.Initialize) */
  /* Insert the implementation of the Initialize method here... */

   int ierr = 0;
   struct bHYPRE_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;
   ierr = HYPRE_IJVectorInitialize( ij_b );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.Initialize) */
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
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_Assemble"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_Assemble(
  /* in */ bHYPRE_IJParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.Assemble) */
  /* Insert the implementation of the Assemble method here... */

   int ierr = 0;
   struct bHYPRE_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;

   ierr = HYPRE_IJVectorAssemble( ij_b );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.Assemble) */
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
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_SetLocalRange"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_SetLocalRange(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t jlower,
  /* in */ int32_t jupper)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.SetLocalRange) */
  /* Insert the implementation of the SetLocalRange method here... */

   /* DEPRECATED ... use Create */

   int ierr = 0;
   struct bHYPRE_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;
   /* Create or SetCommunicator should have be called by now... */
   hypre_assert( data->comm != MPI_COMM_NULL );

   ierr = HYPRE_IJVectorCreate( data->comm, jlower, jupper, &ij_b );
   ierr += HYPRE_IJVectorSetObjectType( ij_b, HYPRE_PARCSR );
   data -> ij_b = ij_b;

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.SetLocalRange) */
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
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_SetValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_SetValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t nvalues,
  /* in */ int32_t* indices,
  /* in */ double* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.SetValues) */
  /* Insert the implementation of the SetValues method here... */

   int ierr = 0;
   struct bHYPRE_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;

   ierr = HYPRE_IJVectorSetValues( ij_b, nvalues,
                                   indices,
                                   values );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.SetValues) */
}

/*
 * Adds to values in vector.  Usage details are analogous to
 * {\tt SetValues}.
 * 
 * Not collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_AddToValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_AddToValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t nvalues,
  /* in */ int32_t* indices,
  /* in */ double* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */

   int ierr = 0;
   struct bHYPRE_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_b = data->ij_b;

   ierr = HYPRE_IJVectorAddToValues( ij_b, nvalues,
                                     indices,
                                     values );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.AddToValues) */
}

/*
 * Returns range of the part of the vector owned by this
 * processor.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_GetLocalRange"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_GetLocalRange(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ int32_t* jlower,
  /* out */ int32_t* jupper)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.GetLocalRange) */
  /* Insert the implementation of the GetLocalRange method here... */

   int ierr = 0;
   struct bHYPRE_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;

   ierr = HYPRE_IJVectorGetLocalRange( ij_b, jlower, jupper );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.GetLocalRange) */
}

/*
 * Gets values in vector.  Usage details are analogous to {\tt
 * SetValues}.
 * 
 * Not collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_GetValues"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_GetValues(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ int32_t nvalues,
  /* in */ int32_t* indices,
  /* inout */ double* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.GetValues) */
  /* Insert the implementation of the GetValues method here... */

   int ierr = 0;
   struct bHYPRE_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_b = data -> ij_b;

   ierr = HYPRE_IJVectorGetValues( ij_b, nvalues,
                                   indices,
                                   values );
   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.GetValues) */
}

/*
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_Print"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_Print(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ const char* filename)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.Print) */
  /* Insert the implementation of the Print method here... */

   int ierr = 0;
   struct bHYPRE_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_b = data->ij_b;

   ierr = HYPRE_IJVectorPrint( ij_b, filename );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.Print) */
}

/*
 * Read the vector from file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_Read"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_Read(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ const char* filename,
  /* in */ bHYPRE_MPICommunicator comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.Read) */
  /* Insert the implementation of the Read method here... */

   int ierr = 0;
   struct bHYPRE_IJParCSRVector__data * data;
   HYPRE_IJVector ij_b;
   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_b = data->ij_b;
   MPI_Comm mpicomm = bHYPRE_MPICommunicator__get_data(comm)->mpi_comm;

   /* HYPRE_IJVectorRead will make a new one */
   ierr = HYPRE_IJVectorDestroy( ij_b );

   ierr = HYPRE_IJVectorRead( filename, mpicomm,
                              HYPRE_PARCSR, &ij_b );
   data->ij_b = ij_b;
   bHYPRE_IJParCSRVector__set_data( self, data );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.Read) */
}

/*
 * Set {\tt self} to 0.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_Clear"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_Clear(
  /* in */ bHYPRE_IJParCSRVector self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.Clear) */
  /* Insert the implementation of the Clear method here... */

   int ierr = 0;
   void * object;
   struct bHYPRE_IJParCSRVector__data * data;
   HYPRE_ParVector xx;
   HYPRE_IJVector ij_x;
   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_x = data -> ij_b;

   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;
   ierr += HYPRE_ParVectorSetConstantValues( xx, 0 );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.Clear) */
}

/*
 * Copy x into {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_Copy"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_Copy(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.Copy) */
  /* Insert the implementation of the Copy method here... */

   /* Copy the contents of x onto self.  This is a deep copy,
    * ultimately done by hypre_SeqVectorCopy.  */
   int ierr = 0;
   int type[1]; /* type[0] produces silly error messages on Sun */
   void * objectx, * objecty;
   struct bHYPRE_IJParCSRVector__data * data_y, * data_x;
   HYPRE_IJVector ij_y, ij_x;
   bHYPRE_IJParCSRVector bHYPREP_x;
   HYPRE_ParVector yy, xx;
   
   /* A bHYPRE_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( bHYPRE_Vector_queryInt(x, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_x = bHYPRE_IJParCSRVector__cast( x );
      bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x ); /* extra ref from queryInt */
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }

   data_y = bHYPRE_IJParCSRVector__get_data( self );
   data_x = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x ); /* extra reference from queryInt */

   data_y->comm = data_x->comm;

   ij_x = data_x -> ij_b;
   ij_y = data_y -> ij_b;

   ierr += HYPRE_IJVectorGetObjectType( ij_y, type );
   /* ... don't know how to deal with other types */
   hypre_assert( *type == HYPRE_PARCSR );
   ierr += HYPRE_IJVectorGetObject( ij_y, &objecty );
   yy = (HYPRE_ParVector) objecty;

   ierr += HYPRE_IJVectorGetObjectType( ij_x, type );
   /* ... don't know how to deal with other types */
   hypre_assert( *type == HYPRE_PARCSR );
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   ierr += HYPRE_ParVectorCopy( xx, yy );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.Copy) */
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
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_Clone"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_Clone(
  /* in */ bHYPRE_IJParCSRVector self,
  /* out */ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.Clone) */
  /* Insert the implementation of the Clone method here... */

   int ierr = 0;
   int type[1];  /* type[0] produces silly error messages on Sun */
   int jlower, jupper, my_id;
   void * objectx, * objecty;
   struct bHYPRE_IJParCSRVector__data * data_y, * data_x;
   HYPRE_IJVector ij_y, ij_x;
   bHYPRE_IJVectorView bHYPRE_ij_x;
   bHYPRE_IJParCSRVector bHYPREP_x;
   HYPRE_ParVector yy, xx;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_id );

   bHYPREP_x = bHYPRE_IJParCSRVector__create();
   bHYPRE_ij_x = bHYPRE_IJVectorView__cast( bHYPREP_x );

   data_y = bHYPRE_IJParCSRVector__get_data( self );
   data_x = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );

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
   hypre_assert( *type == HYPRE_PARCSR );
   ierr += HYPRE_IJVectorGetObject( ij_y, &objecty );
   yy = (HYPRE_ParVector) objecty;

   ierr += HYPRE_IJVectorGetObjectType( ij_x, type );
   /* ... don't know how to deal with other types */
   hypre_assert( *type == HYPRE_PARCSR );
   ierr += HYPRE_IJVectorGetObject( ij_x, &objectx );
   xx = (HYPRE_ParVector) objectx;

   /* Copy data in y to x... */
   HYPRE_ParVectorCopy( yy, xx );

   ierr += bHYPRE_IJVectorView_Initialize( bHYPRE_ij_x );

   *x = bHYPRE_Vector__cast( bHYPRE_ij_x );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.Clone) */
}

/*
 * Scale {\tt self} by {\tt a}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_Scale"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_Scale(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ double a)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.Scale) */
  /* Insert the implementation of the Scale method here... */

   int ierr = 0;
   void * object;
   struct bHYPRE_IJParCSRVector__data * data;
   HYPRE_ParVector xx;
   HYPRE_IJVector ij_x;
   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_x = data -> ij_b;

   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;
   ierr += HYPRE_ParVectorScale( a, xx );

   return ierr;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.Scale) */
}

/*
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_Dot"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_Dot(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ bHYPRE_Vector x,
  /* out */ double* d)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.Dot) */
  /* Insert the implementation of the Dot method here... */

   int ierr = 0;
   void * object;
   struct bHYPRE_IJParCSRVector__data * data;
   bHYPRE_IJParCSRVector bHYPREP_x;
   HYPRE_ParVector xx, yy;
   HYPRE_IJVector ij_x, ij_y;

   /* A bHYPRE_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( bHYPRE_Vector_queryInt(x, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_x = bHYPRE_IJParCSRVector__cast( x );
      bHYPRE_StructVector_deleteRef( bHYPREP_x ); /* extra ref from queryInt */
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }

   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_y = data -> ij_b;
   data = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x );
   ij_x = data -> ij_b;

   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;
   ierr += HYPRE_IJVectorGetObject( ij_y, &object );
   yy = (HYPRE_ParVector) object;

   ierr += HYPRE_ParVectorInnerProd( xx, yy, d );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.Dot) */
}

/*
 * Add {\tt a}*{\tt x} to {\tt self}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRVector_Axpy"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_IJParCSRVector_Axpy(
  /* in */ bHYPRE_IJParCSRVector self,
  /* in */ double a,
  /* in */ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRVector.Axpy) */
  /* Insert the implementation of the Axpy method here... */

   int ierr = 0;
   int type[1];
   void * object;
   struct bHYPRE_IJParCSRVector__data * data, * data_x;
   bHYPRE_IJParCSRVector bHYPREP_x;
   HYPRE_IJVector ij_y, ij_x;
   HYPRE_ParVector yy, xx;
   data = bHYPRE_IJParCSRVector__get_data( self );
   ij_y = data -> ij_b;

   ierr += HYPRE_IJVectorGetObjectType( ij_y, type );
   /* ... don't know how to deal with other types */
   hypre_assert( *type == HYPRE_PARCSR );
   ierr += HYPRE_IJVectorGetObject( ij_y, &object );
   yy = (HYPRE_ParVector) object;

   /* A bHYPRE_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( bHYPRE_Vector_queryInt(x, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_x = bHYPRE_IJParCSRVector__cast( x );
   }
   else
   {
      hypre_assert( "Unrecognized vector type."==(char *)x );
   }

   data_x = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x );
   ij_x = data_x->ij_b;
   ierr += HYPRE_IJVectorGetObjectType( ij_x, type );
   /* ... don't know how to deal with other types */
   hypre_assert( *type == HYPRE_PARCSR );
   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;

   ierr += hypre_ParVectorAxpy( a, (hypre_ParVector *) xx,
                                (hypre_ParVector *) yy );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRVector.Axpy) */
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_IJParCSRVector__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJParCSRVector(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_IJParCSRVector__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_IJParCSRVector(struct 
  bHYPRE_IJParCSRVector__object* obj) {
  return bHYPRE_IJParCSRVector__getURL(obj);
}
struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_MPICommunicator__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) {
  return bHYPRE_MPICommunicator__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRVector_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct bHYPRE_Vector__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_Vector(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_Vector__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_Vector(struct 
  bHYPRE_Vector__object* obj) {
  return bHYPRE_Vector__getURL(obj);
}
struct bHYPRE_IJVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_IJVectorView(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_IJVectorView__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_IJVectorView(struct 
  bHYPRE_IJVectorView__object* obj) {
  return bHYPRE_IJVectorView__getURL(obj);
}
struct bHYPRE_ProblemDefinition__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_ProblemDefinition(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_ProblemDefinition__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_ProblemDefinition(struct 
  bHYPRE_ProblemDefinition__object* obj) {
  return bHYPRE_ProblemDefinition__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRVector_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct bHYPRE_MatrixVectorView__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_bHYPRE_MatrixVectorView(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_MatrixVectorView__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRVector_fgetURL_bHYPRE_MatrixVectorView(struct 
  bHYPRE_MatrixVectorView__object* obj) {
  return bHYPRE_MatrixVectorView__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_IJParCSRVector_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_IJParCSRVector_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
