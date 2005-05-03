/*
 * File:          bHYPRE_SStructParCSRMatrix_Impl.c
 * Symbol:        bHYPRE.SStructParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.SStructParCSRMatrix
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
 * Symbol "bHYPRE.SStructParCSRMatrix" (version 1.0.0)
 * 
 * The SStructParCSR matrix class.
 * 
 * Objects of this type can be cast to SStructBuildMatrix or
 * Operator objects using the {\tt \_\_cast} methods.
 * 
 */

#include "bHYPRE_SStructParCSRMatrix_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "mpi.h"
#include "sstruct_mv.h"
#include "bHYPRE_SStructParCSRVector_Impl.h"
#include "bHYPRE_SStructGraph_Impl.h"
#include "bHYPRE_IJParCSRMatrix_Impl.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix__ctor"

void
impl_bHYPRE_SStructParCSRMatrix__ctor(
  /*in*/ bHYPRE_SStructParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */

   /* To build a SStructParCSRMatrix via Babel: first call the constructor,
      then SetCommunicator, then SetGraph (which internally calls
      HYPRE_SStructMatrixCreate), then any optional parameter set functions
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

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix__dtor"

void
impl_bHYPRE_SStructParCSRMatrix__dtor(
  /*in*/ bHYPRE_SStructParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix matrix;
   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   matrix = data -> matrix;
   ierr += HYPRE_SStructMatrixDestroy( matrix );
   assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix._dtor) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetCommunicator"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetCommunicator(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   data -> comm = (MPI_Comm) mpi_comm;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetIntParameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntParameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ const char* name,
    /*in*/ int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetDoubleParameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleParameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ const char* name,
    /*in*/ double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetDoubleParameter) 
    */
  /* Insert the implementation of the SetDoubleParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetStringParameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetStringParameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ const char* name,
    /*in*/ const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetStringParameter) 
    */
  /* Insert the implementation of the SetStringParameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntArray1Parameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetIntArray2Parameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_GetIntValue"

int32_t
impl_bHYPRE_SStructParCSRMatrix_GetIntValue(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ const char* name,
    /*out*/ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_GetDoubleValue"

int32_t
impl_bHYPRE_SStructParCSRMatrix_GetDoubleValue(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ const char* name,
    /*out*/ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Setup"

int32_t
impl_bHYPRE_SStructParCSRMatrix_Setup(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ bHYPRE_Vector b,
    /*in*/ bHYPRE_Vector x)
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

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Apply"

int32_t
impl_bHYPRE_SStructParCSRMatrix_Apply(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ bHYPRE_Vector b,
    /*inout*/ bHYPRE_Vector* x)
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

   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   /* A bHYPRE_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.SStructParCSRVector" ) )
   {
      bHYPREP_b = bHYPRE_SStructParCSRVector__cast( b );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)b );
   }

   if ( bHYPRE_Vector_queryInt( *x, "bHYPRE.SStructParCSRVector" ) )
   {
      bHYPREP_x = bHYPRE_SStructParCSRVector__cast( *x );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
   }

   data_x = bHYPRE_SStructParCSRVector__get_data( bHYPREP_x );
   Hx = data_x -> vec;
   data_b = bHYPRE_SStructParCSRVector__get_data( bHYPREP_b );
   Hb = data_b -> vec;

   ierr += HYPRE_SStructMatrixMatvec( 1.0, HA, Hb, 0.0, Hx );

   bHYPRE_SStructParCSRVector_deleteRef( bHYPREP_b ); /* ref was created by queryInt */
   bHYPRE_SStructParCSRVector_deleteRef( bHYPREP_x ); /* ref was created by queryInt */

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.Apply) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Initialize"

int32_t
impl_bHYPRE_SStructParCSRMatrix_Initialize(
  /*in*/ bHYPRE_SStructParCSRMatrix self)
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

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Assemble"

int32_t
impl_bHYPRE_SStructParCSRMatrix_Assemble(
  /*in*/ bHYPRE_SStructParCSRMatrix self)
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
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_GetObject"

int32_t
impl_bHYPRE_SStructParCSRMatrix_GetObject(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*out*/ sidl_BaseInterface* A)
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

   HYPRE_IJMatrixGetLocalRange( ijA, &ilower, &iupper, &jlower, &jupper );
   printf( "ilower=%i iupper=%i  ", ilower, iupper );
   printf( "jlower=%i jupper=%i\n", jlower, jupper );

   pA = bHYPRE_IJParCSRMatrix__create();
   p_data = bHYPRE_IJParCSRMatrix__get_data( pA );
   p_data->ij_A = ijA;
   p_data->comm = data -> comm;
   /* The grid and stencil slots of p_data haven't been set, but they shouldn't
      be needed- they are just used for creation of the HYPRE_StructMatrix object.
    */
   *A = sidl_BaseInterface__cast( pA );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.GetObject) */
}

/*
 * Set the matrix graph.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetGraph"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetGraph(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ bHYPRE_SStructGraph graph)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetGraph) */
  /* Insert the implementation of the SetGraph method here... */

   /* To create a matrix one needs a graph and communicator.
      We assume SetCommunicator will be called first.
      So SetGraph can simply call HYPRE_StructMatrixCreate.
      It is an error to call this function
      if HYPRE_StructMatrixCreate has already been called for this matrix.
   */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;
   HYPRE_SStructGraph Hgraph;
   MPI_Comm comm;
   struct bHYPRE_SStructGraph__data * gdata;

   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data->matrix;
   assert( HA==NULL ); /* shouldn't have already been created */
   comm = data->comm;

   gdata = bHYPRE_SStructGraph__get_data( graph );
   Hgraph = gdata->graph;

   ierr += HYPRE_SStructMatrixCreate( comm, Hgraph, &HA );
   data->matrix = HA;

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetGraph) */
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
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetValues"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetValues(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*in*/ int32_t nentries, /*in*/ struct sidl_int__array* entries,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;
   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_SStructMatrixSetValues
      ( HA, part, sidlArrayAddr1( index, 0 ), var, nentries,
        sidlArrayAddr1( entries, 0 ), sidlArrayAddr1( values, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetValues) */
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
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetBoxValues"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetBoxValues(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t var,
    /*in*/ int32_t nentries, /*in*/ struct sidl_int__array* entries,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetBoxValues) */
  /* Insert the implementation of the SetBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;
   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_SStructMatrixSetBoxValues
      ( HA, part, sidlArrayAddr1( ilower, 0 ), sidlArrayAddr1( iupper, 0 ),
        var, nentries, sidlArrayAddr1( entries, 0 ), sidlArrayAddr1( values, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetBoxValues) */
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
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_AddToValues"

int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToValues(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* index, /*in*/ int32_t var,
    /*in*/ int32_t nentries, /*in*/ struct sidl_int__array* entries,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;
   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_SStructMatrixAddToValues
      ( HA, part, sidlArrayAddr1( index, 0 ), var, nentries,
        sidlArrayAddr1( entries, 0 ), sidlArrayAddr1( values, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.AddToValues) */
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
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_AddToBoxValues"

int32_t
impl_bHYPRE_SStructParCSRMatrix_AddToBoxValues(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ int32_t part,
    /*in*/ struct sidl_int__array* ilower,
    /*in*/ struct sidl_int__array* iupper, /*in*/ int32_t var,
    /*in*/ int32_t nentries, /*in*/ struct sidl_int__array* entries,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.AddToBoxValues) */
  /* Insert the implementation of the AddToBoxValues method here... */

   int ierr = 0;
   struct bHYPRE_SStructParCSRMatrix__data * data;
   HYPRE_SStructMatrix HA;
   data = bHYPRE_SStructParCSRMatrix__get_data( self );
   HA = data -> matrix;

   ierr += HYPRE_SStructMatrixAddToBoxValues
      ( HA, part, sidlArrayAddr1( ilower, 0 ), sidlArrayAddr1( iupper, 0 ),
        var, nentries, sidlArrayAddr1( entries, 0 ), sidlArrayAddr1( values, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.AddToBoxValues) */
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
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetSymmetric"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetSymmetric(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ int32_t part,
    /*in*/ int32_t var, /*in*/ int32_t to_var, /*in*/ int32_t symmetric)
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

/*
 * Define symmetry properties for all non-stencil matrix
 * entries.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetNSSymmetric"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetNSSymmetric(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ int32_t symmetric)
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

/*
 * Set the matrix to be complex.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_SetComplex"

int32_t
impl_bHYPRE_SStructParCSRMatrix_SetComplex(
  /*in*/ bHYPRE_SStructParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructParCSRMatrix.SetComplex) */
  /* Insert the implementation of the SetComplex method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructParCSRMatrix.SetComplex) */
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructParCSRMatrix_Print"

int32_t
impl_bHYPRE_SStructParCSRMatrix_Print(
  /*in*/ bHYPRE_SStructParCSRMatrix self, /*in*/ const char* filename,
    /*in*/ int32_t all)
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
