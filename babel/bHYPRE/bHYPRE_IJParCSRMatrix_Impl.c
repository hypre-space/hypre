/*
 * File:          bHYPRE_IJParCSRMatrix_Impl.c
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * sidl Created:  20050317 11:17:39 PST
 * Generated:     20050317 11:17:44 PST
 * Description:   Server-side implementation for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 * source-line   = 794
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.IJParCSRMatrix" (version 1.0.0)
 * 
 * The IJParCSR matrix class.
 * 
 * Objects of this type can be cast to IJBuildMatrix, Operator, or
 * CoefficientAccess objects using the {\tt \_\_cast} methods.
 * 
 */

#include "bHYPRE_IJParCSRMatrix_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "bHYPRE_IJParCSRVector_Impl.h"
#include "HYPRE_parcsr_mv.h"
#include "mpi.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix__ctor"

void
impl_bHYPRE_IJParCSRMatrix__ctor(
  /*in*/ bHYPRE_IJParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct bHYPRE_IJParCSRMatrix__data * data;

   data = hypre_CTAlloc(struct bHYPRE_IJParCSRMatrix__data,1);
   /* data = (struct bHYPRE_IJParCSRMatrix__data *)
      malloc( sizeof ( struct bHYPRE_IJParCSRMatrix__data ) ); */

   data -> comm = MPI_COMM_NULL;
   data -> ij_A = NULL;

   bHYPRE_IJParCSRMatrix__set_data( self, data );
   
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix__dtor"

void
impl_bHYPRE_IJParCSRMatrix__dtor(
  /*in*/ bHYPRE_IJParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixDestroy( ij_A );
   assert( ierr == 0 );

   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix._dtor) */
}

/*
 * (Optional) Set the max number of nonzeros to expect in each
 * row of the diagonal and off-diagonal blocks.  The diagonal
 * block is the submatrix whose column numbers correspond to
 * rows owned by this process, and the off-diagonal block is
 * everything else.  The arrays {\tt diag\_sizes} and {\tt
 * offdiag\_sizes} contain estimated sizes for each row of the
 * diagonal and off-diagonal blocks, respectively.  This routine
 * can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes"

int32_t
impl_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ struct sidl_int__array* diag_sizes,
    /*in*/ struct sidl_int__array* offdiag_sizes)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetDiagOffdSizes) */
  /* Insert the implementation of the SetDiagOffdSizes method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixSetDiagOffdSizes( ij_A, 
                                          sidlArrayAddr1(diag_sizes, 0), 
                                          sidlArrayAddr1(offdiag_sizes, 0) );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetDiagOffdSizes) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetCommunicator"

int32_t
impl_bHYPRE_IJParCSRMatrix_SetCommunicator(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   /* The data type of the last argument, mpi_comm, should be MPI_Comm */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

#ifdef HYPRE_DEBUG
   printf("impl_bHYPRE_IJParCSRMatrix_SetCommunicator\n");
#endif
   
   data -> comm = (MPI_Comm) mpi_comm;

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetIntParameter"

int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntParameter(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ const char* name,
    /*in*/ int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetDoubleParameter"

int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleParameter(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ const char* name,
    /*in*/ double value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetStringParameter"

int32_t
impl_bHYPRE_IJParCSRMatrix_SetStringParameter(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ const char* name,
    /*in*/ const char* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetStringParameter) */
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter"

int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetIntArray1Parameter) */
  /* Insert the implementation of the SetIntArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetIntArray1Parameter) */
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter"

int32_t
impl_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetIntArray2Parameter) */
  /* Insert the implementation of the SetIntArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetIntArray2Parameter) */
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter"

int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetDoubleArray1Parameter) */
  /* Insert the implementation of the SetDoubleArray1Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetDoubleArray1Parameter) */
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter"

int32_t
impl_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ const char* name,
    /*in*/ struct sidl_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetDoubleArray2Parameter) */
  /* Insert the implementation of the SetDoubleArray2Parameter method here... */
   return 1;
  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetDoubleArray2Parameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GetIntValue"

int32_t
impl_bHYPRE_IJParCSRMatrix_GetIntValue(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ const char* name,
    /*out*/ int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GetDoubleValue"

int32_t
impl_bHYPRE_IJParCSRMatrix_GetDoubleValue(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ const char* name,
    /*out*/ double* value)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_Setup"

int32_t
impl_bHYPRE_IJParCSRMatrix_Setup(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ bHYPRE_Vector b,
    /*in*/ bHYPRE_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.Setup) */
  /* Insert the implementation of the Setup method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixAssemble( ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_Apply"

int32_t
impl_bHYPRE_IJParCSRMatrix_Apply(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ bHYPRE_Vector b,
    /*inout*/ bHYPRE_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.Apply) */
  /* Insert the implementation of the Apply method here... */

   /* Apply means to multiply by a vector, y = A*x .  Here, we call
    * the HYPRE Matvec function which performs y = a*A*x + b*y (we set
    * a=1 and b=0).  */
   int ierr=0;
   void * object;
   struct bHYPRE_IJParCSRMatrix__data * data;
   struct bHYPRE_IJParCSRVector__data * data_x, * data_b;
   bHYPRE_IJParCSRVector bHYPREP_b, bHYPREP_x;
   HYPRE_IJMatrix ij_A;
   HYPRE_IJVector ij_x, ij_b;
   HYPRE_ParVector xx, bb;
   HYPRE_ParCSRMatrix A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );
   ij_A = data -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &object );
   A = (HYPRE_ParCSRMatrix) object;

   /* A bHYPRE_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( bHYPRE_Vector_queryInt(b, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_b = bHYPRE_IJParCSRVector__cast( b );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)b );
   }

   if ( bHYPRE_Vector_queryInt( *x, "bHYPRE.IJParCSRVector" ) )
   {
      bHYPREP_x = bHYPRE_IJParCSRVector__cast( *x );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
   }

   data_x = bHYPRE_IJParCSRVector__get_data( bHYPREP_x );
   ij_x = data_x -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;
   data_b = bHYPRE_IJParCSRVector__get_data( bHYPREP_b );
   ij_b = data_b -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &object );
   bb = (HYPRE_ParVector) object;

   ierr += HYPRE_ParCSRMatrixMatvec( 1.0, A, bb, 0.0, xx );

   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_b ); /* ref was created by queryInt */
   bHYPRE_IJParCSRVector_deleteRef( bHYPREP_x ); /* ref was created by queryInt */

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.Apply) */
}

/*
 * The GetRow method will allocate space for its two output
 * arrays on the first call.  The space will be reused on
 * subsequent calls.  Thus the user must not delete them, yet
 * must not depend on the data from GetRow to persist beyond the
 * next GetRow call.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GetRow"

int32_t
impl_bHYPRE_IJParCSRMatrix_GetRow(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ int32_t row, /*out*/ int32_t* size,
    /*out*/ struct sidl_int__array** col_ind,
    /*out*/ struct sidl_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GetRow) */
  /* Insert the implementation of the GetRow method here... */

   int ierr=0;
   void * object;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix bHYPREP_A;
   int * iindices[1];
   double * dvalues[1];

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &object );
   bHYPREP_A = (HYPRE_ParCSRMatrix) object;

   *col_ind = sidl_int__array_create1d( size[0] );
   *values = sidl_double__array_create1d( size[0] );

   *iindices = sidlArrayAddr1( *col_ind, 0 );
   *dvalues = sidlArrayAddr1( *values, 0 );

   /* RestoreRow doesn't do anything but reset a parameter.  Its
    * function is to make sure the user who calls GetRow is aware that
    * the data in the output arrays will be changed. */
   HYPRE_ParCSRMatrixRestoreRow( bHYPREP_A, row, size, iindices, dvalues );
   ierr += HYPRE_ParCSRMatrixGetRow( bHYPREP_A, row, size, iindices, dvalues );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GetRow) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_Initialize"

int32_t
impl_bHYPRE_IJParCSRMatrix_Initialize(
  /*in*/ bHYPRE_IJParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.Initialize) */
  /* Insert the implementation of the Initialize method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixInitialize( ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.Initialize) */
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
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_Assemble"

int32_t
impl_bHYPRE_IJParCSRMatrix_Assemble(
  /*in*/ bHYPRE_IJParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.Assemble) */
  /* Insert the implementation of the Assemble method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixAssemble( ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.Assemble) */
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
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GetObject"

int32_t
impl_bHYPRE_IJParCSRMatrix_GetObject(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*out*/ sidl_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GetObject) */
  /* Insert the implementation of the GetObject method here... */

   bHYPRE_IJParCSRMatrix_addRef( self );
   *A = sidl_BaseInterface__cast( self );
   return( 0 );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GetObject) */
}

/*
 * Set the local range for a matrix object.  Each process owns
 * some unique consecutive range of rows, indicated by the
 * global row indices {\tt ilower} and {\tt iupper}.  The row
 * data is required to be such that the value of {\tt ilower} on
 * any process $p$ be exactly one more than the value of {\tt
 * iupper} on process $p-1$.  Note that the first row of the
 * global matrix may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * For square matrices, {\tt jlower} and {\tt jupper} typically
 * should match {\tt ilower} and {\tt iupper}, respectively.
 * For rectangular matrices, {\tt jlower} and {\tt jupper}
 * should define a partitioning of the columns.  This
 * partitioning must be used for any vector $v$ that will be
 * used in matrix-vector products with the rectangular matrix.
 * The matrix data structure may use {\tt jlower} and {\tt
 * jupper} to store the diagonal blocks (rectangular in general)
 * of the matrix separately from the rest of the matrix.
 * 
 * Collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetLocalRange"

int32_t
impl_bHYPRE_IJParCSRMatrix_SetLocalRange(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ int32_t ilower,
    /*in*/ int32_t iupper, /*in*/ int32_t jlower, /*in*/ int32_t jupper)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetLocalRange) */
  /* Insert the implementation of the SetLocalRange method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   if ( data -> comm == MPI_COMM_NULL )    
   {
#ifdef HYPRE_DEBUG
      printf("Set Communicator must be called before Create in IJBuilder\n");
#endif
      return( -1 );
   }
   else
   {
      ierr = HYPRE_IJMatrixCreate( data -> comm,
                                   ilower, iupper, jlower, jupper, &ij_A );

      ierr = HYPRE_IJMatrixSetObjectType( ij_A, HYPRE_PARCSR );

      data -> ij_A = ij_A;
   
      return( ierr );
   }

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetLocalRange) */
}

/*
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt
 * ncols} and {\tt rows} are of dimension {\tt nrows} and
 * contain the number of columns in each row and the row
 * indices, respectively.  The array {\tt cols} contains the
 * column indices for each of the {\tt rows}, and is ordered by
 * rows.  The data in the {\tt values} array corresponds
 * directly to the column entries in {\tt cols}.  Erases any
 * previous values at the specified locations and replaces them
 * with new ones, or, if there was no value there before,
 * inserts a new one.
 * 
 * Not collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetValues"

int32_t
impl_bHYPRE_IJParCSRMatrix_SetValues(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ int32_t nrows,
    /*in*/ struct sidl_int__array* ncols, /*in*/ struct sidl_int__array* rows,
    /*in*/ struct sidl_int__array* cols,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixSetValues( ij_A, nrows,
                                   sidlArrayAddr1(ncols, 0),
                                   sidlArrayAddr1(rows, 0),
                                   sidlArrayAddr1(cols, 0),
                                   sidlArrayAddr1(values, 0) ); 

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetValues) */
}

/*
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_AddToValues"

int32_t
impl_bHYPRE_IJParCSRMatrix_AddToValues(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ int32_t nrows,
    /*in*/ struct sidl_int__array* ncols, /*in*/ struct sidl_int__array* rows,
    /*in*/ struct sidl_int__array* cols,
    /*in*/ struct sidl_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixAddToValues( ij_A, nrows,
                                     sidlArrayAddr1(ncols, 0) ,
                                     sidlArrayAddr1(rows, 0) ,
                                     sidlArrayAddr1(cols, 0) ,
                                     sidlArrayAddr1(values, 0)  ); 
   
   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.AddToValues) */
}

/*
 * Gets range of rows owned by this processor and range of
 * column partitioning for this processor.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GetLocalRange"

int32_t
impl_bHYPRE_IJParCSRMatrix_GetLocalRange(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*out*/ int32_t* ilower,
    /*out*/ int32_t* iupper, /*out*/ int32_t* jlower, /*out*/ int32_t* jupper)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GetLocalRange) */
  /* Insert the implementation of the GetLocalRange method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixGetLocalRange( ij_A, ilower, iupper, jlower, jupper );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GetLocalRange) */
}

/*
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GetRowCounts"

int32_t
impl_bHYPRE_IJParCSRMatrix_GetRowCounts(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ int32_t nrows,
    /*in*/ struct sidl_int__array* rows,
    /*inout*/ struct sidl_int__array** ncols)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GetRowCounts) */
  /* Insert the implementation of the GetRowCounts method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixGetRowCounts( ij_A, nrows,
                                      sidlArrayAddr1(rows, 0),
                                      sidlArrayAddr1(*ncols, 0));

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GetRowCounts) */
}

/*
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_GetValues"

int32_t
impl_bHYPRE_IJParCSRMatrix_GetValues(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ int32_t nrows,
    /*in*/ struct sidl_int__array* ncols, /*in*/ struct sidl_int__array* rows,
    /*in*/ struct sidl_int__array* cols,
    /*inout*/ struct sidl_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.GetValues) */
  /* Insert the implementation of the GetValues method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixGetValues( ij_A, nrows,
                                   sidlArrayAddr1(ncols, 0),
                                   sidlArrayAddr1(rows, 0),
                                   sidlArrayAddr1(cols, 0),
                                   sidlArrayAddr1(*values, 0) ); 

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.GetValues) */
}

/*
 * (Optional) Set the max number of nonzeros to expect in each
 * row.  The array {\tt sizes} contains estimated sizes for each
 * row on this process.  This call can significantly improve the
 * efficiency of matrix construction, and should always be
 * utilized if possible.
 * 
 * Not collective.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_SetRowSizes"

int32_t
impl_bHYPRE_IJParCSRMatrix_SetRowSizes(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ struct sidl_int__array* sizes)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.SetRowSizes) */
  /* Insert the implementation of the SetRowSizes method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixSetRowSizes( ij_A, sidlArrayAddr1(sizes, 0) );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.SetRowSizes) */
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_Print"

int32_t
impl_bHYPRE_IJParCSRMatrix_Print(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ const char* filename)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.Print) */
  /* Insert the implementation of the Print method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixPrint( ij_A, filename);

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.Print) */
}

/*
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_IJParCSRMatrix_Read"

int32_t
impl_bHYPRE_IJParCSRMatrix_Read(
  /*in*/ bHYPRE_IJParCSRMatrix self, /*in*/ const char* filename,
    /*in*/ void* comm)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.IJParCSRMatrix.Read) */
  /* Insert the implementation of the Read method here... */

   int ierr=0;
   struct bHYPRE_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = bHYPRE_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixRead( filename, data -> comm, HYPRE_PARCSR, &ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(bHYPRE.IJParCSRMatrix.Read) */
}
