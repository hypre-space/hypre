/*
 * File:          Hypre_IJParCSRMatrix_Impl.c
 * Symbol:        Hypre.IJParCSRMatrix-v0.1.7
 * Symbol Type:   class
 * Babel Version: 0.8.0
 * SIDL Created:  20030306 17:05:11 PST
 * Generated:     20030306 17:05:15 PST
 * Description:   Server-side implementation for Hypre.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.0
 * source-line   = 799
 * source-url    = file:/home/falgout/linear_solvers/babel/Interfaces.idl
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.IJParCSRMatrix" (version 0.1.7)
 * 
 * The IJParCSR matrix class.
 * 
 * Objects of this type can be cast to IJBuildMatrix, Operator, or
 * CoefficientAccess objects using the {\tt \_\_cast} methods.
 * 
 */

#include "Hypre_IJParCSRMatrix_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "Hypre_IJParCSRVector_Impl.h"
#include "HYPRE_parcsr_mv.h"
#include "mpi.h"
/* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix__ctor"

void
impl_Hypre_IJParCSRMatrix__ctor(
  Hypre_IJParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct Hypre_IJParCSRMatrix__data * data;

   data = hypre_CTAlloc(struct Hypre_IJParCSRMatrix__data,1);
   /* data = (struct Hypre_IJParCSRMatrix__data *)
      malloc( sizeof ( struct Hypre_IJParCSRMatrix__data ) ); */

   data -> comm = MPI_COMM_NULL;
   data -> ij_A = NULL;

   Hypre_IJParCSRMatrix__set_data( self, data );
   
  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix__dtor"

void
impl_Hypre_IJParCSRMatrix__dtor(
  Hypre_IJParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixDestroy( ij_A );
   assert( ierr == 0 );

   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix._dtor) */
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
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_SetDiagOffdSizes"

int32_t
impl_Hypre_IJParCSRMatrix_SetDiagOffdSizes(
  Hypre_IJParCSRMatrix self, struct SIDL_int__array* diag_sizes,
    struct SIDL_int__array* offdiag_sizes)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.SetDiagOffdSizes) */
  /* Insert the implementation of the SetDiagOffdSizes method here... */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixSetDiagOffdSizes( ij_A, 
                                          SIDLArrayAddr1(diag_sizes, 0), 
                                          SIDLArrayAddr1(offdiag_sizes, 0) );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.SetDiagOffdSizes) */
}

/*
 * Set the MPI Communicator.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_SetCommunicator"

int32_t
impl_Hypre_IJParCSRMatrix_SetCommunicator(
  Hypre_IJParCSRMatrix self, void* mpi_comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   /* The data type of the last argument, mpi_comm, should be MPI_Comm */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;

   data = Hypre_IJParCSRMatrix__get_data( self );

#ifdef HYPRE_DEBUG
   printf("impl_Hypre_IJParCSRMatrix_SetCommunicator\n");
#endif
   
   data -> comm = (MPI_Comm) mpi_comm;

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.SetCommunicator) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_SetIntParameter"

int32_t
impl_Hypre_IJParCSRMatrix_SetIntParameter(
  Hypre_IJParCSRMatrix self, const char* name, int32_t value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.SetIntParameter) */
  /* Insert the implementation of the SetIntParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.SetIntParameter) */
}

/*
 * Set the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_SetDoubleParameter"

int32_t
impl_Hypre_IJParCSRMatrix_SetDoubleParameter(
  Hypre_IJParCSRMatrix self, const char* name, double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.SetDoubleParameter) */
  /* Insert the implementation of the SetDoubleParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.SetDoubleParameter) */
}

/*
 * Set the string parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_SetStringParameter"

int32_t
impl_Hypre_IJParCSRMatrix_SetStringParameter(
  Hypre_IJParCSRMatrix self, const char* name, const char* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.SetStringParameter) */
  /* Insert the implementation of the SetStringParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.SetStringParameter) */
}

/*
 * Set the int array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_SetIntArrayParameter"

int32_t
impl_Hypre_IJParCSRMatrix_SetIntArrayParameter(
  Hypre_IJParCSRMatrix self, const char* name, struct SIDL_int__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.SetIntArrayParameter) */
  /* Insert the implementation of the SetIntArrayParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.SetIntArrayParameter) */
}

/*
 * Set the double array parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_SetDoubleArrayParameter"

int32_t
impl_Hypre_IJParCSRMatrix_SetDoubleArrayParameter(
  Hypre_IJParCSRMatrix self, const char* name, struct SIDL_double__array* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.SetDoubleArrayParameter) 
    */
  /* Insert the implementation of the SetDoubleArrayParameter method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.SetDoubleArrayParameter) */
}

/*
 * Set the int parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_GetIntValue"

int32_t
impl_Hypre_IJParCSRMatrix_GetIntValue(
  Hypre_IJParCSRMatrix self, const char* name, int32_t* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.GetIntValue) */
  /* Insert the implementation of the GetIntValue method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.GetIntValue) */
}

/*
 * Get the double parameter associated with {\tt name}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_GetDoubleValue"

int32_t
impl_Hypre_IJParCSRMatrix_GetDoubleValue(
  Hypre_IJParCSRMatrix self, const char* name, double* value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.GetDoubleValue) */
  /* Insert the implementation of the GetDoubleValue method here... */

   return 1;

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.GetDoubleValue) */
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_Setup"

int32_t
impl_Hypre_IJParCSRMatrix_Setup(
  Hypre_IJParCSRMatrix self, Hypre_Vector b, Hypre_Vector x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.Setup) */
  /* Insert the implementation of the Setup method here... */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixAssemble( ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.Setup) */
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_Apply"

int32_t
impl_Hypre_IJParCSRMatrix_Apply(
  Hypre_IJParCSRMatrix self, Hypre_Vector b, Hypre_Vector* x)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.Apply) */
  /* Insert the implementation of the Apply method here... */

   /* Apply means to multiply by a vector, y = A*x .  Here, we call
    * the HYPRE Matvec function which performs y = a*A*x + b*y (we set
    * a=1 and b=0).  */
   int ierr=0;
   void * object;
   struct Hypre_IJParCSRMatrix__data * data;
   struct Hypre_IJParCSRVector__data * data_x, * data_b;
   Hypre_IJParCSRVector HypreP_b, HypreP_x;
   HYPRE_IJMatrix ij_A;
   HYPRE_IJVector ij_x, ij_b;
   HYPRE_ParVector xx, bb;
   HYPRE_ParCSRMatrix A;

   data = Hypre_IJParCSRMatrix__get_data( self );
   ij_A = data -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &object );
   A = (HYPRE_ParCSRMatrix) object;

   /* A Hypre_Vector is just an interface, we have no knowledge of its
    * contents.  Check whether it's something we know how to handle.
    * If not, die. */
   if ( Hypre_Vector_queryInt(b, "Hypre.IJParCSRVector" ) )
   {
      HypreP_b = Hypre_IJParCSRVector__cast( b );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)b );
   }

   if ( Hypre_Vector_queryInt( *x, "Hypre.IJParCSRVector" ) )
   {
      HypreP_x = Hypre_IJParCSRVector__cast( *x );
   }
   else
   {
      assert( "Unrecognized vector type."==(char *)x );
   }

   data_x = Hypre_IJParCSRVector__get_data( HypreP_x );
   ij_x = data_x -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_x, &object );
   xx = (HYPRE_ParVector) object;
   data_b = Hypre_IJParCSRVector__get_data( HypreP_b );
   ij_b = data_b -> ij_b;
   ierr += HYPRE_IJVectorGetObject( ij_b, &object );
   bb = (HYPRE_ParVector) object;

   ierr += HYPRE_ParCSRMatrixMatvec( 1.0, A, bb, 0.0, xx );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.Apply) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_Initialize"

int32_t
impl_Hypre_IJParCSRMatrix_Initialize(
  Hypre_IJParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.Initialize) */
  /* Insert the implementation of the Initialize method here... */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixInitialize( ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.Initialize) */
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
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_Assemble"

int32_t
impl_Hypre_IJParCSRMatrix_Assemble(
  Hypre_IJParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.Assemble) */
  /* Insert the implementation of the Assemble method here... */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixAssemble( ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.Assemble) */
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
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_GetObject"

int32_t
impl_Hypre_IJParCSRMatrix_GetObject(
  Hypre_IJParCSRMatrix self, SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.GetObject) */
  /* Insert the implementation of the GetObject method here... */

   Hypre_IJParCSRMatrix_addRef( self );
   *A = SIDL_BaseInterface__cast( self );
   return( 0 );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.GetObject) */
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
 * RDF: Changed name from 'Create' (x)
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_SetLocalRange"

int32_t
impl_Hypre_IJParCSRMatrix_SetLocalRange(
  Hypre_IJParCSRMatrix self, int32_t ilower, int32_t iupper, int32_t jlower,
    int32_t jupper)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.SetLocalRange) */
  /* Insert the implementation of the SetLocalRange method here... */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

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

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.SetLocalRange) */
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
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_SetValues"

int32_t
impl_Hypre_IJParCSRMatrix_SetValues(
  Hypre_IJParCSRMatrix self, int32_t nrows, struct SIDL_int__array* ncols,
    struct SIDL_int__array* rows, struct SIDL_int__array* cols,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixSetValues( ij_A, nrows,
                                   SIDLArrayAddr1(ncols, 0),
                                   SIDLArrayAddr1(rows, 0),
                                   SIDLArrayAddr1(cols, 0),
                                   SIDLArrayAddr1(values, 0) ); 

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.SetValues) */
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
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_AddToValues"

int32_t
impl_Hypre_IJParCSRMatrix_AddToValues(
  Hypre_IJParCSRMatrix self, int32_t nrows, struct SIDL_int__array* ncols,
    struct SIDL_int__array* rows, struct SIDL_int__array* cols,
    struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixAddToValues( ij_A, nrows,
                                     SIDLArrayAddr1(ncols, 0) ,
                                     SIDLArrayAddr1(rows, 0) ,
                                     SIDLArrayAddr1(cols, 0) ,
                                     SIDLArrayAddr1(values, 0)  ); 
   
   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.AddToValues) */
}

/*
 * Gets range of rows owned by this processor and range of
 * column partitioning for this processor.
 * 
 * RDF: New (x)
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_GetLocalRange"

int32_t
impl_Hypre_IJParCSRMatrix_GetLocalRange(
  Hypre_IJParCSRMatrix self, int32_t* ilower, int32_t* iupper, int32_t* jlower,
    int32_t* jupper)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.GetLocalRange) */
  /* Insert the implementation of the GetLocalRange method here... */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixGetLocalRange( ij_A, ilower, iupper, jlower, jupper );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.GetLocalRange) */
}

/*
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 * 
 * RDF: New (x)
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_GetRowCounts"

int32_t
impl_Hypre_IJParCSRMatrix_GetRowCounts(
  Hypre_IJParCSRMatrix self, int32_t nrows, struct SIDL_int__array* rows,
    struct SIDL_int__array** ncols)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.GetRowCounts) */
  /* Insert the implementation of the GetRowCounts method here... */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixGetRowCounts( ij_A, nrows,
                                      SIDLArrayAddr1(rows, 0),
                                      SIDLArrayAddr1(*ncols, 0));

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.GetRowCounts) */
}

/*
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 * 
 * RDF: New (x)
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_GetValues"

int32_t
impl_Hypre_IJParCSRMatrix_GetValues(
  Hypre_IJParCSRMatrix self, int32_t nrows, struct SIDL_int__array* ncols,
    struct SIDL_int__array* rows, struct SIDL_int__array* cols,
    struct SIDL_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.GetValues) */
  /* Insert the implementation of the GetValues method here... */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixGetValues( ij_A, nrows,
                                   SIDLArrayAddr1(ncols, 0),
                                   SIDLArrayAddr1(rows, 0),
                                   SIDLArrayAddr1(cols, 0),
                                   SIDLArrayAddr1(*values, 0) ); 

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.GetValues) */
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
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_SetRowSizes"

int32_t
impl_Hypre_IJParCSRMatrix_SetRowSizes(
  Hypre_IJParCSRMatrix self, struct SIDL_int__array* sizes)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.SetRowSizes) */
  /* Insert the implementation of the SetRowSizes method here... */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixSetRowSizes( ij_A, SIDLArrayAddr1(sizes, 0) );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.SetRowSizes) */
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_Print"

int32_t
impl_Hypre_IJParCSRMatrix_Print(
  Hypre_IJParCSRMatrix self, const char* filename)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.Print) */
  /* Insert the implementation of the Print method here... */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixPrint( ij_A, filename);

   return ierr;

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.Print) */
}

/*
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_Read"

int32_t
impl_Hypre_IJParCSRMatrix_Read(
  Hypre_IJParCSRMatrix self, const char* filename, void* comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.Read) */
  /* Insert the implementation of the Read method here... */

   int ierr=0;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   ierr = HYPRE_IJMatrixRead( filename, data -> comm, HYPRE_PARCSR, &ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.Read) */
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
#define __FUNC__ "impl_Hypre_IJParCSRMatrix_GetRow"

int32_t
impl_Hypre_IJParCSRMatrix_GetRow(
  Hypre_IJParCSRMatrix self, int32_t row, int32_t* size,
    struct SIDL_int__array** col_ind, struct SIDL_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.IJParCSRMatrix.GetRow) */
  /* Insert the implementation of the GetRow method here... */

   int ierr=0;
   void * object;
   struct Hypre_IJParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;
   HYPRE_ParCSRMatrix HypreP_A;
   int * iindices[1];
   double * dvalues[1];

   data = Hypre_IJParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;
   ierr += HYPRE_IJMatrixGetObject( ij_A, &object );
   HypreP_A = (HYPRE_ParCSRMatrix) object;

   *col_ind = SIDL_int__array_create1d( size[0] );
   *values = SIDL_double__array_create1d( size[0] );

   *iindices = SIDLArrayAddr1( *col_ind, 0 );
   *dvalues = SIDLArrayAddr1( *values, 0 );

   /* RestoreRow doesn't do anything but reset a parameter.  Its
    * function is to make sure the user who calls GetRow is aware that
    * the data in the output arrays will be changed. */
   HYPRE_ParCSRMatrixRestoreRow( HypreP_A, row, size, iindices, dvalues );
   ierr += HYPRE_ParCSRMatrixGetRow( HypreP_A, row, size, iindices, dvalues );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.IJParCSRMatrix.GetRow) */
}

/**
 * ================= BEGIN UNREFERENCED METHOD(S) ================
 * The following code segment(s) belong to unreferenced method(s).
 * This can result from a method rename/removal in the SIDL file.
 * Move or remove the code in order to compile cleanly.
 */
/* ================== END UNREFERENCED METHOD(S) ================= */
