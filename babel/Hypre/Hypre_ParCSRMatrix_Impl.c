/*
 * File:          Hypre_ParCSRMatrix_Impl.c
 * Symbol:        Hypre.ParCSRMatrix-v0.1.5
 * Symbol Type:   class
 * Babel Version: 0.6.1
 * SIDL Created:  20020104 15:27:10 PST
 * Generated:     20020104 15:27:19 PST
 * Description:   Server-side implementation for Hypre.ParCSRMatrix
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "Hypre.ParCSRMatrix" (version 0.1.5)
 * 
 * A single class that implements both a build interface and an operator
 * interface. It returns itself for <code>GetConstructedObject</code>.
 */

#include "Hypre_ParCSRMatrix_Impl.h"

/* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix._includes) */
/* Put additional includes or other arbitrary code here... */
/* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix__ctor"

void
impl_Hypre_ParCSRMatrix__ctor(
  Hypre_ParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix._ctor) */
  /* Insert the implementation of the constructor method here... */
   struct Hypre_ParCSRMatrix__data * data;

   printf("impl_Hypre_ParCSRMatrix__ctor\n");

   data = hypre_CTAlloc(struct Hypre_ParCSRMatrix__data,1);
   /* data = (struct Hypre_ParCSRMatrix__data *)
      malloc( sizeof ( struct Hypre_ParCSRMatrix__data ) ); */

   data -> comm = NULL;
   data -> ij_A = NULL;

   Hypre_ParCSRMatrix__set_data( self, data );
   
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix__dtor"

void
impl_Hypre_ParCSRMatrix__dtor(
  Hypre_ParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix._dtor) */
  /* Insert the implementation of the destructor method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   printf("impl_Hypre_ParCSRMatrix__dtor\n");

   ierr = HYPRE_IJMatrixDestroy( ij_A );

   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix._dtor) */
}

/*
 * Adds to values for {\tt nrows} of the matrix.  Usage details are
 * analogous to \Ref{HYPRE_IJMatrixSetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value there
 * before, inserts a new one.
 * 
 * Not collective.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix_AddToValues"

int32_t
impl_Hypre_ParCSRMatrix_AddToValues(
  Hypre_ParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.AddToValues) */
  /* Insert the implementation of the AddToValues method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   printf("impl_Hypre_ParCSRMatrix_AddToValues\n");

   ierr = HYPRE_IJMatrixAddToValues( ij_A,
                            nrows,
                            SIDLArrayAddr1(ncols, 0) ,
                            SIDLArrayAddr1(rows, 0) ,
                            SIDLArrayAddr1(cols, 0) ,
                            SIDLArrayAddr1(values, 0)  ); 
   
   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.AddToValues) */
}

/*
 * Method:  Apply
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix_Apply"

int32_t
impl_Hypre_ParCSRMatrix_Apply(
  Hypre_ParCSRMatrix self,
  Hypre_Vector x,
  Hypre_Vector* y)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.Apply) */
  /* Insert the implementation of the Apply method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   printf("impl_Hypre_ParCSRMatrix_Apply\n");
   printf( "Stub\n");
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.Apply) */
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
#define __FUNC__ "impl_Hypre_ParCSRMatrix_Assemble"

int32_t
impl_Hypre_ParCSRMatrix_Assemble(
  Hypre_ParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.Assemble) */
  /* Insert the implementation of the Assemble method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   printf("impl_Hypre_ParCSRMatrix_Assemble\n");

   ierr = HYPRE_IJMatrixAssemble( ij_A );

   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.Assemble) */
}

/*
 * Create a matrix object.  Each process owns some unique consecutive
 * range of rows, indicated by the global row indices {\tt ilower} and
 * {\tt iupper}.  The row data is required to be such that the value
 * of {\tt ilower} on any process $p$ be exactly one more than the
 * value of {\tt iupper} on process $p-1$.  Note that the first row of
 * the global matrix may start with any integer value.  In particular,
 * one may use zero- or one-based indexing.
 * 
 * For square matrices, {\tt jlower} and {\tt jupper} typically should
 * match {\tt ilower} and {\tt iupper}, respectively.  For rectangular
 * matrices, {\tt jlower} and {\tt jupper} should define a
 * partitioning of the columns.  This partitioning must be used for
 * any vector $v$ that will be used in matrix-vector products with the
 * rectangular matrix.  The matrix data structure may use {\tt jlower}
 * and {\tt jupper} to store the diagonal blocks (rectangular in
 * general) of the matrix separately from the rest of the matrix.
 * 
 * Collective.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix_Create"

int32_t
impl_Hypre_ParCSRMatrix_Create(
  Hypre_ParCSRMatrix self,
  int32_t ilower,
  int32_t iupper,
  int32_t jlower,
  int32_t jupper)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.Create) */
  /* Insert the implementation of the Create method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;
   printf("impl_Hypre_ParCSRMatrix_Create\n");

   if ( data-> comm == NULL )    
   {
#ifdef HYPRE_DEBUG
      printf("Set Communicator must be called before Create in IJBuilder\n");
#endif
      return( -1 );
   }
   else
   {
      ierr = HYPRE_IJMatrixCreate( *(data -> comm),
                          ilower,
                          iupper,
                          jlower,
                          jupper,
                          &ij_A );

      ierr = HYPRE_IJMatrixSetObjectType( ij_A, HYPRE_PARCSR );

      data -> ij_A = ij_A;
   
      return( ierr );
   }
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.Create) */
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
#define __FUNC__ "impl_Hypre_ParCSRMatrix_GetObject"

int32_t
impl_Hypre_ParCSRMatrix_GetObject(
  Hypre_ParCSRMatrix self,
  SIDL_BaseInterface* A)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.GetObject) */
  /* Insert the implementation of the GetObject method here... */
   printf("impl_Hypre_ParCSRMatrix_GetObject\n");

   /* This class is implemented
      so that it returns *itself* as the returned Object, because it is both
      the IJBuilder and the matrix object that it builds. */
   Hypre_ParCSRMatrix_addReference( self );
   
   *A = SIDL_BaseInterface__cast( self );
   
   return( 0 );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.GetObject) */
}

/*
 * Method:  GetRow
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix_GetRow"

int32_t
impl_Hypre_ParCSRMatrix_GetRow(
  Hypre_ParCSRMatrix self,
  int32_t row,
  int32_t* size,
  struct SIDL_int__array** col_ind,
  struct SIDL_double__array** values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.GetRow) */
  /* Insert the implementation of the GetRow method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   printf("impl_Hypre_ParCSRMatrix_GetRow\n");
   printf( "Stub\n");

   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.GetRow) */
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix_Initialize"

int32_t
impl_Hypre_ParCSRMatrix_Initialize(
  Hypre_ParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.Initialize) */
  /* Insert the implementation of the Initialize method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   printf("impl_Hypre_ParCSRMatrix_Initialize\n");

   ierr = HYPRE_IJMatrixInitialize( ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.Initialize) */
}

/*
 * Print the matrix to file.  This is mainly for debugging purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix_Print"

int32_t
impl_Hypre_ParCSRMatrix_Print(
  Hypre_ParCSRMatrix self,
  const char* filename)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.Print) */
  /* Insert the implementation of the Print method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   printf("impl_Hypre_ParCSRMatrix_Print\n");

   ierr = HYPRE_IJMatrixPrint( ij_A, filename);

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.Print) */
}

/*
 * Read the matrix from file.  This is mainly for debugging purposes.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix_Read"

int32_t
impl_Hypre_ParCSRMatrix_Read(
  Hypre_ParCSRMatrix self,
  const char* filename,
  void* comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.Read) */
  /* Insert the implementation of the Read method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   printf("impl_Hypre_ParCSRMatrix_Read\n");

   ierr = HYPRE_IJMatrixRead( filename,
		              *(data -> comm),
		              HYPRE_PARCSR,
		              & ij_A );

   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.Read) */
}

/*
 * Method:  SetCommunicator
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix_SetCommunicator"

int32_t
impl_Hypre_ParCSRMatrix_SetCommunicator(
  Hypre_ParCSRMatrix self,
  void* comm)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.SetCommunicator) */
  /* Insert the implementation of the SetCommunicator method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

#ifdef HYPRE_DEBUG
   printf("impl_Hypre_ParCSRMatrix_SetCommunicator\n");
#endif
   
   data -> comm = (MPI_Comm *) comm;

   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.SetCommunicator) */
}

/*
 * (Optional) Set the max number of nonzeros to expect in each row of
 * the diagonal and off-diagonal blocks.  The diagonal block is the
 * submatrix whose column numbers correspond to rows owned by this
 * process, and the off-diagonal block is everything else.  The arrays
 * {\tt diag\_sizes} and {\tt offdiag\_sizes} contain estimated sizes
 * for each row of the diagonal and off-diagonal blocks, respectively.
 * This routine can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix_SetDiagOffdSizes"

int32_t
impl_Hypre_ParCSRMatrix_SetDiagOffdSizes(
  Hypre_ParCSRMatrix self,
  struct SIDL_int__array* diag_sizes,
  struct SIDL_int__array* offdiag_sizes)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.SetDiagOffdSizes) */
  /* Insert the implementation of the SetDiagOffdSizes method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   printf("impl_Hypre_ParCSRMatrix_SetDiagOffdSizes\n");

   ierr = HYPRE_IJMatrixSetDiagOffdSizes( ij_A, 
                                          SIDLArrayAddr1(diag_sizes, 0), 
                                          SIDLArrayAddr1(offdiag_sizes, 0) );

   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.SetDiagOffdSizes) */
}

/*
 * Method:  SetParameter
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix_SetParameter"

int32_t
impl_Hypre_ParCSRMatrix_SetParameter(
  Hypre_ParCSRMatrix self,
  const char* name,
  double value)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.SetParameter) */
  /* Insert the implementation of the SetParameter method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   printf("impl_Hypre_ParCSRMatrix_SetParameter\n");
   printf( "Stub\n");
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.SetParameter) */
}

/*
 * (Optional) Set the max number of nonzeros to expect in each row.
 * The array {\tt sizes} contains estimated sizes for each row on this
 * process.  This call can significantly improve the efficiency of
 * matrix construction, and should always be utilized if possible.
 * 
 * Not collective.
 * 
 * DEVELOPER NOTES: None.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix_SetRowSizes"

int32_t
impl_Hypre_ParCSRMatrix_SetRowSizes(
  Hypre_ParCSRMatrix self,
  struct SIDL_int__array* sizes)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.SetRowSizes) */
  /* Insert the implementation of the SetRowSizes method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   printf("impl_Hypre_ParCSRMatrix_SetRowSizes\n");

   ierr = HYPRE_IJMatrixSetRowSizes( ij_A, SIDLArrayAddr1(sizes, 0) );

   return( ierr );
  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.SetRowSizes) */
}

/*
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt ncols}
 * and {\tt rows} are of dimension {\tt nrows} and contain the number
 * of columns in each row and the row indices, respectively.  The
 * array {\tt cols} contains the column indices for each of the {\tt
 * rows}, and is ordered by rows.  The data in the {\tt values} array
 * corresponds directly to the column entries in {\tt cols}.  Erases
 * any previous values at the specified locations and replaces them
 * with new ones, or, if there was no value there before, inserts a
 * new one.
 * 
 * Not collective.
 * 
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix_SetValues"

int32_t
impl_Hypre_ParCSRMatrix_SetValues(
  Hypre_ParCSRMatrix self,
  int32_t nrows,
  struct SIDL_int__array* ncols,
  struct SIDL_int__array* rows,
  struct SIDL_int__array* cols,
  struct SIDL_double__array* values)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.SetValues) */
  /* Insert the implementation of the SetValues method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   printf("impl_Hypre_ParCSRMatrix_SetValues\n");

   ierr = HYPRE_IJMatrixSetValues( ij_A,
                            nrows,
                            SIDLArrayAddr1(ncols, 0),
                            SIDLArrayAddr1(rows, 0),
                            SIDLArrayAddr1(cols, 0),
                            SIDLArrayAddr1(values, 0) ); 

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.SetValues) */
}

/*
 * Method:  Setup
 */

#undef __FUNC__
#define __FUNC__ "impl_Hypre_ParCSRMatrix_Setup"

int32_t
impl_Hypre_ParCSRMatrix_Setup(
  Hypre_ParCSRMatrix self)
{
  /* DO-NOT-DELETE splicer.begin(Hypre.ParCSRMatrix.Setup) */
  /* Insert the implementation of the Setup method here... */
   int ierr=0;
   struct Hypre_ParCSRMatrix__data * data;
   HYPRE_IJMatrix ij_A;

   data = Hypre_ParCSRMatrix__get_data( self );

   ij_A = data -> ij_A;

   printf("impl_Hypre_ParCSRMatrix_Setup\n");

   ierr = HYPRE_IJMatrixAssemble( ij_A );

   return( ierr );

  /* DO-NOT-DELETE splicer.end(Hypre.ParCSRMatrix.Setup) */
}
