
/******************************************************
 *
 *  File:  Hypre_ParCSRMatrixBuilder.c
 *
 *********************************************************/

#include "Hypre_ParCSRMatrixBuilder_Skel.h" 
#include "Hypre_ParCSRMatrixBuilder_Data.h" 
#include "Hypre_ParCSRMatrix_Skel.h" 
#include "Hypre_ParCSRMatrix_Data.h" 

#include "HYPRE.h"
#include "Hypre_MPI_Com_Skel.h" 
#include "Hypre_MPI_Com_Data.h" 
#include "HYPRE_IJ_mv.h"
#include "IJ_matrix_vector.h"
#include "Hypre_Partition_Skel.h"
#include "Hypre_PartitionBuilder_Skel.h"
#include "Hypre_Map_Stub.h"

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_ParCSRMatrixBuilder_constructor(Hypre_ParCSRMatrixBuilder this) {
   this->Hypre_ParCSRMatrixBuilder_data =
      (struct Hypre_ParCSRMatrixBuilder_private_type *)
      malloc( sizeof( struct Hypre_ParCSRMatrixBuilder_private_type ) );
   this->Hypre_ParCSRMatrixBuilder_data->newmat = NULL;
   this->Hypre_ParCSRMatrixBuilder_data->matgood = 0;
} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_ParCSRMatrixBuilder_destructor(Hypre_ParCSRMatrixBuilder this) {
   if ( this->Hypre_ParCSRMatrixBuilder_data->newmat != NULL ) {
      Hypre_ParCSRMatrix_deleteReference
         ( this->Hypre_ParCSRMatrixBuilder_data->newmat );
      /* ... will delete newmat if there are no other references to it */
   };
   free(this->Hypre_ParCSRMatrixBuilder_data);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderConstructor
 **********************************************************/
Hypre_ParCSRMatrixBuilder  impl_Hypre_ParCSRMatrixBuilder_Constructor
( Hypre_MPI_Com com, int global_m, int global_n ) {
   return Hypre_ParCSRMatrixBuilder_New();
} /* end impl_Hypre_ParCSRMatrixBuilderConstructor */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilder_Start
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_Start
( Hypre_ParCSRMatrixBuilder this, Hypre_MPI_Com com, int global_m, int global_n ) {

   int ierr = 0;

   struct Hypre_MPI_Com_private_type * HMCp = com->Hypre_MPI_Com_data;
   MPI_Comm * comm = HMCp->hcom;

   struct Hypre_ParCSRMatrix_private_type * Mp;
   HYPRE_IJMatrix * M;
   if ( this->Hypre_ParCSRMatrixBuilder_data->newmat != NULL )
      Hypre_ParCSRMatrix_deleteReference( this->Hypre_ParCSRMatrixBuilder_data->newmat );
   this->Hypre_ParCSRMatrixBuilder_data->newmat = Hypre_ParCSRMatrix_New();
   this->Hypre_ParCSRMatrixBuilder_data->matgood = 0;
   Hypre_ParCSRMatrix_addReference( this->Hypre_ParCSRMatrixBuilder_data->newmat );

   Mp = this->Hypre_ParCSRMatrixBuilder_data->newmat->Hypre_ParCSRMatrix_data;
   M = Mp->Hmat;

   ierr += HYPRE_IJMatrixCreate( *comm, M, global_m, global_n );
   ierr += HYPRE_IJMatrixSetLocalStorageType( *M, HYPRE_PARCSR );

   return ierr;

} /* end impl_Hypre_ParCSRMatrixBuilder_Start */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderSetLocalSize
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_SetLocalSize
( Hypre_ParCSRMatrixBuilder this, int local_m, int local_n ) {
   struct Hypre_ParCSRMatrix_private_type * Mp =
      this->Hypre_ParCSRMatrixBuilder_data->newmat->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * M = Mp->Hmat;

   return HYPRE_IJMatrixSetLocalSize( *M, local_m, local_n );
} /* end impl_Hypre_ParCSRMatrixBuilderSetLocalSize */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderSetRowSizes
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_SetRowSizes
( Hypre_ParCSRMatrixBuilder this, array1int sizes ) {
   struct Hypre_ParCSRMatrix_private_type * Mp =
      this->Hypre_ParCSRMatrixBuilder_data->newmat->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * M = Mp->Hmat;
   const int * size_data = &(sizes.data[*(sizes.lower)]);

   return HYPRE_IJMatrixSetRowSizes( *M, size_data );
} /* end impl_Hypre_ParCSRMatrixBuilderSetRowSizes */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderSetDiagRowSizes
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_SetDiagRowSizes
( Hypre_ParCSRMatrixBuilder this, array1int sizes ) {
   struct Hypre_ParCSRMatrix_private_type * Mp =
      this->Hypre_ParCSRMatrixBuilder_data->newmat->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * M = Mp->Hmat;
   const int * size_data = &(sizes.data[*(sizes.lower)]);

   return HYPRE_IJMatrixSetDiagRowSizes( *M, size_data );
} /* end impl_Hypre_ParCSRMatrixBuilderSetDiagRowSizes */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderSetOffDiagRowSizes
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_SetOffDiagRowSizes
( Hypre_ParCSRMatrixBuilder this, array1int sizes ) {
   struct Hypre_ParCSRMatrix_private_type * Mp =
      this->Hypre_ParCSRMatrixBuilder_data->newmat->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * M = Mp->Hmat;
   const int * size_data = &(sizes.data[*(sizes.lower)]);

   return HYPRE_IJMatrixSetOffDiagRowSizes( *M, size_data );
} /* end impl_Hypre_ParCSRMatrixBuilderSetOffDiagRowSizes */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderInsertRow
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_InsertRow
( Hypre_ParCSRMatrixBuilder this, int n, int row,
  array1int cols, array1double values ) {
   struct Hypre_ParCSRMatrix_private_type * Mp =
      this->Hypre_ParCSRMatrixBuilder_data->newmat->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * M = Mp->Hmat;
   const int * col_data = &(cols.data[*(cols.lower)]);
   const double * value_data = &(values.data[*(values.lower)]);

   return HYPRE_IJMatrixInsertRow( *M, n, row, col_data, value_data );
} /* end impl_Hypre_ParCSRMatrixBuilderInsertRow */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderAddToRow
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_AddToRow
( Hypre_ParCSRMatrixBuilder this, int n, int row,
  array1int cols, array1double values ) {
   struct Hypre_ParCSRMatrix_private_type * Mp =
      this->Hypre_ParCSRMatrixBuilder_data->newmat->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * M = Mp->Hmat;
   const int * col_data = &(cols.data[*(cols.lower)]);
   const double * value_data = &(values.data[*(values.lower)]);

   return HYPRE_IJMatrixAddToRow( *M, n, row, col_data, value_data );
} /* end impl_Hypre_ParCSRMatrixBuilderAddToRow */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderInsertBlock
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_InsertBlock
( Hypre_ParCSRMatrixBuilder this, int m, int n,
  array1int rows, array1int cols, array1double values) {
   struct Hypre_ParCSRMatrix_private_type * Mp =
      this->Hypre_ParCSRMatrixBuilder_data->newmat->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * M = Mp->Hmat;
   const int * row_data = &(rows.data[*(rows.lower)]);
   const int * col_data = &(cols.data[*(cols.lower)]);
   const double * value_data = &(values.data[*(values.lower)]);

   return HYPRE_IJMatrixInsertBlock( *M, m, n, row_data, col_data, value_data );
} /* end impl_Hypre_ParCSRMatrixBuilderInsertBlock */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderAddtoBlock
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_AddtoBlock
( Hypre_ParCSRMatrixBuilder this, int m, int n,
  array1int rows, array1int cols, array1double values ) {
   struct Hypre_ParCSRMatrix_private_type * Mp =
      this->Hypre_ParCSRMatrixBuilder_data->newmat->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * M = Mp->Hmat;
   const int * row_data = &(rows.data[*(rows.lower)]);
   const int * col_data = &(cols.data[*(cols.lower)]);
   const double * value_data = &(values.data[*(values.lower)]);

   return HYPRE_IJMatrixAddToBlock( *M, m, n, row_data, col_data, value_data );
} /* end impl_Hypre_ParCSRMatrixBuilderAddtoBlock */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderGetRowPartitioning
 *
 * The data array in partitioning will be replaced, i.e.
 * the pointer int* partitioning->data will get a different value.
 * This is a potential memory leak.  As we don't know how or whether
 * partitioning->data * was allocated, we can't free it here.
 * That will be the user's job.  The best practice is to not allocate
 * space in the first place, just provide a dummy pointer.
 * >>>>> TO DO: decide whether to copy the data instead. <<<<<
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_GetRowPartitioning
( Hypre_ParCSRMatrixBuilder this, array1int* partitioning ) {
   int ierr = 0;

   struct Hypre_ParCSRMatrix_private_type * Mp =
      this->Hypre_ParCSRMatrixBuilder_data->newmat->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * M = Mp->Hmat;

   int ** const_partitioning_data_1 = &(partitioning->data);
   const int ** const_partitioning_data_2 = (const int **) &(partitioning->data);

   ierr = HYPRE_IJMatrixGetRowPartitioning( *M, const_partitioning_data_2 );
   return ierr;
} /* end impl_Hypre_ParCSRMatrixBuilderGetRowPartitioning */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilder_SetMap
 * WARNING: This function will change the hypre internal partitioning data RowStarts
 * represented by the provided map, but does not change any related data distribution
 * parameters (if any exist), nor does it redistribute the data itself. This should
 * be ok only if called sufficiently early in matrix construction.
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_SetMap
( Hypre_ParCSRMatrixBuilder this, Hypre_Map map ) {
   Hypre_ParCSRMatrix mat = this->Hypre_ParCSRMatrixBuilder_data->newmat;
   HYPRE_IJMatrix * Hmat = mat->Hypre_ParCSRMatrix_data->Hmat;
   hypre_IJMatrix * Mij = (hypre_IJMatrix *) (*Hmat);
   hypre_ParCSRMatrix *parM = hypre_IJMatrixLocalStorage(Mij);
   int * partition_data ;
   Hypre_Partition part = (Hypre_Partition) Hypre_Map_castTo( map, "Hypre.Partition" );
   if ( part==NULL ) {
      printf( "wrong kind of map for Hypre_ParCSRMatrixBuilder_SetMap\n" );
      return 1;
   };
   partition_data = part->Hypre_Partition_data->partition;
   if (this->Hypre_ParCSRMatrixBuilder_data->matgood==1) {
      /* ... error to set partitioning on a fully built matrix */
      /* This check would better be done by a design-by-contract style "Require"
         There are many such cases in this code. */
      return 1;
   }
   else {
      hypre_ParCSRMatrixRowStarts(parM) = partition_data;
      return 0;
      /* I'd rather pass the buck, but the following function doesn't exist... */
      /*    return HYPRE_IJMatrixSetPartitioning( *Hmat, partition_data );*/
   }

   return 1;
} /* end impl_Hypre_ParCSRMatrixBuilder_SetMap */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilder_GetMap
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_GetMap
( Hypre_ParCSRMatrixBuilder this, Hypre_Map* map ) {
   Hypre_ParCSRMatrix mat = this->Hypre_ParCSRMatrixBuilder_data->newmat;
   return Hypre_ParCSRVector_GetMap( mat, map );
} /* end impl_Hypre_ParCSRMatrixBuilder_GetMap */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderSetup
 **********************************************************/
int  impl_Hypre_ParCSRMatrixBuilder_Setup(Hypre_ParCSRMatrixBuilder this) {
/* As nearly as I can tell from looking at sample usage, Initialize happens
   after all the sizes are set, and Assemble happens at the end of the build
   process. I'll try doing them both here. We may need to keep track of three
   states and turn on or off the ability to do things: newly constructed (can
   set sizes not data), initialized (can set data not sizes), completed (can use
   matrix, can't change it).
   */
   int ierr = 0;
   struct Hypre_ParCSRMatrix_private_type * Mp =
      this->Hypre_ParCSRMatrixBuilder_data->newmat->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * M = Mp->Hmat;

   ierr += HYPRE_IJMatrixInitialize( *M );
   ierr += HYPRE_IJMatrixAssemble( *M );
   if ( ierr==0 ) this->Hypre_ParCSRMatrixBuilder_data->matgood = 1;
   return ierr;
} /* end impl_Hypre_ParCSRMatrixBuilderSetup */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixBuilderGetConstructedObject
 **********************************************************/
int impl_Hypre_ParCSRMatrixBuilder_GetConstructedObject
( Hypre_ParCSRMatrixBuilder this, Hypre_LinearOperator *op ) {
   Hypre_ParCSRMatrix newmat = this->Hypre_ParCSRMatrixBuilder_data->newmat;

   if ( newmat==NULL || this->Hypre_ParCSRMatrixBuilder_data->matgood==0 ) {
      printf( "Hypre_ParCSRMatrixBuilder: object not constructed yet\n");
      *op = (Hypre_LinearOperator) NULL;
      return 1;
   };
   *op = (Hypre_LinearOperator) Hypre_ParCSRMatrix_castTo
      ( newmat, "Hypre.LinearOperator" );
   return 0;
} /* end impl_Hypre_ParCSRMatrixBuilderGetConstructedObject */


/* ********************************************************
 * ********************************************************
 *
 * The following functions are not declared in the SIDL file.
 *
 * ********************************************************
 * ********************************************************
 */

/* ********************************************************
 * Hypre_ParCSRMatrixBuilder_New_fromHYPRE
 *
 * Input: M, a pointer to an already-constructed HYPRE_IJMatrix.
 * At a minimum, M represents a call of HYPRE_IJMatrixCreate.
 * This function builds a Hypre_ParCSRMatrix which points to it.
 * The new Hypre_ParCSRMatrix is available by calling
 * GetConstructedObject.
 * There is no need to call Setup or Set functions if that has
 * already been done directly to the HYPRE_IJMatrix.
 **********************************************************/
int Hypre_ParCSRMatrixBuilder_New_fromHYPRE
( Hypre_ParCSRMatrixBuilder this, HYPRE_IJMatrix * M )
{
   int ierr = 0;

   struct Hypre_ParCSRMatrix_private_type * Mp;

   ierr += HYPRE_IJMatrixSetLocalStorageType( *M, HYPRE_PARCSR );

   if ( this->Hypre_ParCSRMatrixBuilder_data->newmat != NULL )
      Hypre_ParCSRMatrix_deleteReference(
         this->Hypre_ParCSRMatrixBuilder_data->newmat );
   this->Hypre_ParCSRMatrixBuilder_data->newmat = Hypre_ParCSRMatrix_New();
   this->Hypre_ParCSRMatrixBuilder_data->matgood = 0;
   Hypre_ParCSRMatrix_addReference( this->Hypre_ParCSRMatrixBuilder_data->newmat );

   Mp = this->Hypre_ParCSRMatrixBuilder_data->newmat->Hypre_ParCSRMatrix_data;
   Mp->Hmat = M;
   this->Hypre_ParCSRMatrixBuilder_data->matgood = 1;

   return ierr;

} /* end impl_Hypre_ParCSRMatrixBuilder_New_fromHYPRE */

/* will appear in next Babel run, probably... */
impl_Hypre_ParCSRMatrixBuilder_GetLocalRange
( Hypre_ParCSRMatrixBuilder this,
  int* row_start,
  int* row_end,
  int* col_start,
  int* col_end
   )
{
      printf( "Hypre_ParCSRMatrixBuilder_GetLocalRange doesn't work. TO DO: implement this\n" );
      return 1;
}

