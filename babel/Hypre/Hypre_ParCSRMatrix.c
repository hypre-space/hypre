
/******************************************************
 *
 *  File:  Hypre_ParCSRMatrix.c
 *
 *********************************************************/

#include "Hypre_ParCSRMatrix_Skel.h" 
#include "Hypre_ParCSRMatrix_Data.h" 

#include <assert.h>
#include "Hypre_ParCSRVector_Skel.h" 
#include "Hypre_ParCSRVector_Data.h" 
#include "IJ_matrix_vector.h"
#include "Hypre_Partition_Skel.h"
#include "Hypre_PartitionBuilder_Skel.h"
#include "Hypre_Map_Stub.h"

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_ParCSRMatrix_constructor(Hypre_ParCSRMatrix this) {
   this->Hypre_ParCSRMatrix_data = (struct Hypre_ParCSRMatrix_private_type *)
      malloc( sizeof( struct Hypre_ParCSRMatrix_private_type ) );

   this->Hypre_ParCSRMatrix_data->Hmat = (HYPRE_IJMatrix *)
      malloc( sizeof( HYPRE_IJMatrix ) );

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_ParCSRMatrix_destructor(Hypre_ParCSRMatrix this) {
   struct Hypre_ParCSRMatrix_private_type *Mp = this->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix *M = Mp->Hmat;

   HYPRE_IJMatrixDestroy( *M );
   free(this->Hypre_ParCSRMatrix_data);

} /* end destructor */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixApply
 * y = A*x where this=A
 **********************************************************/
int  impl_Hypre_ParCSRMatrix_Apply
(Hypre_ParCSRMatrix this, Hypre_Vector x, Hypre_Vector* y) {
   int ierr = 0;
/* We just extract the relevant hypre_ParCSRMatrix and hypre_ParVector's, and
   call par_csr_matvec.c:hypre_ParCSRMatrixMatvec. It should stick the new data
   into y (its y is the same as the y here).
   There are lots of places where there could be more defensive coding,
   asserts, requires, etc.
*/
   struct Hypre_ParCSRMatrix_private_type *Mp = this->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix *HM = Mp->Hmat;
   hypre_IJMatrix *hM = (hypre_IJMatrix *) (*HM);
   hypre_ParCSRMatrix *parM;

   Hypre_ParCSRVector xin;
   struct Hypre_ParCSRVector_private_type *HPx;
   HYPRE_IJVector *Hx;
   hypre_IJVector *hx;
   hypre_ParVector *par_x;

   Hypre_ParCSRVector yin;
   struct Hypre_ParCSRVector_private_type *HPy;
   HYPRE_IJVector *Hy;
   hypre_IJVector *hy;
   hypre_ParVector *par_y;

   assert(hypre_IJMatrixLocalStorage(hM));
   parM = hypre_IJMatrixLocalStorage(hM);

   xin = (Hypre_ParCSRVector) Hypre_Vector_castTo( x, "Hypre.ParCSRVector" );
   if ( xin==NULL ) return 1;
   HPx = xin->Hypre_ParCSRVector_data;
   Hx = HPx->Hvec;
   hx = (hypre_IJVector *) *Hx;
   assert( hypre_IJVectorLocalStorage(hx) );
   par_x = hypre_IJVectorLocalStorage(hx);

   yin = (Hypre_ParCSRVector) Hypre_Vector_castTo( *y, "Hypre.ParCSRVector" );
   if ( yin==NULL ) return 1;
   HPy = yin->Hypre_ParCSRVector_data;
   Hy = HPy->Hvec;
   hy = (hypre_IJVector *) *Hy;
   assert( hypre_IJVectorLocalStorage(hy) );
   par_y = hypre_IJVectorLocalStorage(hy);

   ierr += hypre_ParCSRMatrixMatvec( 1.0, parM, par_x, 0.0, par_y );

   return ierr;
} /* end impl_Hypre_ParCSRMatrixApply */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixGetDims
 **********************************************************/
int impl_Hypre_ParCSRMatrix_GetDims(Hypre_ParCSRMatrix this, int* m, int* n) {
   struct Hypre_ParCSRMatrix_private_type * Mp = this->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * MIJ = Mp->Hmat;
   hypre_IJMatrix * Mij = (hypre_IJMatrix *) (*MIJ);
   hypre_ParCSRMatrix *parM = hypre_IJMatrixLocalStorage(Mij);
   HYPRE_ParCSRMatrix HYPRE_mat = (HYPRE_ParCSRMatrix) parM;
   return HYPRE_ParCSRMatrixGetDims( HYPRE_mat, m, n );
} /* end impl_Hypre_ParCSRMatrixGetDims */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixGetLocalRange
 **********************************************************/
int impl_Hypre_ParCSRMatrix_GetLocalRange
( Hypre_ParCSRMatrix this, int* row_start, int* row_end,
  int* col_start, int* col_end ) {
   struct Hypre_ParCSRMatrix_private_type * Mp = this->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * MIJ = Mp->Hmat;
   hypre_IJMatrix * Mij = (hypre_IJMatrix *) (*MIJ);
   hypre_ParCSRMatrix *parM = hypre_IJMatrixLocalStorage(Mij);

   return hypre_ParCSRMatrixGetLocalRange
      ( parM, row_start, row_end, col_start, col_end );

} /* end impl_Hypre_ParCSRMatrixGetLocalRange */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixGetRow
 * This calls a HYPRE function to return the column indices and values of
 * nonzero matrix elements in the given row.  The HYPRE function expects to get
 * the global row number but will return an error if the row be not available
 * on the local processor.
 *
 * How the memory management works: we call hypre_ParCSRMatrixGetRow, which
 * on the first call allocates two arrays to use for the return values, just once
 * for all calls of of the function.  Each time it is called, it copies data into
 * these two arrays, and sets the provided pointers to point to these arrays.
 * The arrays are locked (to prevent other GetRow calls from overwriting the data)
 * until the the user calls RestoreRow to signal that he doesn't need to look at
 * the data any more.
 *
 * Typical use (in pseudo-code) :
 * GetRow( out column_indices, out values )
 * allocate my data structures
 * copy data to my data structures
 * RestoreRow()
 * use row data
 * free my data structures
 *
 **********************************************************/
int  impl_Hypre_ParCSRMatrix_GetRow
( Hypre_ParCSRMatrix this, int row, int* size,
  array1int* col_ind, array1double* values ) {
   int ierr = 0;
   struct Hypre_ParCSRMatrix_private_type * Mp = this->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * MIJ = Mp->Hmat;
   hypre_IJMatrix * Mij = (hypre_IJMatrix *) (*MIJ);
   hypre_ParCSRMatrix *parM = hypre_IJMatrixLocalStorage(Mij);
   HYPRE_ParCSRMatrix HYPRE_mat = (HYPRE_ParCSRMatrix) parM;

   ierr += HYPRE_ParCSRMatrixGetRow
      ( HYPRE_mat, row, size, &((*col_ind).data), &((*values).data) );

   /* c.f. par_csr_matrix.c: the arrays returned above are zero-based .. */
   (*col_ind).lower[0] = 0;   (*col_ind).upper[0] = *size;
   (*values).lower[0] = 0;   (*values).upper[0] = *size;
   
   return ierr;
} /* end impl_Hypre_ParCSRMatrixGetRow */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixRestoreRow
 * This just calls the HYPRE RestoreRow function, which does nothing
 * but reset an active flag to allow future access to a row.
 **********************************************************/
int  impl_Hypre_ParCSRMatrix_RestoreRow
( Hypre_ParCSRMatrix this, int row, int size,
  array1int col_ind, array1double values )
{
   int ierr = 0;
   struct Hypre_ParCSRMatrix_private_type * Mp = this->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * MIJ = Mp->Hmat;
   hypre_IJMatrix * Mij = (hypre_IJMatrix *) (*MIJ);
   hypre_ParCSRMatrix *parM = hypre_IJMatrixLocalStorage(Mij);
   HYPRE_ParCSRMatrix HYPRE_mat = (HYPRE_ParCSRMatrix) parM;

   ierr += HYPRE_ParCSRMatrixRestoreRow( HYPRE_mat, row, &size,
                                         &(col_ind.data), &(values.data) );

   return ierr;
} /* end impl_Hypre_ParCSRMatrixRestoreRow */


/* ********************************************************
 * impl_Hypre_ParCSRMatrix_GetMap
 **********************************************************/
int  impl_Hypre_ParCSRMatrix_GetMap(Hypre_ParCSRMatrix this, Hypre_Map* map) {

/* ********************************************************
 * The data array in map will be replaced, i.e.
 * the pointer int* map->Hypre_Partition_Data->data will get a different value.
 * This is a potential memory leak.  As we don't know how or whether
 * this data pointer was allocated, we can't free it here.
 * That will be the user's job.  The best practice is to not allocate
 * space in the first place, just provide a dummy pointer.
 * >>>>> TO DO: decide whether to copy the data instead. <<<<<
 **********************************************************/
   Hypre_Partition part;
   Hypre_PartitionBuilder partBldr;
   array1int* partitioning;
   array1int partarray;
   int ierr = 0;
   struct Hypre_ParCSRMatrix_private_type * Mp = this->Hypre_ParCSRMatrix_data;
   HYPRE_IJMatrix * M = Mp->Hmat;
   int * new_data_p;
   int ** new_data = &new_data_p;

   int ** const_partitioning_data_1 = &(partitioning->data);
   const int ** const_partitioning_data_2 = (const int **) &(partitioning->data);

   int num_procs = 1;
/* The following would give a correct size for num_procs and hence the partition
   array, if a Hypre_MPI_Com object were available.  It's not, but in most cases
   we can get by because only the data pointer in the map object is looked at
   later.  This is a potential bug. */
/*
   MPI_Comm * MCp = comm->Hypre_MPI_Com_data->hcom;
   int num_procs;
   MPI_Comm_size( *MCp, &num_procs );
*/

   if ( (*map)==NULL ) { /* make one ... */
      partarray.lower[0] = 0;
      partarray.upper[0] = num_procs;
      partarray.data = *new_data;
      partBldr = Hypre_PartitionBuilder_New();
      ierr += Hypre_PartitionBuilder_Start( partBldr, partarray );
      ierr += Hypre_PartitionBuilder_Setup( partBldr );
      ierr += Hypre_PartitionBuilder_GetConstructedObject( partBldr, map );
      Hypre_Map_addReference( (*map) );
      Hypre_PartitionBuilder_deleteReference( partBldr );
   }

   part = (Hypre_Partition) Hypre_Map_castTo( *map, "Hypre.Partition" );
   if ( part==NULL ) {
      printf( "wrong kind of map for Hypre_ParCSRVectorBuilder_SetMap\n" );
      return 1;
   };
   (partitioning->lower)[0] = part->Hypre_Partition_data->lower;
   (partitioning->upper)[0] = part->Hypre_Partition_data->upper;
   (partitioning->data) = part->Hypre_Partition_data->partition;
   const_partitioning_data_2 = (const int **) &(partitioning->data);

   ierr += HYPRE_IJMatrixGetRowPartitioning( *M, const_partitioning_data_2 );

} /* end impl_Hypre_ParCSRMatrix_GetMap */

