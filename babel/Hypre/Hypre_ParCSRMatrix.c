
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

/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_ParCSRMatrix_constructor(Hypre_ParCSRMatrix this) {
   this->d_table = (struct Hypre_ParCSRMatrix_private_type *)
      malloc( sizeof( struct Hypre_ParCSRMatrix_private_type ) );

   this->d_table->Hmat = (HYPRE_IJMatrix *) malloc( sizeof( HYPRE_IJMatrix ) );

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_ParCSRMatrix_destructor(Hypre_ParCSRMatrix this) {
   struct Hypre_ParCSRMatrix_private_type *Mp = this->d_table;
   HYPRE_IJMatrix *M = Mp->Hmat;

   HYPRE_IJMatrixDestroy( *M );
   free(this->d_table);

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
   struct Hypre_ParCSRMatrix_private_type *Mp = this->d_table;
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

   xin = (Hypre_ParCSRVector) Hypre_Vector_castTo( x, "Hypre_ParCSRVector" );
   if ( xin==NULL ) return 1;
   HPx = xin->d_table;
   Hx = HPx->Hvec;
   hx = (hypre_IJVector *) *Hx;
   assert( hypre_IJVectorLocalStorage(hx) );
   par_x = hypre_IJVectorLocalStorage(hx);

   yin = (Hypre_ParCSRVector) Hypre_Vector_castTo( *y, "Hypre_ParCSRVector" );
   if ( yin==NULL ) return 1;
   HPy = yin->d_table;
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
   struct Hypre_ParCSRMatrix_private_type * Mp = this->d_table;
   HYPRE_IJMatrix * MIJ = Mp->Hmat;
   hypre_IJMatrix * Mij = (hypre_IJMatrix *) (*MIJ);
   hypre_ParCSRMatrix *parM = hypre_IJMatrixLocalStorage(Mij);
   HYPRE_ParCSRMatrix HYPRE_mat = (HYPRE_ParCSRMatrix) parM;
   return HYPRE_ParCSRMatrixGetDims( HYPRE_mat, m, n );
} /* end impl_Hypre_ParCSRMatrixGetDims */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixGetLocalRange
/* >>>>>>> TO DO: implement this
 **********************************************************/
int impl_Hypre_ParCSRMatrix_GetLocalRange
( Hypre_ParCSRMatrix this, int* row_start, int* row_end,
  int* col_start, int* col_end ) {
   printf("Hypre_ParCSRMatrix_GetLocalRange has not been implemented!\n");
   return 1;
} /* end impl_Hypre_ParCSRMatrixGetLocalRange */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixGetRow
/* >>>>>>> TO DO: implement this
 **********************************************************/
int  impl_Hypre_ParCSRMatrix_GetRow
( Hypre_ParCSRMatrix this, int row, int* size, array1int* col_ind,
  array1double* values ) {
   printf("Hypre_ParCSRMatrix_GetRow has not been implemented!\n");
   return 1;
} /* end impl_Hypre_ParCSRMatrixGetRow */

/* ********************************************************
 * impl_Hypre_ParCSRMatrixRestoreRow
/* >>>>>>> TO DO: implement this
 **********************************************************/
int  impl_Hypre_ParCSRMatrix_RestoreRow
( Hypre_ParCSRMatrix this, int row, int size, array1int col_ind,
  array1double values) {
   printf("Hypre_ParCSRMatrix_RestoreRow has not been implemented!\n");
   return 1;
} /* end impl_Hypre_ParCSRMatrixRestoreRow */


