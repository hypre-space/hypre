
// THIS IS A C++ FILE, since it needs to call ISIS++ with objects

#include "iostream.h"
#include "RowMatrix.h"  // ISIS++ header file
#include "./distributed_matrix.h"

extern "C" {

typedef struct
{
    int *ind;
    double *val;
} 
RowBuf;

/*--------------------------------------------------------------------------
 * hypre_InitializeDistributedMatrixISIS
 *--------------------------------------------------------------------------*/

// matrix must be set before calling this function

int 
hypre_InitializeDistributedMatrixISIS(hypre_DistributedMatrix *dm)
{
   RowMatrix *mat = (RowMatrix *) hypre_DistributedMatrixLocalStorage(dm);
   int num_rows = mat->getMap().numLocalRows();
   
   hypre_DistributedMatrixM(dm) = num_rows;
   hypre_DistributedMatrixN(dm) = num_rows;

   // allocate space for row buffers

   RowBuf *rowbuf = new RowBuf;
   rowbuf->ind = new int[num_rows];
   rowbuf->val = new double[num_rows];

   dm->auxiliary_data = (void *) rowbuf;

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_FreeDistributedMatrixISIS
 *--------------------------------------------------------------------------*/

int 
hypre_FreeDistributedMatrixISIS( hypre_DistributedMatrix *dm)
{
   RowBuf *rowbuf = (RowBuf *) dm->auxiliary_data;

   delete rowbuf->ind;
   delete rowbuf->val;
   delete rowbuf;

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_PrintDistributedMatrixISIS
 *--------------------------------------------------------------------------*/

int 
hypre_PrintDistributedMatrixISIS( hypre_DistributedMatrix *matrix )
{
   cout << "hypre_PrintDistributedMatrixISIS not implemented" << endl;

   return -1;
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixLocalRangeISIS
 *--------------------------------------------------------------------------*/

int 
hypre_GetDistributedMatrixLocalRangeISIS( hypre_DistributedMatrix *dm,
                             int *start,
                             int *end )
{
   RowMatrix *mat = (RowMatrix *) hypre_DistributedMatrixLocalStorage(dm);

   *start = mat->getMap().startRow() - 1;  // convert to 0-based
   *end = mat->getMap().endRow(); // endRow actually returns 1 less
   
   cout << "LocalRangeISIS " << *start << "  " << *end << endl;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_GetDistributedMatrixRowISIS
 *--------------------------------------------------------------------------*/

// semantics: buffers returned will be overwritten on next call to 
// this get function

int 
hypre_GetDistributedMatrixRowISIS( hypre_DistributedMatrix *dm,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   RowMatrix *mat = (RowMatrix *) hypre_DistributedMatrixLocalStorage(dm);
   RowBuf *rowbuf;
   int i, temp;

   rowbuf = (RowBuf *) dm->auxiliary_data;

   // set pointers to local buffers
   if (col_ind != NULL)
   {
       int *p;

       mat->getRow(row+1, temp, rowbuf->ind);
       *size = temp;
       *col_ind = rowbuf->ind;

       // need to convert to 0-based indexing for output
       for (i=0, p=*col_ind; i<temp; i++, p++)
	   (*p)--;
   }

   if (values != NULL)
   {
       mat->getRow(row+1, temp, rowbuf->val);
       *values = rowbuf->val;
       *size = temp;
   }

   // should try to use this if possible
   // mat->getRow(row+1, *size, *values, *col_ind);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_RestoreDistributedMatrixRowISIS
 *--------------------------------------------------------------------------*/

int 
hypre_RestoreDistributedMatrixRowISIS( hypre_DistributedMatrix *dm,
                             int row,
                             int *size,
                             int **col_ind,
                             double **values )
{
   // does nothing, since we use local buffers

   return 0;
}

} // extern "C"
