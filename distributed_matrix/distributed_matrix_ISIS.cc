
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

   const Map& map = mat->getMap();

   int num_rows = mat->getMap().n();
   int num_cols = mat->getMap().n();
   
   hypre_DistributedMatrixM(dm) = num_rows;
   hypre_DistributedMatrixN(dm) = num_cols;

   // allocate space for row buffers

   RowBuf *rowbuf = new RowBuf;
   rowbuf->ind = new int[num_cols];
   rowbuf->val = new double[num_cols];

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

   mat->getRow(row+1, temp, rowbuf->val, rowbuf->ind);

   // add diagonal element if necessary
   {
       int *p;
       int found = 0;

       for (i=0, p=rowbuf->ind; i<temp; i++, p++)
       {
	   if (*p == row+1) 
	       found = 1;
       }

       if (!found)
       {
	   rowbuf->ind[temp] = row+1;
	   rowbuf->val[temp] = 1.; // pick a value
	   temp++;
       }
   }

   // set pointers to local buffers
   if (col_ind != NULL)
   {
       int *p;

       *size = temp;
       *col_ind = rowbuf->ind;

       // need to convert to 0-based indexing for output
       for (i=0, p=*col_ind; i<temp; i++, p++)
	   (*p)--;
   }

   if (values != NULL)
   {
       *values = rowbuf->val;
       *size = temp;
   }

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
