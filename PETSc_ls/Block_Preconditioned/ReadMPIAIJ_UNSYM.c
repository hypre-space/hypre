/* Include headers for problem and solver data structure */
#include "BlockJacobiPcKsp.h"

/* Since this is building a Petsc specific matrix, include needed Petsc incs */
#include "matrix.h"

int ReadMPIAIJ_UNSYM( Mat *A, char *file_name, int *n )
{
   FILE       *fp;
   int         ierr, i, j;
   Scalar      *data, read_Scalar;
   int        *ia;
   int        *ja;
   int        size, nprocs, myrows, mb, me, first_row, last_row;
   int        dnz, onz, read_int, ja_indx, temp_indx, max_row;
   


   MPI_Comm_rank( MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   fp = fopen(file_name, "r");
   /* read in junk line */
   fscanf(fp, "%*[^\n]\n");

   fscanf(fp, "%d", &size);
   *n = size;

   myrows = (size-1)/nprocs+1; mb = myrows;
   if( (nprocs-1) == me) {
     /* Last processor gets remainder */
      myrows = size-(nprocs-1)*mb;
   }

   first_row = me*mb; last_row = first_row+myrows-1;

   dnz = 0; onz = 0; max_row = 0;

   ia = ctalloc(int, size+1);
   for (j = 0; j < size+1; j++) {
      fscanf(fp, "%d", &read_int);
      ia[j] = read_int-1;
   }


   ja = ctalloc(int, ia[last_row+1]-ia[first_row]);
   ja_indx = 0;

   for (j = 0; j < size; j++) {
      for (i = ia[j]; i < ia[j+1]; i++) {
         max_row = max(max_row, ia[j+1]-ia[j]);
         fscanf(fp, "%d", &read_int);
         if( (j>=first_row)&&(j<=last_row) ) {
            ja[ja_indx++] = read_int-1;
	    if( (read_int-1>=first_row)&&(read_int-1<=last_row) ) {
               dnz++;
            } else {
               onz++;
	    }
         }
      }
   }


   /* Allocate the structure for the Petsc matrix */

   /* Turn off INODE option in Petsc. This is needed so that internal
      Petsc storage will be compatible with ILU. */
   /* No longer needed... -AC, 6/97 
   ierr = OptionsSetValue( "-mat_aij_no_inode", PETSC_NULL ); CHKERRA(ierr);
   */

   /* Set internal Petsc storage to use index-1 indexing. This is needed so that internal
      Petsc storage will be compatible with ILU. */
   ierr = OptionsSetValue( "-mat_aij_oneindex", PETSC_NULL ); CHKERRA(ierr);


   ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,myrows, myrows,
         size,size,NDIMA(dnz)/myrows+1,PETSC_NULL,0,PETSC_NULL,&(*A) ); CHKERRA(ierr);


   /* put the values in by rows with a series of calls to MatSetValues */
   data = ctalloc( Scalar, max_row);
   ja_indx = 0;

   for (j = first_row; j <= last_row; j++) {
      temp_indx = 0;
      for (i = ia[j]; i < ia[j+1]; i++) {
         fscanf(fp, "%le", &read_Scalar);
         if( (j>=first_row)&&(j<=last_row) ) {
            data[temp_indx++] = read_Scalar;
         }
      }
      ierr = MatSetValues( *A, 1, &j, temp_indx, &ja[ja_indx],
                           data, INSERT_VALUES ); CHKERRA(ierr);
      ja_indx += ia[j+1]-ia[j];
      if ((j/1000)*1000 == j) 
        {
        printf("%d\n",j);
        }
   }

   fclose(fp);


   ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
   ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);


   return( 0 );
}
