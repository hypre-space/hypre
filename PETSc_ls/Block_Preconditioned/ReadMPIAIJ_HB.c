#include "sles.h"
#include <stdio.h>
#include "headers.h"

int ConHB2MPIAIJ ( Mat *A, char *file_name )
{
  FILE       *fp;
  int         i, j;
  Scalar      *data;
   int        *ia;
   int        *ja;
   int        size;
   

  ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,
         0,PETSC_NULL,0,PETSC_NULL,&(*A) ); CHKERRA(ierr);

   fp = fopen(file_name, "r");
   /* read in junk line */
   fscanf(fp, "%*[^\n]\n");

   fscanf(fp, "%d", &size);

   ia = ctalloc(int, size+1);
   for (j = 0; j < size+1; j++)
      fscanf(fp, "%d", &ia[j]);

   ja = ctalloc(int, ia[size]-1);
   for (j = 0; j < ia[size]-1; j++)
      fscanf(fp, "%d", &ja[j]);

   data = ctalloc(sizeof (Scalar), ia[size]-1);
   for (j = 0; j < ia[size]-1; j++)
      fscanf(fp, "%le", &data[j]);

   fclose(fp);

  /* put the values in with a series of calls to mat set values */
   for (j = 0; j < size; j++) {
      ierr = MatSetValues( A, 1, &j, ia[j+1]-ia[j], &ja[ia[j]], 
                           &data[ia[j]], INSERT_VALUES );
   }

  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRA(ierr);


  return();
}
