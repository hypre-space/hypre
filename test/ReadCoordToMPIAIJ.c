int ReadCoordToMPIAIJ( MPI_Comm comm, Mat *A, char *file_name, int *n )
{
   FILE       *fp;
   int         ierr, i, j;
   Scalar      read_Scalar;
   int        size, nprocs, myrows, mb, me, first_row, last_row;
  


   MPI_Comm_rank( comm, &me);
   MPI_Comm_size( comm, &nprocs);

   fp = fopen(file_name, "r");

   fscanf(fp, "%d", &size);
   *n = size;

   myrows = (size-1)/nprocs+1; mb = myrows;
   if( (nprocs-1) == me) {
     /* Last processor gets remainder */
      myrows = size-(nprocs-1)*mb;
   }

   first_row = me*mb; last_row = first_row+myrows-1;

   dnz = 0; onz = 0; max_row = 0;

   /* Allocate the structure for the Petsc matrix */

   ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,myrows, myrows,
         size,size,0,PETSC_NULL,0,PETSC_NULL,&(*A) ); CHKERRA(ierr);





   while ( fscanf(fp, "%d %d %le",&i, &j, &read_Scalar) != EOF) 
   {
       if( (j>=first_row)&&(j<=last_row) ) 
       {
          ierr = MatSetValues( *A, 1, &i, 1, &j,
                 &read_Scalar, INSERT_VALUES ); CHKERRA(ierr);
        }
   }


   fclose(fp);


   ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
   ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);


   return( 0 );
}
