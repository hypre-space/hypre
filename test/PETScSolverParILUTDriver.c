/* Include headers for external packages */
#include "HYPRE.h"
#include "sles.h"
#include "../utilities/memory.h"

#include <string.h>

#ifdef HYPRE_DEBUG
/* debugging header */
#include <cegdb.h>
#endif

int ReadCoordToMPIAIJ( MPI_Comm comm, Mat *A, char *file_name, int *n );

int main(int argc,char **args)
{
  HYPRE_PETScSolverParILUT solver;
  Mat         A;
  Vec         x, b;
  char       *problem;
  char        dlamch_arg[1]="e";
  char        file_name[255];
  int         ierr, n;
  int         myid;
  double      one = 1.0, minus_1 = -1.0, zero = 0.0;
  double      norm_A, scaled_resid, norm_resid, norm_soln, eps;
  double      dlamch_( char *arg1, int arg2);



  /* Initialize Petsc */
  PetscInitialize(&argc,&args,(char *)0,NULL);

  MPI_Comm_rank(MPI_COMM_WORLD, &myid );

#ifdef HYPRE_DEBUG
  cegdb(&argc, &args, myid);
#endif

  if (argc < 2)
    {
      fprintf(stderr, "Usage: <driver_program_name> <problem> [Petsc options]\n");
      exit(1);
    }

  problem = args[1];
  strcpy( file_name, problem);
  
  /* Read in matrix and vectors and put in Petsc format */
  /* Create and assemble matrix */
  ierr = ReadCoordToMPIAIJ( MPI_COMM_WORLD, &A, file_name, &n ); 
  CHKERRA( ierr );
  strcpy( file_name, problem);
  PetscPrintf(MPI_COMM_WORLD, "Finished reading in matrix\n");

  /* Create vectors */
  ierr = VecCreateMPI( MPI_COMM_WORLD, PETSC_DECIDE, n, &b ); CHKERRA( ierr );
  ierr = VecSet( &one, b );  CHKERRA( ierr );

  ierr = VecDuplicate (b, &x ); CHKERRA( ierr );
  ierr = VecSet ( &zero, x); CHKERRA( ierr );

  /* Solver structure */
  solver = HYPRE_NewPETScSolverParILUT (MPI_COMM_WORLD );
  ierr = HYPRE_PETScSolverParILUTInitialize( solver ); CHKERRA( ierr );
  ierr = HYPRE_PETScSolverParILUTSetSystemMatrix ( solver, A );
  ierr = HYPRE_PETScSolverParILUTSetFactorRowSize ( solver, 8 );
  ierr = HYPRE_PETScSolverParILUTSetDropTolerance( solver, 0.000001 );

  /* call setup of routine */
  PetscPrintf(MPI_COMM_WORLD, "calling setup routine\n");
  ierr = HYPRE_PETScSolverParILUTSetup( solver, x, b ); CHKERRA( ierr );

  /* call solve routine */
  PetscPrintf(MPI_COMM_WORLD, "calling solver routine\n");
  ierr = HYPRE_PETScSolverParILUTSolve( solver, x, b ); CHKERRA( ierr );

  /* Output */
  /*
  ierr = VecView ( x, 0 );
  */

  /* Cleanup after routine */
  ierr = HYPRE_FreePETScSolverParILUT( solver ); CHKERRA(ierr);

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);  
  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
}

int ReadCoordToMPIAIJ( MPI_Comm comm, Mat *A, char *file_name, int *n )
     /* ASSUMES file stores coordinates in 1-based indexing */
{
   FILE       *fp;
   int         ierr, i, j;
   Scalar      read_Scalar;
   int         *onzz, *dnzz;
   int         size, nprocs, myrows, mb, me, first_row, last_row;
  


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

   /* initial scan to get row lengths */
   onzz = hypre_CTAlloc( int, myrows );
   dnzz = hypre_CTAlloc( int, myrows );

   while ( fscanf(fp, "%d %d %le",&i, &j, &read_Scalar) != EOF) 
   {
       i --; j --;
       if( (i>=first_row)&&(i<=last_row) ) 
       {
         if( (j>=first_row)&&(j<=last_row) ) 
         {
            dnzz[i-first_row] ++;
          } else
         {
            onzz[i-first_row] ++;
          }
       }
   }


   fclose(fp);

   /* Allocate the structure for the Petsc matrix */

   if(nprocs>1)
   {
     ierr = MatCreateMPIAIJ( comm,myrows, myrows,
         size,size,0,dnzz,0,onzz,&(*A) ); CHKERRA(ierr);
   } else
   {
     ierr = MatCreateSeqAIJ(MPI_COMM_SELF,
         size,size,0,dnzz,&(*A) ); CHKERRA(ierr);
   }

   hypre_TFree( dnzz );
   hypre_TFree( onzz );

   /* Fill in values */
   fp = fopen(file_name, "r");

   fscanf(fp, "%d", &size);
   *n = size;

   while ( fscanf(fp, "%d %d %le",&i, &j, &read_Scalar) != EOF) 
   {
       i --; j --;
       if( (i>=first_row)&&(i<=last_row) ) 
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
