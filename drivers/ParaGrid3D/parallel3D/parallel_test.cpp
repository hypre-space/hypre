#include <stdio.h>
#include <string.h>
#include <mpi.h>

//============================================================================
//  Compile using: 
//    CC -c -O3 -64 parallel_test.cpp -I /usr/local/mpi/include
//    mpicc -64 -O3 -w -o test parallel_test.o
//    mpirun -np 4 test
//
//============================================================================

int factorial( int n ) {
  if (n) return n*factorial(n-1) ;
  else return 1;
}

int main ( int argc, char *argv[]) {
  
  int myrank, root=0, tmp=0 ; 

  MPI_Status status;
  int tag=99; 

  // Initialize
  MPI_Init( &argc, &argv ) ; 

  MPI_Comm_rank( MPI_COMM_WORLD, &myrank );
  printf( "My factorial  = %d\n", factorial( myrank )  ); fflush(stdout); 

  // Sums all values in myrank and return in tmp
  MPI_Allreduce( &myrank, &tmp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
  
  char msg[20];
  strcpy( msg, "Helooooo");
  if (myrank==1) 
    MPI_Send( msg, strlen(msg)+1, MPI_CHAR, 2, tag, MPI_COMM_WORLD);
  if (myrank==2) {
    MPI_Recv( msg, 20, MPI_CHAR, 1, tag, MPI_COMM_WORLD, &status );
    printf("String received=%s\n", msg);
  }

  if (myrank==root) 
    printf("Sum=%d\n", tmp );

  MPI_Finalize() ;
}
