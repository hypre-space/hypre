#include <mpi.h>

//============================================================================
// Calling critical() gives us a critical region. In the region there should 
// be no communication between the processors. It's used for printing on
// the terminal statements by the different processors - the idea is that
// the messages don't get mixed. The execution is from processor 0 to np-1.
//============================================================================
void critical(){
  int myrank, flag;
  MPI_Status Status;
  
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  if (myrank != 0)
    MPI_Recv( &flag, 1, MPI_INT, MPI_ANY_SOURCE,
	      MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
}

//============================================================================
void end_critical(){
  int myrank, flag = 1,  numprocs;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  if (myrank +1 < numprocs)
    MPI_Send(&flag, 1, MPI_INT, myrank+1, flag, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}

//============================================================================
