#include <iostream.h>
#include <stdio.h>
#include <iomanip.h>
#include "Method.h"
#include "Subdomain.h"

#include <mpi.h>

// Used to solve Pressure-Concentration Problem (with pressure_bioscreen.cpp)
int Pressure = 1;
real convection[3];
 
#if OUTPUT_LM     == ON
 FILE *plotLM;               
#endif

//============================================================================

int main(int argc, char *argv[]){ 
  int myrank, np, i, interactive, vizualization, exact, before, after;
  char mesh_file[100], hypre_init[100];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  interactive = vizualization = exact = before = after = 0;
  strcpy(mesh_file, "cube.out");
  strcpy(hypre_init,"hypre");

  //system("clear");
  MPI_Barrier(MPI_COMM_WORLD);
  //if (myrank == 0) 
  //  system("clear");

  if (myrank == 0 && argc == 3){
  cout << endl
       <<"+==============================================================+\n"
       <<"|You started the program with its default arguments :          |\n"
       <<"|mpirun -np " 
       << setw(2)<<np<<" dd -m cube.out -i 0 -v 0 -e 0 -b 0 -a 0 -h hypre|\n"
       <<"+==============================================================+\n"
       <<"| -m cube.out  : input mesh file is cube.out                   |\n"
       <<"| -i 0/1       : non-interactive / interactive mode            |\n"
       <<"| -v 0/1       : visualization off/on                          |\n"
       <<"| -e 0/1       : exact solution not provided/provided          |\n"
       <<"| -b num       : refine num times before splitting the domain  |\n"
       <<"| -a num       : refine num times after splitting the domain   |\n"
       <<"| -h file_name : hypre arguments taken from file file_name     |\n"
       <<"+==============================================================+\n";
  }

  if (argc != 3){
    for(i = 3; i<argc; i+=2){
      if (strcmp("-m", argv[i])==0)
	strcpy(mesh_file, argv[i+1]);
      else if (strcmp("-i", argv[i])==0)
	interactive = atoi(argv[i+1]);
      else if (strcmp("-v", argv[i])==0)
	vizualization = atoi(argv[i+1]);
      else if (strcmp("-e", argv[i])==0)
	exact = atoi(argv[i+1]);
      else if (strcmp("-b", argv[i])==0)
	before = atoi(argv[i+1]);
      else if (strcmp("-a", argv[i])==0)
	after  = atoi(argv[i+1]);
      else if (strcmp("-h", argv[i])==0)
	strcpy(hypre_init,argv[i+1]);
    }
  }
    
  Method *m = new Method(mesh_file);
  //m->Solve();
  if (myrank == 0){
    cout << "Start " << before << " levels of mesh refinement\n";
    cout.flush();
  }
  m->Refine(before, 1);

  idxtype *tr;
  tr = new idxtype[m->GetNTR()];
  if (myrank == 0){
    cout << "Start splitting the domain into " << np << endl;
    cout.flush();
  }
  m->MetisDomainSplit(np, tr);  // split the domain into "np" domains (Metis) 
  
  if (myrank == 0){
    cout << "Start initializing the subdomains\n";
    cout.flush();
  }
  Subdomain dd(myrank, m, tr);
  delete [] tr;

  if (myrank == 0){
    cout << "Start " << after << " more levels of parallel refinement\n";
    cout.flush();
  }
  dd.Refine(after, 1);
  dd.Solve_HYPRE(interactive, vizualization, exact, hypre_init);
  //  dd.Solve();

  MPI_Finalize();
}
//============================================================================
// Bench-mark problem :
// start 2 bioscreen.out functions.cpp
//============================================================================
