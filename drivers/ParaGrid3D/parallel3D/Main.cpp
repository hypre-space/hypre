#include <iostream.h>
#include <stdio.h>
#include "definitions.h"

#if PROBLEM == 3
  #include "Method_mixed.h"
#else
  #include "Method.h"
#endif

//#include "debug.h"

// Used to solve Pressure-Concentration Problem (with pressure_bioscreen.cpp)
#if CONCENTRATION == ON
 int Pressure = 1;
 real convection[3];
#endif

#if OUTPUT_LM     == ON
 FILE *plotLM;               
#endif


int main(int argc, char *argv[]){ 
  char fname[100];
  int Refinement;

  if (argc != 2){
    printf("Start the program with : programa input_file\n");
    exit(1);
  }
  strcpy(fname, argv[1]);

  #if PROBLEM == 3
    MethodMixed m(fname);
    printf("Total volume = %f\n", m.volume());
    Refinement = 3;
    m.Solve(Refinement);
  #else
    Method m(fname);
    printf("Total volume = %f\n", m.volume());

    // m.PrintMetis(NULL);
    // m.PrintMeshMaple("maple.off");
    // m.PrintMesh();
    // m.UniformRefine();
    // m.Create_level_matrices();

    cout << "Input the desired refinement (1:ZZ, 2:F, 3:Uni): ";
    cin  >> Refinement; 
    m.Solve(Refinement);
  #endif
}


