#include <iostream.h>
#include <stdio.h>
#include "Mesh.h"
#include "Method.h"


main(){ 
  char fname[100];

  cout << "\nInput the file name : "; cin >> fname;
  Method m(fname);
 
  m.Solve();
  /*
  int *array = new int[10*m.GetNTR()]; 
  for(int i=0; i<m.GetNTR(); i++) array[i] = 0;
  for(int i=0; i<m.GetNTR(); i++) array[i] = 1;
  // for(int i=0; i<50; i++) array[i] = 1;
  //array[0] = 1;
  m.LocalRefine( array);

  
  for(int i=0; i<m.GetNTR(); i++) array[i] = 0;
  for(int i=0; i<m.GetNTR(); i++) array[i] = 1;
  //  m.LocalRefine( array);
  
  double *sol = new double[m.GetNN(1)];

  m.PrintMesh("MeshOutput");
  for(int i=0; i<m.GetNN(1); i++) sol[i] = 0.;sol[0] = 1.;
  m.PrintSol("/home/tomov/Visual/Visualization.off", sol);
  
  for(int i=0; i<m.GetNTR(); i++){
    printf("Tetrahedron %d\n", i);
    m.PrintTetrahedron(i);
  }
  */
}
