#include <iostream.h>
#include <stdio.h>
#include "definitions.h"

#include "Method_mixed.h"

int Pressure = 1;
real convection[3];

int main(int argc, char *argv[]){ 
  char fname[100];

  if (argc != 2){
    printf("Start the program with : mixed input_file\n");
    exit(1);
  }
  strcpy(fname, argv[1]);

  MethodMixed m(fname);
  printf("Total volume = %f\n", m.volume());
  m.PrintLocalMatrices();
}


