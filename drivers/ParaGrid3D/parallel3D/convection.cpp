#include <stdio.h>
#include <math.h>
#include "definitions.h"

//============================================================================
// Given the middle point of face on the bdr return the boundary type :
// DIRICHLET, NEUMANN or ROBIN.
//============================================================================
int Type_Boundary(double x, double y, double z){
  if (x < 0.00001 || x>1.9999)
    return DIRICHLET;
  else
    return NEUMANN;
}

//============================================================================
// This function gives the 3 x 3 tensor from "- div( K grad( u))".
// "c" gives the point where the tensor has to be evaluated, "atribut" is the
// element`s attribute, which value is set during the initial mesh generation.
//= ( 2 ) ====================================================================
void func_K(real *c, double K[][3], int atribut){
  K[0][0] = 1.; K[0][1] = 0.; K[0][2] = 0.;
  K[1][0] = 0.; K[1][1] = 1.; K[1][2] = 0.;
  K[2][0] = 0.; K[2][1] = 0.; K[2][2] = 1.;
}

//============================================================================
// The inverse of the above. It`s used in "Zienkiewizs-Zhu" error analysis.
//= ( 3 ) ====================================================================
void func_K_Inv(real *c, double K[][3], int atribut){
  K[0][0] = 1.; K[0][1] = 0.; K[0][2] = 0.;
  K[1][0] = 0.; K[1][1] = 1.; K[1][2] = 0.;
  K[2][0] = 0.; K[2][1] = 0.; K[2][2] = 1.;
}

//============================================================================
// This is the reaction term.
//= ( 4 ) ====================================================================
double func_c(real *c){
  return 0.;
}

//============================================================================
// The source term.
//= ( 5 ) ====================================================================
double func_f(real *c){
  double x = c[0], y = c[1], z = c[2];
  return (sin(x*(y+1)*(z+2))*(y+1)*(y+1)*(z+2)*(z+2) +
	  sin(x*(y+1)*(z+2))*x*x*(z+2)*(z+2) + 
	  sin(x*(y+1)*(z+2))*x*x*(y+1)*(y+1)
	  + 100.*cos(x*(y+1)*(z+2))*(y+1)*(z+2)
	  + 10.*cos(x*(y+1)*(z+2))*x*(z+2)
	  + 0.*cos(x*(y+1)*(z+2))*x*(y+1));
}

//============================================================================
// Dirichlet boundary value
//= ( 6 ) ====================================================================
double func_u0(real *c){
  return sin(c[0]*(c[1]+1)*(c[2]+2));
}

//============================================================================
// Neumann boundary value
//= ( 7 ) ====================================================================
double func_gn(real *c){
  return 0.;
}

//============================================================================
// The exact solution if "#if EXACT == ON"
//= ( 8 ) ====================================================================
double   exact(real *c){
  double x = c[0], y = c[1], z = c[2];
  
  return sin(c[0]*(c[1]+1)*(c[2]+2));
}

//============================================================================
// if CONCENTRATION == OFF && CONVECTION == ON
//= ( 9 ) ====================================================================
void func_b(real *c, real b[3]){
  b[0] = 100.;
  b[1] = 10.;
  b[2] = 0.;
}

//============================================================================
// "#if WELL == ON" - return 1 (point belong to well) or 0 (point not in well)
// Put the wells and their characteristics in file "Method.cpp".
//= ( 10 ) ===================================================================
int locate_well(int Atribut, real *coord){}

//============================================================================
// array is of dimension the number of elements (NTR). When function refine-
// ment is specified the initialization of array will determine which elements
// to be refined (1 for refine, 0 for no-refine).
//= ( 11 ) ===================================================================
void Refine_F(int *array, int level, int SN, int NTR, vector<vertex> &Z, 
		vector<tetrahedron> &TR){
  int i;
  double coord[3];

  for(i=0; i<NTR; i++){
    TR[i].GetMiddle(coord, Z);
    if (coord[0] < 0.5)
      array[i] = 1;
    else
      array[i] = 0;
  }
}

//============================================================================
//============================================================================

