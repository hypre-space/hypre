#include "definitions.h"
#include <stdio.h>

// This is the problem with one layer. Dirichlet boundary is given for
// y = 0
int Problem1(double x, double y, double z){
  // return Dirichlet;
  if (x < 0.00001|| x>999.9999){
    return Dirichlet;
  }
  else
    return Neumann;
}

//============================================================================
// This is again bioscreen problem. We have one layer and one well. For 
// the pressure equation (which is solved first the pressure is zero 
// on the boundary of the well - given as Dirichlet boundary.
//============================================================================
int Problem2(double x, double y, double z){
  if (x < 0.00001|| x>999.9999  ){
    return Dirichlet;
  }
  else
    if (170.<x && x<230. && 220.<y && y<280. && 190.<z && z<302.)
      return Dirichlet;
    else
      return Neumann;
}


int Lshape(double x, double y, double z){
  if (x > 0.99999 || y > 0.99999 || z > 0.99999)
    return Neumann;
  return Dirichlet;
}

//============================================================================
// Given (x,y,z) on the boundary of the domain this function should say what
// kind of boundary is this (Dirichlet, Neumann or Robin). 
// This function is used in the mesh constructor to find out wether a face
// is on Neumann or Robin boundary (x,y,z is the middle of a face).
//=========================================================================== 
int Type_Boundary(double x, double y, double z){
  //return Lshape(x, y, z);
  return Problem1(x, y, z);
}


