#include <stdio.h>
#include <math.h>
#include "definitions.h"

extern real convection[3];      // used for the velocity
extern int Pressure;            // 1 if pressure eqns are being solved


//============================================================================
// Given the middle point of face on the bdr return the boundary type :
// DIRICHLET, NEUMANN or ROBIN.
//============================================================================
int Type_Boundary(double x, double y, double z){
  if (x < 0.00001 || x > 999.9999)
    return DIRICHLET;
  else
    return NEUMANN;
}

//============================================================================
// This problem determines the pressure (from where the velocity) for mesh in
// bioscreen.out, Mesh_functions in Problem1. The mesh has layer, marked with
// atribut 1 or 2.
//============================================================================
void func_K(real *c, real K[][3], int atribut){
  if (Pressure){                        // Solve the pressure
    double eps;

    if (atribut == 1 || atribut == 2)   // this element is in the layer
      eps = 16.;
    else eps = 32.;

    K[0][0] = eps;  K[0][1] =  0.;  K[0][2] =  0.;
    K[1][0] =  0.;  K[1][1] = eps;  K[1][2] =  0.;
    K[2][0] =  0.;  K[2][1] =  0.;  K[2][2] = eps;
  }
  else {
    double l_l, l_t, l = 0.0001, len;
    l_l = 21.; l_t = 2.1;

    double v11 = convection[0]*convection[0],
      v22 = convection[1]*convection[1],
      v33 = convection[2]*convection[2],
      v12 = convection[0]*convection[1],
      v13 = convection[0]*convection[2],
      v23 = convection[1]*convection[2];
    len = sqrt(v11 + v22 + v33);        // the length
    //fprintf(stderr,"(%f, %f, %f), length = %f\n", convection[0],
    //	    convection[1], convection[2], len);
    if (len < 0.00000001){              // no dispersion
      K[0][0] = K[1][1] = K[2][2] = l; 
      K[0][1] = K[1][0] = K[0][2] = K[2][0] = K[1][2] = K[2][1] = 0.; 
      return;
    }
  
    double c1 = l_l/len, c2 = l_t/len;
  
    K[0][0]=c1*v11+c2*(v22+v33)+l;K[0][1]=c1*v12-c2*v12;K[0][2]=c1*v13-c2*v13;
    K[1][0]=c1*v12-c2*v12;K[1][1]=c1*v22+c2*(v11+v33)+l;K[1][2]=c1*v23-c2*v23;
    K[2][0]=c1*v13-c2*v13;K[2][1]=c1*v13-c2*v13;K[2][2]=c1*v33+c2*(v11+v22)+l;
  }
}

//============================================================================

void func_K_Inv(real *c, real K[][3], int atribut){
  if (Pressure){
    double eps;
    if (atribut == 1 || atribut == 2)
      eps = 1/16.;
    else
      eps = 1/32.;

    K[0][0] =eps;  K[0][1] = 0.;  K[0][2] = 0.;
    K[1][0] = 0.;  K[1][1] =eps;  K[1][2] = 0.;
    K[2][0] = 0.;  K[2][1] = 0.;  K[2][2] =eps;
  }
  else{
    double l_l, l_t, l = 0.0001, len;
    l_l = 21.; l_t = 2.1;
    
    double v11 = convection[0]*convection[0],
      v22 = convection[1]*convection[1],
      v33 = convection[2]*convection[2];
    len = sqrt(v11 + v22 + v33);   // the length
    if (len < 0.00000001){         // no dispersion
      K[0][0] = K[1][1] = K[2][2] = 1/l; 
      K[0][1] = K[1][0] = K[0][2] = K[2][0] = K[1][2] = K[2][1] = 0.; 
      return;
    }
  
    double c1 = l_l/len,  c2 = l_t/len;
  
    K[0][0]= 1/(c1*v11+c2*(v22+v33)+l); K[0][1]= 0.; K[0][2]= 0.;
    K[1][0]= 0.; K[1][1]=1/(c1*v22+c2*(v11+v33)+l); K[1][2] = 0.;
    K[2][0]= 0.; K[2][1]= 0.; K[2][2]= 1/(c1*v33+c2*(v11+v22)+l);
  }
}

//============================================================================

double func_c(real *c){
  if (Pressure)
    return 0.0;
  else 
    // return 1.;        // high biodegradation
    // return 0.2;       // middle biodegradation
    return 0.1;          // low biodegradation
}

//============================================================================

double func_f(real *c){
  return 0.;
}

//============================================================================
// Dirichlet boundary value
//============================================================================
double func_u0(real *c){
  if (Pressure){
    if (c[0] < 100.)
      return 1000.;
    else
      return 0.;
  }
  else{
    if (c[0]<0.00001)                  // the point is on dirichlet boundary
      if ((c[2]>49.999)&&(c[2]<350.00001))
	return 30.;
    
    return 0.;
  }
}

//============================================================================
// Neumann boundary value
//============================================================================
double func_gn(real *c){
  return 0.;
}

//============================================================================
// The following funcitons are not used
//============================================================================
double   exact(real *coord){
  return 0.;
}

//============================================================================
// If concentraiton is not defined this function gives the velocity field
//============================================================================
void func_b(real *c, real b[3]){
  b[0] = b[1] = b[2] = 0.;
}

//============================================================================

int locate_well(int Atribut, real *coord){
  if (coord[0] < 300.0001 && coord[0] > 299.9999)
    if (coord[1] < 0.0001 && coord[1] > -0.0001)
      if (coord[2] > 0. && coord[2] < 400.)
	return 1;
  return 0;
}

//============================================================================

#include "Mesh_Classes.h"                // defines tetrahedron class
void Refine_F(int *array, int level, int NTR, 
	      vector<vertex> &Z, vector<tetrahedron> &TR){
  int i;
  double coord[3];
  
  for(i=0; i<NTR; i++){
    TR[i].GetMiddle(coord, Z);
    if (coord[0] < 0.5)
      array[i] = 0;
    else
      array[i] = 1;
  }
}


//============================================================================

