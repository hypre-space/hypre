#include <stdio.h>
#include "definitions.h"

extern double v1, v2, v3;         // used for the velocity
extern int Pressure;              // 1 if pressure equations are being solved

//============================================================================
// This problem determines the pressure (from where the velocity) for mesh in
// bioscreen.out, Mesh_functions in Problem1. The mesh has layer, marked with
// atribut 1.
//============================================================================
void func_K(double *c, double K[][3], int atribut){
  if (Pressure){       // Solve the pressure
    double eps;

    if (atribut == 1)   // this element is in the layer
      eps = 1/2000.;
    else eps = 1/1000.;
    
    K[0][0] = eps;  K[0][1] =  0.;  K[0][2] =  0.;
    K[1][0] =  0.;  K[1][1] = eps;  K[1][2] =  0.;
    K[2][0] =  0.;  K[2][1] =  0.;  K[2][2] = eps;
  }
  else {
    double l_l, l_t, l = 0.0001, len;
    l_l = 21.; l_t = 2.1;

    double v11=v1*v1,v22=v2*v2,v33=v3*v3,v12=v1*v2,v13=v1*v3,v23=v2*v3;
    len = sqrt(v11 + v22 + v33);   // the length
    if (len < 0.00000001){         // no dispersion
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

void func_K_Inv(double *c, double K[][3], int atribut){
  if (Pressure){
    double eps;
    if (atribut == 1)
      eps = 2000.;
    else
      atribut = 1000.;
    
    K[0][0] =eps;  K[0][1] = 0.;  K[0][2] = 0.;
    K[1][0] = 0.;  K[1][1] =eps;  K[1][2] = 0.;
    K[2][0] = 0.;  K[2][1] = 0.;  K[2][2] =eps;
  }
  else {
    double l_l, l_t, l = 0.0001, len;
    l_l = 21.; l_t = 2.1;
    
    double v11=v1*v1,v22=v2*v2,v33=v3*v3;
    len = sqrt(v11 + v22 + v33);   // the length
    if (len < 0.00000001){         // no dispersion
      K[0][0] = K[1][1] = K[2][2] = 1/l; 
      K[0][1] = K[1][0] = K[0][2] = K[2][0] = K[1][2] = K[2][1] = 0.; 
      return;
    }
  
    double c1 = l_l/len, c2 = l_t/len;
  
    K[0][0]= 1/(c1*v11+c2*(v22+v33)+l); K[0][1]= 0.; K[0][2]= 0.;
    K[1][0]= 0.; K[1][1]=1/(c1*v22+c2*(v11+v33)+l); K[1][2] = 0.;
    K[2][0]= 0.; K[2][1]= 0.; K[2][2]= 1/(c1*v33+c2*(v11+v22)+l);
  }
}


double func_c(double *c){
  if (Pressure)
    return 0.0;
  else 
    // return 1.;       // high biodegradation
    // return 0.2;      // middle biodegradation
    return 0.1;     // low biodegradation
}


double func_f(double *c){
  return 0.;
}

// Dirichlet boundary value
double func_u0(double *c){
  if (Pressure){
    if (c[0] < 100.)
      return 32.;
    else
      if (170.<c[0] && c[0]<230. && 220.<c[1] && c[1]<280.
	  && 190.<c[2] && c[2]<302.) // this is for the well - may write 
	return 0.;                   // directly else return 0.
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

// Neumann boundary value
double func_gn(double *c){
  return 0.;
}


// The following funcitons are not used
double   exact(double *coord){
  return 0.;
}

void func_b(double *c, double b[3]){
  b[0] = b[1] = b[2] = 0.;
}



