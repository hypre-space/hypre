#include <math.h>
#include <stdio.h>
#include "definitions.h"
//============================================================================
// Here are the functions defining the problem that we are solving:
//        | - div( K grad(u) - vu ) + b.grad(u) + c u = f     in  D  
//        |                
// (1)    |                                         u = u_o   on  G1
//        |                   
//        |                            K drad( u) . n = gn    on Neumann
//        |                 ( K grad( u)).n + gamma.u = not i on Robin
// In this file "double *c" is supposed to be coordinate of 3D point:
// (c[0], c[1], c[2]). If only reaction-diffusion problem is solved this
// should be specified by defining CONVECTION 0FF in definitions.h.
//============================================================================

void func_K(real *c, double K[][3], int atribut){
  K[0][0] = 1.; K[0][1] = 0.; K[0][2]= 0.;
  K[1][0] = 0.; K[1][1] = 1.; K[1][2]= 0.;
  K[2][0] = 0.; K[2][1] = 0.; K[2][2]= 1.;
}

void func_K_Inv(real *c, double K[][3], int atribut){
  K[0][0] = 1.; K[0][1] = 0.; K[0][2]= 0.;
  K[1][0] = 0.; K[1][1] = 1.; K[1][2]= 0.;
  K[2][0] = 0.; K[2][1] = 0.; K[2][2]= 1.;
}

// These are functions for the L-shaped domain. The exact solution is not
// known. We have corner singularity. We solve Homogeneous Poisson equation
// with  f = 1

double func_c(real *c){
  return 0.;
}


double func_f(real *c){
  return 1.;
}

// Dirichlet boundary value
double func_u0(real *c){
  return 0.;
}

double func_gn(real *c){
  return 0.;
}

void func_b(real *c, real b[3]){
  b[0] = 0.;
  b[1] = 0.;
  b[2] = 0.;
}

// these are not defined for this problem 
double   exact(real *c){
  return func_u0( c);
}



// The following are functions for the L-shaped cylindrical domain. The
// exact solution is known. The boundary is Dirichlet everywhere.
/*
double func_f(real *c){
  return 0.;
}

// Dirichlet boundary value
double func_u0(real *c){
  double x = c[0], y = c[1], z = c[2], Pi = 4.*atan(1.);

  if (((x==0.)&&(y<0))||((y==0.)&&(x>0.)))
    return 0.;
  
  if ((x == 0.)&&(y==0.)) return 0.;
  else 
    if (x > 0.)
      return (sin(2./3.* asin(y/sqrt(x*x + y*y)))*
	      pow(x*x+y*y, 1/3.));
    else 
      if (y > 0.)
	return (sin(2./3.* (Pi -asin(y/sqrt(x*x + y*y))))
		*pow(x*x+y*y, 1/3.));
      else 
	return (sin(2./3.* (Pi - asin(y/sqrt(x*x + y*y))))
		*pow(x*x+y*y, 1/3.));
}

double func_gn(real *c){
  return 0.;
}

void func_b(real *c, double b[3]){
  b[0] = 0.;
  b[1] = 0.;
  b[2] = 0.;
}

// these are not defined for this problem 
double   exact(real *c){
  return func_u0( c);
}
*/
