#include <math.h>
#include <stdio.h>
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
// should be specified by defining CONVECTION OFF in definitions.h.
//============================================================================

//============================================================================
//
//============================================================================
int Type_Boundary(double x, double y, double z){
  if (x < 0.00001 || x>999.9999  )
    return Dirichlet;
  else
    if (170.<x && x<230. && 220.<y && y<280. && 190.<z && z<302.)
      return Dirichlet;
    else
      return Neumann;
}


//=========================================================================++
//                    K = K_disp + K_diff                                  ||
//                  ========================                               ||
//                                                                         ||
//                |v1^2  v1v2  v1v3|         |v2^2+v3^2  -v1v2    -v1v3  | ||
//           l_l  |                |    l_t  |                           | ||
// K_disp = ----- |v1v2  v2^2  v2v3| + ----- | -v1v2   v1^2+v3^2  -v2v3  | ||
//           |v|  |                |    |v|  |                           | ||
//                |v1v3  v2v3  v3^2|         | -v1v3     -v2v3  v1^2+v2^2| || 
//                                                                         ||
// K_diff = l*I.                                                           ||
//=========================================================================++
void func_K(double *c, double K[][3], int atribut){
  double l_l, l_t, l = 0.0001, v1, v2, v3, len;

  v1 = 31.9; v2 = 0.; v3 = 0.;                   // take the velocity
  l_l = 21.; l_t = 2.1;

  double v11 = v1*v1, v22 = v2*v2, v33 = v3*v3, v12 = v1*v2, v13 = v1*v3,
    v23 = v2*v3;
  len = sqrt(v11 + v22 + v33);   // the length
  if (len < 0.00000001){         // no dispersion
    K[0][0] = K[1][1] = K[2][2] = l; 
    K[0][1] = K[1][0] = K[0][2] = K[2][0] = K[1][2] = K[2][1] = 0.; 
    return;
  }
  
  double c1 = l_l/len, c2 = l_t/len;
  
  K[0][0]=c1*v11+c2*(v22+v33)+l; K[0][1]=c1*v12-c2*v12; K[0][2]=c1*v13-c2*v13;
  K[1][0]=c1*v12-c2*v12; K[1][1]=c1*v22+c2*(v11+v33)+l; K[1][2]=c1*v23-c2*v23;
  K[2][0]=c1*v13-c2*v13; K[2][1]=c1*v13-c2*v13; K[2][2]=c1*v33+c2*(v11+v22)+l;
}

//============================================================================

double func_c(double *c){
  // return 1.;       // high biodegradation
  return 0.2;      // middle biodegradation
  // return 0.05;     // low biodegradation
}

//============================================================================

double func_f(double *c){
  return 0.;
}

//============================================================================
// Dirichlet boundary value
//============================================================================
double func_u0(double *c){
  double x = c[0], z = c[2];

  if (x<0.00001)  // the point is on dirichlet boundary
    if (z>149.999)
      return 30.;

  return 0.;
}

//============================================================================

double func_gn(double *c){
  return 0.;
}

//============================================================================
// "#if CONCENTRATION == OFF && CONVECTION == ON
//= ( 9 ) ====================================================================
void func_b(double *c, double b[3]){
  b[0] = 31.9;
  b[1] = 0.;
  b[2] = 0.;
}

//============================================================================
// these are not defined for this problem 
//============================================================================
double   exact(double *coord){return 0.;}
void func_K_Inv(double *coord, double K[][3], int atribut){}

