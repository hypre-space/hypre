#include <stdio.h>
#include "definitions.h"

//============================================================================
//============= P R O B L E M   D E S C R I P T I O N ========================
//============================================================================

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
// Given the middle point of face on the bdr return the boundary type :
// DIRICHLET, NEUMANN or ROBIN.
//= ( 1 ) ====================================================================
int Type_Boundary(double x, double y, double z);


//============================================================================
// This function gives the 3 x 3 tensor from "- div( K grad( u))".
// "c" gives the point where the tensor has to be evaluated, "atribut" is the
// element's attribute, which value is set during the initial mesh generation.
//= ( 2 ) ====================================================================
void func_K(real *c, double K[][3], int atribut);


//============================================================================
// The inverse of the above. It's used in "Zienkiewizs-Zhu" error analysis.
//= ( 3 ) ====================================================================
void func_K_Inv(real *c, double K[][3], int atribut);


//============================================================================
// This is the reaction term.
//= ( 4 ) ====================================================================
double func_c(real *c);


//============================================================================
// The source term.
//= ( 5 ) ====================================================================
double func_f(real *c);


//============================================================================
// Dirichlet boundary value
//= ( 6 ) ====================================================================
double func_u0(real *c);


//============================================================================
// Neumann boundary value
//= ( 7 ) ====================================================================
double func_gn(real *c);


//============================================================================
// The exact solution if "#if EXACT == ON"
//= ( 8 ) ====================================================================
double   exact(real *coord);


//============================================================================
// "#if CONCENTRATION == OFF && CONVECTION == ON
//= ( 9 ) ====================================================================
void func_b(real *c, real b[3]);


//============================================================================
// "#if WELL == ON" - return 1 (point belong to well) or 0 (point not in well)
// Put the wells and their characteristics in file "Method.cpp".
//= ( 10 ) ===================================================================
int locate_well(int Atribut, real *coord);
