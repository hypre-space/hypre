#include <math.h>
#include <stdio.h>
#include "Matrix.h"
#include "Method.h"
#include "definitions.h"

#if CONCENTRATION == ON || CONVECTION == ON
  extern real convection[3];
  extern int Pressure;
#endif

#if OUTPUT_LM == ON
  extern FILE *plotLM;
#endif


//============================================================================
// Add the local contribution from tetrahedron num_tr to the global stiffness
// matrix (for Reaction-Diffusion problem). The function has extension to add
// convection term doing discontinuous Galerkin approximation (used for con-
// vection dominated problems. The extension is used by setting CONVECTION
// to be ON. Another variant of using is in solving pressure equation, from
// which we get the velocity, which is the convection term in a convection
// dominated problem for the concentration. 
//============================================================================
void Method::Reaction_Diffusion_LM(int num_tr, real *A, double *b){
  int i, j;
  real *coord, middle[3], vol, LM[4][4];
  double f[5];              // used for initializing the RHS b       

  real K[3][3];             // diffusion matrix
  real grad[4][3];          // grad[i]   - gradient vector for node i = 0..3
  double K_grad[4][3];      // K_grad[i] - flux vector ( K * grad[i])
    
  real M, diag, off_diag;

  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      LM[i][j] = 0.;

  GetMiddle(num_tr, middle);

  grad_nodal_basis(num_tr, grad);

  #if CONCENTRATION == ON
    if (Pressure == OFF){   // i.e. if we solve the concentration part - get
                            // the convection part from the pressure
      K_grad[0][0] = K_grad[0][1] = K_grad[0][2] = 0.;
      for(i=0; i<4; i++){
	K_grad[0][0] += grad[i][0]*pressure[TR[num_tr].node[i]];
	K_grad[0][1] += grad[i][1]*pressure[TR[num_tr].node[i]];
	K_grad[0][2] += grad[i][2]*pressure[TR[num_tr].node[i]];
      }

      // get the Darcy velocity
      Pressure = 1;
      func_K(middle, K, TR[num_tr].atribut);
      convection[0] = K[0][0]*K_grad[0][0] + K[0][1]*K_grad[0][1] + 
	K[0][2]*K_grad[0][2];
      convection[1] = K[1][0]*K_grad[0][0] + K[1][1]*K_grad[0][1] + 
	K[1][2]*K_grad[0][2];
      convection[2] = K[2][0]*K_grad[0][0] + K[2][1]*K_grad[0][1] + 
	K[2][2]*K_grad[0][2];
      Pressure = 0;

      // add convection to the local matrix LM
      Convection_LM(num_tr, LM, convection);
    }
  #elif CONVECTION == ON
    func_b(middle, convection);
    Convection_LM(num_tr, LM, convection);// add convection
  #endif

  func_K( middle, K, TR[num_tr].atribut);
  vol = volume(num_tr);
      
  M = func_c( middle);
  off_diag = M*vol*0.05;                         // 0.05 = 1/20
  diag     = 2*off_diag;
 
  for(i=0; i<4; i++)
    for(j=0; j<3; j++)
      K_grad[i][j]=K[j][0]*grad[i][0]+K[j][1]*grad[i][1]+K[j][2]*grad[i][2];

  for(i=0; i<4; i++){
    coord = GetNode(TR[num_tr].node[i]); 
    f[i]  = func_f( coord);
  }
  f[4] = 4*func_f(middle);

  for(i=0;i<4;i++)
    for(j=0;j<4;j++){
      LM[i][j] += ( K_grad[i][0]*grad[j][0] +    // add dispersion
		    K_grad[i][1]*grad[j][1] +
		    K_grad[i][2]*grad[j][2] ) * vol;
      
      if (i==j)                                  // add reaction
	LM[i][j] += diag;
      else
	LM[i][j] += off_diag;
      
      A[ ind_ij(level, TR[num_tr].node[i], TR[num_tr].node[j]) ] += LM[i][j];
    }
      
  vol *= 0.05;                                   // vol=vol/20 (optimization)
  b[TR[num_tr].node[0]]+=(f[0]+f[4])*vol;
  b[TR[num_tr].node[1]]+=(f[1]+f[4])*vol;
  b[TR[num_tr].node[2]]+=(f[2]+f[4])*vol;
  b[TR[num_tr].node[3]]+=(f[3]+f[4])*vol;

  //============= Fix Neumann Boundary =======================================
  real m_edge[3][3];
  real S, ff;
  for(i=0; i<4; i++){            // for the 4 faces
    ff = TR[num_tr].face[i];
    if (F[ff].tetr[1] == NEUMANN){
      S = area(num_tr, i);
      GetMiddleEdge(F[ff].node[1], F[ff].node[2], m_edge[0]);
      GetMiddleEdge(F[ff].node[0], F[ff].node[2], m_edge[1]);
      GetMiddleEdge(F[ff].node[0], F[ff].node[1], m_edge[2]);
     
      b[F[ff].node[0]] += (func_gn(m_edge[1])+func_gn(m_edge[2]))*S/6;
      b[F[ff].node[1]] += (func_gn(m_edge[0])+func_gn(m_edge[2]))*S/6;
      b[F[ff].node[2]] += (func_gn(m_edge[0])+func_gn(m_edge[1]))*S/6;  
    }
  }

  #if OUTPUT_LM == ON                            // print the local matrix
    for(i=0; i<4; i++){
      for(j=0; j<4; j++)
	fprintf(plotLM,"%12.7f ", LM[i][j]);
      fprintf(plotLM,"\n");
    }
    fprintf(plotLM,"\n");
  #endif
}


//============================================================================
// Add the local contribution from tetrahedron num_tr to the global stiffness
// matrix (for Convection-Diffusion problem) using the streamline-diffusion
// method. If we make "delta = 0" (in the code below) we will do standard
// FE approximation for the convection part, which is "good" for "small" conv.
//============================================================================
void Method::Convection_Diffusion_LM(int num_tr, real *A, double *b){
  int i, j, ii, jj, s;
  real vol, a_ij;
  real *coord, middle[3], conv[3]; // conv - convection direction
  double f[4];             // used to initialize the RHS b  

  real K[3][3];            // diffusion matrix
  real grad[4][3];         // grad[i]   - gradient vector for node i = 0..3
  double K_grad[4][3];     // K_grad[i] - flux vector ( K * grad[i] 
  double db[3],lb;         // (lb -length) of convection direction
  double h, delta;         // db = delta . conv          
  
  double M, diag, off_diag;

  GetMiddle(num_tr, middle);
      
  func_K( middle, K, TR[num_tr].atribut);
  func_b( middle, conv);

  lb = sqrt(conv[0]*conv[0]+conv[1]*conv[1]+conv[2]*conv[2]);
  h  = length( F[ TR[num_tr].face[TR[num_tr].refine]].node[0],
	       F[ TR[num_tr].face[TR[num_tr].refine]].node[1]);

  // fix delta : the constant 16
  s = 0;
  for(i=1; i<3; i++)
    if (K[i][i] < K[s][s]) s = i;
  if (K[s][s] < h*lb)
    delta = h/(8*lb);
  else
    delta = 0.;

  // fix db = delta . conv
  for(i=0; i<3; i++)
    db[i] = conv[i]*delta;

  // fix the diffusion matrix K = K + bd conv
  for(i=0; i<3; i++)
    for(j=0; j<3; j++)
      K[i][j] += db[i]*conv[j];

  grad_nodal_basis(num_tr, grad);
  vol = volume(num_tr);
      
  M = func_c( middle);
  off_diag = M*vol/20;
  diag     = 2*off_diag;
 
  for(i=0; i<4; i++)
    for(j=0; j<3; j++)
      K_grad[i][j]=K[j][0]*grad[i][0]+K[j][1]*grad[i][1]+K[j][2]*grad[i][2];

  for(i=0; i<4; i++){
    coord = GetNode(TR[num_tr].node[i]); 
    f[i]  = func_f( coord);
  }

  for(i=0; i<4; i++){
    for(j=0; j<4; j++){
      a_ij = 0.;
      ii = TR[num_tr].node[i];
      jj = TR[num_tr].node[j];
      
      // contribution from (K grad(u), grad(phi_j))
      a_ij += ( K_grad[i][0]*grad[j][0] +
		K_grad[i][1]*grad[j][1] +
		K_grad[i][2]*grad[j][2] ) * vol;
      
      // contribution from (conv . grad(u), phi_j)
      a_ij += ( grad[i][0]*conv[0] + grad[i][1]*conv[1] + 
		grad[i][2]*conv[2] ) * vol * 0.25;
      
      // contribution from (c u, phi_j)
      if (i==j) a_ij += diag;
      else      a_ij += off_diag;
    
      // contribution from (c u, db . grad(phi_j))
      a_ij += (grad[j][0]*db[0]+grad[j][1]*db[1]+grad[j][2]*db[2])*M*vol/4.;

      // initialization of the RHS b
      if (i==j)	b[jj] += 2*f[i]*vol/20;
      else      b[jj] +=   f[i]*vol/20;
      b[jj]+=(grad[j][0]*db[0]+grad[j][1]*db[1]+grad[j][2]*db[2])*f[i]*vol/4;

      A[ ind_ij(level, ii, jj) ] += a_ij;
    }
  }

  //============= Fix Neumann Boundary =======================================
  real n[3], dbn = 0.;           // dbn = delta conv . n
  double S;
  int ff, local;
  for(s=0; s<4; s++){            // for the 4 faces
    ff = TR[num_tr].face[s];
    if (F[ff].tetr[1] == NEUMANN){
      normal(n, num_tr, s);
      S = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])/3;
      for(i=0; i<3; i++) dbn += db[i]*n[i];
      
      for(i=0;i<3;i++)
	for(j=0;j<3;j++){
	  a_ij = 0.;
	  local = (s+i+1)%4;
	  ii = F[ff].node[i];
	  jj = F[ff].node[j];

	  // contribution from -<conv.grad(u), delta conv . n phi_j>
	  a_ij -= (grad[local][0]*conv[0] + grad[local][1]*conv[1] +
		   grad[local][2]*conv[2])*dbn/3;

	  // contribution from -<c u, dbn phi_j>
	  if (i==j) a_ij -= M*dbn/6;
	  else      a_ij -= M*dbn/12;
	  
	  // contribution ftom -<f, dbn phi_j> (goes in the RHS)
	  if (i==j) b[jj] -= f[local]*dbn/6;
	  else      b[jj] -= f[local]*dbn/12;

	  // contribution from <K grad u . n, phi_j> 
	  b[jj] += func_gn( middle)*S;

	  A[ ind_ij(level, ii, jj) ] += a_ij;
	}
    }   // end if face is NEUMANN
  }     // end for the 4 faces    
}


//============================================================================
// Dot produnt of 3D vectors.
//============================================================================
template <class data>
data dot_product(data x[3], data y[3]){
  return (x[0]*y[0]+x[1]*y[1]+x[2]*y[2]);
}

//============================================================================
// Cross product (cross = x X y).
//============================================================================
template <class data>
void cross_product(data x[3], data y[3], data cross[3]){
  cross[0] = x[1]*y[2] - x[2]*y[1];
  cross[1] = x[2]*y[0] - x[0]*y[2];
  cross[2] = x[0]*y[1] - x[1]*y[0];
}


//============================================================================
// This procedure adds convection (of the form div(b u)) to the stiffness 
// matrix. It is a discontinuous Galerkin finite volume approximation. We
// go through the elements and for every element's edge we add the approp-
// riate contributions to the end points with function "add_convection".  
//============================================================================
void Method::Convection_LM(real *A){
  real conv[3], LM[4][4], middle[3], grad[4][3];
  int i, j, k, ii, jj, end, num_tr;
  real K[3][3]; 
  
  for(num_tr=0; num_tr<NTR; num_tr++){
    
    for(i=0; i<4; i++)
      for(j=0; j<4; j++)
	LM[i][j] = 0.;

    GetMiddle(num_tr, middle);

    #if CONCENTRATION == ON
      grad_nodal_basis(num_tr, grad);

      if (Pressure) func_K(middle, K, TR[num_tr].atribut);
      else {Pressure=1; func_K(middle, K, TR[num_tr].atribut); Pressure=0;}

      middle[0] = middle[1] = middle[2] = 0.;
      for(i=0; i<4; i++){
	middle[0] += grad[i][0]*pressure[TR[num_tr].node[i]];
	middle[1] += grad[i][1]*pressure[TR[num_tr].node[i]];
	middle[2] += grad[i][2]*pressure[TR[num_tr].node[i]];
      }
      conv[0] = K[0][0]*middle[0] + K[0][1]*middle[1] + K[0][2]*middle[2];
      conv[1] = K[1][0]*middle[0] + K[1][1]*middle[1] + K[1][2]*middle[2];
      conv[2] = K[2][0]*middle[0] + K[2][1]*middle[1] + K[2][2]*middle[2];
    #else
      func_b( middle, conv);
    #endif

    add_convection(0, 1, 2, 3, conv, LM, num_tr);
    add_convection(1, 2, 0, 3, conv, LM, num_tr);
    add_convection(2, 0, 1, 3, conv, LM, num_tr);
    add_convection(0, 3, 1, 2, conv, LM, num_tr);
    add_convection(1, 3, 2, 0, conv, LM, num_tr);
    add_convection(2, 3, 1, 0, conv, LM, num_tr);
    
    // Now check whether there are faces on the boundary
    for(i=0; i<4; i++)
      if (F[TR[num_tr].face[i]].tetr[1] == NEUMANN ||
	  F[TR[num_tr].face[i]].tetr[1] == ROBIN )
    	add_convection_face( i, conv, LM, num_tr);
    
    for(i=0;i<4;i++)
      for(j=0;j<4;j++){
	ii = TR[num_tr].node[i];
	jj = TR[num_tr].node[j];
	
	end   = V[level][ii+1];
	for(k=V[level][ii]; k< end; k++)
	  if(jj==PN[level][k])
	    A[k] += LM[i][j];
      }
  }
}


//============================================================================
void Method::Convection_LM(int num_tr, real LM[][4], real conv[3]){
  int i;
  
  add_convection(0, 1, 2, 3, conv, LM, num_tr);
  add_convection(1, 2, 0, 3, conv, LM, num_tr);
  add_convection(2, 0, 1, 3, conv, LM, num_tr);
  add_convection(0, 3, 1, 2, conv, LM, num_tr);
  add_convection(1, 3, 2, 0, conv, LM, num_tr);
  add_convection(2, 3, 1, 0, conv, LM, num_tr);
    
  // Now check whether there are faces on the boundary
  for(i=0; i<4; i++)
    if (F[TR[num_tr].face[i]].tetr[1]== NEUMANN ||
	F[TR[num_tr].face[i]].tetr[1]== ROBIN )
      add_convection_face( i, conv, LM, num_tr);
}


//============================================================================
// This routine is used by the previous one. It adds the contribution asso-
// ciated with edge n1-n2 from element num_tr. n3 and n4 are the other two
// vertices.
//============================================================================
void Method::add_convection(int n1, int n2, int n3, int n4, 
			    real conv[3], real LM[][4], int num_tr){
  real v[4][3], dot = 0.;
  int i;
  
  for(i=0; i<3; i++){
    // We put in v0 and v1 vectors which inner-product will give as the 
    // normal to the "integrating" face associated with the considered edge
    // One fourth of the length is the area of that integrating face.
    v[0][i] = (( Z[TR[num_tr].node[n1]].coord[i] +
		 Z[TR[num_tr].node[n2]].coord[i] +
		 Z[TR[num_tr].node[n3]].coord[i] )/3 - 
	       Z[TR[num_tr].node[n4]].coord[i]);
    
    v[1][i] = (( Z[TR[num_tr].node[n1]].coord[i] +
		 Z[TR[num_tr].node[n2]].coord[i] )*0.5 -
	       Z[TR[num_tr].node[n4]].coord[i]);
    
    v[2][i] = Z[TR[num_tr].node[n1]].coord[i]-Z[TR[num_tr].node[n4]].coord[i];
  }
  
  cross_product( v[0], v[1], v[3]);  // we get the normal in v3
  
  if (dot_product(v[2], v[3]) > 0)   // we want the nornal to be oriented 
    for(i=0; i<3; i++)               // from p1 to p2 (always)
      v[3][i] = -v[3][i];
  
  dot = dot_product(conv,v[3])*0.25;
  if (dot > 0){
    LM[n1][n1] += dot;
    LM[n2][n1] -= dot;
  }
  else{
    LM[n1][n2] += dot;
    LM[n2][n2] -= dot;
  } 
}


//============================================================================
// For Face i in tetrahedron num_tr add the corresponding contributions from
// the convection part (the face is on Neumann boundary).
//============================================================================
void Method::add_convection_face(int i, real conv[3], real LM[][4], 
				 int num_tr){
  real n[3], dot;

  normal(n, num_tr, i);
  dot = dot_product(conv, n)/3.;
  if (dot > 0.){
    LM[0][0] += dot; LM[1][1] += dot; LM[2][2] += dot; LM[3][3] += dot;
    LM[i][i] -= dot;
  }
}

//============================================================================
// LM is poiter to "poiters to 4 doubles". This is needed by the Hypre prec.
// Library (otherwise we could have given "double LM[][4]".
void Method::ComputeLocalMatrix(int num_tr, double **LM, double *b){
    #if PROBLEM == 0 || PROBLEM == 1
      Reaction_Diffusion_LM( num_tr, LM, b);
    #elif PROBLEM == 2
      Convection_Diffusion_LM(num_tr, LM, b);
    #endif
}

//============================================================================
// For tetrahedron num_tr this function computes the local matrix LM[4][4]
// and the cotribution to the RHS b[4] for a reaction-diffusion problem.
//============================================================================
void Method::Reaction_Diffusion_LM(int num_tr, real **LM, double *b){
  int i, j;
  real *coord, middle[3], vol;
  double f[5];              // used for initializing the RHS b       

  real K[3][3];             // diffusion matrix
  real grad[4][3];          // grad[i]   - gradient vector for node i = 0..3
  double K_grad[4][3];      // K_grad[i] - flux vector ( K * grad[i])
    
  real M, diag, off_diag;

  for(i=0; i<4; i++)
    for(j=0; j<4; j++)
      LM[i][j] = 0.;

  GetMiddle(num_tr, middle);

  grad_nodal_basis(num_tr, grad);

  #if CONCENTRATION == ON
    if (Pressure == OFF){   // i.e. if we solve the concentration part - get
                            // the convection part from the pressure
      K_grad[0][0] = K_grad[0][1] = K_grad[0][2] = 0.;
      for(i=0; i<4; i++){
	K_grad[0][0] += grad[i][0]*pressure[TR[num_tr].node[i]];
	K_grad[0][1] += grad[i][1]*pressure[TR[num_tr].node[i]];
	K_grad[0][2] += grad[i][2]*pressure[TR[num_tr].node[i]];
      }

      // get the Darcy velocity
      Pressure = 1;
      func_K(middle, K, TR[num_tr].atribut);
      convection[0] = K[0][0]*K_grad[0][0] + K[0][1]*K_grad[0][1] + 
	K[0][2]*K_grad[0][2];
      convection[1] = K[1][0]*K_grad[0][0] + K[1][1]*K_grad[0][1] + 
	K[1][2]*K_grad[0][2];
      convection[2] = K[2][0]*K_grad[0][0] + K[2][1]*K_grad[0][1] + 
	K[2][2]*K_grad[0][2];
      Pressure = 0;

      // add convection to the local matrix LM
      double CM[4][4];
      for(i=0; i<4; i++) for(j=0; j<4; j++) CM[i][j] = 0.;
      Convection_LM(num_tr, CM, convection);
      for(i=0; i<4; i++) for(j=0; j<4; j++) LM[i][j] = CM[i][j];
    }
  #elif CONVECTION == ON
    func_b(middle, convection);
    double CM[4][4];
    for(i=0; i<4; i++) for(j=0; j<4; j++) CM[i][j] = 0.;
    Convection_LM(num_tr, CM, convection);       // add convection
    for(i=0; i<4; i++) for(j=0; j<4; j++) LM[i][j] = CM[i][j];
  #endif

  func_K( middle, K, TR[num_tr].atribut);
  vol = volume(num_tr);
      
  M = func_c( middle);
  off_diag = M*vol*0.05;                         // 0.05 = 1/20
  diag     = 2*off_diag;
 
  for(i=0; i<4; i++)
    for(j=0; j<3; j++)
      K_grad[i][j]=K[j][0]*grad[i][0]+K[j][1]*grad[i][1]+K[j][2]*grad[i][2];

  for(i=0; i<4; i++){
    coord = GetNode(TR[num_tr].node[i]); 
    f[i]  = func_f( coord);
  }
  f[4] = 4*func_f(middle);

  for(i=0;i<4;i++)
    for(j=0;j<4;j++){
      LM[i][j] += ( K_grad[i][0]*grad[j][0] +    // add dispersion
		    K_grad[i][1]*grad[j][1] +
		    K_grad[i][2]*grad[j][2] ) * vol;
      
      if (i==j)                                  // add reaction
	LM[i][j] += diag;
      else
	LM[i][j] += off_diag;
    }
      
  vol *= 0.05;                                   // vol=vol/20 (optimization)
  b[0] = (f[0]+f[4])*vol;
  b[1] = (f[1]+f[4])*vol;
  b[2] = (f[2]+f[4])*vol;
  b[3] = (f[3]+f[4])*vol;

  //============= Fix Neumann Boundary =======================================
  real m_edge[3][3];
  real S, ff;
  for(i=0; i<4; i++){            // for the 4 faces
    ff = TR[num_tr].face[i];
    if (F[ff].tetr[1] == NEUMANN){
      S = area(num_tr, i);
      GetMiddleEdge(F[ff].node[1], F[ff].node[2], m_edge[0]);
      GetMiddleEdge(F[ff].node[0], F[ff].node[2], m_edge[1]);
      GetMiddleEdge(F[ff].node[0], F[ff].node[1], m_edge[2]);

      for(j=0; j<4; j++)
	if (TR[num_tr].node[j]==F[ff].node[0])
	  b[j] += (func_gn(m_edge[1])+func_gn(m_edge[2]))*S/6;
	else if (TR[num_tr].node[j]==F[ff].node[1])
	  b[j] += (func_gn(m_edge[0])+func_gn(m_edge[2]))*S/6;
	else if (TR[num_tr].node[j]==F[ff].node[2])
	  b[j] += (func_gn(m_edge[0])+func_gn(m_edge[1]))*S/6;
    }
  }
}


//============================================================================
// For tetrahedron num_tr this function computes the local matrix LM[4][4]
// and the cotribution to the RHS b[4] for a convection-diffusion problem.
//============================================================================
void Method::Convection_Diffusion_LM(int num_tr, real **LM, double *b){
  int i, j, ii, jj, s;
  real vol, a_ij;
  real *coord, middle[3], conv[3]; // conv - convection direction
  double f[4];             // used to initialize the RHS b  

  real K[3][3];            // diffusion matrix
  real grad[4][3];         // grad[i]   - gradient vector for node i = 0..3
  double K_grad[4][3];     // K_grad[i] - flux vector ( K * grad[i] 
  double db[3],lb;         // (lb -length) of convection direction
  double h, delta;         // db = delta . conv          
  
  double M, diag, off_diag;

  GetMiddle(num_tr, middle);
      
  func_K( middle, K, TR[num_tr].atribut);
  func_b( middle, conv);

  lb = sqrt(conv[0]*conv[0]+conv[1]*conv[1]+conv[2]*conv[2]);
  h  = length( F[ TR[num_tr].face[TR[num_tr].refine]].node[0],
	       F[ TR[num_tr].face[TR[num_tr].refine]].node[1]);

  // fix delta : the constant 16
  s = 0;
  for(i=1; i<3; i++)
    if (K[i][i] < K[s][s]) s = i;

  if (K[s][s] < h*lb)
    delta = h/(8*lb);
  else
    delta = 0.;

  // fix db = delta . conv
  for(i=0; i<3; i++)
    db[i] = conv[i]*delta;

  // fix the diffusion matrix K = K + bd conv
  for(i=0; i<3; i++)
    for(j=0; j<3; j++)
      K[i][j] += db[i]*conv[j];

  grad_nodal_basis(num_tr, grad);
  vol = volume(num_tr);
      
  M = func_c( middle);
  off_diag = M*vol/20;
  diag     = 2*off_diag;
 
  for(i=0; i<4; i++)
    for(j=0; j<3; j++)
      K_grad[i][j]=K[j][0]*grad[i][0]+K[j][1]*grad[i][1]+K[j][2]*grad[i][2];

  for(i=0; i<4; i++){
    coord = GetNode(TR[num_tr].node[i]); 
    f[i]  = func_f( coord);
    b[i]  = 0.;
  }

  for(i=0; i<4; i++){
    for(j=0; j<4; j++){
      // contribution from (K grad(u), grad(phi_j))
      LM[i][j] = ( K_grad[i][0]*grad[j][0] +
		   K_grad[i][1]*grad[j][1] +
		   K_grad[i][2]*grad[j][2] ) * vol;
      
      // contribution from (conv . grad(u), phi_j)
      LM[i][j] += ( grad[i][0]*conv[0] + grad[i][1]*conv[1] + 
		    grad[i][2]*conv[2] ) * vol * 0.25;
      
      // contribution from (c u, phi_j)
      if (i==j) LM[i][j] += diag;
      else      LM[i][j] += off_diag;
    
      // contribution from (c u, db . grad(phi_j))
      LM[i][j]+=(grad[j][0]*db[0]+grad[j][1]*db[1]+grad[j][2]*db[2])*M*vol/4.;

      // initialization of the RHS b
      if (i==j)	b[j] += 2*f[i]*vol/20;
      else      b[j] +=   f[i]*vol/20;
      b[j]+=(grad[j][0]*db[0]+grad[j][1]*db[1]+grad[j][2]*db[2])*f[i]*vol/4;
    }
  }

  //============= Fix Neumann Boundary =======================================
  real n[3], dbn = 0.;           // dbn = delta conv . n
  double S;
  int ff, local, k;
  for(s=0; s<4; s++){            // for the 4 faces
    ff = TR[num_tr].face[s];
    if (F[ff].tetr[1] == NEUMANN){
      normal(n, num_tr, s);
      S = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])/3;
      for(i=0; i<3; i++) dbn += db[i]*n[i];
      
      for(i=0;i<3;i++)
	for(j=0;j<3;j++){
	  local = (s+i+1)%4;
	  for(k=0; k<4; k++){
	    if (TR[num_tr].node[k] == F[ff].node[i]) ii = k;
	    if (TR[num_tr].node[k] == F[ff].node[j]) jj = k;
	  }
	  // contribution from -<conv.grad(u), delta conv . n phi_j>
	  LM[ii][jj] -= (grad[local][0]*conv[0] + grad[local][1]*conv[1] +
			 grad[local][2]*conv[2])*dbn/3;

	  // contribution from -<c u, dbn phi_j>
	  if (ii==jj) LM[ii][jj] -= M*dbn/6;
	  else        LM[ii][jj] -= M*dbn/12;
	  
	  // contribution ftom -<f, dbn phi_j> (goes in the RHS)
	  if (ii==jj) b[jj] -= f[local]*dbn/6;
	  else        b[jj] -= f[local]*dbn/12;

	  // contribution from <K grad u . n, phi_j> 
	  b[jj] += func_gn( middle)*S;
	}
    }   // end if face is NEUMANN
  }     // end for the 4 faces    
}

//============================================================================





