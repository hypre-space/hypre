#include "math.h"
#include <stdio.h>
#include "Matrix.h"
#include "Method_mixed.h"

#if OUTPUT_LM == ON
  extern FILE *plotLM;
#endif

//============================================================================
// Solving : -div( K grad p) = f, p = u0 on G_D, K grad p . n = gn on G_N
// gives the following Mixed finite element formulation:
// (K^{-1} u, phi) + (p, div phi) = <u0, phi . n>|_G_D
// (div u, psi)                   = -(f, psi)
// We solve for (u . n) at the middle of the edges (RT0) which means G_N
// will be Dirichlet boundary (equal to gn). 
//============================================================================
void MethodMixed::Mixed_LM(int num_tr, real *A, real *B,
 			   double *b1, double *b2) {
  real middle[3], m[4][3][3], m_edge[6][3], m_face[3];
  real K[3][3], LM[4][4];
  int i, j, k;
  
  // Basis function i for RT0 element is given by
  // phi_i(x,y) = [a[i], b[i], c[i]] + d[i]*[x, y, z] , div(phi_i) = 3*d[i]
  double a[4], b[4], c[4], d[4], e[4];
  double det = fabs(TR[num_tr].determinant( Z));
  double vol = det/6;
  
  GetMiddleEdge(TR[num_tr].node[0], TR[num_tr].node[1], m_edge[0]);
  GetMiddleEdge(TR[num_tr].node[1], TR[num_tr].node[2], m_edge[1]);
  GetMiddleEdge(TR[num_tr].node[2], TR[num_tr].node[3], m_edge[2]);
  GetMiddleEdge(TR[num_tr].node[3], TR[num_tr].node[1], m_edge[3]);
  GetMiddleEdge(TR[num_tr].node[0], TR[num_tr].node[2], m_edge[4]);
  GetMiddleEdge(TR[num_tr].node[0], TR[num_tr].node[3], m_edge[5]);

  for(i=0; i<4; i++){                      // compute the 4 basis functions
    e[i] = area(num_tr, i);
    if (F[TR[num_tr].face[i]].tetr[0]==num_tr){
      a[i] = -2*e[i]*Z[TR[num_tr].node[i]].coord[0]/det;
      b[i] = -2*e[i]*Z[TR[num_tr].node[i]].coord[1]/det;
      c[i] = -2*e[i]*Z[TR[num_tr].node[i]].coord[2]/det;
      d[i] =  2*e[i]/det;
    }
    else{
      a[i] = 2*e[i]*Z[TR[num_tr].node[i]].coord[0]/det;
      b[i] = 2*e[i]*Z[TR[num_tr].node[i]].coord[1]/det;
      c[i] = 2*e[i]*Z[TR[num_tr].node[i]].coord[2]/det;
      d[i] = -2*e[i]/det;
    }
    
    GetMiddleEdge(F[TR[num_tr].face[i]].node[0],F[TR[num_tr].face[i]].node[1],
		  m[i][0]);
    GetMiddleEdge(F[TR[num_tr].face[i]].node[1],F[TR[num_tr].face[i]].node[2],
		  m[i][1]);
    GetMiddleEdge(F[TR[num_tr].face[i]].node[0],F[TR[num_tr].face[i]].node[2],
		  m[i][2]);
  }  
  for(i=0; i<4; i++) LM[i][0] = LM[i][1] = LM[i][2] = LM[i][3] = 0.;
  
  GetMiddle(num_tr, middle);
  func_K_Inv( middle, K, TR[num_tr].atribut);

                      // First fix the Neumann and Dirichlet BC.
  for(i=0; i<4; i++)  // For the 4 faces          
    if (F[TR[num_tr].face[i]].tetr[1] == DIRICHLET){
      // b1 += < func_u0, phi_i . n >
      GetMiddleFace(TR[num_tr].face[i], m_face);
      b1[TR[num_tr].face[i]] += e[i]*func_u0(m_face);
    } 
    else  // for now we don't handle Robin BC
      if (F[TR[num_tr].face[i]].tetr[1] == NEUMANN){
	// (div phi_i, psi_num_tr) = 3d[i]*V/det = d[i]/2.
	// In 3D | det | = 6V. The minus comes becouse we put it on the RHS
	GetMiddleFace(TR[num_tr].face[i], m_face);
	b2[num_tr] -= 3*d[i]*vol*func_gn(m_face);
      }

  for(i=0; i<4; i++){
    for(j=0; j<4; j++){
      if (F[TR[num_tr].face[j]].tetr[1] != NEUMANN){
	// Compute (K phi_i, phi_j) using 2nd order quadrature (quadrature
	// points in the middle of the edges).
	for(k=0; k<6; k++){
	  LM[j][i] += ( ( K[0][0]*(a[i]+d[i]*m_edge[k][0]) + 
		 	  K[0][1]*(b[i]+d[i]*m_edge[k][1]) +
			  K[0][2]*(c[i]+d[i]*m_edge[k][2])) * 
			( a[j]+d[j]*m_edge[k][0] ) +
			
		 	( K[1][0]*(a[i]+d[i]*m_edge[k][0]) + 
			  K[1][1]*(b[i]+d[i]*m_edge[k][1]) +
			  K[1][2]*(c[i]+d[i]*m_edge[k][2])) * 
			( b[j]+d[j]*m_edge[k][1] ) +

			( K[2][0]*(a[i]+d[i]*m_edge[k][0]) + 
			  K[2][1]*(b[i]+d[i]*m_edge[k][1]) +
			  K[2][2]*(c[i]+d[i]*m_edge[k][2])) * 
			( c[j]+d[j]*m_edge[k][2] ) );
	}
	
	// the coefficients a, b, c and d are scaled by 1/determinant = 
	// 1/(6*V) so the quadrature gives the scaling 1/(6*6).
	LM[j][i] = LM[j][i]*vol/6;  
	A[ ind_ij( V_A[level], PN_A[level], 
	 	   TR[num_tr].face[j], TR[num_tr].face[i] ) ] += LM[j][i];
      }
    }

    // Matrix B has #elements rows and #edges columns
    if ( F[ TR[num_tr].face[i] ].tetr[1] != NEUMANN)
      B[ind_ij(V_B[level],PN_B[level],num_tr,TR[num_tr].face[i])]+=3*d[i]*vol;
  }
  
  double contrib = 0.;
  for(i=0; i<6; i++)
    contrib -= func_f(m_edge[i]);
  b2[num_tr] += contrib * vol/6;
  
  #if OUTPUT_LM == ON
    for(i=0; i<4; i++){
      for(j=0; j<4; j++)
	fprintf(plotLM,"%12.7f ", LM[i][j]);
      fprintf(plotLM,"\n");
    }
    fprintf(plotLM,"\n");
  #endif
}

//============================================================================
// Print the local matrix through the stream out. The local matrix is the 
// followiwng bilinear form : a(u,v) = (u, v) + (div u, div v), where
// u and v are vector-functions in Raviart--Thomas space (RT0).
//============================================================================
void MethodMixed::Print_Mixed_LM(int num_tr, ofstream &out){
  real m_edge[6][3], LM[4][4];
  int i, j, k;
  
  // Basis function i for RT0 element is given by :
  // phi_i(x, y, z) = [a[i], b[i], c[i]] + d[i]*[x, y, z] 
  // div(phi_i) = 3*d[i]
  double a[4], b[4], c[4], d[4], e[4];
  double det = fabs(TR[num_tr].determinant( Z));
  double vol = det/6;
  
  GetMiddleEdge(TR[num_tr].node[0], TR[num_tr].node[1], m_edge[0]);
  GetMiddleEdge(TR[num_tr].node[1], TR[num_tr].node[2], m_edge[1]);
  GetMiddleEdge(TR[num_tr].node[2], TR[num_tr].node[3], m_edge[2]);
  GetMiddleEdge(TR[num_tr].node[3], TR[num_tr].node[1], m_edge[3]);
  GetMiddleEdge(TR[num_tr].node[0], TR[num_tr].node[2], m_edge[4]);
  GetMiddleEdge(TR[num_tr].node[0], TR[num_tr].node[3], m_edge[5]);

  for(i=0; i<4; i++){                      // compute the 4 basis functions
    e[i] = area(num_tr, i);
    if (F[TR[num_tr].face[i]].tetr[0]==num_tr){
      a[i] = -2*e[i]*Z[TR[num_tr].node[i]].coord[0]/det;
      b[i] = -2*e[i]*Z[TR[num_tr].node[i]].coord[1]/det;
      c[i] = -2*e[i]*Z[TR[num_tr].node[i]].coord[2]/det;
      d[i] =  2*e[i]/det;
    }
    else{
      a[i] = 2*e[i]*Z[TR[num_tr].node[i]].coord[0]/det;
      b[i] = 2*e[i]*Z[TR[num_tr].node[i]].coord[1]/det;
      c[i] = 2*e[i]*Z[TR[num_tr].node[i]].coord[2]/det;
      d[i] = -2*e[i]/det;
    }
  }  
  for(i=0; i<4; i++) LM[i][0] = LM[i][1] = LM[i][2] = LM[i][3] = 0.;
  
  for(i=0; i<4; i++)
    for(j=0; j<4; j++){
	// Compute ( phi_i, phi_j) using 2nd order quadrature (quadrature
	// points in the middle of the edges).
	for(k=0; k<6; k++)
	  LM[j][i] += ( (a[i]+d[i]*m_edge[k][0]) * (a[j]+d[j]*m_edge[k][0]) +
			(b[i]+d[i]*m_edge[k][1]) * (b[j]+d[j]*m_edge[k][1]) +
			(c[i]+d[i]*m_edge[k][2]) * (c[j]+d[j]*m_edge[k][2]) );
	
	LM[j][i] = LM[j][i]*vol/6 + 9*d[i]*d[j]*vol;  
    }

  for(i=0; i<4; i++){
    for(j=0; j<4; j++)
      out << LM[i][j] << "  ";
    out << endl;
  }
  out << endl;
}

//============================================================================
// RT0 basis function i (the index is given by TR[num_tr].face[i]) evaluated 
// at point coord. The computation is for triangle num_tr and the output is
// written in result. It's not efficient to use this function - implemented
// only for ilustration how to obtain it.
//============================================================================
void MethodMixed::phi_RT0(int num_tr, int i, real coord[3], real result[3]){
  double det, a, b, c, d, e;

  det = fabs(TR[num_tr].determinant( Z));

  e = area(num_tr, i);
  if (F[TR[num_tr].face[i]].tetr[0]==num_tr){
    a = -2*e*Z[TR[num_tr].node[i]].coord[0]/det;
    b = -2*e*Z[TR[num_tr].node[i]].coord[1]/det;
    c = -2*e*Z[TR[num_tr].node[i]].coord[2]/det;
    d =  2*e/det;
  }
  else{
    a = 2*e*Z[TR[num_tr].node[i]].coord[0]/det;
    b = 2*e*Z[TR[num_tr].node[i]].coord[1]/det;
    c = 2*e*Z[TR[num_tr].node[i]].coord[2]/det;
    d = -2*e/det;
  }
  
  result[0] = a + d * coord[0];
  result[1] = b + d * coord[1];
  result[2] = c + d * coord[2];
}

//============================================================================
// Similar to the above. Return the divergence. Again not efficient to use.
//============================================================================
real MethodMixed::div_phi_RT0(int num_tr, int i){
  if (F[TR[num_tr].face[i]].tetr[0]==num_tr)
    return 6*area(num_tr, i)/fabs(TR[num_tr].determinant( Z));
  else
    return -6*area(num_tr, i)/fabs(TR[num_tr].determinant( Z));
}
//============================================================================
