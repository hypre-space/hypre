#include <iostream.h>
#include <stdio.h>
#include <math.h>

#include "Method.h"

void     func_K(real *coord, real K[][3], int atribut);
void func_K_Inv(real *coord, real K[][3], int atribut);

//============================================================================
// ZZ Refinement.
//============================================================================
void Method::Refine_ZZ(double percent, double *sol, int *array) {
  double est_min = 100., est_max = 0., eps;
  int i, j, k, Nn = NN[level]; 
  
  if (percent!=0){

    double3 *grad = new double3[Nn];
    double *vol=new double[Nn], var;
  
    for(i=0; i<Nn; i++) grad[i][0] = grad[i][1] = grad[i][2] = vol[i] = 0.;
  
    // compute int_Omega(N^T D S N) d Omega sol and A^-1 (as given in ZZ)
    for(i=0; i<NTR; i++) 
      CompGradVert(i, sol, grad, vol); 
  
    for(i=0; i<Nn; i++){      // Take the inverse
      var = 1/vol[i];
      grad[i][0] *= var;
      grad[i][1] *= var;
      grad[i][2] *= var;
    }
    
    // 6 is adjustment factor
    eps = H1_Semi_Norm( sol) * percent/(100. * sqrt(NTR)) * 6;
    printf("||u||*percent/(100*sqrt(NTR)) = %4.2f*%4.1f/(100*%4.1f) = %f\n",
	   H1_Semi_Norm( sol),percent, sqrt(NTR), eps);    
    
    double grad_sol[3];              // nodal basis gradients and sol grad
    real middle[3], grad_phi[4][3];
    real D[3][3];                    // elasticity matrix
    double err_est = 0.;
    double sum = 0.;

    for ( i=0; i< NTR; i++){         // determine the tetrah. to be refined
    
      grad_nodal_basis(i, grad_phi);
      for(j=0; j<3; j++){            // for the 3 coordinates
	grad_sol[j] = 0.;
	for(k=0; k<4; k++)           // for the 4 nodes
	  grad_sol[j] += sol[ TR[i].node[k]] * grad_phi[k][j];
      }
      
      GetMiddle(i, middle);
      // compute the stress error in grad_phi[0] (this is a vector)
      for(j=0; j<3; j++){            // for the 3 coordinates
	grad_phi[0][j] = 0.;
	for(k=0; k<4; k++)           // for the 4 nodes
	  grad_phi[0][j] += grad[TR[i].node[k]][j];
      }

      func_K(middle, D, TR[i].atribut);
      // Compute : grad_phi[2] = D grad_phi[0]
      for(j=0; j<3; j++){   // for the 3 coordinates 
	grad_phi[2][j] = 0.;
	for(k=0; k<3; k++)
	  grad_phi[2][j] += D[j][k] * grad_sol[k];
      }
      
      for(j=0; j<3; j++)
	grad_phi[0][j] = grad_phi[2][j] - grad_phi[0][j]/4.;
      
      func_K_Inv(middle, D, TR[i].atribut);

      // Compute : grad_phi[1] = D grad_phi[0]
      for(j=0; j<3; j++){   // for the 3 coordinates 
	grad_phi[1][j] = 0.;
	for(k=0; k<3; k++)
	  grad_phi[1][j] += D[j][k] * grad_phi[0][k];
      }
      
      // Compute : err_est = sqrt(grad_phi[0] . grad_phi[1] * volume)
      err_est = 0.;
      for(j=0; j<3; j++)
	err_est += grad_phi[0][j]*grad_phi[1][j];
      
      err_est = sqrt( err_est * volume(i));
      
      if (err_est > est_max) est_max = err_est;
      if (err_est < est_min) est_min = err_est;
      
      if (err_est > eps) array[i] = 1;     // refine
      else array[i] = 0;                   // don't refine
   
      //    printf("%10.8f\n", err_est);
      sum += err_est*err_est;
    }    
    
    printf("Min local error est = %10.8f, max = %10.8f\n", est_min, est_max);
    printf("sum = %f\n", sqrt(sum)/2);
    
    delete [] grad;
    delete [] vol;
  }
}


//============================================================================
// compute int_Omega(N^T D S N) d Omega sol and A - in vol (as given in ZZ)
//============================================================================
void Method::CompGradVert(int i, double *sol, real grad[][3], real *vol){
  real grad_phi[4][3], grad_sol[3]; // nodal basis gradients and sol grad
  real middle[3], D[3][3];   
  int j, k;

  grad_nodal_basis(i, grad_phi);
  for(j=0; j<3; j++){   // for the 3 coordinates
    grad_sol[j] = 0.;
    for(k=0; k<4; k++)  // for the 4 nodes
      grad_sol[j] += sol[ TR[i].node[k]] * grad_phi[k][j];
  }

  GetMiddle(i, middle);
  func_K(middle, D, TR[i].atribut);

  // grad_phi[0] = D grad_sol
  for(j=0; j<3; j++){          // for the 3 coordinates 
    grad_phi[0][j] = 0.;
    for(k=0; k<3; k++)
      grad_phi[0][j] += D[j][k] * grad_sol[k];
  }

  real v = volume(i)/4;
  for(j=0; j<4; j++){          // add the contributions in the 4 nodes
    for(k=0; k<3; k++)         // for the 3 coordinates
      grad[ TR[i].node[j]][k] += v*grad_phi[0][k];
    vol[ TR[i].node[j]] += v;
  }
}

//============================================================================
