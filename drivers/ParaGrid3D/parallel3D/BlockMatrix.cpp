#include "BlockMatrix.h"
#include <stdio.h>


//============================================================================

void BlockMatrix::InitMatrix(int *VA, int *VB, int *PNA, int *PNB,
			     real *AA, real *BB, int d1, int d2){
  V_A  =  VA;
  V_B  =  VB;
  PN_A = PNA;
  PN_B = PNB;
  A    =  AA;
  B    =  BB;

  dim1 = d1;
  dim2 = d2;
  dim = dim1 + dim2;
}

//============================================================================

void BlockMatrix::Action(double *v1, double *v2){
  int i, j, end, var;

  for(i=0;i<dim;i++)
    v2[i] = 0.;
  
  for(i=0; i<dim1; i++){               // Action with A
    end = V_A[i+1];
    for(j=V_A[i]; j<end; j++)
      v2[i]+=A[j]*v1[PN_A[j]];
  }
  
  for(i=0; i<dim2; i++){
    end = V_B[i+1];
    var = dim1+i;
    for(j=V_B[i]; j<end; j++){
      v2[PN_B[j]] += B[j]*v1[var];     // Action with B^T
      v2[var]     += B[j]*v1[PN_B[j]]; // Action with B
    }
  }
}

//============================================================================
// This functions prints the global matrix (for testing on small problems).
//============================================================================
void BlockMatrix::Print(){
  FILE *plot;
  double *u = new double[dim], *v = new double[dim];
  double **LM = new p_double[dim];
  int i, j;

  for(i=0; i<dim; i++){
    LM[i] = new double[dim];
    u[i] = 0.;
  }

  for(i=0; i<dim; i++){
    u[i]=1.;
    Action(u, v);
    for(j=0; j<dim; j++)
      LM[j][i] = v[j];
    u[i]=0.;
  }

  plot=fopen("global_matrix","w+");
  printf("The global stiffness matrix is printed in file global_matrix\n");
  fprintf(plot,"dim1(# of edges) = %d, dim2(# of triangles) = %d\n\n", 
	  dim1, dim2); 
  for(i=0; i<dim; i++){
    for(j=0; j<dim; j++)
      fprintf(plot,"%7.3f", LM[i][j]);
    fprintf(plot,"\n");
    delete [] LM[i];
  }
  fclose(plot);

  delete [] LM;
  delete [] u;
  delete [] v;
}

//============================================================================
