#include "Mesh_mixed.h"
#include <stdio.h>
#include <iostream.h>
#include <math.h>

//============================================================================

MeshMixed::MeshMixed(char *f_name):Mesh(f_name){
  InitializeVPN( 0);
}

//============================================================================

void MeshMixed::InitializeVPN(int l){
  int dim, i, j, end, *V;

  V_A[l] = new int[NF  + 1];
  V_B[l] = new int[NTR + 1];

  //=============================
  V = V_A[l];
  end = NF; 
  dim = V[0] = 0; 
  for(i=0; i<end; i++){
    if (F[i].tetr[1] > -1)    // inner face 
      dim += 7;
    else
      switch (F[i].tetr[1]){
      case DIRICHLET:
	dim += 4;
	break;
      case NEUMANN:
	break;
      case ROBIN:               // not yet implemented 
	printf("Robin bdr is not yet implemented. Exit.\n");
	exit(1);
	break;
      }
    V[i+1] = dim;
  }
  DimPN_A[l] = dim;
  PN_A[l]    = new int[dim];
  for(i=0; i<dim; i++) PN_A[l][i] = -1;

  //==============================
  V = V_B[l]; 
  dim = V[0] = 0; 
  for(i=0; i<NTR; i++){
    for(j=0; j<4; j++)          // for the 4 faces
      if (F[TR[i].face[j]].tetr[1] != NEUMANN)
	dim++;
    V[i+1] = dim;
  }
  DimPN_B[l] = dim;
  PN_B[l]    = new int[dim];
  for(i=0; i<dim; i++) PN_B[l][i] = -1;
  
  //==============================
  for(i=0; i<NTR; i++){
    // For the 4 faces. First initialize matrix A. ROBIN bdr not yet impl.
    if ( F[ TR[i].face[0] ].tetr[1] != NEUMANN )  
      add_new_A( TR[i].face[0], TR[i].face[1], TR[i].face[2], TR[i].face[3],
		 V_A[l][TR[i].face[0]], V_A[l][TR[i].face[0]+1], PN_A[l]);
    
    if ( F[ TR[i].face[1] ].tetr[1] != NEUMANN )
      add_new_A( TR[i].face[1], TR[i].face[0],  TR[i].face[2], TR[i].face[3],
		 V_A[l][TR[i].face[1]], V_A[l][TR[i].face[1]+1], PN_A[l]);
 
    if ( F[ TR[i].face[2] ].tetr[1] != NEUMANN )
      add_new_A( TR[i].face[2], TR[i].face[0],  TR[i].face[1], TR[i].face[3], 
		 V_A[l][TR[i].face[2]], V_A[l][TR[i].face[2]+1], PN_A[l]);
    
    if ( F[ TR[i].face[3] ].tetr[1] != NEUMANN )
      add_new_A( TR[i].face[3], TR[i].face[0],  TR[i].face[1], TR[i].face[2], 
		 V_A[l][TR[i].face[3]], V_A[l][TR[i].face[3]+1], PN_A[l]);

    add_new_B( TR[i].face[0], TR[i].face[1], TR[i].face[2], TR[i].face[3],
	       V_B[l][i], V_B[l][i+1], PN_B[l]);
  }
}

//============================================================================
// From start to end (in array PN) add e0, e1, e2, e3 if not there. 
//============================================================================
void MeshMixed::add_new_A(int e0, int e1, int e2, int e3, 
			  int start,int end,int *PN){
  int i;

  for(i=start; i<end; i++){
    if (PN[i]==e0) break;
    if (PN[i]==-1){ PN[i] = e0; break;}
  }
  for(i=start; i<end; i++){
    if (PN[i]==e1) break;
    if (PN[i]==-1){ PN[i] = e1; break;}
  }
  for(i=start; i<end; i++){
    if (PN[i]==e2) break;
    if (PN[i]==-1){ PN[i] = e2; break;}
  }
  for(i=start; i<end; i++){
    if (PN[i]==e3) break;
    if (PN[i]==-1){ PN[i] = e3; break;}
  }
}

//============================================================================
// From start to end (in array PN) add e0, e1, e2 if not on NEUMANN bdr
// and if not there (ROBIN not yet implemented). 
//============================================================================
void MeshMixed::add_new_B(int e0, int e1, int e2, int e3,
			  int start, int end, int *PN){
  int i;
  
  if (F[e0].tetr[1] != NEUMANN)
    for(i=start; i<end; i++){
      if (PN[i]==e0) break;
      if (PN[i]==-1){ PN[i] = e0; break;}
    }
  if (F[e1].tetr[1] != NEUMANN)
    for(i=start; i<end; i++){
      if (PN[i]==e1) break;
      if (PN[i]==-1){ PN[i] = e1; break;}
    }
  if (F[e2].tetr[1] != NEUMANN)
    for(i=start; i<end; i++){
      if (PN[i]==e2) break;
      if (PN[i]==-1){ PN[i] = e2; break;}
    }
  if (F[e3].tetr[1] != NEUMANN)
    for(i=start; i<end; i++){
      if (PN[i]==e3) break;
      if (PN[i]==-1){ PN[i] = e3; break;}
    }
}

//============================================================================

int MeshMixed::ind_ij(int *V, int *PN, int i, int j){
  int k;
  for(k=V[i]; k<V[i+1]; k++)
    if (PN[k] == j)
      return k;
  return -1;                                                 // for not found
}

//============================================================================
