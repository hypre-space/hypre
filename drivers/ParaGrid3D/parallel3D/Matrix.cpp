#include "Matrix.h"
#include <vector.h>
#include <stdio.h>

#if DOM_DECOMP == ON
  #include "Subdomain.h"
#endif

//============================================================================

void Matrix::InitMatrix(int *VV,int *PNPN, real *AA, int dim){
  V=VV;
  PN=PNPN;
  A=AA;
  dimension = dim;
}

//============================================================================

void Matrix::InitMatrixAtr( int *DD, int DD_dim, int *Atr){
  Dir=DD;
  dim_Dir=DD_dim;
  Atribut=Atr;
}


//============================================================================
void Matrix::Action(double *v1, double *v2){
  int i, j, end;

  #if DOM_DECOMP == ON
    Subd->Update_Slave_Values( v1);
  #endif

  #pragma parallel local(i, j, end)
  {
    #pragma pfor
    for(i=0;i<dimension;i++)
      v2[i]=0;
    #pragma pfor
    for(i=0;i<dimension;i++){
      end   = V[i+1];
      for(j=V[i]; j < end; j++){
	v2[i] += A[j]*v1[PN[j]];
      } 
    }
  }

  #if DOM_DECOMP == ON
    Subd->Update( v2);
    Subd->Zero_Slave_Values(v1);
  #endif

  #pragma parallel local(i) 
  # pragma pfor
  for(i=0;i<dim_Dir;i++)
    v2[Dir[i]]=0.;
}


//============================================================================
// Sequential action.
//============================================================================
void Matrix::ActionS(double *v1, double *v2){
  int i, j, end;

  for(i=0;i<dimension;i++)
    v2[i]=0.;

  for(i=0;i<dimension;i++){
    end   = V[i+1];
    for(j=V[i]; j < end; j++){
      v2[i]+= A[j]*v1[PN[j]];
    } 
  }
  
  for(i=0;i<dim_Dir;i++)
    v2[Dir[i]]=0.;
}


//============================================================================
// This action is used for parallel computations. It operates on vector
// v2 from "begin" to "end".
//============================================================================
void Matrix::Action(double *v1, double *v2, int begin, int end){
  int i, j, e;
  
  for(i=begin; i<end; i++) 
    v2[i]=0;
  for(i=begin; i<end; i++){
    e = V[i+1];
    for(j=V[i]; j < e; j++)
      v2[i]+= A[j]*v1[PN[j]];
  }
  for(i=0;i<dim_Dir;i++)
    if (Dir[i]>=begin && Dir[i]<end)
      v2[Dir[i]]=0.;
}

//============================================================================
// dim is the dimension of the vector v2, i.e. "this" matrix has dimension
// dimension x dim

void Matrix::TransposeAction(int dim, double *v1, double *v2){
  int i,j,end;
  
  for(i=0;i<dim;i++)
    v2[i]=0.;

  for(i=0;i<dimension;i++){
    end = V[i+1];
    for(j=V[i]; j < end; j++)
      v2[PN[j]] += A[j]*v1[i];
  }

  for(i=0;i<dim_Dir;i++){
    if (Dir[i] >= dim) break;  // the rest is with bigger than dim indexes
    v2[Dir[i]] = 0.;
  }
}

//============================================================================
// B = this * C
//============================================================================
void Matrix::Multiplication(Matrix *C,Matrix *B){
  int i,j,k,s,found,ll=0,q=0, e1, end;

  B->V = new int[dimension+1];
  for(i=0;i<(dimension+1);i++)
    B->V[i] = 0;
  
  vector<int>    *PN_vec = new vector<int>[dimension];
  vector<double> *B_vec  = new vector<double>[dimension];
  
  for(i=0;i<dimension;i++){
    for(k=V[i]; k<V[i+1]; k++){
      
      if (C->Atribut[k]==DIRICHLET)
      	break;

      e1 = C->V[PN[k]+1];
      for(j=C->V[PN[k]]; j<e1;j++){
	found=0;
	end = PN_vec[i].size();
	for(s=0; s<end; s++){
	  if(C->PN[j]==PN_vec[i][s]){
	    B_vec[i][s]+=A[k]*(C->A[j]);
	    found=1;
	    break;
	  }
	}
	if(found==0){
	  PN_vec[i].push_back(C->PN[j]);
	  B_vec[i].push_back(A[k]*(C->A[j]));
	}
      }
    }
  }

  for(i=0; i<dimension; i++)
    ll += PN_vec[i].size();

  B->PN = new  int[ll];
  B->A  = new real[ll];

  for(i=0; i<dimension; i++){
    end = PN_vec[i].size();
    for(j=0; j< end; j++){
      B->PN[q] = PN_vec[i][j];
      B->A[q]  =  B_vec[i][j];
      q++;  
    }
  }

  B->V[0] = 0;
  end = dimension + 1;
  for(i=1 ;i < end; i++){
    B->V[i] = B->V[i-1] + PN_vec[i-1].size();
  }   
  
  B->Dir = Dir;
  B->dim_Dir =  dim_Dir;
  B->dimension = dimension;
  B->Atribut = Atribut;
  
  delete [] PN_vec;
  delete [] B_vec;
}


void Matrix::Gauss_Seidel_forw(double *v1, double *w){
  int i,j,end;
  double sum = 0.;

  for(i=0;i<dimension;i++){
    if(Atribut[i]!=DIRICHLET){       
      end=V[i+1];
      sum=0.;
      for(j=(V[i]+1); j<end ;j++)
	sum+=A[j]*v1[PN[j]];
         
      v1[i]=(w[i] - sum)/A[V[i]];  
    }
  }
}


void Matrix::Gauss_Seidel_back(double *v1, double *w){
  int i,j,end;
  double sum = 0.;
  
  for(i=dimension-1;i>=0;i--){
    if (Atribut[i]!=DIRICHLET){
      end=V[i+1];
      sum=0.;
      for(j=(V[i]+1); j<end ;j++){
	sum+=A[j]*v1[PN[j]];
      }
    
      v1[i]=(w[i] - sum)/A[V[i]];  
    }
  }
}

/*
void Matrix::Transpose(Matrix *AT){
  int i,j,ii,jj;

  AT->V = new NC[dimension];
  for(i=0;i<dimension;i++)
    AT->V[i].num=0;
  
  AT->PN = new int[V[dimension -1].p+V[dimension-1].num];
  AT->A = new double[dimension -1].p+V[dimension-1].num];
   
  AT->Dir = Dir;
  AT->dim_Dir =  dim_Dir;
  AT->dimension = dimension;
  AT->Atribut = Atribut;
  
  for(i=0;i<dimension;i++){
    for(j=V[i].p;j<(V[i].p+V[i].num);j++){
      if(i>PN[j]){
	for(ii=V[PN[j]].p;ii<(V[PN[j]].p+V[PN[j]].num);ii++){
	  if(ii==i)
	    
	}

      }
	
    }
    
  }

}

*/



