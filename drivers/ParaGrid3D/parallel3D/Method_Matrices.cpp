#include "Method.h"


void Method::Mass( Matrix *Masa){
  int i, j, k, ii, jj, start, end, num_tr;
  double den;
  double *MM=new double [DimPN[level]];
  
  end = DimPN[level];
  for(i=0; i<end; i++)
    MM[i]=0;
  
  for(num_tr=0; num_tr<NTR; num_tr++){
    den = volume(num_tr)/20.;
    for(i=0;i<4;i++){
      for(j=0;j<4;j++){
	ii = TR[num_tr][i];
	jj = TR[num_tr][j];
	start = V[level][ii];
	end   = V[level][ii+1];
	for(k=start; k<end; k++){
	  if(jj==PN[level][k]){
	    if(i==j)
	      MM[k]+=2*den;
	    else
	      MM[k]+=den;
            break;
	  }
        }
      }   
    }
  }
  Masa->InitMatrix(V[level], PN[level], MM, NN[level], &Dir, 
		   dim_Dir[level], &Atribut);
}
