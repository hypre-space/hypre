#ifndef _BLOCK_METRIX_
#define _BLOCK_METRIX_

#include "definitions.h" 
#include "Matrix.h"

class BlockMatrix{

  protected :
    real *A, *B;
    int  *V_A,  *V_B;
    int *PN_A, *PN_B;
    int dim, dim1, dim2;
  
  public :
    BlockMatrix(){}
  
    void InitMatrix(int *VA, int *VB, int *PNA, int *PNB,
		    real *A, real *B, int d1, int d2);
    void Action(double *v1, double *v2);
    void Print();
};

#endif
