#ifndef _METRIX_
#define _METRIX_

#include "definitions.h" 

class Matrix{

  protected :
  
    int *V;
    int *PN;

    double *A;
    int dimension;
    int *Dir;
    int dim_Dir;
    int *Atribut;
    
  public :
    Matrix(){}

    void InitMatrix(int *V,int *PN,double *A,int dimension);
    void InitMatrixAtr(	int *Dir, int dim_Dir, int *Atribut);    
		  
    void Action(double *v1, double *v2);
    void TransposeAction(int dim, double *v1, double *v2);
    void Multiplication(Matrix *C, Matrix *B);
    
    void Gauss_Seidel_forw(double *v1, double *w);
    void Gauss_Seidel_back(double *v1, double *w);
};

#endif
