#ifndef _METRIX_
#define _METRIX_

#include "definitions.h" 

#if DOM_DECOMP == ON
  class Subdomain;
#endif


class Matrix{

  protected :
  
    int *V;
    int *PN;

    real *A;
    int dimension;
    int *Dir;
    int dim_Dir;
    int *Atribut;
    
    #if DOM_DECOMP == ON
      Subdomain *Subd;
    #endif
    
  public :
    Matrix(){}

    void InitMatrix(int *V, int *PN, real *A,int dimension);
    void InitMatrixAtr(int *Dir, int dim_Dir, int *Atribut);    
    #if DOM_DECOMP == ON
      void InitSubdomain(Subdomain *S) { Subd = S;}
    #endif
  
    void Action(double *v1, double *v2);
    void ActionS(double *v1, double *v2);
    void Action(double *v1, double *v2, int begin, int end);
    void TransposeAction(int dim, double *v1, double *v2);
    void Multiplication(Matrix *C, Matrix *B);
    
    void Gauss_Seidel_forw(double *v1, double *w);
    void Gauss_Seidel_back(double *v1, double *w);
};

#endif
