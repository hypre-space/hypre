#ifndef METHOD_MIXED
#define METHOD_MIXED

#include <iostream.h>
#include <fstream.h>

#include "definitions.h"
#include "Mesh_mixed.h"
#include "BlockMatrix.h"
#include "Method.h"

class MethodMixed : public MeshMixed, public Method{

    friend class Subdomain;

  private : 
    
    real **A, **B;                   // used for storing the matrices (A & B)
    BlockMatrix *GLOBAL;
    double *b;
     
  public :
 
    MethodMixed() {}
    MethodMixed(char *f_name);
     
    void Create_level_matrices();
 
    // Initialize the global stiffness matrices A, B and RHS b1 & b2 
    void Global_Init_A_B_b(real *A, real *B, double *b1, double *b2);
    void Mixed_LM(int num_tr, real *A, real *B, double *b1, double *b2);

    void phi_RT0(int num_tr, int i, real coord[3], real result[3]);
    real div_phi_RT0(int num_tr, int i);

    void Init_Dir(int l, double *v1);
    void Null_Dir(int l, double *v1);
    void Check_Dir(int l, double *v1);
    
    void Solve(int Refinement);
    void gmres(int n,int &nit,double *x,double *b, BlockMatrix *A);
    void inprod(int n, double *, double *, double &);

    double error_L2_p( double *sol);
    double error_L2_u( double *sol);

    void PrintLocalMatrices();
    void Print_Mixed_LM(int num_tr, ofstream &out);
    //double error_Hdiv( double *sol);
    //double error_max_p(double *sol);
    //double error_max_u(double *sol);

    //============== Local Refinement ========================================
    int Refine_F(int *array); 
    //========================================================================
};

#endif
