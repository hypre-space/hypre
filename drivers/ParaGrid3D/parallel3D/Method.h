#ifndef METHOD
#define METHOD

#include "Mesh.h"
#include "definitions.h"
#include "Matrix.h"

class Subdomain;


class Method : public virtual Mesh{

  friend class Subdomain;

  private : 
    
    real **A, **A_HB, **I;  // used for storing the matrices (A & HM matrix)
    Matrix *HB, *GLOBAL;    // Global - corresponds to A, HB to A_HB
    Matrix *Interp;         // Interpolation matrix
    double *b;
    
    double *pressure;
     
  public :
 
    Method() {}
    Method(char *f_name);
    ~Method();             // Used in domain decomposition to destroy the
                           // starting mesh (and method dynamically alloc mem)

    void Create_level_matrices();
    void Global_Init_b(double *b);

    // Initialize the global stiffness matrix and RHS
    void Global_Init_A_b(real *A, double *b); 

    // LM is pointer to "pointers to 4 doubles" 
    void ComputeLocalMatrix(int num_tr, double **LM, double *b);

    //add local contr for RD problem
    void Reaction_Diffusion_LM(int num_tr, real *A, double *b);
    void Reaction_Diffusion_LM(int num_tr, real **LM, double *b);

    //add local contr for CD problem
    void Convection_Diffusion_LM(int num_tr, real *A, double *b);
    void Convection_Diffusion_LM(int num_tr, real **LM, double *b);

    //add convection to the local matrix LM; the convection vector is "conv"
    void Convection_LM(int num_tr, real LM[][4], real conv[3]);

    void Convection_LM(real *A);           //add convection to a RD problem
    void add_convection(int, int, int, int, real [3], real [][4], int);
    void add_convection_face(int, real [3], real [][4], int);

    void Init_Dir(int l , double *);
    void Null_Dir(int l , double *);
    void precon(int l, double *v1, double *v2);
    void inprod(int l, double *v1, double *v2, double &res);
    int   vvadd(int l, double *v1, double *v2, double cnst, double *f);
    int  vvcopy(int l, double *v1, double *v2);
    int  vvcopy(int l, double *v1, double alpha, double *v2);
    void    PCG(int l, int num_of_iter, double *y,double *b, Matrix *A);
    void     CG(int l, int num_of_iter, double *y,double *b, Matrix *A);
    void  gmres(int n, int &nit, double *x, double *b, Matrix *A);

    void Solve();
    void SolveConcentration();
    void Print_Solution(char *name, double *solution);

    void grad_nodal_basis(int tetr, real grad[][3]);
    void normal(real n[3], int tetr, int face);            // the length is S

    double error_En( double *sol);
    double error_L2( double *sol);
    double error_max(double *sol);

    double H1_Semi_Norm(double *x);

    //============== Local Refinement ========================================
    void Refine(int num_levels, int Refinement);
    void Refine_ZZ(double percent, double *sol, int *array);	
    void CompGradVert(int i, double *sol, real grad[][3], real *vol);
    //========================================================================

    void Interpolation(int l, double *v1, double *v2);
    //void Restriction(int l, double *v1, double *v2);

    void V_cycle_HB(int l,double *w,double *v);
    void V_cycle_MG(int l,double *w,double *v);
    //void Mass(int n,int l,Matrix *Masa);
    //void test(int l);

    void DomainSplit(int n, int *tr); // split the domain in n 
    void MetisDomainSplit(int np, idxtype *tr);// split using Metis
};


#endif

