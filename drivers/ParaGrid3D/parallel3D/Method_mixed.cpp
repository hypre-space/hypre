#include "Method_mixed.h"
#include "definitions.h"

#include <sys/time.h>           // these two are for measuring the time
#include <sys/resource.h>

#include <iostream.h>
#include <fstream.h>

#if OUTPUT_LM == ON            // used for printing the local matrices
  extern FILE *plotLM;
#endif

//============================================================================
// y = ax + pb
//============================================================================
void axpby(int n, double a, double *x, double p, double *b, double *y);

//============================================================================

MethodMixed::MethodMixed(char *f_name): Mesh(f_name), MeshMixed(f_name){
    A   = new p_real[LEVEL];
    B   = new p_real[LEVEL];

    b   = new double[NF+NTR];

    GLOBAL = new BlockMatrix[LEVEL];

    Create_level_matrices();
}

//============================================================================

void MethodMixed::Create_level_matrices(){
  delete []b;

  if ( level !=0 ) delete [] A[level-1];
  A[level] = new real[DimPN_A[level]];
  B[level] = new real[DimPN_B[level]];
  b        = new double[ NF + NTR ];

  Global_Init_A_B_b(A[level], B[level], b, b + NF);

  GLOBAL[level].InitMatrix(V_A[level], V_B[level], PN_A[level], PN_B[level],
			   A[level], B[level], NF, NTR);
}

//============================================================================

void MethodMixed::Global_Init_A_B_b(real *A, real *B, 
				    double *b1, double *b2){
  int i;
  
  #if OUTPUT_LM == ON
    printf("Output written in file ~/output/element_matrix\n");
    plotLM = fopen("~/output/element_matrix", "w+");
  #endif

  for(i=0; i< NF; i++)            b1[i] = 0.;
  for(i=0; i<NTR; i++)            b2[i] = 0.;
  for(i=0; i<DimPN_A[level]; i++) A[i] = 0.; 
  for(i=0; i<DimPN_B[level]; i++) B[i] = 0.;

  for(i=0; i<NTR; i++)
    Mixed_LM( i, A, B, b1, b2);

  #if OUTPUT_LM == ON
    fclose(plotLM);
  #endif  

}

//============================================================================

void MethodMixed::Solve(int Refinement){
  int dim = NF + NTR;
  double *solution=new double[dim];
  int i, *array, iterations, refine;

  #if EXACT == ON
    double *zero;
  #endif

  struct rusage t;   // used to measure the time
  long int time;
  double t1;

  for(i=0; i<dim; i++) solution[i]=0.;

  do {
    Init_Dir(level, solution);
    Null_Dir(level, b);

    getrusage( RUSAGE_SELF, &t);
    time = 1000000 * (t).ru_utime.tv_sec + ( t).ru_utime.tv_usec;

    iterations = 500;
    gmres(dim, iterations, solution, b, &GLOBAL[level]);
    // GLOBAL[level].Print();

    getrusage( RUSAGE_SELF, &t);
    time = 1000000 * (t).ru_utime.tv_sec + ( t).ru_utime.tv_usec - time;
    t1 = time/1000000.;
    printf("Elapsed time in seconds : %12.6f\n\n", t1); 
    
    #if EXACT == ON
      zero = new double[dim];
      
      for(i=0; i<dim; i++) zero[i] = 0.;
      printf("|| p - p_h ||_L2 = %8.6f,  %5.3f%%\n", 
             error_L2_p(solution), 100*error_L2_p(solution)/error_L2_p(zero));
      printf("|| u - u_h ||_L2 = %8.6f,  %5.3f%%\n", 
             error_L2_u(solution), 100*error_L2_u(solution)/error_L2_u(zero));
      
      delete [] zero;
    #endif
      
    printf("Solution output is not generated\n");
    
    array = new int[NTR];
    cout << "Do you want another refinement (1/0) : "; cin >> refine;
    for(i=0; i<NTR; i++) array[i] = 1;
    
    if (refine){
      LocalRefine( array);
      printf("Total volume = %f\n", volume());
      
      InitializeVPN(level);
      Create_level_matrices();   

      delete [] solution;
      dim = NF + NTR;
      solution = new double[dim];
      for(i=0; i<dim; i++) solution[i]=0.;
    }
    delete []array;
  }
  while (refine);
}

//============================================================================

void MethodMixed::PrintLocalMatrices(){
  int i, j, *array, refine;

  delete []b;
  do{
    cout << "Output written in file ~/output/element_matrix\n";
    cout << "Output written in file ~/output/element_node\n";
    ofstream out_lm("~/output/element_matrix");
    ofstream out_el("~/output/element_node");
    
    out_el << NTR << "  " << NF << endl;
    for(i=0; i<NTR; i++){
      Print_Mixed_LM( i, out_lm);

      for(j=0; j<4; j++)
	out_el << TR[i].face[j] << "  ";
      out_el << endl;
    }

    array = new int[NTR];
    cout << "Do you want another refinement (1/0) : "; cin >> refine;
    for(i=0; i<NTR; i++) array[i] = 1;
    
    if (refine){
      LocalRefine( array);
      printf("Total volume = %f\n", volume());
      
      InitializeVPN(level);
    }
    delete []array;

    out_lm.close();
    out_el.close();
  }
  while (refine);
}

//============================================================================


void MethodMixed::Init_Dir(int l, double *v1){
  real m_face[3];
 
  for(int i=0; i<NF; i++)
    if (F[i].tetr[1] == NEUMANN){
      GetMiddleFace(i, m_face);
      v1[i] = func_gn( m_face);
    }
}

//============================================================================

void MethodMixed::Check_Dir(int l, double *v1){
  real m_face[3];

  for(int i=0; i<NF; i++)
    if (F[i].tetr[1] == NEUMANN){
      GetMiddleFace(i, m_face);
      if (v1[i] != func_gn( m_face))
	printf("Wrong Dirichlet value.\n");
    }
}

//============================================================================
// Null the Diriclet parts of vector v1 on level l.
//============================================================================
void MethodMixed::Null_Dir(int l, double *v1){  
  for(int i=0; i<NF; i++)
    if (F[i].tetr[1] == NEUMANN)
      v1[i] = 0.;
}

//============================================================================
/*
void MethodMixed::gmres(int n,int &nit,double *x,double *b,int Kmax, 
			BlockMatrix *A){
  int i, j, k, m;  
  double H[Kmax+2][Kmax+1], HH[Kmax+1][Kmax+1]; 
  double r[n], ap[n], xNew[n], inpr;
  int iter;
  
  double rNorm, RNorm, y[Kmax], h1[Kmax];
  
  if (Kmax > n) Kmax=n; 
  double *q[Kmax+1];
  for (i=1; i<=Kmax; i++) q[i] = new double[n]; 

  for (iter = 0; iter<nit; iter++)  {
    A->Action(x, ap);   	      	       //  ap   = Ax
    axpby( n, 1., b, -1., ap, r);              //  r    = b-Ax
    inprod( n, r, r, inpr);                    //  inpr = r . r
    H[1][0] = sqrt(inpr); 
    //if ( fabs(H[1][0]) < TOLERANCE) break;

    for(k=1; k<=Kmax; k++) {
      axpby(n, 0., r, 1.0/H[k][k-1], r, q[k]); //  q[k] = 1.0/H[k][k-1] r
      A->Action( q[k], ap);                    //  ap   = A q[k]
      axpby(n,0., ap, 1.0, ap, r);             //  r    = ap 
      for (i=1; i<=k; i++) {
	inprod( n, q[i], ap, H[i][k]);         //  H[i][k] = q[i] . ap
	axpby(n, 1.0, r, -H[i][k], q[i], r);   //  r       = r - H[i][k] q[i]
      }
      inprod( n, r, r, inpr);
      H[k+1][k]=sqrt(inpr);                    //  H[k+1][k] = sqrt(r . r) 
      
      //   Minimization of  || b-Ax ||  in K_k 
      for (i=1; i<=k; i++) {
	HH[k][i] = 0.0;
	for (j=1; j<=i+1; j++)
	  HH[k][i] +=  H[j][k] * H[j][i];
      }       
      h1[k] = H[1][k]*H[1][0];
      
      if (k != 1)
	for (i=1; i<k; i++) {
	  HH[k][i] = HH[k][i]/HH[i][i];
	  for (m=i+1; m<=k; m++)
	    HH[k][m] -= HH[k][i] * HH[m][i] * HH[i][i];
	  h1[k] -= h1[i] * HH[k][i];   
	}    
      y[k] = h1[k]/HH[k][k]; 
      if (k != 1)  
	for (i=k-1; i>=1; i--) {
	  y[i] = h1[i]/HH[i][i];
	  for (j=i+1; j<=k; j++)
	    y[i] -= y[j] * HH[j][i];
	}
      //   Minimization Done      
      axpby(n, 1.0, x, 0.0, x, xNew);           // xNew = x
      for (i=1; i<=k; i++)
	axpby(n, 1.0, xNew, y[i], q[i], xNew);  // xNew += y[i]*q[i] 
      
      rNorm = fabs(H[k+1][k]);
      if (rNorm < TOLERANCE) break;
    }
    A->Action(xNew, ap);                        // ap = A xNew
    axpby( n, 1.0, b, -1.0, ap, ap);            // ap = b - ap
    inprod( n, ap, ap, inpr);
    RNorm = sqrt(inpr);                         // RNorm = sqrt(ap . ap)
    
    if (rNorm < TOLERANCE) break;
    if (RNorm < TOLERANCE) break;
    axpby( n, 1.0, xNew, 0.0, xNew, x);         // x = xNew
    
    #if SILENT == OFF
      fprintf(stderr, "Iteration %d done!\n", iter);
      fprintf(stderr, "The current residual RNorm = %e\n", RNorm);
    #endif
  }
  
  nit = iter;
  A->Action( xNew, ap);
  axpby( n, 1.0, b, -1.0, ap, r);
  inprod( n, r, r, inpr);
  rNorm = sqrt(inpr);
  for (i=1; i<=Kmax; i++) delete(q[i]);

  printf("\nNumber of iterations %d\n", nit);
  printf("The final residual is %e\n\n", rNorm);
}
*/
//============================================================================
// There are some optimizations in this version
//============================================================================
void MethodMixed::gmres(int n,int &nit, double *x, double *b,
			BlockMatrix *A){
  int i, j, k, m;  
  double H[Kmax+2][Kmax+1], HH[Kmax+1][Kmax+1]; 
  double r[n], ap[n], xNew[n], inpr, var;
  int iter;
  
  double rNorm, RNorm, y[Kmax], h1[Kmax];
  
  double *q[Kmax+1];
  for (i=1; i<=Kmax; i++) q[i] = new double[n]; 

  for (iter = 0; iter<nit; iter++)  {
    A->Action(x, ap);   	      	       //  ap   = Ax
    inpr = 0.;
    for(m=0; m < n; m++){
      r[m] = b[m] - ap[m];                     //  r    = b-Ax
      inpr += r[m]*r[m];                       //  inpr = r . r
    }
    H[1][0] = sqrt(inpr); 
    //if ( fabs(H[1][0]) < TOLERANCE) break;

    for(k=1; k<=Kmax; k++) {
      var = 1.0/H[k][k-1];
      for(m=0; m <n; m++) q[k][m] = var*r[m];  //  q[k] = 1.0/H[k][k-1] r
      A->Action( q[k], ap);                    //  ap   = A q[k]
      for(m=0; m <n; m++) r[m] = ap[m];        //  r    = ap 
      for (i=1; i<=k; i++) {
	inprod(n, q[i], ap, H[i][k]);          //  H[i][k] = q[i] . ap
	var = -H[i][k];
	for(m=0;m<n;m++) r[m]+=var*q[i][m];    //  r       = r - H[i][k] q[i]
      }
      inprod(n, r, r, inpr);
      H[k+1][k]=sqrt(inpr);                    //  H[k+1][k] = sqrt(r . r) 
      
      //   Minimization of  || b-Ax ||  in K_k 
      for (i=1; i<=k; i++) {
	HH[k][i] = 0.0;
	for (j=1; j<=i+1; j++)
	  HH[k][i] +=  H[j][k] * H[j][i];
      } 
      
      h1[k] = H[1][k]*H[1][0];
      
      if (k != 1)
	for (i=1; i<k; i++) {
	  HH[k][i] = HH[k][i]/HH[i][i];
	  for (m=i+1; m<=k; m++)
	    HH[k][m] -= HH[k][i] * HH[m][i] * HH[i][i];
	  h1[k] -= h1[i] * HH[k][i];   
	}    
      y[k] = h1[k]/HH[k][k]; 
      if (k != 1)  
	for (i=k-1; i>=1; i--) {
	  y[i] = h1[i]/HH[i][i];
	  for (j=i+1; j<=k; j++)
	    y[i] -= y[j] * HH[j][i];
	}
      //   Minimization Done      
      m = k;
      rNorm = fabs(H[k+1][k]);
      if (rNorm < TOLERANCE) break;
    }

    for(i=0; i<n; i++){
      xNew[i] = x[i];                           // xNew = x
      for(j=1; j<=m; j++)
	xNew[i] += y[j]*q[j][i];                // xNew += y[i]*q[i] 
    }

    A->Action(xNew, ap);                        // ap = A xNew
    inpr = 0.;
    for(i=0; i<n; i++){
      ap[i] = b[i] - ap[i];                     // ap = b - ap
      inpr += ap[i]*ap[i];
      x[i]  = xNew[i];                          // x = xNew
    }
    RNorm = sqrt(inpr);                         // RNorm = sqrt(ap . ap)
    
    if (rNorm < TOLERANCE) break;
    if (RNorm < TOLERANCE) break;
    
    #if SILENT == OFF
      fprintf(stderr, "Iteration %d done!\n", iter);
      fprintf(stderr, "The current residual RNorm = %e\n", RNorm);
    #endif
  }
  
  nit = iter;
  A->Action( xNew, ap);
  inpr = 0.;
  for(i=0; i<n; i++){
    r[i] = b[i] - ap[i];
    inpr += r[i]*r[i];
  }

  rNorm = sqrt(inpr);
  for (i=1; i<=Kmax; i++) delete(q[i]);

  printf("\n The final residual is %e\n\n", rNorm);
}

//============================================================================

void MethodMixed::inprod(int n, double *a, double *b, double &result){
  result = 0.;
  for(int i=0; i<n; i++)
    result += a[i]*b[i];
}

//============================================================================
// Compute the discrete L2 norm of the error for p.
//============================================================================
double MethodMixed::error_L2_p( double *sol){
  real middle[3];
  double error = 0., *s = sol + NF;
  
  for(int num_tr=0; num_tr<NTR; num_tr++){ // for all tetrahedra
    GetMiddle(num_tr, middle);
    error+=(s[num_tr]-exact(middle))*(s[num_tr]-exact(middle))*volume(num_tr);
  }
  return sqrt(error);
}

//============================================================================

void exact_K_grad_p(real *c, real result[3]);

//============================================================================
// Compute the discrete L2 norm of the error for u.
//============================================================================
double MethodMixed::error_L2_u(double *sol){
  int i, k, num_tr;
  real m_edge[6][3], value[3];
  double error[3]={0.,0.,0.}, s[3], local[3], scale;
  
  for(num_tr=0; num_tr<NTR; num_tr++){ // for all triangles  
    local[0] = local[1] = local[2] =  0.;
    scale = volume(num_tr)/6;

    GetMiddleEdge(TR[num_tr].node[0], TR[num_tr].node[1], m_edge[0]);
    GetMiddleEdge(TR[num_tr].node[1], TR[num_tr].node[2], m_edge[1]);
    GetMiddleEdge(TR[num_tr].node[2], TR[num_tr].node[3], m_edge[2]);
    GetMiddleEdge(TR[num_tr].node[3], TR[num_tr].node[1], m_edge[3]);
    GetMiddleEdge(TR[num_tr].node[0], TR[num_tr].node[2], m_edge[4]);
    GetMiddleEdge(TR[num_tr].node[0], TR[num_tr].node[3], m_edge[5]);

    for(i=0; i<6; i++){                // for the 6 edges (quadrature)
      s[0] = s[1] = s[2] = 0.;
      for(k=0; k<4; k++){              // compute the approximate at edge i
	phi_RT0(num_tr, k, m_edge[i], value);
	s[0] += sol[TR[num_tr].face[k]]*value[0];
	s[1] += sol[TR[num_tr].face[k]]*value[1];
	s[2] += sol[TR[num_tr].face[k]]*value[2];
      }
      exact_K_grad_p(m_edge[i], value);
      local[0] += (s[0]-value[0])*(s[0]-value[0]);
      local[1] += (s[1]-value[1])*(s[1]-value[1]);
      local[2] += (s[2]-value[2])*(s[2]-value[2]);
    }
    error[0] += local[0]*scale;
    error[1] += local[1]*scale;
    error[2] += local[2]*scale;
  }
  return sqrt(error[0]) + sqrt(error[1]) + sqrt(error[2]); 
}

//============================================================================
