#include <iostream.h>
#include <iomanip.h>
#include <stdio.h>
#include <math.h>
#include "Method.h"
#include "Matrix.h"


#include <sys/time.h>       // these twow are for measuring the time
#include <sys/resource.h>

#if DOM_DECOMP == ON
  #include "mpi.h"
  #include "extension_MPI.h"
#endif

#if WELL == ON
  #include "Well.h"
  int locate_well(int Atribut, real *coord);
#endif

#if CONCENTRATION == ON
  extern real convection[3];
  extern int Pressure;
#endif

#if OUTPUT_LM == ON
  extern FILE *plotLM;
#endif

//============================================================================

Method::Method(char *f_name):Mesh(f_name){
  A   = new p_real[LEVEL];
  b   = new double[NN[0]];
  
  GLOBAL = new Matrix[LEVEL];

  #if MAINTAIN_HB == ON
    A_HB   = new p_real[LEVEL];
    HB     = new Matrix[LEVEL];
  #endif

  I      = new p_real[LEVEL];
  Interp = new Matrix[LEVEL];
  
  Create_level_matrices();
}

//============================================================================

Method::~Method(){
  int i;

  // Free the Mesh allocated memory
  delete [] Dir;
  delete [] Atribut;
  // delete Z; delete TR;
  // delete E; delete F;

  if (MAINTAIN_HB == ON)
    for(i=0; i<level; i++){
      delete []  A[i]; delete [] A_HB[i+1];
      delete []  V[i]; delete []  VHB[i+1];
      delete [] PN[i]; delete [] PNHB[i+1];
    }
  else 
    if (MAINTAIN_MG == ON)
      for(i=0; i<level; i++){
	delete []  A[i];
	delete []  V[i];
	delete [] PN[i];
      }

  // TTT
  /*
  for(i=1; i<=level; i++){
    delete []  VI[i]; 
    delete [] PNI[i];
    delete []   I[i];
  }

  delete []  V[level];
  delete [] PN[level];
  delete []  A[level];
  
  delete [] Interp;
  delete [] I;
  #if MAINTAIN_HB == ON
    delete [] HB;
    delete [] A_HB;
  #endif
  delete [] GLOBAL;
  delete [] b;
  delete [] A;
  */
}

//============================================================================
// Init the matrix structure for the current level "level".
//============================================================================
void Method::Create_level_matrices(){
  int i;
  
  delete []b;
  if ( MAINTAIN_MG == OFF && level!=0 ) delete [] A[level-1];
  A[level]=new real[DimPN[level]];

  b = new double[NN[level]];
  Global_Init_A_b(A[level], b);

  GLOBAL[level].InitMatrix( V[level], PN[level], A[level], NN[level]);
  for(i=0; i<=level; i++) 
    GLOBAL[i].InitMatrixAtr( Dir, dim_Dir[i], Atribut);

  int e1 = NN[level] - NN[level-1], e2;
  if (level!=0){                                // create HB matrices
    #if MAINTAIN_HB == ON
      int j, end, s, ss;
      A_HB[level] = new real[DimPNHB[level]];
      end = NN[level] - NN[level-1];
      for(j=0; j<end; j++)
	for(s = VHB[level][j]; s<VHB[level][j+1]; s++){
	  e2 = V[level][NN[level-1]+j+1];
	  for(ss=V[level][NN[level-1]+j]; ss<e2; ss++)
	    if ( PN[level][ss] == (PNHB[level][s] + NN[level-1])){
	      A_HB[level][s] = A[level][ss];
	      break;
	    }
	}
      
      // We don't initialize the right dirichlet nodes here but in the HB
      // matrices we don't use matrix action. These matrices are used only for
      // smoothing.
      HB[level].InitMatrix(VHB[level], PNHB[level], A_HB[level], e1);
      for(i=0; i<=level; i++)
	HB[i].InitMatrixAtr( Dir+dim_Dir[i-1], dim_Dir[i]-dim_Dir[i-1], 
			     Atribut+NN[i-1]);
    #endif
    
    I[level] = new real[DimPNI[level]];
    e1 = NN[level-1];
    for(i=0; i<e1; i++)
      I[level][i] = 1.;
      
    e2 = DimPNI[level];
    for(i=NN[level-1]; i<e2; i++)
      I[level][i] = 1/2.;
    Interp[level].InitMatrix(VI[level], PNI[level], I[level], NN[level]);
    
    for(i=1; i<=level; i++)
      Interp[i].InitMatrixAtr(Dir, dim_Dir[i], Atribut);
  }
}


//============================================================================
// Init the Global matrix and b on the last level "level"
//============================================================================
void Method::Global_Init_A_b(real *A, double *b){
  int i, N;

  #if OUTPUT_LM == ON
    cout << "Output written in file ~/output/element_matrix\n";
    plotLM = fopen("../output/element_matrix", "w+");
  #endif

  N = NN[level];
  for(i=0; i<N; i++) b[i]=0.;
  N = DimPN[level];
  for(i=0; i<N; i++) A[i]=0.;
   
  for (i=0; i<NTR; i++){
    #if PROBLEM == 0 || PROBLEM == 1
      Reaction_Diffusion_LM( i, A, b);
    #elif PROBLEM == 2
      Convection_Diffusion_LM( i, A, b);
    #endif
  }

  #if WELL == ON
    cout << "Adding well contribution ... ";
    if (Pressure){
      Well w(this, 0., 400., 0.0, 2000., locate_well);
      w.add_contribution(level, A, b);
    }
    else{
      Well w(this, 0., 400., 0.0, 2000., locate_well, pressure);
      w.add_contribution(level, A, b);
    }
    cout << "done.\n";
  #endif

  #if OUTPUL_LM == ON
    fclose(plotLM);
  #endif  
}



//============================================================================
// Compute the gradients of the nodal basis functions for tetrahedron num_tr.
//============================================================================
void Method::grad_nodal_basis(int num_tr, real grad[][3]){
  int i;
  real *c[4], det_inv = 1./TR[num_tr].determinant( Z);

  for(i=0;i<4; i++)
    c[i] = GetNode(TR[num_tr].node[i]);

  // First index gives nodal basis function (0..3), second gives the 
  // coordinate (0..2).
  grad[0][0] = (-c[2][1]*c[3][2] + c[3][1]*c[2][2] -     
		 c[3][1]*c[1][2] + c[1][1]*c[3][2] -
		 c[1][1]*c[2][2] + c[2][1]*c[1][2]   )*det_inv;  
  grad[0][1] = (-c[2][2]*c[3][0] + c[3][2]*c[2][0] -     
		 c[3][2]*c[1][0] + c[1][2]*c[3][0] -
		 c[1][2]*c[2][0] + c[2][2]*c[1][0]   )*det_inv;
  grad[0][2] = (-c[2][0]*c[3][1] + c[3][0]*c[2][1] -     
		 c[3][0]*c[1][1] + c[1][0]*c[3][1] -
		 c[1][0]*c[2][1] + c[2][0]*c[1][1]   )*det_inv;

  grad[1][0] = ( c[3][1]*c[0][2] - c[0][1]*c[3][2] +     
		 c[0][1]*c[2][2] - c[2][1]*c[0][2] +
		 c[2][1]*c[3][2] - c[3][1]*c[2][2]   )*det_inv;  
  grad[1][1] = ( c[3][2]*c[0][0] - c[0][2]*c[3][0] +     
		 c[0][2]*c[2][0] - c[2][2]*c[0][0] +
		 c[2][2]*c[3][0] - c[3][2]*c[2][0]   )*det_inv;
  grad[1][2] = ( c[3][0]*c[0][1] - c[0][0]*c[3][1] +     
		 c[0][0]*c[2][1] - c[2][0]*c[0][1] +
		 c[2][0]*c[3][1] - c[3][0]*c[2][1]   )*det_inv;

  grad[2][0] = (-c[0][1]*c[1][2] + c[1][1]*c[0][2] -     
		 c[1][1]*c[3][2] + c[3][1]*c[1][2] -
		 c[3][1]*c[0][2] + c[0][1]*c[3][2]   )*det_inv;  
  grad[2][1] = (-c[0][2]*c[1][0] + c[1][2]*c[0][0] -     
		 c[1][2]*c[3][0] + c[3][2]*c[1][0] -
		 c[3][2]*c[0][0] + c[0][2]*c[3][0]   )*det_inv;
  grad[2][2] = (-c[0][0]*c[1][1] + c[1][0]*c[0][1] -     
		 c[1][0]*c[3][1] + c[3][0]*c[1][1] -
		 c[3][0]*c[0][1] + c[0][0]*c[3][1]   )*det_inv;

  grad[3][0] = ( c[1][1]*c[2][2] - c[2][1]*c[1][2] +     
		 c[2][1]*c[0][2] - c[0][1]*c[2][2] +
		 c[0][1]*c[1][2] - c[1][1]*c[0][2]   )*det_inv;  
  grad[3][1] = ( c[1][2]*c[2][0] - c[2][2]*c[1][0] +     
		 c[2][2]*c[0][0] - c[0][2]*c[2][0] +
		 c[0][2]*c[1][0] - c[1][2]*c[0][0]   )*det_inv;
  grad[3][2] = ( c[1][0]*c[2][1] - c[2][0]*c[1][1] +     
		 c[2][0]*c[0][1] - c[0][0]*c[2][1] +
		 c[0][0]*c[1][1] - c[1][0]*c[0][1]   )*det_inv;
}


//============================================================================
// Compute the "normal" vector to face "f" in tetrahedron tetr, where
// "normal" = normal * S (f is local 0..3 index of face in tetr)
//============================================================================
void Method::normal(real n[3], int tetr, int f){
  real *p[4], v[3][3], flag;
  int i;

  for(i=0; i<4; i++) 
    p[i] = Z[TR[tetr].node[(f+i)%4]].GetCoord();
 
  for(i=0; i<3; i++){
    v[i][0] = p[i][0] - p[i+1][0];
    v[i][1] = p[i][1] - p[i+1][1];
    v[i][2] = p[i][2] - p[i+1][2];
  }

  n[0] = v[1][1]*v[2][2] - v[1][2]*v[2][1];
  n[1] = v[1][2]*v[2][0] - v[1][0]*v[2][2];
  n[2] = v[1][0]*v[2][1] - v[1][1]*v[2][0];
    
  if ((v[0][0]*n[0] + v[0][1]*n[1] + v[0][2]*n[2])<0.) flag =  0.5;
  else                                                 flag = -0.5;
  
  for(i=0; i<3; i++) n[i] *= flag;
}

//============================================================================
// Init the RHS "b" on the last level "level".
//============================================================================
void Method::Global_Init_b(double *b){
  int i, num_tr;
  real *coord;
  double f[4], nom;

  for(i=0;i<NN[level];i++)
    b[i]=0.;
  
  for(num_tr=0; num_tr<NTR; num_tr++){ // for all tetrahedrons
    
    nom = volume(num_tr)/20;
    
    for(i=0; i<4; i++){
      coord = GetNode(TR[num_tr].node[i]); 
      f[i]  = func_f( coord);
    }

    for(i=0; i<4; i++)
      b[TR[num_tr].node[i]] += (2*f[i]+f[(i+1)%4]+f[(i+2)%4]+f[(i+3)%4])*nom;
  }
}

//============================================================================

void Method::precon(int l, double *v1, double *v2) { 
  #if PRECONDITIONER == 1
     for(int i=0; i<NN[l]; i++)
       v2[i] = v1[i];
  #elif PRECONDITIONER == 2
     V_cycle_HB( l, v1, v2);
  #elif PRECONDITIONER == 3
     V_cycle_MG(l, v1, v2);
  #endif
}                    

//============================================================================

void Method::inprod(int l, double *v1, double *v2, double &res){
  int n = NN[l];

  res = 0.;
  for(int i=0; i < n; i++)
     res += v1[i]*v2[i];
}

//============================================================================

int Method::vvadd(int l, double *v1, double *v2, double cnst, double *ff) {
  int i, n = NN[l];

  #pragma parallel local(i)
  #pragma pfor
  for( i=0; i<n; i++)
    ff[i] = v1[i] + cnst * v2[i];

  return(0);
}

//============================================================================

int Method::vvcopy(int l, double *v1, double *v2) {
  int i, n = NN[l];

  #pragma parallel local(i)
  #pragma pfor
  for( i= 0; i< n;i++) 
    v2[i]=v1[i];

  return(0);
}


//============================================================================

int Method::vvcopy(int l, double *v1, double alpha, double *v2) {
  int i, n = NN[l];

  #pragma parallel local(i)
  #pragma pfor
  for( i= 0; i< n;i++) 
    v2[i] = alpha*v1[i];

  return(0);
}


//============================================================================
// r is residual
// b is the right hand side
// d is the search vector
//============================================================================
void Method::PCG(int l, int num_of_iter,  double *y, double *b, Matrix *A){
  double *r = new double[NN[l]], *d=new double[NN[l]], *z=new double[NN[l]];
  double den0, den, nom, betanom, alpha, beta;
  int i;

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  for(i=0; i<NN[l]; i++)
    d[i] = 0.;

  // precon(l, b, y);                         //    y = B b 
  A->ActionS(y, r);                           //    r = A y
  vvadd(l, b, r, -1.0, r);                    //    r = b  - r
  precon(l,r,z);                              //    z = B r
  vvcopy(l,z, d);                             //    d = z
  inprod(l,z, r, nom);                        //  nom = z dot r
  A->ActionS(d, z);                           //    z = A d
  inprod(l,z, d, den0);                       // den0 = z dot d
  den = den0;
  
  if (nom <TOLERANCE) 
    return;

  if (den <= 0.0) {
    cout << "Negative or zero denominator in step 0 of PCG. Exiting!\n";
    exit(1);
  }
  
  // start iteration
  for(i= 0; i<num_of_iter ;i++) { 
    #if SILENT == OFF
      cout.precision(2);
      if (myrank == 0)
	cout << "Iteration : " << setw(3) << i 
	     << "    Norm : " << setw(6) << nom << endl;
    #endif  
    alpha = nom/den;
    vvadd(l,y, d, alpha, y);              //       y = y + alpha d
    vvadd(l,r, z,-alpha, r);              //       r = r - alpha z
    precon(l, r, z);                      //       z = B r
    inprod(l, r, z, betanom);             // betanom = r dot z
    
    if ( betanom <= TOLERANCE) {
      if (myrank == 0)
	cout << "Number of CG iterations: " << i << endl;
      break;
    }
    
    beta = betanom/nom;                   //  beta = betanom/nom
    vvadd(l, z, d, beta, d);              //     d = z + beta d
    A->ActionS(d, z);                      //     z = A d
    inprod(l, d, z, den);                 //   den = d dot z
    nom = betanom;
    
  } // end iteration
  delete [] r;
  delete [] d;
  delete [] z;
}


//============================================================================
void Method::Solve(){
  double *solution=new double [NN[level]];
  int i, *array, iterations, refine, Refinement;
  char FName1[128];

  #if EXACT == ON
    double *zero;
  #endif

  struct rusage t;   // used to measure the time
  long int time;
  double t1;

  for(i=0; i<NN[level]; i++) solution[i]=0.;

  do {
    for(i=0; i<NN[level]; i++) solution[i]=0.;
    iterations = 300;
    Init_Dir(level, solution);
    Null_Dir(level, b);

    getrusage( RUSAGE_SELF, &t);
    time = 1000000 * (t).ru_utime.tv_sec + ( t).ru_utime.tv_usec;

    #if CONVECTION == ON
      #if CONCENTRATION == ON
        if (Pressure)
	  if (level)
	    PCG(level, 3000, solution, b, &GLOBAL[level]);
	  else 
	    CG( level, 3000, solution, b, &GLOBAL[level]);
	else
	  gmres(NN[level], iterations, solution, b, &GLOBAL[level]);
      #else
	gmres(NN[level], iterations, solution, b, &GLOBAL[level]);
      #endif
    #else
      if (level)
	PCG(level, 3000, solution, b, &GLOBAL[level]);
      else 
	CG( level, 3000, solution, b, &GLOBAL[level]);
    #endif

    getrusage( RUSAGE_SELF, &t);
    time = 1000000 * (t).ru_utime.tv_sec + ( t).ru_utime.tv_usec - time;
    t1 = time/1000000.;

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0){
      //cout.precision(5);
      //cout << "Elapsed time in seconds : " 
      //     << setw(10) << t1 << "\n\n"; 

      #if EXACT == ON
        zero = new double[NN[level]];
	for(i=0; i<NN[level]; i++) zero[i] = 0.;

	cout << "The discrete En. norm error is : " 
	     << error_En( solution) << endl; 
	/*
	cout << "The discrete L^2 norm error is : "
             << error_L2( solution) << "  " << setw(5) << precision(3)
             << 100*error_L2(solution)/error_L2(zero) << endl;
	cout << "The discrete max norm error is : "
             << error_max(solution) << "  " << setw(5) << precision(3)
             << 100*error_max(solution)/error_max(zero) << endl;
	*/
	delete [] zero;
      #endif
    
      switch (OUTPUT_TYPE){
      case 1:
	// PrintMCGL(solution);
	// PrintSolBdr("../Visual/Visualization.off", solution);
	sprintf(FName1,"~/Visual/Visualization%d.off", 0);
	PrintMCGL_sequential(FName1, solution);
	// PrintMCGL(FName1, solution, 0);
	
	// MPI_Barrier(MPI_COMM_WORLD);
	break;
      case 2:
	printf("Output written in file ~/output/bdr_mesh.off\n");
	PrintBdr("../output/bdr_mesh.off"); 
	break;
      case 3:
	printf("Output written in file ~/output/scale_mesh\n");
	PrintMeshMaple("../output/scale_mesh.off"); //plot the mesh scaled
	break;
      case 4:
	printf("Output written in file ~/output/maple.off\n");
	// first argument 0, 1, 2 - for x, y, z, second x, y or z = second  
	PrintFaceMaple( solution, "../output/maple.off", 1, 0.);
	break;
      case 5:
	printf("Output written in file ~/output/plot_mtv\n");
	PrintCutMTVPlot(solution, "../output/plot_mtv", 0, 1, 0, -250.);
	break;
      case 0:
	// there are some more printing procedures defined in Mesh.h (P.V.)
	// PrintMesh();       //print TR_Z, TR_F, F_Z, F_E, Z_coordinates, E_Z
	// PrintEdgeStructure();//print connectivity information
	break;
      };

      cout << "Refinement before the splitting (1:ZZ, 2:F, 3:Uni, 4:Split): ";
      cout.flush();
      cin  >> Refinement;
    } 
    MPI_Bcast(&Refinement, 1, MPI_INT, 0, MPI_COMM_WORLD);

    refine = 1;
    array = new int[NTR];
    switch (Refinement){
    case 1:
      double percent;
      if (myrank == 0){
	cout << "Input desired percent error reduction (0 for exit) : ";
	cout.flush();
	cin  >> percent;
      }
      MPI_Bcast(&percent, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      Refine_ZZ(percent, solution, array);
      break;
    case 2:
      Refine_F(array, level, myrank, NTR, Z, TR);
      break;
    case 3:
      for(i=0; i<NTR; i++) array[i] = 1;
      break;
    case 4:
      refine = 0;
      break;
    }

    if (refine){
      LocalRefine( array);

      Create_level_matrices();
      double *new_sol = new double[NN[level]];
      Interp[level].ActionS(solution, new_sol);
      
      delete [] solution;
      solution = new_sol;
    }
    delete []array;
  }  
  while (refine);

  #if CONCENTRATION == ON
    pressure = solution;
    Pressure = 0;
    SolveConcentration(Refinement);
  #endif
}


//============================================================================
void Method::SolveConcentration(){
  #if CONCENTRATION == ON
  double *solution=new double [NN[level]];
  int i, *array, iterations, refine, Refinement;

  struct rusage t;   // used to measure the time
  long int time;
  double t1;

  Global_Init_A_b(A[level], b);
  for(i=0; i<NN[level]; i++) solution[i]=0.;
  
  do {
    iterations = 700;

    Init_Dir(level, solution);
    Null_Dir(level, b);

    getrusage( RUSAGE_SELF, &t);
    time = 1000000 * (t).ru_utime.tv_sec + ( t).ru_utime.tv_usec;

    gmres(NN[level], iterations, solution, b, 15, &GLOBAL[level]);
  
    getrusage( RUSAGE_SELF, &t);
    time = 1000000 * (t).ru_utime.tv_sec + ( t).ru_utime.tv_usec - time;
    t1 = time/1000000.;

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0){
      // printf("Elapsed time in seconds : %12.6f\n\n", t1); 
      
      switch (OUTPUT_TYPE){
      case 1:
        //PrintMCGL(solution);
        //break;
	// PrintSolBdr("../Visual/Visualization.off", solution);
	break;
      case 2:
	printf("Output written in file ~/output/bdr_mesh.off\n");
	PrintBdr("../output/bdr_mesh.off"); 
	break;
      case 3:
	printf("Output written in file ~/output/scale_mesh\n");
	PrintMeshMaple("../output/scale_mesh.off"); //plot the mesh scaled
	break;
      case 4:
	printf("Output written in file ~/output/maple.off\n");
	// first argument 0, 1, 2 - for x, y, z, second x, y or z = second  
	PrintFaceMaple( solution, "../output/maple.off", 1, 0.);
	break;
      case 5:
	printf("Output written in file ~/output/plot_mtv\n");
	PrintCutMTVPlot(solution, "../output/plot_mtv", 0, 1, 0, -250.);
	break;
      case 0:
	// there are some more printing procedures defined in Mesh.h (P.V.)
	// PrintMesh();      //print TR_Z, TR_F, F_Z, F_E, Z_coordinates, E_Z
	// PrintEdgeStructure();//print connectivity information
	break;
      };

      cout << "Refinement before the splitting (1:ZZ, 2:F, 3:Uni, 4:Stop): ";
      cin  >> Refinement;
    } 
    MPI_Bcast(&Refinement, 1, MPI_INT, 0, MPI_COMM_WORLD);

    refine = 1;
    array = new int[NTR];
    switch (Refinement){
    case 1:
      double percent;
      if (myrank == 0){
	cout << "Input desired percent error reduction (0 for exit) : ";
	cin  >> percent;
      }
      MPI_Bcast(&percent, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      Refine_ZZ(percent, solution, array);
      break;
    case 2:
      Refine_F(array, level, myrank, NTR, Z, TR);
      break;
    case 3:
      for(i=0; i<NTR; i++) array[i] = 1;
      break;
    case 4:
      refine = 0;
      break;
    }

    if (refine){
      LocalRefine( array);
      
      double *new_pressure = new double[NN[level]];  // first fix the pressure
      Interpolation(level, pressure, new_pressure);
      delete [] pressure;
      pressure = new_pressure;

      Pressure = 1;                                  // new 1
      Create_level_matrices();
      CG( level, 3000, pressure, b, &GLOBAL[level]); // new 2

      Pressure = 0;                                  // new 3
      Global_Init_A_b(A[level], b);                  // new 4
      double *new_sol = new double[NN[level]];
      Interp[level].ActionS(solution, new_sol);
      
      delete [] solution;
      solution = new_sol;
    }
    delete []array;
  } 
  while (refine);
  #endif
}


//============================================================================
// Init vector v1 on level l with the right values on its Dirichlet nodes.
//============================================================================
void Method::Init_Dir(int l, double *v1){
  for(int i=0; i<dim_Dir[l]; i++)
    v1[Dir[i]] = func_u0( Z[Dir[i]].GetCoord());
}


//============================================================================
// Null the Diriclet parts of vector v1 on level l.
//============================================================================
void Method::Null_Dir(int l, double *v1){
  for(int i=0;i<dim_Dir[l];i++)
    v1[Dir[i]]=0.;
}


//============================================================================

void Method::Interpolation(int l,double *v1,double *v2){
 int i, j, end;

 for(i=0;i<NN[l-1];i++)
   v2[i] = v1[i];

 for(i=NN[l-1];i<NN[l];i++){
   v2[i] = 0.;
   end = VI[level][i+1];
   for(j = VI[level][i]; j < end; j++){
     v2[i] += v1[PNI[level][j]]/2.;  
   }
 }
}

/*
//============================================================================

void Method::Restriction(int l,double *v2,double *v1){
 int i,j;

 for(i=0;i<NN[l-1];i++)
   v1[i]=v2[i];
 
 for(i=0;i<NN[l-1];i++){
   for(j=(V[l-1][i].p+1);j< V[l-1][i].p+V[l-1][i].num;j++){
     if(i>PN[l-1][j]){
     v1[i]+=v2[NEW[l-1][j]]/2.;
     v1[PN[l-1][j]]+=v2[NEW[l-1][j]]/2.;
     }
   }
 }
 Null_Dir(l-1,v1);
}
*/
//============================================================================

void Method::V_cycle_MG(int l,double *w,double *v){
  int i;
  double *residual=new double[NN[l]];
  double *residual1=new double[NN[l-1]];
  double *c1=new double[NN[l-1]];
  
  for(i=0;i<NN[l];i++) v[i]=0.;
  for(i=0;i<NN[l-1];i++) c1[i]=0.;
  
  GLOBAL[l].Gauss_Seidel_forw(v, w);
 
  GLOBAL[l].ActionS(v, residual);                     // residual  = A vec
  
  for(i=0;i<NN[l];i++)  
    residual[i] = w[i] - residual[i]; 
  
  //  Restriction(l,residual,residual1);
  Interp[l].TransposeAction(NN[l-1], residual, residual1);
  
  if (l>1)
    V_cycle_MG(l-1,residual1,c1);    
  else
    CG(l-1,10000,c1,residual1,&GLOBAL[l-1]);      
   
    
  //  Interpolation(l,c1,residual);
  Interp[l].ActionS(c1, residual);

  for(i=0;i<NN[l];i++)
    v[i] += residual[i];

  GLOBAL[l].Gauss_Seidel_back(v, w);
  
  delete []residual;
  delete []residual1;
  delete []c1;
}

//============================================================================

void Method::V_cycle_HB(int l,double *w, double *v){
  int i;
  double *residual = new double[NN[l]];
  double *residual1 = new double[NN[l-1]];
  double *residual2 = new double[NN[l]];
  double *c1=new double[NN[l-1]];
  double *c=new double[NN[l]];
 
  for(i=0;i<  NN[l];i++)  v[i] = 0.;
  for(i=0;i<NN[l-1];i++) c1[i] = 0.;

  HB[l].Gauss_Seidel_forw(v+NN[l-1], w+NN[l-1]);
  
  GLOBAL[l].ActionS(v, residual);                       // residual  = A v
  
  for(i=0;i<NN[l];i++)  
    residual[i] = w[i] - residual[i]; 
  
  //  Restriction(l,residual,residual1);
  Interp[l].TransposeAction(NN[l-1], residual, residual1);

  if (l>1)
    V_cycle_HB(l-1,residual1,c1);
  else
    CG(l-1, 1000, c1,residual1,&GLOBAL[l-1]);      
    
  Interp[l].ActionS(c1, c);
  
  for(i=0;i<NN[l];i++)
    v[i]+=c[i];

  double *vec1 = new double[NN[l]];
 
  GLOBAL[l].ActionS(v, residual2);                  // residual  = A v
  
  for(i=0;i<NN[l];i++)  
    residual2[i] = w[i] - residual2[i]; 

  for(i=0;i<NN[l];i++)
    vec1[i]=0.;

  HB[l].Gauss_Seidel_back(vec1+NN[l-1],residual2+NN[l-1]);
  
  for(i=0;i<NN[l];i++)
    v[i] += vec1[i];

  delete []vec1;
  delete []residual;
  delete []residual1;
  delete []residual2;
  delete []c1;
  delete []c;
}

//============================================================================
/*
void Method::Print_Solution(char *name, double *solution){
  FILE *plot;
  int i;
  plot=fopen(name,"w+");

  double maximum = 0., minimum = 0.;
  for(i=0;i<NN[level-1];i++){
    if(solution[i] > maximum)
      maximum=solution[i];
    if(solution[i] < minimum)
      minimum=solution[i];
  }
  double scale = 0.5/(maximum - minimum);
  cout << "Scale = " << scale << "\n";
  
  fprintf(plot,"(geometry MC_geometry { : MC_mesh })\n");
  fprintf(plot,"(bbox-draw g1 yes)\n(read geometry {\n");
  fprintf(plot,"    appearance { +edge }\n    define MC_mesh\n\n");
  fprintf(plot,"# This OFF file was generated by MC.\nOFF\n");
  fprintf(plot, "%d %d %d\n", NN[level], NTR,0);
  
  for(i=0;i<NN[level];i++)
    fprintf(plot, "%f %f %f\n", Z[i][0],Z[i][1],scale*solution[i]);
  double s; 
  for(i=0; i<NTR; i++){
    s=(solution[TR[i][0]]+solution[TR[i][1]]+solution[TR[i][2]])/3.0;
    fprintf(plot, "%d %d %d %d %4.2f %4.2f %4.2f %4.2f\n", 
	           3, TR[i].node[0], TR[i].node[1], TR[i].node[2],
                   1-(maximum-s)/(maximum - minimum), 0.,
	    (maximum-s)/(maximum - minimum), 0.);
  }

  fprintf(plot,"\n})\n");
  fclose(plot);
}

//============================================================================

void Method::test(int l){
  int i;
  double *v = new double[NN[l]];
  double *w = new double[NN[l]];
  double *u = new double[NN[l]];

  for(i=0;i<NN[l];i++)
    v[i]=(1.e0 +i)/(17.e0+i);

  for(i=0;i<NN[l-1];i++)
    v[i]=0.;
  ActionS(l,v,w);
  Action_HB(l,v+NN[l-1],u+NN[l-1]);
  for(i=NN[l-1];i<NN[l];i++){
    cout<<"w["<< i <<"]=" <<w[i]<<"\n";
    cout<<"u["<< i <<"]=" <<u[i]<<"\n";
    w[i]-=u[i];
  }

  for(i=NN[l-1];i<NN[l];i++)
    cout<<"w["<< i <<"]=" <<w[i]<<"\n";

  delete []v;
  delete []w;
  delete []u;
}

//============================================================================

void Method::Interpolation_Matrix(int l,Matrix *Interp){
  int i,j;
  int *VV=new int[NN[l]];
  int *PNPN=new int[G[l]];
  double *AA=new double[G[l]];

  for(i=0;i<NN[l-1];i++){
    VV[i].p=i;
    VV[i].num=1;
    PNPN[i]=i;
    AA[i]=1.;
  }

  for(i=NN[l-1];i<NN[l];i++){
    VV[i].p=VV[i-1].p+VV[i-1].num;
    VV[i].num=2;
  }

  for(i=0;i<NN[l-1];i++){
    for(j=(V[l-1][i].p+1);j< V[l-1][i].p+V[l-1][i].num;j++){
      PNPN[V[NEW[l-1][j]].p]=i;
      PNPN[V[NEW[l-1][j]].p+1]=PN[l-1][j];
      AA[V[NEW[l-1][j]].p]=0.5;
      AA[V[NEW[l-1][j]].p+1]=0.5;
    }
  }

  Interp->InitMatrix(VV,PNPN,AA,NN[l],Dir[l],dim_Dir[l],Atribut);
}
*/


//============================================================================
// Compute the discrete L2 norm of the error. We may call this function when
// we have the exact solution : in file "functions.cpp" function exact.
//============================================================================
double Method::error_L2( double *solution){
  int i, num_tr;
  real *coord, c[3];
  double f[5], error = 0., vm;
  
  for(num_tr=0; num_tr<NTR; num_tr++){ // for all tetrahedrons
    for(i=0; i<4; i++){
      coord = GetNode(TR[num_tr].node[i]); 
      f[i]  = ((exact( coord)-solution[TR[num_tr].node[i]])*
	       (exact( coord)-solution[TR[num_tr].node[i]]));
    }
    GetMiddle(num_tr, c);
    vm =(solution[TR[num_tr].node[0]]+solution[TR[num_tr].node[1]]+
	 solution[TR[num_tr].node[2]]+solution[TR[num_tr].node[3]])*0.25;
    f[4] = (exact( c)-vm)*(exact( c)-vm);
    
    error += ((f[0]+f[1]+f[2]+f[3])*0.05 + f[4]*0.8)*volume(num_tr);
  }
  return sqrt(error);

}

//============================================================================
double Method::error_max( double *solution){
  int i;
  double max = 0.;
  for(i=0; i<NN[level]; i++){
    if (max < fabs(solution[i] - exact( Z[i].GetCoord())))
      max = fabs(solution[i] - exact( Z[i].GetCoord()));
  }
  return max;
}


//============================================================================
double Method::error_En( double *sol){
  int i;
  double *e=new double[NN[level]], *Ae=new double[NN[level]], res;

  for(i=0; i<NN[level]; i++)
    e[i] = exact(Z[i].GetCoord()) - sol[i];
  GLOBAL[level].ActionS(e, Ae);
  inprod(level, e, Ae, res);

  delete [] e;
  delete [] Ae;
  return sqrt(res);
}


//============================================================================
double Method::H1_Semi_Norm(double *x){
  int i, j, k;
  real grad_phi[4][3], middle[3], D[3][3];
  double norm = 0., local, grad[3];

  for ( i=0; i< NTR; i++){  // determine the tetrahedrons to be refined
    local = 0.;

    grad_nodal_basis(i, grad_phi);
    for(j=0; j<3; j++){   // for the 3 coordinates
      grad[j] = 0.;
      for(k=0; k<4; k++)  // for the 4 nodes
	grad[j] += x[ TR[i].node[k]] * grad_phi[k][j];
    }

    GetMiddle(i, middle);
    func_K(middle, D, TR[i].atribut);

    // Compute : grad_phi[1] = D grad
    for(j=0; j<3; j++){   // for the 3 coordinates 
      grad_phi[1][j] = 0.;
      for(k=0; k<3; k++)
	grad_phi[1][j] += D[j][k] * grad[k];
    }

    // Compute : norm = sqrt(grad . grad_phi[1] * volume)
    for(j=0; j<3; j++)
      local += grad[j]*grad_phi[1][j];
    norm += local * volume(i);
  }
  return sqrt(norm);
}


//============================================================================
void Method::CG(int l, int num_of_iter,  double *y, double *b, Matrix *A ){
  double *r=new double[NN[l]], *d=new double[NN[l]], *z=new double[NN[l]];
  double den0, den, nom, betanom, alpha, beta;
  int i, j;

  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  for(i=0; i<NN[l]; i++)
    d[i] = 0.;
  
  A->ActionS(y, r);                            // r = A y
  vvadd(l, b, r, -1.0, r);                    // r = b  - r

  for(i=0;i<NN[l];i++)
    z[i]=r[i];
                           
  vvcopy(l,z, d);                            // d = z
  inprod(l,z, r, nom);                       // nom = z dot r
  A->ActionS(d, z);                           // z = A d
  inprod(l,z, d, den0);                      // den0 = z dot d
  den = den0;
  
  if (nom <TOLERANCE) 
    return;
   
  if (den <= 0.0) {
    cout <<"den na CG"<<den<<
      "Negative or zero denominator in step 0 of CG. Exiting!" << "\n";
    exit(1);
  }

  // start iteration
  for(i= 0; i<num_of_iter ;i++) {   
    alpha= nom/den;
    
    vvadd(l,y, d, alpha, y);                   //  y = y + alpha d
    vvadd(l,r, z,-alpha, r);                   //  r = r - alpha z      
    
    for(j=0;j<NN[l];j++)
      z[j]=r[j];
    
    inprod(l,r, z, betanom);                   //  betanom = r dot z
    #if SILENT == OFF
      if (myrank == 0)
        cout << "Iteration : " << i << "    Norm : " << betanom << endl;
    #endif
    if ( betanom < TOLERANCE) {
      if (myrank == 0)
	cout << "Number of CG iterations: " << i << endl;
      break;
    }

    beta = betanom/nom;                         // beta = betanom/nom
    vvadd(l,z, d, beta, d);                     // d = z + beta d
    A->ActionS(d, z);                            // z = A d
    inprod(l,d, z, den);                        // den = d dot z
    nom = betanom;
  } // end iteration
  delete [] r;
  delete [] d;
  delete [] z;
}

//============================================================================
// y = ax + pb
//============================================================================
void axpby( int n, double a, double *x, double p, double *b, double *y){
  int i;

  for(i=0; i<n; i++)
    y[i] = a*x[i] + p*b[i];
}

//============================================================================
//#include "pgmres.cpp"

void Method::gmres(int n, int &nit, double *x, double *b, Matrix *A){
  int i, j, k, m;  
  double H[Kmax+2][Kmax+1], HH[Kmax+1][Kmax+1]; 
  double *r=new double[n], *ap=new double[n], *xNew=new double[n], inpr;
  int iter;
  
  double rNorm, RNorm, y[Kmax], h1[Kmax];
  
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  double *q[Kmax+1];
  for (i=1; i<=Kmax; i++) q[i] = new double[n]; 

  for (iter = 0; iter<nit; iter++)  {
    A->ActionS(x, ap);   	      	       //  ap   = Ax
    //vvadd(level, b, ap, -1., r);               //  r    = b-Ax
    //inprod(level, r, r, inpr);                 //  inpr = r . r
    inpr = 0.;
    for(m=0; m < n; m++){
      r[m] = b[m] - ap[m];
      inpr += r[m]*r[m];
    }

    H[1][0] = sqrt(inpr); 
    //if ( fabs(H[1][0]) < TOLERANCE) break;

    for(k=1; k<=Kmax; k++) {
      vvcopy(level, r, 1.0/H[k][k-1], q[k]);   //  q[k] = 1.0/H[k][k-1] r
      A->ActionS( q[k], ap);                    //  ap   = A q[k]
      vvcopy(level, ap, r);                    //  r    = ap 
      for (i=1; i<=k; i++) {
	inprod(level, q[i], ap, H[i][k]);      //  H[i][k] = q[i] . ap
	vvadd(level, r, q[i], -H[i][k], r);    //  r       = r - H[i][k] q[i]
      }
      inprod(level, r, r, inpr);
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

      //vvcopy(level, x, xNew);           // xNew = x
      //for (i=1; i<=k; i++)
      // vvadd(level, xNew, q[i], y[i], xNew);  // xNew += y[i]*q[i] 
      m = k;
      rNorm = fabs(H[k+1][k]);
      if (rNorm < TOLERANCE) break;
    }

    for(i=0; i<n; i++){
      xNew[i] = x[i];
      for(j=1; j<=m; j++)
	xNew[i] += y[j]*q[j][i];  
    }

    A->ActionS(xNew, ap);                        // ap = A xNew
    //vvadd(level, b, ap, -1., ap);               // ap = b - ap
    //inprod(level, ap, ap, inpr);
    inpr = 0.;
    for(i=0; i<n; i++){
      ap[i] = b[i] - ap[i];
      inpr += ap[i]*ap[i];
    }

    RNorm = sqrt(inpr);                         // RNorm = sqrt(ap . ap)
    
    if (rNorm < TOLERANCE) break;
    if (RNorm < TOLERANCE) break;
    vvcopy(level, xNew, x);                     // x = xNew
    
    #if SILENT == OFF
      if (myrank == 0){
	cout << "Iteration " << iter << " done!\n";
	cout << "The current residual RNorm = " << RNorm << endl;
      }
    #endif
  }
  
  nit = iter;
  A->ActionS( xNew, ap);
  //vvadd(level, b, ap, -1., r);
  //inprod(level, r, r, inpr);
  inpr = 0.;
  for(i=0; i<n; i++){
    r[i] = b[i] - ap[i];
    inpr += r[i]*r[i];
  }

  rNorm = sqrt(inpr);
  for (i=1; i<=Kmax; i++) delete(q[i]);

  // cout << "\n The final residual is " << rNorm << "\n\n";
  delete [] r;
  delete [] ap;
  delete [] xNew;
}


//============================================================================
// This function splits the domain into "n". The result is in tr - gives the
// the corresponding tetrahedron to which subdomain belongs. 
//============================================================================
void Method::DomainSplit(int n, int *tr){
  int i;
  real coord[3];

  for(i=0; i<NTR; i++){
    switch ( n){

    case 1:
      tr[i] = 0;
      break;

    case 2:
      GetMiddle(i, coord);
      if (coord[0] < 500.)
	tr[i] = 0;
      else 
	tr[i] = 1;
      break;

    case 4:
      GetMiddle(i, coord);
      if (coord[0] < 500.)
	if (coord[1] < 250.)
	  tr[i] = 0;
	else
	  tr[i] = 1;
      else
	if (coord[1] < 250.)
	  tr[i] = 2;
	else
	  tr[i] = 3;
      break;
      
    case 8:
      GetMiddle(i, coord);
      if (coord[0] > 500.)
	if (coord[1] < 250.)
	  if (coord[2] < 250.)
	    tr[i] = 0;
	  else
	    tr[i] = 1;
	else
	  if (coord[2] < 250.)
	    tr[i] = 2;
	  else
	    tr[i] = 3;
      else
	if (coord[1] < 250.)
	  if (coord[2] < 250.)
	    tr[i] = 4;
	  else
	    tr[i] = 5;
	else
	  if (coord[2] < 250.)
	    tr[i] = 6;
	  else
	    tr[i] = 7;
      break;
    } // end switch 
  }   // end for
}


//============================================================================
// This function splits the domain into "n". The result is in tr - gives the
// the corresponding triangle to which subdomain belongs. The splitting is 
// with Metis. 
//============================================================================
extern "C" 
void METIS_MeshToDual(int *, int *, idxtype *, int *, int *, 
		      idxtype *, idxtype *);
extern "C" 
void METIS_PartGraphRecursive(int *, idxtype *, idxtype *, idxtype *, 
			      idxtype *, int *, int *, int *, int *, 
			      int *, idxtype *);
extern "C" 
void METIS_PartGraphKway(int *, idxtype *, idxtype *, idxtype *, 
			 idxtype *, int *, int *, int *, int *, 
			 int *, idxtype *);
//============================================================================
void Method::MetisDomainSplit(int np, idxtype *tr){
  int i, j, n, ne = NTR, nn = NN[level], etype = 2, numflag = 0;
  int options[5], edgecut, wght = 0;
  idxtype *elmnts, *dxadj, *dadjncy, *vpntr = NULL, *epntr = NULL;
  
  n = 0;
  elmnts = new idxtype[4*NTR];
  for(i=0; i<NTR; i++)
    for(j=0; j<4; j++)
      elmnts[n++] = TR[i].node[j];
  dadjncy = new idxtype[4*NTR+1];
  dxadj   = new idxtype[NTR+1];

  // Create the mesh graph where the verteces are the elements and the
  // connections between them are the edges. The result - the arrays 
  // dxadj and dadjncy are passed to the splitting routine (~pmetis).
  if (np!=1){
    METIS_MeshToDual(&ne, &nn, elmnts, &etype, &numflag, dxadj, dadjncy);
    options[0] = 0;
    if (np <= 8)
      METIS_PartGraphRecursive(&ne, dxadj, dadjncy, vpntr, epntr, &wght, 
			       &numflag, &np, options, &edgecut, tr);
    else
      METIS_PartGraphKway(&ne, dxadj, dadjncy, vpntr, epntr, &wght, 
			  &numflag, &np, options, &edgecut, tr);
      
  }
  else
    for(i=0; i<NTR; i++) tr[i] = 0;

  delete [] dxadj;
  delete [] dadjncy;
  delete [] elmnts;
}

//============================================================================
// Here Refinement is 1 for uniform refinement and 2 for refinement based on
// user provided function.
//============================================================================
void Method::Refine(int num_levels, int Refinement){
  int i, j, *array, myrank;

  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  for(i=0; i<num_levels; i++){
    array = new int[NTR];
    switch (Refinement){
    case 1:
      for(j=0; j<NTR; j++) array[j] = 1;
      break;
    case 2:
      Refine_F(array, level, myrank, NTR, Z, TR);
      break;
    }
    LocalRefine( array);
    delete [] array;
  }
}

//============================================================================
