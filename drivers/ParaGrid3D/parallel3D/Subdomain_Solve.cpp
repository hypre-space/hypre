#include <iostream.h>
#include <stdio.h>
#include <math.h>
#include "Method.h"
#include "Matrix.h"
#include "Subdomain.h"

#include <sys/time.h>           // these two are for measuring the time
#include <sys/resource.h>

#include <mpi.h>
#include "extension_MPI.h"


#if OUTPUT_LM == ON            // used for printing the local matrices
  extern FILE *plotLM;
#endif

//============================================================================

void Subdomain::inprod(int l, double *v1, double *v2, double &res){
  res = 0.;

  for(int i=0; i<NN[l]; i++)
    res+=v1[i]*v2[i];

  double total;
  MPI_Allreduce(&res, &total, 1, MPI_DOUBLE, 
		MPI_SUM, MPI_COMM_WORLD);
  res = total;
}

//============================================================================

// r is residual
// b is the right hand side
// d is the search vector

void Subdomain::PCG(int l, int num_of_iter,  double *y, double *b, Matrix *A){
  double *r, *d, *z;
  double den0, den, nom, betanom, alpha, beta;
  int i;

  r = new double[NN[l]];
  d = new double[NN[l]];
  z = new double[NN[l]];

  for(i=0; i<NN[l]; i++)
    d[i] = 0.;

  // precon(l, b, y);                         //    y = B b 
  A->Action(y, r);                            //    r = A y
  vvadd(l, b, r, -1.0, r);                    //    r = b  - r
  precon(l,r,z);                              //    z = B r
  vvcopy(l,z, d);                             //    d = z
  inprod(l,z, r, nom);                        //  nom = z dot r
  A->Action(d, z);                            //    z = A d
  inprod(l,z, d, den0);                       // den0 = z dot d
  den = den0;
  
  if (nom <TOLERANCE) 
    return;

  if (den <= 0.0) {
    cout << "Negative or zero denominator in step 0 of PCG. Exiting!\n";
    exit(1);
  }
  
  for(i= 0; i<num_of_iter ;i++) { 
    fprintf(stderr,"Iteration : %3d    Norm : %e\n", i, nom);
    alpha = nom/den;
    vvadd(l,y, d, alpha, y);              //       y = y + alpha d
    vvadd(l,r, z,-alpha, r);              //       r = r - alpha z
    precon(l, r, z);                      //       z = B r
    inprod(l, r, z, betanom);             // betanom = r dot z
    
    if ( betanom <= TOLERANCE) {
      printf("Number of iterations: %d\n", i);
      break;
    }
    
    beta = betanom/nom;                   //  beta = betanom/nom
    vvadd(l, z, d, beta, d);              //     d = z + beta d
    A->Action(d, z);                      //     z = A d
    inprod(l, d, z, den);                 //   den = d dot z
    nom = betanom;
    
  }
  delete []z;
  delete []d;
  delete []r;
}


//============================================================================

void Subdomain::V_cycle_MG(int l,double *w,double *v){
  int i;
  double *residual=new double[NN[l]];
  double *residual1=new double[NN[l-1]];
  double *c1=new double[NN[l-1]];
  
  for(i=0;i<NN[l];i++) v[i]=0.;
  for(i=0;i<NN[l-1];i++) c1[i]=0.;
  
  GLOBAL[l].Gauss_Seidel_forw(v, w);

  GLOBAL[l].Action(v, residual);                     // residual  = A vec
  
  for(i=0;i<NN[l];i++)  
    residual[i] = w[i] - residual[i]; 
  
  //  Restriction(l,residual,residual1);
  Interp[l].TransposeAction(NN[l-1], residual, residual1);

  if (l>1)
    V_cycle_MG(l-1,residual1,c1);    
  else
    CG(l-1,10000,c1,residual1,&GLOBAL[l-1]);      
   
    
  //  Interpolation(l,c1,residual);
  Interp[l].Action(c1, residual);

  for(i=0;i<NN[l];i++)
    v[i] += residual[i];

  GLOBAL[l].Gauss_Seidel_back(v, w);

  delete []residual;
  delete []residual1;
  delete []c1;
}

//============================================================================

void Subdomain::V_cycle_HB(int l,double *w, double *v){
  int i;
  double *residual = new double[NN[l]];
  double *residual1 = new double[NN[l-1]];
  double *residual2 = new double[NN[l]];
  double *c1=new double[NN[l-1]];
  double *c=new double[NN[l]];
 
  for(i=0;i<  NN[l];i++)  v[i] = 0.;
  for(i=0;i<NN[l-1];i++) c1[i] = 0.;

  HB[l].Gauss_Seidel_forw(v+NN[l-1], w+NN[l-1]);
  
  GLOBAL[l].Action(v, residual);                       // residual  = A v
  
  for(i=0;i<NN[l];i++)  
    residual[i] = w[i] - residual[i]; 
  
  //  Restriction(l,residual,residual1);
  Interp[l].TransposeAction(NN[l-1], residual, residual1);

  if (l>1)
    V_cycle_HB(l-1,residual1,c1);
  else
    CG(l-1, 1000, c1,residual1,&GLOBAL[l-1]);      
    
  Interp[l].Action(c1, c);
  
  for(i=0;i<NN[l];i++)
    v[i]+=c[i];

  double *vec1 = new double[NN[l]];
 
  GLOBAL[l].Action(v, residual2);                  // residual  = A v
  
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

void Subdomain::CG(int l, int num_of_iter,  double *y, double *b, Matrix *A ){
  double *r=new double[NN[l]], *d=new double[NN[l]], *z=new double[NN[l]];
  double den0, den, nom, betanom, alpha, beta;
  int i;

  A->Action(y, r);                           // r = A y
  vvadd(l, b, r, -1.0, r);                   // r = b  - r

  vvcopy(l, r, d);                           // d = r
  inprod(l, r, r, nom);                      // nom = r dot r
  A->Action(r, z);                           // z = A r
  inprod(l, z, r, den0);                     // den0 = z dot r
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
    
    inprod(l, r, r, betanom);                  //  betanom = r dot r
    //printf("Iteration : %3d    Norm : %e\n", i, betanom);
    if ( betanom < TOLERANCE) {
      int myrank;
      MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
      if (myrank == 0)
	fprintf(stderr,"Number of CG iterations: %d\n", i);
      break;
    }

    beta = betanom/nom;                         // beta = betanom/nom
    vvadd(l, r, d, beta, d);                    // d = r + beta d
    A->Action(d, z);                            // z = A d
    inprod(l,d, z, den);                        // den = d dot z
    nom = betanom;
  } // end iteration
  delete [] r;
  delete [] z;
  delete [] d;
}

//============================================================================
void axpby( int n, double a, double *x, double p, double *b, double *y);

//============================================================================

void Subdomain::gmres(int n,int &nit,double *x,double *b, Matrix *A){
  int i, j, k, m;  
  double H[Kmax+2][Kmax+1], HH[Kmax+1][Kmax+1]; 
  double *r=new double[n], *ap=new double[n], *xNew=new double[n], inpr;
  int iter;
  
  double rNorm, RNorm, y[Kmax], h1[Kmax];
  
  // if (Kmax > n) Kmax=n; 
  double *q[Kmax+1];
  for (i=1; i<=Kmax; i++) q[i] = new double[n]; 

  for (iter = 0; iter<nit; iter++)  {
    A->Action(x, ap);   	      	       //  ap   = Ax
    axpby( n, 1., b, -1., ap, r);              //  r    = b-Ax
    inprod(level, r, r, inpr);                 //  inpr = r . r

    H[1][0] = sqrt(inpr); 
    //if ( fabs(H[1][0]) < TOLERANCE) break;

    for(k=1; k<=Kmax; k++) {
      axpby(n,0., r, 1.0/H[k][k-1], r, q[k]);  //  q[k] = 1.0/H[k][k-1] r
      A->Action( q[k], ap);                    //  ap   = A q[k]
      axpby(n,0., ap, 1.0, ap, r);             //  r    = ap 
      for (i=1; i<=k; i++) {
	inprod(level, q[i], ap, H[i][k]);      //  H[i][k] = q[i] . ap
	axpby(n, 1.0, r, -H[i][k], q[i], r);   //  r       = r - H[i][k] q[i]
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
      axpby(n, 1.0, x, 0.0, x, xNew);           // xNew = x
      for (i=1; i<=k; i++)
	axpby(n, 1.0, xNew, y[i], q[i], xNew);  // xNew += y[i]*q[i] 
      
      rNorm = fabs(H[k+1][k]);
      if (rNorm < TOLERANCE) break;
    }
    A->Action(xNew, ap);                        // ap = A xNew
    axpby( n, 1.0, b, -1.0, ap, ap);            // ap = b - ap
    inprod(level, ap, ap, inpr);
    RNorm = sqrt(inpr);                         // RNorm = sqrt(ap . ap)
    
    if (rNorm < TOLERANCE) break;
    if (RNorm < TOLERANCE) break;
    axpby( n, 1.0, xNew, 0.0, xNew, x);         // x = xNew
    
    fprintf(stderr, "Iteration %d done!\n", iter);
    fprintf(stderr, "The current residual RNorm = %e\n", RNorm);
  }
  
  nit = iter;
  A->Action( xNew, ap);
  axpby( n, 1.0, b, -1.0, ap, r);
  inprod(level, r, r, inpr);
  rNorm = sqrt(inpr);
  for (i=1; i<=Kmax; i++) delete(q[i]);

  printf("\n The final residual is %e\n\n", rNorm);
  delete [] r;
  delete [] ap;
  delete [] xNew;
}

//============================================================================
