#include <ulocks.h>
#include <task.h>

typedef void (*pfunction)();
typedef double *pdouble;

struct gmres_data{
  int nit; 
  double **H, **HH, *r, *ap, *xNew, *y, *h1, **q, *inpr;
};


void pgmres(int &, int &, double *, double *, Matrix *, gmres_data &);

//============================================================================
// This is parallel implementation of the GMRES routine.
//============================================================================
void Method::gmres(int n, int &nit, double *x, double *b, int Kmax,Matrix *A){
  /*
  int i,ii, j, k, m;  
  double H[Kmax+2][Kmax+1];//, HH[Kmax][Kmax]; 
  double r[n], ap[n], xNew[n];
  int iter;
  
  double rNorm, RNorm, y[Kmax], h1[Kmax];
  
  //if (Kmax > n) Kmax=n; 
  //double *q[Kmax+1];
  //for (i=1; i<=Kmax; i++) q[i] = new double[n]; 

  int np = 1, begin = 0, end = n, p = 0;
  double inpr[1];


  //  gmres_data data;
  //double **H, **HH, **q;
  double **q, **HH;

  if (Kmax > n) Kmax=n; 
  q = new pdouble[Kmax+1];
  for (i=1; i<=Kmax; i++) q[i] = new double[n];
  
  //H  = new pdouble[Kmax+2];
  //for(i=0; i<(Kmax+2); i++) H[i] = new double[Kmax+1];
  
  HH = new pdouble[Kmax+1];
  for(i=0; i<Kmax+1; i++) HH[i] = new double[Kmax+1];
  */
  
  int i, np = m_get_numprocs();  
  //double r[n], ap[n], xNew[n], inpr[np];
  //double y[Kmax], h1[Kmax];
  double *r, *ap, *xNew, *inpr;
  double *y, *h1;
  r    = new double[n];
  ap   = new double[n];
  xNew = new double[n];
  inpr = new double[np];
  y    = new double[Kmax];
  h1   = new double[Kmax];

  gmres_data data;

  if (Kmax > n) Kmax=n; 
  data.q = new pdouble[Kmax+1];
  for (i=1; i<=Kmax; i++) data.q[i] = new double[n];

  data.H  = new pdouble[Kmax+2];
  for(i=0; i<(Kmax+2); i++) data.H[i] = new double[Kmax+1];

  data.HH = new pdouble[Kmax+1];
  for(i=0; i<=Kmax; i++) data.HH[i] = new double[Kmax+1];

  double **H = data.H, **HH = data.HH, **q = data.q;

  data.nit = nit;
  data.r   = r;
  data.ap  = ap;
  data.xNew= xNew;
  data.y   = y;
  data.h1  = h1;
  data.inpr= inpr;

  m_fork((pfunction)pgmres, &n, &Kmax, x, b, A, &data);
  //pgmres(n, Kmax, x, b, A, data);
  
  //==========================================================================
  /*
  for (i=1; i<=Kmax; i++) delete []data.q[i];
  delete [] data.q;
  for(i=0; i<(Kmax+2); i++) delete []data.H[i];
  delete [] data.H;
  for(i=0; i<Kmax; i++) delete [] data.HH[i];
  delete [] data.HH;
  */
}


//============================================================================
void pgmres(int &n,int &Kmax, double *x, double *b, Matrix *A, gmres_data &D){
  int p = m_get_myid(), np = m_get_numprocs();
  //int p = 0, np = 1;
  int begin, end, i, k, m, j, ii, iter, nit = D.nit;
  double rNorm, RNorm;
  double **H = D.H, **HH = D.HH, *r = D.r, *ap = D.ap;
  double *xNew = D.xNew, *y = D.y, *h1 = D.h1, **q = D.q, *inpr = D.inpr;

  if (n%np >= 2){ begin = p*(n/np + 1); end = (p+1)*(n/np+1);} 
  else { begin = p*n/np; end = begin + n/np;}
  if (p == (np-1)) end = n;
  
  //fprintf(stderr,"begin = %d, end = %d\n", begin, end);

  for(iter = 0; iter<nit; iter++){
    A->Action(x, ap, begin, end);       	         //  ap   = Ax
fprintf(stderr,"%d\n", p);
    m_sync();
    inpr[p] = 0.;
    for(i=begin; i<end; i++){                            //  r    = b-Ax
      r[i]  = b[i]-ap[i];                                //  inpr = (r, r)
      inpr[p] += r[i]*r[i];
    }
    m_sync();
    if (p==0){
      for(i=1; i<np; i++)
	inpr[0]+=inpr[i];
      H[1][0] = sqrt(inpr[0]);
    }
    m_sync();
    //fprintf(stderr,"1. %d\n", p);
    
    for(k=1; k<=Kmax; k++) {
      for(i=begin; i<end; i++)                 //  q[k] = 1.0/H[k][k-1] r
	q[k][i] = r[i]/H[k][k-1];
      m_sync();
      //fprintf(stderr,"2. %d\n", p);
      A->Action( q[k], ap, begin, end);        //  ap   = A q[k]
      for(i=begin;i<end;i++) r[i] = ap[i];     //  r    = ap 
      for(ii=1; ii<=k; ii++) {
	inpr[p] = 0;
	for(i=begin; i<end; i++)               //  H[ii][k] = q[ii] . ap
	  inpr[p] += q[ii][i]*ap[i];
	m_sync();
	//fprintf(stderr,"3. %d\n", p);
	if (p==0){
	  for(i=1; i<np; i++)
	    inpr[0] += inpr[i];
	  H[ii][k] = inpr[0];
	}
	m_sync();
	//fprintf(stderr,"4. %d\n", p);
	
	for(i=begin; i<end; i++)               //  r        = r-H[ii][k] q[ii]
	  r[i] -= H[ii][k] * q[ii][i]; 
      }
      inpr[p] = 0;
      for(i=begin; i<end; i++)                 //  inpr = r . r
	inpr[p] += r[i]*r[i];
      m_sync();
      //fprintf(stderr,"5. %d\n", p);
      if (p==0){
	for(i=1; i<np; i++)
	  inpr[0]+=inpr[i];    
	H[k+1][k]=sqrt(inpr[0]);                //  H[k+1][k] = sqrt(r . r) 
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
	// Minimization Done
      }
      m_sync();
      rNorm = fabs(H[k+1][k]);
      //fprintf(stderr,"6. %d\n", p);
      if (rNorm < TOLERANCE || k == Kmax){
	for(i=begin; i<end; i++) xNew[i]=x[i];  // xNew = x
	for (ii=1; ii<=k; ii++)
	  for(i=begin; i<end; i++)              // xNew += y[ii]*q[ii]
	    xNew[i] += y[ii]*q[ii][i];
	break;
      }
    }
    A->Action(xNew, ap, begin, end);            // ap = A xNew
    m_sync();
    //fprintf(stderr,"7. %d\n", p);
    inpr[p] = 0.;
    for(i=begin; i<end; i++){                   // ap = b - ap
      ap[i] = b[i] - ap[i];
      inpr[p] += ap[i]*ap[i];
    }
    m_sync();
    //fprintf(stderr,"8. %d\n", p);
    if (p==0){
      for(i=1; i<np; i++)
	inpr[0]+=inpr[i];  
      fprintf(stderr, "Iteration %d done!\n", iter);
      fprintf(stderr, "The current residual RNorm = %e\n", RNorm);
    }
    m_sync();
    RNorm = sqrt(inpr[0]);                    // RNorm = sqrt(ap . ap)
    if (rNorm < TOLERANCE) break;
    if (RNorm < TOLERANCE) break;

    for(i=begin;i<end;i++) x[i] = xNew[i];     // x = xNew
  }
}
