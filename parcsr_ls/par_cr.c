
/*====================
 * Functions to run cr
 *====================*/
#include <headers.h>

#define RelaxScheme1 3 /* cr type */
#define fptOmegaJac 1  /* 1 is f pt weighted jacobi */
#define omega1 1.0     /* weight */
#define fptgs 3        /* 3 is f pt GS */
                                                                                                             
#define theta_global1 .7    /* cr stop criteria */
#define mu1            5    /* # of cr sweeps */
                                                                                                             
#define cpt  1
#define fpt -1
#define cand 0
                                                                                                             
int
hypre_BoomerAMGCoarsenCR( hypre_ParCSRMatrix    *A,
                 int               **CF_marker_ptr,
                 hypre_ParCSRMatrix    *S,
                 int                *coarse_size_ptr,
                 int                num_CR_relax_steps,
                 int                IS_type,
                 int                CRaddCpoints)
{
   int i;
   double theta_global;
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   int             *A_i           = hypre_CSRMatrixI(A_diag);
   int             *A_j           = hypre_CSRMatrixJ(A_diag);
   double          *A_data        = hypre_CSRMatrixData(A_diag);
   int              num_variables = hypre_CSRMatrixNumRows(A_diag);
                                                                                                             
   int             *S_i;
   int             *S_j;
   double          *S_data;
                                                                                                             
   int             *CF_marker;
   int              coarse_size;
   int              ierr = 0;
                                                                                                             
   if(CRaddCpoints == 0){
      CF_marker = hypre_CTAlloc(int, num_variables);
      for ( i = 0; i < num_variables; i++)
          CF_marker[i] = fpt;
   } else {
      CF_marker = *CF_marker_ptr;
   }
                                                                                                             
  /* Run the CR routine */
                                                                                                             
   fprintf(stdout,"\n... Building CF using CR ...\n\n");
   cr(A_i, A_j, A_data, num_variables, CF_marker,
      RelaxScheme1, omega1, theta_global1,mu1);
                                                                                                             
   fprintf(stdout,"\n... Done \n\n");
   coarse_size = 0;
   for ( i =0 ;i < num_variables;i++){
      if ( CF_marker[i] == cpt){
         coarse_size++;
      }
    }
   *CF_marker_ptr   = CF_marker;
   *coarse_size_ptr = coarse_size;
                                                                                                             
   return (ierr);
}

/* main cr routine */
int cr(int *A_i, int *A_j, double *A_data, int n, int *cf, 
       int rlx, double omega, double tg, int mu)
{ 
   int i,j,nstages=0;  
   double nc,rho,rho0,rho1,*e0,*e1;

   e0=hypre_CTAlloc(double, n); e1=hypre_CTAlloc(double, n);    

   fprintf(stdout,"Stage  \t rho \t alpha \n");
   fprintf(stdout,"-----------------------\n");

   for (i = 0; i < n; i++) 
      e1[i] = 1.0e0+.1*drand48();
   
  /* stages */
   while(1){
      if (nstages > 0){
         for (i=0;i<n;i++){
            if(cf[i] == cpt){
               e0[i] = 0.0e0;
               e1[i] = 0.0e0;
            }
         }
      }

      switch(rlx){
         case fptOmegaJac: 
            for (i=0;i<mu;i++)
               fptjaccr(cf,A_i,A_j,A_data,n,e0,omega,e1); 
         break;
         case fptgs:
            for (i=0;i<mu;i++)  
               fptgscr(cf,A_i,A_j,A_data,n,e0,e1); 
         break;
      }

      rho=0.0e0; rho0=0.0e0; rho1=0.0e0;
      for(i=0;i<n;i++){ 
         rho0 += pow(e0[i],2);
         rho1 += pow(e1[i],2);
      }
      rho = sqrt(rho1)/sqrt(rho0);

      if (rho > tg){
         formu(cf,n,e1,A_i,rho);
         IndepSetGreedy(A_i,A_j,n,cf);

         fprintf(stdout,"  %d \t%2.3lf  \t%2.3lf \n",
                 nstages,rho,nc/n);
        /* update for next sweep */
         nc = 0.0e0;
         for (i=0;i<n;i++){
	    if (cf[i] ==  cpt) 
               nc+=1.0e0;
	    else if (cf[i] ==  fpt){ 
               e0[i] = 1.0e0+.1*drand48();
               e1[i] = 1.0e0+.1*drand48();
            }
         }
         nstages += 1;
      } else {
         fprintf(stdout,"  %d \t%2.3lf  \t%2.3lf \n",
                 nstages,rho,nc/n);
         break;
      }
   }
   free(e0); free(e1); return 0;
}

/* take an ind. set over the candidates*/
int GraphAdd( Link *list, int *head, int *tail, int index, int istack )
{
   int prev = tail[-istack];
                                                                                                       
   list[index].prev = prev;
   if (prev < 0){
      head[-istack] = index;
   } else {
      list[prev].next = index;
   }
   list[index].next = -istack;
   tail[-istack] = index;
                                                                                                       
   return 0;
}
                                                                                                       
int GraphRemove( Link *list, int *head, int *tail, int index )
{
   int prev = list[index].prev;
   int next = list[index].next;
                                                                                                       
   if (prev < 0){
      head[prev] = next;
   } else {
      list[prev].next = next;
   }
   if (next < 0){
      tail[next] = prev;
   } else {
      list[next].prev = prev;
   }
                                                                                                       
   return 0;
}

int IndepSetGreedy(int *A_i, int *A_j, int n, int *cf) 
{
   Link *list;
   int  *head, *head_mem, *ma;
   int  *tail, *tail_mem;
                                                                                                       
   int i, ji, jj, jl, index, istack, stack_size;

   ma = hypre_CTAlloc(int, n);
                                                                                                       
   /* Initialize the graph and measure array
    *
    * ma: cands >= 1
    *     cpts  = -1
    *     else  =  0
    * Note: only cands are put into graph */
                                                                                                       
   istack = 0;
   for (i = 0; i < n; i++){
      if (cf[i] == cand){
         ma[i] = 1;
         for (ji = A_i[i]+1; ji < A_i[i+1]; ji++){
            jj = A_j[ji];
            if (cf[jj] != cpt){
               ma[i]++;
            }
         }
         if (ma[i] > istack){
            istack = (int) ma[i];
         }
      }
      else if (cf[i] == cpt){
         ma[i] = -1;
      } else {
         ma[i] = 0;
      }
   }
   stack_size = 2*istack;
                                                                                                       
   /* initialize graph */
   head_mem = hypre_CTAlloc(int, stack_size); head = head_mem + stack_size;
   tail_mem = hypre_CTAlloc(int, stack_size); tail = tail_mem + stack_size;
   list = hypre_CTAlloc(Link, n);
                                                                                                       
   for (i = -1; i >= -stack_size; i--){
      head[i] = i;
      tail[i] = i;
   }
   for (i = 0; i < n; i++){
      if (ma[i] > 0){
         GraphAdd(list, head, tail, i, (int) ma[i]);
      }
   }
                                                                                                       
  /* Loop until all points are either F or C */
   while (istack > 0){
     /* i w/ max measure at head of stacks */
      i = head[-istack];
                                                                                                       
     /* make i C point */
      cf[i] = cpt;
      ma[i] = -1;
                                                                                                       
     /* remove i from graph */
      GraphRemove(list, head, tail, i);
                                                                                                       
     /* update nbs and nbs-of-nbs */
      for (ji = A_i[i]+1; ji < A_i[i+1]; ji++){
         jj = A_j[ji];
        /* if not "decided" C or F */
         if (ma[jj] > -1){
           /* if a candidate, remove jj from graph */
            if (ma[jj] > 0){
               GraphRemove(list, head, tail, jj);
            }
                                                                                                       
           /* make jj an F point and mark "decided" */
            cf[jj] = fpt;
            ma[jj] = -1;
                                                                                                       
            for (jl = A_i[jj]+1; jl < A_i[jj+1]; jl++){
               index = A_j[jl];
              /* if a candidate, increase ma */
               if (ma[index] > 0){
                  ma[index]++;
                                                                                                       
                 /* move index in graph */
                  GraphRemove(list, head, tail, index);
                  GraphAdd(list, head, tail, index,
                           (int) ma[index]);
                  if (ma[index] > istack){
                     istack = (int) ma[index];
                  }
               }
            }
         }
      }
     /* reset istack to point to biggest non-empty stack */
      for ( ; istack > 0; istack--){
        /* if non-negative, break */
         if (head[-istack] > -1){
            break;
         }
      }
   }
   free(ma); free(list); free(head_mem); free(tail_mem);
   return 0;
}

/* f point jac cr */
int fptjaccr(int *cf, int *A_i, int *A_j, double *A_data,
       int n, double *e0, double omega, double *e1)
{
   int i, j;
   double res;

   for (i=0;i<n;i++)
      if (cf[i] == fpt)
         e0[i] = e1[i];

   for (i=0;i<n;i++){
      res = 0.0e0;  
      if (cf[i] == fpt){
         for (j=A_i[i]+1;j<A_i[i+1];j++){ 
            if (cf[A_j[j]] == fpt){
               res -= (A_data[j]*e0[A_j[j]]);
            }
         }
         e1[i] *= (1.0-omega);
         e1[i] += omega*res/A_data[A_i[i]];
      }
   }
   return 0;
}


/* f point GS cr */
int fptgscr(int *cf, int *A_i, int *A_j, double *A_data, int n,
       double *e0, double *e1)
{
   int i, j;
   double res; 

   for (i=0;i<n;i++)
      if (cf[i] == fpt)
         e0[i] = e1[i];

   for (i=0;i<n;i++){
      if (cf[i] == fpt){  
         res = 0.0e0; 
         for ( j = A_i[i]+1; j < A_i[i+1]; j++){ 
            if (cf[A_j[j]] == fpt){
               res -= (A_data[j]*e1[A_j[j]]);
            }
         }
         e1[i] = res/A_data[A_i[i]];  
      }
   }
return 0;
}

/* form the candidate set U */
int formu(int *cf,int n, double *e1, int *A_i, double rho){
   int i;
   double candmeas=0.0e0, max=0.0e0;
   double thresh=1-rho;

   for(i=0;i<n;i++)
      if(fabs(e1[i]) > max)
         max = fabs(e1[i]);

   for (i=0;i<n;i++){
      if (cf[i] == fpt){
	 candmeas = pow(fabs(e1[i]),1.0)/max;
	 if (candmeas > thresh && A_i[i+1]-A_i[i] > 1){
            cf[i] = cand; 
         }
      }
   }
   return 0;
}
