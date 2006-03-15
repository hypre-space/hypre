
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
hypre_BoomerAMGCoarsenCR1( hypre_ParCSRMatrix    *A,
                 int               **CF_marker_ptr,
                 int                *coarse_size_ptr,
                 int                num_CR_relax_steps,
                 int                IS_type,
                 int                CRaddCpoints)
{
   int i;
   /* double theta_global;*/
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   int             *A_i           = hypre_CSRMatrixI(A_diag);
   int             *A_j           = hypre_CSRMatrixJ(A_diag);
   double          *A_data        = hypre_CSRMatrixData(A_diag);
   int              num_variables = hypre_CSRMatrixNumRows(A_diag);
                                                                                                             
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
   int i,nstages=0;  
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
/*==========================================================================
 * Ruge's coarsening algorithm                        
 *==========================================================================*/

#define C_PT 1
#define F_PT -1
#define Z_PT -2
#define SF_PT -3  /* special fine points */
#define UNDECIDED 0 


/**************************************************************
 *
 *      Ruge Coarsening routine
 *
 **************************************************************/
int
hypre_BoomerAMGIndepRS( hypre_ParCSRMatrix    *S,
                        int                    measure_type,
                        int                    debug_flag,
                        int                   *CF_marker)
{
   MPI_Comm         comm          = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg   *comm_pkg      = hypre_ParCSRMatrixCommPkg(S);
   hypre_CSRMatrix *S_diag        = hypre_ParCSRMatrixDiag(S);
   hypre_CSRMatrix *S_offd        = hypre_ParCSRMatrixOffd(S);
   int             *S_i           = hypre_CSRMatrixI(S_diag);
   int             *S_j           = hypre_CSRMatrixJ(S_diag);
   int             *S_offd_i      = hypre_CSRMatrixI(S_offd);
   int             *S_offd_j;
   int              num_variables = hypre_CSRMatrixNumRows(S_diag);
   int              num_cols_offd = hypre_CSRMatrixNumCols(S_offd);
                  
   hypre_ParCSRCommHandle *comm_handle;
   hypre_CSRMatrix *ST;
   int             *ST_i;
   int             *ST_j;
                 
   int             *measure_array;
   int             *CF_marker_offd;
   int             *int_buf_data;

   int              i, j, k, jS;
   int		    index;
   int		    num_procs, my_id;
   int		    num_sends = 0;
   int		    start, jrow;

   hypre_LinkList   LoL_head;
   hypre_LinkList   LoL_tail;

   int             *lists, *where;
   int              measure, new_meas;
   int              num_left = 0;
   int              nabor, nabor_two;

   int              ierr = 0;
   int              f_pnt = F_PT;
   double	    wall_time;

   /*-------------------------------------------------------
    * Initialize the C/F marker, LoL_head, LoL_tail  arrays
    *-------------------------------------------------------*/

   LoL_head = NULL;
   LoL_tail = NULL;
   lists = hypre_CTAlloc(int, num_variables);
   where = hypre_CTAlloc(int, num_variables);

#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

   /*--------------------------------------------------------------
    * Compute a CSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: the entries are negative initially, corresponding
    * to "unaccounted-for" dependence.
    *----------------------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   if (!comm_pkg)
   {
        comm_pkg = hypre_ParCSRMatrixCommPkg(S); 
   }

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
      hypre_NewCommPkgCreate(S);
#else
        hypre_MatvecCommPkgCreate(S);
#endif
        comm_pkg = hypre_ParCSRMatrixCommPkg(S); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   if (num_cols_offd) S_offd_j = hypre_CSRMatrixJ(S_offd);

   jS = S_i[num_variables];

   ST = hypre_CSRMatrixCreate(num_variables, num_variables, jS);
   ST_i = hypre_CTAlloc(int,num_variables+1);
   ST_j = hypre_CTAlloc(int,jS);
   hypre_CSRMatrixI(ST) = ST_i;
   hypre_CSRMatrixJ(ST) = ST_j;

   /*----------------------------------------------------------
    * generate transpose of S, ST
    *----------------------------------------------------------*/

   for (i=0; i <= num_variables; i++)
      ST_i[i] = 0;
 
   for (i=0; i < jS; i++)
   {
	 ST_i[S_j[i]+1]++;
   }
   for (i=0; i < num_variables; i++)
   {
      ST_i[i+1] += ST_i[i];
   }
   for (i=0; i < num_variables; i++)
   {
      for (j=S_i[i]; j < S_i[i+1]; j++)
      {
	 index = S_j[j];
 	 ST_j[ST_i[index]] = i;
       	 ST_i[index]++;
      }
   }      
   for (i = num_variables; i > 0; i--)
   {
      ST_i[i] = ST_i[i-1];
   }
   ST_i[0] = 0;

   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are given by the row sums of ST.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    * correct actual measures through adding influences from
    * neighbor processors
    *----------------------------------------------------------*/

   if (measure_type == 0)
   {
      measure_array = hypre_CTAlloc(int, num_variables);
      for (i=0; i < num_variables; i++)
         measure_array[i] = 0;
      for (i=0; i < num_variables; i++)
      {
         if (CF_marker[i] < 1)
         {
            for (j = S_i[i]; j < S_i[i+1]; j++)
            {
               if (CF_marker[S_j[j]] < 1)
                  measure_array[S_j[j]]++;
            }
         }
      }
 
   }
   else
   {

      /* now the off-diagonal part of CF_marker */
      if (num_cols_offd)
         CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
      else
         CF_marker_offd = NULL;
 
      for (i=0; i < num_cols_offd; i++)
         CF_marker_offd[i] = 0;
   
      /*------------------------------------------------
       * Communicate the CF_marker values to the external nodes
       *------------------------------------------------*/
      int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
         {
            jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
            int_buf_data[index++] = CF_marker[jrow];
         }
      }
    
      if (num_procs > 1)
      {
         comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                  CF_marker_offd);
         hypre_ParCSRCommHandleDestroy(comm_handle);
      }

      measure_array = hypre_CTAlloc(int, num_variables+num_cols_offd);
      for (i=0; i < num_variables+num_cols_offd; i++)
         measure_array[i] = 0;
 
      for (i=0; i < num_variables; i++)
      {
         if (CF_marker[i] < 1)
         {
            for (j = S_i[i]; j < S_i[i+1]; j++)
            {
               if (CF_marker[S_j[j]] < 1)
                  measure_array[S_j[j]]++;
            }
            for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
            {
               if (CF_marker_offd[S_offd_j[j]] < 1)
                  measure_array[num_variables + S_offd_j[j]]++;
            }
         }
      }
      /* now send those locally calculated values for the external nodes to the neighboring processors */
      if (num_procs > 1)
         comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg,
                        &measure_array[num_variables], int_buf_data);
 
      /* finish the communication */
      if (num_procs > 1)
         hypre_ParCSRCommHandleDestroy(comm_handle);
       
      /* now add the externally calculated part of the local nodes to the local nodes */
      index = 0;
      for (i=0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += int_buf_data[index++];
      }
      hypre_TFree(int_buf_data);
   }


   if (measure_type == 2 && num_procs > 1)
   {
      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] == 0)
         {
            if ((S_offd_i[i+1]-S_offd_i[i]) == 0)
            {
	       num_left++;
            }
            else 
            {
	       measure_array[i] = 0;
	       CF_marker[i] = 2;
            }
         }
         else if (CF_marker[i] < 0)
	    measure_array[i] = 0;
         else
	    measure_array[i] = -1;
      }
   }
   else
   {
      for (i = 0; i < num_variables; i++)
      {
         if (CF_marker[i] == 0)
         {
	    num_left++;
         }
         else if (CF_marker[i] < 0)
	    measure_array[i] = 0;
         else
	    measure_array[i] = -1;
      }
   }

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   if (debug_flag == 3) wall_time = time_getWallclockSeconds();

   /* first coarsening phase */

  /*************************************************************
   *
   *   Initialize the lists
   *
   *************************************************************/

   for (j = 0; j < num_variables; j++) 
   {    
      measure = measure_array[j];
      if (CF_marker[j] == 0)
      {
         if (measure > 0)
         {
            enter_on_lists(&LoL_head, &LoL_tail, measure, j, lists, where);
         }
         else
         {
            if (measure < 0) printf("negative measure!\n");
            CF_marker[j] = f_pnt;
            for (k = S_i[j]; k < S_i[j+1]; k++)
            {
               nabor = S_j[k];
               if (CF_marker[nabor] != SF_PT && CF_marker[nabor] < 1)
               {
                  if (nabor < j)
                  {
                     new_meas = measure_array[nabor];
	             if (new_meas > 0)
                        remove_point(&LoL_head, &LoL_tail, new_meas, 
                               nabor, lists, where);

                     new_meas = ++(measure_array[nabor]);
                     enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                 nabor, lists, where);
                  }
	          else
                  {
                     new_meas = ++(measure_array[nabor]);
                  }
               }
            }
            --num_left;
         }
      }
   }

   /****************************************************************
    *
    *  Main loop of Ruge-Stueben first coloring pass.
    *
    *  WHILE there are still points to classify DO:
    *        1) find first point, i,  on list with max_measure
    *           make i a C-point, remove it from the lists
    *        2) For each point, j,  in S_i^T,
    *           a) Set j to be an F-point
    *           b) For each point, k, in S_j
    *                  move k to the list in LoL with measure one
    *                  greater than it occupies (creating new LoL
    *                  entry if necessary)
    *        3) For each point, j,  in S_i,
    *                  move j to the list in LoL with measure one
    *                  smaller than it occupies (creating new LoL
    *                  entry if necessary)
    *
    ****************************************************************/

   while (num_left > 0)
   {
      index = LoL_head -> head;

      CF_marker[index] = C_PT;
      measure = measure_array[index];
      measure_array[index] = 0;
      --num_left;
      
      remove_point(&LoL_head, &LoL_tail, measure, index, lists, where);
  
      for (j = ST_i[index]; j < ST_i[index+1]; j++)
      {
         nabor = ST_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            CF_marker[nabor] = F_PT;
            measure = measure_array[nabor];

            remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);
            --num_left;

            for (k = S_i[nabor]+1; k < S_i[nabor+1]; k++)
            {
               nabor_two = S_j[k];
               if (CF_marker[nabor_two] == UNDECIDED)
               {
                  measure = measure_array[nabor_two];
                  remove_point(&LoL_head, &LoL_tail, measure, 
                               nabor_two, lists, where);

                  new_meas = ++(measure_array[nabor_two]);
                 
                  enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                 nabor_two, lists, where);
               }
            }
         }
      }
      for (j = S_i[index]; j < S_i[index+1]; j++)
      {
         nabor = S_j[j];
         if (CF_marker[nabor] == UNDECIDED)
         {
            measure = measure_array[nabor];

            remove_point(&LoL_head, &LoL_tail, measure, nabor, lists, where);

            measure_array[nabor] = --measure;
	
	    if (measure > 0)
               enter_on_lists(&LoL_head, &LoL_tail, measure, nabor, 
				lists, where);
	    else
	    {
               CF_marker[nabor] = F_PT;
               --num_left;

               for (k = S_i[nabor]+1; k < S_i[nabor+1]; k++)
               {
                  nabor_two = S_j[k];
                  if (CF_marker[nabor_two] == UNDECIDED)
                  {
                     new_meas = measure_array[nabor_two];
                     remove_point(&LoL_head, &LoL_tail, new_meas, 
                               nabor_two, lists, where);

                     new_meas = ++(measure_array[nabor_two]);
                 
                     enter_on_lists(&LoL_head, &LoL_tail, new_meas,
                                 nabor_two, lists, where);
                  }
               }
	    }
         }
      }
   }

   hypre_TFree(measure_array);
   hypre_CSRMatrixDestroy(ST);

   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Coarsen 1st pass = %f\n",
                     my_id, wall_time); 
   }

   if (measure_type == 2)
   {
      for (i=0; i < num_variables; i++)
         if (CF_marker[i] == 2) CF_marker[i] = 0;
   }

   hypre_TFree(lists);
   hypre_TFree(where);
   hypre_TFree(LoL_head);
   hypre_TFree(LoL_tail);

   return (ierr);
}


int
hypre_BoomerAMGIndepHMIS( hypre_ParCSRMatrix    *S,
                          int                    measure_type,
                          int                    debug_flag,
                          int                   *CF_marker)
{
   int              ierr = 0;
   int		    num_procs;
   MPI_Comm comm = hypre_ParCSRMatrixComm(S);

   MPI_Comm_size(comm,&num_procs);
   /*-------------------------------------------------------
    * Perform Ruge coarsening followed by CLJP coarsening
    *-------------------------------------------------------*/

   ierr += hypre_BoomerAMGIndepRS (S, 2, debug_flag,
                                CF_marker);

   if (num_procs > 1)
      ierr += hypre_BoomerAMGIndepPMIS (S, 0, debug_flag,
                                CF_marker);

   return (ierr);
}

/*--------------------------------------------------------------------------*/

#define C_PT  1
#define F_PT -1
#define SF_PT -3
#define COMMON_C_PT  2
#define Z_PT -2

      /* begin HANS added */
/**************************************************************
 *
 *      Modified Independent Set Coarsening routine
 *          (don't worry about strong F-F connections
 *           without a common C point)
 *
 **************************************************************/
int
hypre_BoomerAMGIndepPMIS( hypre_ParCSRMatrix    *S,
                        int                    CF_init,
                        int                    debug_flag,
                        int                   *CF_marker)
{
   MPI_Comm 	       comm            = hypre_ParCSRMatrixComm(S);
   hypre_ParCSRCommPkg      *comm_pkg        = hypre_ParCSRMatrixCommPkg(S);
   hypre_ParCSRCommHandle   *comm_handle;

   hypre_CSRMatrix    *S_diag          = hypre_ParCSRMatrixDiag(S);
   int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix    *S_offd          = hypre_ParCSRMatrixOffd(S);
   int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   int                *S_offd_j;

   int                 num_variables   = hypre_CSRMatrixNumRows(S_diag);
   int 		       num_cols_offd = 0;
                  
   int		       num_sends = 0;
   int  	      *int_buf_data;
   double	      *buf_data;

   int                *CF_marker_offd;
                      
   double             *measure_array;
   int                *graph_array;
   int                *graph_array_offd;
   int                 graph_size;
   int                 graph_offd_size;
   int                 global_graph_size;
                      
   int                 i, j, jj, jS, ig;
   int		       index, start, my_id, num_procs, jrow, cnt, elmt;
                      
   int                 ierr = 0;

   double	    wall_time;
   int   iter = 0;



#if 0 /* debugging */
   char  filename[256];
   FILE *fp;
   int   iter = 0;
#endif

   /*******************************************************************************
    BEFORE THE INDEPENDENT SET COARSENING LOOP:
      measure_array: calculate the measures, and communicate them
        (this array contains measures for both local and external nodes)
      CF_marker, CF_marker_offd: initialize CF_marker
        (separate arrays for local and external; 0=unassigned, negative=F point, positive=C point)
   ******************************************************************************/      

   /*--------------------------------------------------------------
    * Use the ParCSR strength matrix, S.
    *
    * For now, the "strength" of dependence/influence is defined in
    * the following way: i depends on j if
    *     aij > hypre_max (k != i) aik,    aii < 0
    * or
    *     aij < hypre_min (k != i) aik,    aii >= 0
    * Then S_ij = 1, else S_ij = 0.
    *
    * NOTE: S_data is not used; in stead, only strong columns are retained
    *       in S_j, which can then be used like S_data
    *----------------------------------------------------------------*/

   /*S_ext = NULL; */
   if (debug_flag == 3) wall_time = time_getWallclockSeconds();
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   if (!comm_pkg)
   {
        comm_pkg = hypre_ParCSRMatrixCommPkg(S); 
   }

   if (!comm_pkg)
   {
#ifdef HYPRE_NO_GLOBAL_PARTITION
      hypre_NewCommPkgCreate(S);
#else
        hypre_MatvecCommPkgCreate(S);
#endif
        comm_pkg = hypre_ParCSRMatrixCommPkg(S); 
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   int_buf_data = hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
   buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                num_sends));
 
   num_cols_offd = hypre_CSRMatrixNumCols(S_offd);

   S_diag_j = hypre_CSRMatrixJ(S_diag);

   if (num_cols_offd)
   {
      S_offd_j = hypre_CSRMatrixJ(S_offd);
   }

   /* now the off-diagonal part of CF_marker */
   if (num_cols_offd)
     CF_marker_offd = hypre_CTAlloc(int, num_cols_offd);
   else
     CF_marker_offd = NULL;

   for (i=0; i < num_cols_offd; i++)
	CF_marker_offd[i] = 0;
  
   /*------------------------------------------------
    * Communicate the CF_marker values to the external nodes
    *------------------------------------------------*/
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
      {
   	 jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
	 int_buf_data[index++] = CF_marker[jrow];
      }
   }
   
   if (num_procs > 1)
   { 
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, 
						  CF_marker_offd);
       
      hypre_ParCSRCommHandleDestroy(comm_handle);   
   } 
      
   /*----------------------------------------------------------
    * Compute the measures
    *
    * The measures are currently given by the column sums of S.
    * Hence, measure_array[i] is the number of influences
    * of variable i.
    *
    * The measures are augmented by a random number
    * between 0 and 1.
    *----------------------------------------------------------*/

   measure_array = hypre_CTAlloc(double, num_variables+num_cols_offd);
   for (i=0; i < num_variables+num_cols_offd; i++)
      measure_array[i] = 0;

   /* calculate the local part for the local nodes */
   for (i=0; i < num_variables; i++)
   { 
      if (CF_marker[i] < 1) 
      {
         for (j = S_diag_i[i]; j < S_diag_i[i+1]; j++)
         { 
            if (CF_marker[S_diag_j[j]] < 1) 
	       measure_array[S_diag_j[j]] += 1.0;
         } 
         for (j = S_offd_i[i]; j < S_offd_i[i+1]; j++)
         { 
            if (CF_marker_offd[S_offd_j[j]] < 1) 
	       measure_array[num_variables + S_offd_j[j]] += 1.0;
         } 
      }
   }

   /* now send those locally calculated values for the external nodes to the neighboring processors */
   if (num_procs > 1)
   comm_handle = hypre_ParCSRCommHandleCreate(2, comm_pkg, 
                        &measure_array[num_variables], buf_data);

   /* finish the communication */
   if (num_procs > 1)
   hypre_ParCSRCommHandleDestroy(comm_handle);
      
   /* now add the externally calculated part of the local nodes to the local nodes */
   index = 0;
   for (i=0; i < num_sends; i++)
   {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            measure_array[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
                        += buf_data[index++];
   }

   /* set the measures of the external nodes to zero */
   for (i=num_variables; i < num_variables+num_cols_offd; i++)
   { 
      measure_array[i] = 0;
   }

   /* this augments the measures with a random number between 0 and 1 */
   /* (only for the local part) */
   /* this augments the measures */
   i = 2747+my_id;
   hypre_SeedRand(i);
   for (i = 0; i < num_variables; i++)
   {
      measure_array[i] += hypre_Rand();
   }

   /*---------------------------------------------------
    * Initialize the graph arrays, and CF_marker arrays
    *---------------------------------------------------*/

   /* first the off-diagonal part of the graph array */
   if (num_cols_offd) 
      graph_array_offd = hypre_CTAlloc(int, num_cols_offd);
   else
      graph_array_offd = NULL;

   for (ig = 0; ig < num_cols_offd; ig++)
      graph_array_offd[ig] = ig;

   graph_offd_size = num_cols_offd;

   /* now the local part of the graph array, and the local CF_marker array */
   graph_array = hypre_CTAlloc(int, num_variables);

   if (CF_init==1)
   { 
      cnt = 0;
      for (i=0; i < num_variables; i++)
      {
         if ( (S_offd_i[i+1]-S_offd_i[i]) > 0 || CF_marker[i] == -1)
	 {
	   CF_marker[i] = 0;
	 }
         if (CF_marker[i] == SF_PT)
            measure_array[i] = 0;
         else if ( CF_marker[i] < 1)
         {
            if (measure_array[i] >= 1.0 )
            {
               CF_marker[i] = 0;
               graph_array[cnt++] = i;
            }
            else
            {
               CF_marker[i] = F_PT;
               measure_array[i] = 0;
            }
         }
         else 
            measure_array[i] = 0;
      }
   }
   else
   {
      cnt = 0;
      for (i=0; i < num_variables; i++)
      {
         if (CF_marker[i] == 0 && measure_array[i] >= 1.0 )
         {
            graph_array[cnt++] = i;
         }
         else 
            measure_array[i] = 0;
      }
   }
   graph_size = cnt;

   /*------------------------------------------------
    * Communicate the local measures, which are complete,
      to the external nodes
    *------------------------------------------------*/
   index = 0;
   for (i = 0; i < num_sends; i++)
   {
       start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
       for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
       {
	   jrow = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
	   buf_data[index++] = measure_array[jrow];
       }
   }
   
   if (num_procs > 1)
   { 
       comm_handle = hypre_ParCSRCommHandleCreate(1, comm_pkg, buf_data, 
						  &measure_array[num_variables]);
       
       hypre_ParCSRCommHandleDestroy(comm_handle);   
       
   } 
      
   if (debug_flag == 3)
   {
      wall_time = time_getWallclockSeconds() - wall_time;
      printf("Proc = %d    Initialize CLJP phase = %f\n",
                     my_id, wall_time); 
   }

   /*******************************************************************************
    THE INDEPENDENT SET COARSENING LOOP:
   ******************************************************************************/      

   /*---------------------------------------------------
    * Loop until all points are either fine or coarse.
    *---------------------------------------------------*/

   while (1)
   {

     /* stop the coarsening if nothing left to be coarsened */
     MPI_Allreduce(&graph_size,&global_graph_size,1,MPI_INT,MPI_SUM,comm);

     if (global_graph_size == 0)
       break;

     /*     printf("\n");
     printf("*** MIS iteration %d\n",iter);
     printf("graph_size remaining %d\n",graph_size);*/

     /*------------------------------------------------
      * Pick an independent set of points with
      * maximal measure.
        At the end, CF_marker is complete, but still needs to be
        communicated to CF_marker_offd
      *------------------------------------------------*/
      if (1)
      {
          /* hypre_BoomerAMGIndepSet(S, measure_array, graph_array, 
				graph_size, 
				graph_array_offd, graph_offd_size, 
				CF_marker, CF_marker_offd);*/
         for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];
            if (measure_array[i] > 1)
               CF_marker[i] = 1;
         }
         for (ig = 0; ig < graph_offd_size; ig++)
         {
            i = graph_array_offd[ig];
            if (measure_array[i+num_variables] > 1)
               CF_marker_offd[i] = 1;
         }
   /*-------------------------------------------------------
    * Remove nodes from the initial independent set
    *-------------------------------------------------------*/
                                                                                              
         for (ig = 0; ig < graph_size; ig++)
         {
            i = graph_array[ig];
            if (measure_array[i] > 1)
            {
               for (jS = S_diag_i[i]; jS < S_diag_i[i+1]; jS++)
               {
                  j = S_diag_j[jS];
                                                                                              
                  if (measure_array[j] > 1)
                  {
                     if (measure_array[i] > measure_array[j])
                        CF_marker[j] = 0;
                     else if (measure_array[j] > measure_array[i])
                        CF_marker[i] = 0;
                  }
               }
               for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++)
               {
                  jj = S_offd_j[jS];
                  j = num_variables+jj;
                                                                                              
                  if (measure_array[j] > 1)
                  {
                     if (measure_array[i] > measure_array[j])
                        CF_marker_offd[jj] = 0;
                     else if (measure_array[j] > measure_array[i])
                        CF_marker[i] = 0;
                  }
               }
            }
         }

      /*------------------------------------------------
       * Exchange boundary data for CF_marker: send internal
         points to external points
       *------------------------------------------------*/

         if (num_procs > 1)
         {
            comm_handle = hypre_ParCSRCommHandleCreate(12, comm_pkg, 
		CF_marker_offd, int_buf_data);
 
            hypre_ParCSRCommHandleDestroy(comm_handle);   
         }

         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            {
               elmt = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);
               if (!int_buf_data[index] && CF_marker[elmt] > 0)
               {
                  CF_marker[elmt] = 0; 
                  index++;
               }
               else
               {
                  int_buf_data[index++] = CF_marker[elmt];
               }
            }
         }
 
         if (num_procs > 1)
         {
            comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, 
        	CF_marker_offd);
 
            hypre_ParCSRCommHandleDestroy(comm_handle);   
         }
      }

      iter++;
     /*------------------------------------------------
      * Set C-pts and F-pts.
      *------------------------------------------------*/

     for (ig = 0; ig < graph_size; ig++) {
       i = graph_array[ig];

       /*---------------------------------------------
	* First treat the case where point i is in the
	* independent set: make i a C point, 
	*---------------------------------------------*/
       
       if (CF_marker[i] > 0) 
       {
	 /* set to be a C-pt */
	 CF_marker[i] = C_PT;
       }  

       /*---------------------------------------------
	* Now treat the case where point i is not in the
	* independent set: loop over
	* all the points j that influence equation i; if
	* j is a C point, then make i an F point.
	*---------------------------------------------*/

       else 
       {

	 /* first the local part */
	 for (jS = S_diag_i[i]+1; jS < S_diag_i[i+1]; jS++) {
	   /* j is the column number, or the local number of the point influencing i */
	   j = S_diag_j[jS];
	   if (CF_marker[j] > 0){ /* j is a C-point */
	     CF_marker[i] = F_PT;
	   }
	 }
	 /* now the external part */
	 for (jS = S_offd_i[i]; jS < S_offd_i[i+1]; jS++) {
	   j = S_offd_j[jS];
	   if (CF_marker_offd[j] > 0){ /* j is a C-point */
	     CF_marker[i] = F_PT;
	   }
	 }

       } /* end else */
     } /* end first loop over graph */

     /* now communicate CF_marker to CF_marker_offd, to make
        sure that new external F points are known on this processor */

      /*------------------------------------------------
       * Exchange boundary data for CF_marker: send internal
         points to external points
       *------------------------------------------------*/

      index = 0;
      for (i = 0; i < num_sends; i++)
      {
        start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
        for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
                int_buf_data[index++] 
                 = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
 
      if (num_procs > 1)
      {
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data, 
        CF_marker_offd);
 
      hypre_ParCSRCommHandleDestroy(comm_handle);   
      }

     /*------------------------------------------------
      * Update subgraph
      *------------------------------------------------*/

     for (ig = 0; ig < graph_size; ig++) {
       i = graph_array[ig];
       
       if (!CF_marker[i]==0) /* C or F point */
	 {
	   /* the independent set subroutine needs measure 0 for
              removed nodes */
	   measure_array[i] = 0;
	   /* take point out of the subgraph */
	   graph_size--;
	   graph_array[ig] = graph_array[graph_size];
	   graph_array[graph_size] = i;
	   ig--;
	 }
     }
     for (ig = 0; ig < graph_offd_size; ig++) {
       i = graph_array_offd[ig];
       
       if (!CF_marker_offd[i]==0) /* C or F point */
	 {
	   /* the independent set subroutine needs measure 0 for
              removed nodes */
	   measure_array[i+num_variables] = 0;
	   /* take point out of the subgraph */
	   graph_offd_size--;
	   graph_array_offd[ig] = graph_array_offd[graph_offd_size];
	   graph_array_offd[graph_offd_size] = i;
	   ig--;
	 }
     }
     
   } /* end while */

   /*   printf("*** MIS iteration %d\n",iter);
   printf("graph_size remaining %d\n",graph_size);

   printf("num_cols_offd %d\n",num_cols_offd);
   for (i=0;i<num_variables;i++)
     {
              if(CF_marker[i]==1)
       printf("node %d CF %d\n",i,CF_marker[i]);
       }*/


   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/

   hypre_TFree(measure_array);
   hypre_TFree(graph_array);
   if (num_cols_offd) hypre_TFree(graph_array_offd);
   hypre_TFree(buf_data);
   hypre_TFree(int_buf_data);
   hypre_TFree(CF_marker_offd);

   return (ierr);
}

int
hypre_BoomerAMGCoarsenCR( hypre_ParCSRMatrix    *A,
                 int               **CF_marker_ptr,
                 int                *coarse_size_ptr,
                 int                num_CR_relax_steps,
                 int                IS_type,
                 int                rlx_type,
                 double             relax_weight,
                 double             omega,
                 double             theta,
                 int                CRaddCpoints)
{
   /* double theta_global;*/
   MPI_Comm         comm = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   int              global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   int             *row_starts = hypre_ParCSRMatrixRowStarts(A);
   int             *A_i           = hypre_CSRMatrixI(A_diag);
   int             *A_j           = hypre_CSRMatrixJ(A_diag);
   /*double          *A_data        = hypre_CSRMatrixData(A_diag);*/
   double          *Vtemp_data        = hypre_CSRMatrixData(A_diag);
   int              num_variables = hypre_CSRMatrixNumRows(A_diag);
   int             *A_offd_i     = hypre_CSRMatrixI(A_offd);
   hypre_ParVector *e0_vec, *e1_vec, *Vtemp;                                                                                                             
   int             *CF_marker;
   int              coarse_size;
   int              ierr = 0;
   int i,j, nstages=0;  
   int              num_procs, my_id;
   double rho,rho0,rho1,*e0,*e1;
   int		    num_coarse, global_num_variables, global_nc = 0;
   MPI_Comm_size(comm,&num_procs);
   MPI_Comm_rank(comm,&my_id);

   global_num_variables = hypre_ParCSRMatrixGlobalNumRows(A);
   if(CRaddCpoints == 0)
   {
      CF_marker = hypre_CTAlloc(int, num_variables);
      for ( i = 0; i < num_variables; i++)
          CF_marker[i] = fpt;
   } 
   else 
   {
      CF_marker = *CF_marker_ptr;
   }
                                                                                                             
  /* Run the CR routine */
                                                                                                             
   if (my_id == 0) fprintf(stdout,"\n... Building CF using CR ...\n\n");
   /*cr(A_i, A_j, A_data, num_variables, CF_marker,
      RelaxScheme1, omega1, theta_global1,mu1);*/
                                                                                                             
/* main cr routine */
/*int cr(int *A_i, int *A_j, double *A_data, int n, int *cf, 
       int rlx, double omega, double tg, int mu)*/

   e0_vec = hypre_ParVectorCreate(comm,global_num_rows,row_starts);
   hypre_ParVectorInitialize(e0_vec);
   hypre_ParVectorSetPartitioningOwner(e0_vec,0);
   e1_vec = hypre_ParVectorCreate(comm,global_num_rows,row_starts);
   hypre_ParVectorInitialize(e1_vec);
   hypre_ParVectorSetPartitioningOwner(e1_vec,0);
   Vtemp = hypre_ParVectorCreate(comm,global_num_rows,row_starts);
   hypre_ParVectorInitialize(Vtemp);
   hypre_ParVectorSetPartitioningOwner(Vtemp,0);
   Vtemp_data = hypre_VectorData(hypre_ParVectorLocalVector(Vtemp));

   e0 = hypre_VectorData(hypre_ParVectorLocalVector(e0_vec));
   e1 = hypre_VectorData(hypre_ParVectorLocalVector(e1_vec));

   if (my_id == 0)
   {
     fprintf(stdout,"Stage  \t rho \t alpha \n");
     fprintf(stdout,"-----------------------\n");
   }

   for (i = 0; i < num_variables; i++) 
      e1[i] = 1.0e0+.1*drand48();
   
  /* stages */
   while(1)
   {
      if (nstages > 0)
      {
         for (i=0;i<num_variables;i++)
	 {
            Vtemp_data[i] = 0.0e0;
            if(CF_marker[i] == cpt)
	    {
               e0[i] = 0.0e0;
               e1[i] = 0.0e0;
            }
         }
      }

      /*for (i=0;i<num_CR_relax_steps;i++)
               fptgscr(CF_marker,A_i,A_j,A_data,num_variables,e0,e1); */
      /*switch(rlx_type){
         case fptOmegaJac: 
            for (i=0;i<mu;i++)
               fptjaccr(cf,A_i,A_j,A_data,n,e0,omega,e1); 
         break;
         case fptgs:
            for (i=0;i<mu;i++)  
               fptgscr(cf,A_i,A_j,A_data,n,e0,e1); 
         break;
      }*/

      for (i=0;i<num_CR_relax_steps;i++)
      {
         for (j=0; j < num_variables; j++)
 	    if (CF_marker[i] == fpt) e0[j] = e1[j];
         hypre_BoomerAMGRelax(A, Vtemp, CF_marker, rlx_type, fpt,
		relax_weight, omega, e1_vec, e0_vec);
      }
      rho=0.0e0; rho0=0.0e0; rho1=0.0e0;
      /*for(i=0;i<num_variables;i++){ 
         rho0 += pow(e0[i],2);
         rho1 += pow(e1[i],2);
      }*/

      rho0 = hypre_ParVectorInnerProd(e0_vec,e0_vec);
      rho1 = hypre_ParVectorInnerProd(e1_vec,e1_vec);
      rho = sqrt(rho1)/sqrt(rho0);
      if (rho > theta)
      {
         /*formu(CF_marker,num_variables,e1,A_i,rho);*/
         double candmeas=0.0e0, local_max=0.0e0, global_max = 0;
         double thresh=1-rho;

         for(i=0;i<num_variables;i++)
            if(fabs(e1[i]) > local_max)
               local_max = fabs(e1[i]);

	 MPI_Allreduce(&local_max,&global_max,1,MPI_DOUBLE,MPI_MAX,comm);
         for (i=0;i<num_variables;i++)
	 {
            if (CF_marker[i] == fpt)
	    {
	       candmeas = pow(fabs(e1[i]),1.0)/global_max;
	       if (candmeas > thresh && 
		  	(A_i[i+1]-A_i[i]+A_offd_i[i+1]-A_offd_i[i]) > 1)
               {
                  CF_marker[i] = cand; 
               }
            }
         }
   	 if (IS_type == 1)
	     hypre_BoomerAMGIndepHMIS(A,0,0,CF_marker);
   	 else if (IS_type == 2)
	     hypre_BoomerAMGIndepPMIS(A,0,0,CF_marker);
   	 else if (IS_type == 3)
	 {
            IndepSetGreedy(A_i,A_j,num_variables,CF_marker);
	 }
   	 else 
	    hypre_BoomerAMGIndepRS(A,0,0,CF_marker);

         if (my_id == 0) fprintf(stdout,"  %d \t%2.3lf  \t%2.3lf \n",
                 nstages,rho,(double)global_nc/(double)global_num_variables);
        /* update for next sweep */
         num_coarse = 0;
         for (i=0;i<num_variables;i++)
         {
	    if (CF_marker[i] ==  cpt) 
               num_coarse++;
	    else if (CF_marker[i] ==  fpt)
            { 
               e0[i] = 1.0e0+.1*drand48();
               e1[i] = 1.0e0+.1*drand48();
            }
         }
         nstages += 1;
	 MPI_Allreduce(&num_coarse,&global_nc,1,MPI_INT,MPI_MAX,comm);
      } 
      else 
      {
         if (my_id == 0) fprintf(stdout,"  %d \t%2.3lf  \t%2.3lf \n",
                 nstages,rho,(double)global_nc/(double)global_num_variables);
         break;
      }
   }
   hypre_ParVectorDestroy(e0_vec); 
   hypre_ParVectorDestroy(e1_vec); 
   hypre_ParVectorDestroy(Vtemp); 

   if (my_id == 0) fprintf(stdout,"\n... Done \n\n");
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

