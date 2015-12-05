#include "headers.h"
#include "Common.h"


#ifdef HYPRE_USING_ESSL
#include <essl.h>
#else
HYPRE_Int hypre_F90_NAME_BLAS(dgemm, DGEMM) (char *, char *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *, double *, double *, HYPRE_Int *, double *, HYPRE_Int *, double *, double *, HYPRE_Int *);
HYPRE_Int hypre_F90_NAME_BLAS(dgemv, DGEMV) (char *, HYPRE_Int * , HYPRE_Int * , double *, double *, HYPRE_Int *, double *, HYPRE_Int *, double *, double *, HYPRE_Int *);

HYPRE_Int hypre_F90_NAME_LAPACK(dgetrf, DGETRF) (HYPRE_Int *, HYPRE_Int *, double *, HYPRE_Int *, HYPRE_Int *, HYPRE_Int *);
HYPRE_Int hypre_F90_NAME_LAPACK(dgetrs, DGETRS) (char *, HYPRE_Int *, HYPRE_Int *, double *, HYPRE_Int *, HYPRE_Int *, double *b, HYPRE_Int*, HYPRE_Int *);

#endif

#define ADJUST(a,b)  (adjust_list[(a)*(num_functions-1)+(b)])

/******************************************************************************
 * hypre_BoomerAMGFitInterpVectors
 *
  This routine for updating the interp operator to interpolate the
  supplied smooth vectors with a L.S. fitting.  This code (varient 0)
  was used for the Baker, Kolev and Yang elasticity paper in section 3
  to evaluate the least squares fitting methed proposed by Stuben in
  his talk (see paper for details).  So this code is basically a
  post-processing step that performs the LS fit (the size and sparsity
  of P do not change).

  Note: truncation only works correctly for 1 processor - needs to
        just use the other truncation rouitne


  Variant = 0: do L.S. fit to existing interp weights (default)


  Variant = 1: extends the neighborhood to incl. other unknowns on the
  same node - ASSUMES A NODAL COARSENING, ASSUMES VARIABLES ORDERED
  GRID POINT, THEN UNKNOWN (e.g., u0, v0, u1, v1, etc. ), AND AT MOST
  3 FCNS (NOTE: **only** works with 1 processor)

  This code is not compiled or accessible through hypre at this time
  (it was not particularly effective - compared to the LN and GM
  approaches), but is checked-in in case there is interest in the
  future.

 ******************************************************************************/
HYPRE_Int hypre_BoomerAMGFitInterpVectors( hypre_ParCSRMatrix *A,
                                           hypre_ParCSRMatrix **P,
                                           HYPRE_Int num_smooth_vecs,
                                           hypre_ParVector **smooth_vecs,
                                           hypre_ParVector **coarse_smooth_vecs,
                                           double delta,
                                           HYPRE_Int num_functions, 
                                           HYPRE_Int *dof_func,
                                           HYPRE_Int *CF_marker, 
                                           HYPRE_Int max_elmts, 
                                           double trunc_factor, 
                                           HYPRE_Int variant, HYPRE_Int level) 
{
   
   HYPRE_Int  i,j, k;
   
   HYPRE_Int  one_i = 1;
   HYPRE_Int  info;
   HYPRE_Int  coarse_index;;
   HYPRE_Int  num_coarse_diag;
   HYPRE_Int  num_coarse_offd;
   HYPRE_Int  num_nonzeros = 0;
   HYPRE_Int  coarse_point = 0;
   HYPRE_Int  k_size;
   HYPRE_Int  k_alloc;
   HYPRE_Int  counter;
   HYPRE_Int  *piv;
   HYPRE_Int  tmp_int;
   HYPRE_Int  num_sends;

   double *alpha;
   double *Beta;
   double *w;
   double *w_old;
   double *B_s;
  
   double tmp_double;
   double one = 1.0;
   double mone = -1.0;;
   double *vec_data;

   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(*P);
   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(*P);
   double          *P_diag_data = hypre_CSRMatrixData(P_diag);
   HYPRE_Int       *P_diag_i = hypre_CSRMatrixI(P_diag);
   HYPRE_Int       *P_diag_j = hypre_CSRMatrixJ(P_diag);
   double          *P_offd_data = hypre_CSRMatrixData(P_offd);
   HYPRE_Int       *P_offd_i = hypre_CSRMatrixI(P_offd);
   HYPRE_Int       *P_offd_j = hypre_CSRMatrixJ(P_offd);
   HYPRE_Int	    num_rows_P = hypre_CSRMatrixNumRows(P_diag);
   HYPRE_Int        P_diag_size = P_diag_i[num_rows_P];
   HYPRE_Int        P_offd_size = P_offd_i[num_rows_P];
   HYPRE_Int        num_cols_P_offd = hypre_CSRMatrixNumCols(P_offd);
   HYPRE_Int       *col_map_offd_P;

   hypre_CSRMatrix  *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int         num_cols_A_offd = hypre_CSRMatrixNumCols(A_offd);

   hypre_ParCSRCommPkg     *comm_pkg = hypre_ParCSRMatrixCommPkg(*P);
   hypre_ParCSRCommHandle  *comm_handle;
   MPI_Comm                 comm;
   

   double  *dbl_buf_data;
   double  *smooth_vec_offd = NULL;
   double  *offd_vec_data;
   
   HYPRE_Int   index, start;
   HYPRE_Int  *P_marker;
   HYPRE_Int   num_procs;
 
   hypre_ParVector *vector;
  
   HYPRE_Int   new_nnz, orig_start, j_pos, fcn_num, num_elements;
   HYPRE_Int  *P_diag_j_new;
   double     *P_diag_data_new;
   HYPRE_Int   adjust_3D[] = {1, 2, -1, 1, -2, -1};
   HYPRE_Int   adjust_2D[] = {1, -1};
   HYPRE_Int  *adjust_list;
  
   if (variant ==1 && num_functions > 1)
   {
      /* First add new entries to P with value 0.0 corresponding to weights from 
         other unknowns on the same grid point */
      /* Loop through each row */
  
      new_nnz = P_diag_size*num_functions; /* this is an over-estimate */
      P_diag_j_new = hypre_CTAlloc(HYPRE_Int, new_nnz);
      P_diag_data_new = hypre_CTAlloc (double, new_nnz);
      

      if (num_functions ==2)
         adjust_list = adjust_2D;
      else if (num_functions ==3)
         adjust_list = adjust_3D;
  
      j_pos = 0;
      orig_start = 0;
      /* loop through rows */
      for (i=0; i < num_rows_P; i++)
      {
         fcn_num = (HYPRE_Int) fmod(i, num_functions);
         if (fcn_num != dof_func[i]) 
            printf("WARNING - ROWS incorrectly ordered!\n");
         
         /* loop through elements */
         num_elements = P_diag_i[i+1] - orig_start;
         
         /* add zeros corrresponding to other unknowns */
         if (num_elements > 1)
         {
            for (j=0; j < num_elements; j++)
            {
               P_diag_j_new[j_pos] = P_diag_j[orig_start+j];
               P_diag_data_new[j_pos++] = P_diag_data[orig_start+j];
               
               for (k=0; k < num_functions-1; k++)
               {
                  P_diag_j_new[j_pos] = P_diag_j[orig_start+j]+ ADJUST(fcn_num,k);
                  P_diag_data_new[j_pos++] = 0.0;
               }
            }
         }
         else if (num_elements == 1)/* only one element - just copy to new */
         {
            P_diag_j_new[j_pos] = P_diag_j[orig_start];
            P_diag_data_new[j_pos++] = P_diag_data[orig_start];
         }
         orig_start = P_diag_i[i+1];
         if (num_elements > 1)
            P_diag_i[i+1] =  P_diag_i[i] + num_elements*num_functions;
         else
            P_diag_i[i+1] = P_diag_i[i] + num_elements;

         if (j_pos != P_diag_i[i+1]) printf("Problem!\n");
         

      }/* end loop through rows */

      /* modify P */
      hypre_TFree(P_diag_j);
      hypre_TFree(P_diag_data);
      hypre_CSRMatrixJ(P_diag) = P_diag_j_new;
      hypre_CSRMatrixData(P_diag) = P_diag_data_new;
      hypre_CSRMatrixNumNonzeros(P_diag) = P_diag_i[num_rows_P];
      P_diag_j = P_diag_j_new;
      P_diag_data = P_diag_data_new;
      
      /* check if there is already a comm pkg - if so, destroy*/
      if (comm_pkg)
      {
          hypre_MatvecCommPkgDestroy(comm_pkg );
          comm_pkg = NULL;
          
      }
      

   } /* end variant == 1 and num functions > 0 */



   /* For each row, we are updating the weights by 
      solving w = w_old + (delta)(Beta^T)Bs^(-1)(alpha - (Beta)w_old).
      let s = num_smooth_vectors
      let k = # of interp points for fine point i
      Then:
      w = new weights (k x 1)
      w_old = old weights (k x 1)
      delta is a scalar weight in [0,1]
      alpha = s x 1 vector of s smooth vector values at fine point i
      Beta = s x k matrix of s smooth vector values at k interp points of i
      Bs = delta*Beta*Beta^T+(1-delta)*I_s (I_s is sxs identity matrix)
   */



#if 0
   /* print smoothvecs */
   {
      char new_file[80];

      for (i=0; i < num_smooth_vecs; i++)
      {
         sprintf(new_file,"%s.%d.level.%d","smoothvec", i, level );
         hypre_ParVectorPrint(smooth_vecs[i], new_file); 
      }
   }
   
#endif

   /*initial*/
   if (num_smooth_vecs == 0)
      return hypre_error_flag;

   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate ( *P );
      comm_pkg = hypre_ParCSRMatrixCommPkg(*P);
   }
   

   comm      = hypre_ParCSRCommPkgComm(comm_pkg);

   MPI_Comm_size(comm, &num_procs);

   num_nonzeros = hypre_CSRMatrixNumNonzeros(P_diag)
      + hypre_CSRMatrixNumNonzeros(P_offd);
  
   /* number of coarse points = number of cols */
   coarse_points = hypre_CSRMatrixNumCols(P_diag) + hypre_CSRMatrixNumCols(P_offd);

   /* allocate */
   alpha = hypre_CTAlloc(double, num_smooth_vecs);
   piv = hypre_CTAlloc(HYPRE_Int, num_smooth_vecs);
   B_s = hypre_CTAlloc(double, num_smooth_vecs*num_smooth_vecs);

   /*estimate the max number of weights per row (coarse points only have one weight)*/
   k_alloc = (num_nonzeros - coarse_points)/(num_rows_P - coarse_points);
   k_alloc += 5;

   Beta = hypre_CTAlloc(double, k_alloc*num_smooth_vecs);
   w = hypre_CTAlloc(double, k_alloc);                                             
   w_old = hypre_CTAlloc(double, k_alloc);  

   /* Get smooth vec components for the off-processor columns */

   if (num_procs > 1)
   {
      
      smooth_vec_offd =  hypre_CTAlloc(double, num_cols_P_offd*num_smooth_vecs);
      
      /* for now, do a seperate comm for each smooth vector */
      for (k = 0; k< num_smooth_vecs; k++)
      {
         
         vector = smooth_vecs[k];
         vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
         
         num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
         dbl_buf_data = hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                              num_sends));
         /* point into smooth_vec_offd */
         offd_vec_data =  smooth_vec_offd + k*num_cols_P_offd;
         
         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
               dbl_buf_data[index++] 
                  = vec_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
         }
         
         comm_handle = hypre_ParCSRCommHandleCreate( 1, comm_pkg, dbl_buf_data, 
                                                     offd_vec_data);
         
         hypre_ParCSRCommHandleDestroy(comm_handle); 
         
         hypre_TFree(dbl_buf_data);
      }
   }/*end num procs > 1 */
   /* now off-proc smooth vec data is in smoothvec_offd */

   /* Loop through each row */
   for (i=0; i < num_rows_P; i++)
   {

      /* only need to modify rows belonging to fine points */
      if (CF_marker[i] >= 0) /* coarse */
         continue;

      num_coarse_diag = P_diag_i[i+1] - P_diag_i[i];
      num_coarse_offd =  P_offd_i[i+1] - P_offd_i[i];

      k_size = num_coarse_diag + num_coarse_offd;
      

      /* only need to modify rows that interpolate from coarse points */
      if (k_size == 0)
         continue;

#if 0
      /* only change the weights if we have at least as many coarse points
         as smooth vectors - do we want to do this? NO */
    
      too_few = 0;
      if (k_size < num_smooth_vecs)
      {
         too_few++;
         continue;
      }
#endif      

      /*verify that we have enough space allocated */
      if (k_size > k_alloc)
      {
         k_alloc = k_size + 2;

         Beta = hypre_TReAlloc(Beta, double, k_alloc*num_smooth_vecs);
         w = hypre_TReAlloc(w, double, k_alloc);
         w_old = hypre_TReAlloc(w_old, double, k_alloc);
      }

      /* put current weights into w*/
      counter = 0;
      for (j=P_diag_i[i];j <  P_diag_i[i+1]; j++)
      {
         w[counter++] = P_diag_data[j];
      }
      for (j=P_offd_i[i];j <  P_offd_i[i+1]; j++)
      {
         w[counter++] = P_offd_data[j];
      }

      /* copy w to w_old */
      for (j=0; j< k_size; j++)
         w_old[j] = w[j];
            
      /* get alpha and Beta */
      /* alpha is the smooth vector values at fine point i */
      /* Beta is the smooth vector values at the points that
         i interpolates from */

      /* Note - for using BLAS/LAPACK - need to store Beta in
       * column-major order */

      for (j = 0; j< num_smooth_vecs; j++)
      {
         vector = smooth_vecs[j];
         vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
         /* point into smooth_vec_offd */
         offd_vec_data = smooth_vec_offd + j*num_cols_P_offd;

         alpha[j] = vec_data[i];
         
         vector = coarse_smooth_vecs[j];
         vec_data = hypre_VectorData(hypre_ParVectorLocalVector(vector));
         /* on processor */
         counter = 0;
         
         for (k = P_diag_i[i]; k <  P_diag_i[i+1]; k++)
         {
            coarse_index = P_diag_j[k];
            /*Beta(j, counter) */
            Beta[counter*num_smooth_vecs + j] = vec_data[coarse_index];
            counter++;
         }
         /* off-processor */
         for (k = P_offd_i[i];k <  P_offd_i[i+1]; k++)
         {
            coarse_index = P_offd_j[k];
            Beta[counter*num_smooth_vecs + j] = offd_vec_data[coarse_index];
            counter++;
            
         }

      }

      /* form B_s: delta*Beta*Beta^T + (1-delta)*I_s */

      /* first B_s <- (1-delta)*I_s */
      tmp_double = 1.0 - delta;
      for (j = 0; j < num_smooth_vecs*num_smooth_vecs; j++)
         B_s[j] = 0.0;
       for (j = 0; j < num_smooth_vecs; j++)
          B_s[j*num_smooth_vecs + j] = tmp_double;
       
       /* now  B_s <-delta*Beta*Beta^T + B_s */ 
       /* usage: DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
                 C := alpha*op( A )*op( B ) + beta*C */
       hypre_F90_NAME_BLAS(dgemm,DGEMM)("N", "T", &num_smooth_vecs, 
                                        &num_smooth_vecs, &k_size, 
                                        &delta, Beta, &num_smooth_vecs, Beta, 
                                        &num_smooth_vecs, &one, B_s, &num_smooth_vecs);
       
       /* now do alpha <- (alpha - beta*w)*/
       /* usage: DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
                 y := alpha*A*x + beta*y */
       hypre_F90_NAME_BLAS(dgemv,DGEMV)("N", &num_smooth_vecs, &k_size, &mone, 
                                        Beta, &num_smooth_vecs, w_old, &one_i, 
                                        &one, alpha, &one_i);
       
      
       /* now get alpha <- inv(B_s)*alpha */
           /*write over B_s with LU */
       hypre_F90_NAME_LAPACK(dgetrf, DGETRF)(&num_smooth_vecs, &num_smooth_vecs, 
                                             B_s, &num_smooth_vecs, piv, &info);
       
           /*now get alpha  */
       hypre_F90_NAME_LAPACK(dgetrs, DGETRS)("N", &num_smooth_vecs, &one_i, B_s, 
                                             &num_smooth_vecs, piv, alpha, 
                                             &num_smooth_vecs, &info);
       


       /* now w <- w + (delta)*(Beta)^T*(alpha) */
       hypre_F90_NAME_BLAS(dgemv,DGEMV)("T", &num_smooth_vecs, &k_size, &delta, 
                                        Beta, &num_smooth_vecs, alpha, &one_i, 
                                        &one, w, &one_i);

       
       /* note:we have w_old still, but we don't need it unless we
        * want to use it in the future for something */

       /* now update the weights in P*/
       counter = 0;
       for (j=P_diag_i[i];j <  P_diag_i[i+1]; j++)
       {
          P_diag_data[j] = w[counter++];
       }
       for (j=P_offd_i[i];j <  P_offd_i[i+1]; j++)
       {
          P_offd_data[j] = w[counter++];
       }
   }/* end of loop through each row */
                       
                                              
   /* clean up from L.S. fitting*/
   hypre_TFree(alpha);
   hypre_TFree(Beta);
   hypre_TFree(w);
   hypre_TFree(w_old);
   hypre_TFree(piv);
   hypre_TFree(B_s);
   hypre_TFree(smooth_vec_offd);
   
   /* Now we truncate here (instead of after forming the interp matrix) */

   /* SAME code as in othr interp routines:
      Compress P, removing coefficients smaller than trunc_factor * Max , 
      or when there are more than max_elements*/

   if (trunc_factor != 0.0 || max_elmts > 0)
   {

      /* To DO: THIS HAS A BUG IN PARALLEL! */

      tmp_int =  P_offd_size;
      
      hypre_BoomerAMGInterpTruncation(*P, trunc_factor, max_elmts);
      P_diag_data = hypre_CSRMatrixData(P_diag);
      P_diag_i = hypre_CSRMatrixI(P_diag);
      P_diag_j = hypre_CSRMatrixJ(P_diag);
      P_offd_data = hypre_CSRMatrixData(P_offd);
      P_offd_i = hypre_CSRMatrixI(P_offd);
      P_offd_j = hypre_CSRMatrixJ(P_offd);
      P_diag_size = P_diag_i[num_rows_P];
      
      P_offd_size = P_offd_i[num_rows_P];

   
      /* if truncation occurred, we need to re-do the col_map_offd... */
      if (tmp_int != P_offd_size)
      {
         num_cols_P_offd = 0;
         P_marker = hypre_CTAlloc(HYPRE_Int, num_cols_A_offd);

         for (i=0; i < num_cols_A_offd; i++)
            P_marker[i] = 0;

         num_cols_P_offd = 0;
         for (i=0; i < P_offd_size; i++)
         {
            index = P_offd_j[i];
            if (!P_marker[index])
            {
               num_cols_P_offd++;
               P_marker[index] = 1;
            }
         }

         col_map_offd_P = hypre_CTAlloc(HYPRE_Int, num_cols_P_offd);

         index = 0;
         for (i=0; i < num_cols_P_offd; i++)
         {
            while (P_marker[index]==0) index++;
            col_map_offd_P[i] = index++;
         }
         for (i=0; i < P_offd_size; i++)
            P_offd_j[i] = hypre_BinarySearch(col_map_offd_P,
                                             P_offd_j[i],
                                             num_cols_P_offd);
         hypre_TFree(P_marker); 
         hypre_TFree( hypre_ParCSRMatrixColMapOffd(*P));

         /* assign new col map */
         hypre_ParCSRMatrixColMapOffd(*P) = col_map_offd_P;
         hypre_CSRMatrixNumCols(P_offd) = num_cols_P_offd;


         /* destroy the old and get a new commpkg....*/
         hypre_MatvecCommPkgDestroy(comm_pkg);
         hypre_MatvecCommPkgCreate ( *P );


      }/*end re-do col_map_offd */
      
   }/*end trucation */
   
   return hypre_error_flag;
   
   
}
