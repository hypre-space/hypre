
#include "headers.h"

/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBlockRelaxIF

   This is the block version of the relaxation routines. 

   A is now a Block matrix.  

   CF_marker is size number of nodes

 *--------------------------------------------------------------------------*/

int  hypre_BoomerAMGBlockRelaxIF( hypre_ParCSRBlockMatrix *A,
                             hypre_ParVector    *f,
                             int                *cf_marker,
                             int                 relax_type,
                             int                 relax_order,
                             int                 cycle_type,
                             double              relax_weight,
                             double              omega,
                             hypre_ParVector    *u,
                             hypre_ParVector    *Vtemp )
{

   int i, Solve_err_flag = 0;
   int relax_points[2];
   
   if (relax_order == 1 && cycle_type < 3)
      /* if do C/F and not on the cg */
   {
      if (cycle_type < 2) /* 0 = fine, 1 = down */
      {
         relax_points[0] = 1;
	 relax_points[1] = -1;
      }
      else  /* 2 = up */
      {
	 relax_points[0] = -1;
	 relax_points[1] = 1;
      }
      for (i=0; i < 2; i++)
      {
         Solve_err_flag = hypre_BoomerAMGBlockRelax(A,
                                                    f,
                                                    cf_marker,
                                                    relax_type,
                                                    relax_points[i],
                                                    relax_weight,
                                                    omega,
                                                    u,
                                                    Vtemp); 
      }
   }
   else /* either on the cg or doing normal relaxation (no C/F relaxation) */
   {
      Solve_err_flag = hypre_BoomerAMGBlockRelax(A,
                                                 f,
                                                 cf_marker,
                                                 relax_type,
                                                 0,
                                                 relax_weight,
                                                 omega,
                                                 u,
                                                 Vtemp); 
   }

   return Solve_err_flag;
}


/*---------------------------------------------------------------------------
 * hypre_BoomerAMGBlockRelax

   This is the block version of the relaxation routines. 

   A is now a Block matrix.  

   CF_marker is size number of nodes.

 *--------------------------------------------------------------------------*/
int  hypre_BoomerAMGBlockRelax( hypre_ParCSRBlockMatrix *A,
                                hypre_ParVector    *f,
                                int                *cf_marker,
                                int                 relax_type,
                                int                 relax_points,
                                double              relax_weight,
                                double              omega,
                                hypre_ParVector    *u,
                                hypre_ParVector    *Vtemp )

{
   


   MPI_Comm	         comm = hypre_ParCSRBlockMatrixComm(A);

   hypre_CSRBlockMatrix *A_diag       = hypre_ParCSRBlockMatrixDiag(A);
   double               *A_diag_data  = hypre_CSRBlockMatrixData(A_diag);
   int                  *A_diag_i     = hypre_CSRBlockMatrixI(A_diag);
   int                  *A_diag_j     = hypre_CSRBlockMatrixJ(A_diag);

   hypre_CSRBlockMatrix *A_offd       = hypre_ParCSRBlockMatrixOffd(A);
   int                  *A_offd_i     = hypre_CSRBlockMatrixI(A_offd);
   double               *A_offd_data  = hypre_CSRBlockMatrixData(A_offd);
   int                  *A_offd_j     = hypre_CSRBlockMatrixJ(A_offd);

   hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRBlockMatrixCommPkg(A);
   hypre_ParCSRCommHandle *comm_handle;

   int             block_size = hypre_CSRBlockMatrixBlockSize(A_diag);
   int             bnnz = block_size*block_size;

   int             n_global;
   int             n             = hypre_CSRBlockMatrixNumRows(A_diag);
   int             num_cols_offd = hypre_CSRBlockMatrixNumCols(A_offd);
   int	      	   first_index = hypre_ParVectorFirstIndex(u);
   
   hypre_Vector   *u_local = hypre_ParVectorLocalVector(u);
   double         *u_data  = hypre_VectorData(u_local);

   hypre_Vector   *f_local = hypre_ParVectorLocalVector(f);
   double         *f_data  = hypre_VectorData(f_local);

   hypre_Vector   *Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   double         *Vtemp_data  = hypre_VectorData(Vtemp_local);
   double 	  *Vext_data;
   double 	  *v_buf_data;
   
   double         *tmp_data;

   int            size, rest, ne, ns;
   

   int             i, j, k;
   int             ii, jj;

   int             relax_error = 0;
   int		   num_sends;
   int		   index, start;
   int		   num_procs, num_threads, my_id;
 
   double	   *res_vec, *out_vec, *tmp_vec;
   double          *res0_vec, *res2_vec;
   double          one_minus_weight;
   double          one_minus_omega;
   double          prod;

   hypre_CSRMatrix *A_CSR;
   int		   *A_CSR_i;   
   int		   *A_CSR_j;
   double	   *A_CSR_data;
 
   hypre_Vector    *f_vector;
   double	   *f_vector_data;

   hypre_ParCSRMatrix *A_ParCSR;

   double         *A_mat;
   double         *b_vec;

   int             column;

   /* initialize some stuff */
   one_minus_weight = 1.0 - relax_weight;
   one_minus_omega = 1.0 - omega;
   MPI_Comm_size(comm, &num_procs);  
   MPI_Comm_rank(comm, &my_id);  
   num_threads = hypre_NumThreads();

   res_vec = hypre_CTAlloc(double, block_size);
   out_vec = hypre_CTAlloc(double, block_size);
   tmp_vec = hypre_CTAlloc(double, block_size);

   /*-----------------------------------------------------------------------
    * Switch statement to direct control based on relax_type:
    *     relax_type = 20 -> Jacobi or CF-Jacobi

    *     relax_type = 23 -> hybrid: SOR-J mix off-processor, SOR on-processor
    *     		    with outer relaxation parameters (forward solve)

    *     relax_type = 29 -> Direct Solve
    *-----------------------------------------------------------------------*/
   switch (relax_type)
   {            

/*---------------------------------------------------------------------------
            Jacobi
---------------------------------------------------------------------------*/
      case 20:
      {

         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
            v_buf_data = hypre_CTAlloc(double, 
                                       hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends)*block_size);
            Vext_data = hypre_CTAlloc(double, num_cols_offd*block_size);
            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRBlockMatrixJ(A_offd);
               A_offd_data = hypre_CSRBlockMatrixData(A_offd);
            }
            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
               {
                  for (k = 0; k < block_size; k++)
                  {
                     v_buf_data[index++] 
                        = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)*block_size + k];
                  }
               }
            }
 
            /* we need to use the block comm handle here - since comm_pkg is nodal based */   
            comm_handle = hypre_ParCSRBlockCommHandleCreate(1, block_size, comm_pkg,
                                                           v_buf_data, Vext_data );
           
         }

         /*-----------------------------------------------------------------
          * Copy current approximation into temporary vector.
          *-----------------------------------------------------------------*/
         
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
         for (i = 0; i < n*block_size; i++)
         {
            Vtemp_data[i] = u_data[i];
         }
         if (num_procs > 1)
         { 
            hypre_ParCSRBlockCommHandleDestroy(comm_handle); /* now Vext_data 
                                                                is populated */
            comm_handle = NULL;
         } 
         /*-----------------------------------------------------------------
          * Relax all points.
          *-----------------------------------------------------------------*/

         if (relax_points == 0)
         {
#define HYPRE_SMP_PRIVATE i,ii,jj,res
#include "../utilities/hypre_smp_forloop.h"
            for (i = 0; i < n; i++)
            {
               
               /*-----------------------------------------------------------
                * If diagonal is nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
               
               for (k=0; k< block_size; k++) 
               {
                  res_vec[k] = f_data[i*block_size+k];
               }
               for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
               {
                  ii = A_diag_j[jj];
                  /* res -= A_diag_data[jj] * Vtemp_data[ii]; */
                  hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                  &Vtemp_data[ii*block_size], 
                                                  1.0, res_vec, block_size);
               }
               for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
               {
                  ii = A_offd_j[jj];
                  /* res -= A_offd_data[jj] * Vext_data[ii]; */
                  hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj*bnnz], 
                                                  &Vext_data[ii*block_size], 
                                                  1.0, res_vec, block_size);
               }
               
               /* if diag is singular, then skip this point */ 
               if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec, 
                                                       out_vec, block_size) == 0)
               {
                  for (k=0; k< block_size; k++) 
                  {                     
                     u_data[i*block_size+k] *= one_minus_weight; 
                     u_data[i*block_size+k] += relax_weight *out_vec[k];
                  }
                  
               }
            }
         }
         
         /*-----------------------------------------------------------------
          * Relax only C or F points as determined by relax_points.
          *-----------------------------------------------------------------*/
         
         else
         {
#define HYPRE_SMP_PRIVATE i,ii,jj,res
#include "../utilities/hypre_smp_forloop.h"
            for (i = 0; i < n; i++)
            {
               
               /*-----------------------------------------------------------
                * If i is of the right type ( C or F ) and diagonal is
                * nonzero, relax point i; otherwise, skip it.
                *-----------------------------------------------------------*/
               
               if (cf_marker[i] == relax_points) 
               {
                  
                  for (k=0; k< block_size; k++) 
                  {
                     res_vec[k] = f_data[i*block_size+k];
                  }
                  for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                  {
                     ii = A_diag_j[jj];
                     /* res -= A_diag_data[jj] * Vtemp_data[ii]; */
                     hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                     &Vtemp_data[ii*block_size], 
                                                     1.0, res_vec, block_size);
                  }
                  for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                  {
                     ii = A_offd_j[jj];
                     /* res -= A_offd_data[jj] * Vext_data[ii]; */
                     hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj*bnnz], 
                                                     &Vext_data[ii*block_size], 
                                                     1.0, res_vec, block_size);
                  }
                  
                  /* if diag is singular, then skip this point */ 
                  if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec, 
                                                          out_vec, block_size) == 0)
                  {
                     for (k=0; k< block_size; k++) 
                     {                     
                        u_data[i*block_size+k] *= one_minus_weight; 
                        u_data[i*block_size+k] += relax_weight *out_vec[k];
                     }
                     
                  }
               }
            }     
         }
         if (num_procs > 1)
         {
            hypre_TFree(Vext_data);
            hypre_TFree(v_buf_data);
         }
         
         break;
         
      } /* end case 20 */

/*---------------------------------------------------------------------------
 Hybrid: G-S on proc. and Jacobi off proc.
---------------------------------------------------------------------------*/
      case 23: 
      {
         if (num_procs > 1)
         {
            num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
            v_buf_data = hypre_CTAlloc(double, 
                                       hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends)*block_size);
            Vext_data = hypre_CTAlloc(double, num_cols_offd*block_size);
            if (num_cols_offd)
            {
               A_offd_j = hypre_CSRBlockMatrixJ(A_offd);
               A_offd_data = hypre_CSRBlockMatrixData(A_offd);
            }
            index = 0;
            for (i = 0; i < num_sends; i++)
            {
               start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
               for (j=start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
               {
                  for (k = 0; k < block_size; k++)
                  {
                     v_buf_data[index++] 
                        = u_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)*block_size + k];
                  }
               }
            }
 
            /* we need to use the block comm handle here - since comm_pkg is nodal based */   
            comm_handle = hypre_ParCSRBlockCommHandleCreate(1, block_size, comm_pkg,
                                                           v_buf_data, Vext_data );
           
         }

        /*-----------------------------------------------------------------
         * Copy current approximation into temporary vector.
         *-----------------------------------------------------------------*/
         
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
         for (i = 0; i < n*block_size; i++)
         {
            Vtemp_data[i] = u_data[i];
         }

         if (num_procs > 1)
         { 
            hypre_ParCSRBlockCommHandleDestroy(comm_handle); /* now Vext_data 
                                                                is populated */
            comm_handle = NULL;
        } 
         

        /*-----------------------------------------------------------------
         * relax weight and omega = 1
         *-----------------------------------------------------------------*/
         
         if (relax_weight == 1 && omega == 1)
         {

            /*-----------------------------------------------------------------
             * Relax all points.
             *-----------------------------------------------------------------*/
            if (relax_points == 0)
            {
               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(double,n);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++)	/* interior points first */
                     {
                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/
                        
                        for (k=0; k< block_size; k++) 
                        {
                           res_vec[k] = f_data[i*block_size+k];
                        }
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           if (ii >= ns && ii < ne)
                           {
                              /*  res -= A_diag_data[jj] * u_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                              &u_data[ii*block_size], 
                                                              1.0, res_vec, block_size);
                           }
                           else
                           {
                              /* res -= A_diag_data[jj] * tmp_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                              &tmp_data[ii*block_size], 
                                                              1.0, res_vec, block_size);
                           }
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj*bnnz], 
                                                           &Vext_data[ii*block_size], 
                                                           1.0, res_vec, block_size);
                           
                        }
                        /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                        /* if diag is singular, then skip this point */ 
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec, 
                                                                out_vec, block_size) == 0)
                        {
                           for (k=0; k< block_size; k++) 
                           {                     
                              u_data[i*block_size+k] = out_vec[k];
                           }
                        }
                     } /* for loop over points */
                  } /* foor loop over threads */
                  hypre_TFree(tmp_data);
               }
               else /* num_threads = 1 */
               {
                  for (i = 0; i < n; i++)	/* interior points first */
                  {
                     
                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/
                     for (k=0; k< block_size; k++) 
                     {
                        res_vec[k] = f_data[i*block_size+k];
                     }
                     for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                     {
                        ii = A_diag_j[jj];
                        /* res -= A_diag_data[jj] * u_data[ii]; */
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                        &u_data[ii*block_size], 
                                                        1.0, res_vec, block_size);
                     }
                     for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                     {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii]; */
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj*bnnz], 
                                                        &Vext_data[ii*block_size], 
                                                        1.0, res_vec, block_size);
                     }
                    /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                     if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec, 
                                                            out_vec, block_size) == 0)
                     {
                        for (k=0; k< block_size; k++) 
                        {                     
                           u_data[i*block_size+k] = out_vec[k];
                        }
                     }
                  } /* for loop over points */
               } /* end of num_threads = 1 */
            } /* end of non-CF relaxation */
           
           /*-----------------------------------------------------------------
            * Relax only C or F points as determined by relax_points.
            *-----------------------------------------------------------------*/
            else
            {
               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(double,n);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++) /* relax interior points */
                     {
                        /*-----------------------------------------------------------
                        * If i is of the right type ( C or F ) and diagonal is
                        * nonzero, relax point i; otherwise, skip it.
                        *-----------------------------------------------------------*/
                        if (cf_marker[i] == relax_points )
                        {
                           for (k=0; k< block_size; k++) 
                           {
                              res_vec[k] = f_data[i*block_size+k];
                           }
                           for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                           {
                              ii = A_diag_j[jj];
                              if (ii >= ns && ii < ne)
                              {
                                 /* res -= A_diag_data[jj] * u_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                                 &u_data[ii*block_size], 
                                                                 1.0, res_vec, block_size);
                              }
                              else
                              {
                                 /* res -= A_diag_data[jj] * tmp_data[ii]; */
                                 hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                                 &tmp_data[ii*block_size], 
                                                                 1.0, res_vec, block_size);
                              }
                           }
                           for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                           {
                              ii = A_offd_j[jj];
                              /* res -= A_offd_data[jj] * Vext_data[ii];*/
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj*bnnz], 
                                                              &Vext_data[ii*block_size], 
                                                              1.0, res_vec, block_size);
                             
                           }
                           /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                           /* if diag is singular, then skip this point */ 
                           if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec, 
                                                                   out_vec, block_size) == 0)
                           {
                              for (k=0; k< block_size; k++) 
                              {                     
                                 u_data[i*block_size+k] = out_vec[k];
                              }
                           }
                        }
                     } /* loop over points */    
                  } /* loop over threads */     
                  hypre_TFree(tmp_data);
               }
               else /* num_threads = 1 */
               {
                  for (i = 0; i < n; i++) /* relax interior points */
                  {
                     /*-----------------------------------------------------------
                      * If i is of the right type ( C or F ) and diagonal is
                      * nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/
                     if (cf_marker[i] == relax_points )
                     {
                        for (k=0; k< block_size; k++) 
                        {
                           res_vec[k] = f_data[i*block_size+k];
                        }
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           /* res -= A_diag_data[jj] * u_data[ii]; */
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                              &u_data[ii*block_size], 
                                                           1.0, res_vec, block_size);
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj*bnnz], 
                                                           &Vext_data[ii*block_size], 
                                                           1.0, res_vec, block_size);
                           
                        }
                        /* u_data[i] = res / A_diag_data[A_diag_i[i]]; */
                        /* if diag is singular, then skip this point */ 
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], res_vec, 
                                                                out_vec, block_size) == 0)
                        {
                           for (k=0; k< block_size; k++) 
                           {                     
                              u_data[i*block_size+k] = out_vec[k];
                           }
                        }
                     }
                  } /* end of loop over points */     
               } /* end of num_threads = 1 */
            } /* end of C/F option */
         }
         else 
         {
           /*-----------------------------------------------------------------
            * relax weight and omega do not = 1
            *-----------------------------------------------------------------*/
            prod = (1.0-relax_weight*omega);
            res0_vec = hypre_CTAlloc(double, block_size);
            res2_vec = hypre_CTAlloc(double, block_size);

            if (relax_points == 0)
            {
               /*-----------------------------------------------------------------
                * Relax all points.
                *-----------------------------------------------------------------*/

               if (num_threads > 1)
               {
                  tmp_data = hypre_CTAlloc(double,n);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
                  for (i = 0; i < n; i++)
                     tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
                  for (j = 0; j < num_threads; j++)
                  {
                     size = n/num_threads;
                     rest = n - size*num_threads;
                     if (j < rest)
                     {
                        ns = j*size+j;
                        ne = (j+1)*size+j+1;
                     }
                     else
                     {
                        ns = j*size+rest;
                        ne = (j+1)*size+rest;
                     }
                     for (i = ns; i < ne; i++)	/* interior points first */
                     {
                        /*-----------------------------------------------------------
                         * If diagonal is nonzero, relax point i; otherwise, skip it.
                         *-----------------------------------------------------------*/
                        for (k=0; k< block_size; k++) 
                        {
                           res_vec[k] = f_data[i*block_size+k];
                           res0_vec[k] = 0.0;
                           res2_vec[k] = 0.0;
                        }
                        for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                        {
                           ii = A_diag_j[jj];
                           if (ii >= ns && ii < ne)
                           {
                              /* res0 -= A_diag_data[jj] * u_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                              &u_data[ii*block_size], 
                                                              1.0, res0_vec, block_size);
                              /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                              hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj*bnnz], 
                                                              &Vtemp_data[ii*block_size], 
                                                              1.0, res2_vec, block_size);
                           }
                           else
                           {
                              /* res -= A_diag_data[jj] * tmp_data[ii]; */
                              hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                              &tmp_data[ii*block_size], 
                                                              1.0, res_vec, block_size);
                           }
                        }
                        for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                        {
                           ii = A_offd_j[jj];
                           /* res -= A_offd_data[jj] * Vext_data[ii];*/
                           hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj*bnnz], 
                                                           &Vext_data[ii*block_size], 
                                                           1.0, res_vec, block_size);
                        }
                        /* u_data[i] *= prod;
                           u_data[i] += relax_weight*(omega*res + res0 +
                           one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                        for (k=0; k< block_size; k++) 
                        {
                           tmp_vec[k] =  omega*res_vec[k] + res0_vec[k] + one_minus_omega*res2_vec[k];
                        }
                        if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec, 
                                                                out_vec, block_size) == 0)
                        {
                           for (k=0; k< block_size; k++) 
                           {                     
                              u_data[i*block_size+k] *= prod; 
                              u_data[i*block_size+k] += relax_weight *out_vec[k];
                           }
                        }
                        
                     } /* end of loop over points */
                  } /* end of loop over threads */
                  hypre_TFree(tmp_data);
               }
               else /* num_threads = 1 */
               {
                  for (i = 0; i < n; i++)	/* interior points first */
                  {
                     /*-----------------------------------------------------------
                      * If diagonal is nonzero, relax point i; otherwise, skip it.
                      *-----------------------------------------------------------*/
                     for (k=0; k< block_size; k++) 
                     {
                        res_vec[k] = f_data[i*block_size+k];
                        res0_vec[k] = 0.0;
                        res2_vec[k] = 0.0;
                     }
                     for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                     {
                        ii = A_diag_j[jj];
                        /* res0 -= A_diag_data[jj] * u_data[ii]; */
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                        &u_data[ii*block_size], 
                                                        1.0, res0_vec, block_size);
                        /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                        hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj*bnnz], 
                                                        &Vtemp_data[ii*block_size], 
                                                        1.0, res2_vec, block_size);
                     }
                     for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                     {
                        ii = A_offd_j[jj];
                        /* res -= A_offd_data[jj] * Vext_data[ii];*/
                        hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj*bnnz], 
                                                        &Vext_data[ii*block_size], 
                                                        1.0, res_vec, block_size);
                     }
                     /* u_data[i] *= prod;
                        u_data[i] += relax_weight*(omega*res + res0 +
                        one_minus_omega*res2) / A_diag_data[A_diag_i[i]]; */
                     for (k=0; k< block_size; k++) 
                     {
                        tmp_vec[k] =  omega*res_vec[k] + res0_vec[k] + one_minus_omega*res2_vec[k];
                     }
                     if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec, 
                                                             out_vec, block_size) == 0)
                     {
                        for (k=0; k< block_size; k++) 
                        {                     
                           u_data[i*block_size+k] *= prod; 
                           u_data[i*block_size+k] += relax_weight *out_vec[k];
                        }
                     }
                  } /* end of loop over points */
               } /* end num_threads = 1 */
            }
            /*-----------------------------------------------------------------
             * Relax only C or F points as determined by relax_points.
             *-----------------------------------------------------------------*/
           else
           {
              if (num_threads > 1)
              {
                 tmp_data = hypre_CTAlloc(double,n);
#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
                 for (i = 0; i < n; i++)
                    tmp_data[i] = u_data[i];
#define HYPRE_SMP_PRIVATE i,ii,j,jj,ns,ne,res,rest,size
#include "../utilities/hypre_smp_forloop.h"
                 for (j = 0; j < num_threads; j++)
                 {
                    size = n/num_threads;
                    rest = n - size*num_threads;
                    if (j < rest)
                    {
                       ns = j*size+j;
                       ne = (j+1)*size+j+1;
                    }
                    else
                    {
                       ns = j*size+rest;
                       ne = (j+1)*size+rest;
                    }
                    for (i = ns; i < ne; i++) /* relax interior points */
                    {
                       
                       /*-----------------------------------------------------------
                        * If i is of the right type ( C or F ) and diagonal is
                        * nonzero, relax point i; otherwise, skip it.
                        *-----------------------------------------------------------*/
                       
                       if (cf_marker[i] == relax_points) 
                       {
                          /*-----------------------------------------------------------
                           * If diagonal is nonzero, relax point i; otherwise, skip it.
                           *-----------------------------------------------------------*/
                          for (k=0; k< block_size; k++) 
                          {
                             res_vec[k] = f_data[i*block_size+k];
                             res0_vec[k] = 0.0;
                             res2_vec[k] = 0.0;
                          }
                          for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                          {
                             ii = A_diag_j[jj];
                             if (ii >= ns && ii < ne)
                             {
                                /* res0 -= A_diag_data[jj] * u_data[ii]; */
                                hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                                &u_data[ii*block_size], 
                                                                1.0, res0_vec, block_size);
                                /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                                hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj*bnnz], 
                                                                &Vtemp_data[ii*block_size], 
                                                                1.0, res2_vec, block_size);
                             }
                             else
                             {
                                /* res -= A_diag_data[jj] * tmp_data[ii]; */
                                hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                                &tmp_data[ii*block_size], 
                                                                1.0, res_vec, block_size);
                             }
                          }
                          for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                          {
                             ii = A_offd_j[jj];
                             /* res -= A_offd_data[jj] * Vext_data[ii];*/
                             hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj*bnnz], 
                                                             &Vext_data[ii*block_size], 
                                                             1.0, res_vec, block_size);
                          }
                          /* u_data[i] *= prod;
                             u_data[i] += relax_weight*(omega*res + res0 +
                             one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                          for (k=0; k< block_size; k++) 
                          {
                             tmp_vec[k] =  omega*res_vec[k] + res0_vec[k] + one_minus_omega*res2_vec[k];
                          }
                          if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec, 
                                                                  out_vec, block_size) == 0)
                          {
                             for (k=0; k< block_size; k++) 
                             {                     
                                u_data[i*block_size+k] *= prod; 
                                u_data[i*block_size+k] += relax_weight *out_vec[k];
                             }
                          }
                       } /* end of if cf_marker */
                    } /* end loop over points */     
                 } /* end loop over threads */     
                 hypre_TFree(tmp_data);
              }
              else /* num_threads = 1 */
              {
                 for (i = 0; i < n; i++) /* relax interior points */
                 {
                    
                    /*-----------------------------------------------------------
                     * If i is of the right type ( C or F ) and diagonal is
                     * nonzero, relax point i; otherwise, skip it.
                     *-----------------------------------------------------------*/
                    
                    if (cf_marker[i] == relax_points) 
                    {
                       /*-----------------------------------------------------------
                        * If diagonal is nonzero, relax point i; otherwise, skip it.
                        *-----------------------------------------------------------*/
                       for (k=0; k< block_size; k++) 
                       {
                          res_vec[k] = f_data[i*block_size+k];
                          res0_vec[k] = 0.0;
                          res2_vec[k] = 0.0;
                       }
                       for (jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
                       {
                          ii = A_diag_j[jj];
                          /* res0 -= A_diag_data[jj] * u_data[ii]; */
                          hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_diag_data[jj*bnnz], 
                                                          &u_data[ii*block_size], 
                                                          1.0, res0_vec, block_size);
                          /* res2 += A_diag_data[jj] * Vtemp_data[ii];*/
                          hypre_CSRBlockMatrixBlockMatvec(1.0, &A_diag_data[jj*bnnz], 
                                                          &Vtemp_data[ii*block_size], 
                                                          1.0, res2_vec, block_size);
                       }
                       for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
                       {
                          ii = A_offd_j[jj];
                          /* res -= A_offd_data[jj] * Vext_data[ii];*/
                          hypre_CSRBlockMatrixBlockMatvec(-1.0, &A_offd_data[jj*bnnz], 
                                                          &Vext_data[ii*block_size], 
                                                          1.0, res_vec, block_size);
                       }
                       /* u_data[i] *= prod;
                          u_data[i] += relax_weight*(omega*res + res0 +
                          one_minus_omega*res2) / A_diag_data[A_diag_i[i]];*/
                       for (k=0; k< block_size; k++) 
                       {
                          tmp_vec[k] =  omega*res_vec[k] + res0_vec[k] + one_minus_omega*res2_vec[k];
                       }
                       if (hypre_CSRBlockMatrixBlockInvMatvec( &A_diag_data[A_diag_i[i]*bnnz], tmp_vec, 
                                                               out_vec, block_size) == 0)
                       {
                          for (k=0; k< block_size; k++) 
                          {                     
                             u_data[i*block_size+k] *= prod; 
                             u_data[i*block_size+k] += relax_weight *out_vec[k];
                          }
                       } 
                    } /* end cf_marker */
                 }  /* end loop over points */   
              } /* end num_threads = 1 */
           } /* end C/F option */ 
            hypre_TFree(res0_vec);
            hypre_TFree(res2_vec);
         } /* end of check relax weight and omega */
         
         if (num_procs > 1)
         {
            hypre_TFree(Vext_data);
            hypre_TFree(v_buf_data);
         }
         

         break;
      }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
      case 29: /* Direct solve: use gaussian elimination */
      {
      
         /* for now, we convert to a parcsr and 
            then proceed as in non-block case  - shouldn't be too expensive
            since this is a small matrix.  Would be better to just store the CSR matrix
            in the amg_data structure - esp. in the case of no global partition */
         
         A_ParCSR =  hypre_ParCSRBlockMatrixConvertToParCSRMatrix(A);
         n_global = hypre_ParCSRMatrixGlobalNumRows(A_ParCSR);

         /*  Generate CSR matrix from ParCSRMatrix A */

#ifdef HYPRE_NO_GLOBAL_PARTITION
         /* all processors are needed for these routines */
         A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A_ParCSR);
         f_vector = hypre_ParVectorToVectorAll(f);
	 if (n)
	 {
	 
#else
	 if (n)
	 {
	    A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A_ParCSR);
	    f_vector = hypre_ParVectorToVectorAll(f);
#endif
 	    A_CSR_i = hypre_CSRMatrixI(A_CSR);
 	    A_CSR_j = hypre_CSRMatrixJ(A_CSR);
 	    A_CSR_data = hypre_CSRMatrixData(A_CSR);
   	    f_vector_data = hypre_VectorData(f_vector);

            A_mat = hypre_CTAlloc(double, n_global*n_global);
            b_vec = hypre_CTAlloc(double, n_global);    

            /*  Load CSR matrix into A_mat. */


            for (i = 0; i < n_global; i++)
            {
               for (jj = A_CSR_i[i]; jj < A_CSR_i[i+1]; jj++)
               {
                  column = A_CSR_j[jj];
                  A_mat[i*n_global+column] = A_CSR_data[jj];
               }
               b_vec[i] = f_vector_data[i];
            }

            relax_error = gselim(A_mat,b_vec,n_global); 

            for (i = 0; i < n; i++)
            {
               for (k=0; k < block_size; k++)
               {
                  u_data[i*block_size + k] = b_vec[first_index+i*block_size + k];
               }
            }

	    hypre_TFree(A_mat); 
            hypre_TFree(b_vec);
            hypre_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         
         }
#ifdef HYPRE_NO_GLOBAL_PARTITION
         else 
         {
            hypre_CSRMatrixDestroy(A_CSR);
            A_CSR = NULL;
            hypre_SeqVectorDestroy(f_vector);
            f_vector = NULL;
         }
#endif

         hypre_ParCSRMatrixDestroy(A_ParCSR);
         A_ParCSR = NULL;

         break;
      }
      
   }
   

   hypre_TFree(res_vec);
   hypre_TFree(out_vec);
   hypre_TFree(tmp_vec);

   return (relax_error);
   
}
