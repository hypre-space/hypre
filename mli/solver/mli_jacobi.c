/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <stdio.h>
#include "parcsr_mv.h"
#include "mli_matrix.h"
#include "mli_vector.h"
#include "mli_smoother.h"

extern int  MLI_Smoother_Apply_DampedJacobi(void *smoother, MLI_Vector *f,
                                            MLI_Vector *u);

/******************************************************************************
 * Damped Jacobi relaxation scheme
 *****************************************************************************/

typedef struct MLI_Smoother_Jacobi_Struct
{
   MLI_Matrix  *Amat;
   int         nsweeps;
   double      *relax_weights;
} MLI_Smoother_Jacobi;

/*--------------------------------------------------------------------------
 * MLI_Smoother_Destroy_DampedJacobi
 *--------------------------------------------------------------------------*/

void MLI_Smoother_Destroy_DampedJacobi(void *smoother_obj)
{
   MLI_Smoother_Jacobi *jacobi_smoother = (MLI_Smoother_Jacobi *) smoother_obj;
   if ( jacobi_smoother != NULL ) 
   {
      if ( jacobi_smoother->Amat != NULL ) 
         MLI_Matrix_Destroy( jacobi_smoother->Amat );
      if ( jacobi_smoother->relax_weights != NULL ) 
         hypre_TFree( jacobi_smoother->relax_weights );
      hypre_TFree( jacobi_smoother );
   }
   return;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Setup_DampedJacobi
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Setup_DampedJacobi(void *smoother_obj, MLI_Matrix *Amat, 
                                    int ntimes, double *relax_weights)
{
   int                 i;
   MLI_Smoother_Jacobi *jacobi_smoother;
   MLI_Smoother        *generic_smoother = (MLI_Smoother *) smoother_obj;
   hypre_ParCSRMatrix  *A;

   jacobi_smoother = hypre_CTAlloc(MLI_Smoother_Jacobi, 1); 
   if ( jacobi_smoother == NULL ) { return 1; }
   if ( ntimes > 0 ) jacobi_smoother->nsweeps = ntimes;
   else              jacobi_smoother->nsweeps = 1;
   if ( relax_weights != NULL )
   {
      jacobi_smoother->relax_weights = 
                       hypre_CTAlloc(double, jacobi_smoother->nsweeps); 
      for ( i = 0; i < jacobi_smoother->nsweeps; i++ )
      {
         if ( relax_weights[i] < 0 || relax_weights[i] > 2. )
            printf("MLI_Smoother_Setup_DampedJacobi : weight %d = %e ?\n", i,
                   relax_weights[i]);
         jacobi_smoother->relax_weights[i] = relax_weights[i];
      }
   }
   else jacobi_smoother->relax_weights = NULL;
   jacobi_smoother->Amat = Amat;
   generic_smoother->object = (void *) jacobi_smoother;
   generic_smoother->destroy_func = MLI_Smoother_Destroy_DampedJacobi;
   generic_smoother->apply_func = MLI_Smoother_Apply_DampedJacobi;
   return 0;
}

/*--------------------------------------------------------------------------
 * MLI_Smoother_Apply_DampedJacobi
 *--------------------------------------------------------------------------*/

int MLI_Smoother_Apply_DampedJacobi(void *smoother_obj,
                                    MLI_Vector *f_in, MLI_Vector *u_in)
{
   hypre_ParCSRMatrix  *A;
   hypre_CSRMatrix     *A_diag;
   double              *A_diag_data;
   int                 *A_diag_i;
   hypre_Vector        *u_local, *Vtemp_local;
   double              *u_data, *Vtemp_data;
   hypre_ParVector     *Vtemp;
   int                 i, n, relax_error = 0, global_size, *partitioning1;
   int                 is, num_procs, num_threads, *partitioning2, nsweeps;
   double              zero = 0.0, *relax_weights, relax_weight;
   MPI_Comm            comm;
   MLI_Smoother_Jacobi *jacobi_smoother;
   hypre_ParVector     *f, *u;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   MPI_Comm_size(comm,&num_procs);  
   num_threads     = hypre_NumThreads();
   jacobi_smoother = (MLI_Smoother_Jacobi *) smoother_obj;
   relax_weights   = jacobi_smoother->relax_weights;
   nsweeps         = jacobi_smoother->nsweeps;
   A               = jacobi_smoother->Amat->matrix;
   comm            = hypre_ParCSRMatrixComm(A);
   A_diag          = hypre_ParCSRMatrixDiag(A);
   A_diag_data     = hypre_CSRMatrixData(A_diag);
   A_diag_i        = hypre_CSRMatrixI(A_diag);
   n               = hypre_CSRMatrixNumRows(A_diag);
   f               = f_in->vector;
   u               = u_in->vector;
   u_local         = hypre_ParVectorLocalVector(u);
   u_data          = hypre_VectorData(u_local);
   
   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   global_size   = hypre_ParVectorGlobalSize(f);
   partitioning1 = hypre_ParVectorPartitioning(f);
   partitioning2 = hypre_CTAlloc( int, num_procs+1 );
   for ( i = 0; i <= num_procs; i++ ) partitioning2[i] = partitioning1[i];
   Vtemp = hypre_ParVectorCreate(comm, global_size, partitioning2);
   Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   Vtemp_data  = hypre_VectorData(Vtemp_local);

   /*-----------------------------------------------------------------
    * Copy f into temporary vector.
    *-----------------------------------------------------------------*/
        
   hypre_ParVectorCopy(f,Vtemp); 
 
   /*-----------------------------------------------------------------
    * Perform Matvec Vtemp=f-Au
    *-----------------------------------------------------------------*/
 
   for( is = 0; is < nsweeps; is++ )
   {
      if ( relax_weights != NULL ) relax_weight = relax_weights[i];
      else                         relax_weight = 1.0;

      hypre_ParCSRMatrixMatvec(-1.0,A, u, 1.0, Vtemp);

#define HYPRE_SMP_PRIVATE i
#include "../utilities/hypre_smp_forloop.h"
      for (i = 0; i < n; i++)
      {
         /*-----------------------------------------------------------
          * If diagonal is nonzero, relax point i; otherwise, skip it.
          *-----------------------------------------------------------*/
           
         if (A_diag_data[A_diag_i[i]] != zero)
         {
            u_data[i] += relax_weight * Vtemp_data[i] 
                                   / A_diag_data[A_diag_i[i]];
         }
      }
   }

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   hypre_ParVectorDestroy( Vtemp ); 

   return(relax_error); 
}

