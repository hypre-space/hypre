/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "parcsr_mv/parcsr_mv.h"
#include "base/mli_defs.h"
#include "solver/mli_solver_jacobi.h"

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_Jacobi::MLI_Solver_Jacobi() : MLI_Solver(MLI_SOLVER_JACOBI_ID)
{
   Amat             = NULL;
   nsweeps          = 1;
   relax_weights    = new double[1];
   relax_weights[0] = 0.5;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_Jacobi::~MLI_Solver_Jacobi()
{
   if ( relax_weights != NULL ) delete [] relax_weights;
   relax_weights = NULL;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_Jacobi::setup(MLI_Matrix *mat)
{
   Amat = mat;
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_Jacobi::solve(MLI_Vector *f_in, MLI_Vector *u_in)
{
   hypre_ParCSRMatrix  *A;
   hypre_CSRMatrix     *A_diag;
   double              *A_diag_data;
   int                 *A_diag_i;
   hypre_Vector        *u_local, *Vtemp_local;
   double              *u_data, *Vtemp_data;
   hypre_ParVector     *Vtemp;
   int                 i, n, relax_error = 0, global_size, *partitioning1;
   int                 is, num_procs, *partitioning2;
   double              zero = 0.0, relax_weight;
   MPI_Comm            comm;
   hypre_ParVector     *f, *u;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A               = (hypre_ParCSRMatrix *) Amat->getMatrix();
   comm            = hypre_ParCSRMatrixComm(A);
   A_diag          = hypre_ParCSRMatrixDiag(A);
   A_diag_data     = hypre_CSRMatrixData(A_diag);
   A_diag_i        = hypre_CSRMatrixI(A_diag);
   n               = hypre_CSRMatrixNumRows(A_diag);
   f               = (hypre_ParVector *) f_in->getVector();
   u               = (hypre_ParVector *) u_in->getVector();
   u_local         = hypre_ParVectorLocalVector(u);
   u_data          = hypre_VectorData(u_local);
   MPI_Comm_size(comm,&num_procs);  
   
   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   global_size   = hypre_ParVectorGlobalSize(f);
   partitioning1 = hypre_ParVectorPartitioning(f);
   partitioning2 = hypre_CTAlloc( int, num_procs+1 );
   for ( i = 0; i <= num_procs; i++ ) partitioning2[i] = partitioning1[i];
   Vtemp = hypre_ParVectorCreate(comm, global_size, partitioning2);
   hypre_ParVectorInitialize(Vtemp);
   Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   Vtemp_data  = hypre_VectorData(Vtemp_local);

   /*-----------------------------------------------------------------
    * Perform Jacobi iterations
    *-----------------------------------------------------------------*/
 
   for( is = 0; is < nsweeps; is++ )
   {
      if ( relax_weights != NULL ) relax_weight = relax_weights[is];
      else                         relax_weight = 1.0;

      hypre_ParVectorCopy(f,Vtemp); 
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);

#define HYPRE_SMP_PRIVATE i
#include "utilities/hypre_smp_forloop.h"
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

/******************************************************************************
 * set Jacobi parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_Jacobi::setParams( char *param_string, int argc, char **argv )
{
   int    i;
   double *weights;
   char   param1[200];

   if ( !strcasecmp(param_string, "numSweeps") )
   {
      sscanf(param_string, "%s %d", param1, &nsweeps);
      if ( nsweeps < 1 ) nsweeps = 1;
      
      return 0;
   }
   else if ( !strcasecmp(param_string, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         printf("MLI_Solver_Jacobi::setParams ERROR : needs 1 or 2 args.\n");
         return 1;
      }
      if ( argc >= 1 ) nsweeps = *(int*)   argv[0];
      if ( argc == 2 ) weights = (double*) argv[1];
      if ( nsweeps < 1 ) nsweeps = 1;
      if ( relax_weights != NULL ) delete [] relax_weights;
      relax_weights = NULL;
      if ( weights != NULL )
      {
         relax_weights = new double[nsweeps];
         for ( i = 0; i < nsweeps; i++ ) relax_weights[i] = weights[i];
      }
   }
   else if ( strcasecmp(param_string, "zeroInitialGuess") )
   {   
      printf("MLI_Solver_Jacobi::setParams - parameter not recognized.\n");
      printf("                Params = %s\n", param_string);
      return 1;
   }
   return 0;
}

/******************************************************************************
 * set Jacobi parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_Jacobi::setParams( int ntimes, double *weights )
{
   if ( ntimes <= 0 )
   {
      printf("MLI_Solver_Jacobi::setParams WARNING : nsweeps set to 1.\n");
      ntimes = 1;
   }
   nsweeps = ntimes;
   if ( relax_weights != NULL ) delete [] relax_weights;
   relax_weights = new double[ntimes];
   if ( weights == NULL )
   {
      printf("MLI_Solver_Jacobi::setParams - relax_weights set to 0.5.\n");
      for ( int i = 0; i < ntimes; i++ ) relax_weights[i] = 0.5;
   }
   else
   {
      for ( int j = 0; j < ntimes; j++ ) 
      {
         if (weights[j] >= 0. && weights[j] <= 2.) 
            relax_weights[j] = weights[j];
         else 
         {
            printf("MLI_Solver_Jacobi::setParams - weights set to 0.5.\n");
            relax_weights[j] = 0.5;
         }
      }
   }
   return 0;
}

