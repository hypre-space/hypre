/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <stdio.h>

#include "mli_solver_mls.h"
#include "../base/mli_defs.h"
#include "parcsr_mv/parcsr_mv.h"

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_MLS::MLI_Solver_MLS() : MLI_Solver(MLI_SOLVER_MLS_ID)
{
   Amat      = NULL;
   max_eigen = -1.0;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_MLS::~MLI_Solver_MLS()
{
   Amat = NULL;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_MLS::setup(MLI_Matrix *mat)
{
   Amat = mat;
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_MLS::solve(MLI_Vector *f_in, MLI_Vector *u_in)
{
   hypre_ParCSRMatrix  *A;
   hypre_CSRMatrix     *A_diag;
   hypre_Vector        *u_local, *Vtemp_local;
   hypre_ParVector     *Vtemp, *Wtemp, *f, *u;
   int                 i, n, global_size, *partitioning1, *partitioning2;
   int                 *A_diag_i, num_procs;
   double              *A_diag_data, omega, omega2, *u_data, *Vtemp_data;
   double              mls_over=1.1, mls_boost=1.019, alpha;
   MPI_Comm            comm;

   /*-----------------------------------------------------------------
    * check that proper spectral radius is passed in
    *-----------------------------------------------------------------*/

   if ( max_eigen <= 0.0 )
   {
      cout << "Solver_MLS::solver ERROR : max_eig <= 0.\n"; 
      exit(1);
   }

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
   partitioning2 = hypre_CTAlloc( int, num_procs+1 );
   for ( i = 0; i <= num_procs; i++ ) partitioning2[i] = partitioning1[i];
   Wtemp = hypre_ParVectorCreate(comm, global_size, partitioning2);
   hypre_ParVectorInitialize(Wtemp);

   /*-----------------------------------------------------------------
    * Perform MLS iterations
    *-----------------------------------------------------------------*/
 
   omega  = 2.0 / (max_eigen * 1.5 * mls_over);
   omega2 = (1 - omega * max_eigen * mls_over);
   omega2 = omega2 * omega2 * max_eigen * mls_over;

   /* u = u + omega * (f - A u) */

   hypre_ParVectorCopy(f,Vtemp); 
   hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);
   alpha = omega * mls_over;

#define HYPRE_SMP_PRIVATE i
#include "utilities/hypre_smp_forloop.h"
   for (i = 0; i < n; i++)
   {
      if (A_diag_data[A_diag_i[i]] != 0.0) u_data[i] += (alpha*Vtemp_data[i]);
   }

   /* compute residual Vtemp = f - A u */

   hypre_ParVectorCopy(f,Vtemp); 
   hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);

   /* compute residual Wtemp = (I - omega * A) Vtemp */

   alpha = omega;
   hypre_ParVectorCopy(Vtemp,Wtemp); 
   hypre_ParCSRMatrixMatvec(-alpha, A, Vtemp, 1.0, Wtemp);

   /* compute residual Vtemp = (I - omega * A) Wtemp */

   hypre_ParVectorCopy(Wtemp,Vtemp); 
   hypre_ParCSRMatrixMatvec(-omega, A, Wtemp, 1.0, Vtemp);

   /* compute u = u + alpha * Vtemp */

   alpha = 2.0 / (omega2 * mls_boost) * mls_over;

#define HYPRE_SMP_PRIVATE i
#include "utilities/hypre_smp_forloop.h"
   for (i = 0; i < n; i++)
   {
      if (A_diag_data[A_diag_i[i]] != 0.0) u_data[i] += (alpha*Vtemp_data[i]);
   }

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   hypre_ParVectorDestroy( Vtemp ); 
   hypre_ParVectorDestroy( Wtemp ); 

   return(0); 
}

/******************************************************************************
 * set MLS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_MLS::setParams( char *param_string, int argc, char **argv )
{
   int    nsweeps;
   double *weights;

   if ( !strcmp(param_string, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         cout << "Solver_MLS::setParams ERROR : needs 1 or 2 args.\n";
         return 1;
      }
      if ( argc >= 1 ) nsweeps = *(int*)   argv[0];
      if ( argc == 2 ) weights = (double*) argv[1];
      max_eigen = weights[0];
      if ( max_eigen < 0.0 ) 
      {
         cout << "Solver_MLS::setParams ERROR : max_eig <= 0 (" 
              << max_eigen << ")\n";
         return 1;
      }
   }
   return 0;
}

/******************************************************************************
 * set MLS parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_MLS::setParams( double eigen_in )
{
   if ( max_eigen <= 0.0 )
   {
      cerr << "Solver_MLS::setParams WARNING : max_eigen <= 0." << endl;
      return 1; 
   }
   max_eigen = eigen_in;
   return 0;
}

