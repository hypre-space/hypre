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

#include "mli_mls.h"
#include "parcsr_mv/parcsr_mv.h"

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_SolverMLS::MLI_SolverMLS()
{
   Amat      = NULL;
   max_eigen = -1.0;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_SolverMLS::~MLI_SolverMLS()
{
   Amat = NULL;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_SolverMLS::setup(MLI_Matrix *mat)
{
   Amat = mat;
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_SolverMLS::solve(MLI_Vector *f_in, MLI_Vector *u_in)
{
   hypre_ParCSRMatrix  *A;
   hypre_CSRMatrix     *A_diag;
   double              *A_diag_data;
   int                 *A_diag_i;
   hypre_Vector        *u_local, *Vtemp_local;
   double              *u_data, *Vtemp_data;
   hypre_ParVector     *Vtemp;
   int                 i, n, relax_error = 0, global_size, *partitioning1;
   int                 is, num_procs, num_threads, *partitioning2;
   double              zero = 0.0, omega, omega2;
   MPI_Comm            comm;
   hypre_ParVector     *f, *u;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   num_threads     = hypre_NumThreads();
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
    * Perform MLS iterations
    *-----------------------------------------------------------------*/
 
   hypre_ParVectorCopy(u, Vtemp); 
   hypre_ParCSRMatrixMatvec(1.0, A, Vtemp, 0.0, u);
   omega  = 2.0 / max_eigen;
   omega2 = max_eigen;
   for( is = 0; is < 2; is++ )
   {
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
            u_data[i] += ( omega * Vtemp_data[i] ); 
         }
      }
      omega2 = ( 1.0 - omega * max_eigen ) * omega2;
   }
   hypre_ParVectorCopy(f,Vtemp); 
   hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);
   omega2 = 2.0 / omega2;
#define HYPRE_SMP_PRIVATE i
#include "utilities/hypre_smp_forloop.h"
   for (i = 0; i < n; i++)
   {
      if (A_diag_data[A_diag_i[i]] != zero)
         u_data[i] += ( omega2 * Vtemp_data[i] ); 
   }

   /*-----------------------------------------------------------------
    * clean up and return
    *-----------------------------------------------------------------*/

   hypre_ParVectorDestroy( Vtemp ); 

   return(relax_error); 
}

/******************************************************************************
 * set MLS parameters
 *---------------------------------------------------------------------------*/

int MLI_SolverMLS::setParams( char *param_string, int argc, char **argv )
{
   int    nsweeps;
   double *weights;

   if ( !strcmp(param_string, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         cout << "SolverMLS::setParams ERROR : needs 1 or 2 args.\n";
         return 1;
      }
      if ( argc >= 1 ) nsweeps = *(int*)   argv[0];
      if ( argc == 2 ) weights = (double*) argv[1];
printf("MLS : eigen = %e\n", weights[0]);
      max_eigen = weights[0];
      if ( max_eigen < 0.0 ) 
      {
         cout << "SolverMLS::setParams ERROR : max_eig <= 0 (" 
              << max_eigen << ")\n";
         return 1;
      }
   }
   return 0;
}

/******************************************************************************
 * set MLS parameters
 *---------------------------------------------------------------------------*/

int MLI_SolverMLS::setParams( double eigen_in )
{
   if ( max_eigen <= 0.0 )
   {
      cerr << "SolverMLS::setParams WARNING : max_eigen <= 0." << endl;
      return 1; 
   }
   max_eigen = eigen_in;
   return 0;
}

