/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <string.h>
#include <iostream.h>

#include "parcsr_mv/parcsr_mv.h"
#include "base/mli_defs.h"
#include "solver/mli_solver_parasails.h"

/******************************************************************************
 * ParaSails relaxation scheme
 *****************************************************************************/

/******************************************************************************
 * constructor
 *--------------------------------------------------------------------------*/

MLI_Solver_ParaSails::MLI_Solver_ParaSails() 
                     : MLI_Solver(MLI_SOLVER_PARASAILS_ID)
{
#ifdef MLI_PARASAILS
   Amat       = NULL;
   ps         = NULL;
   nlevels    = 0;        /* number of levels */
   symmetric  = 0;        /* nonsymmetric */
   transpose  = 0;        /* non-transpose */
   threshold  = 0.0;
   filter     = 0.0;
   loadbal    = 0;        /* no load balance */
   factorized = 0;        /* not factorized */
#else
   cout << "ParaSails smoother not available.\n";
   exit(1);
#endif
}

/******************************************************************************
 * destructor
 *--------------------------------------------------------------------------*/

MLI_Solver_ParaSails::~MLI_Solver_ParaSails()
{
#ifdef MLI_PARASAILS
   if ( ps != NULL ) ParaSailsDestroy(ps);
   ps = NULL;
#endif
}

/******************************************************************************
 * Setup
 *--------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setup(MLI_Matrix *Amat_in)
{
#ifdef MLI_PARASAILS
   hypre_ParCSRMatrix *A;
   int                *partition, mypid, start_row, end_row;
   int                row, row_length, *col_indices;
   double             *col_values;
   Matrix             *mat;
   MPI_Comm           comm;

   /*-----------------------------------------------------------------
    * fetch machine and matrix parameters
    *-----------------------------------------------------------------*/

   Amat = Amat_in;
   A = (hypre_ParCSRMatrix *) Amat->getMatrix();
   comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm_rank(comm,&mypid);  
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) A, &partition);
   start_row = partition[mypid];
   end_row   = partition[mypid+1] - 1;

   /*-----------------------------------------------------------------
    * construct a ParaSails matrix
    *-----------------------------------------------------------------*/

   mat = MatrixCreate(comm, start_row, end_row);
   for (row = start_row; row <= end_row; row++)
   {
      hypre_ParCSRMatrixGetRow(A, row, &row_length, &col_indices, &col_values);
      MatrixSetRow(mat, row, row_length, col_indices, col_values);
      hypre_ParCSRMatrixRestoreRow(A,row,&row_length,&col_indices,&col_values);
   }
   MatrixComplete(mat);

   /*-----------------------------------------------------------------
    * construct a ParaSails smoother object
    *-----------------------------------------------------------------*/

   ps = ParaSailsCreate(comm, start_row, end_row, factorized);
   ps->loadbal_beta = loadbal;
   ParaSailsSetupPattern(ps, mat, threshold, nlevels);
   ParaSailsStatsPattern(ps, mat);
   ParaSailsSetupValues(ps, mat, filter);
   ParaSailsStatsValues(ps, mat);

   /*-----------------------------------------------------------------
    * clean up and return object and function
    *-----------------------------------------------------------------*/

   MatrixDestroy(mat);
   return 0;
#else
   cout << "ParaSails smoother not available.\n";
   exit(1);
   return 1;
#endif
}

/******************************************************************************
 * solve
 *--------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::solve(MLI_Vector *f_in, MLI_Vector *u_in)
{
#ifdef MLI_PARASAILS
   if (transpose) return (applyParaSailsTrans( f_in, u_in ));
   else           return (applyParaSails( f_in, u_in ));
#else
   cout << "ParaSails smoother not available.\n";
   exit(1);
   return 1;
#endif
}

/******************************************************************************
 * set parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setParams(char *param_string, int argc, char **argv)
{
   char param1[100];

   if ( !strcmp(param_string, "nLevels") )
   {
      sscanf(param_string, "%s %d", param1, &nlevels);
      if ( nlevels < 0 ) nlevels = 0;
   }
   else if ( !strcmp(param_string, "symmetric") )   symmetric  = 1;
   else if ( !strcmp(param_string, "unsymmetric") ) symmetric  = 0;
   else if ( !strcmp(param_string, "factorized") )  factorized = 1;
   else if ( !strcmp(param_string, "transpose") )   transpose  = 1;
   else if ( !strcmp(param_string, "loadbal") )     loadbal    = 1;
   else if ( !strcmp(param_string, "threshold") )
   {
      sscanf(param_string, "%s %lg", param1, &threshold);
      if ( threshold < 0 || threshold > 1. ) threshold = 0.;
   }
   else if ( !strcmp(param_string, "filter") )
   {
      sscanf(param_string, "%s %lg", param1, &filter);
      if ( filter < 0 || filter > 1. ) filter = 0.;
   }
   else
   {   
      cout << "MLI_Solver_ParaSails::setParams - parameter not recognized.\n";
      cout << "              Params = " << param_string << endl;
      return 1;
   }
   return 0;
}

/******************************************************************************
 * apply as it is
 *--------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::applyParaSails(MLI_Vector *f_in, MLI_Vector *u_in)
{
#ifdef MLI_PARASAILS
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *A_diag;
   hypre_ParVector    *Vtemp;
   hypre_Vector       *u_local, *Vtemp_local;
   double             *u_data, *Vtemp_data;
   int                i, n, relax_error = 0, global_size;
   int                num_procs, *partition1, *partition2;
   double             *tmp_data;
   MPI_Comm           comm;
   hypre_ParVector    *u, *f;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A       = (hypre_ParCSRMatrix *) Amat->getMatrix();
   comm    = hypre_ParCSRMatrixComm(A);
   A_diag  = hypre_ParCSRMatrixDiag(A);
   n       = hypre_CSRMatrixNumRows(A_diag);
   u       = (hypre_ParVector *) u_in->getVector();
   u_local = hypre_ParVectorLocalVector(u);
   u_data  = hypre_VectorData(u_local);
   MPI_Comm_size(comm,&num_procs);  

   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   f           = (hypre_ParVector *) f_in->getVector();
   global_size = hypre_ParVectorGlobalSize(f);
   partition1  = hypre_ParVectorPartitioning(f);
   partition2  = hypre_CTAlloc( int, num_procs+1 );
   for ( i = 0; i <= num_procs; i++ ) partition2[i] = partition1[i];
   Vtemp = hypre_ParVectorCreate(comm, global_size, partition2);
   hypre_ParVectorInitialize(Vtemp);
   Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   Vtemp_data  = hypre_VectorData(Vtemp_local);

   /*-----------------------------------------------------------------
    * perform smoothing
    *-----------------------------------------------------------------*/

   hypre_ParVectorCopy(f, Vtemp);
   hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);
   tmp_data = new double[n];

   if (!factorized)
   {
      MatrixMatvec(ps->M, Vtemp_data, tmp_data);
      for (i = 0; i < n; i++) u_data[i] += tmp_data[i];
   }
   else
   {
      MatrixMatvec(ps->M, Vtemp_data, tmp_data);
      MatrixMatvecTrans(ps->M, tmp_data, tmp_data);
      for (i = 0; i < n; i++) u_data[i] += tmp_data[i];
   }

   /*-----------------------------------------------------------------
    * clean up 
    *-----------------------------------------------------------------*/

   delete [] tmp_data;

   return(relax_error); 
#else
   cout << "ParaSails smoother not available.\n";
   exit(1);
   return 1;
#endif
}

/******************************************************************************
 * apply its transpose 
 *--------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::applyParaSailsTrans(MLI_Vector *f_in, 
                                              MLI_Vector *u_in)
{
#ifdef MLI_PARASAILS
   hypre_ParCSRMatrix *A;
   hypre_CSRMatrix    *A_diag;
   hypre_ParVector    *Vtemp;
   hypre_Vector       *u_local, *Vtemp_local;
   double             *u_data, *Vtemp_data;
   int                i, n, relax_error = 0, global_size;
   int                num_procs, *partition1, *partition2;
   double             *tmp_data;
   MPI_Comm           comm;
   hypre_ParVector    *u, *f;

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A       = (hypre_ParCSRMatrix *) Amat->getMatrix();
   comm    = hypre_ParCSRMatrixComm(A);
   A_diag  = hypre_ParCSRMatrixDiag(A);
   n       = hypre_CSRMatrixNumRows(A_diag);
   u       = (hypre_ParVector *) u_in->getVector();
   u_local = hypre_ParVectorLocalVector(u);
   u_data  = hypre_VectorData(u_local);
   MPI_Comm_size(comm,&num_procs);  

   /*-----------------------------------------------------------------
    * create temporary vector
    *-----------------------------------------------------------------*/

   f           = (hypre_ParVector *) f_in->getVector();
   global_size = hypre_ParVectorGlobalSize(f);
   partition1  = hypre_ParVectorPartitioning(f);
   partition2  = hypre_CTAlloc( int, num_procs+1 );
   for ( i = 0; i <= num_procs; i++ ) partition2[i] = partition1[i];
   Vtemp = hypre_ParVectorCreate(comm, global_size, partition2);
   Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   Vtemp_data  = hypre_VectorData(Vtemp_local);

   /*-----------------------------------------------------------------
    * perform smoothing
    *-----------------------------------------------------------------*/

   hypre_ParVectorCopy(f, Vtemp);
   hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);
   tmp_data = new double[n];

   if (!factorized)
   {
      MatrixMatvecTrans(ps->M, Vtemp_data, tmp_data);
      for (i = 0; i < n; i++) u_data[i] += tmp_data[i];
   }
   else
   {
      MatrixMatvec(ps->M, Vtemp_data, tmp_data);
      MatrixMatvecTrans(ps->M, tmp_data, tmp_data);
      for (i = 0; i < n; i++) u_data[i] += tmp_data[i];
   }

   /*-----------------------------------------------------------------
    * clean up 
    *-----------------------------------------------------------------*/

   delete [] tmp_data;

   return(relax_error); 
#else
   cout << "ParaSails smoother not available.\n";
   exit(1);
   return 1;
#endif
}

/******************************************************************************
 * set ParaSails number of levels parameter
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setNumLevels( int levels )
{
   if ( levels < 0 )
   {
      cerr << "MLI_Solver_ParaSails::setNumLevels WARNING : nlevels = 0.\n";
      nlevels = 0;
   }
   else nlevels = levels;
   return 0;
}

/******************************************************************************
 * set ParaSails symmetry parameter
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setSymmetric()
{
   symmetric = 1;
   return 0;
}

/*---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setUnSymmetric()
{
   symmetric = 0;
   return 0;
}

/******************************************************************************
 * set ParaSails threshold
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setThreshold( double thresh )
{
   if ( thresh < 0 || thresh > 1. )
   {
      cerr << "MLI_Solver_ParaSails::setThreshold WARNING : thresh = 0.\n";
      threshold = 0.;
   }
   else threshold = thresh;
   return 0;
}

/******************************************************************************
 * set ParaSails filter
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setFilter( double data )
{
   if ( data < 0 || data > 1. )
   {
      cerr << "MLI_Solver_ParaSails::setThreshold WARNING : filter = 0.\n";
      filter = 0.;
   }
   else filter = data;
   return 0;
}

/******************************************************************************
 * set ParaSails loadbal parameter
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setLoadBal()
{
   loadbal = 1;
   return 0;
}

/******************************************************************************
 * set ParaSails factorized parameter
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setFactorized()
{
   factorized = 1;
   return 0;
}

/******************************************************************************
 * set ParaSails tranpose parameter
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setTranspose()
{
   transpose = 1;
   return 0;
}

