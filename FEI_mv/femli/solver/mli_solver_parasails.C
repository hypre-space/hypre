/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <string.h>
#include <strings.h>

#include "parcsr_mv/parcsr_mv.h"
#include "base/mli_defs.h"
#include "solver/mli_solver_parasails.h"

/******************************************************************************
 * ParaSails relaxation scheme
 *****************************************************************************/

/******************************************************************************
 * constructor
 *--------------------------------------------------------------------------*/

MLI_Solver_ParaSails::MLI_Solver_ParaSails(char *name) : MLI_Solver(name)
{
#ifdef MLI_PARASAILS
   Amat_            = NULL;
   ps_              = NULL;
   nlevels_         = 1;     /* number of levels */
   symmetric_       = 1;     /* symmetric */
   transpose_       = 0;     /* non-transpose */
   correction_      = 0.8;   /* 0.8 confirmed a good value for 4-8 proc */
   threshold_       = 1.0e-4;
   filter_          = 1.0e-4;
   loadbal_         = 0;        /* no load balance */
   zeroInitialGuess_ = 0;
#else
   printf("MLI_Solver_ParaSails::constructor - ParaSails smoother ");
   printf("not available.\n");
   exit(1);
#endif
}

/******************************************************************************
 * destructor
 *--------------------------------------------------------------------------*/

MLI_Solver_ParaSails::~MLI_Solver_ParaSails()
{
#ifdef MLI_PARASAILS
   if ( ps_ != NULL ) ParaSailsDestroy(ps_);
   ps_ = NULL;
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

   Amat_ = Amat_in;
   A = (hypre_ParCSRMatrix *) Amat_->getMatrix();
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

   ps_ = ParaSailsCreate(comm, start_row, end_row, symmetric_);
   ps_->loadbal_beta = loadbal_;
   ParaSailsSetupPattern(ps_, mat, threshold_, nlevels_);
   ParaSailsStatsPattern(ps_, mat);
   ParaSailsSetupValues(ps_, mat, filter_);
   ParaSailsStatsValues(ps_, mat);

   /*-----------------------------------------------------------------
    * clean up and return object and function
    *-----------------------------------------------------------------*/

   MatrixDestroy(mat);
   return 0;
#else
   (void) Amat_in;
   printf("MLI_Solver_ParaSails::setup ERROR - ParaSails smoother ");
   printf("not available.\n");
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
   if (transpose_) return (applyParaSailsTrans( f_in, u_in ));
   else            return (applyParaSails( f_in, u_in ));
#else
   (void) f_in;
   (void) u_in;
   printf("MLI_Solver_ParaSails::solve ERROR - ParaSails smoother \n");
   printf("not available.\n");
   exit(1);
   return 1;
#endif
}

/******************************************************************************
 * set parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setParams(char *paramString, int argc, char **argv)
{
   char param1[100];

   sscanf(paramString, "%s", param1);
   if ( !strcmp(param1, "nLevels") )
   {
      sscanf(paramString, "%s %d", param1, &nlevels_);
      if ( nlevels_ < 0 ) nlevels_ = 0;
   }
   else if ( !strcmp(param1, "symmetric") )   symmetric_ = 1;
   else if ( !strcmp(param1, "unsymmetric") ) symmetric_ = 0;
   else if ( !strcmp(param1, "transpose") )   transpose_ = 1;
   else if ( !strcmp(param1, "loadbal") )     loadbal_   = 1;
   else if ( !strcmp(param1, "threshold") )
   {
      sscanf(paramString, "%s %lg", param1, &threshold_);
      if ( threshold_< 0 || threshold_> 1. ) threshold_= 0.;
   }
   else if ( !strcmp(param1, "filter") )
   {
      sscanf(paramString, "%s %lg", param1, &filter_);
      if ( filter_ < 0 || filter_ > 1. ) filter_= 0.;
   }
   else if ( !strcmp(param1, "correction") )
   {
      sscanf(paramString, "%s %lg", param1, &correction_);
      if ( correction_<= 0 ) correction_= 0.5;
   }
   else if ( !strcmp(param1, "zeroInitialGuess") )
   {
      zeroInitialGuess_ = 1;
   }
   else if ( strcmp(param1, "relaxWeight") )
   {   
      printf("MLI_Solver_ParaSails::setParams - parameter not recognized.\n");
      printf("              Params = %s\n", paramString);
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

   A       = (hypre_ParCSRMatrix *) Amat_->getMatrix();
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

   tmp_data = new double[n];
   hypre_ParVectorCopy(f, Vtemp);
   if ( zeroInitialGuess_ == 0 )
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);

   ParaSailsApply(ps_, Vtemp_data, tmp_data);

   if ( zeroInitialGuess_ == 0 )
      for (i = 0; i < n; i++) u_data[i] += correction_ * tmp_data[i];
   else
      for (i = 0; i < n; i++) u_data[i] = correction_ * tmp_data[i];

   zeroInitialGuess_ = 0;

   /*-----------------------------------------------------------------
    * clean up 
    *-----------------------------------------------------------------*/

   delete [] tmp_data;

   return(relax_error); 
#else
   (void) f_in;
   (void) u_in;
   printf("MLI_Solver_ParaSails::applyParaSails ERROR - ParaSails not");
   printf(" available.\n");
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

   A       = (hypre_ParCSRMatrix *) Amat_->getMatrix();
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

   tmp_data = new double[n];
   hypre_ParVectorCopy(f, Vtemp);
   if ( zeroInitialGuess_ == 0 )
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Vtemp);

   ParaSailsApplyTrans(ps_, Vtemp_data, tmp_data);

   if ( zeroInitialGuess_ == 0 )
      for (i = 0; i < n; i++) u_data[i] += correction_ * tmp_data[i];
   else
      for (i = 0; i < n; i++) u_data[i] = correction_ * tmp_data[i];

   zeroInitialGuess_ = 0;

   /*-----------------------------------------------------------------
    * clean up 
    *-----------------------------------------------------------------*/

   delete [] tmp_data;

   return(relax_error); 
#else
   (void) f_in;
   (void) u_in;
   printf("MLI_Solver_ParaSails::applyParaSailsTrans ERROR - ParaSails");
   printf(" not available.\n");
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
      printf("MLI_Solver_ParaSails::setNumLevels WARNING : nlevels = 0.\n");
      nlevels_ = 0;
   }
   else nlevels_ = levels;
   return 0;
}

/******************************************************************************
 * set ParaSails symmetry parameter
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setSymmetric()
{
   symmetric_ = 1;
   return 0;
}

/*---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setUnSymmetric()
{
   symmetric_ = 0;
   return 0;
}

/******************************************************************************
 * set ParaSails threshold
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setThreshold( double thresh )
{
   if ( thresh < 0 || thresh > 1. )
   {
      printf("MLI_Solver_ParaSails::setThreshold WARNING - thresh = 0.\n");
      threshold_ = 0.;
   }
   else threshold_ = thresh;
   return 0;
}

/******************************************************************************
 * set ParaSails filter
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setFilter( double data )
{
   if ( data < 0 || data > 1. )
   {
      printf("MLI_Solver_ParaSails::setThreshold WARNING - filter = 0.\n");
      filter_ = 0.;
   }
   else filter_ = data;
   return 0;
}

/******************************************************************************
 * set ParaSails loadbal parameter
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setLoadBal()
{
   loadbal_ = 1;
   return 0;
}

/******************************************************************************
 * set ParaSails tranpose parameter
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setTranspose()
{
   transpose_ = 1;
   return 0;
}

/******************************************************************************
 * set ParaSails smoother correction factor
 *---------------------------------------------------------------------------*/

int MLI_Solver_ParaSails::setUnderCorrection(double factor)
{
   correction_ = factor;
   return 0;
}

