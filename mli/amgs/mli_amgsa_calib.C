/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

// *********************************************************************
// This file is customized to use HYPRE matrix format
// *********************************************************************

// *********************************************************************
// local includes
// ---------------------------------------------------------------------

#include <string.h>
#include <iostream.h>
#include <assert.h>

#include "HYPRE.h"
#include "utilities/utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "seq_mv/seq_mv.h"
#include "parcsr_mv/parcsr_mv.h"

#include "mli_method_amgsa.h"
#include "../util/mli_utils.h"
 
/***********************************************************************
 * generate multilevel structure
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setupCalibration( MLI *mli ) 
{
   int          mypid, nprocs, *partition, ndofs, nrows, n_null;
   int          i, j, k, level, local_nrows, relax_num, targc, calib_size_tmp;
   double       *dble_array, *relax_wts, start_time;
   double       *sol_data, *nullspace_store, dtime, *Q_array, *R_array;
   char         param_string[100], **targv;
   MLI_Matrix   *mli_Amat;
   MLI_Vector   *mli_rhs, *mli_sol;
   MLI_OneLevel *single_level;
   MLI          *new_mli;
   MPI_Comm     mpi_comm;
   MLI_Method         *new_amgsa;
   hypre_Vector       *sol_local;
   hypre_ParVector    *trial_sol, *zero_rhs;
   hypre_ParCSRMatrix *hypreA;

#ifdef MLI_DEBUG_DETAILED
   cout << " MLI_Method_AMGSA::setupCalibration begins..." << endl;
   cout.flush();
#endif

   /* --------------------------------------------------------------- */
   /* fetch machine and matrix information                            */
   /* --------------------------------------------------------------- */

   mpi_comm = getComm();
   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   single_level = mli->getOneLevelObject( 0 );
   mli_Amat     = single_level->getAmat();
   hypreA       = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   targv         = new char*[4];

   /* --------------------------------------------------------------- */
   /* create trial vectors for calibration (trial_sol, zero_rhs)      */
   /* --------------------------------------------------------------- */

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partition);
   trial_sol = hypre_ParVectorCreate(mpi_comm, partition[nprocs], partition);
   hypre_ParVectorInitialize( trial_sol );
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partition);
   local_nrows = partition[mypid+1] - partition[mypid];
   zero_rhs = hypre_ParVectorCreate(mpi_comm, partition[nprocs], partition);
   hypre_ParVectorInitialize( zero_rhs );
   hypre_ParVectorSetConstantValues( zero_rhs, 0.0 );
   sol_local = hypre_ParVectorLocalVector(trial_sol);
   sol_data  = hypre_VectorData(sol_local);

   /* --------------------------------------------------------------- */
   /* fetch initial set of null space                                 */
   /* --------------------------------------------------------------- */

   getNullSpace(ndofs, n_null, nullspace_store, nrows);
   if ( nullspace_store != NULL )
   {
      dble_array = nullspace_store;
      nullspace_store = new double[nrows*(n_null+calibration_size)];
      for (i = 0; i < nrows*n_null; i++) nullspace_store[i] = dble_array[i]; 
   }
   else
   {
      nrows = local_nrows;
      nullspace_store = new double[nrows*(n_null+calibration_size)];
      for ( j = 0; j < n_null; j++ ) 
      {
         for ( k = 0; k < nrows; k++ ) 
            if ( k % n_null == j ) nullspace_store[j*nrows+k] = 1.0; 
            else                   nullspace_store[j*nrows+k] = 0.0;
      }
   }

   /* --------------------------------------------------------------- */
   /* clone the method definition (use SGS as approx coarse solver)   */
   /* And allocate temporaray arrays for QR decomposition.            */
   /* --------------------------------------------------------------- */

   relax_num = 20;
   relax_wts = new double[relax_num];
   for ( i = 0; i < relax_num; i++ ) relax_wts[i] = 1.0;
   new_amgsa = MLI_Method_CreateFromID( MLI_METHOD_AMGSA_ID, mpi_comm );
   sprintf( param_string, "setCoarseSolver SGS" );
   targc = 2;
   targv[0] = (char *) &relax_num;
   targv[1] = (char *) relax_wts;
   new_amgsa->setParams( param_string, targc, targv );
   Q_array = new double[nrows*(n_null+calibration_size)];
   R_array = new double[(n_null+calibration_size)*(n_null+calibration_size)];
   new_mli = new MLI( mpi_comm );
   new_mli->setMaxIterations(2);
   new_mli->setMethod( new_amgsa );
   new_mli->setSystemMatrix( 0, mli_Amat );

   /* --------------------------------------------------------------- */
   /* recover the other null space vectors                            */
   /* --------------------------------------------------------------- */

   start_time = MLI_Utils_WTime();

   for ( i = 0; i < calibration_size; i++ )
   {
      sprintf( param_string, "setNullSpace" );
      targc = 4;
      targv[0] = (char *) &ndofs;  
      targv[1] = (char *) &n_null;  
      targv[2] = (char *) nullspace_store;  
      targv[3] = (char *) &nrows;  
      new_amgsa->setParams( param_string, targc, targv );

      dtime = time_getWallclockSeconds();
      hypre_ParVectorSetRandomValues( trial_sol, (int) dtime );

      new_mli->setup();

      sprintf(param_string, "HYPRE_ParVector");
      mli_sol = new MLI_Vector( (void*) trial_sol, param_string, NULL );
      mli_rhs = new MLI_Vector( (void*) zero_rhs, param_string, NULL );
      new_mli->cycle( mli_sol, mli_rhs );


      for ( j = nrows*n_null; j < nrows*(n_null+1); j++ )
         nullspace_store[j] = sol_data[j-nrows*n_null];
      n_null++;
      for ( j = 0; j < nrows*n_null; j++ ) Q_array[j] = nullspace_store[j];
/*
      MLI_Utils_QR( Q_array, R_array, nrows, n_null );
      for ( j = 0; j < n_null; j++ ) 
         printf("P%d : Norm of Null %d = %e\n", mypid,j,R_array[j*n_null+j]);
*/
   }
   new_mli->resetSystemMatrix(0);
   delete new_mli;
   total_time += ( MLI_Utils_WTime() - start_time );
   delete [] Q_array;
   delete [] R_array;
   delete [] relax_wts;
   delete [] targv;
   setNullSpace(ndofs, n_null, nullspace_store, nrows);
   delete [] nullspace_store;
   calib_size_tmp = calibration_size;
   calibration_size = 0;
   level = setup( mli );
   calibration_size = calib_size_tmp;
   hypre_ParVectorDestroy( trial_sol );
   hypre_ParVectorDestroy( zero_rhs );
   return level;
}

