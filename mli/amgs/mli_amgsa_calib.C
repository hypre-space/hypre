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
#include "../util/mli_utils.h"
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/parcsr_mv.h"
#include "utilities/utilities.h"
#include "seq_mv/seq_mv.h"
#include "mli_amgsa.h"
 
/***********************************************************************
 * generate multilevel structure
 * --------------------------------------------------------------------- */

int MLI_AMGSA::genMLStructureCalibration( MLI *mli ) 
{
   int          mypid, nprocs, *partition, targc, ndofs, nrows, n_null;
   int          i, j, k, level, *int_array, local_nrows, nsweeps;
   double       *dble_array, *dble2_array, start_time, elapsed_time;
   double       *sol_data, *nullspace_store, dtime, *Q_array, *R_array;
   char         param_string[100], *targv[10];
   MLI_Matrix   *mli_Amat;
   MLI_Vector   *mli_rhs, *mli_sol;
   MLI_OneLevel *single_level;
   MLI_Function *func_ptr;
   MLI          *new_mli;
   MLI_Method   *method_data;
   hypre_Vector       *sol_local;
   hypre_ParVector    *trial_sol, *zero_rhs;
   hypre_ParCSRMatrix *hypreA;

#ifdef MLI_DEBUG_DETAILED
   cout << " MLI_AMGSA::genMLStructureCalibration begins..." << endl;
   cout.flush();
#endif
cout << " MLI_AMGSA::genMLStructureCalibration begins..." << endl;

   /* --------------------------------------------------------------- */
   /* fetch machine and matrix information                            */
   /* --------------------------------------------------------------- */

   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   single_level = mli->getOneLevelObject( 0 );
   mli_Amat     = single_level->getAmat();
   hypreA       = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();

   /* --------------------------------------------------------------- */
   /* create trial vectors for calibration                            */
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
   func_ptr = new MLI_Function();
   MLI_Utils_HypreVectorGetDestroyFunc(func_ptr);
   sol_local = hypre_ParVectorLocalVector(trial_sol);
   sol_data  = hypre_VectorData(sol_local);

   /* --------------------------------------------------------------- */
   /* fetch current null space (since it will be destroyed inside     */
   /* MLI_AMGSA when setup is called.)                                */
   /* --------------------------------------------------------------- */

   sprintf( param_string, "getNullSpace" );
   method_data = mli->getMethod();
   method_data->getParams(param_string, &int_array, &nullspace_store);
   ndofs  = int_array[0];
   n_null = int_array[1];
   nrows  = int_array[2];
   nrows  = local_nrows;
   delete [] int_array;
   dble_array = nullspace_store;
   nullspace_store = new double[nrows*(n_null+calibration_size)];
   if ( dble_array != NULL )
   {
      for (i = 0; i < nrows*n_null; i++) nullspace_store[i] = dble_array[i]; 
      delete [] dble_array;
   }
   else
   {
      for ( j = 0; j < n_null; j++ ) 
      {
         for ( k = 0; k < nrows; k++ ) 
            if ( k % n_null == j ) nullspace_store[j*nrows+k] = 1.0; 
            else                   nullspace_store[j*nrows+k] = 0.0;
      }
   }

   /* --------------------------------------------------------------- */
   /* recover the other null space vectors                            */
   /* --------------------------------------------------------------- */

   sprintf( param_string, "MLI_AMGSA" );
   method_data->setName( param_string );
   sprintf( param_string, "setCoarseSolver SGS" );
   targc = 2;
   nsweeps = 20;
   dble_array = new double[nsweeps];
   for ( i = 0; i < nsweeps; i++ ) dble_array[i] = 1.0;
   targv[0] = (char *) &nsweeps;
   targv[1] = (char *) dble_array;
   method_data->setParams( param_string, targc, targv );
   delete [] dble_array;
   Q_array = new double[nrows*(n_null+calibration_size)];
   R_array = new double[(n_null+calibration_size)*(n_null+calibration_size)];
   for ( i = 0; i < calibration_size; i++ )
   {
      dtime = time_getWallclockSeconds();
      hypre_ParVectorSetRandomValues( trial_sol, (int) dtime );
      new_mli     = new MLI( mpi_comm );
      new_mli->setMaxIterations(2);
      new_mli->setMethod( method_data );
      new_mli->setSystemMatrix( 0, mli_Amat );
      new_mli->setup();
      sprintf(param_string, "HYPRE_ParVector");
      mli_sol = new MLI_Vector( (void*) trial_sol, param_string, NULL );
      mli_rhs = new MLI_Vector( (void*) zero_rhs, param_string, NULL );
      new_mli->cycle( mli_sol, mli_rhs );
      new_mli->resetSystemMatrix(0);
      new_mli->resetMethod();
      delete new_mli;
      for ( j = nrows*n_null; j < nrows*(n_null+1); j++ )
         nullspace_store[j] = sol_data[j-nrows*n_null];
      n_null++;
      for ( j = 0; j < nrows*n_null; j++ ) Q_array[j] = nullspace_store[j];
      MLI_Utils_QR( Q_array, R_array, nrows, n_null );
      for ( j = 0; j < n_null; j++ ) 
         printf("P%d : Norm of Null %d = %e\n", mypid,j,R_array[j*n_null+j]);
      targv[0] = (char *) &ndofs;
      targv[1] = (char *) &n_null;
      targv[2] = (char *) nullspace_store;
      targv[3] = (char *) &nrows;
      targc = 4;
      sprintf( param_string, "setNullSpace" );
      method_data->setParams(param_string, targc, targv);
      sprintf( param_string, "reinitialize" );
      method_data->setParams(param_string, 0, NULL);
   }
   delete [] Q_array;
   delete [] R_array;
   delete [] nullspace_store;
   level = genMLStructure( mli );
   hypre_ParVectorDestroy( trial_sol );
   hypre_ParVectorDestroy( zero_rhs );
   delete func_ptr;
   return level;
}

