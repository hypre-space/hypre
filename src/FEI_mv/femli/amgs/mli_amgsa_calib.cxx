/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/





// *********************************************************************
// This function is a first experiment to study calibrated smoothed
// aggregation methods.  Users first load the first set of null space
// vectors, and subsequent generation of addition approximate null space
// vectors are based on the previous set.
// *********************************************************************

// *********************************************************************
// This file is customized to use HYPRE matrix format
// *********************************************************************

// *********************************************************************
// local includes
// ---------------------------------------------------------------------

#include <string.h>
#include <assert.h>

#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "seq_mv/seq_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"

#include "amgs/mli_method_amgsa.h"
#include "util/mli_utils.h"
 
/***********************************************************************
 * generate multilevel structure using an adaptive method
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
   MLI          *new_mli;
   MPI_Comm     comm;
   MLI_Method         *new_amgsa;
   hypre_Vector       *sol_local;
   hypre_ParVector    *trial_sol, *zero_rhs;
   hypre_ParCSRMatrix *hypreA;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupCalibration begins...\n");
#endif

   /* --------------------------------------------------------------- */
   /* fetch machine and matrix information                            */
   /* --------------------------------------------------------------- */

   comm = getComm();
   MPI_Comm_rank( comm, &mypid );
   MPI_Comm_size( comm, &nprocs );
   mli_Amat = mli->getSystemMatrix( 0 );
   hypreA   = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   targv    = new char*[4];

   /* --------------------------------------------------------------- */
   /* create trial vectors for calibration (trial_sol, zero_rhs)      */
   /* --------------------------------------------------------------- */

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partition);
   trial_sol = hypre_ParVectorCreate(comm, partition[nprocs], partition);
   hypre_ParVectorInitialize( trial_sol );
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partition);
   local_nrows = partition[mypid+1] - partition[mypid];
   zero_rhs = hypre_ParVectorCreate(comm, partition[nprocs], partition);
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
      nullspace_store = new double[nrows*(n_null+calibrationSize_)];
      for (i = 0; i < nrows*n_null; i++) nullspace_store[i] = dble_array[i]; 
      delete [] dble_array;
   }
   else
   {
      nrows = local_nrows;
      nullspace_store = new double[nrows*(n_null+calibrationSize_)];
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
   new_amgsa = MLI_Method_CreateFromID( MLI_METHOD_AMGSA_ID, comm );
   copy( new_amgsa );
   sprintf( param_string, "setCoarseSolver SGS" );
   targc = 2;
   targv[0] = (char *) &relax_num;
   targv[1] = (char *) relax_wts;
   new_amgsa->setParams( param_string, targc, targv );
   Q_array = new double[nrows*(n_null+calibrationSize_)];
   R_array = new double[(n_null+calibrationSize_)*(n_null+calibrationSize_)];
   new_mli = new MLI( comm );
   new_mli->setMaxIterations(2);
   new_mli->setMethod( new_amgsa );
   new_mli->setSystemMatrix( 0, mli_Amat );

   /* --------------------------------------------------------------- */
   /* recover the other null space vectors                            */
   /* --------------------------------------------------------------- */

   start_time = MLI_Utils_WTime();

   for ( i = 0; i < calibrationSize_; i++ )
   {
      /* ------------------------------------------------------------ */
      /* set the current set of null space vectors                    */
      /* ------------------------------------------------------------ */

      sprintf( param_string, "setNullSpace" );
      targc = 4;
      targv[0] = (char *) &ndofs;  
      targv[1] = (char *) &n_null;  
      targv[2] = (char *) nullspace_store;  
      targv[3] = (char *) &nrows;  
      new_amgsa->setParams( param_string, targc, targv );

      dtime = time_getWallclockSeconds();

      /* ------------------------------------------------------------ */
      /* use random initial vectors for now and then call setup       */
      /* ------------------------------------------------------------ */

      hypre_ParVectorSetRandomValues( trial_sol, (int) dtime );

      new_mli->setup();

      /* ------------------------------------------------------------ */
      /* solve using a random initial vector and a zero rhs           */
      /* ------------------------------------------------------------ */

      sprintf(param_string, "HYPRE_ParVector");
      mli_sol = new MLI_Vector( (void*) trial_sol, param_string, NULL );
      mli_rhs = new MLI_Vector( (void*) zero_rhs, param_string, NULL );
      new_mli->cycle( mli_sol, mli_rhs );

      /* ------------------------------------------------------------ */
      /* add the new approximate null space vector to the current set */
      /* ------------------------------------------------------------ */

      for ( j = nrows*n_null; j < nrows*(n_null+1); j++ )
         nullspace_store[j] = sol_data[j-nrows*n_null];
      n_null++;
      for ( j = 0; j < nrows*n_null; j++ ) Q_array[j] = nullspace_store[j];
#if 0
      MLI_Utils_QR( Q_array, R_array, nrows, n_null );
      for ( j = 0; j < n_null; j++ ) 
         printf("P%d : Norm of Null %d = %e\n", mypid,j,R_array[j*n_null+j]);
#endif
   }

   totalTime_ += ( MLI_Utils_WTime() - start_time );

   /* --------------------------------------------------------------- */
   /* store the new set of null space vectors, and call mli setup     */
   /* --------------------------------------------------------------- */

   setNullSpace(ndofs, n_null, nullspace_store, nrows);
   calib_size_tmp = calibrationSize_;
   calibrationSize_ = 0;
   level = setup( mli );
   calibrationSize_ = calib_size_tmp;

   /* --------------------------------------------------------------- */
   /* clean up                                                        */
   /* --------------------------------------------------------------- */

   new_mli->resetSystemMatrix(0);
   delete new_mli;
   delete [] Q_array;
   delete [] R_array;
   delete [] relax_wts;
   delete [] targv;
   delete [] nullspace_store;
   hypre_ParVectorDestroy( trial_sol );
   hypre_ParVectorDestroy( zero_rhs );

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupCalibration ends.\n");
#endif
   return level;
}

/***********************************************************************
 * generate multilevel structure using an adaptive method (not done yet)
 * --------------------------------------------------------------------- */
#if 0
int MLI_Method_AMGSA::setupCalibration( MLI *mli ) 
{
   int          mypid, nprocs, *partition, ndofs, nrows, n_null;
   int          i, j, k, level, local_nrows, relax_num, targc, calib_size_tmp;
   double       *dble_array, *relax_wts, start_time;
   double       *sol_data, *nullspace_store, dtime, *Q_array, *R_array;
   char         param_string[100], **targv;
   MLI_Matrix   *mli_Amat;
   MLI_Vector   *mli_rhs, *mli_sol;
   MLI          *new_mli;
   MPI_Comm     comm;
   MLI_Method         *new_amgsa;
   hypre_Vector       *sol_local;
   hypre_ParVector    *trial_sol, *zero_rhs;
   hypre_ParCSRMatrix *hypreA;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupCalibration begins...\n");
#endif

   /* --------------------------------------------------------------- */
   /* fetch machine and matrix information                            */
   /* --------------------------------------------------------------- */

   comm = getComm();
   MPI_Comm_rank( comm, &mypid );
   MPI_Comm_size( comm, &nprocs );
   mli_Amat = mli->getSystemMatrix( 0 );
   hypreA   = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   targv    = new char*[4];
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partition);
   local_nrows = partition[mypid+1] - partition[mypid];
   free( partition );

   /* --------------------------------------------------------------- */
   /* fetch initial set of null space                                 */
   /* --------------------------------------------------------------- */

   getNullSpace(ndofs, n_null, nullspace_store, nrows);
   if ( nullspace_store != NULL ) delete [] nullspace_store;
   n_null = 0;
   nrows  = local_nrows;

   /* --------------------------------------------------------------- */
   /* create trial vectors for calibration (trial_sol, zero_rhs)      */
   /* --------------------------------------------------------------- */

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partition);
   trial_sol = hypre_ParVectorCreate(comm, partition[nprocs], partition);
   hypre_ParVectorInitialize( trial_sol );
   hypre_ParVectorSetRandomValues( trial_sol, (int) dtime );

   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partition);
   zero_rhs = hypre_ParVectorCreate(comm, partition[nprocs], partition);
   hypre_ParVectorInitialize( zero_rhs );
   hypre_ParVectorSetConstantValues( zero_rhs, 0.0 );

   sol_data  = hypre_VectorData(hypre_ParVectorLocalVector(trial_sol));

   sprintf(param_string, "HYPRE_ParVector");
   mli_sol = new MLI_Vector( (void*) trial_sol, param_string, NULL );
   mli_rhs = new MLI_Vector( (void*) zero_rhs, param_string, NULL );

   /* --------------------------------------------------------------- */
   /* compute approximate null vectors                                */
   /* --------------------------------------------------------------- */

   start_time = MLI_Utils_WTime();

   new_mli = NULL;

   for ( i = 0; i < calibrationSize_; i++ )
   {
      dtime = time_getWallclockSeconds();
      hypre_ParVectorSetRandomValues( trial_sol, (int) dtime );

      if ( i ==  0 ) 
      {
         smoother_ptr->solve(mli_rhs, mli_sol);
      }
      else
      {
         new_mli->cycle( mli_sol, mli_rhs );
         if (new_amgsa != NULL) delete new_amgsa;
         if (new_mli   != NULL) delete new_mli;
         delete mli_sol;
         delete mli_rhs;
      }

      /* ------------------------------------------------------------ */
      /* clone new amgsa and mli                                      */
      /* ------------------------------------------------------------ */

      new_amgsa = MLI_Method_CreateFromID( MLI_METHOD_AMGSA_ID, comm );
      copy( new_amgsa );
      new_amgsa->setNumLevels(2);
      new_mli = new MLI( comm );
      new_mli->setMaxIterations(2);
      new_mli->setMethod( new_amgsa );
      new_mli->setSystemMatrix( 0, mli_Amat );

      /* ------------------------------------------------------------ */
      /* construct and initialize the new null space                  */
      /* ------------------------------------------------------------ */

      offset = local_nrows * n_null;
      for (i = offset; i < offset+local_nrows; i++) 
         nullspace_store[i] = sol_data[i-offset]; 
      n_null++;

      sprintf( param_string, "setNullSpace" );
      targc = 4;
      targv[0] = (char *) &ndofs;  
      targv[1] = (char *) &n_null;  
      targv[2] = (char *) nullspace_store;  
      targv[3] = (char *) &nrows;  
      new_amgsa->setParams( param_string, targc, targv );

      if ( i < calibrationSize_-1 ) new_mli->setup();
   }
   delete new_amgsa;
   delete new_mli;

   totalTime_ += ( MLI_Utils_WTime() - start_time );

   /* --------------------------------------------------------------- */
   /* store the new set of null space vectors, and call mli setup     */
   /* --------------------------------------------------------------- */

   setNullSpace(ndofs, n_null, nullspace_store, nrows);
   calib_size_tmp = calibrationSize_;
   calibrationSize_ = 0;
   level = setup( mli );
   calibrationSize_ = calib_size_tmp;

   /* --------------------------------------------------------------- */
   /* clean up                                                        */
   /* --------------------------------------------------------------- */

   new_mli->resetSystemMatrix(0);
   delete new_mli;
   delete [] Q_array;
   delete [] R_array;
   delete [] relax_wts;
   delete [] targv;
   delete [] nullspace_store;
   hypre_ParVectorDestroy( trial_sol );
   hypre_ParVectorDestroy( zero_rhs );

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setupCalibration ends.\n");
#endif
   return level;
}
#endif

