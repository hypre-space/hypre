/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <string.h>
#include <iostream.h>
#include <assert.h>
#include "HYPRE.h"
#include "../util/mli_utils.h"
#include "../matrix/mli_matrix.h"
#include "../vector/mli_vector.h"
#include "../solver/mli_solver.h"
#include "../base/mli_defs.h"
#include "mli_amgsa.h"
 
/* ********************************************************************* *
 * constructor
 * --------------------------------------------------------------------- */

MLI_AMGSA::MLI_AMGSA( MPI_Comm comm )
{
   mpi_comm          = comm;
   strcpy(method_name, "MLI_AMGSA");
   max_levels        = 40;
   num_levels        = 40;
   curr_level        = 0;
   output_level      = 1;
   node_dofs         = 3;
   threshold         = 0.08;
   nullspace_dim     = 3;
   nullspace_vec     = NULL;
   P_weight          = 4.0/3.0;
   drop_tol_for_P    = 0.0;            /* tolerance to sparsify P*/
   sa_counts         = new int[40];    /* number of aggregates   */
   sa_data           = new int*[40];   /* node to aggregate data */
   for ( int i = 0; i < 40; i++ ) sa_data[i] = NULL;
   spectral_norms    = new double[40]; /* calculated max eigen   */
   for ( int k = 0; k < 40; k++ ) spectral_norms[k] = 0.0;
   calc_norm_scheme  = 0;              /* use matrix rowsum norm */
   min_coarse_size   = 20;             /* smallest coarse grid   */
   coarsen_scheme    = MLI_AMGSA_LOCAL;
   pre_smoother      = MLI_SOLVER_JACOBI_ID;
   postsmoother      = MLI_SOLVER_JACOBI_ID;
   pre_smoother_num  = 10;
   postsmoother_num  = 10;
   pre_smoother_wgt  = new double[2];
   postsmoother_wgt  = new double[2];
   pre_smoother_wgt[0] = pre_smoother_wgt[1] = 0.667;
   postsmoother_wgt[0] = postsmoother_wgt[1] = 0.667;
   coarse_solver       = MLI_SOLVER_GS_ID;
   coarse_solver_num   = 20;
   coarse_solver_wgt   = new double[20];
   for ( int j = 0; j < 20; j++ ) coarse_solver_wgt[j] = 1.0;
   calibration_size  = 1;
   RAP_time          = 0.0;
   total_time        = 0.0;
}

/* ********************************************************************* *
 * destructor
 * --------------------------------------------------------------------- */

MLI_AMGSA::~MLI_AMGSA()
{
   if ( nullspace_vec != NULL )
   {
      delete [] nullspace_vec;
      nullspace_vec = NULL;
   }
   if ( sa_counts != NULL )
   {
      delete [] sa_counts;
      sa_counts = NULL;
   }
   if ( sa_data != NULL )
   {
      for ( int i = 0; i < max_levels; i++ )
      {
         if ( sa_data[i] != NULL )
              delete [] sa_data[i];
         else break;
      }
      delete [] sa_data;
      sa_data = NULL;
   }
   if ( spectral_norms != NULL )
   {
      delete [] spectral_norms;
      spectral_norms = NULL;
   }
   if ( pre_smoother_wgt != NULL ) 
   {
      delete [] pre_smoother_wgt;
      pre_smoother_wgt = NULL;
   }
   if ( postsmoother_wgt != NULL ) 
   {
      delete [] postsmoother_wgt;
      postsmoother_wgt = NULL;
   }
   if ( coarse_solver_wgt != NULL ) 
   {
      delete [] coarse_solver_wgt;
      coarse_solver_wgt = NULL;
   }
}

/* ********************************************************************* *
 * set diagnostics output level
 * --------------------------------------------------------------------- */

int MLI_AMGSA::setOutputLevel( int level )
{
   output_level = level;
   return 0;
}

/* ********************************************************************* *
 * set number of levels 
 * --------------------------------------------------------------------- */

int MLI_AMGSA::setNumLevels( int nlevels )
{
   if ( nlevels < max_levels && nlevels > 0 ) num_levels = nlevels;
   return 0;
}

/* ********************************************************************* *
 * set smoother
 * --------------------------------------------------------------------- */

int MLI_AMGSA::setSmoother( int pre_post, int set_id, int num, double *wgt )
{
   int i;

   if ( pre_post != MLI_SMOOTHER_PRE && pre_post != MLI_SMOOTHER_BOTH &&
        pre_post != MLI_SMOOTHER_POST )
   {
      cout << "MLI_AMGSA:setSmoother ERROR - invalid info (1)." << endl;
      cout.flush();
      return 1;
   }
   if ( pre_post == MLI_SMOOTHER_PRE || pre_post == MLI_SMOOTHER_BOTH )
   {
      switch ( set_id )
      {
         case MLI_SOLVER_JACOBI_ID    : pre_smoother = MLI_SOLVER_JACOBI_ID;
                                        break;
         case MLI_SOLVER_GS_ID        : pre_smoother = MLI_SOLVER_GS_ID;
                                        break;
         case MLI_SOLVER_SGS_ID       : pre_smoother = MLI_SOLVER_SGS_ID;
                                        break;
         case MLI_SOLVER_PARASAILS_ID : pre_smoother = MLI_SOLVER_PARASAILS_ID;
                                        break;
         case MLI_SOLVER_SCHWARZ_ID   : pre_smoother = MLI_SOLVER_SCHWARZ_ID;
                                        break;
         case MLI_SOLVER_MLS_ID       : pre_smoother = MLI_SOLVER_MLS_ID;
                                        break;
         default : cout << "MLI_AMGSA::setSmoother ERROR(2)\n";
                   exit(1);
      }
      if ( num > 0 ) pre_smoother_num = num; else pre_smoother_num = 1;
      delete [] pre_smoother_wgt;
      pre_smoother_wgt = new double[pre_smoother_num];
      if ( wgt == NULL )
         for (i = 0; i < pre_smoother_num; i++) pre_smoother_wgt[i] = 1.;
      else
         for (i = 0; i < pre_smoother_num; i++) pre_smoother_wgt[i] = wgt[i];
   }
   if ( pre_post == MLI_SMOOTHER_POST || pre_post == MLI_SMOOTHER_BOTH )
   {
      switch ( set_id )
      {
         case MLI_SOLVER_JACOBI_ID    : postsmoother = MLI_SOLVER_JACOBI_ID;
                                        break;
         case MLI_SOLVER_GS_ID        : postsmoother = MLI_SOLVER_GS_ID;
                                        break;
         case MLI_SOLVER_SGS_ID       : postsmoother = MLI_SOLVER_SGS_ID;
                                        break;
         case MLI_SOLVER_PARASAILS_ID : postsmoother = MLI_SOLVER_PARASAILS_ID;
                                        break;
         case MLI_SOLVER_SCHWARZ_ID   : postsmoother = MLI_SOLVER_SCHWARZ_ID;
                                        break;
         case MLI_SOLVER_MLS_ID       : postsmoother = MLI_SOLVER_MLS_ID;
                                        break;
         default : cout << "MLI_AMGSA::setSmoother ERROR(3)\n";
                   exit(1);
      }
      if ( num > 0 ) postsmoother_num = num; else postsmoother_num = 1;
      delete [] postsmoother_wgt;
      postsmoother_wgt = new double[postsmoother_num];
      if ( wgt == NULL )
         for (i = 0; i < postsmoother_num; i++) postsmoother_wgt[i] = 1.;
      else
         for (i = 0; i < postsmoother_num; i++) postsmoother_wgt[i] = wgt[i];
   }
   return 0;
}

/* ********************************************************************* *
 * set coarse solver 
 * --------------------------------------------------------------------- */

int MLI_AMGSA::setCoarseSolver( int set_id, int num, double *wgt )
{
   int i;

   switch ( set_id )
   {
      case MLI_SOLVER_JACOBI_ID    : coarse_solver = MLI_SOLVER_JACOBI_ID;
                                     break;
      case MLI_SOLVER_GS_ID        : coarse_solver = MLI_SOLVER_GS_ID;
                                     break;
      case MLI_SOLVER_SGS_ID       : coarse_solver = MLI_SOLVER_SGS_ID;
                                     break;
      case MLI_SOLVER_PARASAILS_ID : coarse_solver = MLI_SOLVER_PARASAILS_ID;
                                     break;
      case MLI_SOLVER_SCHWARZ_ID   : coarse_solver = MLI_SOLVER_SCHWARZ_ID;
                                     break;
      case MLI_SOLVER_MLS_ID       : coarse_solver = MLI_SOLVER_MLS_ID;
                                     break;
      case MLI_SOLVER_SUPERLU_ID   : coarse_solver = MLI_SOLVER_SUPERLU_ID;
                                     break;
      default : cout << "MLI_AMGSA::setCoarseSolver ERROR : invalid\n";
                exit(1);
   }
   if ( num > 0 ) coarse_solver_num = num; else coarse_solver_num = 1;
   delete [] coarse_solver_wgt;
   if ( wgt != NULL && coarse_solver != MLI_SOLVER_SUPERLU_ID ) 
   {
      coarse_solver_wgt = new double[coarse_solver_num]; 
      for (i = 0; i < coarse_solver_num; i++) coarse_solver_wgt[i] = wgt[i];
   }
   else coarse_solver_wgt = NULL;
   return 0;
}

/* ********************************************************************* *
 * set coarsening scheme 
 * --------------------------------------------------------------------- */

int MLI_AMGSA::setCoarsenScheme( int scheme )
{
   if ( scheme == MLI_AMGSA_LOCAL ) 
   {
      coarsen_scheme = MLI_AMGSA_LOCAL;
      return 0;
   }
   else
   {
      cout << "MLI_AMGSA:setCoarsenScheme ERROR - invalid scheme." << endl;
      cout.flush();
      return 1;
   }
}

/* ********************************************************************* *
 * set minimum coarse size
 * --------------------------------------------------------------------- */

int MLI_AMGSA::setMinCoarseSize( int coarse_size  )
{
   if ( coarse_size > 0 ) min_coarse_size = coarse_size;
   return 0;
}

/* ********************************************************************* *
 * set coarsening threshold
 * --------------------------------------------------------------------- */

int MLI_AMGSA::setStrengthThreshold( double thresh )
{
   if ( thresh > 0.0 ) threshold = thresh;
   else                threshold = 0.0;
   return 0;
}

/* ********************************************************************* *
 * set damping factor for smoother prolongator
 * --------------------------------------------------------------------- */

int MLI_AMGSA::setPweight( double weight )
{
   if ( weight >= 0.0 && weight <= 2.0 ) P_weight = weight;
   return 0;
}

/* ********************************************************************* *
 * indicate spectral norm is to be calculated
 * --------------------------------------------------------------------- */

int MLI_AMGSA::setCalcSpectralNorm()
{
   calc_norm_scheme = 1;
   return 0;
}

/* ********************************************************************* *
 * load the null space
 * --------------------------------------------------------------------- */

int MLI_AMGSA::setNullSpace( int ndofs, int ndim, double *nullvec, 
                             int length ) 
{
   if ( (nullvec == NULL) && (ndofs != ndim) )
   {
      cout << "WARNING:  When no nullspace vector is specified, the nodal\n";
      cout << "DOFS must be equal to the nullspace dimension.\n";
      cout.flush();
      ndim = ndofs;
   }
   node_dofs     = ndofs;
   nullspace_dim = ndim;
   nullspace_len = length;
   if ( nullspace_vec != NULL ) delete [] nullspace_vec;
   if ( nullvec != NULL )
   {
      nullspace_vec = new double[length * ndim];
      for ( int i = 0; i < length*ndim; i++ )
         nullspace_vec[i] = nullvec[i];
   }
   else nullspace_vec = NULL;
   return 0;
}

/* ********************************************************************* *
 * set parameter for calibration AMG 
 * --------------------------------------------------------------------- */

int MLI_AMGSA::setCalibrationSize( int size )
{
   if ( size > 0 ) calibration_size = size;
   return 0;
}

/***********************************************************************
 * generate multilevel structure
 * --------------------------------------------------------------------- */

int MLI_AMGSA::genMLStructure( MLI *mli ) 
{
   int          level, nsweeps, mypid;
   double       start_time, elapsed_time, max_eigen;
   char         param_string[100], *targv[10];
   MLI_Matrix   *mli_Pmat, *mli_Rmat, *mli_Amat, *mli_cAmat;
   MLI_OneLevel *single_level, *next_level;
   MLI_Solver   *smoother_ptr, *csolve_ptr;

#ifdef MLI_DEBUG_DETAILED
   cout << " MLI_AMGSA::genMLStructure begins..." << endl;
   cout.flush();
#endif

   /* --------------------------------------------------------------- */
   /* traverse all levels                                             */
   /* --------------------------------------------------------------- */

   MPI_Comm_rank( mpi_comm, &mypid );
   single_level = mli->getOneLevelObject( 0 );
   mli_Amat     = single_level->getAmat();
   total_time   = MLI_Utils_WTime();
   RAP_time     = 0.0;

   for (level = 0; level < num_levels; level++ )
   {
      if ( mypid == 0 && output_level > 0 )
      {
         printf("\t*************************************************************\n");
         printf("\t*** Aggregation (uncoupled) : level = %d\n", level);
         printf("\t-------------------------------------------------------------\n");
      }
      curr_level   = level;
      single_level = mli->getOneLevelObject( level );
      next_level   = mli->getOneLevelObject( level+1 );

      // ----- fetch fine grid matrix

      mli_Amat     = single_level->getAmat();
      assert ( mli_Amat != NULL );

      // ----- perform coarsening

      switch ( coarsen_scheme )
      {
         case MLI_AMGSA_LOCAL :
              max_eigen = genPLocal(mli_Amat, &mli_Pmat); 
              break;
      }
      if ( max_eigen != 0.0 ) spectral_norms[level] = max_eigen;
      if ( mli_Pmat == NULL ) break;
      start_time = MLI_Utils_WTime();

      // ----- construct and set the coarse grid matrix

      if ( mypid == 0 && output_level > 0 ) cout << "\tComputing RAP\n";
      MLI_Matrix_ComputePtAP(mli_Pmat, mli_Amat, &mli_cAmat);
      elapsed_time = (MLI_Utils_WTime() - start_time);
      RAP_time += elapsed_time;
      if ( mypid == 0 && output_level > 0 ) 
         cout << "\tRAP computed, time = " << elapsed_time << endl;
      //mli_Amat->print("Amat");
      //mli_Pmat->print("Pmat");
      //mli_cAmat->print("cAmat");
      next_level->setAmat( mli_cAmat );

      // ----- set the prolongation matrix

      next_level->setPmat( mli_Pmat );

      // ----- set the restriction matrix

      sprintf(param_string, "HYPRE_ParCSRT");
      mli_Rmat = new MLI_Matrix(mli_Pmat->getMatrix(), param_string, NULL);
      single_level->setRmat( mli_Rmat );

      // ----- set the smoothers 

      if ( pre_smoother == MLI_SOLVER_MLS_ID ) 
         pre_smoother_wgt[0] = max_eigen;
      if ( postsmoother == MLI_SOLVER_MLS_ID ) 
         postsmoother_wgt[0] = max_eigen;
      smoother_ptr = MLI_Solver_Construct(pre_smoother);
      targv[0] = (char *) &pre_smoother_num;
      targv[1] = (char *) pre_smoother_wgt;
      sprintf( param_string, "relaxWeight" );
      smoother_ptr->setParams(param_string, 2, targv);
      smoother_ptr->setup(mli_Amat);
      single_level->setSmoother( MLI_SMOOTHER_PRE, smoother_ptr );
      smoother_ptr = MLI_Solver_Construct(postsmoother);
      targv[0] = (char *) &postsmoother_num;
      targv[1] = (char *) postsmoother_wgt;
      sprintf( param_string, "relaxWeight" );
      smoother_ptr->setParams(param_string, 2, targv);
      smoother_ptr->setup(mli_Amat);
      single_level->setSmoother( MLI_SMOOTHER_POST, smoother_ptr );
   }

   // ----- set the coarse grid solver 

   if (coarse_solver == MLI_SOLVER_MLS_ID) coarse_solver_wgt[0] = max_eigen;
   csolve_ptr = MLI_Solver_Construct(coarse_solver);
   targv[0] = (char *) &coarse_solver_num;
   targv[1] = (char *) coarse_solver_wgt;
   sprintf( param_string, "relaxWeight" );
   csolve_ptr->setParams(param_string, 2, targv);
   single_level = mli->getOneLevelObject( level );
   mli_Amat     = single_level->getAmat();
   csolve_ptr->setup(mli_Amat);
   single_level->setCoarseSolve( csolve_ptr );
   total_time = MLI_Utils_WTime() - total_time;

   /* --------------------------------------------------------------- */
   /* return the coarsest grid level number                           */
   /* --------------------------------------------------------------- */

   printStatistics(mli);

#ifdef MLI_DEBUG_DETAILED
   cout << " MLI_AMGSA::genMLStructure ends." << endl;
   cout.flush();
#endif
   return (level+1);
}

/* ********************************************************************* *
 * print AMGSA information
 * --------------------------------------------------------------------- */

int MLI_AMGSA::print()
{
   int mypid;
   MPI_Comm_rank( mpi_comm, &mypid);
   if ( mypid == 0 )
   {
      cout << "\t*************************************************************\n";
      cout << "\t*** method name             = " << method_name << endl;
      cout << "\t*** number of levels        = " << num_levels << endl;
      cout << "\t*** coarsen_scheme          = " << coarsen_scheme << endl;
      cout << "\t*** nodal degree of freedom = " << node_dofs << endl;
      cout << "\t*** null space dimension    = " << nullspace_dim << endl;
      cout << "\t*** strength threshold      = " << threshold << endl;
      cout << "\t*** Prolongator factor      = " << P_weight << endl;
      cout << "\t*** drop tolerance for P    = " << drop_tol_for_P << endl;
      cout << "\t*** calc_norm_scheme        = " << calc_norm_scheme << endl;
      cout << "\t*** minimum coarse size     = " << min_coarse_size << endl;
      cout << "\t*** pre  smoother type      = " << pre_smoother << endl;
      cout << "\t*** post smoother type      = " << postsmoother << endl;
      cout << "\t*************************************************************\n";
      cout.flush();
   }
   return 0;
}

/* ********************************************************************* *
 * print AMGSA statistics information
 * --------------------------------------------------------------------- */

int MLI_AMGSA::printStatistics(MLI *mli)
{
   int                mypid, level, global_nrows, tot_nrows, fine_nrows;
   int                max_nnz, min_nnz, fine_nnz, tot_nnz, this_nnz, itemp;
   double             max_val, min_val, dtemp;
   char               param_string[100];
   MLI_Matrix         *mli_Amat, *mli_Pmat;
   MLI_OneLevel       *single_level;

   /* --------------------------------------------------------------- */
   /* output header                                                   */
   /* --------------------------------------------------------------- */

   MPI_Comm_rank( mpi_comm, &mypid);
   if ( mypid == 0 )
      cout << "\t********************* AMGSA Statistics **********************\n";

   /* --------------------------------------------------------------- */
   /* output processing time                                          */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      cout << "\t*** number of levels = " << curr_level+1 << endl;
      cout << "\t*** total RAP   time = " << RAP_time     << " seconds\n";
      cout << "\t*** total GenML time = " << total_time   << " seconds\n";
      cout << "\t*********************** Amatrix *****************************\n";
      cout << "\t*level   Nrows MaxNnz MinNnz TotalNnz   maxValue   minValue *\n";
      cout.flush();
   }

   /* --------------------------------------------------------------- */
   /* fine and coarse matrix complexity information                   */
   /* --------------------------------------------------------------- */

   tot_nnz = tot_nrows = 0;
   for ( level = 0; level <= curr_level; level++ )
   {
      single_level = mli->getOneLevelObject( level );
      mli_Amat     = single_level->getAmat();
      sprintf(param_string, "nrows");
      mli_Amat->getMatrixInfo(param_string, global_nrows, dtemp);
      sprintf(param_string, "maxnnz");
      mli_Amat->getMatrixInfo(param_string, max_nnz, dtemp);
      sprintf(param_string, "minnnz");
      mli_Amat->getMatrixInfo(param_string, min_nnz, dtemp);
      sprintf(param_string, "totnnz");
      mli_Amat->getMatrixInfo(param_string, this_nnz, dtemp);
      sprintf(param_string, "maxval");
      mli_Amat->getMatrixInfo(param_string, itemp, max_val);
      sprintf(param_string, "minval");
      mli_Amat->getMatrixInfo(param_string, itemp, min_val);
      if ( mypid == 0 )
      {
         printf("\t*%3d %9d %5d  %5d %10d %9.3e %9.3e *\n",level,
                global_nrows, max_nnz, min_nnz, this_nnz, max_val, min_val);
      }
      if ( level == 0 ) fine_nnz = this_nnz;
      tot_nnz += this_nnz;
      if ( level == 0 ) fine_nrows = global_nrows;
      tot_nrows += global_nrows;
   }

   /* --------------------------------------------------------------- */
   /* prolongation operator complexity information                    */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      cout << "\t*********************** Pmatrix *****************************\n";
      cout << "\t*level   Nrows MaxNnz MinNnz TotalNnz   maxValue   minValue *\n";
      cout.flush();
   }
   for ( level = 1; level <= curr_level; level++ )
   {
      single_level = mli->getOneLevelObject( level );
      mli_Pmat     = single_level->getPmat();
      sprintf(param_string, "nrows");
      mli_Pmat->getMatrixInfo(param_string, global_nrows, dtemp);
      sprintf(param_string, "maxnnz");
      mli_Pmat->getMatrixInfo(param_string, max_nnz, dtemp);
      sprintf(param_string, "minnnz");
      mli_Pmat->getMatrixInfo(param_string, min_nnz, dtemp);
      sprintf(param_string, "totnnz");
      mli_Pmat->getMatrixInfo(param_string, this_nnz, dtemp);
      sprintf(param_string, "maxval");
      mli_Pmat->getMatrixInfo(param_string, itemp, max_val);
      sprintf(param_string, "minval");
      mli_Pmat->getMatrixInfo(param_string, itemp, min_val);
      if ( mypid == 0 )
      {
         printf("\t*%3d %9d %5d  %5d %10d %9.3e %9.3e *\n",level,
                global_nrows, max_nnz, min_nnz, this_nnz, max_val, min_val);
      }
   }

   /* --------------------------------------------------------------- */
   /* other complexity information                                    */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      cout << "\t*************************************************************\n";
      dtemp = (double) tot_nnz / (double) fine_nnz;
      cout << "\t*** Amat complexity  = " << dtemp << endl;
      dtemp = (double) tot_nrows / (double) fine_nrows;
      cout << "\t*** grid complexity  = " << dtemp << endl;
      cout << "\t*************************************************************\n";
      cout.flush();
   }
   return 0;
}

/* ********************************************************************* *
 * get the null space
 * --------------------------------------------------------------------- */

int MLI_AMGSA::getNullSpace(int &ndofs,int &ndim,double *&nullvec,int &leng) 
{
   ndofs   = node_dofs;
   ndim    = nullspace_dim;
   nullvec = nullspace_vec;
   leng    = nullspace_len;
   return 0;
}

/* ********************************************************************* *
 * reinitialize
 * --------------------------------------------------------------------- */

int MLI_AMGSA::reinitialize()
{
   curr_level = 0;
   for ( int i = 0; i < 40; i++ ) 
   {
      if ( sa_data[i] != NULL ) delete [] sa_data[i];
      sa_data[i] = NULL;
   }
   RAP_time   = 0.0;
   total_time = 0.0;
   return 0;
}

