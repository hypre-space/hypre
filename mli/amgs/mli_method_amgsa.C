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
#include "mli_method_amgsa.h"
 
/* ********************************************************************* *
 * constructor
 * --------------------------------------------------------------------- */

MLI_Method_AMGSA::MLI_Method_AMGSA( MPI_Comm comm ) : MLI_Method( comm )
{
   char name[100];

   strcpy(name, "AMGSA");
   setName( name );
   setID( MLI_METHOD_AMGSA_ID );
   max_levels        = 40;
   num_levels        = 40;
   curr_level        = 0;
   output_level      = 0;
   node_dofs         = 1;
   curr_node_dofs    = 1;
   threshold         = 0.08;
   nullspace_dim     = 1;
   nullspace_vec     = NULL;
   nullspace_len     = 0;
   P_weight          = 4.0/3.0;
   drop_tol_for_P    = 0.0;            /* tolerance to sparsify P*/
   sa_counts         = new int[40];    /* number of aggregates   */
   sa_data           = new int*[40];   /* node to aggregate data */
   for ( int i = 0; i < 40; i++ ) sa_data[i] = NULL;
   spectral_norms    = new double[40]; /* calculated max eigen   */
   for ( int k = 0; k < 40; k++ ) spectral_norms[k] = 0.0;
   calc_norm_scheme  = 0;              /* use matrix rowsum norm */
   min_coarse_size   = 5;              /* smallest coarse grid   */
   coarsen_scheme    = MLI_METHOD_AMGSA_LOCAL;
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
   calibration_size  = 0;
   RAP_time          = 0.0;
   total_time        = 0.0;
}

/* ********************************************************************* *
 * destructor
 * --------------------------------------------------------------------- */

MLI_Method_AMGSA::~MLI_Method_AMGSA()
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
 * set parameters
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setParams(char *in_name, int argc, char *argv[])
{
   int        level, size, ndofs, num_ns, length, nsweeps=1, set_id;
   int        pre_post, nnodes;
   double     thresh, pweight, *nullspace, *weights=NULL, *coords, *scales;
   char       param1[256], param2[256];

   sscanf(in_name, "%s", param1);
   if ( !strcmp(param1, "setOutputLevel" ))
   {
      sscanf(in_name,"%s %d", param1, &level);
      return ( setOutputLevel( level ) );
   }
   else if ( !strcmp(param1, "setNumLevels" ))
   {
      sscanf(in_name,"%s %d", param1, &level);
      return ( setNumLevels( level ) );
   }
   else if ( !strcmp(param1, "setCoarsenScheme" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( !strcmp(param2, "local" ) )
         return ( setCoarsenScheme( MLI_METHOD_AMGSA_LOCAL ) );
      else      
      {
         cout << "MLI_Method_AMGSA ERROR : coarsen scheme not valid." << endl;
         cout << "     valid options are : local\n";
         return 1;
      }
   }
   else if ( !strcmp(param1, "setMinCoarseSize" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setMinCoarseSize( size ) );
   }
   else if ( !strcmp(param1, "setStrengthThreshold" ))
   {
      sscanf(in_name,"%s %lg", param1, &thresh);
      return ( setStrengthThreshold( thresh ) );
   }
   else if ( !strcmp(param1, "setPweight" ))
   {
      sscanf(in_name,"%s %lg", param1, &pweight);
      return ( setPweight( pweight ) );
   }
   else if ( !strcmp(param1, "setCalcSpectralNorm" ))
   {
      return ( setCalcSpectralNorm() );
   }
   else if ( !strcmp(param1, "setCalibrationSize" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setCalibrationSize( size ) );
   }
   else if ( !strcmp(param1, "setPreSmoother" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if      (!strcmp(param2, "Jacobi"))    set_id = MLI_SOLVER_JACOBI_ID;
      else if (!strcmp(param2, "GS"))        set_id = MLI_SOLVER_GS_ID;
      else if (!strcmp(param2, "SGS"))       set_id = MLI_SOLVER_SGS_ID;
      else if (!strcmp(param2, "Schwarz"))   set_id = MLI_SOLVER_SCHWARZ_ID;
      else if (!strcmp(param2, "MLS"))       set_id = MLI_SOLVER_MLS_ID;
      else if (!strcmp(param2, "ParaSails")) set_id = MLI_SOLVER_PARASAILS_ID;
      else 
      {
         cout << "MLI_Method_AMGSA ERROR : setSmoother - invalid smoother.\n";
         cout << "     valid options are : Jacobi, GS, SGS, Schwarz\n";
         cout << "                         MLS, ParaSails\n";
         return 1;
      } 
      if ( argc != 2 )
      {
         cout << "MLI_Method_AMGSA ERROR : setSmoother needs 2 arguments.\n";
         cout << "     argument[0] : number of relaxation sweeps \n";
         cout << "     argument[1] : relaxation weights\n";
         return 1;
      } 
      pre_post = MLI_SMOOTHER_PRE;
      nsweeps   = *(int *)   argv[0];
      weights   = (double *) argv[1];
      return ( setSmoother(pre_post,set_id,nsweeps,weights) );
   }
   else if ( !strcmp(param1, "setPostSmoother" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if      (!strcmp(param2, "Jacobi"))    set_id = MLI_SOLVER_JACOBI_ID;
      else if (!strcmp(param2, "GS"))        set_id = MLI_SOLVER_GS_ID;
      else if (!strcmp(param2, "SGS"))       set_id = MLI_SOLVER_SGS_ID;
      else if (!strcmp(param2, "Schwarz"))   set_id = MLI_SOLVER_SCHWARZ_ID;
      else if (!strcmp(param2, "MLS"))       set_id = MLI_SOLVER_MLS_ID;
      else if (!strcmp(param2, "ParaSails")) set_id = MLI_SOLVER_PARASAILS_ID;
      else 
      {
         cout << "MLI_Method_AMGSA ERROR : setSmoother - invalid smoother.\n";
         cout << "     valid options are : Jacobi, GS, SGS, Schwarz\n";
         cout << "                         MLS, ParaSails\n";
         return 1;
      } 
      if ( argc != 2 )
      {
         cout << "MLI_Method_AMGSA ERROR : setSmoother needs 2 arguments.\n";
         cout << "     argument[0] : number of relaxation sweeps \n";
         cout << "     argument[1] : relaxation weights\n";
         return 1;
      } 
      pre_post = MLI_SMOOTHER_POST;
      nsweeps  = *(int *)   argv[0];
      weights  = (double *) argv[1];
      return ( setSmoother(pre_post,set_id,nsweeps,weights) );
   }
   else if ( !strcmp(param1, "setCoarseSolver" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if      (!strcmp(param2, "Jacobi"))    set_id = MLI_SOLVER_JACOBI_ID;
      else if (!strcmp(param2, "GS"))        set_id = MLI_SOLVER_GS_ID;
      else if (!strcmp(param2, "SGS"))       set_id = MLI_SOLVER_SGS_ID;
      else if (!strcmp(param2, "Schwarz"))   set_id = MLI_SOLVER_SCHWARZ_ID;
      else if (!strcmp(param2, "ParaSails")) set_id = MLI_SOLVER_PARASAILS_ID;
      else if (!strcmp(param2, "SuperLU"))   set_id = MLI_SOLVER_SUPERLU_ID;
      else 
      {
         cout << "MLI_Method_AMGSA ERROR : setCoarseSolver - invalid solver\n";
         cout << "     valid options are : Jacobi, GS, SGS, Schwarz\n";
         cout << "                         ParaSails, SuperLu\n";
         return 1;
      } 
      if ( set_id != MLI_SOLVER_SUPERLU_ID && argc != 2 )
      {
         cout << "MLI_Method_AMGSA ERROR : setCoarseSolver needs 2 arguments.\n";
         cout << "     argument[0] : number of relaxation sweeps \n";
         cout << "     argument[1] : relaxation weights\n";
         return 1;
      } 
      else if ( set_id != MLI_SOLVER_SUPERLU_ID )
      {
         nsweeps   = *(int *)   argv[0];
         weights   = (double *) argv[1];
      }
      else if ( set_id == MLI_SOLVER_SUPERLU_ID )
      {
         nsweeps = 1;
         weights = NULL;
      }
      return ( setCoarseSolver(set_id,nsweeps,weights) );
   }
   else if ( !strcmp(param1, "setNullSpace" ))
   {
      if ( argc != 4 )
      {
         cout << "MLI_Method_AMGSA ERROR : setNullSpace needs 4 arguments.\n";
         cout << "     argument[0] : node degree of freedom\n";
         cout << "     argument[1] : number of null space vectors\n";
         cout << "     argument[2] : null space information\n";
         cout << "     argument[3] : vector length\n";
         return 1;
      } 
      ndofs     = *(int *)   argv[0];
      num_ns    = *(int *)   argv[1];
      nullspace = (double *) argv[2];
      length    = *(int *)   argv[3];
      return ( setNullSpace(ndofs,num_ns,nullspace,length) );
   }
   else if ( !strcmp(param1, "setNodalCoord" ))
   {
      if ( argc != 3 && argc != 4 )
      {
         cout << "MLI_Method_AMGSA ERROR : setNodalCoord needs 4 arguments.\n";
         cout << "     argument[0] : number of nodes\n";
         cout << "     argument[1] : node degree of freedom\n";
         cout << "     argument[2] : coordinate information\n";
         cout << "     argument[3] : scalings (can be null)\n";
         return 1;
      } 
      nnodes = *(int *)   argv[0];
      ndofs  = *(int *)   argv[1];
      coords = (double *) argv[2];
      if ( argc == 4 ) scales = (double *) argv[3]; else scales = NULL;
      return ( setNodalCoordinates(nnodes,ndofs,coords,scales) );
   }
   else if ( !strcmp(param1, "print" ))
   {
      return ( print() );
   }
   return 1;
}

/*****************************************************************************
 * get parameters 
 *--------------------------------------------------------------------------*/

int MLI_Method_AMGSA::getParams(char *in_name, int *argc, char *argv[])
{
   int    ndofs, num_ns, length;
   double *nullspace;

   if ( !strcmp(in_name, "getNullSpace" ))
   {
      if ( (*argc) < 4 )
      {
         cout << "MLI_Method_AMGSA ERROR : getNullSpace needs 4 args.\n";
         exit(1);
      }
      getNullSpace(node_dofs,num_ns,nullspace,length);
      argv[0] = (char *) &ndofs;
      argv[1] = (char *) &num_ns;
      argv[2] = (char *) nullspace;
      argv[3] = (char *) &length;
      (*argc) = 4;
      return 0;
   }
   else
   {
      cout << "MLI_Method_AMGSA ERROR : getParams string not valid.\n";
      return 1;
   }
}

/***********************************************************************
 * generate multilevel structure
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setup( MLI *mli ) 
{
   int             level, nsweeps, mypid;
   double          start_time, elapsed_time, max_eigen;
   char            param_string[100], *targv[10];
   MLI_Matrix      *mli_Pmat, *mli_Rmat, *mli_Amat, *mli_cAmat;
   MLI_Solver      *smoother_ptr, *csolve_ptr;
   MPI_Comm        comm;

#ifdef MLI_DEBUG_DETAILED
   cout << " MLI_Method_AMGSA::setup begins..." << endl;
   cout.flush();
#endif

   /* --------------------------------------------------------------- */
   /* clean up some mess made previously                              */
   /* --------------------------------------------------------------- */

   if ( sa_data != NULL )
   {
      for ( level = 0; level < max_levels; level++ )
      {
         if ( sa_data[level] != NULL ) delete [] sa_data[level];
         sa_data[level] = NULL;
      }
   }

   /* --------------------------------------------------------------- */
   /* call calibration if calibration size > 0                        */
   /* --------------------------------------------------------------- */

   if ( calibration_size > 0 ) return( setupCalibration( mli ) );
      
   /* --------------------------------------------------------------- */
   /* traverse all levels                                             */
   /* --------------------------------------------------------------- */

   RAP_time = 0.0;
   level    = 0;
   comm     = getComm();
   MPI_Comm_rank( comm, &mypid );
   mli_Amat   = mli->getSystemMatrix(level);
   total_time = MLI_Utils_WTime();

   for (level = 0; level < num_levels; level++ )
   {
      if ( mypid == 0 && output_level > 0 )
      {
         printf("\t*********************************************************\n");
         printf("\t*** Aggregation (uncoupled) : level = %d\n", level);
         printf("\t---------------------------------------------------------\n");
      }
      curr_level = level;
      if ( level == num_levels-1 ) break;

      /* ------fetch fine grid matrix----------------------------------- */

      mli_Amat = mli->getSystemMatrix(level);
      assert ( mli_Amat != NULL );

      /* ------perform coarsening--------------------------------------- */

      switch ( coarsen_scheme )
      {
         case MLI_METHOD_AMGSA_LOCAL :
              max_eigen = genPLocal(mli_Amat, &mli_Pmat); 
              break;
      }
      if ( max_eigen != 0.0 ) spectral_norms[level] = max_eigen;
      if ( mli_Pmat == NULL ) break;
      start_time = MLI_Utils_WTime();

      /* ------construct and set the coarse grid matrix----------------- */

      if ( mypid == 0 && output_level > 0 ) cout << "\tComputing RAP\n";
      MLI_Matrix_ComputePtAP(mli_Pmat, mli_Amat, &mli_cAmat);
      elapsed_time = (MLI_Utils_WTime() - start_time);
      RAP_time += elapsed_time;
      if ( mypid == 0 && output_level > 0 ) 
         cout << "\tRAP computed, time = " << elapsed_time << endl;
      //mli_Amat->print("Amat");
      //mli_Pmat->print("Pmat");
      //mli_cAmat->print("cAmat");
      mli->setSystemMatrix(level+1, mli_cAmat);

      /* ------set the prolongation matrix------------------------------ */

      mli->setProlongation(level+1, mli_Pmat);

      /* ------set the restriction matrix------------------------------- */

      sprintf(param_string, "HYPRE_ParCSRT");
      mli_Rmat = new MLI_Matrix(mli_Pmat->getMatrix(), param_string, NULL);
      mli->setRestriction(level, mli_Rmat);

      /* ------if MLS smoother, need to give the spectral radius-------- */

      if (pre_smoother == MLI_SOLVER_MLS_ID) pre_smoother_wgt[0] = max_eigen;
      if (postsmoother == MLI_SOLVER_MLS_ID) postsmoother_wgt[0] = max_eigen;

      /* ------set the smoothers---------------------------------------- */

      smoother_ptr = MLI_Solver_CreateFromID( pre_smoother );
      targv[0] = (char *) &pre_smoother_num;
      targv[1] = (char *) pre_smoother_wgt;
      sprintf( param_string, "relaxWeight" );
      smoother_ptr->setParams(param_string, 2, targv);
      smoother_ptr->setup(mli_Amat);
      mli->setSmoother( level, MLI_SMOOTHER_PRE, smoother_ptr );

      smoother_ptr = MLI_Solver_CreateFromID( postsmoother );
      targv[0] = (char *) &postsmoother_num;
      targv[1] = (char *) postsmoother_wgt;
      sprintf( param_string, "relaxWeight" );
      smoother_ptr->setParams(param_string, 2, targv);
      smoother_ptr->setup(mli_Amat);
      mli->setSmoother( level, MLI_SMOOTHER_POST, smoother_ptr );
   }

   /* ------set the coarse grid solver---------------------------------- */

   if ( mypid == 0 && output_level > 0 ) printf("\tCoarse level = %d\n", level);
   if (coarse_solver == MLI_SOLVER_MLS_ID) coarse_solver_wgt[0] = max_eigen;
   csolve_ptr = MLI_Solver_CreateFromID( coarse_solver );
   if (coarse_solver != MLI_SOLVER_SUPERLU_ID) 
   {
      targv[0] = (char *) &coarse_solver_num;
      targv[1] = (char *) coarse_solver_wgt;
      sprintf( param_string, "relaxWeight" );
      csolve_ptr->setParams(param_string, 2, targv);
   }
   mli_Amat = mli->getSystemMatrix(level);
   csolve_ptr->setup(mli_Amat);
   mli->setCoarseSolve(csolve_ptr);
   total_time = MLI_Utils_WTime() - total_time;

   /* --------------------------------------------------------------- */
   /* return the coarsest grid level number                           */
   /* --------------------------------------------------------------- */

   if ( output_level >= 2 ) printStatistics(mli);

#ifdef MLI_DEBUG_DETAILED
   cout << " MLI_Method_AMGSA::setup ends." << endl;
   cout.flush();
#endif
   return (level+1);
}

/* ********************************************************************* *
 * set diagnostics output level
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setOutputLevel( int level )
{
   output_level = level;
   return 0;
}

/* ********************************************************************* *
 * set number of levels 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setNumLevels( int nlevels )
{
   if ( nlevels < max_levels && nlevels > 0 ) num_levels = nlevels;
   return 0;
}

/* ********************************************************************* *
 * set smoother
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setSmoother(int pre_post,int set_id,int num,double *wgt)
{
   int i;

   if ( pre_post != MLI_SMOOTHER_PRE && pre_post != MLI_SMOOTHER_BOTH &&
        pre_post != MLI_SMOOTHER_POST )
   {
      cout << "MLI_Method_AMGSA:setSmoother ERROR - invalid info (1).\n";
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
         default : cout << "MLI_Method_AMGSA::setSmoother ERROR(2)\n";
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
         default : cout << "MLI_Method_AMGSA::setSmoother ERROR(3)\n";
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

int MLI_Method_AMGSA::setCoarseSolver( int set_id, int num, double *wgt )
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
      default : cout << "MLI_Method_AMGSA::setCoarseSolver ERROR : invalid\n";
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

int MLI_Method_AMGSA::setCoarsenScheme( int scheme )
{
   if ( scheme == MLI_METHOD_AMGSA_LOCAL ) 
   {
      coarsen_scheme = MLI_METHOD_AMGSA_LOCAL;
      return 0;
   }
   else
   {
      cout << "MLI_Method_AMGSA:setCoarsenScheme ERROR - invalid scheme.\n";
      cout.flush();
      return 1;
   }
}

/* ********************************************************************* *
 * set minimum coarse size
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setMinCoarseSize( int coarse_size )
{
   if ( coarse_size > 0 ) min_coarse_size = coarse_size;
   return 0;
}

/* ********************************************************************* *
 * set coarsening threshold
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setStrengthThreshold( double thresh )
{
   if ( thresh > 0.0 ) threshold = thresh;
   else                threshold = 0.0;
   return 0;
}

/* ********************************************************************* *
 * set damping factor for smoother prolongator
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setPweight( double weight )
{
   if ( weight >= 0.0 && weight <= 2.0 ) P_weight = weight;
   return 0;
}

/* ********************************************************************* *
 * indicate spectral norm is to be calculated
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setCalcSpectralNorm()
{
   calc_norm_scheme = 1;
   return 0;
}

/* ********************************************************************* *
 * load the null space
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setNullSpace( int ndofs, int ndim, double *nullvec, 
                                    int length ) 
{
   if ( (nullvec == NULL) && (ndofs != ndim) )
   {
      cout << "WARNING:  When no nullspace vector is specified, the nodal\n";
      cout << "DOFS must be equal to the nullspace dimension.\n";
      cout.flush();
      ndim = ndofs;
   }
   node_dofs      = ndofs;
   curr_node_dofs = ndofs;
   nullspace_dim  = ndim;
   nullspace_len  = length;
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
 * load nodal coordinates (translates into rigid body modes)
 * (abridged from similar function in ML)
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setNodalCoordinates( int num_nodes, int ndofs, 
                                           double *coords, double *scalings)
{
   int i, j, k, offset, voffset;

   if ( ndofs == 1 )
   {
      node_dofs      = 1;
      curr_node_dofs = 1;
      nullspace_len  = num_nodes;
      nullspace_dim  = 1;
   }
   else if ( ndofs == 3 )
   {
      node_dofs      = 3;
      curr_node_dofs = 3;
      nullspace_len  = num_nodes * 3;
      nullspace_dim  = 6;
   }
   else
   {
      printf("setNodalCoordinates: ndofs = %d not supported\n",ndofs);
      exit(1);
   }
   if ( nullspace_vec != NULL ) delete [] nullspace_vec;
   nullspace_vec = new double[nullspace_len * nullspace_dim];

   for( i = 0 ; i < num_nodes; i++ ) 
   {
      voffset = i * node_dofs;
      if      ( node_dofs == 1 ) nullspace_vec[i] = 1.0;
      else if ( node_dofs == 3 ) 
      {
         for ( j = 0; j < 3; j++ )
         {
            for( k = 0; k < 3; k++ )
            {
               offset = k * nullspace_len + voffset + j;
               if ( j == k ) nullspace_vec[offset] = 1.0;
               else          nullspace_vec[offset] = 0.0;
            }
         }
         for ( j = 0; j < 3; j++ )
         { 
            for ( k = 3; k < 6; k++ )
            {
               offset = k * nullspace_len + voffset + j;
               if ( j == k-3 ) nullspace_vec[offset] = 0.0;
               else 
               {
                  if      (j+k == 4) nullspace_vec[offset] = coords[i*3+2];
                  else if (j+k == 5) nullspace_vec[offset] = coords[i*3+1];
                  else if (j+k == 6) nullspace_vec[offset] = coords[i*3];
                  else nullspace_vec[offset] = 0.0;
               }
            }
         }
         j = 0; k = 5; offset = k * nullspace_len + voffset + j; 
         nullspace_vec[offset] *= -1.0;
         j = 1; k = 3; offset = k * nullspace_len + voffset + j; 
         nullspace_vec[offset] *= -1.0;
         j = 2; k = 4; offset = k * nullspace_len + voffset + j; 
         nullspace_vec[offset] *= -1.0;
      }
   }
   if ( scalings != NULL )
   {
      for ( i = 0 ; i < nullspace_dim; i++ ) 
         for ( j = 0 ; j < nullspace_len; j++ ) 
            nullspace_vec[i*nullspace_len+j] *= scalings[j];
   }
   return 0;
}

/* ********************************************************************* *
 * set parameter for calibration AMG 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setCalibrationSize( int size )
{
   if ( size > 0 ) calibration_size = size;
   return 0;
}

/* ********************************************************************* *
 * print AMGSA information
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::print()
{
   int      mypid;
   MPI_Comm comm = getComm();

   MPI_Comm_rank( comm, &mypid);
   if ( mypid == 0 )
   {
      cout << "\t************************************************************\n";
      cout << "\t*** method name             = " << getName() << endl;
      cout << "\t*** number of levels        = " << num_levels << endl;
      cout << "\t*** coarsen_scheme          = " << coarsen_scheme << endl;
      cout << "\t*** nodal degree of freedom = " << node_dofs << endl;
      cout << "\t*** null space dimension    = " << nullspace_dim << endl;
      cout << "\t*** strength threshold      = " << threshold << endl;
      cout << "\t*** Prolongator factor      = " << P_weight << endl;
      cout << "\t*** drop tolerance for P    = " << drop_tol_for_P << endl;
      cout << "\t*** calc_norm_scheme        = " << calc_norm_scheme << endl;
      cout << "\t*** minimum coarse size     = " << min_coarse_size << endl;
      switch ( pre_smoother )
      {
         case MLI_SOLVER_JACOBI_ID :
              cout << "\t*** pre  smoother type      = Jacobi\n"; break;
         case MLI_SOLVER_GS_ID :
              cout << "\t*** pre  smoother type      = Gauss Seidel\n"; break;
         case MLI_SOLVER_SGS_ID :
              cout << "\t*** pre  smoother type      = symm Gauss Seidel\n"; 
              break;
         case MLI_SOLVER_PARASAILS_ID :
              cout << "\t*** pre  smoother type      = ParaSails\n"; break; 
         case MLI_SOLVER_SCHWARZ_ID :
              cout << "\t*** pre  smoother type      = Schwarz\n"; break; 
         case MLI_SOLVER_MLS_ID :
              cout << "\t*** pre  smoother type      = MLS\n"; break; 
         case MLI_SOLVER_SUPERLU_ID :
              cout << "\t*** pre  smoother type      = SuperLU\n"; break; 
      }
      cout << "\t*** pre  smoother nsweeps   = " << pre_smoother_num << endl;  
      switch ( postsmoother )
      {
         case MLI_SOLVER_JACOBI_ID :
              cout << "\t*** post smoother type      = Jacobi\n"; break;
         case MLI_SOLVER_GS_ID :
              cout << "\t*** post smoother type      = Gauss Seidel\n"; break;
         case MLI_SOLVER_SGS_ID :
              cout << "\t*** post smoother type      = symm Gauss Seidel\n"; 
              break;
         case MLI_SOLVER_PARASAILS_ID :
              cout << "\t*** post smoother type      = ParaSails\n"; break; 
         case MLI_SOLVER_SCHWARZ_ID :
              cout << "\t*** post smoother type      = Schwarz\n"; break; 
         case MLI_SOLVER_MLS_ID :
              cout << "\t*** post smoother type      = MLS\n"; break; 
         case MLI_SOLVER_SUPERLU_ID :
              cout << "\t*** post smoother type      = SuperLU\n"; break; 
      }
      cout << "\t*** post smoother nsweeps   = " << postsmoother_num << endl;  
      switch ( coarse_solver )
      {
         case MLI_SOLVER_JACOBI_ID :
              cout << "\t*** coarse solver type      = Jacobi\n"; break;
         case MLI_SOLVER_GS_ID :
              cout << "\t*** coarse solver type      = Gauss Seidel\n"; break;
         case MLI_SOLVER_SGS_ID :
              cout << "\t*** coarse solver type      = symm Gauss Seidel\n"; 
              break;
         case MLI_SOLVER_PARASAILS_ID :
              cout << "\t*** coarse solver type      = ParaSails\n"; break; 
         case MLI_SOLVER_SCHWARZ_ID :
              cout << "\t*** coarse solver type      = Schwarz\n"; break; 
         case MLI_SOLVER_MLS_ID :
              cout << "\t*** coarse solver type      = MLS\n"; break; 
         case MLI_SOLVER_SUPERLU_ID :
              cout << "\t*** coarse solver type      = SuperLU\n"; break; 
      }
      cout << "\t*** coarse solver nsweeps   = " << coarse_solver_num << endl;  
      cout << "\t*** calibration size        = " << calibration_size << endl;  
      cout << "\t************************************************************\n";
      cout.flush();
   }
   return 0;
}

/* ********************************************************************* *
 * print AMGSA statistics information
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::printStatistics(MLI *mli)
{
   int          mypid, level, global_nrows, tot_nrows, fine_nrows;
   int          max_nnz, min_nnz, fine_nnz, tot_nnz, this_nnz, itemp;
   double       max_val, min_val, dtemp;
   char         param_string[100];
   MLI_Matrix   *mli_Amat, *mli_Pmat;
   MPI_Comm     comm = getComm();

   /* --------------------------------------------------------------- */
   /* output header                                                   */
   /* --------------------------------------------------------------- */

   MPI_Comm_rank( comm, &mypid);
   if ( mypid == 0 )
      cout << "\t******************** AMGSA Statistics **********************\n";

   /* --------------------------------------------------------------- */
   /* output processing time                                          */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      cout << "\t*** number of levels = " << curr_level+1 << endl;
      cout << "\t*** total RAP   time = " << RAP_time     << " seconds\n";
      cout << "\t*** total GenML time = " << total_time   << " seconds\n";
      cout << "\t********************** Amatrix *****************************\n";
      cout << "\t*level   Nrows MaxNnz MinNnz TotalNnz   maxValue   minValue*\n";
      cout.flush();
   }

   /* --------------------------------------------------------------- */
   /* fine and coarse matrix complexity information                   */
   /* --------------------------------------------------------------- */

   tot_nnz = tot_nrows = 0;
   for ( level = 0; level <= curr_level; level++ )
   {
      mli_Amat = mli->getSystemMatrix( level );
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
      cout << "\t********************** Pmatrix *****************************\n";
      cout << "\t*level   Nrows MaxNnz MinNnz TotalNnz   maxValue   minValue*\n";
      cout.flush();
   }
   for ( level = 1; level <= curr_level; level++ )
   {
      mli_Pmat = mli->getProlongation( level );
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
      cout << "\t************************************************************\n";
      dtemp = (double) tot_nnz / (double) fine_nnz;
      cout << "\t*** Amat complexity  = " << dtemp << endl;
      dtemp = (double) tot_nrows / (double) fine_nrows;
      cout << "\t*** grid complexity  = " << dtemp << endl;
      cout << "\t************************************************************\n";
      cout.flush();
   }
   return 0;
}

/* ********************************************************************* *
 * get the null space
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::getNullSpace(int &ndofs,int &ndim,double *&nullvec,
                                   int &leng) 
{
   ndofs   = curr_node_dofs;
   ndim    = nullspace_dim;
   nullvec = nullspace_vec;
   leng    = nullspace_len;
   return 0;
}

/* ********************************************************************* *
 * clone another object
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::copy( MLI_Method *new_obj )
{
   MLI_Method_AMGSA *new_amgsa;

   if ( ! strcmp(new_obj->getName(), "AMGSA" ) )
   {
      new_amgsa = (MLI_Method_AMGSA *) new_obj;
      new_amgsa->max_levels = max_levels;
      new_amgsa->setOutputLevel( output_level );
      new_amgsa->setNumLevels( num_levels );
      new_amgsa->setSmoother( MLI_SMOOTHER_PRE, pre_smoother, 
                              pre_smoother_num, pre_smoother_wgt );
      new_amgsa->setSmoother( MLI_SMOOTHER_POST, postsmoother, 
                              postsmoother_num, postsmoother_wgt );
      new_amgsa->setCoarseSolver(coarse_solver,coarse_solver_num,
                                 coarse_solver_wgt); 
      new_amgsa->setCoarsenScheme( coarsen_scheme );
      new_amgsa->setMinCoarseSize( min_coarse_size );
      if ( calc_norm_scheme ) new_amgsa->setCalcSpectralNorm();
      new_amgsa->setPweight( P_weight );
      new_amgsa->setNullSpace(node_dofs,nullspace_dim,nullspace_vec,
                              nullspace_len);
      new_amgsa->setStrengthThreshold( threshold );
   }
   else
   {
      cout << "MLI_Method_AMGSA::copy ERROR - incoming object not AMGSA.\n";
      exit(1);
   }
   return 0;
}

