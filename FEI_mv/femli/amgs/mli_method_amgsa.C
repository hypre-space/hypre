/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <string.h>
#include <strings.h>
#include <assert.h>
#include "HYPRE.h"
#include "util/mli_utils.h"
#include "matrix/mli_matrix.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"
#include "base/mli_defs.h"
#include "amgs/mli_method_amgsa.h"

/* ********************************************************************* *
 * functions external to MLI 
 * --------------------------------------------------------------------- */

extern "C"
{
   /* ARPACK function to compute eigenvalues/eigenvectors */

   void dnstev_(int *n, int *nev, char *which, double *sigmar, 
                double *sigmai, int *colptr, int *rowind, double *nzvals, 
                double *dr, double *di, double *z, int *ldz, int *info);
}

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
   spectral_norms    = new double[40]; /* calculated max eigen   */
   for ( int i = 0; i < 40; i++ ) 
   {
      sa_counts[i] = 0;
      sa_data[i]   = NULL;
      spectral_norms[i] = 0.0;
   }
   calc_norm_scheme  = 0;              /* use matrix rowsum norm */
   min_coarse_size   = 5;              /* smallest coarse grid   */
   coarsen_scheme    = MLI_METHOD_AMGSA_LOCAL;
   strcpy(pre_smoother, "Jacobi");
   strcpy(postsmoother, "Jacobi");
   pre_smoother_num  = 2;
   postsmoother_num  = 2;
   pre_smoother_wgt  = new double[2];
   postsmoother_wgt  = new double[2];
   pre_smoother_wgt[0] = pre_smoother_wgt[1] = 0.667;
   postsmoother_wgt[0] = postsmoother_wgt[1] = 0.667;
   strcpy(coarse_solver, "SGS");
   coarse_solver_num   = 20;
   coarse_solver_wgt   = new double[20];
   for ( int j = 0; j < 20; j++ ) coarse_solver_wgt[j] = 1.0;
   calibration_size    = 0;
   useSAMGeFlag_       = 0;
   RAP_time            = 0.0;
   total_time          = 0.0;
   ddObj               = NULL;
   ARPACKSuperLUExists_ = 0;
   sa_labels            = NULL;
   useSAMGDDFlag_       = 0;
   strcpy( paramFile_, "empty" );
}

/* ********************************************************************* *
 * destructor
 * --------------------------------------------------------------------- */

MLI_Method_AMGSA::~MLI_Method_AMGSA()
{
   char paramString[20];

   if ( nullspace_vec != NULL ) delete [] nullspace_vec;
   if ( sa_counts != NULL ) delete [] sa_counts;
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
   if ( sa_labels != NULL )
   {
      for ( int i = 0; i < max_levels; i++ )
      {
         if ( sa_labels[i] != NULL )
              delete [] sa_labels[i];
         else break;
      }
      delete [] sa_labels;
      sa_labels = NULL;
   }
   if ( spectral_norms    != NULL ) delete [] spectral_norms;
   if ( pre_smoother_wgt  != NULL ) delete [] pre_smoother_wgt;
   if ( postsmoother_wgt  != NULL ) delete [] postsmoother_wgt;
   if ( coarse_solver_wgt != NULL ) delete [] coarse_solver_wgt;
   if ( ddObj != NULL ) 
   {
      if ( ddObj->sendProcs != NULL ) delete [] ddObj->sendProcs;
      if ( ddObj->recvProcs != NULL ) delete [] ddObj->recvProcs;
      if ( ddObj->sendLengs != NULL ) delete [] ddObj->sendLengs;
      if ( ddObj->recvLengs != NULL ) delete [] ddObj->recvLengs;
      if ( ddObj->sendMap   != NULL ) delete [] ddObj->sendMap;
      if ( ddObj->ANodeEqnList != NULL ) delete [] ddObj->ANodeEqnList;
      if ( ddObj->SNodeEqnList != NULL ) delete [] ddObj->SNodeEqnList;
      delete ddObj;
   }
   if ( ARPACKSuperLUExists_ ) 
   {
      strcpy( paramString, "destroy" );
#ifdef MLI_ARPACK
      int  info;
      dnstev_(NULL, NULL, paramString, NULL, NULL, NULL, NULL, NULL, NULL, 
              NULL, NULL, NULL, &info);
#else
      printf("FATAL ERROR : ARPACK not installed.\n");
      exit(1);
#endif
   }
}

/* ********************************************************************* *
 * set parameters
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setParams(char *in_name, int argc, char *argv[])
{
   int        level, size, nDOF, numNS, length, nSweeps=1, offset;
   int        prePost, nnodes, nAggr, *aggrInfo, *labels, is, *indices;
   double     thresh, pweight, *nullspace, *weights=NULL, *coords, *scales;
   double     *nsAdjust;
   char       param1[256], param2[256], *param3;

   sscanf(in_name, "%s", param1);
   if ( !strcasecmp(param1, "setOutputLevel" ))
   {
      sscanf(in_name,"%s %d", param1, &level);
      return ( setOutputLevel( level ) );
   }
   else if ( !strcasecmp(param1, "setNumLevels" ))
   {
      sscanf(in_name,"%s %d", param1, &level);
      return ( setNumLevels( level ) );
   }
   else if ( !strcasecmp(param1, "useSAMGe" ))
   {
      useSAMGeFlag_ = 1;
      return 0;
   }
   else if ( !strcasecmp(param1, "useSAMGDD" ))
   {
      useSAMGDDFlag_ = 1;
      return 0;
   }
   else if ( !strcasecmp(param1, "setCoarsenScheme" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( !strcasecmp(param2, "local" ) )
         return ( setCoarsenScheme( MLI_METHOD_AMGSA_LOCAL ) );
      else      
      {
         printf("MLI_Method_AMGSA::setParams ERROR : setCoarsenScheme not");
         printf(" valid.  Valid options are : local \n");
         return 1;
      }
   }
   else if ( !strcasecmp(param1, "setMinCoarseSize" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setMinCoarseSize( size ) );
   }
   else if ( !strcasecmp(param1, "setStrengthThreshold" ))
   {
      sscanf(in_name,"%s %lg", param1, &thresh);
      return ( setStrengthThreshold( thresh ) );
   }
   else if ( !strcasecmp(param1, "setPweight" ))
   {
      sscanf(in_name,"%s %lg", param1, &pweight);
      return ( setPweight( pweight ) );
   }
   else if ( !strcasecmp(param1, "setCalcSpectralNorm" ))
   {
      return ( setCalcSpectralNorm() );
   }
   else if ( !strcasecmp(param1, "setAggregateInfo" ))
   {
      if ( argc != 4 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setAggregateInfo");
         printf(" needs 4 args.\n");
         printf("     argument[0] : level number \n");
         printf("     argument[1] : number of aggregates \n");
         printf("     argument[2] : total degree of freedom \n");
         printf("     argument[3] : aggregate information \n");
         return 1;
      } 
      level    = *(int *) argv[0];
      nAggr    = *(int *) argv[1];
      length   = *(int *) argv[2];
      aggrInfo = (int *)  argv[3];
      return ( setAggregateInfo(level,nAggr,length,aggrInfo) );
   }
   else if ( !strcasecmp(param1, "setCalibrationSize" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setCalibrationSize( size ) );
   }
   else if ( !strcasecmp(param1, "setPreSmoother" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( argc != 2 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setPreSmoother needs");
         printf(" 2 arguments.\n");
         printf("     argument[0] : number of relaxation sweeps \n");
         printf("     argument[1] : relaxation weights\n");
         return 1;
      } 
      prePost = MLI_SMOOTHER_PRE;
      nSweeps = *(int *)   argv[0];
      weights = (double *) argv[1];
      return ( setSmoother(prePost, param2, nSweeps, weights) );
   }
   else if ( !strcasecmp(param1, "setPostSmoother" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( argc != 2 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setPostSmoother needs");
         printf(" 2 arguments.\n");
         printf("     argument[0] : number of relaxation sweeps \n");
         printf("     argument[1] : relaxation weights\n");
         return 1;
      } 
      prePost = MLI_SMOOTHER_POST;
      nSweeps = *(int *)   argv[0];
      weights = (double *) argv[1];
      return ( setSmoother(prePost, param2, nSweeps, weights) );
   }
   else if ( !strcasecmp(param1, "setCoarseSolver" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( strcmp(param2, "SuperLU") && argc != 2 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setCoarseSolver needs");
         printf(" 2 arguments.\n");
         printf("     argument[0] : number of relaxation sweeps \n");
         printf("     argument[1] : relaxation weights\n");
         return 1;
      } 
      else if ( strcmp(param2, "SuperLU") )
      {
         nSweeps   = *(int *)   argv[0];
         weights   = (double *) argv[1];
      }
      else if ( !strcmp(param2, "SuperLU") )
      {
         nSweeps = 1;
         weights = NULL;
      }
      return ( setCoarseSolver(param2, nSweeps, weights) );
   }
   else if ( !strcasecmp(param1, "setNullSpace" ))
   {
      if ( argc != 4 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setNullSpace needs");
         printf(" 4 arguments.\n");
         printf("     argument[0] : node degree of freedom \n");
         printf("     argument[1] : number of null space vectors \n");
         printf("     argument[2] : null space information \n");
         printf("     argument[3] : vector length \n");
         return 1;
      } 
      nDOF      = *(int *)   argv[0];
      numNS     = *(int *)   argv[1];
      nullspace = (double *) argv[2];
      length    = *(int *)   argv[3];
      return ( setNullSpace(nDOF,numNS,nullspace,length) );
   }
   else if ( !strcasecmp(param1, "adjustNullSpace" ))
   {
      if ( argc != 1 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - adjustNullSpace needs");
         printf(" 1 argument.\n");
         printf("     argument[0] : adjustment vectors \n");
         return 1;
      } 
      nsAdjust = (double *) argv[0];
      return ( adjustNullSpace( nsAdjust ) );
   }
   else if ( !strcasecmp(param1, "resetNullSpaceComponents" ))
   {
      if ( argc != 3 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - resetNSComponents needs");
         printf(" 2 arguments.\n");
         printf("     argument[0] : number of equations \n");
         printf("     argument[1] : equation number offset \n");
         printf("     argument[2] : list of equation numbers \n");
         return 1;
      } 
      length  = *(int *) argv[0];
      offset  = *(int *) argv[1];
      indices =  (int *) argv[2];
      return ( resetNullSpaceComponents(length, offset, indices) );
   }
   else if ( !strcasecmp(param1, "setNodalCoord" ))
   {
      if ( argc != 3 && argc != 4 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setNodalCoord needs");
         printf(" 4 arguments.\n");
         printf("     argument[0] : number of nodes \n");
         printf("     argument[1] : node degree of freedom \n");
         printf("     argument[2] : coordinate information \n");
         printf("     argument[3] : scalings (can be null) \n");
         return 1;
      } 
      nnodes = *(int *)   argv[0];
      nDOF   = *(int *)   argv[1];
      coords = (double *) argv[2];
      if ( argc == 4 ) scales = (double *) argv[3]; else scales = NULL;
      return ( setNodalCoordinates(nnodes,nDOF,coords,scales) );
   }
   else if ( !strcasecmp(param1, "setLabels" ))
   {
      if ( argc != 4 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setLabels needs");
         printf(" 3 arguments.\n");
         printf("     argument[0] : vector length \n");
         printf("     argument[1] : level number \n");
         printf("     argument[2] : label information \n");
         return 1;
      } 
      length = *(int *) argv[0];
      level  = *(int *) argv[1];
      labels =  (int *) argv[2];
      if ( sa_labels == NULL ) 
      {
         sa_labels = new int*[max_levels];
         for ( is = 0; is < max_levels; is++ ) sa_labels[is] = NULL;
      }
      if ( level < 0 || level >= max_levels )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setLabels has \n");
         printf("invalid level number = %d (%d)\n", level, max_levels);
         return 1;
      }
      if ( sa_labels[level] != NULL ) delete [] sa_labels[level];
      sa_labels[level] = new int[length];
      for ( is = 0; is < length; is++ ) sa_labels[level][is] = labels[is];
      return 0;
   }
   else if ( !strcasecmp(param1, "setParamFile" ))
   {
      param3 = (char *) argv[0];
      strcpy( paramFile_, param3 ); 
      return 0;
   }
   else if ( !strcasecmp(param1, "print" ))
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
   int    nDOF, numNS, length;
   double *nullspace;

   if ( !strcasecmp(in_name, "getNullSpace" ))
   {
      if ( (*argc) < 4 )
      {
         printf("MLI_Method_AMGSA::getParams ERROR - getNullSpace needs");
         printf(" 4 arguments.\n");
         exit(1);
      }
      getNullSpace(node_dofs,numNS,nullspace,length);
      argv[0] = (char *) &nDOF;
      argv[1] = (char *) &numNS;
      argv[2] = (char *) nullspace;
      argv[3] = (char *) &length;
      (*argc) = 4;
      return 0;
   }
   else
   {
      printf("MLI_Method_AMGSA::getParams ERROR - invalid param string.\n");
      return 1;
   }
}

/***********************************************************************
 * generate multilevel structure
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setup( MLI *mli ) 
{
   int             level, mypid;
   double          start_time, elapsed_time, max_eigen;
   char            param_string[100], *targv[10];
   MLI_Matrix      *mli_Pmat, *mli_Rmat, *mli_Amat, *mli_cAmat;
   MLI_Solver      *smoother_ptr, *csolve_ptr;
   MPI_Comm        comm;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setup begins...\n");
#endif

   /* --------------------------------------------------------------- */
   /* clean up some mess made previously                              */
   /* --------------------------------------------------------------- */

   if ( sa_data != NULL )
   {
      for ( level = 1; level < max_levels; level++ )
      {
         if ( sa_data[level] != NULL ) delete [] sa_data[level];
         sa_data[level] = NULL;
      }
   }

   /* --------------------------------------------------------------- */
   /* call SAe and/or setupDD if flag is set                          */
   /* --------------------------------------------------------------- */

   if ( useSAMGeFlag_ )  setupSubdomainNullSpaceUsingFEData(mli);
   if ( useSAMGDDFlag_ ) setupDDFormSubdomainAggregate(mli);

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
   if ( nullspace_dim != node_dofs && nullspace_vec == NULL )
      nullspace_dim = node_dofs;

#if HAVE_LOBPCG
   relaxNullSpaces(mli_Amat);
#endif
   
   for (level = 0; level < num_levels; level++ )
   {
      if ( mypid == 0 && output_level > 0 )
      {
         printf("\t*****************************************************\n");
         printf("\t*** Aggregation (uncoupled) : level = %d\n", level);
         printf("\t-----------------------------------------------------\n");
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
              if ( level == 0 )
                 max_eigen = genPLocal(mli_Amat, &mli_Pmat, sa_counts[0], 
                                       sa_data[0]); 
              else
                 max_eigen = genPLocal(mli_Amat, &mli_Pmat, 0, NULL); 
              break;
      }
      if ( max_eigen != 0.0 ) spectral_norms[level] = max_eigen;
      if ( mli_Pmat == NULL ) break;
      start_time = MLI_Utils_WTime();

      /* ------construct and set the coarse grid matrix----------------- */

      if ( mypid == 0 && output_level > 0 ) printf("\tComputing RAP\n");
      MLI_Matrix_ComputePtAP(mli_Pmat, mli_Amat, &mli_cAmat);
      mli->setSystemMatrix(level+1, mli_cAmat);
      elapsed_time = (MLI_Utils_WTime() - start_time);
      RAP_time += elapsed_time;
      if ( mypid == 0 && output_level > 0 ) 
         printf("\tRAP computed, time = %e seconds.\n", elapsed_time);

#if 0
      mli_Amat->print("Amat");
      mli_Pmat->print("Pmat");
      mli_cAmat->print("cAmat");
#endif

      /* ------set the prolongation matrix------------------------------ */

      mli->setProlongation(level+1, mli_Pmat);

      /* ------set the restriction matrix------------------------------- */

      sprintf(param_string, "HYPRE_ParCSRT");
      mli_Rmat = new MLI_Matrix(mli_Pmat->getMatrix(), param_string, NULL);
      mli->setRestriction(level, mli_Rmat);

      /* ------set the smoothers---------------------------------------- */

      if ( useSAMGDDFlag_ && num_levels == 2 && 
           !strcmp(pre_smoother, "ARPACKSuperLU") )
      {
         setupDDSuperLUSmoother(mli, level);
         smoother_ptr = MLI_Solver_CreateFromName(pre_smoother);
         targv[0] = (char *) ddObj;
         sprintf( param_string, "ARPACKSuperLUObject" );
         smoother_ptr->setParams(param_string, 1, targv);
         smoother_ptr->setup(mli_Amat);
         mli->setSmoother( level, MLI_SMOOTHER_PRE, smoother_ptr );
#if 0
         smoother_ptr = MLI_Solver_CreateFromName(pre_smoother);
         smoother_ptr->setParams(param_string, 1, targv);
         smoother_ptr->setup(mli_Amat);
         mli->setSmoother( level, MLI_SMOOTHER_POST, smoother_ptr );
#endif
         continue;
      }
      smoother_ptr = MLI_Solver_CreateFromName( pre_smoother );
      targv[0] = (char *) &pre_smoother_num;
      targv[1] = (char *) pre_smoother_wgt;
      sprintf( param_string, "relaxWeight" );
      smoother_ptr->setParams(param_string, 2, targv);
      if ( !strcmp(pre_smoother, "MLS") ) 
      {
         sprintf( param_string, "maxEigen" );
         targv[0] = (char *) &max_eigen;
         smoother_ptr->setParams(param_string, 1, targv);
      }
      smoother_ptr->setup(mli_Amat);
      mli->setSmoother( level, MLI_SMOOTHER_PRE, smoother_ptr );

      if ( strcmp(pre_smoother, postsmoother) )
      {
         smoother_ptr = MLI_Solver_CreateFromName( postsmoother );
         targv[0] = (char *) &postsmoother_num;
         targv[1] = (char *) postsmoother_wgt;
         sprintf( param_string, "relaxWeight" );
         smoother_ptr->setParams(param_string, 2, targv);
         if ( !strcmp(postsmoother, "MLS") ) 
         {
            sprintf( param_string, "maxEigen" );
            targv[0] = (char *) &max_eigen;
            smoother_ptr->setParams(param_string, 1, targv);
         }
         smoother_ptr->setup(mli_Amat);
      }
      mli->setSmoother( level, MLI_SMOOTHER_POST, smoother_ptr );
   }

   /* ------set the coarse grid solver---------------------------------- */

   if (mypid == 0 && output_level > 0) printf("\tCoarse level = %d\n",level);
   csolve_ptr = MLI_Solver_CreateFromName( coarse_solver );
   if ( strcmp(coarse_solver, "SuperLU") )
   {
      targv[0] = (char *) &coarse_solver_num;
      targv[1] = (char *) coarse_solver_wgt;
      sprintf( param_string, "relaxWeight" );
      csolve_ptr->setParams(param_string, 2, targv);
      if ( !strcmp(coarse_solver, "MLS") )
      {
         sprintf( param_string, "maxEigen" );
         targv[0] = (char *) &max_eigen;
         csolve_ptr->setParams(param_string, 1, targv);
      }
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
   printf("MLI_Method_AMGSA::setup ends.");
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

int MLI_Method_AMGSA::setSmoother(int prePost, char *stype, int num, 
                                  double *wgt)
{
   int i;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setSmoother - type = %s.\n", stype);
#endif

   if ( prePost != MLI_SMOOTHER_PRE && prePost != MLI_SMOOTHER_BOTH &&
        prePost != MLI_SMOOTHER_POST )
   {
      printf("MLI_Method_AMGSA::setSmoother ERROR - invalid info (1).\n");
      return 1;
   }
   if ( prePost == MLI_SMOOTHER_PRE || prePost != MLI_SMOOTHER_BOTH )
   {
      strcpy( pre_smoother, stype );
      if ( num > 0 ) pre_smoother_num = num; else pre_smoother_num = 1;
      delete [] pre_smoother_wgt;
      pre_smoother_wgt = new double[pre_smoother_num];
      if ( wgt == NULL )
         for (i = 0; i < pre_smoother_num; i++) pre_smoother_wgt[i] = 0.;
      else
         for (i = 0; i < pre_smoother_num; i++) pre_smoother_wgt[i] = wgt[i];
   }
   if ( prePost == MLI_SMOOTHER_POST || prePost == MLI_SMOOTHER_BOTH )
   {
      strcpy( postsmoother, stype );
      if ( num > 0 ) postsmoother_num = num; else postsmoother_num = 1;
      delete [] postsmoother_wgt;
      postsmoother_wgt = new double[postsmoother_num];
      if ( wgt == NULL )
         for (i = 0; i < postsmoother_num; i++) postsmoother_wgt[i] = 0.;
      else
         for (i = 0; i < postsmoother_num; i++) postsmoother_wgt[i] = wgt[i];
   }
   return 0;
}

/* ********************************************************************* *
 * set coarse solver 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setCoarseSolver( char *stype, int num, double *wgt )
{
   int i;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setCoarseSolver - type = %s.\n", stype);
#endif

   strcpy( coarse_solver, stype );
   if ( num > 0 ) coarse_solver_num = num; else coarse_solver_num = 1;
   delete [] coarse_solver_wgt;
   if ( wgt != NULL && strcmp(coarse_solver, "SuperLU") )
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
      printf("MLI_Method_AMGSA::setCoarsenScheme ERROR - invalid scheme.\n");
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
 * load the initial aggregate information 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setAggregateInfo(int level, int aggrCnt, int length,
                                       int *aggrInfo) 
{
   if ( level != 0 )
   {
      printf("MLI_Method_AMGSA::setAggregateInfo ERROR : invalid level");
      printf(" number = %d.", level);
      return 1;
   }
   sa_counts[level] = aggrCnt;
   if ( sa_data[level] != NULL ) delete [] sa_data[level];
   sa_data[level] = new int[length];
   for ( int i = 0; i < length; i++ ) sa_data[level][i] = aggrInfo[i];
   return 0;
}

/* ********************************************************************* *
 * load the null space
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setNullSpace( int nDOF, int ndim, double *nullvec, 
                                    int length ) 
{
#if 0
   if ( (nullvec == NULL) && (nDOF != ndim) )
   {
      printf("MLI_Method_AMGSA::setNullSpace WARNING -  When no nullspace\n");
      printf(" vector is specified, the nodal DOFS must be equal to the \n");
      printf("nullspace dimension.\n");
      ndim = nDOF;
   }
#endif
   node_dofs      = nDOF;
   curr_node_dofs = nDOF;
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
 * adjust null space vectors
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::adjustNullSpace(double *vecAdjust)
{
   int i;

   if ( useSAMGeFlag_ ) return 0;

   for ( i = 0; i < nullspace_len*nullspace_dim; i++ )
      nullspace_vec[i] += vecAdjust[i];

   return 0;
}

/* ********************************************************************* *
 * reset some entry in the null space vectors
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::resetNullSpaceComponents(int length, int start,
                                               int *eqnIndices)
{
   int i, j, index;

   if ( useSAMGeFlag_ ) return 0;

   for ( i = 0; i < length; i++ )
   {
      index = eqnIndices[i] - start;
      for ( j = 0; j < nullspace_dim; j++ )
         nullspace_vec[j*nullspace_len+index] = 0.;
   }
   return 0;
}

/* ********************************************************************* *
 * load nodal coordinates (translates into rigid body modes)
 * (abridged from similar function in ML)
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setNodalCoordinates(int num_nodes, int nDOF, 
                                          double *coords, double *scalings)
{
   int i, j, k, offset, voffset, mypid;
   MPI_Comm comm = getComm();
   MPI_Comm_rank( comm, &mypid );

   if ( useSAMGeFlag_ ) return 0;

   if ( nDOF == 1 )
   {
      node_dofs      = 1;
      curr_node_dofs = 1;
      nullspace_len  = num_nodes;
      nullspace_dim  = 1;
   }
   else if ( nDOF == 3 )
   {
      node_dofs      = 3;
      curr_node_dofs = 3;
      nullspace_len  = num_nodes * 3;
      nullspace_dim  = 6;
   }
   else
   {
      printf("setNodalCoordinates: nDOF = %d not supported\n",nDOF);
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
      printf("\t********************************************************\n");
      printf("\t*** method name             = %s\n", getName());
      printf("\t*** number of levels        = %d\n", num_levels);
      printf("\t*** coarsen_scheme          = %d\n", coarsen_scheme);
      printf("\t*** nodal degree of freedom = %d\n", node_dofs);
      printf("\t*** null space dimension    = %d\n", nullspace_dim);
      printf("\t*** strength threshold      = %e\n", threshold);
      printf("\t*** Prolongator factor      = %e\n", P_weight);
      printf("\t*** drop tolerance for P    = %e\n", drop_tol_for_P);
      printf("\t*** calc_norm_scheme        = %d\n", calc_norm_scheme);
      printf("\t*** minimum coarse size     = %d\n", min_coarse_size);
      printf("\t*** pre  smoother type      = %s\n", pre_smoother); 
      printf("\t*** pre  smoother nsweeps   = %d\n", pre_smoother_num);
      printf("\t*** post smoother type      = %s\n", postsmoother); 
      printf("\t*** post smoother nsweeps   = %d\n", postsmoother_num);
      printf("\t*** coarse solver type      = %s\n", coarse_solver); 
      printf("\t*** coarse solver nsweeps   = %d\n", coarse_solver_num);  
      printf("\t*** calibration size        = %d\n", calibration_size);
      printf("\t********************************************************\n");
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
      printf("\t****************** AMGSA Statistics ********************\n");

   /* --------------------------------------------------------------- */
   /* output processing time                                          */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      printf("\t*** number of levels = %d\n", curr_level+1);
      printf("\t*** total RAP   time = %e seconds\n", RAP_time);
      printf("\t*** total GenML time = %e seconds\n", total_time);
      printf("\t******************** Amatrix ***************************\n");
      printf("\t*level   Nrows MaxNnz MinNnz TotalNnz  maxValue  minValue*\n");
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
         printf("\t*%3d %9d %5d  %5d %10d %8.3e %8.3e *\n",level,
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
      printf("\t******************** Pmatrix ***************************\n");
      printf("\t*level   Nrows MaxNnz MinNnz TotalNnz  maxValue  minValue*\n");
      fflush(stdout);
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
         printf("\t*%3d %9d %5d  %5d %10d %8.3e %8.3e *\n",level,
                global_nrows, max_nnz, min_nnz, this_nnz, max_val, min_val);
      }
   }

   /* --------------------------------------------------------------- */
   /* other complexity information                                    */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      printf("\t********************************************************\n");
      dtemp = (double) tot_nnz / (double) fine_nnz;
      printf("\t*** Amat complexity  = %e\n", dtemp);
      dtemp = (double) tot_nrows / (double) fine_nrows;
      printf("\t*** grid complexity  = %e\n", dtemp);
      printf("\t********************************************************\n");
      fflush(stdout);
   }
   return 0;
}

/* ********************************************************************* *
 * get the null space
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::getNullSpace(int &nDOF,int &ndim,double *&nullvec,
                                   int &leng) 
{
   nDOF    = curr_node_dofs;
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

   if ( ! strcasecmp(new_obj->getName(), "AMGSA" ) )
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
      printf("MLI_Method_AMGSA::copy ERROR - incoming object not AMGSA.\n");
      exit(1);
   }
   return 0;
}

/* ********************************************************************* *
 * LOBPCG subroutine calls
 * ********************************************************************* */

#ifdef HAVE_LOBPCG
#ifdef __cplusplus
extern "C" {
#endif

#include "../../../eigen/lobpcg/lobpcg.h"
#include "../../../IJ_mv/IJ_mv.h"
#include "../../../parcsr_mv/parcsr_mv.h"
#include "../../../seq_mv/seq_mv.h"
#include "../../../parcsr_ls/parcsr_ls.h"
HYPRE_Solver	   lobHYPRESolver;
HYPRE_ParCSRMatrix lobHYPREA;

int Funct_Solve(HYPRE_ParVector b,HYPRE_ParVector x)
{
   int ierr=0;
   ierr=HYPRE_ParCSRPCGSolve(lobHYPRESolver,lobHYPREA,b,x);assert2(ierr);
   return 0;
}
int Func_Matvec(HYPRE_ParVector x,HYPRE_ParVector y)
{
   int ierr=0;
   ierr=HYPRE_ParCSRMatrixMatvec(1.0,lobHYPREA,x,0.0,y);assert2(ierr);
   return 0;
}
#ifdef __cplusplus
}
#endif
#endif

/* ********************************************************************* *
 * relax null spaces 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::relaxNullSpaces(MLI_Matrix *mli_Amat)
{
#ifdef HAVE_LOBPCG
   int                mypid, *partitioning, startRow, endRow, localNRows;
   int                iV, i, offset, *cols;
   double             *eigval, *uData;
   MPI_Comm           comm;
   HYPRE_IJVector     tempIJ;
   hypre_ParVector    **lobVecs = new hypre_ParVector*[nullspace_dim];
   hypre_ParCSRMatrix *hypreA;
   HYPRE_Solver       HYPrecon=NULL;
   HYPRE_LobpcgData   lobpcgdata;
   int (*FuncT)(HYPRE_ParVector x,HYPRE_ParVector y);

   comm     = getComm();
   MPI_Comm_rank( comm, &mypid );
   hypreA = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA, 
                                        &partitioning );
   startRow   = partitioning[mypid];
   endRow     = partitioning[mypid+1] - 1;
   localNRows = endRow - startRow + 1;
   cols       = new int[localNRows];
   for ( i = startRow; i <= endRow; i++ ) cols[i] = startRow + i;

   for ( iV = 0; iV < nullspace_dim; iV++ )
   {
      HYPRE_IJVectorCreate(comm, startRow, endRow, &tempIJ);
      HYPRE_IJVectorSetObjectType(tempIJ, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(tempIJ);
      HYPRE_IJVectorAssemble(tempIJ);
      offset = nullspace_len * iV ;
      HYPRE_IJVectorSetValues(tempIJ, localNRows, (const int *) cols,
                              (const double *) &(nullspace_vec[offset]));
      HYPRE_IJVectorGetObject(tempIJ, (void **) &(lobVecs[iV]));
//HYPRE_ParVectorSetRandomValues( (HYPRE_ParVector) lobVecs[iV], 9001*iV*7901 );
      HYPRE_IJVectorSetObjectType(tempIJ, -1);
      HYPRE_IJVectorDestroy(tempIJ);
   }
   delete [] cols;
   free(partitioning);

   printf("LOBPCG Solve\n");
   HYPRE_LobpcgCreate(&lobpcgdata);
   HYPRE_LobpcgSetVerbose(lobpcgdata);
   HYPRE_LobpcgSetBlocksize(lobpcgdata, nullspace_dim);
   FuncT = Funct_Solve;
   HYPRE_LobpcgSetSolverFunction(lobpcgdata,FuncT);
   HYPRE_LobpcgSetup(lobpcgdata);
   lobHYPREA      = (HYPRE_ParCSRMatrix) hypreA;
   HYPRE_ParCSRPCGCreate(comm, &lobHYPRESolver);
   HYPRE_ParCSRPCGSetMaxIter(lobHYPRESolver, 10);
   HYPRE_ParCSRPCGSetTol(lobHYPRESolver, 1.0e-1);
   HYPRE_ParCSRPCGSetup(lobHYPRESolver, lobHYPREA, 
          (HYPRE_ParVector) lobVecs[0], (HYPRE_ParVector) lobVecs[1]);
   HYPRE_ParCSRPCGSetPrecond(lobHYPRESolver, HYPRE_ParCSRDiagScale,
                             HYPRE_ParCSRDiagScaleSetup, HYPrecon);
   HYPRE_LobpcgSetTolerance(lobpcgdata, 1.0e-1);

   HYPRE_LobpcgSolve(lobpcgdata,Func_Matvec,(HYPRE_ParVector*)lobVecs,&eigval);
   for ( iV = 0; iV < nullspace_dim; iV++ )
   {
      uData = hypre_VectorData(
                 hypre_ParVectorLocalVector((hypre_ParVector *)lobVecs[iV]));
      offset = nullspace_len * iV;
      for ( i = 0; i < nullspace_len; i++ )
         nullspace_vec[offset+i] = uData[i];
      hypre_ParVectorDestroy(lobVecs[iV]);
   }
   HYPRE_LobpcgDestroy(lobpcgdata);
#endif
   return 0;
}

