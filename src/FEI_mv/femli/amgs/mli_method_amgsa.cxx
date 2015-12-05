/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.36 $
 ***********************************************************************EHEADER*/





#ifdef WIN32
#define strcmp _stricmp
#endif

#include <string.h>
#include <assert.h>
#include "HYPRE.h"
#include "../../IJ_mv/HYPRE_IJ_mv.h"
#include "util/mli_utils.h"
#include "matrix/mli_matrix.h"
#include "matrix/mli_matrix_misc.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"
#include "amgs/mli_method_amgsa.h"

#define MABS(x) (((x) > 0) ? (x) : -(x))
/* ********************************************************************* *
 * functions external to MLI 
 * --------------------------------------------------------------------- */

#ifdef MLI_ARPACK
extern "C"
{
   /* ARPACK function to compute eigenvalues/eigenvectors */
   void dnstev_(int *n, int *nev, char *which, double *sigmar, 
                double *sigmai, int *colptr, int *rowind, double *nzvals, 
                double *dr, double *di, double *z, int *ldz, int *info,
                double *tol);
}
#endif

/* ********************************************************************* *
 * constructor
 * --------------------------------------------------------------------- */

MLI_Method_AMGSA::MLI_Method_AMGSA( MPI_Comm comm ) : MLI_Method( comm )
{
   char name[100];

   strcpy(name, "AMGSA");
   setName( name );
   setID( MLI_METHOD_AMGSA_ID );
   maxLevels_     = 40;
   numLevels_     = 40;
   currLevel_     = 0;
   outputLevel_   = 0;
   scalar_        = 0;
   nodeDofs_      = 1;
   currNodeDofs_  = 1;
   threshold_     = 0.0;
   nullspaceDim_  = 1;
   nullspaceVec_  = NULL;
   nullspaceLen_  = 0;
   numSmoothVec_  = 0;              /* smooth vectors instead of null vectors */
   numSmoothVecSteps_ = 0;
   Pweight_       = 0.0;
   SPLevel_       = 0;
   dropTolForP_   = 0.0;            /* tolerance to sparsify P*/
   saCounts_      = new int[40];    /* number of aggregates   */
   saData_        = new int*[40];   /* node to aggregate data */
   saDataAux_     = NULL;
   spectralNorms_ = new double[40]; /* calculated max eigen   */
   for ( int i = 0; i < 40; i++ ) 
   {
      saCounts_[i] = 0;
      saData_[i]   = NULL;
      spectralNorms_[i] = 0.0;
   }
   calcNormScheme_ = 0;              /* use matrix rowsum norm */
   minCoarseSize_  = 3000;           /* smallest coarse grid   */
   minAggrSize_    = 3;              /* smallest aggregate size */
   coarsenScheme_  = MLI_METHOD_AMGSA_LOCAL;
   strcpy(preSmoother_, "HSGS");
   strcpy(postSmoother_, "HSGS");
   preSmootherNum_  = 2;
   postSmootherNum_  = 2;
   preSmootherWgt_  = new double[2];
   postSmootherWgt_  = new double[2];
   preSmootherWgt_[0] = preSmootherWgt_[1] = 1.0;
   postSmootherWgt_[0] = postSmootherWgt_[1] = 1.0;
   smootherPrintRNorm_ = 0;
   smootherFindOmega_  = 0;
   strcpy(coarseSolver_, "SuperLU");
   coarseSolverNum_    = 0;
   coarseSolverWgt_    = NULL;
   calibrationSize_    = 0;
   useSAMGeFlag_       = 0;
   RAPTime_            = 0.0;
   totalTime_          = 0.0;
   ddObj_              = NULL;
   ARPACKSuperLUExists_ = 0;
   saLabels_            = NULL;
   useSAMGDDFlag_       = 0;
   printToFile_         = 0;
   strcpy( paramFile_, "empty" );
   symmetric_           = 1;
   arpackTol_           = 1.0e-10;
}

/* ********************************************************************* *
 * destructor
 * --------------------------------------------------------------------- */

MLI_Method_AMGSA::~MLI_Method_AMGSA()
{
   char paramString[20];

   if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   if ( saDataAux_ != NULL )
   {
      for ( int j = 0; j < saCounts_[0]; j++ )
         if ( saDataAux_[j] != NULL ) delete [] saDataAux_[j];
      delete [] saDataAux_;
   }
   if ( saCounts_ != NULL ) delete [] saCounts_;
   if ( saData_ != NULL )
   {
      for ( int i = 0; i < maxLevels_; i++ )
      {
         if ( saData_[i] != NULL ) delete [] saData_[i];
         else break;
      }
      delete [] saData_;
      saData_ = NULL;
   }
   if ( saLabels_ != NULL )
   {
      for ( int k = 0; k < maxLevels_; k++ )
      {
         if ( saLabels_[k] != NULL ) delete [] saLabels_[k];
         else break;
      }
      delete [] saLabels_;
      saLabels_ = NULL;
   }
   if ( spectralNorms_   != NULL ) delete [] spectralNorms_;
   if ( preSmootherWgt_  != NULL ) delete [] preSmootherWgt_;
   if ( postSmootherWgt_ != NULL ) delete [] postSmootherWgt_;
   if ( coarseSolverWgt_ != NULL ) delete [] coarseSolverWgt_ ;
   if ( ddObj_!= NULL ) 
   {
      if ( ddObj_->sendProcs != NULL ) delete [] ddObj_->sendProcs;
      if ( ddObj_->recvProcs != NULL ) delete [] ddObj_->recvProcs;
      if ( ddObj_->sendLengs != NULL ) delete [] ddObj_->sendLengs;
      if ( ddObj_->recvLengs != NULL ) delete [] ddObj_->recvLengs;
      if ( ddObj_->sendMap   != NULL ) delete [] ddObj_->sendMap;
      if ( ddObj_->ANodeEqnList != NULL ) delete [] ddObj_->ANodeEqnList;
      if ( ddObj_->SNodeEqnList != NULL ) delete [] ddObj_->SNodeEqnList;
      delete ddObj_;
   }
   if ( ARPACKSuperLUExists_ ) 
   {
      strcpy( paramString, "destroy" );
#ifdef MLI_ARPACK
      int  info;
      dnstev_(NULL, NULL, paramString, NULL, NULL, NULL, NULL, NULL, NULL, 
              NULL, NULL, NULL, &info, &arpackTol_);
#endif
   }
}

/* ********************************************************************* *
 * set parameters
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setParams(char *in_name, int argc, char *argv[])
{
   int        level, size, nDOF, numNS, length, nSweeps=1, offset, nsDim;
   int        prePost, nnodes, nAggr, *aggrInfo, *labels, is, *indices;
   int        mypid;
   double     thresh, pweight, *nullspace, *weights=NULL, *coords, *scales;
   double     *nsAdjust;
   char       param1[256], param2[256], *param3;
   MPI_Comm   comm;

   comm = getComm();
   MPI_Comm_rank( comm, &mypid );
   sscanf(in_name, "%s", param1);
   if ( outputLevel_ > 1 && mypid == 0 ) 
      printf("\tMLI_Method_AMGSA::setParam = %s\n", in_name);
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
   else if ( !strcmp(param1, "useSAMGe" ))
   {
      useSAMGeFlag_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "useSAMGDD" ))
   {
      useSAMGDDFlag_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "useSAMGDDExt" ))
   {
      useSAMGDDFlag_ = 2;
      return 0;
   }
   else if ( !strcmp(param1, "useSAMGDDExt2" ))
   {
      useSAMGDDFlag_ = 3;
      return 0;
   }
   else if ( !strcmp(param1, "setCoarsenScheme" ))
   {
      sscanf(in_name,"%s %s", param1, param2);
      if ( !strcmp(param2, "local" ) )
         return ( setCoarsenScheme( MLI_METHOD_AMGSA_LOCAL ) );
      else if ( !strcmp(param2, "hybrid" ) )
         return ( setCoarsenScheme( MLI_METHOD_AMGSA_HYBRID ) );
      else      
      {
         printf("MLI_Method_AMGSA::setParams ERROR : setCoarsenScheme not");
         printf(" valid.  Valid options are : local \n");
         return 1;
      }
   }
   else if ( !strcmp(param1, "setMinCoarseSize" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setMinCoarseSize( size ) );
   }
   else if ( !strcmp(param1, "setMinAggrSize" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setMinAggregateSize( size ) );
   }
   else if ( !strcmp(param1, "setStrengthThreshold" ))
   {
      sscanf(in_name,"%s %lg", param1, &thresh);
      return ( setStrengthThreshold( thresh ) );
   }
   else if ( !strcmp(param1, "setSmoothVec" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setSmoothVec( size ) );
   }
   else if ( !strcmp(param1, "setSmoothVecSteps" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setSmoothVecSteps( size ) );
   }
   else if ( !strcmp(param1, "setPweight" ))
   {
      sscanf(in_name,"%s %lg", param1, &pweight);
      return ( setPweight( pweight ) );
   }
   else if ( !strcmp(param1, "setSPLevel" ))
   {
      sscanf(in_name,"%s %d", param1, &level);
      return ( setSPLevel( level ) );
   }
   else if ( !strcmp(param1, "setCalcSpectralNorm" ))
   {
      return ( setCalcSpectralNorm() );
   }
   else if ( !strcmp(param1, "useNonsymmetric" ))
   {
      symmetric_ = 0;
      return ( 0 );
   }
   else if ( !strcmp(param1, "setAggregateInfo" ))
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
   else if ( !strcmp(param1, "setCalibrationSize" ))
   {
      sscanf(in_name,"%s %d", param1, &size);
      return ( setCalibrationSize( size ) );
   }
   else if ( !strcmp(param1, "setPreSmoother" ))
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
   else if ( !strcmp(param1, "setPostSmoother" ))
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
   else if ( !strcmp(param1, "setSmootherPrintRNorm" ))
   {
      smootherPrintRNorm_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "setSmootherFindOmega" ))
   {
      smootherFindOmega_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "setCoarseSolver" ))
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
   else if ( !strcmp(param1, "setNullSpace" ))
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
   else if ( !strcmp(param1, "adjustNullSpace" ))
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
   else if ( !strcmp(param1, "resetNullSpaceComponents" ))
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
   else if ( !strcmp(param1, "setNodalCoord" ))
   {
      if ( argc != 5 && argc != 6 )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setNodalCoord needs");
         printf(" 4 arguments.\n");
         printf("     argument[0] : number of nodes \n");
         printf("     argument[1] : node degree of freedom \n");
         printf("     argument[2] : number of space dimension\n");
         printf("     argument[3] : coordinate information \n");
         printf("     argument[4] : null space dimension \n");
         printf("     argument[5] : scalings (can be null) \n");
         return 1;
      } 
      nnodes = *(int *)   argv[0];
      nDOF   = *(int *)   argv[1];
      nsDim  = *(int *)   argv[2];
      coords = (double *) argv[3];
      numNS  = *(int *)   argv[4];
      if ( argc == 6 ) scales = (double *) argv[5]; else scales = NULL;
      return ( setNodalCoordinates(nnodes,nDOF,nsDim,coords,numNS,scales) );
   }
   else if ( !strcmp(param1, "setLabels" ))
   {
      if ( argc != 3 )
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
      if ( saLabels_ == NULL ) 
      {
         saLabels_ = new int*[maxLevels_];
         for ( is = 0; is < maxLevels_; is++ ) saLabels_[is] = NULL;
      }
      if ( level < 0 || level >= maxLevels_ )
      {
         printf("MLI_Method_AMGSA::setParams ERROR - setLabels has \n");
         printf("invalid level number = %d (%d)\n", level, maxLevels_);
         return 1;
      }
      if ( saLabels_[level] != NULL ) delete [] saLabels_[level];
      saLabels_[level] = new int[length];
      for ( is = 0; is < length; is++ ) saLabels_[level][is] = labels[is];
      return 0;
   }
   else if ( !strcmp(param1, "scalar" ))
   {
      scalar_ = 1;
   }
   else if ( !strcmp(param1, "setParamFile" ))
   {
      param3 = (char *) argv[0];
      strcpy( paramFile_, param3 ); 
      return 0;
   }
   else if ( !strcmp(param1, "printNodalCoord" ))
   {
      printToFile_ |= 2;
      return 0;
   }
   else if ( !strcmp(param1, "printNullSpace" ))
   {
      printToFile_ |= 4;
      return 0;
   }
   else if ( !strcmp(param1, "printElemNodeList" ))
   {
      printToFile_ |= 8;
      return 0;
   }
   else if ( !strcmp(param1, "print" ))
   {
      return ( print() );
   }
   else if ( !strcmp(param1, "arpackTol" ))
   {
      sscanf(in_name, "%s %lg", param1, &arpackTol_);
      if ( arpackTol_ <= 1.0e-10 ) arpackTol_ = 1.0e-10;
      if ( arpackTol_ >  1.0e-1  ) arpackTol_ = 1.0e-1;
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

   if ( !strcmp(in_name, "getNullSpace" ))
   {
      if ( (*argc) < 4 )
      {
         printf("MLI_Method_AMGSA::getParams ERROR - getNullSpace needs");
         printf(" 4 arguments.\n");
         exit(1);
      }
      getNullSpace(nodeDofs_,numNS,nullspace,length);
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
   int             level, mypid, nRows, nullspaceDimKeep, ii, jj;
   double          startTime, elapsedTime, maxEigen, maxEigenT, dtemp=0.0;
   char            paramString[100], *targv[10];
   MLI_Matrix      *mli_Pmat, *mli_Rmat, *mli_Amat, *mli_ATmat, *mli_cAmat;
   MLI_Solver      *smootherPtr, *csolvePtr;
   MPI_Comm        comm;
   MLI_Function    *funcPtr;
   hypre_ParCSRMatrix *hypreRT;
   MLI_FEData      *fedata;
   MLI_SFEI        *sfei;

#define DEBUG
#ifdef DEBUG
   int                *partition, ANRows, AStart, AEnd; 
   double             *XData, rnorm;
   HYPRE_IJVector     IJX, IJY;
   hypre_ParCSRMatrix *hypreA;
   hypre_ParVector    *hypreX, *hypreY;

   if (nullspaceVec_ != NULL)
   {
      mli_Amat = mli->getSystemMatrix(0);
      hypreA = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
      comm = hypre_ParCSRMatrixComm(hypreA);
      MPI_Comm_rank(comm,&mypid);
      HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix) hypreA,&partition);
      AStart = partition[mypid];
      AEnd = partition[mypid+1];
      ANRows = AEnd - AStart;
      free(partition);
      HYPRE_IJVectorCreate(comm, AStart, AEnd-1,&IJX);
      HYPRE_IJVectorSetObjectType(IJX, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(IJX);
      HYPRE_IJVectorAssemble(IJX);
      HYPRE_IJVectorGetObject(IJX, (void **) &hypreX);
      HYPRE_IJVectorCreate(comm, AStart, AEnd-1,&IJY);
      HYPRE_IJVectorSetObjectType(IJY, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(IJY);
      HYPRE_IJVectorAssemble(IJY);
      HYPRE_IJVectorGetObject(IJY, (void **) &hypreY);
      XData = (double *) hypre_VectorData(hypre_ParVectorLocalVector(hypreX));
      for (ii = 0; ii < nullspaceDim_; ii++)
      {
         for (jj = 0; jj < ANRows; jj++) XData[jj] = nullspaceVec_[ii*ANRows+jj];
         hypre_ParCSRMatrixMatvec(1.0, hypreA, hypreX, 0.0, hypreY);
         rnorm = sqrt(hypre_ParVectorInnerProd(hypreY, hypreY));
         if (mypid == 0) printf("HYPRE FEI: check null space = %e\n", rnorm);
      }
      HYPRE_IJVectorDestroy(IJX);
      HYPRE_IJVectorDestroy(IJY);
   }
   else
   {
      printf("MLI::setup - no nullspace vector.\n");
   }
#endif

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGSA::setup begins...\n");
#endif

   /* --------------------------------------------------------------- */
   /* if using the extended DD method, needs special processing       */
   /* --------------------------------------------------------------- */

#if 1
   if (useSAMGDDFlag_ == 2) 
   {
      return(setupExtendedDomainDecomp(mli));
   }
   if (useSAMGDDFlag_ == 3) 
   {
      return(setupExtendedDomainDecomp2(mli));
   }
#endif

   /* --------------------------------------------------------------- */
   /* clean up some mess made previously for levels other than the    */
   /* finest level                                                    */
   /* --------------------------------------------------------------- */

   if (saData_ != NULL)
   {
      for (level = 1; level < maxLevels_; level++)
      {
         if (saData_[level] != NULL) delete [] saData_[level];
         saData_[level] = NULL;
      }
   }
   nullspaceDimKeep = nullspaceDim_;

   /* --------------------------------------------------------------- */
   /* if requested, compute null spaces from finite element stiffness */
   /* matrices                                                        */
   /* --------------------------------------------------------------- */

   if (useSAMGeFlag_)  
   {
      level = 0;
      fedata = mli->getFEData( level );
      if (fedata != NULL) setupFEDataBasedNullSpaces(mli);
      else
      {
         sfei = mli->getSFEI( level );
         if (sfei != NULL) setupSFEIBasedNullSpaces(mli);
         else              useSAMGeFlag_ = 0;
      }
   }

   /* --------------------------------------------------------------- */
   /* if domain decomposition requested, compute aggregate based on   */
   /* subdomains                                                      */
   /* --------------------------------------------------------------- */

   if (useSAMGDDFlag_ == 1) 
   {
      level = 0;
      fedata = mli->getFEData( level );
      if (fedata != NULL) setupFEDataBasedAggregates(mli);
      else
      {
         sfei = mli->getSFEI( level );
         if (sfei != NULL) setupSFEIBasedAggregates(mli);
         else              useSAMGDDFlag_ = 0;
      }
   }

   /* --------------------------------------------------------------- */
   /* update nullspace dimension                                      */
   /* --------------------------------------------------------------- */

   if ( nullspaceVec_ != NULL )
   {
      for ( ii = nullspaceDim_-1; ii >= 0; ii-- )
      {
         dtemp = 0.0;
         for ( jj = 0; jj < nullspaceLen_; jj++ )
            dtemp += MABS(nullspaceVec_[ii*nullspaceLen_+jj]);
         if (dtemp != 0.0) break;
      }
      nullspaceDim_ = nullspaceDim_ + ii - nullspaceDim_ + 1;
   }

   /* --------------------------------------------------------------- */
   /* call calibration if calibration size > 0                        */
   /* --------------------------------------------------------------- */

   if (calibrationSize_ > 0) return(setupCalibration(mli));
      
   /* --------------------------------------------------------------- */
   /* if no null spaces have been provided nor computed, set null     */
   /* space dimension equal to node degree of freedom                 */
   /* --------------------------------------------------------------- */

   if (nullspaceDim_ != nodeDofs_ && nullspaceVec_ == NULL 
       && numSmoothVec_ == 0)
      nullspaceDim_ = nodeDofs_;

   /* --------------------------------------------------------------- */
   /* traverse all levels                                             */
   /* --------------------------------------------------------------- */

   RAPTime_ = 0.0;
   level    = 0;
   comm     = getComm();
   MPI_Comm_rank( comm, &mypid );
   mli_Amat   = mli->getSystemMatrix(level);
   totalTime_ = MLI_Utils_WTime();

#if HAVE_LOBPCG
   relaxNullSpaces(mli_Amat);
#endif
   
   for (level = 0; level < numLevels_; level++ )
   {
      if (mypid == 0 && outputLevel_ > 0)
      {
         printf("\t*****************************************************\n");
         printf("\t*** Aggregation (uncoupled) : level = %d\n", level);
         printf("\t-----------------------------------------------------\n");
      }
      currLevel_ = level;
      if (level == numLevels_-1) break;

      /* -------------------------------------------------- */
      /* fetch fine grid matrix                             */
      /* -------------------------------------------------- */

      mli_Amat = mli->getSystemMatrix(level);
      assert (mli_Amat != NULL);

      /* -------------------------------------------------- */
      /* perform coarsening                                 */
      /* -------------------------------------------------- */

      switch (coarsenScheme_)
      {
         case MLI_METHOD_AMGSA_LOCAL :
              if (level == 0)
                 maxEigen = genP(mli_Amat,&mli_Pmat,saCounts_[0],saData_[0]); 
              else
                 maxEigen = genP(mli_Amat, &mli_Pmat, 0, NULL); 
              break;

         case MLI_METHOD_AMGSA_HYBRID :
              if (level == 0)
                 maxEigen = genP(mli_Amat,&mli_Pmat,saCounts_[0],saData_[0]); 
              else
                 maxEigen = genP(mli_Amat, &mli_Pmat, 0, NULL); 
              break;
      }
      if (maxEigen != 0.0) spectralNorms_[level] = maxEigen;
      if (mli_Pmat == NULL) break;
      startTime = MLI_Utils_WTime();

      /* -------------------------------------------------- */
      /* construct and set the coarse grid matrix           */
      /* -------------------------------------------------- */

      if (mypid == 0 && outputLevel_ > 0) printf("\tComputing RAP\n");
      MLI_Matrix_ComputePtAP(mli_Pmat, mli_Amat, &mli_cAmat);
      mli->setSystemMatrix(level+1, mli_cAmat);
      elapsedTime = (MLI_Utils_WTime() - startTime);
      RAPTime_ += elapsedTime;
      if (mypid == 0 && outputLevel_ > 0) 
         printf("\tRAP computed, time = %e seconds.\n", elapsedTime);

#if 0
      mli_Amat->print("Amat");
      mli_Pmat->print("Pmat");
      mli_cAmat->print("cAmat");
#endif

      /* -------------------------------------------------- */
      /* set the prolongation matrix                        */
      /* -------------------------------------------------- */

      mli->setProlongation(level+1, mli_Pmat);

      /* -------------------------------------------------- */
      /* if nonsymmetric, generate a different R            */
      /* then set restriction operator                      */
      /* -------------------------------------------------- */

      if (symmetric_ == 0 && Pweight_ == 0.0)
      {
         MLI_Matrix_Transpose(mli_Amat, &mli_ATmat);
         switch (coarsenScheme_)
         {
            case MLI_METHOD_AMGSA_LOCAL :
                 maxEigenT = genP(mli_ATmat, &mli_Rmat, saCounts_[level], 
                                  saData_[level]); 
                 if ( maxEigenT < 0.0 ) 
                    printf("MLI_Method_AMGSA::setup ERROR : maxEigenT < 0.\n");
                 break;

            case MLI_METHOD_AMGSA_HYBRID :
                 maxEigenT = genP(mli_ATmat, &mli_Rmat, saCounts_[level], 
                                  saData_[level]); 
                 if ( maxEigenT < 0.0 ) 
                    printf("MLI_Method_AMGSA::setup ERROR : maxEigenT < 0.\n");
                 break;
         }
         delete mli_ATmat;
         hypreRT = (hypre_ParCSRMatrix *) mli_Rmat->takeMatrix();
         delete mli_Rmat;
         sprintf(paramString, "HYPRE_ParCSRT");
         funcPtr = new MLI_Function();
         MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
         sprintf(paramString, "HYPRE_ParCSRT" ); 
         mli_Rmat = new MLI_Matrix( (void *) hypreRT, paramString, funcPtr );
         delete funcPtr;
      }
      else
      {
         sprintf(paramString, "HYPRE_ParCSRT");
         mli_Rmat = new MLI_Matrix(mli_Pmat->getMatrix(), paramString, NULL);
      }
      mli->setRestriction(level, mli_Rmat);

      /* -------------------------------------------------- */
      /* if a global coarsening step has been called, this  */
      /* is the coarsest grid. So quit.                     */
      /* -------------------------------------------------- */

      if (spectralNorms_[level] == 1.0e39) 
      {
         spectralNorms_[level] = 0.0;
         level++;
         currLevel_ = level;
         break;
      }

      /* -------------------------------------------------- */
      /* set the smoothers                                  */
      /* (if domain decomposition and ARPACKA SuperLU       */
      /* smoothers is requested, perform special treatment, */
      /* and if domain decomposition and SuperLU smoother   */
      /* is requested with multiple local subdomains, again */
      /* perform special treatment.)                        */
      /* -------------------------------------------------- */

      if ( useSAMGDDFlag_ == 1 && numLevels_ == 2 && 
           !strcmp(preSmoother_, "ARPACKSuperLU") )
      {
         setupFEDataBasedSuperLUSmoother(mli, level);
         smootherPtr = MLI_Solver_CreateFromName(preSmoother_);
         targv[0] = (char *) ddObj_;
         sprintf( paramString, "ARPACKSuperLUObject" );
         smootherPtr->setParams(paramString, 1, targv);
         smootherPtr->setup(mli_Amat);
         mli->setSmoother( level, MLI_SMOOTHER_PRE, smootherPtr );
#if 0
         smootherPtr = MLI_Solver_CreateFromName(preSmoother_);
         smootherPtr->setParams(paramString, 1, targv);
         smootherPtr->setup(mli_Amat);
         mli->setSmoother( level, MLI_SMOOTHER_POST, smootherPtr );
#endif
         continue;
      }
      else if ( useSAMGDDFlag_ == 1 && numLevels_ == 2 && 
                !strcmp(preSmoother_, "SeqSuperLU") && saDataAux_ != NULL)
      {
         smootherPtr = MLI_Solver_CreateFromName(preSmoother_);
         sprintf( paramString, "setSubProblems" );
         targv[0] = (char *) &(saDataAux_[0][0]); 
         targv[1] = (char *) &(saDataAux_[0][1]); 
         targv[2] = (char *) &(saDataAux_[1]); 
         smootherPtr->setParams(paramString, 3, targv);
         smootherPtr->setup(mli_Amat);
         mli->setSmoother(level, MLI_SMOOTHER_PRE, smootherPtr);
         mli->setSmoother(level, MLI_SMOOTHER_POST, NULL);
      }
      else
      {
         smootherPtr = MLI_Solver_CreateFromName( preSmoother_ );
         targv[0] = (char *) &preSmootherNum_;
         targv[1] = (char *) preSmootherWgt_;
         sprintf( paramString, "relaxWeight" );
         smootherPtr->setParams(paramString, 2, targv);
         if ( !strcmp(preSmoother_, "Jacobi") ) 
         {
            sprintf( paramString, "setModifiedDiag" );
            smootherPtr->setParams(paramString, 0, NULL);
         }
         if ( !strcmp(preSmoother_, "MLS") ) 
         {
            sprintf( paramString, "maxEigen" );
            targv[0] = (char *) &maxEigen;
            smootherPtr->setParams(paramString, 1, targv);
         }
         if ( smootherPrintRNorm_ == 1 )
         {
            sprintf( paramString, "printRNorm" );
            smootherPtr->setParams(paramString, 0, NULL);
         }
         if ( smootherFindOmega_ == 1 )
         {
            sprintf( paramString, "findOmega" );
            smootherPtr->setParams(paramString, 0, NULL);
         }
         smootherPtr->setup(mli_Amat);
         mli->setSmoother( level, MLI_SMOOTHER_PRE, smootherPtr );

         if ( strcmp(preSmoother_, postSmoother_) )
         {
            smootherPtr = MLI_Solver_CreateFromName( postSmoother_ );
            targv[0] = (char *) &postSmootherNum_;
            targv[1] = (char *) postSmootherWgt_;
            sprintf( paramString, "relaxWeight" );
            smootherPtr->setParams(paramString, 2, targv);
            if ( !strcmp(postSmoother_, "MLS") ) 
            {
               sprintf( paramString, "maxEigen" );
               targv[0] = (char *) &maxEigen;
               smootherPtr->setParams(paramString, 1, targv);
            }
            if ( smootherPrintRNorm_ == 1 )
            {
               sprintf( paramString, "printRNorm" );
               smootherPtr->setParams(paramString, 0, NULL);
            }
            if ( smootherFindOmega_ == 1 )
            {
               sprintf( paramString, "findOmega" );
               smootherPtr->setParams(paramString, 0, NULL);
            }
            smootherPtr->setup(mli_Amat);
         }
         mli->setSmoother( level, MLI_SMOOTHER_POST, smootherPtr );
      }
   }

   /* --------------------------------------------------------------- */
   /* set the coarse grid solver                                      */
   /* --------------------------------------------------------------- */

   if (mypid == 0 && outputLevel_ > 0) printf("\tCoarse level = %d\n",level);
   mli_Amat = mli->getSystemMatrix(level);
   strcpy(paramString, "nrows");
   mli_Amat->getMatrixInfo(paramString, nRows, dtemp);
   if (nRows > 10000)
   {
      if ( outputLevel_ > 1 && mypid == 0 )
         printf("ML_Method_AMGSA::message - nCoarse too large => GMRESSGS.\n"); 
      strcpy(coarseSolver_, "GMRESSGS");
      csolvePtr = MLI_Solver_CreateFromName( coarseSolver_ );
      sprintf(paramString, "maxIterations %d", coarseSolverNum_);
      csolvePtr->setParams(paramString, 0, NULL);
   }
   else
   {
      csolvePtr = MLI_Solver_CreateFromName( coarseSolver_ );
      if (strcmp(coarseSolver_, "SuperLU"))
      {
         targv[0] = (char *) &coarseSolverNum_;
         targv[1] = (char *) coarseSolverWgt_ ;
         sprintf( paramString, "relaxWeight" );
         csolvePtr->setParams(paramString, 2, targv);
         if (!strcmp(coarseSolver_, "MLS"))
         {
            sprintf(paramString, "maxEigen");
            targv[0] = (char *) &maxEigen;
            csolvePtr->setParams(paramString, 1, targv);
         }
      }
   }
   mli_Amat = mli->getSystemMatrix(level);
   csolvePtr->setup(mli_Amat);
   mli->setCoarseSolve(csolvePtr);
   totalTime_ = MLI_Utils_WTime() - totalTime_;

   /* --------------------------------------------------------------- */
   /* return the coarsest grid level number                           */
   /* --------------------------------------------------------------- */

   if ( outputLevel_ >= 2 ) printStatistics(mli);
   nullspaceDim_ = nullspaceDimKeep;

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
   outputLevel_ = level;
   return 0;
}

/* ********************************************************************* *
 * set number of levels 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setNumLevels( int nlevels )
{
   if ( nlevels < maxLevels_ && nlevels > 0 ) numLevels_ = nlevels;
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
      strcpy( preSmoother_, stype );
      if ( num > 0 ) preSmootherNum_ = num; else preSmootherNum_ = 1;
      delete [] preSmootherWgt_;
      preSmootherWgt_ = new double[preSmootherNum_];
      if ( wgt == NULL )
         for (i = 0; i < preSmootherNum_; i++) preSmootherWgt_[i] = 1.0;
      else
         for (i = 0; i < preSmootherNum_; i++) preSmootherWgt_[i] = wgt[i];
   }
   if ( prePost == MLI_SMOOTHER_POST || prePost == MLI_SMOOTHER_BOTH )
   {
      strcpy( postSmoother_, stype );
      if ( num > 0 ) postSmootherNum_ = num; else postSmootherNum_ = 1;
      delete [] postSmootherWgt_;
      postSmootherWgt_ = new double[postSmootherNum_];
      if ( wgt == NULL )
         for (i = 0; i < postSmootherNum_; i++) postSmootherWgt_[i] = 1.0;
      else
         for (i = 0; i < postSmootherNum_; i++) postSmootherWgt_[i] = wgt[i];
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

   strcpy( coarseSolver_, stype );
   if ( num > 0 ) coarseSolverNum_ = num; else coarseSolverNum_ = 1;
   delete [] coarseSolverWgt_ ;
   if ( wgt != NULL && strcmp(coarseSolver_, "SuperLU") )
   {
      coarseSolverWgt_  = new double[coarseSolverNum_]; 
      for (i = 0; i < coarseSolverNum_; i++) coarseSolverWgt_ [i] = wgt[i];
   }
   else coarseSolverWgt_  = NULL;
   return 0;
}

/* ********************************************************************* *
 * set coarsening scheme 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setCoarsenScheme( int scheme )
{
   if ( scheme == MLI_METHOD_AMGSA_LOCAL ) 
   {
      coarsenScheme_ = MLI_METHOD_AMGSA_LOCAL;
      return 0;
   }
   else if ( scheme == MLI_METHOD_AMGSA_HYBRID ) 
   {
      coarsenScheme_ = MLI_METHOD_AMGSA_HYBRID;
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

int MLI_Method_AMGSA::setMinCoarseSize( int coarseSize )
{
   if ( coarseSize > 0 ) minCoarseSize_ = coarseSize;
   return 0;
}

/* ********************************************************************* *
 * set minimum aggregate size
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setMinAggregateSize( int aggrSize )
{
   if ( aggrSize > 0 ) minAggrSize_ = aggrSize;
   return 0;
}

/* ********************************************************************* *
 * set coarsening threshold
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setStrengthThreshold( double thresh )
{
   if ( thresh > 0.0 ) threshold_ = thresh;
   else                threshold_ = 0.0;
   return 0;
}

/* ********************************************************************* *
 * set number of smooth vectors
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setSmoothVec( int num )
{
   if ( num >= 0 ) numSmoothVec_ = num;
   return 0;
}

/* ********************************************************************* *
 * set number of steps for generating smooth vectors
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setSmoothVecSteps( int num )
{
   if ( num >= 0 ) numSmoothVecSteps_ = num;
   return 0;
}

/* ********************************************************************* *
 * set damping factor for smoother prolongator
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setPweight( double weight )
{
   if ( weight >= 0.0 && weight <= 2.0 ) Pweight_ = weight;
   return 0;
}

/* ********************************************************************* *
 * set starting level for smoother prolongator
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setSPLevel( int level )
{
   if ( level > 0 ) SPLevel_ = level;
   return 0;
}

/* ********************************************************************* *
 * indicate spectral norm is to be calculated
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setCalcSpectralNorm()
{
   calcNormScheme_ = 1;
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
   saCounts_[level] = aggrCnt;
   if ( saData_[level] != NULL ) delete [] saData_[level];
   saData_[level] = new int[length];
   for ( int i = 0; i < length; i++ ) saData_[level][i] = aggrInfo[i];
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
   nodeDofs_     = nDOF;
   currNodeDofs_ = nDOF;
   nullspaceDim_ = ndim;
   nullspaceLen_ = length;
   if ( nullspaceVec_ != NULL ) delete [] nullspaceVec_;
   if ( nullvec != NULL )
   {
      nullspaceVec_ = new double[length * ndim];
      for ( int i = 0; i < length*ndim; i++ )
         nullspaceVec_[i] = nullvec[i];
   }
   else nullspaceVec_ = NULL;
   return 0;
}

/* ********************************************************************* *
 * adjust null space vectors
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::adjustNullSpace(double *vecAdjust)
{
   int i;

   if ( useSAMGeFlag_ ) return 0;

   for ( i = 0; i < nullspaceLen_*nullspaceDim_; i++ )
      nullspaceVec_[i] += vecAdjust[i];

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
      for ( j = 0; j < nullspaceDim_; j++ )
         nullspaceVec_[j*nullspaceLen_+index] = 0.;
   }
   return 0;
}

/* ********************************************************************* *
 * load nodal coordinates (translates into rigid body modes)
 * (abridged from similar function in ML)
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setNodalCoordinates(int num_nodes,int nDOF,int nsDim, 
                         double *coords, int numNS, double *scalings)
{
   int  i, j, k, offset, voffset, mypid;
   char fname[100];
   FILE *fp;

   MPI_Comm comm = getComm();
   MPI_Comm_rank( comm, &mypid );

   if ( nDOF == 1 )
   {
      nodeDofs_     = 1;
      currNodeDofs_ = 1;
      nullspaceLen_ = num_nodes;
      nullspaceDim_ = numNS;
      if (useSAMGeFlag_ == 0 && numNS != 1) nullspaceDim_ = 1;
   }
   else if ( nDOF == 3 )
   {
      nodeDofs_     = 3;
      currNodeDofs_ = 3;
      nullspaceLen_ = num_nodes * 3;
      nullspaceDim_ = numNS;
      if (useSAMGeFlag_ == 0 && numNS != 6 && numNS != 12 && numNS != 21)
         nullspaceDim_ = 6;
   }
   else
   {
      printf("setNodalCoordinates: nDOF = %d not supported\n",nDOF);
      exit(1);
   }
   if (nullspaceVec_ != NULL) delete [] nullspaceVec_;

   if ((printToFile_ & 2) != 0 && nodeDofs_ == 3 )
   {
      sprintf(fname, "nodalCoord.%d", mypid); 
      fp = fopen(fname, "w");
      fprintf(fp, "%d\n", num_nodes);
      for ( i = 0 ; i < num_nodes; i++ ) 
      {
         for ( j = 0 ; j < nodeDofs_; j++ ) 
            fprintf(fp," %25.16e", coords[i*nodeDofs_+j]);
         fprintf(fp,"\n");
      }
      fclose(fp);
   }

   nullspaceVec_ = new double[nullspaceLen_ * nullspaceDim_];
   for( i = 0 ; i < nullspaceLen_*nullspaceDim_; i++ ) nullspaceVec_[i] = 0.0;

   for( i = 0 ; i < num_nodes; i++ ) 
   {
      if ( nodeDofs_ == 1 )
      {
         for( k = 0; k < nsDim; k++ )
            nullspaceVec_[k*nullspaceLen_+i] = 0.0;
         nullspaceVec_[i] = 1.0;
         if ( nullspaceDim_ == 4 ) 
         {
            for( k = 0; k < nsDim; k++ )
               nullspaceVec_[(k+1)*nullspaceLen_+i] = coords[i*nsDim+k];
         }
      }
      else if ( nodeDofs_ == 3 )
      {
         if ( nullspaceDim_ == 6 ) 
         {
            voffset = i * nodeDofs_;
            for ( j = 0; j < 3; j++ )
            {
               for( k = 0; k < 3; k++ )
               {
                  offset = k * nullspaceLen_ + voffset + j;
                  if ( j == k ) nullspaceVec_[offset] = 1.0;
                  else          nullspaceVec_[offset] = 0.0;
               }
            }
            for ( j = 0; j < 3; j++ )
            {
               for ( k = 3; k < 6; k++ )
               {
                  offset = k * nullspaceLen_ + voffset + j;
                  if ( j == k-3 ) nullspaceVec_[offset] = 0.0;
                  else 
                  {
                     if      (j+k == 4) nullspaceVec_[offset] = coords[i*3+2];
                     else if (j+k == 5) nullspaceVec_[offset] = coords[i*3+1];
                     else if (j+k == 6) nullspaceVec_[offset] = coords[i*3];
                     else nullspaceVec_[offset] = 0.0;
                  }
               }
            }
            j = 0; k = 5; offset = k * nullspaceLen_ + voffset + j; 
            nullspaceVec_[offset] *= -1.0;
            j = 1; k = 3; offset = k * nullspaceLen_ + voffset + j; 
            nullspaceVec_[offset] *= -1.0;
            j = 2; k = 4; offset = k * nullspaceLen_ + voffset + j; 
            nullspaceVec_[offset] *= -1.0;
         }
         else if ( (nullspaceDim_ == 12 || nullspaceDim_ == 21 || nullspaceDim_ == 24) &&
                    useSAMGeFlag_ == 0 )
         {
            voffset = i * nodeDofs_;
            for ( j = 0; j < 3; j++ )
            {
               for( k = 0; k < 3; k++ )
               {
                  offset = k * nullspaceLen_ + voffset + j;
                  if ( j == k ) nullspaceVec_[offset] = 1.0;
                  else          nullspaceVec_[offset] = 0.0;
               }
               for( k = 3; k < 6; k++ )
               {
                  offset = k * nullspaceLen_ + voffset + j;
                  if ( j == (k-3) ) nullspaceVec_[offset] = coords[i*3];
                  else              nullspaceVec_[offset] = 0.0;
               }
               for( k = 6; k < 9; k++ )
               {
                  offset = k * nullspaceLen_ + voffset + j;
                  if ( j == (k-6) ) nullspaceVec_[offset] = coords[i*3+1];
                  else              nullspaceVec_[offset] = 0.0;
               }
               for( k = 9; k < 12; k++ )
               {
                  offset = k * nullspaceLen_ + voffset + j;
                  if ( j == (k-9) ) nullspaceVec_[offset] = coords[i*3+2];
                  else              nullspaceVec_[offset] = 0.0;
               }
            }
         }
         if ( (nullspaceDim_ == 21 || nullspaceDim_ == 24) && useSAMGeFlag_ == 0 )
         {
            voffset = i * nodeDofs_;
            for ( j = 0; j < 3; j++ )
            {
               for( k = 12; k < 15; k++ )
               {
                  offset = k * nullspaceLen_ + voffset + j;
                  if ( j == (k-12) )
                       nullspaceVec_[offset] = coords[i*3]*coords[i*3+1];
                  else nullspaceVec_[offset] = 0.0;
               }
               for( k = 15; k < 18; k++ )
               {
                  offset = k * nullspaceLen_ + voffset + j;
                  if ( j == (k-15) )
                       nullspaceVec_[offset] = coords[i*3+1]*coords[i*3+2];
                  else nullspaceVec_[offset] = 0.0;
               }
               for( k = 18; k < 21; k++ )
               {
                  offset = k * nullspaceLen_ + voffset + j;
                  if (j == (k-18))
                       nullspaceVec_[offset] = coords[i*3]*coords[i*3+2];
                  else nullspaceVec_[offset] = 0.0;
               }
            }
         }
         if ( nullspaceDim_ == 24 && useSAMGeFlag_ == 0 )
         {
            voffset = i * nodeDofs_;
            for ( j = 0; j < 3; j++ )
            {
               for( k = 21; k < 24; k++ )
               {
                  offset = k * nullspaceLen_ + voffset + j;
                  if (j == (k-21))
                       nullspaceVec_[offset] = coords[i*3]*coords[i*3+1]*coords[i*3+2];
                  else nullspaceVec_[offset] = 0.0;
               }
            }
         }
      }
   }
   if ( scalings != NULL )
   {
      for ( i = 0 ; i < nullspaceDim_; i++ ) 
         for ( j = 0 ; j < nullspaceLen_; j++ ) 
            nullspaceVec_[i*nullspaceLen_+j] /= scalings[j];
   }
   return 0;
}

/* ********************************************************************* *
 * set parameter for calibration AMG 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::setCalibrationSize( int size )
{
   if ( size > 0 ) calibrationSize_ = size;
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
      printf("\t*** number of levels        = %d\n", numLevels_);
      printf("\t*** coarsen scheme          = %d\n", coarsenScheme_);
      printf("\t*** nodal degree of freedom = %d\n", nodeDofs_);
      printf("\t*** null space dimension    = %d\n", nullspaceDim_);
      printf("\t*** Smooth vectors          = %d\n", numSmoothVec_);
      printf("\t*** Smooth vector steps     = %d\n", numSmoothVecSteps_);
      printf("\t*** strength threshold      = %e\n", threshold_);
      printf("\t*** Prolongator factor      = %e\n", Pweight_);
      printf("\t*** S Prolongator level     = %d\n", SPLevel_);
      printf("\t*** drop tolerance for P    = %e\n", dropTolForP_);
      printf("\t*** A-norm scheme           = %d\n", calcNormScheme_);
      printf("\t*** minimum coarse size     = %d\n", minCoarseSize_);
      printf("\t*** pre  smoother type      = %s\n", preSmoother_); 
      printf("\t*** pre  smoother nsweeps   = %d\n", preSmootherNum_);
      printf("\t*** post smoother type      = %s\n", postSmoother_); 
      printf("\t*** post smoother nsweeps   = %d\n", postSmootherNum_);
      printf("\t*** coarse solver type      = %s\n", coarseSolver_); 
      printf("\t*** coarse solver nsweeps   = %d\n", coarseSolverNum_);  
      printf("\t*** calibration size        = %d\n", calibrationSize_);
      printf("\t********************************************************\n");
   }
   return 0;
}

/* ********************************************************************* *
 * print AMGSA statistics information
 * --------------------------------------------------------------------- */

int MLI_Method_AMGSA::printStatistics(MLI *mli)
{
   int          mypid, level, globalNRows, totNRows, fineNRows;
   int          maxNnz, minNnz, fineNnz, totNnz, thisNnz, itemp;
   double       maxVal, minVal, dtemp, dthisNnz, dtotNnz, dfineNnz;
   char         paramString[100];
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
      printf("\t*** number of levels = %d\n", currLevel_+1);
      printf("\t*** total RAP   time = %e seconds\n", RAPTime_);
      printf("\t*** total GenML time = %e seconds\n", totalTime_);
      printf("\t******************** Amatrix ***************************\n");
      printf("\t*level   Nrows  MaxNnz MinNnz  TotalNnz  maxValue  minValue*\n");
   }

   /* --------------------------------------------------------------- */
   /* fine and coarse matrix complexity information                   */
   /* --------------------------------------------------------------- */

   totNnz = totNRows = 0;
   dtotNnz = 0.0;
   for ( level = 0; level <= currLevel_; level++ )
   {
      mli_Amat = mli->getSystemMatrix( level );
      sprintf(paramString, "nrows");
      mli_Amat->getMatrixInfo(paramString, globalNRows, dtemp);
      sprintf(paramString, "maxnnz");
      mli_Amat->getMatrixInfo(paramString, maxNnz, dtemp);
      sprintf(paramString, "minnnz");
      mli_Amat->getMatrixInfo(paramString, minNnz, dtemp);
      sprintf(paramString, "totnnz");
      mli_Amat->getMatrixInfo(paramString, thisNnz, dtemp);
      sprintf(paramString, "maxval");
      mli_Amat->getMatrixInfo(paramString, itemp, maxVal);
      sprintf(paramString, "minval");
      mli_Amat->getMatrixInfo(paramString, itemp, minVal);
      sprintf(paramString, "dtotnnz");
      mli_Amat->getMatrixInfo(paramString, itemp, dthisNnz);
      if ( mypid == 0 )
      {
         if (globalNRows > 25000000)
            printf("\t*%3d %10d %5d  %5d %11.5e %8.3e %8.3e *\n",level,
                   globalNRows, maxNnz, minNnz, dthisNnz, maxVal, minVal);
         else
            printf("\t*%3d %10d %5d  %5d %11d %8.3e %8.3e *\n",level,
                   globalNRows, maxNnz, minNnz, thisNnz, maxVal, minVal);
      }
      if ( level == 0 ) 
      {
         fineNnz = thisNnz;
         dfineNnz = dthisNnz;
      }
      totNnz += thisNnz;
      dtotNnz += dthisNnz;
      if ( level == 0 ) fineNRows = globalNRows;
      totNRows += globalNRows;
   }

   /* --------------------------------------------------------------- */
   /* prolongation operator complexity information                    */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      printf("\t******************** Pmatrix ***************************\n");
      printf("\t*level   Nrows  MaxNnz MinNnz  TotalNnz  maxValue  minValue*\n");
      fflush(stdout);
   }
   for ( level = 1; level <= currLevel_; level++ )
   {
      mli_Pmat = mli->getProlongation( level );
      sprintf(paramString, "nrows");
      mli_Pmat->getMatrixInfo(paramString, globalNRows, dtemp);
      sprintf(paramString, "maxnnz");
      mli_Pmat->getMatrixInfo(paramString, maxNnz, dtemp);
      sprintf(paramString, "minnnz");
      mli_Pmat->getMatrixInfo(paramString, minNnz, dtemp);
      sprintf(paramString, "totnnz");
      mli_Pmat->getMatrixInfo(paramString, thisNnz, dtemp);
      sprintf(paramString, "maxval");
      mli_Pmat->getMatrixInfo(paramString, itemp, maxVal);
      sprintf(paramString, "minval");
      mli_Pmat->getMatrixInfo(paramString, itemp, minVal);
      if ( mypid == 0 )
      {
         printf("\t*%3d %10d %5d  %5d %11d %8.3e %8.3e *\n",level,
                globalNRows, maxNnz, minNnz, thisNnz, maxVal, minVal);
      }
   }

   /* --------------------------------------------------------------- */
   /* other complexity information                                    */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      printf("\t********************************************************\n");
      if ( fineNnz > 1000000000 ) dtemp = dtotNnz / dfineNnz;
      else                        dtemp = dtotNnz / (double) fineNnz;
      printf("\t*** Amat complexity  = %e\n", dtemp);
      dtemp = (double) totNRows / (double) fineNRows;
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
   nDOF    = currNodeDofs_;
   ndim    = nullspaceDim_;
   nullvec = nullspaceVec_;
   leng    = nullspaceLen_;
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
      new_amgsa->maxLevels_ = maxLevels_;
      new_amgsa->setOutputLevel( outputLevel_ );
      new_amgsa->setNumLevels( numLevels_ );
      new_amgsa->setSmoother( MLI_SMOOTHER_PRE, preSmoother_, 
                              preSmootherNum_, preSmootherWgt_ );
      new_amgsa->setSmoother( MLI_SMOOTHER_POST, postSmoother_, 
                              postSmootherNum_, postSmootherWgt_ );
      new_amgsa->setCoarseSolver(coarseSolver_,coarseSolverNum_,
                                 coarseSolverWgt_ ); 
      new_amgsa->setCoarsenScheme( coarsenScheme_ );
      new_amgsa->setMinCoarseSize( minCoarseSize_ );
      if ( calcNormScheme_ ) new_amgsa->setCalcSpectralNorm();
      new_amgsa->setPweight( Pweight_ );
      new_amgsa->setSPLevel( SPLevel_ );
      new_amgsa->setNullSpace(nodeDofs_,nullspaceDim_,nullspaceVec_,
                              nullspaceLen_);
      new_amgsa->setSmoothVec( numSmoothVec_ );
      new_amgsa->setSmoothVecSteps( numSmoothVecSteps_ );
      new_amgsa->setStrengthThreshold( threshold_ );
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
#include "../../../parcsr_mv/_hypre_parcsr_mv.h"
#include "../../../seq_mv/seq_mv.h"
#include "../../../parcsr_ls/_hypre_parcsr_ls.h"
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
   hypre_ParVector    **lobVecs = new hypre_ParVector*[nullspaceDim_];
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

   for ( iV = 0; iV < nullspaceDim_; iV++ )
   {
      HYPRE_IJVectorCreate(comm, startRow, endRow, &tempIJ);
      HYPRE_IJVectorSetObjectType(tempIJ, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(tempIJ);
      HYPRE_IJVectorAssemble(tempIJ);
      offset = nullspaceLen_ * iV ;
      HYPRE_IJVectorSetValues(tempIJ, localNRows, (const int *) cols,
                              (const double *) &(nullspaceVec_[offset]));
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
   HYPRE_LobpcgSetBlocksize(lobpcgdata, nullspaceDim_);
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
   for ( iV = 0; iV < nullspaceDim_; iV++ )
   {
      uData = hypre_VectorData(
                 hypre_ParVectorLocalVector((hypre_ParVector *)lobVecs[iV]));
      offset = nullspaceLen_ * iV;
      for ( i = 0; i < nullspaceLen_; i++ )
         nullspaceVec_[offset+i] = uData[i];
      hypre_ParVectorDestroy(lobVecs[iV]);
   }
   HYPRE_LobpcgDestroy(lobpcgdata);
#endif
   return 0;
}

