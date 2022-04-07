/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

//******************************************************************************
//******************************************************************************
// This module supports the solution of linear systems arising from the finite
// element discretization of the incompressible Navier Stokes equations.
// The steps in using this module are :
//
//    (1)  precond = new HYPRE_LSI_BlockP(HYPRE_IJMatrix Amat)
//    (2a) precond->setSchemeBlockDiag(), or
//    (2b) precond->setSchemeBlockTriangular(), or
//    (2c) precond->setSchemeBlockInverse()
//    (3)  If lumped mass matrix is to be loaded, do the following :
//         -- call directly to HYPRE : beginCreateMapFromSoln 
//         -- use FEI function to load initial guess with map
//         -- call directly to HYPRE : endCreateMapFromSoln 
//    (4)  precond->setup(mapFromSolnList_,mapFromSolnList2_,mapFromSolnLeng_)
//    (5)  precond->solve( HYPRE_IJVector x, HYPRE_IJVector f )
// 
//******************************************************************************
//******************************************************************************

//******************************************************************************
// system include files
//------------------------------------------------------------------------------

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if 0 /* RDF: Not sure this is really needed */
#ifdef WIN32
#define strcmp _stricmp
#endif
#endif

//******************************************************************************
// HYPRE include files
//------------------------------------------------------------------------------

#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_LSI_ddilut.h"
#include "HYPRE_LSI_mli.h"
#include "HYPRE_LSI_blkprec.h"
#ifdef HAVE_ML
extern "C" {
   int HYPRE_LSI_MLCreate( MPI_Comm, HYPRE_Solver *);
   int HYPRE_LSI_MLDestroy( HYPRE_Solver );
   int HYPRE_LSI_MLSetup( HYPRE_Solver, HYPRE_ParCSRMatrix,
                          HYPRE_ParVector, HYPRE_ParVector );
   int HYPRE_LSI_MLSolve( HYPRE_Solver, HYPRE_ParCSRMatrix,
                          HYPRE_ParVector, HYPRE_ParVector );
   int HYPRE_LSI_MLSetStrongThreshold( HYPRE_Solver, double );
   int HYPRE_LSI_MLSetNumPreSmoothings( HYPRE_Solver, int );
   int HYPRE_LSI_MLSetNumPostSmoothings( HYPRE_Solver, int );
   int HYPRE_LSI_MLSetPreSmoother( HYPRE_Solver, int );
   int HYPRE_LSI_MLSetPostSmoother( HYPRE_Solver, int );
   int HYPRE_LSI_MLSetCoarseSolver( HYPRE_Solver, int );
}
#endif

//******************************************************************************
// external functions needed here and local defines
//------------------------------------------------------------------------------

extern "C" {
   int hypre_BoomerAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix**);
   int HYPRE_LSI_Search(int *, int, int);
}

//******************************************************************************
//******************************************************************************
// C-Interface data structure 
//------------------------------------------------------------------------------

typedef struct HYPRE_LSI_BlockPrecond_Struct
{
   void *precon;
} 
HYPRE_LSI_BlockPrecond;

//******************************************************************************
//******************************************************************************
// C-Interface functions to solver
//------------------------------------------------------------------------------

extern "C" 
int HYPRE_LSI_BlockPrecondCreate(MPI_Comm mpi_comm, HYPRE_Solver *solver)
{
   (void) mpi_comm;
   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *)
                                     hypre_CTAlloc(HYPRE_LSI_BlockPrecond, 1, HYPRE_MEMORY_HOST);
   HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) new HYPRE_LSI_BlockP();
   cprecon->precon = (void *) precon;
   (*solver) = (HYPRE_Solver) cprecon;
   return 0;
}

//------------------------------------------------------------------------------

extern "C" int HYPRE_LSI_BlockPrecondDestroy(HYPRE_Solver solver)
{
   int err=0;

   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      if ( precon != NULL ) delete precon;
      else                  err = 1;
      free( cprecon );
   }
   return err; 
}

//------------------------------------------------------------------------------

extern "C"
int HYPRE_LSI_BlockPrecondSetLumpedMasses(HYPRE_Solver solver, int length, 
                                          double *mass_v)
{
   int err=0;

   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      err = precon->setLumpedMasses(length, mass_v);
   }
   return err;
}
   
//------------------------------------------------------------------------------

extern "C" int HYPRE_LSI_BlockPrecondSetParams(HYPRE_Solver solver,
                                               char *params)
{
   int err=0;

   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      err = precon->setParams(params);
   }
   return err;
}

//------------------------------------------------------------------------------

extern "C" int HYPRE_LSI_BlockPrecondSetLookup(HYPRE_Solver solver, 
                                               HYPRE_Lookup *lookup)
{
   int err=0;

   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      err = precon->setLookup((Lookup *)lookup->object);
   }
   return err;
}

//------------------------------------------------------------------------------

extern "C" 
int HYPRE_LSI_BlockPrecondSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix Amat,
                                HYPRE_ParVector b, HYPRE_ParVector x)
{
   int err=0;

   (void) b;
   (void) x;
   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      err = precon->setup(Amat);
   }
   return err;
}

//------------------------------------------------------------------------------

extern "C" 
int HYPRE_LSI_BlockPrecondSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix Amat,
                                HYPRE_ParVector b, HYPRE_ParVector x)
{
   int err=0;

   (void) Amat;
   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      err = precon->solve(b, x);
   }
   return err;
}

//------------------------------------------------------------------------------

extern "C" int HYPRE_LSI_BlockPrecondSetA11Tolerance(HYPRE_Solver solver,
                                                     double tol)
{
   int    err=0;
   double tol2;
   char   paramString[30];

   HYPRE_LSI_BlockPrecond *cprecon = (HYPRE_LSI_BlockPrecond *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_BlockP *precon = (HYPRE_LSI_BlockP *) cprecon->precon;
      tol2 = tol;
      if ( tol2 > 0.1 ) tol2 = 0.1;
      sprintf( paramString, "blockP A11Tolerance %e", tol2 );
      err = precon->setParams(paramString);
   }
   return err;
}

//******************************************************************************
//******************************************************************************
//******************************************************************************
// Constructor
//------------------------------------------------------------------------------

HYPRE_LSI_BlockP::HYPRE_LSI_BlockP()
{
   Amat_                     = NULL;
   A11mat_                   = NULL;
   A12mat_                   = NULL;
   A22mat_                   = NULL;
   F1vec_                    = NULL;
   F2vec_                    = NULL;
   X1vec_                    = NULL;
   X2vec_                    = NULL;
   X1aux_                    = NULL;
   APartition_               = NULL;
   P22LocalInds_             = NULL;
   P22GlobalInds_            = NULL;
   P22Offsets_               = NULL;
   P22Size_                  = -1;
   P22GSize_                 = -1;
   block1FieldID_            = 0;
   block2FieldID_            = 1;
   assembled_                = 0;
   outputLevel_              = 0;
   lumpedMassScheme_         = 0;
   lumpedMassThresh_         = 0.1;
   lumpedMassNlevels_        = 1;
   lumpedMassLength_         = 0;
   lumpedMassDiag_           = NULL;
   scheme_                   = HYPRE_INCFLOW_BDIAG;
   printFlag_                = 0;
   A11Solver_                = NULL;
   A11Precond_               = NULL;
   A22Solver_                = NULL;
   A22Precond_               = NULL;
   A11Params_.SolverID_      = 1;       /* default : gmres */
   A22Params_.SolverID_      = 0;       /* default : cg */
   A11Params_.PrecondID_     = 1;       /* default : diagonal */
   A22Params_.PrecondID_     = 1;       /* default : diagonal */
   A11Params_.Tol_           = 1.0e-2;
   A22Params_.Tol_           = 1.0e-2;
   A11Params_.MaxIter_       = 1000;
   A22Params_.MaxIter_       = 1000;
   A11Params_.PSNLevels_     = 1;
   A22Params_.PSNLevels_     = 1;
   A11Params_.PSThresh_      = 1.0e-1;
   A22Params_.PSThresh_      = 1.0e-1;
   A11Params_.PSFilter_      = 2.0e-1;
   A22Params_.PSFilter_      = 2.0e-1;
   A11Params_.AMGThresh_     = 5.0e-1;
   A22Params_.AMGThresh_     = 5.0e-1;
   A11Params_.AMGRelaxType_  = 6;
   A22Params_.AMGRelaxType_  = 6;
   A11Params_.AMGNSweeps_    = 2;
   A22Params_.AMGNSweeps_    = 2;
   A11Params_.AMGSystemSize_ = 1;
   A22Params_.AMGSystemSize_ = 1;
   A11Params_.PilutFillin_   = 100;
   A22Params_.PilutFillin_   = 100;
   A11Params_.PilutDropTol_  = 0.1;
   A22Params_.PilutDropTol_  = 0.1;
   A11Params_.EuclidNLevels_ = 1;
   A22Params_.EuclidNLevels_ = 1;
   A11Params_.EuclidThresh_  = 0.1;
   A22Params_.EuclidThresh_  = 0.1;
   A11Params_.DDIlutFillin_  = 3.0;
   A22Params_.DDIlutFillin_  = 3.0;
   A11Params_.DDIlutDropTol_ = 0.2;
   A22Params_.DDIlutDropTol_ = 0.2;
   A11Params_.MLThresh_      = 0.08;
   A22Params_.MLThresh_      = 0.08;
   A11Params_.MLNSweeps_     = 1;
   A22Params_.MLNSweeps_     = 1;
   A11Params_.MLIThresh_     = 0.08;
   A22Params_.MLIThresh_     = 0.08;
   A11Params_.MLINSweeps_    = 1;
   A22Params_.MLINSweeps_    = 1;
   A11Params_.MLIRelaxType_  = 2;
   A22Params_.MLIRelaxType_  = 2;
   A11Params_.MLIPweight_    = 0.0;
   A22Params_.MLIPweight_    = 0.0;
   A11Params_.MLINodeDOF_    = 3;
   A22Params_.MLINodeDOF_    = 3;
   A11Params_.MLINullDim_    = 3;
   A22Params_.MLINullDim_    = 3;
}

//******************************************************************************
// destructor
//------------------------------------------------------------------------------

HYPRE_LSI_BlockP::~HYPRE_LSI_BlockP()
{
   if ( A11mat_         != NULL ) HYPRE_IJMatrixDestroy(A11mat_);
   if ( A12mat_         != NULL ) HYPRE_IJMatrixDestroy(A12mat_);
   if ( A22mat_         != NULL ) HYPRE_IJMatrixDestroy(A22mat_);
   if ( APartition_     != NULL ) free( APartition_ );
   if ( P22LocalInds_   != NULL ) delete [] P22LocalInds_;
   if ( P22GlobalInds_  != NULL ) delete [] P22GlobalInds_;
   if ( P22Offsets_     != NULL ) delete [] P22Offsets_;
   if ( lumpedMassDiag_ != NULL ) delete [] lumpedMassDiag_;
   if ( F1vec_          != NULL ) HYPRE_IJVectorDestroy( F1vec_ );
   if ( F2vec_          != NULL ) HYPRE_IJVectorDestroy( F2vec_ );
   if ( X1vec_          != NULL ) HYPRE_IJVectorDestroy( X1vec_ );
   if ( X2vec_          != NULL ) HYPRE_IJVectorDestroy( X2vec_ );
   if ( X1aux_          != NULL ) HYPRE_IJVectorDestroy( X1aux_ );
   destroySolverPrecond();
}

//******************************************************************************
// load mass matrix for pressure
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::setLumpedMasses(int length, double *Mdata)
{
   if ( length <= 0 )
   {
      printf("HYPRE_LSI_BlockP setLumpedMasses ERROR : M has length <= 0\n");
      exit(1);
   }
   lumpedMassLength_ = length;
   if ( lumpedMassDiag_ != NULL ) delete [] lumpedMassDiag_;
   lumpedMassDiag_ = new double[length];
   for ( int i = 0; i < length; i++ ) lumpedMassDiag_[i] = Mdata[i];
   return 0;
}

//******************************************************************************
// set internal parameters
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::setParams(char *params)
{
   char   param1[256], param2[256], param3[256];

   sscanf(params,"%s", param1);
   if ( strcmp(param1, "blockP") )
   {
      printf("HYPRE_LSI_BlockP::parameters not for me.\n");
      return 1;
   }
   sscanf(params,"%s %s", param1, param2);
   if ( !strcmp(param2, "help") )
   {
      printf("Available options for blockP are : \n");
      printf("      blockD (block diagonal)\n");
      printf("      blockS (block diagonal with Schur complement\n");
      printf("      blockT (block tridiagonal)\n");
      printf("      blockLU (block LU) \n");
      printf("      outputLevel <d> \n");
      printf("      block1FieldID <d> (-3 : special for implicitHydro)\n");
      printf("      block2FieldID <d> (-3 : special for implicitHydro)\n");
      printf("      printInfo \n");
      printf("      lumpedMassScheme <diag,ainv> \n");
      printf("      invA11PSNlevels <d> \n");
      printf("      invA11PSThresh <f> \n");
      printf("      A11Solver <cg,gmres,boomeramg,diag,identity> \n");
      printf("      A11Tolerance <f> \n");
      printf("      A11MaxIterations <d> \n");
      printf("      A11Precon <pilut,boomeramg,euclid,parasails,ddilut,mli>\n");
      printf("      A11PreconPSNlevels <d> \n");
      printf("      A11PreconPSThresh <f> \n");
      printf("      A11PreconPSFilter <f> \n");
      printf("      A11PreconAMGThresh <f> \n");
      printf("      A11PreconAMGRelaxType <jacobi,hybrid,hybridsym> \n");
      printf("      A11PreconAMGNumSweeps <d> \n");
      printf("      A11PreconAMGSystemSize <d> \n");
      printf("      A11PreconEuclidNLevels <d> \n");
      printf("      A11PreconEuclidThresh <f> \n");
      printf("      A11PreconPilutFillin <d> \n");
      printf("      A11PreconPilutDropTol <f> \n");
      printf("      A11PreconDDIlutFillin <f> \n");
      printf("      A11PreconDDIlutDropTol <f> \n");
      printf("      A11PreconMLIThresh <f> \n");
      printf("      A11PreconMLIRelaxType <jacobi,gs,sgs,bsgs,parasails> \n");
      printf("      A11PreconMLINumSweeps <d> \n");
      printf("      A11PreconMLIPweight <f> \n");
      printf("      A11PreconMLINodeDOF <d> \n");
      printf("      A11PreconMLINullDim <d> \n");
      printf("      A22Solver <cg,gmres,boomeramg,diag,identity>\n");
      printf("      A22Tolerance <f> \n");
      printf("      A22MaxIterations <d> \n");
      printf("      A22Precon <pilut,boomeramg,euclid,parasails,ddilut,mli>\n");
      printf("      A22PreconPSNlevels <d> \n");
      printf("      A22PreconPSThresh <f> \n");
      printf("      A22PreconPSFilter <f> \n");
      printf("      A22PreconAMGThresh <f> \n");
      printf("      A22PreconAMGRelaxType <jacobi,hybrid,hybridsym> \n");
      printf("      A22PreconAMGNumSweeps <d> \n");
      printf("      A22PreconAMGSystemSize <d> \n");
      printf("      A22PreconEuclidNLevels <d> \n");
      printf("      A22PreconEuclidThresh <f> \n");
      printf("      A22PreconPilutFillin <d> \n");
      printf("      A22PreconPilutDropTol <f> \n");
      printf("      A22PreconDDIlutFillin <f> \n");
      printf("      A22PreconDDIlutDropTol <f> \n");
      printf("      A22PreconMLIThresh <f> \n");
      printf("      A22PreconMLIRelaxType <jacobi,gs,sgs,bsgs,parasails> \n");
      printf("      A22PreconMLINumSweeps <d> \n");
      printf("      A22PreconMLIPweight <f> \n");
      printf("      A22PreconMLINodeDOF <d> \n");
      printf("      A22PreconMLINullDim <d> \n");
   }
   else if ( !strcmp(param2, "blockD") )
   {
      scheme_ = HYPRE_INCFLOW_BDIAG;
      if ( outputLevel_ > 0 ) 
         printf("HYPRE_LSI_BlockP::select block diagonal.\n");
   }
   else if ( !strcmp(param2, "blockS") )
   {
      scheme_ = HYPRE_INCFLOW_SDIAG;
      if ( outputLevel_ > 0 ) 
         printf("HYPRE_LSI_BlockP::select block diagonal with Schur.\n");
   }
   else if ( !strcmp(param2, "blockT") )
   {
      scheme_ = HYPRE_INCFLOW_BTRI;
      if ( outputLevel_ > 0 ) 
         printf("HYPRE_LSI_BlockP::select block triangular.\n");
   }
   else if ( !strcmp(param2, "blockLU") )
   {
      scheme_ = HYPRE_INCFLOW_BLU;
      if ( outputLevel_ > 0 ) 
         printf("HYPRE_LSI_BlockP::select block LU.\n");
   }
   else if ( !strcmp(param2, "outputLevel") )
   {
      sscanf(params,"%s %s %d", param1, param2, &outputLevel_);
      if ( outputLevel_ > 0 ) 
         printf("HYPRE_LSI_BlockP::outputLevel = %d.\n", outputLevel_);
   }
   else if ( !strcmp(param2, "block1FieldID") )
   {
      sscanf(params,"%s %s %d", param1, param2, &block1FieldID_);
      if ( outputLevel_ > 0 ) 
         printf("HYPRE_LSI_BlockP::block1FieldID = %d.\n", block1FieldID_);
   }
   else if ( !strcmp(param2, "block2FieldID") )
   {
      sscanf(params,"%s %s %d", param1, param2, &block2FieldID_);
      if ( outputLevel_ > 0 ) 
         printf("HYPRE_LSI_BlockP::block2FieldID = %d.\n", block2FieldID_);
   }
   else if ( !strcmp(param2, "printInfo") )
   {
      printFlag_ = 1;
      if ( outputLevel_ > 0 ) 
         printf("HYPRE_LSI_BlockP::set print flag.\n");
   }
   else if ( !strcmp(param2, "invA11PSNlevels") )
   {
      sscanf(params,"%s %s %d", param1, param2, &lumpedMassNlevels_);
      if ( outputLevel_ > 0 ) 
         printf("HYPRE_LSI_BlockP::invA11PSNlevels = %d.\n",lumpedMassNlevels_);
   }
   else if ( !strcmp(param2, "invA11PSThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &lumpedMassThresh_);
      if ( outputLevel_ > 0 ) 
         printf("HYPRE_LSI_BlockP::invA11PSThresh = %e.\n", lumpedMassThresh_);
   }
   else if ( !strcmp(param2, "invA11Scheme") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if ( !strcmp(param3, "diag") ) 
      {
         lumpedMassScheme_ = 0;
         if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::invA11Scheme=diag\n");
      }
      else if ( !strcmp(param3, "ainv") ) 
      {
         lumpedMassScheme_ = 1;
         if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::invA11Scheme = ainv\n");
      }
   }
   else if ( !strcmp(param2, "A11Solver") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if ( !strcmp(param3, "cg") ) 
      {
         A11Params_.SolverID_ = 0;
         if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11 solver = cg\n");
      }
      else if ( !strcmp(param3, "gmres") ) 
      {
         A11Params_.SolverID_ = 1;
         if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11 solver = gmres\n");
      }
      else if ( !strcmp(param3, "boomeramg") ) 
      {
         A11Params_.SolverID_ = 2;
         if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11 solver = boomeramg\n");
      }
      else if ( !strcmp(param3, "diagonal") ) 
      {
         A11Params_.SolverID_ = 3;
         if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11 solver = diag\n");
      }
      else if ( !strcmp(param3, "identity") ) 
      {
         A11Params_.SolverID_ = 9;
         if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11 solver = identity\n");
      }
      else
      {
         printf("HYPRE_LSI_BlockP::invalid A11 solver %s.\n",param3);
      }
   }
   else if ( !strcmp(param2, "A22Solver") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if ( !strcmp(param3, "cg") ) 
      {
         A22Params_.SolverID_ = 0;
         if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22 solver = cg\n");
      }
      else if ( !strcmp(param3, "gmres") ) 
      {
         A22Params_.SolverID_ = 1;
         if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22 solver = gmres\n");
      }
      else if ( !strcmp(param3, "boomeramg") ) 
      {
         A22Params_.SolverID_ = 2;
         if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22 solver = boomeramg\n");
      }
      else if ( !strcmp(param3, "diagonal") ) 
      {
         A22Params_.SolverID_ = 3;
         if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22 solver = diag\n");
      }
      else if ( !strcmp(param3, "identity") ) 
      {
         A22Params_.SolverID_ = 9;
         if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22 solver = identity\n");
      }
      else
      {
         printf("HYPRE_LSI_BlockP::invalid A22 solver %s.\n",param3);
      }
   }
   else if ( !strcmp(param2, "A11Tolerance") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.Tol_));
      if ( A11Params_.Tol_ >= 1.0 || A11Params_.Tol_ <= 0.0 ) 
         A11Params_.Tol_ = 1.0e-12;
      if (outputLevel_ > 0) 
         printf("HYPRE_LSI_BlockP::A11 tol = %e\n", A11Params_.Tol_);
   }
   else if ( !strcmp(param2, "A22Tolerance") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A22Params_.Tol_));
      if ( A22Params_.Tol_ >= 1.0 || A22Params_.Tol_ <= 0.0 ) 
         A22Params_.Tol_ = 1.0e-12;
      if (outputLevel_ > 0) 
         printf("HYPRE_LSI_BlockP::A22 tol = %e\n", A22Params_.Tol_);
   }
   else if ( !strcmp(param2, "A11MaxIterations") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.MaxIter_));
      if ( A11Params_.MaxIter_ <= 0 ) A11Params_.MaxIter_ = 10;
      if (outputLevel_ > 0) 
         printf("HYPRE_LSI_BlockP::A11 maxiter = %d\n", A11Params_.MaxIter_);
   }
   else if ( !strcmp(param2, "A22MaxIterations") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A22Params_.MaxIter_));
      if ( A22Params_.MaxIter_ <= 0 ) A22Params_.MaxIter_ = 10;
      if (outputLevel_ > 0) 
         printf("HYPRE_LSI_BlockP::A22 maxiter = %d\n", A22Params_.MaxIter_);
   }
   else if ( !strcmp(param2, "A11Precon") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if ( !strcmp(param3, "diagonal") ) 
      {
         A11Params_.PrecondID_ = 1;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A11 precon = diagonal\n");
      }
      else if ( !strcmp(param3, "parasails") ) 
      {
         A11Params_.PrecondID_ = 2;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A11 precon = parasails\n");
      }
      else if ( !strcmp(param3, "boomeramg") ) 
      {
         A11Params_.PrecondID_ = 3;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A11 precon = boomeramg\n");
      }
      else if ( !strcmp(param3, "pilut") ) 
      {
         A11Params_.PrecondID_ = 4;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A11 precon = pilut\n");
      }
      else if ( !strcmp(param3, "euclid") ) 
      {
         A11Params_.PrecondID_ = 5;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A11 precon = euclid\n");
      }
      else if ( !strcmp(param3, "ddilut") ) 
      {
         A11Params_.PrecondID_ = 6;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A11 precon = ddilut\n");
      }
      else if ( !strcmp(param3, "ml") ) 
      {
         A11Params_.PrecondID_ = 7;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A11 precon = ml\n");
      }
      else if ( !strcmp(param3, "mli") ) 
      {
         A11Params_.PrecondID_ = 8;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A11 precon = mli\n");
      }
   }
   else if ( !strcmp(param2, "A22Precon") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if ( !strcmp(param3, "diagonal") ) 
      {
         A22Params_.PrecondID_ = 1;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A22 precon = diagonal\n");
      }
      else if ( !strcmp(param3, "parasails") ) 
      {
         A22Params_.PrecondID_ = 2;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A22 precon = parasails\n");
      }
      else if ( !strcmp(param3, "boomeramg") ) 
      {
         A22Params_.PrecondID_ = 3;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A22 precon = boomeramg\n");
      }
      else if ( !strcmp(param3, "pilut") ) 
      {
         A22Params_.PrecondID_ = 4;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A22 precon = pilut\n");
      }
      else if ( !strcmp(param3, "euclid") ) 
      {
         A22Params_.PrecondID_ = 5;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A22 precon = euclid\n");
      }
      else if ( !strcmp(param3, "ddilut") ) 
      {
         A22Params_.PrecondID_ = 6;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A22 precon = ddilut\n");
      }
      else if ( !strcmp(param3, "ml") ) 
      {
         A22Params_.PrecondID_ = 7;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A22 precon = ml\n");
      }
      else if ( !strcmp(param3, "mli") ) 
      {
         A22Params_.PrecondID_ = 8;
         if (outputLevel_ > 0) 
            printf("HYPRE_LSI_BlockP::A22 precon = mli\n");
      }
   }
   else if ( !strcmp(param2, "A11PreconPSNlevels") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.PSNLevels_));
      if ( A11Params_.PSNLevels_ < 0 ) A11Params_.PSNLevels_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconPSNLevels\n");
   }
   else if ( !strcmp(param2, "A22PreconPSNlevels") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A22Params_.PSNLevels_));
      if ( A22Params_.PSNLevels_ < 0 ) A22Params_.PSNLevels_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconPSNLevels\n");
   }
   else if ( !strcmp(param2, "A11PreconPSThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.PSThresh_));
      if ( A11Params_.PSThresh_ < 0 ) A11Params_.PSThresh_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconPSThresh\n");
   }
   else if ( !strcmp(param2, "A22PreconPSThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A22Params_.PSThresh_));
      if ( A22Params_.PSThresh_ < 0 ) A22Params_.PSThresh_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconPSThresh\n");
   }
   else if ( !strcmp(param2, "A11PreconPSFilter") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.PSFilter_));
      if ( A11Params_.PSFilter_ < 0 ) A11Params_.PSFilter_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconPSFilter\n");
   }
   else if ( !strcmp(param2, "A22PreconPSFilter") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A22Params_.PSFilter_));
      if ( A22Params_.PSFilter_ < 0 ) A22Params_.PSFilter_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconPSFilter\n");
   }
   else if ( !strcmp(param2, "A11PreconAMGThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.AMGThresh_));
      if ( A11Params_.AMGThresh_ < 0 ) A11Params_.AMGThresh_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconAMGThresh\n");
   }
   else if ( !strcmp(param2, "A22PreconAMGThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A22Params_.AMGThresh_));
      if ( A22Params_.AMGThresh_ < 0 ) A22Params_.AMGThresh_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconAMGThresh\n");
   }
   else if ( !strcmp(param2, "A11PreconAMGRelaxType") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if      ( !strcmp(param3, "jacobi" ) ) A11Params_.AMGRelaxType_ = 0;
      else if ( !strcmp(param3, "gsSlow") )  A11Params_.AMGRelaxType_ = 1;
      else if ( !strcmp(param3, "gsFast") )  A11Params_.AMGRelaxType_ = 4;
      else if ( !strcmp(param3, "hybrid" ) ) A11Params_.AMGRelaxType_ = 3;
      else if ( !strcmp(param3, "hybridsym"))A11Params_.AMGRelaxType_ = 6;
      else                                       A11Params_.AMGRelaxType_ = 4;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconAMGRelaxType\n");
   }
   else if ( !strcmp(param2, "A22PreconAMGRelaxType") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if      ( !strcmp(param3, "jacobi" ) ) A22Params_.AMGRelaxType_ = 0;
      else if ( !strcmp(param3, "gsSlow") )  A22Params_.AMGRelaxType_ = 1;
      else if ( !strcmp(param3, "gsFast") )  A22Params_.AMGRelaxType_ = 4;
      else if ( !strcmp(param3, "hybrid" ) ) A22Params_.AMGRelaxType_ = 3;
      else if ( !strcmp(param3, "hybridsym"))A22Params_.AMGRelaxType_ = 6;
      else                                       A22Params_.AMGRelaxType_ = 4;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconAMGRelaxType\n");
   }
   else if ( !strcmp(param2, "A11PreconAMGNumSweeps") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.AMGNSweeps_));
      if ( A11Params_.AMGNSweeps_ < 0 ) A11Params_.AMGNSweeps_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconAMGNSweeps\n");
   }
   else if ( !strcmp(param2, "A22PreconAMGNumSweeps") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A22Params_.AMGNSweeps_));
      if ( A22Params_.AMGNSweeps_ < 0 ) A22Params_.AMGNSweeps_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconAMGNSweeps\n");
   }
   else if ( !strcmp(param2, "A11PreconAMGSystemSize") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.AMGSystemSize_));
      if ( A11Params_.AMGSystemSize_ < 0 ) A11Params_.AMGSystemSize_ = 1;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconAMGSystemSize\n");
   }
   else if ( !strcmp(param2, "A22PreconAMGSystemSize") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A22Params_.AMGSystemSize_));
      if ( A22Params_.AMGSystemSize_ < 0 ) A22Params_.AMGSystemSize_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconAMGSystemSize\n");
   }
   else if ( !strcmp(param2, "A11PreconEuclidNLevels") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.EuclidNLevels_));
      if ( A11Params_.EuclidNLevels_ < 0 ) A11Params_.EuclidNLevels_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconEuclidNLevels\n");
   }
   else if ( !strcmp(param2, "A22PreconEuclidNLevels") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A22Params_.EuclidNLevels_));
      if ( A22Params_.EuclidNLevels_ < 0 ) A22Params_.EuclidNLevels_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconEuclidNLevels\n");
   }
   else if ( !strcmp(param2, "A11PreconEuclidThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.EuclidThresh_));
      if ( A11Params_.EuclidThresh_ < 0 ) A11Params_.EuclidThresh_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconEuclidThresh\n");
   }
   else if ( !strcmp(param2, "A22PreconEuclidThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A22Params_.EuclidThresh_));
      if ( A22Params_.EuclidThresh_ < 0 ) A22Params_.EuclidThresh_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconEuclidThresh\n");
   }
   else if ( !strcmp(param2, "A11PreconPilutFillin") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.PilutFillin_));
      if ( A11Params_.PilutFillin_ < 0 ) A11Params_.PilutFillin_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconPilutFillin\n");
   }
   else if ( !strcmp(param2, "A22PreconPilutFillin") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A22Params_.PilutFillin_));
      if ( A22Params_.PilutFillin_ < 0 ) A22Params_.PilutFillin_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconPilutFillin\n");
   }
   else if ( !strcmp(param2, "A11PreconPilutDropTol") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.PilutDropTol_));
      if ( A11Params_.PilutDropTol_ < 0 ) A11Params_.PilutDropTol_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconPilutDropTol\n");
   }
   else if ( !strcmp(param2, "A22PreconPilutDropTol") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A22Params_.PilutDropTol_));
      if ( A22Params_.PilutDropTol_ < 0 ) A22Params_.PilutDropTol_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconPilutDropTol\n");
   }
   else if ( !strcmp(param2, "A11PreconDDIlutFillin") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.DDIlutFillin_));
      if ( A11Params_.DDIlutFillin_ < 0 ) A11Params_.DDIlutFillin_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconDDIlutFillin\n");
   }
   else if ( !strcmp(param2, "A22PreconDDIlutFillin") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A22Params_.DDIlutFillin_));
      if ( A22Params_.DDIlutFillin_ < 0 ) A22Params_.DDIlutFillin_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconDDIlutFillin\n");
   }
   else if ( !strcmp(param2, "A11PreconDDIlutDropTol") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.DDIlutDropTol_));
      if ( A11Params_.DDIlutDropTol_ < 0 ) A11Params_.DDIlutDropTol_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconDDIlutDropTol\n");
   }
   else if ( !strcmp(param2, "A22PreconDDIlutDropTol") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A22Params_.DDIlutDropTol_));
      if ( A22Params_.DDIlutDropTol_ < 0 ) A22Params_.DDIlutDropTol_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconDDIlutDropTol\n");
   }
   else if ( !strcmp(param2, "A11PreconMLThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.MLThresh_));
      if ( A11Params_.MLThresh_ < 0 ) A11Params_.MLThresh_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconMLThresh\n");
   }
   else if ( !strcmp(param2, "A22PreconMLThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A22Params_.MLThresh_));
      if ( A22Params_.MLThresh_ < 0 ) A22Params_.MLThresh_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconMLThresh\n");
   }
   else if ( !strcmp(param2, "A11PreconMLNumSweeps") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.MLNSweeps_));
      if ( A11Params_.MLNSweeps_ < 0 ) A11Params_.MLNSweeps_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconMLNumSweeps\n");
   }
   else if ( !strcmp(param2, "A22PreconMLNumSweeps") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A22Params_.MLNSweeps_));
      if ( A22Params_.MLNSweeps_ < 0 ) A22Params_.MLNSweeps_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconMLNumSweeps\n");
   }
   else if ( !strcmp(param2, "A11PreconMLIThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.MLIThresh_));
      if ( A11Params_.MLIThresh_ < 0.0 ) A11Params_.MLIThresh_ = 0.0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconMLIThresh\n");
   }
   else if ( !strcmp(param2, "A22PreconMLIThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A22Params_.MLIThresh_));
      if ( A22Params_.MLIThresh_ < 0.0 ) A22Params_.MLIThresh_ = 0.0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconMLIThresh\n");
   }
   else if ( !strcmp(param2, "A11PreconMLIRelaxType") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if      ( !strcmp(param3, "jacobi" ) ) A11Params_.MLIRelaxType_ = 0;
      else if ( !strcmp(param3, "gs") )      A11Params_.MLIRelaxType_ = 1;
      else if ( !strcmp(param3, "sgs") )     A11Params_.MLIRelaxType_ = 2;
      else if ( !strcmp(param3, "bsgs") )    A11Params_.MLIRelaxType_ = 3;
      else if ( !strcmp(param3, "parasails"))A11Params_.MLIRelaxType_ = 4;
      else                                       A11Params_.MLIRelaxType_ = 2;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconMLIRelaxType\n");
   }
   else if ( !strcmp(param2, "A22PreconMLIRelaxType") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if      ( !strcmp(param3, "jacobi" ) ) A22Params_.MLIRelaxType_ = 0;
      else if ( !strcmp(param3, "gs") )      A22Params_.MLIRelaxType_ = 1;
      else if ( !strcmp(param3, "sgs") )     A22Params_.MLIRelaxType_ = 2;
      else if ( !strcmp(param3, "bsgs") )    A22Params_.MLIRelaxType_ = 3;
      else if ( !strcmp(param3, "parasails"))A22Params_.MLIRelaxType_ = 4;
      else                                       A22Params_.MLIRelaxType_ = 2;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconMLIRelaxType\n");
   }
   else if ( !strcmp(param2, "A11PreconMLINumSweeps") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.MLINSweeps_));
      if ( A11Params_.MLINSweeps_ < 0 ) A11Params_.MLINSweeps_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconMLINSweeps\n");
   }
   else if ( !strcmp(param2, "A22PreconMLINumSweeps") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A22Params_.MLINSweeps_));
      if ( A22Params_.MLINSweeps_ < 0 ) A22Params_.MLINSweeps_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconMLINSweeps\n");
   }
   else if ( !strcmp(param2, "A11PreconMLIPweight") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.MLIPweight_));
      if ( A11Params_.MLIPweight_ < 0.0 ) A11Params_.MLIPweight_ = 0.0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconMLIPweight\n");
   }
   else if ( !strcmp(param2, "A22PreconMLIPweight") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A22Params_.MLIPweight_));
      if ( A22Params_.MLIPweight_ < 0.0 ) A22Params_.MLIPweight_ = 0.0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconMLIPweight\n");
   }
   else if ( !strcmp(param2, "A11PreconMLINodeDOF") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.MLINodeDOF_));
      if ( A11Params_.MLINodeDOF_ < 1 ) A11Params_.MLINodeDOF_ = 1;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconMLINodeDOF\n");
   }
   else if ( !strcmp(param2, "A22PreconMLINodeDOF") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A22Params_.MLINodeDOF_));
      if ( A22Params_.MLINodeDOF_ < 1 ) A22Params_.MLINodeDOF_ = 1;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconMLINodeDOF\n");
   }
   else if ( !strcmp(param2, "A11PreconMLINullDim") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.MLINullDim_));
      if ( A11Params_.MLINullDim_ < 1 ) A11Params_.MLINullDim_ = 1;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A11PreconMLINullDim\n");
   }
   else if ( !strcmp(param2, "A22PreconMLINullDim") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A22Params_.MLINullDim_));
      if ( A22Params_.MLINullDim_ < 1 ) A22Params_.MLINullDim_ = 1;
      if (outputLevel_ > 0) printf("HYPRE_LSI_BlockP::A22PreconMLINullDim\n");
   }
   else 
   {
      printf("HYPRE_LSI_BlockP:: string not recognized %s\n", params);
   }
   return 0;
}

//******************************************************************************
// set lookup object
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::setLookup(Lookup *object)
{
   lookup_ = object;
   return 0;
}

//******************************************************************************
// set up routine
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::setup(HYPRE_ParCSRMatrix Amat)
{
   int      i, j, irow, checkZeros, mypid, nprocs, AStart, AEnd, ANRows; 
   int      rowSize, *colInd, searchInd, newRow, one=1, maxRowSize;
   int      *colInd2, newRowSize, count, *newColInd, rowSize2;
   int      MNRows, MStartRow, *MRowLengs, SNRows, SStartRow, *SRowLengs;
   int      V1Leng, V1Start, V2Leng, V2Start;
   double   dtemp, *colVal, *colVal2, *newColVal;
   char     fname[100];
   FILE     *fp;
   MPI_Comm mpi_comm;
   HYPRE_IJMatrix     Mmat, B22mat;
   HYPRE_ParCSRMatrix Cmat_csr, Mmat_csr, Smat_csr, A22mat_csr, B22mat_csr;
   HYPRE_ParCSRMatrix A11mat_csr;
   HYPRE_Solver       parasails;

   //------------------------------------------------------------------
   // diagnostics
   //------------------------------------------------------------------

   if ( printFlag_ ) print();

   //------------------------------------------------------------------
   // build the blocks A11, A12, and the A22 block, if any
   //------------------------------------------------------------------

   Mmat = NULL;
   Mmat_csr = NULL;
   B22mat = NULL;
   B22mat_csr = NULL;
   Cmat_csr = NULL;
   Smat_csr = NULL;
   A11mat_csr = NULL;
   A22mat_csr = NULL;
   Amat_ = Amat;
   computeBlockInfo();
   buildBlocks();

   //------------------------------------------------------------------
   // If no approximate Schur complement needed, create zero matrix.
   // Extract the velocity mass matrix in HYPRE_ParCSRMatrix format :
   // the mass matrix comes either from user (lumpedMassDiag_) or 
   // extracted from the diagonal of the A(1,1) matrix => mass_v
   //------------------------------------------------------------------

   HYPRE_ParCSRMatrixGetComm( Amat_, &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   AStart = APartition_[mypid];
   AEnd   = APartition_[mypid+1] - 1;
   ANRows = AEnd - AStart + 1;

   if (lumpedMassScheme_ == 1)
   {
      HYPRE_ParaSailsCreate(mpi_comm, &parasails);
      HYPRE_ParaSailsSetParams(parasails,lumpedMassThresh_,
                               lumpedMassNlevels_);
      HYPRE_ParaSailsSetFilter(parasails, 0.1);
      HYPRE_ParaSailsSetLogging(parasails, 1);
      HYPRE_IJMatrixGetObject(A11mat_, (void **) &A11mat_csr);
      HYPRE_ParaSailsSetup(parasails, A11mat_csr, NULL, NULL);
      HYPRE_ParaSailsBuildIJMatrix(parasails, &Mmat);
   }
   else
   {
      if ( lumpedMassDiag_ != NULL )
      {
         checkZeros = 1;
         for ( i = 0; i < lumpedMassLength_; i++ )
            if ( lumpedMassDiag_[i] == 0.0 ) {checkZeros = 0; break;}
      }
      else checkZeros = 0;
      MNRows    = ANRows - P22Size_;
      MStartRow = AStart - P22Offsets_[mypid];
      MRowLengs = new int[MNRows];
      for ( irow = 0; irow < MNRows; irow++ ) MRowLengs[irow] = 1;
      HYPRE_IJMatrixCreate(mpi_comm, MStartRow, MStartRow+MNRows-1,
                           MStartRow, MStartRow+MNRows-1, &Mmat);
      HYPRE_IJMatrixSetObjectType(Mmat, HYPRE_PARCSR);
      HYPRE_IJMatrixSetRowSizes(Mmat, MRowLengs);
      HYPRE_IJMatrixInitialize(Mmat);
      delete [] MRowLengs;
      newRow = MStartRow;
      for ( irow = AStart; irow <= AEnd; irow++ ) 
      {
         searchInd = hypre_BinarySearch(P22LocalInds_, irow, P22Size_);
         if ( searchInd < 0 )
         {
            if ( checkZeros ) dtemp = lumpedMassDiag_[irow-AStart];
            else
            {
               HYPRE_ParCSRMatrixGetRow(Amat_,irow,&rowSize,&colInd,&colVal);
               for ( j = 0; j < rowSize; j++ ) 
                  if ( colInd[j] == irow ) { dtemp = colVal[j]; break;}
               HYPRE_ParCSRMatrixRestoreRow(Amat_,irow,&rowSize,&colInd,
                                            &colVal);
            }
            dtemp = 1.0 / dtemp;
            if (scheme_ == HYPRE_INCFLOW_BDIAG) dtemp = 0.0;
            HYPRE_IJMatrixSetValues(Mmat, 1, &one, (const int *) &newRow, 
                          (const int *) &newRow, (const double *) &dtemp);
            newRow++;
         }
      }
      HYPRE_IJMatrixAssemble(Mmat);
   }
   HYPRE_IJMatrixGetObject(Mmat, (void **) &Mmat_csr);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) Mmat_csr);

   //------------------------------------------------------------------
   // create Pressure Poisson matrix (S = C^T M^{-1} C)
   //------------------------------------------------------------------
   
   if (outputLevel_ >= 1) printf("BlockPrecond setup: C^T M^{-1}C begins\n");

   HYPRE_IJMatrixGetObject(A12mat_, (void **) &Cmat_csr);
   hypre_BoomerAMGBuildCoarseOperator((hypre_ParCSRMatrix *) Cmat_csr,
                                      (hypre_ParCSRMatrix *) Mmat_csr,
                                      (hypre_ParCSRMatrix *) Cmat_csr,
                                      (hypre_ParCSRMatrix **) &Smat_csr);

   if (outputLevel_ >= 1) printf("BlockPrecond setup: C^T M^{-1} C ends\n");

   //------------------------------------------------------------------
   // construct new A22 = A22 - S
   //------------------------------------------------------------------

   if ( A22mat_ != NULL )
   {
      B22mat = A22mat_;
      HYPRE_IJMatrixGetObject(B22mat, (void **) &B22mat_csr);
   } 
   else B22mat = NULL;
      
   SNRows    = P22Size_;
   SStartRow = P22Offsets_[mypid];
   HYPRE_IJMatrixCreate(mpi_comm, SStartRow, SStartRow+SNRows-1,
                        SStartRow, SStartRow+SNRows-1, &A22mat_);
   HYPRE_IJMatrixSetObjectType(A22mat_, HYPRE_PARCSR);

   SRowLengs = new int[SNRows];
   maxRowSize = 0;
   for ( irow = SStartRow; irow < SStartRow+SNRows; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(Smat_csr,irow,&rowSize,&colInd,NULL);
      newRowSize = rowSize;
      if ( B22mat != NULL )
      {
         HYPRE_ParCSRMatrixGetRow(B22mat_csr,irow,&rowSize2,&colInd2,NULL);
#if 1
         //==========================================
         // extract the whole S
         //------------------------------------------
         newRowSize += rowSize2;
         newColInd = new int[newRowSize];
         for (j = 0; j < rowSize;  j++) newColInd[j] = colInd[j];
         for (j = 0; j < rowSize2; j++) newColInd[j+rowSize] = colInd2[j];
#else
         //==========================================
         // extract only the diagonal from S
         //------------------------------------------
         //newRowSize++;
         //newColInd = new int[newRowSize];
         //for (j = 0; j < rowSize;  j++)
         //   if (colInd[j] == irow) {newColInd[0] = colInd[j]; break;}
         //if (j == rowSize) printf("ERROR : diagonal of S not found.\n"); 
         //for (j = 0; j < rowSize2; j++) newColInd[j+1] = colInd2[j];
         //==========================================
#endif
         hypre_qsort0(newColInd, 0, newRowSize-1);
         count = 0;
         for ( j = 1; j < newRowSize; j++ )
         {
            if ( newColInd[j] != newColInd[count] )
            {
               count++;
               newColInd[count] = newColInd[j];
            }
         }
         if ( newRowSize > 0 ) count++;
         newRowSize = count;
         HYPRE_ParCSRMatrixRestoreRow(B22mat_csr,irow,&rowSize2,&colInd2,NULL);
         delete [] newColInd;
      }
      SRowLengs[irow-SStartRow] = newRowSize;
      maxRowSize = ( newRowSize > maxRowSize ) ? newRowSize : maxRowSize;
      HYPRE_ParCSRMatrixRestoreRow(Smat_csr,irow,&rowSize,&colInd,NULL);
   }
   HYPRE_IJMatrixSetRowSizes(A22mat_, SRowLengs);
   HYPRE_IJMatrixInitialize(A22mat_);
   delete [] SRowLengs;

   for ( irow = SStartRow; irow < SStartRow+SNRows; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(Smat_csr,irow,&rowSize,&colInd,&colVal);
      if ( B22mat == NULL )
      {
#if 1
         //==========================================
         // extract the whole S
         //------------------------------------------
         newRowSize = rowSize;
         newColInd  = new int[newRowSize];
         newColVal  = new double[newRowSize];
         for (j = 0; j < rowSize; j++) 
         {
            newColInd[j] = colInd[j];
            newColVal[j] = - colVal[j];
         }
#else
         //==========================================
         // extract only the diagonal from S
         //------------------------------------------
         //newRowSize = 1;
         //newColInd  = new int[newRowSize];
         //newColVal  = new double[newRowSize];
         //for (j = 0; j < rowSize; j++) 
         //{
         //   if (colInd[j] == irow)
         //   {
         //      newColInd[0] = colInd[j];
         //      newColVal[0] = - colVal[j];
         //      break;
         //   }
         //}
         //if (j == rowSize) printf("ERROR : diagonal of S not found.\n"); 
         //==========================================
#endif
      }
      else
      {
         HYPRE_ParCSRMatrixGetRow(B22mat_csr,irow,&rowSize2,&colInd2,&colVal2);
#if 1
         //==========================================
         // extract the whole S
         //------------------------------------------
         newRowSize = rowSize + rowSize2;
         newColInd = new int[newRowSize];
         newColVal = new double[newRowSize];
         for (j = 0; j < rowSize; j++) 
         {
            newColInd[j] = colInd[j];
            newColVal[j] = - colVal[j];
         }
         for (j = 0; j < rowSize2; j++) 
         {
            newColInd[j+rowSize] = colInd2[j];
            newColVal[j+rowSize] = colVal2[j];
         }
#else
         //==========================================
         // extract only the diagonal from S
         //------------------------------------------
         //newRowSize = rowSize2 + 1;
         //newColInd = new int[newRowSize];
         //newColVal = new double[newRowSize];
         //for (j = 0; j < rowSize; j++) 
         //{
         //   if (colInd[j] == irow)
         //   {
         //      newColInd[0] = colInd[j];
         //      newColVal[0] = - colVal[j] * 1000;
         //      break;
         //   }
         //}
         //if (j == rowSize) printf("ERROR : diagonal of S not found.\n"); 
         //for (j = 0; j < rowSize2; j++) 
         //{
         //   newColInd[j+1] = colInd2[j];
         //   newColVal[j+1] = colVal2[j];
         //}
#endif

         hypre_qsort1(newColInd, newColVal, 0, newRowSize-1);
         count = 0;
         for ( j = 1; j < newRowSize; j++ )
         {
            if ( newColInd[j] != newColInd[count] )
            {
               count++;
               newColInd[count] = newColInd[j];
               newColVal[count] = newColVal[j];
            }
            else newColVal[count] += newColVal[j];
         }
         if ( newRowSize > 0 ) count++;
         newRowSize = count;
         HYPRE_ParCSRMatrixRestoreRow(B22mat_csr,irow,&rowSize2,
                                      &colInd2,&colVal2);
      }
      HYPRE_IJMatrixSetValues(A22mat_, 1, &newRowSize, (const int *) &irow,
	                  (const int *) newColInd, (const double *) newColVal);
      HYPRE_ParCSRMatrixRestoreRow(Smat_csr,irow,&rowSize,&colInd,&colVal);
      delete [] newColInd;
      delete [] newColVal;
   }
   HYPRE_IJMatrixAssemble(A22mat_);
   HYPRE_IJMatrixGetObject(A22mat_, (void **) &A22mat_csr);
   if ( B22mat != NULL ) HYPRE_IJMatrixDestroy(B22mat);
   if ( Mmat != NULL ) HYPRE_IJMatrixDestroy(Mmat);
   if ( Smat_csr != NULL ) HYPRE_ParCSRMatrixDestroy(Smat_csr);

   if ( outputLevel_ > 2 && A22mat_csr != NULL )
   {
      sprintf( fname, "A22.%d", mypid);
      fp = fopen( fname, "w" );
      for ( irow = SStartRow; irow < SStartRow+SNRows; irow++ ) 
      {
         HYPRE_ParCSRMatrixGetRow(A22mat_csr,irow,&rowSize,&colInd,&colVal);
         for ( j = 0; j < rowSize; j++ )
            fprintf(fp," %9d %9d %25.16e\n", irow+1, colInd[j]+1, colVal[j]);
         HYPRE_ParCSRMatrixRestoreRow(A22mat_csr,irow,&rowSize,&colInd,&colVal);
      }
      fclose(fp);
   }

   //------------------------------------------------------------------
   // build temporary vectors for solution steps
   //------------------------------------------------------------------

   V1Leng  = ANRows - P22Size_;
   V1Start = AStart - P22Offsets_[mypid];
   HYPRE_IJVectorCreate(mpi_comm, V1Start, V1Start+V1Leng-1, &F1vec_);
   HYPRE_IJVectorSetObjectType(F1vec_, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(F1vec_);
   HYPRE_IJVectorAssemble(F1vec_);

   HYPRE_IJVectorCreate(mpi_comm, V1Start, V1Start+V1Leng-1, &X1vec_);
   HYPRE_IJVectorSetObjectType(X1vec_, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(X1vec_);
   HYPRE_IJVectorAssemble(X1vec_);

   if ( scheme_ == HYPRE_INCFLOW_BLU )
   {
      HYPRE_IJVectorCreate(mpi_comm, V1Start, V1Start+V1Leng-1, &X1aux_);
      HYPRE_IJVectorSetObjectType(X1aux_, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(X1aux_);
      HYPRE_IJVectorAssemble(X1aux_);
   }

   V2Leng  = P22Size_;
   V2Start = P22Offsets_[mypid];
   HYPRE_IJVectorCreate(mpi_comm, V2Start, V2Start+V2Leng-1, &F2vec_);
   HYPRE_IJVectorSetObjectType(F2vec_, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(F2vec_);
   HYPRE_IJVectorAssemble(F2vec_);

   HYPRE_IJVectorCreate(mpi_comm, V2Start, V2Start+V2Leng-1, &X2vec_);
   HYPRE_IJVectorSetObjectType(X2vec_, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(X2vec_);
   HYPRE_IJVectorAssemble(X2vec_);

   assembled_ = 1;

   //------------------------------------------------------------------
   // setup solvers and preconditioners
   //------------------------------------------------------------------

   destroySolverPrecond();
   setupPrecon(&A11Precond_, A11mat_, A11Params_); 
   setupSolver(&A11Solver_, A11mat_, F1vec_, X1vec_, A11Precond_, A11Params_); 
   setupPrecon(&A22Precond_, A22mat_, A22Params_); 
   setupSolver(&A22Solver_, A22mat_, F2vec_, X2vec_, A22Precond_, A22Params_); 
   return 0;
}

//******************************************************************************
// solve 
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::solve(HYPRE_ParVector fvec, HYPRE_ParVector xvec)
{
   int       AStart, AEnd, irow, searchInd;
   int       mypid, nprocs, V1Start, V2Start, V1Cnt, V2Cnt;
   double    *fvals, *xvals, ddata;
   MPI_Comm  mpi_comm;

   //------------------------------------------------------------------
   // check for errors
   //------------------------------------------------------------------

   if ( assembled_ != 1 )
   {
      printf("BlockPrecond Solve ERROR : not assembled yet.\n");
      exit(1);
   }

   //------------------------------------------------------------------
   // extract matrix and machine information
   //------------------------------------------------------------------

   HYPRE_ParCSRMatrixGetComm( Amat_, &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   AStart  = APartition_[mypid];
   AEnd    = APartition_[mypid+1];

   //------------------------------------------------------------------
   // extract subvectors for the right hand side
   //------------------------------------------------------------------

   V1Start = AStart - P22Offsets_[mypid];
   V2Start = P22Offsets_[mypid];
   fvals = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector*)fvec));
   V1Cnt   = V1Start;
   V2Cnt   = V2Start;
   for ( irow = AStart; irow < AEnd; irow++ ) 
   {
      searchInd = hypre_BinarySearch( P22LocalInds_, irow, P22Size_);
      if ( searchInd >= 0 )
      {
         ddata = fvals[irow-AStart];
         HYPRE_IJVectorSetValues(F2vec_, 1, (const int *) &V2Cnt,
                                 (const double *) &ddata);
         V2Cnt++;
      }
      else
      {
         HYPRE_IJVectorSetValues(F1vec_, 1, (const int *) &V1Cnt,
	                         (const double *) &fvals[irow-AStart]);
         V1Cnt++;
      }
   } 

   //------------------------------------------------------------------
   // solve them according to the requested scheme 
   //------------------------------------------------------------------

   if ( A11Params_.SolverID_ == 0 )
      HYPRE_ParCSRPCGSetTol( A11Solver_, A11Params_.Tol_ );
   else if ( A11Params_.SolverID_ == 1 )
      HYPRE_ParCSRGMRESSetTol( A11Solver_, A11Params_.Tol_);
   else if ( A11Params_.SolverID_ == 2 )
      HYPRE_BoomerAMGSetTol( A11Solver_, A11Params_.Tol_);

   switch (scheme_)
   {
      case HYPRE_INCFLOW_BDIAG : solveBDSolve(X1vec_, X2vec_, F1vec_, F2vec_);
                                 break;

      case HYPRE_INCFLOW_SDIAG : solveBDSolve(X1vec_, X2vec_, F1vec_, F2vec_);
                                 break;

      case HYPRE_INCFLOW_BTRI :  solveBTSolve(X1vec_, X2vec_, F1vec_, F2vec_);
                                 break;

      case HYPRE_INCFLOW_BLU  :  solveBLUSolve(X1vec_, X2vec_, F1vec_, F2vec_);
                                 break;

      default :
           printf("HYPRE_LSI_BlockP ERROR : scheme not recognized.\n");
           exit(1);
   }

   //------------------------------------------------------------------
   // put the solution back to xvec
   //------------------------------------------------------------------

   V1Cnt = V1Start;
   V2Cnt = V2Start;
   xvals = hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector*)xvec));
   for ( irow = AStart; irow < AEnd; irow++ ) 
   {
      searchInd = hypre_BinarySearch( P22LocalInds_, irow, P22Size_);
      if ( searchInd >= 0 )
      {
         HYPRE_IJVectorGetValues(X2vec_, 1, &V2Cnt, &xvals[irow-AStart]);
         V2Cnt++;
      }
      else
      {
         HYPRE_IJVectorGetValues(X1vec_, 1, &V1Cnt, &xvals[irow-AStart]);
         V1Cnt++;
      }
   } 
   return 0;
}

//******************************************************************************
// print parameter settings
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::print()
{
   int      mypid;
   MPI_Comm mpi_comm;

   if ( Amat_ != NULL ) 
   {
      HYPRE_ParCSRMatrixGetComm( Amat_, &mpi_comm );
      MPI_Comm_rank( mpi_comm, &mypid );
   } 
   else mypid = 0;

   if ( mypid == 0 )
   {
      printf("*****************************************************\n");
      printf("***********HYPRE_LSI_BlockP Information**************\n");
      switch ( A11Params_.SolverID_ )
      {
         case 0 : printf("* A11 solver            = cg\n"); break;
         case 1 : printf("* A11 solver            = gmres\n"); break;
      }
      switch ( A11Params_.PrecondID_ )
      {
         case 1 : printf("* A11 preconditioner    = diagonal\n"); break;
         case 2 : printf("* A11 preconditioner    = parasails\n"); break;
         case 3 : printf("* A11 preconditioner    = boomeramg\n"); break;
         case 4 : printf("* A11 preconditioner    = pilut\n"); break;
         case 5 : printf("* A11 preconditioner    = euclid\n"); break;
         case 6 : printf("* A11 preconditioner    = ddilut\n"); break;
         case 7 : printf("* A11 preconditioner    = ml\n"); break;
         case 8 : printf("* A11 preconditioner    = ml\n"); break;
      }
      printf("* A11 solver tol        = %e\n", A11Params_.Tol_);
      printf("* A11 solver maxiter    = %d\n", A11Params_.MaxIter_);
      printf("* A11 ParaSails Nlevels = %d\n", A11Params_.PSNLevels_);
      printf("* A11 ParaSails thresh  = %e\n", A11Params_.PSThresh_);
      printf("* A11 ParaSails filter  = %e\n", A11Params_.PSFilter_);
      printf("* A11 BoomerAMG thresh  = %e\n", A11Params_.AMGThresh_);
      printf("* A11 BoomerAMG nsweeps = %d\n", A11Params_.AMGNSweeps_);
      printf("* A11 Pilut Fill-in     = %d\n", A11Params_.PilutFillin_);
      printf("* A11 Pilut Drop Tol    = %e\n", A11Params_.PilutDropTol_);
      printf("* A11 Euclid NLevels    = %d\n", A11Params_.EuclidNLevels_);
      printf("* A11 Euclid threshold  = %e\n", A11Params_.EuclidThresh_);
      printf("* A11 DDIlut Fill-in    = %e\n", A11Params_.DDIlutFillin_);
      printf("* A11 DDIlut Drop Tol   = %e\n", A11Params_.DDIlutDropTol_);
      printf("* A11 ML threshold      = %e\n", A11Params_.MLThresh_);
      printf("* A11 ML nsweeps        = %d\n", A11Params_.MLNSweeps_);
      printf("* A11 MLI threshold     = %e\n", A11Params_.MLIThresh_);
      printf("* A11 MLI nsweeps       = %d\n", A11Params_.MLINSweeps_);
      printf("* A11 MLI Pweight       = %e\n", A11Params_.MLIPweight_);
      printf("* A11 MLI nodeDOF       = %d\n", A11Params_.MLINodeDOF_);
      printf("* A11 MLI Null Dim      = %d\n", A11Params_.MLINullDim_);

      switch ( A22Params_.SolverID_ )
      {
         case 0 : printf("* A22 solver            = cg\n"); break;
         case 1 : printf("* A22 solver            = gmres\n"); break;
      }
      switch ( A22Params_.PrecondID_ )
      {
         case 1 : printf("* A22 preconditioner    = diagonal\n"); break;
         case 2 : printf("* A22 preconditioner    = parasails\n"); break;
         case 3 : printf("* A22 preconditioner    = boomeramg\n"); break;
         case 4 : printf("* A22 preconditioner    = pilut\n"); break;
         case 5 : printf("* A22 preconditioner    = euclid\n"); break;
         case 6 : printf("* A22 preconditioner    = ddilut\n"); break;
         case 7 : printf("* A22 preconditioner    = ml\n"); break;
         case 8 : printf("* A22 preconditioner    = mli\n"); break;
         case 9 : printf("* A22 preconditioner    = identity\n"); break;
      }
      printf("* A22 solver tol        = %e\n", A22Params_.Tol_);
      printf("* A22 solver maxiter    = %d\n", A22Params_.MaxIter_);
      printf("* A22 ParaSails Nlevels = %d\n", A22Params_.PSNLevels_);
      printf("* A22 ParaSails thresh  = %e\n", A22Params_.PSThresh_);
      printf("* A22 ParaSails filter  = %e\n", A22Params_.PSFilter_);
      printf("* A22 BoomerAMG thresh  = %e\n", A22Params_.AMGThresh_);
      printf("* A22 BoomerAMG nsweeps = %d\n", A22Params_.AMGNSweeps_);
      printf("* A22 Pilut Fill-in     = %d\n", A22Params_.PilutFillin_);
      printf("* A22 Pilut Drop Tol    = %e\n", A22Params_.PilutDropTol_);
      printf("* A22 Euclid NLevels    = %d\n", A22Params_.EuclidNLevels_);
      printf("* A22 Euclid threshold  = %e\n", A22Params_.EuclidThresh_);
      printf("* A22 DDIlut Fill-in    = %e\n", A22Params_.DDIlutFillin_);
      printf("* A22 DDIlut Drop Tol   = %e\n", A22Params_.DDIlutDropTol_);
      printf("* A22 ML threshold      = %e\n", A22Params_.MLThresh_);
      printf("* A22 ML nsweeps        = %d\n", A22Params_.MLNSweeps_);
      printf("* A22 MLI threshold     = %e\n", A22Params_.MLIThresh_);
      printf("* A22 MLI nsweeps       = %d\n", A22Params_.MLINSweeps_);
      printf("* A22 MLI Pweight       = %e\n", A22Params_.MLIPweight_);
      printf("* A22 MLI nodeDOF       = %d\n", A22Params_.MLINodeDOF_);
      printf("* A22 MLI Null Dim      = %d\n", A22Params_.MLINullDim_);
      printf("*****************************************************\n");
   }
   return 0;
} 

//******************************************************************************
// load mass matrix for pressure
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::destroySolverPrecond()
{
   if ( A11Solver_ != NULL ) 
   {
      if      (A11Params_.SolverID_ == 0) HYPRE_ParCSRPCGDestroy(A11Solver_);
      else if (A11Params_.SolverID_ == 1) HYPRE_ParCSRGMRESDestroy(A11Solver_);
      else if (A11Params_.SolverID_ == 2) HYPRE_BoomerAMGDestroy(A11Solver_);
   }
   if ( A22Solver_ != NULL ) 
   {
      if      (A22Params_.SolverID_ == 0) HYPRE_ParCSRPCGDestroy(A22Solver_);
      else if (A22Params_.SolverID_ == 1) HYPRE_ParCSRGMRESDestroy(A22Solver_);
      else if (A22Params_.SolverID_ == 2) HYPRE_BoomerAMGDestroy(A22Solver_);
   }
   if ( A11Precond_ != NULL ) 
   {
      if (A11Params_.PrecondID_ == 2) HYPRE_ParCSRParaSailsDestroy(A11Precond_);
      else if (A11Params_.PrecondID_ == 3) HYPRE_BoomerAMGDestroy(A11Precond_);
      else if (A11Params_.PrecondID_ == 4) HYPRE_ParCSRPilutDestroy(A11Precond_);
      else if (A11Params_.PrecondID_ == 5) HYPRE_EuclidDestroy(A11Precond_);
      else if (A11Params_.PrecondID_ == 6) HYPRE_LSI_DDIlutDestroy(A11Precond_);
#ifdef HAVE_ML
      else if (A11Params_.PrecondID_ == 7) HYPRE_LSI_MLDestroy(A11Precond_);
#endif
#ifdef HAVE_MLI
      else if (A11Params_.PrecondID_ == 8) HYPRE_LSI_MLIDestroy(A11Precond_);
#endif
   }
   if ( A22Precond_ != NULL ) 
   {
      if (A22Params_.PrecondID_ == 2) HYPRE_ParCSRParaSailsDestroy(A22Precond_);
      else if (A22Params_.PrecondID_ == 3) HYPRE_BoomerAMGDestroy(A22Precond_);
      else if (A22Params_.PrecondID_ == 4) HYPRE_ParCSRPilutDestroy(A22Precond_);
      else if (A22Params_.PrecondID_ == 5) HYPRE_EuclidDestroy(A22Precond_);
      else if (A22Params_.PrecondID_ == 6) HYPRE_LSI_DDIlutDestroy(A22Precond_);
#ifdef HAVE_ML
      else if (A22Params_.PrecondID_ == 7) HYPRE_LSI_MLDestroy(A22Precond_);
#endif
#ifdef HAVE_MLI
      else if (A22Params_.PrecondID_ == 8) HYPRE_LSI_MLIDestroy(A22Precond_);
#endif
   }
   A11Solver_  = NULL;
   A22Solver_  = NULL;
   A11Precond_ = NULL;
   A22Precond_ = NULL;
   return 0;
}

//******************************************************************************
// Given a matrix A, compute the sizes and indices of the 2 x 2 blocks
// (P22Size_,P22GSize_,P22LocalInds_,P22GlobalInds_,P22Offsets_,APartition_)
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::computeBlockInfo()
{
   int      mypid, nprocs, start_row, end_row, irow, nblks, *elem_blk_ids;
   int      j, row_size, *col_ind, *disp_array, interleave, lumping, elemDOF;
   int      field_id, jcol, zero_diag, nElems, nNodes, nEqns;
   double   *col_val;
   MPI_Comm mpi_comm;

   //------------------------------------------------------------------
   // check that the system matrix has been set, clean up previous
   // allocations, and extract matrix information
   //------------------------------------------------------------------

   if ( Amat_ == NULL )
   {
      printf("BlockPrecond ERROR : Amat not initialized.\n");
      exit(1);
   }
   if ( APartition_    != NULL ) free( APartition_ );
   if ( P22LocalInds_  != NULL ) delete [] P22LocalInds_;
   if ( P22GlobalInds_ != NULL ) delete [] P22GlobalInds_;
   if ( P22Offsets_    != NULL ) delete [] P22Offsets_;
   APartition_    = NULL;
   P22LocalInds_  = NULL;
   P22GlobalInds_ = NULL;
   P22Offsets_    = NULL;
   assembled_     = 0;
   HYPRE_ParCSRMatrixGetRowPartitioning( Amat_, &APartition_ );
   HYPRE_ParCSRMatrixGetComm( Amat_, &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   start_row    = APartition_[mypid];
   end_row      = APartition_[mypid+1] - 1;
    
   //------------------------------------------------------------------
   // find the local size of the (2,2) block
   //------------------------------------------------------------------

   P22Size_ = 0;
   if ( block1FieldID_ == -3 && block2FieldID_ == -3 )
   {
      for ( irow = end_row; irow >= start_row; irow-- ) 
      {
         HYPRE_ParCSRMatrixGetRow(Amat_,irow,&row_size,&col_ind,&col_val);
         zero_diag = 1;
         for ( jcol = 0;  jcol < row_size;  jcol++ ) 
         {
            if ( col_ind[jcol] == irow && col_val[jcol] != 0.0 ) 
            {
               zero_diag = 0; 
               break;
            }
         }
         HYPRE_ParCSRMatrixRestoreRow(Amat_,irow,&row_size,&col_ind,&col_val);
         if ( zero_diag ) P22Size_++;
         else             break;
      }
   }
   else if ( block1FieldID_ == -7 )
   {
      nblks = lookup_->getNumElemBlocks();
      elem_blk_ids = (int *) lookup_->getElemBlockIDs();
      P22Size_ = 0;
      for ( irow = 0; irow < nblks; irow++ )
      {
         lookup_->getElemBlockInfo(elem_blk_ids[irow], interleave, lumping, 
                                   elemDOF, nElems, nNodes, nEqns);
         P22Size_ += nElems;
      }
   }
   else
   {
      for ( irow = start_row; irow <= end_row; irow++ )
      {
         field_id = lookup_->getAssociatedFieldID(irow);
         if      (block2FieldID_ >= 0 && field_id == block2FieldID_) P22Size_++;
         else if (block2FieldID_ <  0 && field_id != block1FieldID_) P22Size_++;
      }
   }
   if ( outputLevel_ > 0 )
   {
      printf("%4d computeBlockInfo : P22_size = %d\n", mypid, P22Size_);
   }

   //------------------------------------------------------------------
   // allocate array for storing indices of (2,2) block variables 
   //------------------------------------------------------------------

   if ( P22Size_ > 0 ) P22LocalInds_ = new int[P22Size_];
   else                P22LocalInds_ = NULL; 

   //------------------------------------------------------------------
   // compose a local list of rows for the (2,2) block
   //------------------------------------------------------------------

   if ( block1FieldID_ == -3 && block2FieldID_ == -3 )
   {
      for ( irow = end_row-P22Size_+1; irow <= end_row; irow++ )
         P22LocalInds_[irow-end_row+P22Size_-1] = irow;
   }
   else if ( block1FieldID_ == -7 )
   {
      for ( irow = end_row-P22Size_+1; irow <= end_row; irow++ )
         P22LocalInds_[irow-end_row+P22Size_-1] = irow;
   }
   else
   {
      P22Size_ = 0;
      for ( irow = start_row; irow <= end_row; irow++ )
      {
         field_id = lookup_->getAssociatedFieldID(irow);
         if ( block2FieldID_ >= 0 && field_id == block2FieldID_ ) 
            P22LocalInds_[P22Size_++] = irow;
         else if ( block2FieldID_ <  0 && field_id != block1FieldID_ ) 
            P22LocalInds_[P22Size_++] = irow;
      }
   }

   //------------------------------------------------------------------
   // compose a global list of rows for the (2,2) block
   //------------------------------------------------------------------

   MPI_Allreduce(&P22Size_, &P22GSize_, 1, MPI_INT, MPI_SUM, mpi_comm);

   if (outputLevel_ > 0)
   {
      if ( P22GSize_ == 0 && mypid == 0 )
         printf("computeBlockInfo WARNING : P22Size = 0 on all processors.\n");
   }
   if ( P22GSize_ == 0 )
   {
      if ( APartition_ != NULL ) free( APartition_ );
      APartition_ = NULL;
      return 1;
   }

   if ( P22GSize_ > 0 ) P22GlobalInds_ = new int[P22GSize_];
   else                 P22GlobalInds_ = NULL;
   disp_array     = new int[nprocs];
   P22Offsets_    = new int[nprocs];
   MPI_Allgather(&P22Size_, 1, MPI_INT, P22Offsets_, 1, MPI_INT, mpi_comm);
   disp_array[0] = 0;
   for ( j = 1; j < nprocs; j++ ) 
      disp_array[j] = disp_array[j-1] + P22Offsets_[j-1];
   MPI_Allgatherv(P22LocalInds_, P22Size_, MPI_INT, P22GlobalInds_,
                  P22Offsets_, disp_array, MPI_INT, mpi_comm);
   delete [] P22Offsets_;
   P22Offsets_ = disp_array;

   if ( outputLevel_ > 2 )
   {
      for ( j = 0; j < P22Size_; j++ )
         printf("%4d computeBlockInfo : P22Inds %8d = %d\n", mypid,
                j, P22LocalInds_[j]);
   }
   return 0;
} 

//******************************************************************************
// Given a matrix A, build the 2 x 2 blocks
// (This function is to be called after computeBlockInfo
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::buildBlocks()
{
   int    mypid, nprocs, *partition, index, searchInd;
   int    ANRows, AGNRows, AStartRow;
   int    A11NRows, A11NCols, A11GNRows, A11GNCols, A11StartRow, A11StartCol;
   int    A12NRows, A12NCols, A12GNRows, A12GNCols, A12StartRow, A12StartCol;
   int    A22NRows, A22NCols, A22GNRows, A22GNCols, A22StartRow, A22StartCol;
   int    *A11RowLengs, A11MaxRowLeng, A11RowCnt, A11NewSize, *A11_inds;
   int    *A12RowLengs, A12MaxRowLeng, A12RowCnt, A12NewSize, *A12_inds;
   int    *A22RowLengs, A22MaxRowLeng, A22RowCnt, A22NewSize, *A22_inds;
   int    irow, j, rowSize, *inds;
   double *vals, *A11_vals, *A12_vals, *A22_vals;
   char   fname[200];
   FILE   *fp;
   MPI_Comm mpi_comm;
   HYPRE_ParCSRMatrix A11mat_csr, A22mat_csr, A12mat_csr;

   //------------------------------------------------------------------
   // extract information about the system matrix
   //------------------------------------------------------------------

   HYPRE_ParCSRMatrixGetRowPartitioning( Amat_, &partition );
   HYPRE_ParCSRMatrixGetComm( Amat_, &mpi_comm );
   MPI_Comm_rank( mpi_comm, &mypid );
   MPI_Comm_size( mpi_comm, &nprocs );
   AStartRow = partition[mypid];
   ANRows    = partition[mypid+1] - AStartRow;
   AGNRows   = partition[nprocs];

   //------------------------------------------------------------------
   // calculate the dimensions of the 2 x 2 blocks
   //------------------------------------------------------------------

   A11NRows    = ANRows - P22Size_;
   A11NCols    = A11NRows;
   A11GNRows   = AGNRows - P22GSize_;
   A11GNCols   = A11GNRows;
   A11StartRow = AStartRow - P22Offsets_[mypid];
   A11StartCol = A11StartRow;

   A12NRows    = ANRows - P22Size_;
   A12NCols    = P22Size_;
   A12GNRows   = AGNRows - P22GSize_;
   A12GNCols   = P22GSize_;
   A12StartRow = AStartRow - P22Offsets_[mypid];
   A12StartCol = P22Offsets_[mypid];

   A22NRows    = P22Size_;
   A22NCols    = P22Size_;
   A22GNRows   = P22GSize_;
   A22GNCols   = P22GSize_;
   A22StartRow = P22Offsets_[mypid];
   A22StartCol = P22Offsets_[mypid];

   if ( outputLevel_ >= 1 )
   {
      printf("%4d buildBlock (1,1) : StartRow  = %d\n", mypid, A11StartRow);
      printf("%4d buildBlock (1,1) : GlobalDim = %d %d\n", mypid, A11GNRows, 
                                                           A11GNCols);
      printf("%4d buildBlock (1,1) : LocalDim  = %d %d\n", mypid, A11NRows, 
                                                           A11NCols);
      printf("%4d buildBlock (1,2) : StartRow  = %d\n", mypid, A12StartRow);
      printf("%4d buildBlock (1,2) : GlobalDim = %d %d\n", mypid, A12GNRows, 
                                                           A12GNCols);
      printf("%4d buildBlock (1,2) : LocalDim  = %d %d\n", mypid, A12NRows, 
                                                           A12NCols);
      printf("%4d buildBlock (2,2) : StartRow  = %d\n", mypid, A22StartRow);
      printf("%4d buildBlock (2,2) : GlobalDim = %d %d\n", mypid, A22GNRows, 
                                                           A22GNCols);
      printf("%4d buildBlock (2,2) : LocalDim  = %d %d\n", mypid, A22NRows, 
                                                           A22NCols);
   }

   //------------------------------------------------------------------
   // figure the row sizes of the block matrices
   //------------------------------------------------------------------

   A11RowLengs = new int[A11NRows];
   A12RowLengs = new int[A12NRows];
   A22RowLengs = new int[A22NRows];
   A11MaxRowLeng = 0;
   A12MaxRowLeng = 0;
   A22MaxRowLeng = 0;
   A11RowCnt = 0;
   A12RowCnt = 0;
   A22RowCnt = 0;

   for ( irow = AStartRow; irow < AStartRow+ANRows; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(Amat_, irow, &rowSize, &inds, &vals);
      searchInd = hypre_BinarySearch(P22LocalInds_, irow, P22Size_);
      if ( searchInd < 0 )   // A(1,1) or A(1,2) block
      {
         A11NewSize = A12NewSize = 0;
         for ( j = 0; j < rowSize; j++ ) 
         {
            index = inds[j];
            searchInd = hypre_BinarySearch(P22GlobalInds_,index,P22GSize_);
            if (searchInd >= 0) A12NewSize++;
            else                A11NewSize++;
         }
         if ( A11NewSize <= 0 ) A11NewSize = 1;
         if ( A12NewSize <= 0 ) A12NewSize = 1;
         A11RowLengs[A11RowCnt++] = A11NewSize;
         A12RowLengs[A12RowCnt++] = A12NewSize;
         A11MaxRowLeng = (A11NewSize > A11MaxRowLeng) ? 
                          A11NewSize : A11MaxRowLeng;
         A12MaxRowLeng = (A12NewSize > A12MaxRowLeng) ? 
                          A12NewSize : A12MaxRowLeng;
      }
      else // A(2,2) block
      {
         A22NewSize = 0;
         for ( j = 0; j < rowSize; j++ ) 
         {
            index = inds[j];
            searchInd = hypre_BinarySearch(P22GlobalInds_,index,P22GSize_);
            if (searchInd >= 0) A22NewSize++;
         }
         A22RowLengs[A22RowCnt++] = A22NewSize;
         A22MaxRowLeng = (A22NewSize > A22MaxRowLeng) ? 
                          A22NewSize : A22MaxRowLeng;
      }
      HYPRE_ParCSRMatrixRestoreRow(Amat_, irow, &rowSize, &inds, &vals);
   }

   //------------------------------------------------------------------
   // create matrix contexts for the blocks
   //------------------------------------------------------------------

   HYPRE_IJMatrixCreate(mpi_comm, A11StartRow, A11StartRow+A11NRows-1,
                                A11StartCol, A11StartCol+A11NCols-1, &A11mat_);
   HYPRE_IJMatrixSetObjectType(A11mat_, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(A11mat_, A11RowLengs);
   HYPRE_IJMatrixInitialize(A11mat_);
   delete [] A11RowLengs;
   HYPRE_IJMatrixCreate(mpi_comm, A12StartRow, A12StartRow+A12NRows-1,
                        A12StartCol, A12StartCol+A12NCols-1, &A12mat_);
   HYPRE_IJMatrixSetObjectType(A12mat_, HYPRE_PARCSR);
   HYPRE_IJMatrixSetRowSizes(A12mat_, A12RowLengs);
   HYPRE_IJMatrixInitialize(A12mat_);
   delete [] A12RowLengs;
   if ( A22MaxRowLeng > 0 )
   {
      HYPRE_IJMatrixCreate(mpi_comm,A22StartRow,A22StartRow+A22NRows-1,
                           A22StartCol, A22StartCol+A22NCols-1, &A22mat_);
      HYPRE_IJMatrixSetObjectType(A22mat_, HYPRE_PARCSR);
      HYPRE_IJMatrixSetRowSizes(A22mat_, A22RowLengs);
      HYPRE_IJMatrixInitialize(A22mat_);
   }
   else A22mat_ = NULL;
   delete [] A22RowLengs;

   //------------------------------------------------------------------
   // load the matrices extracted from A
   //------------------------------------------------------------------

   A11_inds = new int[A11MaxRowLeng+1];
   A11_vals = new double[A11MaxRowLeng+1];
   A12_inds = new int[A12MaxRowLeng+1];
   A12_vals = new double[A12MaxRowLeng+1];
   A22_inds = new int[A22MaxRowLeng+1];
   A22_vals = new double[A22MaxRowLeng+1];

   A11RowCnt = A11StartRow;
   A12RowCnt = A12StartRow;
   A22RowCnt = A22StartRow;

   for ( irow = AStartRow; irow < AStartRow+ANRows; irow++ ) 
   {
      HYPRE_ParCSRMatrixGetRow(Amat_, irow, &rowSize, &inds, &vals);
      searchInd = hypre_BinarySearch(P22LocalInds_, irow, P22Size_);
      if ( searchInd < 0 )   // A(1,1) or A(1,2) block
      {
         A11NewSize = A12NewSize = 0;
         for ( j = 0; j < rowSize; j++ ) 
         {
            index = inds[j];
            searchInd = HYPRE_LSI_Search(P22GlobalInds_,index,P22GSize_);
            if (searchInd >= 0) // A(1,2) block 
            {
               A12_inds[A12NewSize] = searchInd;
               A12_vals[A12NewSize++] = vals[j];
            }
            else
            {
               searchInd = - searchInd - 1;
               A11_inds[A11NewSize] = index - searchInd;
               A11_vals[A11NewSize++] = vals[j];
            }
         }
         if ( A11NewSize == 0 )
         {
            A11_inds[0] = AStartRow - P22Offsets_[mypid];
            A11_vals[0] = 0.0;
            A11NewSize  = 1;
         }
         if ( A12NewSize == 0 )
         {
            A12_inds[0] = P22Offsets_[mypid];
            A12_vals[0] = 0.0;
            A12NewSize  = 1;
         }
         HYPRE_IJMatrixSetValues(A11mat_, 1, &A11NewSize, 
	                    (const int *) &A11RowCnt, (const int *) A11_inds,
                            (const double *) A11_vals);
         HYPRE_IJMatrixSetValues(A12mat_, 1, &A12NewSize, 
	                    (const int *) &A12RowCnt, (const int *) A12_inds,
                            (const double *) A12_vals);
         A11RowCnt++;
         A12RowCnt++;
      }
      else if ( A22MaxRowLeng > 0 ) // A(2,2) block
      {
         A22NewSize = 0;
         for ( j = 0; j < rowSize; j++ ) 
         {
            index = inds[j];
            searchInd = hypre_BinarySearch(P22GlobalInds_,index,P22GSize_);
            if (searchInd >= 0) 
            {
               A22_inds[A22NewSize] = searchInd;
               A22_vals[A22NewSize++] = vals[j];
            }
         }
         if ( A22NewSize == 0 )
         {
            A22_inds[0] = P22Offsets_[mypid];
            A22_vals[0] = 0.0;
            A22NewSize  = 1;
         }
         HYPRE_IJMatrixSetValues(A22mat_, 1, &A22NewSize, 
	                    (const int *) &A22RowCnt, (const int *) A22_inds,
                            (const double *) A22_vals);
         A22RowCnt++;
      }
      HYPRE_ParCSRMatrixRestoreRow(Amat_, irow, &rowSize, &inds, &vals);
   }
   delete [] A11_inds;
   delete [] A11_vals;
   delete [] A12_inds;
   delete [] A12_vals;
   delete [] A22_inds;
   delete [] A22_vals;

   //------------------------------------------------------------------
   // finally assemble the matrix 
   //------------------------------------------------------------------

   HYPRE_IJMatrixAssemble(A11mat_);
   HYPRE_IJMatrixGetObject(A11mat_, (void **) &A11mat_csr);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A11mat_csr);
   HYPRE_IJMatrixAssemble(A12mat_);
   HYPRE_IJMatrixGetObject(A12mat_, (void **) &A12mat_csr);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A12mat_csr);
   if ( A22mat_ != NULL )
   {
      HYPRE_IJMatrixAssemble(A22mat_);
      HYPRE_IJMatrixGetObject(A22mat_, (void **) &A22mat_csr);
      hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A22mat_csr);
   }
   else A22mat_csr = NULL;

   free( partition );

   if ( outputLevel_ >= 3 )
   {
      sprintf( fname, "A11.%d", mypid);
      fp = fopen( fname, "w" );
      for ( irow = A11StartRow; irow < A11StartRow+A11NRows; irow++ ) 
      {
         HYPRE_ParCSRMatrixGetRow(A11mat_csr,irow,&rowSize,&inds,&vals);
         for ( j = 0; j < rowSize; j++ )
            fprintf(fp," %9d %9d %25.16e\n", irow+1, inds[j]+1, vals[j]);
         HYPRE_ParCSRMatrixRestoreRow(A11mat_csr,irow,&rowSize,&inds,&vals);
      }
      fclose(fp);
      sprintf( fname, "A12.%d", mypid);
      fp = fopen( fname, "w" );
      for ( irow = A12StartRow; irow < A12StartRow+A12NRows; irow++ ) 
      {
         HYPRE_ParCSRMatrixGetRow(A12mat_csr,irow,&rowSize,&inds,&vals);
         for ( j = 0; j < rowSize; j++ )
            fprintf(fp, " %9d %9d %25.16e\n", irow+1, inds[j]+1, vals[j]);
         HYPRE_ParCSRMatrixRestoreRow(A12mat_csr,irow,&rowSize,&inds,&vals);
      }
      fclose(fp);
      if ( A22mat_csr != NULL )
      {
         sprintf( fname, "A22.%d", mypid);
         fp = fopen( fname, "w" );
         for ( irow = A22StartRow; irow < A22StartRow+A22NRows; irow++ ) 
         {
            HYPRE_ParCSRMatrixGetRow(A22mat_csr,irow,&rowSize,&inds,&vals);
            for ( j = 0; j < rowSize; j++ )
               fprintf(fp," %9d %9d %25.16e\n", irow+1, inds[j]+1, vals[j]);
            HYPRE_ParCSRMatrixRestoreRow(A22mat_csr,irow,&rowSize,&inds,&vals);
         }
         fclose(fp);
      }
   }
   return 0;
}

//******************************************************************************
// setup preconditioner
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::setupPrecon(HYPRE_Solver *precon, HYPRE_IJMatrix Amat,
                                  HYPRE_LSI_BLOCKP_PARAMS param_ptr)
{
   int                i, nprocs, *nsweeps, *relaxType;
   char               **targv;
#ifdef HAVE_MLI
   char               paramString[100];
#endif
   MPI_Comm           mpi_comm;
   HYPRE_ParCSRMatrix Amat_csr;

   //------------------------------------------------------------------
   // fetch machine parameters 
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject( Amat, (void **) &Amat_csr );
   HYPRE_ParCSRMatrixGetComm( Amat_csr, &mpi_comm );
   MPI_Comm_size(mpi_comm, &nprocs);
   precon = NULL;

   //------------------------------------------------------------------
   // set up the solvers and preconditioners
   //------------------------------------------------------------------

   switch( param_ptr.PrecondID_ )
   {
      case 2 : 
          HYPRE_ParCSRParaSailsCreate( mpi_comm, precon ); 
          if (param_ptr.SolverID_ == 0) HYPRE_ParCSRParaSailsSetSym(*precon,1);
          else                          HYPRE_ParCSRParaSailsSetSym(*precon,0);
          HYPRE_ParCSRParaSailsSetParams(*precon, param_ptr.PSThresh_,
                                         param_ptr.PSNLevels_);
          HYPRE_ParCSRParaSailsSetFilter(*precon, param_ptr.PSFilter_);
          break;
      case 3 :
          HYPRE_BoomerAMGCreate(precon);
          HYPRE_BoomerAMGSetMaxIter(*precon, 1);
          HYPRE_BoomerAMGSetCycleType(*precon, 1);
          HYPRE_BoomerAMGSetPrintLevel(*precon, outputLevel_);
          HYPRE_BoomerAMGSetMaxLevels(*precon, 25);
          HYPRE_BoomerAMGSetMeasureType(*precon, 0);
          HYPRE_BoomerAMGSetCoarsenType(*precon, 0);
          HYPRE_BoomerAMGSetMeasureType(*precon, 1);
          HYPRE_BoomerAMGSetStrongThreshold(*precon,param_ptr.AMGThresh_);
          HYPRE_BoomerAMGSetNumFunctions(*precon, param_ptr.AMGSystemSize_);
          nsweeps = hypre_CTAlloc(int,4,HYPRE_MEMORY_HOST);
          for ( i = 0; i < 4; i++ ) nsweeps[i] = param_ptr.AMGNSweeps_;
          HYPRE_BoomerAMGSetNumGridSweeps(*precon, nsweeps);
          relaxType = hypre_CTAlloc(int,4,HYPRE_MEMORY_HOST);
          for ( i = 0; i < 4; i++ ) relaxType[i] = param_ptr.AMGRelaxType_;
          HYPRE_BoomerAMGSetGridRelaxType(*precon, relaxType);
          //double relax_wt[25];
          //if (nprocs > 1)
          //{
          //   for ( i = 0; i < 25; i++ ) relax_wt[i] = -10.0;
          //   HYPRE_BoomerAMGSetRelaxOmega(*precon, relax_wt);
          //}
          break;
      case 4 :
          HYPRE_ParCSRPilutCreate( mpi_comm, precon );
          HYPRE_ParCSRPilutSetMaxIter( *precon, 1 );
          HYPRE_ParCSRPilutSetFactorRowSize(*precon,param_ptr.PilutFillin_);
          HYPRE_ParCSRPilutSetDropTolerance(*precon,param_ptr.PilutDropTol_);
          break;
      case 5 :
          HYPRE_EuclidCreate( mpi_comm, precon );
          targv = hypre_TAlloc(char*,  4 , HYPRE_MEMORY_HOST);
          for ( i = 0; i < 4; i++ ) targv[i] = hypre_TAlloc(char, 50, HYPRE_MEMORY_HOST);
          strcpy(targv[0], "-level");
          sprintf(targv[1], "%1d", param_ptr.EuclidNLevels_);
          strcpy(targv[2], "-sparseA");
          sprintf(targv[3], "%f", param_ptr.EuclidThresh_);
          HYPRE_EuclidSetParams(*precon, 4, targv);
          for ( i = 0; i < 4; i++ ) free(targv[i]);
          free(targv);
          break;
      case 6 :
          HYPRE_LSI_DDIlutCreate( mpi_comm, precon );
          HYPRE_LSI_DDIlutSetFillin(*precon, param_ptr.DDIlutFillin_);
          HYPRE_LSI_DDIlutSetDropTolerance(*precon, param_ptr.DDIlutDropTol_);
          break;
      case 7 :
#ifdef HAVE_ML
          HYPRE_LSI_MLCreate( mpi_comm, precon );
          HYPRE_LSI_MLSetCoarseSolver(*precon, 0);
          HYPRE_LSI_MLSetStrongThreshold(*precon, param_ptr.MLThresh_);
          HYPRE_LSI_MLSetNumPreSmoothings(*precon, param_ptr.MLNSweeps_);
          HYPRE_LSI_MLSetNumPostSmoothings(*precon,param_ptr.MLNSweeps_);
          HYPRE_LSI_MLSetPreSmoother(*precon, 1);
          HYPRE_LSI_MLSetPostSmoother(*precon, 1);
          break;
#else
          printf("blockP setupPrecon ERROR : ml not available.\n");
          exit(1);
#endif
      case 8 :
#ifdef HAVE_MLI
          HYPRE_LSI_MLICreate( mpi_comm, precon );
          sprintf(paramString, "MLI outputLevel %d", outputLevel_); 
          HYPRE_LSI_MLISetParams(*precon, paramString); 
          sprintf(paramString,"MLI strengthThreshold %e",param_ptr.MLIThresh_); 
          HYPRE_LSI_MLISetParams(*precon, paramString); 
          sprintf(paramString, "MLI method AMGSA");
          HYPRE_LSI_MLISetParams(*precon, paramString); 
          switch (param_ptr.MLIRelaxType_)
          {
             case 0 : strcpy(paramString,"MLI smoother Jacobi"); break;
             case 1 : strcpy(paramString,"MLI smoother GS"); break;
             case 2 : strcpy(paramString,"MLI smoother SGS"); break;
             case 3 : strcpy(paramString,"MLI smoother BSGS"); break;
             case 4 : strcpy(paramString,"MLI smoother ParaSails"); break;
          }
          HYPRE_LSI_MLISetParams(*precon, paramString); 
          sprintf(paramString, "MLI numSweeps %d",param_ptr.MLINSweeps_);
          HYPRE_LSI_MLISetParams(*precon, paramString); 
          sprintf(paramString, "MLI Pweight %e",param_ptr.MLIPweight_);
          HYPRE_LSI_MLISetParams(*precon, paramString); 
          sprintf(paramString, "MLI nodeDOF %d",param_ptr.MLINodeDOF_);
          HYPRE_LSI_MLISetParams(*precon, paramString); 
          sprintf(paramString, "MLI nullSpaceDim %d",param_ptr.MLINullDim_);
          HYPRE_LSI_MLISetParams(*precon, paramString); 
#else
          printf("blockP setupPrecon ERROR : mli not available.\n");
          exit(1);
#endif
   }
   return 0;
}

//******************************************************************************
// setup solver
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::setupSolver(HYPRE_Solver *solver, HYPRE_IJMatrix Amat,
                                  HYPRE_IJVector fvec, HYPRE_IJVector xvec,
                                  HYPRE_Solver precon, 
                                  HYPRE_LSI_BLOCKP_PARAMS param_ptr)
{
   int                i, nprocs, *nsweeps, *relaxType;
   MPI_Comm           mpi_comm;
   HYPRE_ParCSRMatrix Amat_csr;
   HYPRE_ParVector    f_csr, x_csr;

   //------------------------------------------------------------------
   // fetch machine parameters 
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject( Amat, (void **) &Amat_csr );
   HYPRE_IJVectorGetObject( fvec, (void **) &f_csr );
   HYPRE_IJVectorGetObject( xvec, (void **) &x_csr );
   HYPRE_ParCSRMatrixGetComm( Amat_csr, &mpi_comm );
   MPI_Comm_size(mpi_comm, &nprocs);

   //------------------------------------------------------------------
   // create solver context
   //------------------------------------------------------------------

   switch ( param_ptr.SolverID_ )
   {
      case 0 :
          HYPRE_ParCSRPCGCreate(mpi_comm, solver);
          HYPRE_ParCSRPCGSetMaxIter(*solver, param_ptr.MaxIter_ );
          HYPRE_ParCSRPCGSetTol(*solver, param_ptr.Tol_);
          HYPRE_ParCSRPCGSetLogging(*solver, outputLevel_);
          HYPRE_ParCSRPCGSetRelChange(*solver, 0);
          HYPRE_ParCSRPCGSetTwoNorm(*solver, 1);
          switch ( param_ptr.PrecondID_ )
          {
             case 1 : 
                  HYPRE_ParCSRPCGSetPrecond(*solver, HYPRE_ParCSRDiagScale,
                                         HYPRE_ParCSRDiagScaleSetup,precon);
                  break;
             case 2 : 
                  HYPRE_ParCSRPCGSetPrecond(*solver,HYPRE_ParCSRParaSailsSolve,
                                         HYPRE_ParCSRParaSailsSetup,precon);
                  break;
             case 3 : 
                  HYPRE_ParCSRPCGSetPrecond(*solver, HYPRE_BoomerAMGSolve,
                                         HYPRE_BoomerAMGSetup, precon);
                  break;
             case 4 : 
                  HYPRE_ParCSRPCGSetPrecond(*solver, HYPRE_ParCSRPilutSolve,
                                            HYPRE_ParCSRPilutSetup, precon); 
                  break;
             case 5 : 
                  HYPRE_ParCSRPCGSetPrecond(*solver, HYPRE_EuclidSolve,
                                            HYPRE_EuclidSetup, precon);
                  break;
             case 6 : 
                  HYPRE_ParCSRPCGSetPrecond(*solver, HYPRE_LSI_DDIlutSolve,
                                            HYPRE_LSI_DDIlutSetup, precon);
                  break;
             case 7 : 
#ifdef HAVE_ML
                  HYPRE_ParCSRPCGSetPrecond(*solver,HYPRE_LSI_MLSolve,
                                            HYPRE_LSI_MLSetup, precon);
#else
                  printf("blockP setupSolver ERROR : ml not available.\n");
                  exit(1);
#endif
                  break;
             case 8 : 
#ifdef HAVE_MLI
                  HYPRE_ParCSRPCGSetPrecond(*solver,HYPRE_LSI_MLISolve,
                                            HYPRE_LSI_MLISetup, precon);
#else
                  printf("blockP setupSolver ERROR : mli not available.\n");
                  exit(1);
#endif
                  break;
          }
          HYPRE_ParCSRPCGSetup(*solver, Amat_csr, f_csr, x_csr);
          break;

      case 1 :
          HYPRE_ParCSRGMRESCreate(mpi_comm, solver);
          HYPRE_ParCSRGMRESSetMaxIter(*solver, param_ptr.MaxIter_);
          HYPRE_ParCSRGMRESSetTol(*solver, param_ptr.Tol_);
          HYPRE_ParCSRGMRESSetLogging(*solver, outputLevel_);
          HYPRE_ParCSRGMRESSetKDim(*solver, 50);
          switch ( param_ptr.PrecondID_ )
          {
             case 1 : 
                  HYPRE_ParCSRGMRESSetPrecond(*solver, HYPRE_ParCSRDiagScale,
                                         HYPRE_ParCSRDiagScaleSetup,precon);
                  break;
             case 2 : 
                  HYPRE_ParCSRGMRESSetPrecond(*solver,HYPRE_ParCSRParaSailsSolve,
                                         HYPRE_ParCSRParaSailsSetup,precon);
                  break;
             case 3 : 
                  HYPRE_ParCSRGMRESSetPrecond(*solver, HYPRE_BoomerAMGSolve,
                                         HYPRE_BoomerAMGSetup, precon);
                  break;
             case 4 : 
                  HYPRE_ParCSRGMRESSetPrecond(*solver, HYPRE_ParCSRPilutSolve,
                                            HYPRE_ParCSRPilutSetup, precon); 
                  break;
             case 5 : 
                  HYPRE_ParCSRGMRESSetPrecond(*solver, HYPRE_EuclidSolve,
                                            HYPRE_EuclidSetup, precon);
                  break;
             case 6 : 
                  HYPRE_ParCSRGMRESSetPrecond(*solver, HYPRE_LSI_DDIlutSolve,
                                            HYPRE_LSI_DDIlutSetup, precon);
                  break;
             case 7 : 
#ifdef HAVE_ML
                  HYPRE_ParCSRGMRESSetPrecond(*solver,HYPRE_LSI_MLSolve,
                                              HYPRE_LSI_MLSetup, precon);
#else
                  printf("blockP setupSolver ERROR : ml not available.\n");
                  exit(1);
#endif
                  break;
             case 8 : 
#ifdef HAVE_MLI
                  HYPRE_ParCSRGMRESSetPrecond(*solver,HYPRE_LSI_MLISolve,
                                              HYPRE_LSI_MLISetup, precon);
#else
                  printf("blockP setupSolver ERROR : mli not available.\n");
                  exit(1);
#endif
                  break;
          }
          HYPRE_ParCSRGMRESSetup(*solver, Amat_csr, f_csr, x_csr);
          break;

      case 2 :
          HYPRE_BoomerAMGCreate(solver);
          HYPRE_BoomerAMGSetMaxIter(*solver, param_ptr.MaxIter_);
          HYPRE_BoomerAMGSetCycleType(*solver, 1);
          //HYPRE_BoomerAMGSetTol(*solver, 1.0e-20);
          HYPRE_BoomerAMGSetPrintLevel(*solver, outputLevel_);
          HYPRE_BoomerAMGSetMaxLevels(*solver, 25);
          HYPRE_BoomerAMGSetMeasureType(*solver, 0);
          HYPRE_BoomerAMGSetCoarsenType(*solver, 0);
          HYPRE_BoomerAMGSetMeasureType(*solver, 1);
          HYPRE_BoomerAMGSetStrongThreshold(*solver,param_ptr.AMGThresh_);
          HYPRE_BoomerAMGSetNumFunctions(*solver, param_ptr.AMGSystemSize_);
          nsweeps = hypre_CTAlloc(int,4,HYPRE_MEMORY_HOST);
          for ( i = 0; i < 4; i++ ) nsweeps[i] = param_ptr.AMGNSweeps_;
          HYPRE_BoomerAMGSetNumGridSweeps(*solver, nsweeps);
          relaxType = hypre_CTAlloc(int,4,HYPRE_MEMORY_HOST);
          for ( i = 0; i < 4; i++ ) relaxType[i] = param_ptr.AMGRelaxType_;
          HYPRE_BoomerAMGSetGridRelaxType(*solver, relaxType);
          //double relax_wt[25];
          //if (nprocs > 1)
          //{
          //   for ( i = 0; i < 25; i++ ) relax_wt[i] = -10.0;
          //   HYPRE_BoomerAMGSetRelaxOmega(*solver, relax_wt);
          //}
          precon = NULL;
          HYPRE_BoomerAMGSetup(*solver, Amat_csr, f_csr, x_csr);
          break;

      case 3 :
          HYPRE_ParCSRDiagScaleSetup(*solver, Amat_csr, f_csr, x_csr);
          precon = NULL;
          break;
   }
   return 0;
}

//******************************************************************************
// solve with block diagonal or block triangular preconditioner
// (1) for diagonal block solve :
//     (a) A11 solve
//     (b) A22 solve
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::solveBDSolve(HYPRE_IJVector x1,HYPRE_IJVector x2,
                                   HYPRE_IJVector f1,HYPRE_IJVector f2)
{
   HYPRE_ParCSRMatrix A11mat_csr, A22mat_csr;
   HYPRE_ParVector    x1_csr, x2_csr, f1_csr, f2_csr;

   //------------------------------------------------------------------
   // fetch machine paramters and matrix and vector pointers
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject( A11mat_, (void **) &A11mat_csr );
   HYPRE_IJMatrixGetObject( A22mat_, (void **) &A22mat_csr );
   HYPRE_IJVectorGetObject( F1vec_, (void **) &f1_csr );
   HYPRE_IJVectorGetObject( F2vec_, (void **) &f2_csr );
   HYPRE_IJVectorGetObject( X1vec_, (void **) &x1_csr );
   HYPRE_IJVectorGetObject( X2vec_, (void **) &x2_csr );

   //------------------------------------------------------------------
   // (1)  A22 solve
   // (2)  A11 solve
   //------------------------------------------------------------------

   if ( A22Params_.SolverID_ == 0 )
      HYPRE_ParCSRPCGSolve(A22Solver_, A22mat_csr, f2_csr, x2_csr);
   else if ( A22Params_.SolverID_ == 1 )
      HYPRE_ParCSRGMRESSolve(A22Solver_, A22mat_csr, f2_csr, x2_csr);
   else if ( A22Params_.SolverID_ == 2 )
      HYPRE_BoomerAMGSolve( A22Solver_, A22mat_csr, f2_csr, x2_csr );
   else if ( A22Params_.SolverID_ == 3 )
      HYPRE_ParCSRDiagScale( A22Solver_, A22mat_csr, f2_csr, x2_csr );
   else if ( A22Params_.SolverID_ == 9 )
      hypre_ParVectorAxpy((double) 1.0, (hypre_ParVector *) f2_csr, 
                                        (hypre_ParVector *) x2_csr);
   else 
   {
      printf("HYPRE_LSI_BlockP ERROR : invalid A22 solver.\n");
      exit(1);
   }

   if ( A11Params_.SolverID_ == 0 )
      HYPRE_ParCSRPCGSolve(A11Solver_, A11mat_csr, f1_csr, x1_csr);
   else if ( A11Params_.SolverID_ == 1 )
      HYPRE_ParCSRGMRESSolve(A11Solver_, A11mat_csr, f1_csr, x1_csr);
   else if ( A11Params_.SolverID_ == 2 )
      HYPRE_BoomerAMGSolve( A11Solver_, A11mat_csr, f1_csr, x1_csr );
   else if ( A11Params_.SolverID_ == 3 )
      HYPRE_ParCSRDiagScale( A11Solver_, A11mat_csr, f1_csr, x1_csr );
   else if ( A11Params_.SolverID_ == 9 )
      hypre_ParVectorAxpy((double) 1.0, (hypre_ParVector *) f1_csr, 
                                        (hypre_ParVector *) x1_csr);
   else
   {
      printf("HYPRE_LSI_BlockP ERROR : invalid A11 solver.\n");
      exit(1);
   }

   return 0;
}

//******************************************************************************
// solve with block triangular preconditioner
//     (a) A11 solve
//     (b) A22 solve (A_p^{-1} - delta t \hat{M}_p^{-1}) or
//     (c) A22 solve (A22^{-1})
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::solveBTSolve(HYPRE_IJVector x1,HYPRE_IJVector x2,
                                   HYPRE_IJVector f1,HYPRE_IJVector f2)
{
   HYPRE_ParCSRMatrix A11mat_csr, A22mat_csr, A12mat_csr;
   HYPRE_ParVector    x1_csr, x2_csr, f1_csr, f2_csr;

   //------------------------------------------------------------------
   // fetch machine paramters and matrix and vector pointers
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject( A11mat_, (void **) &A11mat_csr );
   HYPRE_IJMatrixGetObject( A22mat_, (void **) &A22mat_csr );
   HYPRE_IJMatrixGetObject( A12mat_, (void **) &A12mat_csr );
   HYPRE_IJVectorGetObject( F1vec_, (void **) &f1_csr );
   HYPRE_IJVectorGetObject( F2vec_, (void **) &f2_csr );
   HYPRE_IJVectorGetObject( X1vec_, (void **) &x1_csr );
   HYPRE_IJVectorGetObject( X2vec_, (void **) &x2_csr );

   //------------------------------------------------------------------
   // (1) A22 solve
   // (2) compute f1 = f1 - C x2 
   // (3) A11 solve
   //------------------------------------------------------------------

   if ( A22Params_.SolverID_ == 0 )
      HYPRE_ParCSRPCGSolve(A22Solver_, A22mat_csr, f2_csr, x2_csr);
   else if ( A22Params_.SolverID_ == 1 )
      HYPRE_ParCSRGMRESSolve(A22Solver_, A22mat_csr, f2_csr, x2_csr);
   else if ( A22Params_.SolverID_ == 2 )
      HYPRE_BoomerAMGSolve( A22Solver_, A22mat_csr, f2_csr, x2_csr );
   else if ( A22Params_.SolverID_ == 3 )
      HYPRE_ParCSRDiagScale( A22Solver_, A22mat_csr, f2_csr, x2_csr );
   else if ( A22Params_.SolverID_ == 9 )
      hypre_ParVectorAxpy((double) 1.0, (hypre_ParVector *) f2_csr, 
                                        (hypre_ParVector *) x2_csr);
   else
   {
      printf("HYPRE_LSI_BlockP ERROR : invalid A22 solver.\n");
      exit(1);
   }
   HYPRE_ParCSRMatrixMatvec(-1.0, A12mat_csr, x2_csr, 1.0, f1_csr);
   if ( A11Params_.SolverID_ == 0 )
      HYPRE_ParCSRPCGSolve(A11Solver_, A11mat_csr, f1_csr, x1_csr);
   else if ( A11Params_.SolverID_ == 1 )
      HYPRE_ParCSRGMRESSolve(A11Solver_, A11mat_csr, f1_csr, x1_csr);
   else if ( A11Params_.SolverID_ == 2 )
      HYPRE_BoomerAMGSolve( A11Solver_, A11mat_csr, f1_csr, x1_csr );
   else if ( A11Params_.SolverID_ == 3 )
      HYPRE_ParCSRDiagScale( A11Solver_, A11mat_csr, f1_csr, x1_csr );
   else if ( A11Params_.SolverID_ == 9 )
      hypre_ParVectorAxpy((double) 1.0, (hypre_ParVector *) f1_csr, 
                                        (hypre_ParVector *) x1_csr);
   else
   {
      printf("HYPRE_LSI_BlockP ERROR : invalid A11 solver.\n");
      exit(1);
   }
   return 0;
}

//******************************************************************************
// solve with block LU preconditioner
// y1 = A11 \ f1
// x2 = A22 \ (C' * y1 - f2)
// x1 = y1 - A11 \ (C x2 )
//------------------------------------------------------------------------------

int HYPRE_LSI_BlockP::solveBLUSolve(HYPRE_IJVector x1,HYPRE_IJVector x2,
                                    HYPRE_IJVector f1,HYPRE_IJVector f2)
{
   HYPRE_ParCSRMatrix A11mat_csr, A22mat_csr, A12mat_csr;
   HYPRE_ParVector    x1_csr, x2_csr, f1_csr, f2_csr, y1_csr;

   //------------------------------------------------------------------
   // fetch matrix and vector pointers
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject( A11mat_, (void **) &A11mat_csr );
   HYPRE_IJMatrixGetObject( A22mat_, (void **) &A22mat_csr );
   HYPRE_IJMatrixGetObject( A12mat_, (void **) &A12mat_csr );
   HYPRE_IJVectorGetObject( f1, (void **) &f1_csr );
   HYPRE_IJVectorGetObject( f2, (void **) &f2_csr );
   HYPRE_IJVectorGetObject( x1, (void **) &x1_csr );
   HYPRE_IJVectorGetObject( x2, (void **) &x2_csr );
   HYPRE_IJVectorGetObject( X1aux_, (void **) &y1_csr );

   //------------------------------------------------------------------
   // (1) y1 = A11 \ f1
   // (2) x2 = S \ ( f2 - C' * y1 )
   // (3) x1 = y1 - A11 \ ( C * x2 )
   //------------------------------------------------------------------

   if ( A11Params_.SolverID_ == 0 )
      HYPRE_ParCSRPCGSolve(A11Solver_, A11mat_csr, f1_csr, y1_csr);
   else if ( A11Params_.SolverID_ == 1 )
      HYPRE_ParCSRGMRESSolve(A11Solver_, A11mat_csr, f1_csr, y1_csr);
   else if ( A11Params_.SolverID_ == 2 )
      HYPRE_BoomerAMGSolve( A11Solver_, A11mat_csr, f1_csr, x1_csr );
   else if ( A11Params_.SolverID_ == 3 )
      HYPRE_ParCSRDiagScale( A11Solver_, A11mat_csr, f1_csr, x1_csr );
   else if ( A11Params_.SolverID_ == 9 )
      hypre_ParVectorAxpy((double) 1.0, (hypre_ParVector *) f1_csr, 
                                        (hypre_ParVector *) x1_csr);
   else
   {
      printf("HYPRE_LSI_BlockP ERROR : invalid A11 solver.\n");
      exit(1);
   }
   HYPRE_ParCSRMatrixMatvecT(1.0, A12mat_csr, y1_csr, -1.0, f2_csr);
   if ( A22Params_.SolverID_ == 0 )
      HYPRE_ParCSRPCGSolve(A22Solver_, A22mat_csr, f2_csr, x2_csr);
   else if ( A22Params_.SolverID_ == 1 )
      HYPRE_ParCSRGMRESSolve(A22Solver_, A22mat_csr, f2_csr, x2_csr);
   else if ( A22Params_.SolverID_ == 2 )
      HYPRE_BoomerAMGSolve( A22Solver_, A22mat_csr, f2_csr, x2_csr );
   else if ( A22Params_.SolverID_ == 3 )
      HYPRE_ParCSRDiagScale( A22Solver_, A22mat_csr, f2_csr, x2_csr );
   else if ( A22Params_.SolverID_ == 9 )
      hypre_ParVectorAxpy((double) 1.0, (hypre_ParVector *) f2_csr, 
                                        (hypre_ParVector *) x2_csr);
   else
   {
      printf("HYPRE_LSI_BlockP ERROR : invalid A22 solver.\n");
      exit(1);
   }
   HYPRE_ParCSRMatrixMatvec(-1.0, A12mat_csr, x2_csr, 0.0, f1_csr);
   if ( A11Params_.SolverID_ == 0 )
      HYPRE_ParCSRPCGSolve(A11Solver_, A11mat_csr, f1_csr, x1_csr);
   else if ( A11Params_.SolverID_ == 1 )
      HYPRE_ParCSRGMRESSolve(A11Solver_, A11mat_csr, f1_csr, x1_csr);
   else if ( A11Params_.SolverID_ == 2 )
      HYPRE_BoomerAMGSolve( A11Solver_, A11mat_csr, f1_csr, x1_csr );
   else if ( A11Params_.SolverID_ == 3 )
      HYPRE_ParCSRDiagScale( A11Solver_, A11mat_csr, f1_csr, x1_csr );
   else if ( A11Params_.SolverID_ == 9 )
      hypre_ParVectorAxpy((double) 1.0, (hypre_ParVector *) f1_csr, 
                                        (hypre_ParVector *) x1_csr);
   else
   {
      printf("HYPRE_LSI_BlockP ERROR : invalid A11 solver.\n");
      exit(1);
   }
   hypre_ParVectorAxpy((double) 1.0, (hypre_ParVector *) y1_csr, 
                                     (hypre_ParVector *) x1_csr);
   return 0;
}

