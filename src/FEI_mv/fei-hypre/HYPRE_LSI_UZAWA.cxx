/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

//***************************************************************************
// Date : Apr 26, 2002
//***************************************************************************
// system includes
//---------------------------------------------------------------------------

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if 0 /* RDF: Not sure this is really needed */
#ifdef WIN32
#define strcmp _stricmp
#endif
#endif

//***************************************************************************
// local includes
//---------------------------------------------------------------------------

#include "HYPRE.h"
#include "HYPRE_LSI_UZAWA.h"

//---------------------------------------------------------------------------
// MLI include files
//---------------------------------------------------------------------------

#ifdef HAVE_MLI
#include "HYPRE_LSI_mli.h"
#endif

//***************************************************************************
// local defines and external functions
//---------------------------------------------------------------------------

extern "C"
{
   int hypre_BoomerAMGBuildCoarseOperator(hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix*,
                                          hypre_ParCSRMatrix**);
}

//***************************************************************************
//***************************************************************************
// C-Interface data structure
//---------------------------------------------------------------------------

typedef struct HYPRE_LSI_Uzawa_Struct
{
   void *precon;
}
HYPRE_LSI_UzawaStruct;

//***************************************************************************
//***************************************************************************
// C-Interface functions to solver
//---------------------------------------------------------------------------

extern "C" int HYPRE_LSI_UzawaCreate(MPI_Comm mpi_comm, HYPRE_Solver *solver)
{
   (void) mpi_comm;
   HYPRE_LSI_UzawaStruct *cprecon = (HYPRE_LSI_UzawaStruct *)
                                     hypre_CTAlloc(HYPRE_LSI_UzawaStruct, 1, HYPRE_MEMORY_HOST);
   HYPRE_LSI_Uzawa *precon = (HYPRE_LSI_Uzawa *) new HYPRE_LSI_Uzawa(mpi_comm);
   cprecon->precon = (void *) precon;
   (*solver) = (HYPRE_Solver) cprecon;
   return 0;
}

//***************************************************************************

extern "C" int HYPRE_LSI_UzawaDestroy(HYPRE_Solver solver)
{
   int err=0;

   HYPRE_LSI_UzawaStruct *cprecon = (HYPRE_LSI_UzawaStruct *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_Uzawa *precon = (HYPRE_LSI_Uzawa *) cprecon->precon;
      if ( precon != NULL ) delete precon;
      else                  err = 1;
      free( cprecon );
   }
   return err;
}

//***************************************************************************

extern "C" int HYPRE_LSI_UzawaSetParams(HYPRE_Solver solver, char *params)
{
   int err=0;

   HYPRE_LSI_UzawaStruct *cprecon = (HYPRE_LSI_UzawaStruct *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_Uzawa *precon = (HYPRE_LSI_Uzawa *) cprecon->precon;
      err = precon->setParams(params);
   }
   return err;
}

//***************************************************************************

extern "C" int HYPRE_LSI_UzawaSetMaxIterations(HYPRE_Solver solver, int iter)
{
   int err=0;

   HYPRE_LSI_UzawaStruct *cprecon = (HYPRE_LSI_UzawaStruct *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_Uzawa *precon = (HYPRE_LSI_Uzawa *) cprecon->precon;
      err = precon->setMaxIterations(iter);
   }
   return err;
}

//***************************************************************************

extern "C" int HYPRE_LSI_UzawaSetTolerance(HYPRE_Solver solver, double tol)
{
   int err=0;

   HYPRE_LSI_UzawaStruct *cprecon = (HYPRE_LSI_UzawaStruct *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_Uzawa *precon = (HYPRE_LSI_Uzawa *) cprecon->precon;
      err = precon->setTolerance(tol);
   }
   return err;
}

//***************************************************************************

extern "C" int HYPRE_LSI_UzawaGetNumIterations(HYPRE_Solver solver, int *iter)
{
   int err=0;

   HYPRE_LSI_UzawaStruct *cprecon = (HYPRE_LSI_UzawaStruct *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_Uzawa *precon = (HYPRE_LSI_Uzawa *) cprecon->precon;
      err = precon->getNumIterations(*iter);
   }
   return err;
}

//***************************************************************************

extern "C"
int HYPRE_LSI_UzawaSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix Amat,
                         HYPRE_ParVector b, HYPRE_ParVector x)
{
   int err=0;

   (void) b;
   (void) x;
   HYPRE_LSI_UzawaStruct *cprecon = (HYPRE_LSI_UzawaStruct *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_Uzawa *precon = (HYPRE_LSI_Uzawa *) cprecon->precon;
      err = precon->setup(Amat, x, b);
   }
   return err;
}

//***************************************************************************

extern "C"
int HYPRE_LSI_UzawaSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix Amat,
                         HYPRE_ParVector b, HYPRE_ParVector x)
{
   int err=0;

   (void) Amat;
   HYPRE_LSI_UzawaStruct *cprecon = (HYPRE_LSI_UzawaStruct *) solver;
   if ( cprecon == NULL ) err = 1;
   else
   {
      HYPRE_LSI_Uzawa *precon = (HYPRE_LSI_Uzawa *) cprecon->precon;
      err = precon->solve(b, x);
   }
   return err;
}

//***************************************************************************
//***************************************************************************
// Constructor
//---------------------------------------------------------------------------

HYPRE_LSI_Uzawa::HYPRE_LSI_Uzawa(MPI_Comm comm)
{
   Amat_                     = NULL;
   A11mat_                   = NULL;
   A12mat_                   = NULL;
   S22mat_                   = NULL;
   mpiComm_                  = comm;
   outputLevel_              = 2;
   S22Scheme_                = 0;
   maxIterations_            = 1;
   tolerance_                = 1.0e-6;
   numIterations_            = 0;
   procA22Sizes_             = NULL;
   A11Solver_                = NULL;
   A11Precond_               = NULL;
   S22Solver_                = NULL;
   S22Precond_               = NULL;
   modifiedScheme_           = 0;
   S22SolverDampFactor_      = 1.0;
   A11Params_.SolverID_      = 1;       /* default : gmres */
   S22Params_.SolverID_      = 1;       /* default : cg */
   A11Params_.PrecondID_     = 1;       /* default : diagonal */
   S22Params_.PrecondID_     = 1;       /* default : diagonal */
   A11Params_.Tol_           = 1.0e-3;
   S22Params_.Tol_           = 1.0e-3;
   A11Params_.MaxIter_       = 1000;
   S22Params_.MaxIter_       = 1000;
   A11Params_.PSNLevels_     = 1;
   S22Params_.PSNLevels_     = 1;
   A11Params_.PSThresh_      = 1.0e-1;
   S22Params_.PSThresh_      = 1.0e-1;
   A11Params_.PSFilter_      = 2.0e-1;
   S22Params_.PSFilter_      = 2.0e-1;
   A11Params_.AMGThresh_     = 7.5e-1;
   S22Params_.AMGThresh_     = 7.5e-1;
   A11Params_.AMGNSweeps_    = 2;
   S22Params_.AMGNSweeps_    = 2;
   A11Params_.AMGSystemSize_ = 1;
   S22Params_.AMGSystemSize_ = 1;
   A11Params_.PilutFillin_   = 100;
   S22Params_.PilutFillin_   = 100;
   A11Params_.PilutDropTol_  = 0.1;
   S22Params_.PilutDropTol_  = 0.1;
   A11Params_.EuclidNLevels_ = 1;
   S22Params_.EuclidNLevels_ = 1;
   A11Params_.EuclidThresh_  = 0.1;
   S22Params_.EuclidThresh_  = 0.1;
   A11Params_.MLIThresh_     = 0.08;
   S22Params_.MLIThresh_     = 0.08;
   A11Params_.MLINSweeps_    = 2;
   S22Params_.MLINSweeps_    = 2;
   A11Params_.MLIPweight_    = 0.0;
   S22Params_.MLIPweight_    = 0.0;
   A11Params_.MLINodeDOF_    = 3;
   S22Params_.MLINodeDOF_    = 3;
   A11Params_.MLINullDim_    = 3;
   S22Params_.MLINullDim_    = 3;
}

//***************************************************************************
// destructor
//---------------------------------------------------------------------------

HYPRE_LSI_Uzawa::~HYPRE_LSI_Uzawa()
{
   Amat_    = NULL;
   mpiComm_ = 0;
   if ( procA22Sizes_ != NULL ) delete [] procA22Sizes_;
   if ( A11mat_       != NULL ) HYPRE_ParCSRMatrixDestroy(A11mat_);
   if ( A12mat_       != NULL ) HYPRE_ParCSRMatrixDestroy(A12mat_);
   if ( S22mat_       != NULL ) HYPRE_ParCSRMatrixDestroy(S22mat_);
}

//***************************************************************************
// set internal parameters
//---------------------------------------------------------------------------

int HYPRE_LSI_Uzawa::setParams(char *params)
{
   char   param1[256], param2[256], param3[256];

   sscanf(params,"%s", param1);
   if ( strcmp(param1, "Uzawa") )
   {
      printf("HYPRE_LSI_Uzawa::parameters not for me.\n");
      return 1;
   }
   sscanf(params,"%s %s", param1, param2);
   if ( !strcmp(param2, "help") )
   {
      printf("Available options for Uzawa are : \n");
      printf("      outputLevel <d> \n");
      printf("      A11Solver <cg,gmres> \n");
      printf("      A11Tolerance <f> \n");
      printf("      A11MaxIterations <d> \n");
      printf("      A11Precon <pilut,boomeramg,euclid,parasails,ddilut,mli>\n");
      printf("      A11PreconPSNlevels <d> \n");
      printf("      A11PreconPSThresh <f> \n");
      printf("      A11PreconPSFilter <f> \n");
      printf("      A11PreconAMGThresh <f> \n");
      printf("      A11PreconAMGNumSweeps <d> \n");
      printf("      A11PreconAMGSystemSize <d> \n");
      printf("      A11PreconEuclidNLevels <d> \n");
      printf("      A11PreconEuclidThresh <f> \n");
      printf("      A11PreconPilutFillin <d> \n");
      printf("      A11PreconPilutDropTol <f> \n");
      printf("      S22SolverDampingFactor <f> \n");
      printf("      S22Solver <cg,gmres> \n");
      printf("      S22Tolerance <f> \n");
      printf("      S22MaxIterations <d> \n");
      printf("      S22Precon <pilut,boomeramg,euclid,parasails,ddilut,mli>\n");
      printf("      S22PreconPSNlevels <d> \n");
      printf("      S22PreconPSThresh <f> \n");
      printf("      S22PreconPSFilter <f> \n");
      printf("      S22PreconAMGThresh <f> \n");
      printf("      S22PreconAMGNumSweeps <d> \n");
      printf("      S22PreconAMGSystemSize <d> \n");
      printf("      S22PreconEuclidNLevels <d> \n");
      printf("      S22PreconEuclidThresh <f> \n");
      printf("      S22PreconPilutFillin <d> \n");
      printf("      S22PreconPilutDropTol <f> \n");
   }
   else if ( !strcmp(param2, "outputLevel") )
   {
      sscanf(params,"%s %s %d", param1, param2, &outputLevel_);
      if ( outputLevel_ > 0 )
         printf("HYPRE_LSI_Uzawa::outputLevel = %d.\n", outputLevel_);
   }
   else if ( !strcmp(param2, "modified") )
   {
      modifiedScheme_ = 1;
      if ( outputLevel_ > 0 ) printf("HYPRE_LSI_Uzawa::3 level scheme.\n");
   }
   else if ( !strcmp(param2, "A11Solver") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if ( !strcmp(param3, "none") )
      {
         A11Params_.SolverID_ = 0;
         if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11 solver = cg\n");
      }
      else if ( !strcmp(param3, "cg") )
      {
         A11Params_.SolverID_ = 1;
         if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11 solver = cg\n");
      }
      else if ( !strcmp(param3, "gmres") )
      {
         A11Params_.SolverID_ = 2;
         if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11 solver = gmres\n");
      }
   }
   else if ( !strcmp(param2, "S22Solver") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if ( !strcmp(param3, "none") )
      {
         S22Params_.SolverID_ = 0;
         if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22 solver = cg\n");
      }
      else if ( !strcmp(param3, "cg") )
      {
         S22Params_.SolverID_ = 1;
         if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22 solver = cg\n");
      }
      else if ( !strcmp(param3, "gmres") )
      {
         S22Params_.SolverID_ = 2;
         if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22 solver = gmres\n");
      }
   }
   else if ( !strcmp(param2, "S22SolverDampingFactor") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &S22SolverDampFactor_);
      if ( S22SolverDampFactor_ < 0.0 ) S22SolverDampFactor_ = 1.0;
   }
   else if ( !strcmp(param2, "A11Tolerance") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.Tol_));
      if ( A11Params_.Tol_ >= 1.0 || A11Params_.Tol_ <= 0.0 )
         A11Params_.Tol_ = 1.0e-12;
      if (outputLevel_ > 0)
         printf("HYPRE_LSI_Uzawa::A11 tol = %e\n", A11Params_.Tol_);
   }
   else if ( !strcmp(param2, "S22Tolerance") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(S22Params_.Tol_));
      if ( S22Params_.Tol_ >= 1.0 || S22Params_.Tol_ <= 0.0 )
         S22Params_.Tol_ = 1.0e-12;
      if (outputLevel_ > 0)
         printf("HYPRE_LSI_Uzawa::S22 tol = %e\n", S22Params_.Tol_);
   }
   else if ( !strcmp(param2, "A11MaxIterations") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.MaxIter_));
      if ( A11Params_.MaxIter_ <= 0 ) A11Params_.MaxIter_ = 10;
      if (outputLevel_ > 0)
         printf("HYPRE_LSI_Uzawa::A11 maxiter = %d\n", A11Params_.MaxIter_);
   }
   else if ( !strcmp(param2, "S22MaxIterations") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(S22Params_.MaxIter_));
      if ( S22Params_.MaxIter_ <= 0 ) S22Params_.MaxIter_ = 10;
      if (outputLevel_ > 0)
         printf("HYPRE_LSI_Uzawa::S22 maxiter = %d\n", S22Params_.MaxIter_);
   }
   else if ( !strcmp(param2, "A11Precon") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if ( !strcmp(param3, "diagonal") )
      {
         A11Params_.PrecondID_ = 1;
         if (outputLevel_ > 0)
            printf("HYPRE_LSI_Uzawa::A11 precon = diagonal\n");
      }
      else if ( !strcmp(param3, "parasails") )
      {
         A11Params_.PrecondID_ = 2;
         if (outputLevel_ > 0)
            printf("HYPRE_LSI_Uzawa::A11 precon = parasails\n");
      }
      else if ( !strcmp(param3, "boomeramg") )
      {
         A11Params_.PrecondID_ = 3;
         if (outputLevel_ > 0)
            printf("HYPRE_LSI_Uzawa::A11 precon = boomeramg\n");
      }
      else if ( !strcmp(param3, "pilut") )
      {
         A11Params_.PrecondID_ = 4;
         if (outputLevel_ > 0)
            printf("HYPRE_LSI_Uzawa::A11 precon = pilut\n");
      }
      else if ( !strcmp(param3, "euclid") )
      {
         A11Params_.PrecondID_ = 5;
         if (outputLevel_ > 0)
            printf("HYPRE_LSI_Uzawa::A11 precon = euclid\n");
      }
      else if ( !strcmp(param3, "mli") )
      {
         A11Params_.PrecondID_ = 6;
         if (outputLevel_ > 0)
            printf("HYPRE_LSI_Uzawa::A11 precon = MLISA\n");
      }
   }
   else if ( !strcmp(param2, "S22Precon") )
   {
      sscanf(params,"%s %s %s", param1, param2, param3);
      if ( !strcmp(param3, "diagonal") )
      {
         S22Params_.PrecondID_ = 1;
         if (outputLevel_ > 0)
            printf("HYPRE_LSI_Uzawa::S22 precon = diagonal\n");
      }
      else if ( !strcmp(param3, "parasails") )
      {
         S22Params_.PrecondID_ = 2;
         if (outputLevel_ > 0)
            printf("HYPRE_LSI_Uzawa::S22 precon = parasails\n");
      }
      else if ( !strcmp(param3, "boomeramg") )
      {
         S22Params_.PrecondID_ = 3;
         if (outputLevel_ > 0)
            printf("HYPRE_LSI_Uzawa::S22 precon = boomeramg\n");
      }
      else if ( !strcmp(param3, "pilut") )
      {
         S22Params_.PrecondID_ = 4;
         if (outputLevel_ > 0)
            printf("HYPRE_LSI_Uzawa::S22 precon = pilut\n");
      }
      else if ( !strcmp(param3, "euclid") )
      {
         S22Params_.PrecondID_ = 5;
         if (outputLevel_ > 0)
            printf("HYPRE_LSI_Uzawa::S22 precon = euclid\n");
      }
      else if ( !strcmp(param3, "mli") )
      {
         S22Params_.PrecondID_ = 6;
         if (outputLevel_ > 0)
            printf("HYPRE_LSI_Uzawa::S22 precon = MLISA\n");
      }
   }
   else if ( !strcmp(param2, "A11PreconPSNlevels") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.PSNLevels_));
      if ( A11Params_.PSNLevels_ < 0 ) A11Params_.PSNLevels_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconPSNLevels\n");
   }
   else if ( !strcmp(param2, "S22PreconPSNlevels") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(S22Params_.PSNLevels_));
      if ( S22Params_.PSNLevels_ < 0 ) S22Params_.PSNLevels_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconPSNLevels\n");
   }
   else if ( !strcmp(param2, "A11PreconPSThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.PSThresh_));
      if ( A11Params_.PSThresh_ < 0 ) A11Params_.PSThresh_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconPSThresh\n");
   }
   else if ( !strcmp(param2, "S22PreconPSThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(S22Params_.PSThresh_));
      if ( S22Params_.PSThresh_ < 0 ) S22Params_.PSThresh_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconPSThresh\n");
   }
   else if ( !strcmp(param2, "A11PreconPSFilter") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.PSFilter_));
      if ( A11Params_.PSFilter_ < 0 ) A11Params_.PSFilter_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconPSFilter\n");
   }
   else if ( !strcmp(param2, "S22PreconPSFilter") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(S22Params_.PSFilter_));
      if ( S22Params_.PSFilter_ < 0 ) S22Params_.PSFilter_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconPSFilter\n");
   }
   else if ( !strcmp(param2, "A11PreconAMGThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.AMGThresh_));
      if ( A11Params_.AMGThresh_ < 0.0 ) A11Params_.AMGThresh_ = 0.0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconAMGThresh\n");
   }
   else if ( !strcmp(param2, "S22PreconAMGThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(S22Params_.AMGThresh_));
      if ( S22Params_.AMGThresh_ < 0.0 ) S22Params_.AMGThresh_ = 0.0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconAMGThresh\n");
   }
   else if ( !strcmp(param2, "A11PreconAMGNumSweeps") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.AMGNSweeps_));
      if ( A11Params_.AMGNSweeps_ < 0 ) A11Params_.AMGNSweeps_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconAMGNSweeps\n");
   }
   else if ( !strcmp(param2, "S22PreconAMGNumSweeps") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(S22Params_.AMGNSweeps_));
      if ( S22Params_.AMGNSweeps_ < 0 ) S22Params_.AMGNSweeps_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconAMGNSweeps\n");
   }
   else if ( !strcmp(param2, "A11PreconAMGSystemSize") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.AMGSystemSize_));
      if ( A11Params_.AMGSystemSize_ < 1 ) A11Params_.AMGSystemSize_ = 1;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconAMGSystemSize\n");
   }
   else if ( !strcmp(param2, "S22PreconAMGSystemSize") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(S22Params_.AMGSystemSize_));
      if ( S22Params_.AMGSystemSize_ < 1 ) S22Params_.AMGSystemSize_ = 1;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconAMGSystemSize\n");
   }
   else if ( !strcmp(param2, "A11PreconEuclidNLevels") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.EuclidNLevels_));
      if ( A11Params_.EuclidNLevels_ < 0 ) A11Params_.EuclidNLevels_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconEuclidNLevels\n");
   }
   else if ( !strcmp(param2, "S22PreconEuclidNLevels") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(S22Params_.EuclidNLevels_));
      if ( S22Params_.EuclidNLevels_ < 0 ) S22Params_.EuclidNLevels_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconEuclidNLevels\n");
   }
   else if ( !strcmp(param2, "A11PreconEuclidThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.EuclidThresh_));
      if ( A11Params_.EuclidThresh_ < 0 ) A11Params_.EuclidThresh_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconEuclidThresh\n");
   }
   else if ( !strcmp(param2, "S22PreconEuclidThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(S22Params_.EuclidThresh_));
      if ( S22Params_.EuclidThresh_ < 0 ) S22Params_.EuclidThresh_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconEuclidThresh\n");
   }
   else if ( !strcmp(param2, "A11PreconPilutFillin") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.PilutFillin_));
      if ( A11Params_.PilutFillin_ < 0 ) A11Params_.PilutFillin_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconPilutFillin\n");
   }
   else if ( !strcmp(param2, "S22PreconPilutFillin") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(S22Params_.PilutFillin_));
      if ( S22Params_.PilutFillin_ < 0 ) S22Params_.PilutFillin_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconPilutFillin\n");
   }
   else if ( !strcmp(param2, "A11PreconPilutDropTol") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.PilutDropTol_));
      if ( A11Params_.PilutDropTol_ < 0 ) A11Params_.PilutDropTol_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconPilutDropTol\n");
   }
   else if ( !strcmp(param2, "S22PreconPilutDropTol") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(S22Params_.PilutDropTol_));
      if ( S22Params_.PilutDropTol_ < 0 ) S22Params_.PilutDropTol_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconPilutDropTol\n");
   }
   else if ( !strcmp(param2, "A11PreconMLIThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.MLIThresh_));
      if ( A11Params_.MLIThresh_ < 0.0 ) A11Params_.MLIThresh_ = 0.0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconMLIThresh\n");
   }
   else if ( !strcmp(param2, "S22PreconMLIThresh") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(S22Params_.MLIThresh_));
      if ( S22Params_.MLIThresh_ < 0.0 ) S22Params_.MLIThresh_ = 0.0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconMLIThresh\n");
   }
   else if ( !strcmp(param2, "A11PreconMLINumSweeps") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.MLINSweeps_));
      if ( A11Params_.MLINSweeps_ < 0 ) A11Params_.MLINSweeps_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconMLINSweeps\n");
   }
   else if ( !strcmp(param2, "S22PreconMLINumSweeps") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(S22Params_.MLINSweeps_));
      if ( S22Params_.MLINSweeps_ < 0 ) S22Params_.MLINSweeps_ = 0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconMLINSweeps\n");
   }
   else if ( !strcmp(param2, "A11PreconMLIPweight") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(A11Params_.MLIPweight_));
      if ( A11Params_.MLIPweight_ < 0.0 ) A11Params_.MLIPweight_ = 0.0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconMLIPweight\n");
   }
   else if ( !strcmp(param2, "S22PreconMLIPweight") )
   {
      sscanf(params,"%s %s %lg", param1, param2, &(S22Params_.MLIPweight_));
      if ( S22Params_.MLIPweight_ < 0.0 ) S22Params_.MLIPweight_ = 0.0;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconMLIPweight\n");
   }
   else if ( !strcmp(param2, "A11PreconMLINodeDOF") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.MLINodeDOF_));
      if ( A11Params_.MLINodeDOF_ < 1 ) A11Params_.MLINodeDOF_ = 1;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconMLINodeDOF\n");
   }
   else if ( !strcmp(param2, "S22PreconMLINodeDOF") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(S22Params_.MLINodeDOF_));
      if ( S22Params_.MLINodeDOF_ < 1 ) S22Params_.MLINodeDOF_ = 1;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconMLINodeDOF\n");
   }
   else if ( !strcmp(param2, "A11PreconMLINullDim") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(A11Params_.MLINullDim_));
      if ( A11Params_.MLINullDim_ < 1 ) A11Params_.MLINullDim_ = 1;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::A11PreconMLINullDim\n");
   }
   else if ( !strcmp(param2, "S22PreconMLINullDim") )
   {
      sscanf(params,"%s %s %d", param1, param2, &(S22Params_.MLINullDim_));
      if ( S22Params_.MLINullDim_ < 1 ) S22Params_.MLINullDim_ = 1;
      if (outputLevel_ > 0) printf("HYPRE_LSI_Uzawa::S22PreconMLINullDim\n");
   }
   else
   {
      printf("HYPRE_LSI_Uzawa:: string not recognized %s\n", params);
   }
   return 0;
}

//***************************************************************************
// set maximum number of iterations
//---------------------------------------------------------------------------

int HYPRE_LSI_Uzawa::setMaxIterations(int niter)
{
   maxIterations_ = niter;
   return 0;
}

//***************************************************************************
// set tolerance
//---------------------------------------------------------------------------

int HYPRE_LSI_Uzawa::setTolerance(double tol)
{
   tolerance_ = tol;
   return 0;
}

//***************************************************************************
// get number of iterations
//---------------------------------------------------------------------------

int HYPRE_LSI_Uzawa::getNumIterations(int &iter)
{
   iter = numIterations_;
   return 0;
}

//***************************************************************************
// Given the matrix (A) within the object, separate the blocks
//---------------------------------------------------------------------------

int HYPRE_LSI_Uzawa::setup(HYPRE_ParCSRMatrix A, HYPRE_ParVector x,
                           HYPRE_ParVector b)
{
   int  mypid;

   //------------------------------------------------------------------
   // initial set up
   //------------------------------------------------------------------

   MPI_Comm_rank( mpiComm_, &mypid );
   if ( mypid == 0 && outputLevel_ >= 1 )
      printf("%4d : HYPRE_LSI_Uzawa begins....\n", mypid);

   Amat_ = A;

   //------------------------------------------------------------------
   // clean up first
   //------------------------------------------------------------------

   if ( procA22Sizes_ != NULL ) delete [] procA22Sizes_;
   if ( A11mat_       != NULL ) HYPRE_ParCSRMatrixDestroy(A11mat_);
   if ( A12mat_       != NULL ) HYPRE_ParCSRMatrixDestroy(A12mat_);
   if ( S22mat_       != NULL ) HYPRE_ParCSRMatrixDestroy(S22mat_);
   procA22Sizes_ = NULL;
   A11mat_       = NULL;
   A12mat_       = NULL;
   S22mat_       = NULL;

   //------------------------------------------------------------------
   // find the size of A22 block in the local processor (procA22Sizes_)
   //------------------------------------------------------------------

   if ( findA22BlockSize() == 0 ) return 0;

   //------------------------------------------------------------------
   // build the reduced matrix
   //------------------------------------------------------------------

   buildBlockMatrices();

   //------------------------------------------------------------------
   // setup preconditioners
   //------------------------------------------------------------------

   setupPrecon(&A11Precond_, A11mat_, A11Params_);
   setupPrecon(&S22Precond_, S22mat_, S22Params_);

   //------------------------------------------------------------------
   // return
   //------------------------------------------------------------------

   if ( mypid == 0 && outputLevel_ >= 1 )
      printf("%4d : HYPRE_LSI_Uzawa ends.\n", mypid);
   return 0;
}

//***************************************************************************
// Solve using the Uzawa algorithm
//---------------------------------------------------------------------------

int HYPRE_LSI_Uzawa::solve(HYPRE_ParVector b, HYPRE_ParVector x)
{
   int             mypid, *procNRows, startRow, endRow, localNRows, ierr;
   int             irow, A22NRows;
   double          rnorm0, rnorm;
   double          *b_data, *x_data, *u1_data, *u2_data, *f1_data, *f2_data;
   double          *v1_data;
   HYPRE_IJVector  IJR, IJF1, IJF2, IJU1, IJU2, IJT1, IJT2, IJV1, IJV2;
   HYPRE_ParVector r_csr, f1_csr, f2_csr, u1_csr, u2_csr, t1_csr, t2_csr;
   HYPRE_ParVector v1_csr, v2_csr;
   hypre_Vector    *b_local, *f1_local, *f2_local;
   hypre_Vector    *x_local, *u1_local, *u2_local, *v1_local;

   //------------------------------------------------------------------
   // get machine information
   //------------------------------------------------------------------

   MPI_Comm_rank( mpiComm_, &mypid );

   //------------------------------------------------------------------
   // create temporary vectors
   //------------------------------------------------------------------

   HYPRE_ParCSRMatrixGetRowPartitioning( Amat_, &procNRows );
   startRow   = procNRows[mypid];
   endRow     = procNRows[mypid+1] - 1;
   localNRows = endRow - startRow + 1;
   ierr = HYPRE_IJVectorCreate(mpiComm_, startRow, endRow, &IJR);
   ierr = HYPRE_IJVectorSetObjectType(IJR, HYPRE_PARCSR);
   ierr = HYPRE_IJVectorInitialize(IJR);
   ierr = HYPRE_IJVectorAssemble(IJR);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(IJR, (void **) &r_csr);

   startRow = procNRows[mypid] - procA22Sizes_[mypid];
   endRow   = procNRows[mypid+1] - procA22Sizes_[mypid+1] - 1;
   ierr = HYPRE_IJVectorCreate(mpiComm_, startRow, endRow, &IJF1);
   ierr = HYPRE_IJVectorSetObjectType(IJF1, HYPRE_PARCSR);
   ierr = HYPRE_IJVectorInitialize(IJF1);
   ierr = HYPRE_IJVectorAssemble(IJF1);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(IJF1, (void **) &f1_csr);
   ierr = HYPRE_IJVectorCreate(mpiComm_, startRow, endRow, &IJU1);
   ierr = HYPRE_IJVectorSetObjectType(IJU1, HYPRE_PARCSR);
   ierr = HYPRE_IJVectorInitialize(IJU1);
   ierr = HYPRE_IJVectorAssemble(IJU1);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(IJU1, (void **) &u1_csr);
   ierr = HYPRE_IJVectorCreate(mpiComm_, startRow, endRow, &IJT1);
   ierr = HYPRE_IJVectorSetObjectType(IJT1, HYPRE_PARCSR);
   ierr = HYPRE_IJVectorInitialize(IJT1);
   ierr = HYPRE_IJVectorAssemble(IJT1);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(IJT1, (void **) &t1_csr);
   ierr = HYPRE_IJVectorCreate(mpiComm_, startRow, endRow, &IJV1);
   ierr = HYPRE_IJVectorSetObjectType(IJV1, HYPRE_PARCSR);
   ierr = HYPRE_IJVectorInitialize(IJV1);
   ierr = HYPRE_IJVectorAssemble(IJV1);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(IJV1, (void **) &v1_csr);

   startRow = procA22Sizes_[mypid];
   endRow   = procA22Sizes_[mypid+1] - 1;
   ierr = HYPRE_IJVectorCreate(mpiComm_, startRow, endRow, &IJF2);
   ierr = HYPRE_IJVectorSetObjectType(IJF2, HYPRE_PARCSR);
   ierr = HYPRE_IJVectorInitialize(IJF2);
   ierr = HYPRE_IJVectorAssemble(IJF2);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(IJF2, (void **) &f2_csr);
   ierr = HYPRE_IJVectorCreate(mpiComm_, startRow, endRow, &IJU2);
   ierr = HYPRE_IJVectorSetObjectType(IJU2, HYPRE_PARCSR);
   ierr = HYPRE_IJVectorInitialize(IJU2);
   ierr = HYPRE_IJVectorAssemble(IJU2);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(IJU2, (void **) &u2_csr);
   ierr = HYPRE_IJVectorCreate(mpiComm_, startRow, endRow, &IJT2);
   ierr = HYPRE_IJVectorSetObjectType(IJT2, HYPRE_PARCSR);
   ierr = HYPRE_IJVectorInitialize(IJT2);
   ierr = HYPRE_IJVectorAssemble(IJT2);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(IJT2, (void **) &t2_csr);
   ierr = HYPRE_IJVectorCreate(mpiComm_, startRow, endRow, &IJV2);
   ierr = HYPRE_IJVectorSetObjectType(IJV2, HYPRE_PARCSR);
   ierr = HYPRE_IJVectorInitialize(IJV2);
   ierr = HYPRE_IJVectorAssemble(IJV2);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(IJV2, (void **) &v2_csr);
   free( procNRows );

   //------------------------------------------------------------------
   // compute initial residual
   //------------------------------------------------------------------

   if ( maxIterations_ > 1 )
   {
      HYPRE_ParVectorCopy( b, r_csr );
      HYPRE_ParCSRMatrixMatvec( -1.0, Amat_, x, 1.0, r_csr );
      HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
      rnorm  = sqrt( rnorm );
      rnorm0 = rnorm * tolerance_;
      if ( rnorm < rnorm0 ) return 0;
      if ( mypid == 0 ) printf("Uzawa : initial rnorm = %e\n",rnorm);
   }
   else rnorm = rnorm0 = 1.0;

   //------------------------------------------------------------------
   // set up solvers
   //------------------------------------------------------------------

   if ( A11Solver_ == NULL )
      setupSolver(&A11Solver_,A11mat_,f1_csr,u1_csr,A11Precond_,A11Params_);
   if ( S22Params_.SolverID_ != 0 && S22Solver_ == NULL )
      setupSolver(&S22Solver_,S22mat_,f2_csr,u2_csr,S22Precond_,S22Params_);

   //------------------------------------------------------------------
   // distribute the vectors
   //------------------------------------------------------------------

   A22NRows = procA22Sizes_[mypid+1] - procA22Sizes_[mypid];
   b_local  = hypre_ParVectorLocalVector((hypre_ParVector *) b);
   b_data   = (double *) hypre_VectorData(b_local);
   f1_local = hypre_ParVectorLocalVector((hypre_ParVector *) f1_csr);
   f1_data  = (double *) hypre_VectorData(f1_local);
   f2_local = hypre_ParVectorLocalVector((hypre_ParVector *) f2_csr);
   f2_data  = (double *) hypre_VectorData(f2_local);
   x_local  = hypre_ParVectorLocalVector((hypre_ParVector *) x);
   x_data   = (double *) hypre_VectorData(x_local);
   u1_local = hypre_ParVectorLocalVector((hypre_ParVector *) u1_csr);
   u1_data  = (double *) hypre_VectorData(u1_local);
   u2_local = hypre_ParVectorLocalVector((hypre_ParVector *) u2_csr);
   u2_data  = (double *) hypre_VectorData(u2_local);
   v1_local = hypre_ParVectorLocalVector((hypre_ParVector *) v1_csr);
   v1_data  = (double *) hypre_VectorData(v1_local);

   for ( irow = 0; irow < localNRows-A22NRows; irow++ )
      f1_data[irow] = b_data[irow];
   for ( irow = localNRows-A22NRows; irow < localNRows; irow++ )
      f2_data[irow-localNRows+A22NRows] = b_data[irow];

   //------------------------------------------------------------------
   // iterate (using the Cheng-Zou method)
   //------------------------------------------------------------------

   numIterations_ = 0;

   while ( numIterations_ < maxIterations_ && rnorm >= rnorm0 )
   {
      numIterations_++;

      //----------------------------------------------------------------
      // copy x to split vectors
      //---------------------------------------------------------------

      for ( irow = 0; irow < localNRows-A22NRows; irow++ )
         v1_data[irow] = u1_data[irow] = x_data[irow];
      for ( irow = localNRows-A22NRows; irow < localNRows; irow++ )
         u2_data[irow-localNRows+A22NRows] = x_data[irow];

      //----------------------------------------------------------------
      // x_{i+1/2} = x_i + Q_A^{-1} (f1 - A x_i - A12 y_i)
      //---------------------------------------------------------------

      HYPRE_ParVectorCopy( f1_csr, t1_csr );
      HYPRE_ParCSRMatrixMatvec( -1.0, A11mat_, u1_csr, 1.0, t1_csr );
      HYPRE_ParCSRMatrixMatvec( -1.0, A12mat_, u2_csr, 1.0, t1_csr );

      if ( A11Params_.SolverID_ == 1 )
         HYPRE_ParCSRPCGSolve(A11Solver_, A11mat_, t1_csr, u1_csr);
      else if ( A11Params_.SolverID_ == 2 )
         HYPRE_ParCSRGMRESSolve(A11Solver_, A11mat_, t1_csr, u1_csr);

      hypre_ParVectorAxpy( 1.0, (hypre_ParVector*)v1_csr ,
                                (hypre_ParVector*)u1_csr );

      //----------------------------------------------------------------
      // y_{i+1/2} = y_i + Q_B^{-1} (A21 x_{i+1/2} - f2)
      //---------------------------------------------------------------

      if ( modifiedScheme_ >= 1 )
      {
         HYPRE_ParVectorCopy( f2_csr, t2_csr );
         HYPRE_ParCSRMatrixMatvecT( 1.0, A12mat_, u1_csr, -1.0, t2_csr );

         if ( S22Params_.SolverID_ == 1 )
            HYPRE_ParCSRPCGSolve(S22Solver_, S22mat_, t2_csr, v2_csr);
         else if ( S22Params_.SolverID_ == 2 )
            HYPRE_ParCSRGMRESSolve(S22Solver_, S22mat_, t2_csr, v2_csr);
         else
         {
            HYPRE_ParVectorCopy( t2_csr, v2_csr );
            HYPRE_ParVectorScale( S22SolverDampFactor_, v2_csr );
         }

         hypre_ParVectorAxpy( 1.0, (hypre_ParVector*)v2_csr ,
                                   (hypre_ParVector*)u2_csr );

         //----------------------------------------------------------------
         // x_{i+1} = x_i + Q_A^{-1} (f - A x_i - A12 y_{i+1/2})
         //---------------------------------------------------------------

         HYPRE_ParVectorCopy( f1_csr, t1_csr );
         HYPRE_ParCSRMatrixMatvec( -1.0, A11mat_, v1_csr, 1.0, t1_csr );
         HYPRE_ParCSRMatrixMatvec( -1.0, A12mat_, u2_csr, 1.0, t1_csr );

         if ( A11Params_.SolverID_ == 1 )
            HYPRE_ParCSRPCGSolve(A11Solver_, A11mat_, t1_csr, u1_csr);
         else if ( A11Params_.SolverID_ == 2 )
            HYPRE_ParCSRGMRESSolve(A11Solver_, A11mat_, t1_csr, u1_csr);

         hypre_ParVectorAxpy( 1.0, (hypre_ParVector*)v1_csr ,
                                   (hypre_ParVector*)u1_csr );

         //----------------------------------------------------------------
         // y_{i+1} = y_{i+1/2} + Q_B^{-1} (A21 x_{i+1} - f2)
         //---------------------------------------------------------------

         HYPRE_ParVectorCopy( f2_csr, t2_csr );
         HYPRE_ParCSRMatrixMatvecT( 1.0, A12mat_, u1_csr, -1.0, t2_csr );

         if ( S22Params_.SolverID_ == 1 )
            HYPRE_ParCSRPCGSolve(S22Solver_, S22mat_, t2_csr, v2_csr);
         else if ( S22Params_.SolverID_ == 2 )
            HYPRE_ParCSRGMRESSolve(S22Solver_, S22mat_, t2_csr, v2_csr);
         else
         {
            HYPRE_ParVectorCopy( t2_csr, v2_csr );
            HYPRE_ParVectorScale( S22SolverDampFactor_, v2_csr );
         }

         hypre_ParVectorAxpy( 1.0, (hypre_ParVector*)v2_csr ,
                                   (hypre_ParVector*)u2_csr );
      }

      //----------------------------------------------------------------
      // merge solution vector and compute residual norm
      //---------------------------------------------------------------

      for ( irow = 0; irow < localNRows-A22NRows; irow++ )
         x_data[irow] = u1_data[irow];
      for ( irow = localNRows-A22NRows; irow < localNRows; irow++ )
         x_data[irow] = u2_data[irow-localNRows+A22NRows];

      if ( maxIterations_ > 1 )
      {
         HYPRE_ParVectorCopy( b, r_csr );
         HYPRE_ParCSRMatrixMatvec( -1.0, Amat_, x, 1.0, r_csr );
         HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
         rnorm  = sqrt( rnorm );
         if ( mypid == 0 )
            printf("Uzawa : iteration = %5d, rnorm = %e\n",numIterations_,
                   rnorm);
      }
   }
   HYPRE_IJVectorDestroy(IJR);
   HYPRE_IJVectorDestroy(IJF1);
   HYPRE_IJVectorDestroy(IJF2);
   HYPRE_IJVectorDestroy(IJU1);
   HYPRE_IJVectorDestroy(IJU2);
   HYPRE_IJVectorDestroy(IJT1);
   HYPRE_IJVectorDestroy(IJT2);
   HYPRE_IJVectorDestroy(IJV2);
   return 0;
}

//***************************************************************************
// search for the separator for
//    A = |A_11    A_12|
//        |A_12^T  A_22|
//---------------------------------------------------------------------------

int HYPRE_LSI_Uzawa::findA22BlockSize()
{
   int    mypid, nprocs, *procNRows, startRow, endRow;
   int    A22LocalSize, irow, zeroDiag, jcol, rowSize, *colInd;
   int    *iTempList, ip, ncnt, A22GlobalSize;
   double *colVal;

   //------------------------------------------------------------------
   // get matrix information
   //------------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   MPI_Comm_size(mpiComm_, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning( Amat_, &procNRows );
   startRow     = procNRows[mypid];
   endRow       = procNRows[mypid+1] - 1;
   free( procNRows );

   //------------------------------------------------------------------
   // search for dimension of A_22
   //------------------------------------------------------------------

   A22LocalSize = 0;
   for ( irow = endRow; irow >= startRow; irow-- )
   {
      HYPRE_ParCSRMatrixGetRow(Amat_,irow,&rowSize,&colInd,&colVal);
      zeroDiag = 1;
      for ( jcol = 0;  jcol < rowSize;  jcol++ )
      {
         if ( colInd[jcol] == irow && colVal[jcol] != 0.0 )
         {
            zeroDiag = 0;
            break;
         }
      }
      HYPRE_ParCSRMatrixRestoreRow(Amat_,irow,&rowSize,&colInd,&colVal);
      if ( zeroDiag ) A22LocalSize++;
      else            break;
   }
   if ( outputLevel_ >= 1 )
      printf("%4d : findA22BlockSize - local nrows = %d\n",mypid,A22LocalSize);

   //------------------------------------------------------------------
   // gather the block size information on all processors
   //------------------------------------------------------------------

   iTempList = new int[nprocs];
   if ( procA22Sizes_ != NULL ) delete [] procA22Sizes_;
   procA22Sizes_ = new int[nprocs+1];
   for ( ip = 0; ip < nprocs; ip++ ) iTempList[ip] = 0;
   iTempList[mypid] = A22LocalSize;
   MPI_Allreduce(iTempList,procA22Sizes_,nprocs,MPI_INT,MPI_SUM,mpiComm_);
   delete [] iTempList;
   A22GlobalSize = 0;
   ncnt = 0;
   for ( ip = 0; ip < nprocs; ip++ )
   {
      ncnt = procA22Sizes_[ip];
      procA22Sizes_[ip] = A22GlobalSize;
      A22GlobalSize += ncnt;
   }
   procA22Sizes_[nprocs] = A22GlobalSize;
   return A22GlobalSize;
}

//****************************************************************************
// build the block matrices A11, A21, and S22
//----------------------------------------------------------------------------

int HYPRE_LSI_Uzawa::buildBlockMatrices()
{
   int  ierr=0;

   ierr += buildA11A12Mat();
   ierr += buildS22Mat();
   return ierr;
}

//****************************************************************************
// build A11 and A12 matrix
//----------------------------------------------------------------------------

int HYPRE_LSI_Uzawa::buildA11A12Mat()
{
   int    mypid, nprocs, *procNRows, startRow, endRow, newEndRow, ierr;
   int    A12NCols, A11NRows, A11StartRow, A12StartCol, *A11MatSize, ip;
   int    *A12MatSize, irow, jcol, colIndex, uBound, A11RowSize, A12RowSize;
   int    *A11ColInd, *A12ColInd, rowIndex, rowSize, *colInd, ncnt;
   int    localNRows, maxA11RowSize, maxA12RowSize;
   double *colVal, *A11ColVal, *A12ColVal;
   HYPRE_IJMatrix     IJA11, IJA12;

   //------------------------------------------------------------------
   // get matrix information
   //------------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   MPI_Comm_size(mpiComm_, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning( Amat_, &procNRows );
   startRow      = procNRows[mypid];
   endRow        = procNRows[mypid+1] - 1;
   localNRows    = endRow - startRow + 1;
   newEndRow     = endRow - (procA22Sizes_[mypid+1] - procA22Sizes_[mypid]);

   //------------------------------------------------------------------
   // calculate the dimension of A11 and A12
   //------------------------------------------------------------------

   A12NCols       = procA22Sizes_[mypid+1] - procA22Sizes_[mypid];
   A11NRows       = localNRows - A12NCols;
   A11StartRow    = procNRows[mypid] - procA22Sizes_[mypid];
   A12StartCol    = procA22Sizes_[mypid];

   if ( outputLevel_ >= 1 )
   {
      printf("%4d : buildA11A12Mat - A11StartRow  = %d\n", mypid, A11StartRow);
      printf("%4d : buildA11A12Mat - A11LocalDim  = %d\n", mypid, A11NRows);
      printf("%4d : buildA11A12Mat - A12StartRow  = %d\n", mypid, A12StartCol);
      printf("%4d : buildA11A12Mat - A12LocalCol  = %d\n", mypid, A12NCols);
   }

   //------------------------------------------------------------------
   // create a matrix context for A11 and A12
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixCreate(mpiComm_,A11StartRow,A11StartRow+A11NRows-1,
                                A11StartRow,A11StartRow+A11NRows-1,&IJA11);
   ierr += HYPRE_IJMatrixSetObjectType(IJA11, HYPRE_PARCSR);
   hypre_assert(!ierr);
   ierr  = HYPRE_IJMatrixCreate(mpiComm_,A11StartRow,A11StartRow+A11NRows-1,
                                A12StartCol,A12StartCol+A12NCols-1,&IJA12);
   ierr += HYPRE_IJMatrixSetObjectType(IJA12, HYPRE_PARCSR);
   hypre_assert(!ierr);

   //------------------------------------------------------------------
   // compute the number of nonzeros in each matrix
   //------------------------------------------------------------------

   A11MatSize = new int[A11NRows];
   A12MatSize = new int[A11NRows];
   maxA11RowSize = maxA12RowSize = 0;

   for ( irow = startRow; irow <= newEndRow ; irow++ )
   {
      A11RowSize = A12RowSize = 0;
      HYPRE_ParCSRMatrixGetRow(Amat_,irow,&rowSize,&colInd,NULL);
      for ( jcol = 0;  jcol < rowSize;  jcol++ )
      {
         colIndex = colInd[jcol];
         for ( ip = 1; ip <= nprocs; ip++ )
            if ( procNRows[ip] > colIndex ) break;
         uBound = procNRows[ip] - (procA22Sizes_[ip] - procA22Sizes_[ip-1]);
         if ( colIndex < uBound ) A11RowSize++;
         else                     A12RowSize++;
      }
      A11MatSize[irow-startRow] = A11RowSize;
      A12MatSize[irow-startRow] = A12RowSize;
      maxA11RowSize = (A11RowSize > maxA11RowSize) ? A11RowSize : maxA11RowSize;
      maxA12RowSize = (A12RowSize > maxA12RowSize) ? A12RowSize : maxA12RowSize;
      HYPRE_ParCSRMatrixRestoreRow(Amat_,irow,&rowSize,&colInd,NULL);
   }

   //------------------------------------------------------------------
   // after fetching the row sizes, initialize the matrices
   //------------------------------------------------------------------

   ierr  = HYPRE_IJMatrixSetRowSizes(IJA11, A11MatSize);
   ierr += HYPRE_IJMatrixInitialize(IJA11);
   hypre_assert(!ierr);
   ierr  = HYPRE_IJMatrixSetRowSizes(IJA12, A12MatSize);
   ierr += HYPRE_IJMatrixInitialize(IJA12);
   hypre_assert(!ierr);

   //------------------------------------------------------------------
   // next load the matrices
   //------------------------------------------------------------------

   A11ColInd = new int[maxA11RowSize+1];
   A11ColVal = new double[maxA11RowSize+1];
   A12ColInd = new int[maxA12RowSize+1];
   A12ColVal = new double[maxA12RowSize+1];

   for ( irow = startRow; irow <= newEndRow ; irow++ )
   {
      A11RowSize = A12RowSize = 0;
      HYPRE_ParCSRMatrixGetRow(Amat_,irow,&rowSize,&colInd,&colVal);
      for ( jcol = 0;  jcol < rowSize;  jcol++ )
      {
         colIndex = colInd[jcol];
         for ( ip = 1; ip <= nprocs; ip++ )
            if ( procNRows[ip] > colIndex ) break;
         uBound = procNRows[ip] - (procA22Sizes_[ip] - procA22Sizes_[ip-1]);
         if ( colIndex < uBound )
         {
            A11ColInd[A11RowSize] = colIndex - procA22Sizes_[ip-1];
            A11ColVal[A11RowSize++] = colVal[jcol];
         }
         else
         {
            A12ColInd[A12RowSize] = colIndex - uBound + procA22Sizes_[ip-1];
            A12ColVal[A12RowSize++] = colVal[jcol];
         }
      }
      HYPRE_ParCSRMatrixRestoreRow(Amat_,irow,&rowSize,&colInd,&colVal);
      rowIndex = irow - procA22Sizes_[mypid];
      ierr = HYPRE_IJMatrixSetValues(IJA11, 1, &A11RowSize,
                   (const int *) &rowIndex, (const int *) A11ColInd,
                   (const double *) A11ColVal);
      hypre_assert( !ierr );
      ierr = HYPRE_IJMatrixSetValues(IJA12, 1, &A12RowSize,
                   (const int *) &rowIndex, (const int *) A12ColInd,
                   (const double *) A12ColVal);
      hypre_assert( !ierr );
   }

   //------------------------------------------------------------------
   // finally assemble the matrix and sanitize
   //------------------------------------------------------------------

   HYPRE_IJMatrixAssemble(IJA11);
   HYPRE_IJMatrixGetObject(IJA11, (void **) &A11mat_);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A11mat_);
   HYPRE_IJMatrixAssemble(IJA12);
   HYPRE_IJMatrixGetObject(IJA12, (void **) &A12mat_);
   hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) A11mat_);

   HYPRE_IJMatrixSetObjectType(IJA11, -1);
   HYPRE_IJMatrixDestroy(IJA11);
   HYPRE_IJMatrixSetObjectType(IJA12, -1);
   HYPRE_IJMatrixDestroy(IJA12);
   delete [] A11MatSize;
   delete [] A12MatSize;
   delete [] A11ColInd;
   delete [] A11ColVal;
   delete [] A12ColInd;
   delete [] A12ColVal;
   free( procNRows );

   //------------------------------------------------------------------
   // diagnostics
   //------------------------------------------------------------------

   if ( outputLevel_ >= 3 )
   {
      ncnt = 0;
      MPI_Barrier(mpiComm_);
      while ( ncnt < nprocs )
      {
         if ( mypid == ncnt )
         {
            printf("====================================================\n");
            printf("%4d : Printing A11 matrix... \n", mypid);
            fflush(stdout);
            for (irow = A11StartRow;irow < A11StartRow+A11NRows;irow++)
            {
               HYPRE_ParCSRMatrixGetRow(A11mat_,irow,&rowSize,&colInd,&colVal);
               for ( jcol = 0; jcol < rowSize; jcol++ )
                  if ( colVal[jcol] != 0.0 )
                     printf("%6d  %6d  %25.16e \n",irow+1,colInd[jcol]+1,
                            colVal[jcol]);
               HYPRE_ParCSRMatrixRestoreRow(A11mat_, irow, &rowSize,
                                            &colInd, &colVal);
            }
            printf("====================================================\n");
         }
         ncnt++;
         MPI_Barrier(mpiComm_);
      }
   }
   if ( outputLevel_ >= 3 )
   {
      ncnt = 0;
      MPI_Barrier(mpiComm_);
      while ( ncnt < nprocs )
      {
         if ( mypid == ncnt )
         {
            printf("====================================================\n");
            printf("%4d : Printing A12 matrix... \n", mypid);
            fflush(stdout);
            for (irow = A11StartRow;irow < A11StartRow+A11NRows;irow++)
            {
               HYPRE_ParCSRMatrixGetRow(A12mat_,irow,&rowSize,&colInd,&colVal);
               for ( jcol = 0; jcol < rowSize; jcol++ )
                  if ( colVal[jcol] != 0.0 )
                     printf("%6d  %6d  %25.16e \n",irow+1,colInd[jcol]+1,
                            colVal[jcol]);
               HYPRE_ParCSRMatrixRestoreRow(A12mat_, irow, &rowSize,
                                            &colInd, &colVal);
            }
            printf("====================================================\n");
         }
         ncnt++;
         MPI_Barrier(mpiComm_);
      }
   }
   return 0;
}

//****************************************************************************
// build the block matrix S22 (B diag(A)^{-1} B^T)
//----------------------------------------------------------------------------

int HYPRE_LSI_Uzawa::buildS22Mat()
{
   int    mypid, nprocs, *procNRows, one=1, A11StartRow, A11NRows, irow, jcol;
   int    rowSize, *colInd, *A11MatSize, ierr;
   double *colVal, ddata;
   HYPRE_ParCSRMatrix ainvA11_csr;
   HYPRE_IJMatrix     ainvA11;
   HYPRE_Solver       parasails;

   //------------------------------------------------------------------
   // get machine information
   //------------------------------------------------------------------

   MPI_Comm_rank(mpiComm_, &mypid);
   MPI_Comm_size(mpiComm_, &nprocs);

   //------------------------------------------------------------------
   // choose between diagonal or approximate inverse
   //------------------------------------------------------------------

   if ( S22Scheme_ == 1 )
   {
      //---------------------------------------------------------------
      // build approximate inverse of A11
      //---------------------------------------------------------------

      HYPRE_ParaSailsCreate(mpiComm_, &parasails);
      HYPRE_ParaSailsSetParams(parasails, 0.1, 1);
      HYPRE_ParaSailsSetFilter(parasails, 0.1);
      HYPRE_ParaSailsSetLogging(parasails, 1);
      HYPRE_ParaSailsSetup(parasails, A11mat_, NULL, NULL);
      HYPRE_ParaSailsBuildIJMatrix(parasails, &ainvA11);
   }
   else
   {
      //---------------------------------------------------------------
      // build inverse of diagonal of A11
      //---------------------------------------------------------------

      HYPRE_ParCSRMatrixGetRowPartitioning( A11mat_, &procNRows );
      A11StartRow = procNRows[mypid];
      A11NRows    = procNRows[mypid+1] - A11StartRow;

      ierr  = HYPRE_IJMatrixCreate(mpiComm_,A11StartRow,
                    A11StartRow+A11NRows-1, A11StartRow,
                    A11StartRow+A11NRows-1,&ainvA11);
      ierr += HYPRE_IJMatrixSetObjectType(ainvA11, HYPRE_PARCSR);
      hypre_assert(!ierr);

      A11MatSize = new int[A11NRows];
      for ( irow = 0; irow < A11NRows; irow++ ) A11MatSize[irow] = 1;
      ierr  = HYPRE_IJMatrixSetRowSizes(ainvA11, A11MatSize);
      ierr += HYPRE_IJMatrixInitialize(ainvA11);
      hypre_assert(!ierr);

      for ( irow = A11StartRow; irow < A11StartRow+A11NRows; irow++ )
      {
         HYPRE_ParCSRMatrixGetRow(A11mat_,irow,&rowSize,&colInd,&colVal);
         ddata = 0.0;
         for ( jcol = 0; jcol < rowSize; jcol++ )
         {
            if ( colInd[jcol] == irow )
            {
               ddata = 1.0 / colVal[jcol];
               break;
            }
         }
         HYPRE_ParCSRMatrixRestoreRow(A11mat_,irow,&rowSize,&colInd,&colVal);
         ierr = HYPRE_IJMatrixSetValues(ainvA11, 1, &one, (const int *) &irow,
                              (const int *) &irow, (const double *) &ddata);
         hypre_assert( !ierr );
      }
      HYPRE_IJMatrixAssemble(ainvA11);
      free( procNRows );
      delete [] A11MatSize;
   }

   //------------------------------------------------------------------
   // perform the triple matrix product A12' * diagA11 * A12
   //------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(ainvA11, (void **) &ainvA11_csr);
   hypre_BoomerAMGBuildCoarseOperator((hypre_ParCSRMatrix *) A12mat_,
                                      (hypre_ParCSRMatrix *) ainvA11_csr,
                                      (hypre_ParCSRMatrix *) A12mat_,
                                      (hypre_ParCSRMatrix **) &S22mat_);

   //------------------------------------------------------------------
   // clean up and return
   //------------------------------------------------------------------

   HYPRE_IJMatrixDestroy(ainvA11);
   return 0;
}

//***************************************************************************
// setup preconditioner
//---------------------------------------------------------------------------

int HYPRE_LSI_Uzawa::setupPrecon(HYPRE_Solver *precon,HYPRE_ParCSRMatrix Amat,
                                 HYPRE_Uzawa_PARAMS paramPtr)
{
   int  i, *nsweeps, *relaxType;
   char **targv;
#ifdef HAVE_MLI
   char paramString[100];
#endif

   (void) Amat;
   if ( paramPtr.SolverID_ == 0 ) return 0;

   //------------------------------------------------------------------
   // set up the solvers and preconditioners
   //------------------------------------------------------------------

   switch( paramPtr.PrecondID_ )
   {
      case 2 :
          HYPRE_ParCSRParaSailsCreate( mpiComm_, precon );
          if (paramPtr.SolverID_ == 0) HYPRE_ParCSRParaSailsSetSym(*precon,1);
          else                         HYPRE_ParCSRParaSailsSetSym(*precon,0);
          HYPRE_ParCSRParaSailsSetParams(*precon, paramPtr.PSThresh_,
                                         paramPtr.PSNLevels_);
          HYPRE_ParCSRParaSailsSetFilter(*precon, paramPtr.PSFilter_);
          break;
      case 3 :
          HYPRE_BoomerAMGCreate(precon);
          HYPRE_BoomerAMGSetMaxIter(*precon, 1);
          HYPRE_BoomerAMGSetCycleType(*precon, 1);
          HYPRE_BoomerAMGSetPrintLevel(*precon, outputLevel_);
          HYPRE_BoomerAMGSetMaxLevels(*precon, 25);
          HYPRE_BoomerAMGSetMeasureType(*precon, 0);
          HYPRE_BoomerAMGSetCoarsenType(*precon, 0);
          HYPRE_BoomerAMGSetStrongThreshold(*precon,paramPtr.AMGThresh_);
          if ( paramPtr.AMGSystemSize_ > 1 )
             HYPRE_BoomerAMGSetNumFunctions(*precon,paramPtr.AMGSystemSize_);
          nsweeps = hypre_CTAlloc(int,4,HYPRE_MEMORY_HOST);
          for ( i = 0; i < 4; i++ ) nsweeps[i] = paramPtr.AMGNSweeps_;
          HYPRE_BoomerAMGSetNumGridSweeps(*precon, nsweeps);
          relaxType = hypre_CTAlloc(int,4,HYPRE_MEMORY_HOST);
          for ( i = 0; i < 4; i++ ) relaxType[i] = 6;
          HYPRE_BoomerAMGSetGridRelaxType(*precon, relaxType);
          break;
      case 4 :
          HYPRE_ParCSRPilutCreate( mpiComm_, precon );
          HYPRE_ParCSRPilutSetMaxIter( *precon, 1 );
          HYPRE_ParCSRPilutSetFactorRowSize(*precon,paramPtr.PilutFillin_);
          HYPRE_ParCSRPilutSetDropTolerance(*precon,paramPtr.PilutDropTol_);
          break;
      case 5 :
          HYPRE_EuclidCreate( mpiComm_, precon );
          targv = hypre_TAlloc(char*,  4 , HYPRE_MEMORY_HOST);
          for ( i = 0; i < 4; i++ ) targv[i] = hypre_TAlloc(char, 50, HYPRE_MEMORY_HOST);
          strcpy(targv[0], "-level");
          sprintf(targv[1], "%1d", paramPtr.EuclidNLevels_);
          strcpy(targv[2], "-sparseA");
          sprintf(targv[3], "%f", paramPtr.EuclidThresh_);
          HYPRE_EuclidSetParams(*precon, 4, targv);
          for ( i = 0; i < 4; i++ ) free(targv[i]);
          free(targv);
          break;
      case 6 :
#ifdef HAVE_MLI
          HYPRE_LSI_MLICreate(mpiComm_, precon);
          sprintf(paramString, "MLI outputLevel %d", outputLevel_);
          HYPRE_LSI_MLISetParams(*precon, paramString);
          sprintf(paramString, "MLI strengthThreshold %e",paramPtr.MLIThresh_);
          HYPRE_LSI_MLISetParams(*precon, paramString);
          sprintf(paramString, "MLI method AMGSA");
          HYPRE_LSI_MLISetParams(*precon, paramString);
          sprintf(paramString, "MLI smoother SGS");
          HYPRE_LSI_MLISetParams(*precon, paramString);
          sprintf(paramString, "MLI numSweeps %d",paramPtr.MLINSweeps_);
          HYPRE_LSI_MLISetParams(*precon, paramString);
          sprintf(paramString, "MLI Pweight %e",paramPtr.MLIPweight_);
          HYPRE_LSI_MLISetParams(*precon, paramString);
          sprintf(paramString, "MLI nodeDOF %d",paramPtr.MLINodeDOF_);
          HYPRE_LSI_MLISetParams(*precon, paramString);
          sprintf(paramString, "MLI nullSpaceDim %d",paramPtr.MLINullDim_);
          HYPRE_LSI_MLISetParams(*precon, paramString);
#else
          printf("Uzawa setupPrecon ERROR : mli not available.\n");
          exit(1);
#endif
          break;
   }
   return 0;
}

//***************************************************************************
// setup solver
//---------------------------------------------------------------------------

int HYPRE_LSI_Uzawa::setupSolver(HYPRE_Solver *solver,HYPRE_ParCSRMatrix Amat,
                          HYPRE_ParVector fvec, HYPRE_ParVector xvec,
                          HYPRE_Solver precon, HYPRE_Uzawa_PARAMS paramPtr)
{
   //------------------------------------------------------------------
   // create solver context
   //------------------------------------------------------------------

   switch ( paramPtr.SolverID_ )
   {
      case 1 :
          HYPRE_ParCSRPCGCreate(mpiComm_, solver);
          HYPRE_ParCSRPCGSetMaxIter(*solver, paramPtr.MaxIter_ );
          HYPRE_ParCSRPCGSetTol(*solver, paramPtr.Tol_);
          HYPRE_ParCSRPCGSetLogging(*solver, outputLevel_);
          HYPRE_ParCSRPCGSetRelChange(*solver, 0);
          HYPRE_ParCSRPCGSetTwoNorm(*solver, 1);
          switch ( paramPtr.PrecondID_ )
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
#ifdef HAVE_MLI
                  HYPRE_ParCSRPCGSetPrecond(*solver,HYPRE_LSI_MLISolve,
                                            HYPRE_LSI_MLISetup, precon);
#else
                  printf("Uzawa setupSolver ERROR : mli not available.\n");
                  exit(1);
#endif
                  break;
          }
          HYPRE_ParCSRPCGSetup(*solver, Amat, fvec, xvec);
          break;

      case 2 :
          HYPRE_ParCSRGMRESCreate(mpiComm_, solver);
          HYPRE_ParCSRGMRESSetMaxIter(*solver, paramPtr.MaxIter_ );
          HYPRE_ParCSRGMRESSetTol(*solver, paramPtr.Tol_);
          HYPRE_ParCSRGMRESSetLogging(*solver, outputLevel_);
          HYPRE_ParCSRGMRESSetKDim(*solver, 50);
          switch ( paramPtr.PrecondID_ )
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
#ifdef HAVEL_MLI
                  HYPRE_ParCSRGMRESSetPrecond(*solver,HYPRE_LSI_MLISolve,
                                              HYPRE_LSI_MLISetup, precon);
#else
                  printf("Uzawa setupSolver ERROR : mli not available.\n");
                  exit(1);
#endif
                  break;
          }
          HYPRE_ParCSRGMRESSetup(*solver, Amat, fvec, xvec);
          break;
   }
   return 0;
}

