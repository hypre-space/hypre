/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Incomplete LU factorization smoother
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_ILUCreate
 *--------------------------------------------------------------------------*/

void *
hypre_ILUCreate( void )
{
   hypre_ParILUData  *ilu_data;
   hypre_Solver      *base;

   ilu_data = hypre_CTAlloc(hypre_ParILUData, 1, HYPRE_MEMORY_HOST);
   base     = (hypre_Solver*) ilu_data;

   /* Set base solver function pointers */
   hypre_SolverSetup(base)   = (HYPRE_PtrToSolverFcn)  HYPRE_ILUSetup;
   hypre_SolverSolve(base)   = (HYPRE_PtrToSolverFcn)  HYPRE_ILUSolve;
   hypre_SolverDestroy(base) = (HYPRE_PtrToDestroyFcn) HYPRE_ILUDestroy;

#if defined(HYPRE_USING_GPU)
   hypre_ParILUDataAperm(ilu_data)                        = NULL;
   hypre_ParILUDataMatBILUDevice(ilu_data)                = NULL;
   hypre_ParILUDataMatSILUDevice(ilu_data)                = NULL;
   hypre_ParILUDataMatEDevice(ilu_data)                   = NULL;
   hypre_ParILUDataMatFDevice(ilu_data)                   = NULL;
   hypre_ParILUDataR(ilu_data)                            = NULL;
   hypre_ParILUDataP(ilu_data)                            = NULL;
   hypre_ParILUDataFTempUpper(ilu_data)                   = NULL;
   hypre_ParILUDataUTempLower(ilu_data)                   = NULL;
   hypre_ParILUDataADiagDiag(ilu_data)                    = NULL;
   hypre_ParILUDataSDiagDiag(ilu_data)                    = NULL;
#endif

   /* general data */
   hypre_ParILUDataGlobalSolver(ilu_data)                 = 0;
   hypre_ParILUDataMatA(ilu_data)                         = NULL;
   hypre_ParILUDataMatL(ilu_data)                         = NULL;
   hypre_ParILUDataMatD(ilu_data)                         = NULL;
   hypre_ParILUDataMatU(ilu_data)                         = NULL;
   hypre_ParILUDataMatS(ilu_data)                         = NULL;
   hypre_ParILUDataSchurSolver(ilu_data)                  = NULL;
   hypre_ParILUDataSchurPrecond(ilu_data)                 = NULL;
   hypre_ParILUDataRhs(ilu_data)                          = NULL;
   hypre_ParILUDataX(ilu_data)                            = NULL;

   /* TODO (VPM): Transform this into a stack array */
   hypre_ParILUDataDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 3, HYPRE_MEMORY_HOST);
   hypre_ParILUDataDroptol(ilu_data)[0]                   = 1.0e-02; /* droptol for B */
   hypre_ParILUDataDroptol(ilu_data)[1]                   = 1.0e-02; /* droptol for E and F */
   hypre_ParILUDataDroptol(ilu_data)[2]                   = 1.0e-02; /* droptol for S */
   hypre_ParILUDataLfil(ilu_data)                         = 0;
   hypre_ParILUDataMaxRowNnz(ilu_data)                    = 1000;
   hypre_ParILUDataCFMarkerArray(ilu_data)                = NULL;
   hypre_ParILUDataPerm(ilu_data)                         = NULL;
   hypre_ParILUDataQPerm(ilu_data)                        = NULL;
   hypre_ParILUDataTolDDPQ(ilu_data)                      = 1.0e-01;
   hypre_ParILUDataF(ilu_data)                            = NULL;
   hypre_ParILUDataU(ilu_data)                            = NULL;
   hypre_ParILUDataFTemp(ilu_data)                        = NULL;
   hypre_ParILUDataUTemp(ilu_data)                        = NULL;
   hypre_ParILUDataXTemp(ilu_data)                        = NULL;
   hypre_ParILUDataYTemp(ilu_data)                        = NULL;
   hypre_ParILUDataZTemp(ilu_data)                        = NULL;
   hypre_ParILUDataUExt(ilu_data)                         = NULL;
   hypre_ParILUDataFExt(ilu_data)                         = NULL;
   hypre_ParILUDataResidual(ilu_data)                     = NULL;
   hypre_ParILUDataRelResNorms(ilu_data)                  = NULL;
   hypre_ParILUDataNumIterations(ilu_data)                = 0;
   hypre_ParILUDataMaxIter(ilu_data)                      = 20;
   hypre_ParILUDataTriSolve(ilu_data)                     = 1;
   hypre_ParILUDataLowerJacobiIters(ilu_data)             = 5;
   hypre_ParILUDataUpperJacobiIters(ilu_data)             = 5;
   hypre_ParILUDataTol(ilu_data)                          = 1.0e-7;
   hypre_ParILUDataLogging(ilu_data)                      = 0;
   hypre_ParILUDataPrintLevel(ilu_data)                   = 0;
   hypre_ParILUDataL1Norms(ilu_data)                      = NULL;
   hypre_ParILUDataOperatorComplexity(ilu_data)           = 0.;
   hypre_ParILUDataIluType(ilu_data)                      = 0;
   hypre_ParILUDataNLU(ilu_data)                          = 0;
   hypre_ParILUDataNI(ilu_data)                           = 0;
   hypre_ParILUDataUEnd(ilu_data)                         = NULL;

   /* Iterative setup variables */
   hypre_ParILUDataIterativeSetupType(ilu_data)           = 0;
   hypre_ParILUDataIterativeSetupOption(ilu_data)         = 0;
   hypre_ParILUDataIterativeSetupMaxIter(ilu_data)        = 100;
   hypre_ParILUDataIterativeSetupNumIter(ilu_data)        = 0;
   hypre_ParILUDataIterativeSetupTolerance(ilu_data)      = 1.e-6;
   hypre_ParILUDataIterativeSetupHistory(ilu_data)        = NULL;

   /* reordering_type default to use local RCM */
   hypre_ParILUDataReorderingType(ilu_data)               = 1;

   /* see hypre_ILUSetType for more default values */
   hypre_ParILUDataTestOption(ilu_data)                   = 0;

   /* -> General slots */
   hypre_ParILUDataSchurSolverLogging(ilu_data)           = 0;
   hypre_ParILUDataSchurSolverPrintLevel(ilu_data)        = 0;

   /* -> Schur-GMRES */
   hypre_ParILUDataSchurGMRESKDim(ilu_data)               = 5;
   hypre_ParILUDataSchurGMRESMaxIter(ilu_data)            = 5;
   hypre_ParILUDataSchurGMRESTol(ilu_data)                = 0.0;
   hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data)        = 0.0;
   hypre_ParILUDataSchurGMRESRelChange(ilu_data)          = 0;

   /* -> Schur precond data */
   hypre_ParILUDataSchurPrecondIluType(ilu_data)          = 0;
   hypre_ParILUDataSchurPrecondIluLfil(ilu_data)          = 0;
   hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data)     = 100;
   hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)       = NULL;
   hypre_ParILUDataSchurPrecondPrintLevel(ilu_data)       = 0;
   hypre_ParILUDataSchurPrecondMaxIter(ilu_data)          = 1;
   hypre_ParILUDataSchurPrecondTriSolve(ilu_data)         = 1;
   hypre_ParILUDataSchurPrecondLowerJacobiIters(ilu_data) = 5;
   hypre_ParILUDataSchurPrecondUpperJacobiIters(ilu_data) = 5;
   hypre_ParILUDataSchurPrecondTol(ilu_data)              = 0.0;

   /* -> Schur-NSH */
   hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data)         = 5;
   hypre_ParILUDataSchurNSHSolveTol(ilu_data)             = 0.0;
   hypre_ParILUDataSchurNSHDroptol(ilu_data)              = NULL;
   hypre_ParILUDataSchurNSHMaxNumIter(ilu_data)           = 2;
   hypre_ParILUDataSchurNSHMaxRowNnz(ilu_data)            = 1000;
   hypre_ParILUDataSchurNSHTol(ilu_data)                  = 1e-09;
   hypre_ParILUDataSchurMRMaxIter(ilu_data)               = 2;
   hypre_ParILUDataSchurMRColVersion(ilu_data)            = 0;
   hypre_ParILUDataSchurMRMaxRowNnz(ilu_data)             = 200;
   hypre_ParILUDataSchurMRTol(ilu_data)                   = 1e-09;

   return ilu_data;
}

/*--------------------------------------------------------------------------
 * hypre_ILUDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUDestroy( void *data )
{
   hypre_ParILUData      *ilu_data = (hypre_ParILUData*) data;
   HYPRE_MemoryLocation   memory_location;

   if (ilu_data)
   {
      /* Get memory location from L factor */
      if (hypre_ParILUDataMatL(ilu_data))
      {
         memory_location = hypre_ParCSRMatrixMemoryLocation(hypre_ParILUDataMatL(ilu_data));
      }
      else
      {
         /* Use default memory location */
         HYPRE_GetMemoryLocation(&memory_location);
      }

      /* GPU additional data */
#if defined(HYPRE_USING_GPU)
      hypre_ParCSRMatrixDestroy( hypre_ParILUDataAperm(ilu_data) );
      hypre_ParCSRMatrixDestroy( hypre_ParILUDataR(ilu_data) );
      hypre_ParCSRMatrixDestroy( hypre_ParILUDataP(ilu_data) );

      hypre_CSRMatrixDestroy( hypre_ParILUDataMatAILUDevice(ilu_data) );
      hypre_CSRMatrixDestroy( hypre_ParILUDataMatBILUDevice(ilu_data) );
      hypre_CSRMatrixDestroy( hypre_ParILUDataMatSILUDevice(ilu_data) );
      hypre_CSRMatrixDestroy( hypre_ParILUDataMatEDevice(ilu_data) );
      hypre_CSRMatrixDestroy( hypre_ParILUDataMatFDevice(ilu_data) );
      hypre_SeqVectorDestroy( hypre_ParILUDataFTempUpper(ilu_data) );
      hypre_SeqVectorDestroy( hypre_ParILUDataUTempLower(ilu_data) );
      hypre_SeqVectorDestroy( hypre_ParILUDataADiagDiag(ilu_data) );
      hypre_SeqVectorDestroy( hypre_ParILUDataSDiagDiag(ilu_data) );
#endif

      /* final residual vector */
      hypre_ParVectorDestroy( hypre_ParILUDataResidual(ilu_data) );
      hypre_TFree( hypre_ParILUDataRelResNorms(ilu_data), HYPRE_MEMORY_HOST );

      /* temp vectors for solve phase */
      hypre_ParVectorDestroy( hypre_ParILUDataUTemp(ilu_data) );
      hypre_ParVectorDestroy( hypre_ParILUDataFTemp(ilu_data) );
      hypre_ParVectorDestroy( hypre_ParILUDataXTemp(ilu_data) );
      hypre_ParVectorDestroy( hypre_ParILUDataYTemp(ilu_data) );
      hypre_ParVectorDestroy( hypre_ParILUDataZTemp(ilu_data) );
      hypre_ParVectorDestroy( hypre_ParILUDataRhs(ilu_data) );
      hypre_ParVectorDestroy( hypre_ParILUDataX(ilu_data) );
      hypre_TFree( hypre_ParILUDataUExt(ilu_data), HYPRE_MEMORY_HOST );
      hypre_TFree( hypre_ParILUDataFExt(ilu_data), HYPRE_MEMORY_HOST );

      /* l1_norms */
      hypre_TFree( hypre_ParILUDataL1Norms(ilu_data), HYPRE_MEMORY_HOST );

      /* u_end */
      hypre_TFree( hypre_ParILUDataUEnd(ilu_data), HYPRE_MEMORY_HOST );

      /* Factors */
      hypre_ParCSRMatrixDestroy( hypre_ParILUDataMatS(ilu_data) );
      hypre_ParCSRMatrixDestroy( hypre_ParILUDataMatL(ilu_data) );
      hypre_ParCSRMatrixDestroy( hypre_ParILUDataMatU(ilu_data) );
      hypre_ParCSRMatrixDestroy( hypre_ParILUDataMatLModified(ilu_data) );
      hypre_ParCSRMatrixDestroy( hypre_ParILUDataMatUModified(ilu_data) );
      hypre_TFree( hypre_ParILUDataMatD(ilu_data), memory_location );
      hypre_TFree( hypre_ParILUDataMatDModified(ilu_data), memory_location );

      if (hypre_ParILUDataSchurSolver(ilu_data))
      {
         switch (hypre_ParILUDataIluType(ilu_data))
         {
            case 10: case 11: case 40: case 41: case 50:
               /* GMRES for Schur */
               HYPRE_ParCSRGMRESDestroy(hypre_ParILUDataSchurSolver(ilu_data));
               break;

            case 20: case 21:
               /* NSH for Schur */
               hypre_NSHDestroy(hypre_ParILUDataSchurSolver(ilu_data));
               break;

            default:
               break;
         }
      }

      /* ILU as precond for Schur */
      if ( hypre_ParILUDataSchurPrecond(ilu_data)  &&
#if defined(HYPRE_USING_GPU)
           hypre_ParILUDataIluType(ilu_data) != 10 &&
           hypre_ParILUDataIluType(ilu_data) != 11 &&
#endif
           (hypre_ParILUDataIluType(ilu_data) == 10 ||
            hypre_ParILUDataIluType(ilu_data) == 11 ||
            hypre_ParILUDataIluType(ilu_data) == 40 ||
            hypre_ParILUDataIluType(ilu_data) == 41) )
      {
         HYPRE_ILUDestroy( hypre_ParILUDataSchurPrecond(ilu_data) );
      }

      /* CF marker array */
      hypre_TFree( hypre_ParILUDataCFMarkerArray(ilu_data), HYPRE_MEMORY_HOST );

      /* permutation array */
      hypre_TFree( hypre_ParILUDataPerm(ilu_data), memory_location );
      hypre_TFree( hypre_ParILUDataQPerm(ilu_data), memory_location );

      /* Iterative ILU data */
      hypre_TFree( hypre_ParILUDataIterativeSetupHistory(ilu_data), HYPRE_MEMORY_HOST );

      /* droptol array - TODO (VPM): remove this after changing to static array */
      hypre_TFree( hypre_ParILUDataDroptol(ilu_data), HYPRE_MEMORY_HOST );
      hypre_TFree( hypre_ParILUDataSchurPrecondIluDroptol(ilu_data), HYPRE_MEMORY_HOST );
      hypre_TFree( hypre_ParILUDataSchurNSHDroptol(ilu_data), HYPRE_MEMORY_HOST );
   }

   /* ILU data */
   hypre_TFree(ilu_data, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetLevelOfFill
 *
 * Set fill level for ILUK
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetLevelOfFill( void      *ilu_vdata,
                         HYPRE_Int  lfil )
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataLfil(ilu_data) = lfil;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetMaxNnzPerRow
 *
 * Set max non-zeros per row in factors for ILUT
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetMaxNnzPerRow( void      *ilu_vdata,
                          HYPRE_Int  nzmax )
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataMaxRowNnz(ilu_data) = nzmax;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetDropThreshold
 *
 * Set threshold for dropping in LU factors for ILUT
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetDropThreshold( void       *ilu_vdata,
                           HYPRE_Real  threshold )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   if (!(hypre_ParILUDataDroptol(ilu_data)))
   {
      hypre_ParILUDataDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 3, HYPRE_MEMORY_HOST);
   }
   hypre_ParILUDataDroptol(ilu_data)[0] = threshold;
   hypre_ParILUDataDroptol(ilu_data)[1] = threshold;
   hypre_ParILUDataDroptol(ilu_data)[2] = threshold;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetDropThresholdArray
 *
 * Set array of threshold for dropping in LU factors for ILUT
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetDropThresholdArray( void       *ilu_vdata,
                                HYPRE_Real *threshold )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   if (!(hypre_ParILUDataDroptol(ilu_data)))
   {
      hypre_ParILUDataDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 3, HYPRE_MEMORY_HOST);
   }

   hypre_ParILUDataDroptol(ilu_data)[0] = threshold[0];
   hypre_ParILUDataDroptol(ilu_data)[1] = threshold[1];
   hypre_ParILUDataDroptol(ilu_data)[2] = threshold[2];

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetType
 *
 * Set ILU factorization type
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetType( void      *ilu_vdata,
                  HYPRE_Int  ilu_type )
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;

   /* Destroy schur solver and/or preconditioner if already have one */
   if (hypre_ParILUDataSchurSolver(ilu_data))
   {
      switch (hypre_ParILUDataIluType(ilu_data))
      {
         case 10: case 11: case 40: case 41: case 50:
            //GMRES for Schur
            HYPRE_ParCSRGMRESDestroy(hypre_ParILUDataSchurSolver(ilu_data));
            break;

         case 20: case 21:
            //  NSH for Schur
            hypre_NSHDestroy(hypre_ParILUDataSchurSolver(ilu_data));
            break;

         default:
            break;
      }
      hypre_ParILUDataSchurSolver(ilu_data) = NULL;
   }

   /* ILU as precond for Schur */
   if ( hypre_ParILUDataSchurPrecond(ilu_data)    &&
#if defined(HYPRE_USING_GPU)
        (hypre_ParILUDataIluType(ilu_data) != 10  &&
         hypre_ParILUDataIluType(ilu_data) != 11) &&
#endif
        (hypre_ParILUDataIluType(ilu_data) == 10  ||
         hypre_ParILUDataIluType(ilu_data) == 11  ||
         hypre_ParILUDataIluType(ilu_data) == 40  ||
         hypre_ParILUDataIluType(ilu_data) == 41) )
   {
      HYPRE_ILUDestroy(hypre_ParILUDataSchurPrecond(ilu_data));
      hypre_ParILUDataSchurPrecond(ilu_data) = NULL;
   }

   hypre_ParILUDataIluType(ilu_data) = ilu_type;

   /* reset default value, not a large cost
    * assume we won't change back from
    */
   switch (ilu_type)
   {
      /* NSH type */
      case 20: case 21:
      {
         /* only set value when user has not assiged value before */
         if (!(hypre_ParILUDataSchurNSHDroptol(ilu_data)))
         {
            hypre_ParILUDataSchurNSHDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 2, HYPRE_MEMORY_HOST);
            hypre_ParILUDataSchurNSHDroptol(ilu_data)[0] = 1e-02;
            hypre_ParILUDataSchurNSHDroptol(ilu_data)[1] = 1e-02;
         }
         break;
      }

      case 10: case 11: case 40: case 41: case 50:
      {
         /* Set value of droptol for solving Schur system (if not set by user) */
         /* NOTE: This is currently not exposed to users */
         if (!(hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)))
         {
            hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 3, HYPRE_MEMORY_HOST);
            hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[0] = 1e-02;
            hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[1] = 1e-02;
            hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[2] = 1e-02;
         }
         break;
      }

      default:
         break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetMaxIter
 *
 * Set max number of iterations for ILU solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetMaxIter( void      *ilu_vdata,
                     HYPRE_Int  max_iter )
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataMaxIter(ilu_data) = max_iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetIterativeSetupType
 *
 * Set iterative ILU setup algorithm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetIterativeSetupType( void      *ilu_vdata,
                                HYPRE_Int  iter_setup_type)
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataIterativeSetupType(ilu_data) = iter_setup_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetIterativeSetupOption
 *
 * Set iterative ILU compute option
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetIterativeSetupOption( void      *ilu_vdata,
                                  HYPRE_Int  iter_setup_option)
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;

   /* Compute residuals when using the stopping criteria, if not chosen by the user */
   iter_setup_option |= ((iter_setup_option & 0x02) && !(iter_setup_option & 0x0C)) ? 0x08 : 0;

   /* Compute residuals when asking for conv. history, if not chosen by the user */
   iter_setup_option |= ((iter_setup_option & 0x10) && !(iter_setup_option & 0x08)) ? 0x08 : 0;

   /* Zero out first bit of option (turn off rocSPARSE logging) */
   iter_setup_option &= ~0x01;

   /* Set internal iter_setup_option */
   hypre_ParILUDataIterativeSetupOption(ilu_data) = iter_setup_option;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetIterativeSetupMaxIter
 *
 * Set maximum number of iterations for iterative ILU setup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetIterativeSetupMaxIter( void      *ilu_vdata,
                                   HYPRE_Int  iter_setup_max_iter)
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataIterativeSetupMaxIter(ilu_data) = iter_setup_max_iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetIterativeSetupTolerance
 *
 * Set dropping tolerance for iterative ILU setup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetIterativeSetupTolerance( void       *ilu_vdata,
                                     HYPRE_Real  iter_setup_tolerance)
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataIterativeSetupTolerance(ilu_data) = iter_setup_tolerance;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUGetIterativeSetupHistory
 *
 * Get array of corrections and/or residual norms computed during ILU's
 * iterative setup algorithm.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUGetIterativeSetupHistory( void           *ilu_vdata,
                                   HYPRE_Complex **iter_setup_history)
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;

   *iter_setup_history = hypre_ParILUDataIterativeSetupHistory(ilu_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetTriSolve
 *
 * Set ILU triangular solver type
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetTriSolve( void      *ilu_vdata,
                      HYPRE_Int  tri_solve )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataTriSolve(ilu_data) = tri_solve;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetLowerJacobiIters
 *
 * Set Lower Jacobi iterations for iterative triangular solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetLowerJacobiIters( void     *ilu_vdata,
                              HYPRE_Int lower_jacobi_iters )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataLowerJacobiIters(ilu_data) = lower_jacobi_iters;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetUpperJacobiIters
 *
 * Set Upper Jacobi iterations for iterative triangular solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetUpperJacobiIters( void      *ilu_vdata,
                              HYPRE_Int  upper_jacobi_iters )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataUpperJacobiIters(ilu_data) = upper_jacobi_iters;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetTol
 *
 * Set convergence tolerance for ILU solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetTol( void       *ilu_vdata,
                 HYPRE_Real  tol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataTol(ilu_data) = tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetPrintLevel
 *
 * Set print level for ILU solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetPrintLevel( void      *ilu_vdata,
                        HYPRE_Int  print_level )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataPrintLevel(ilu_data) = print_level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetLogging
 *
 * Set print level for ilu solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetLogging( void      *ilu_vdata,
                     HYPRE_Int  logging )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataLogging(ilu_data) = logging;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetLocalReordering
 *
 * Set type of reordering for local matrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetLocalReordering( void      *ilu_vdata,
                             HYPRE_Int  ordering_type )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataReorderingType(ilu_data) = ordering_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurSolverKDIM
 *
 * Set KDim (for GMRES) for Solver of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurSolverKDIM( void      *ilu_vdata,
                             HYPRE_Int  ss_kDim )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurGMRESKDim(ilu_data) = ss_kDim;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurSolverMaxIter
 *
 * Set max iteration for Solver of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurSolverMaxIter( void      *ilu_vdata,
                                HYPRE_Int  ss_max_iter )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   /* for the GMRES solve, the max iter is same as kdim by default */
   hypre_ParILUDataSchurGMRESKDim(ilu_data) = ss_max_iter;
   hypre_ParILUDataSchurGMRESMaxIter(ilu_data) = ss_max_iter;

   /* also set this value for NSH solve */
   hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data) = ss_max_iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurSolverTol
 *
 * Set convergence tolerance for Solver of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurSolverTol( void       *ilu_vdata,
                            HYPRE_Real  ss_tol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurGMRESTol(ilu_data) = ss_tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurSolverAbsoluteTol
 *
 * Set absolute tolerance for Solver of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurSolverAbsoluteTol( void       *ilu_vdata,
                                    HYPRE_Real  ss_absolute_tol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data) = ss_absolute_tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurSolverLogging
 *
 * Set logging for Solver of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurSolverLogging( void      *ilu_vdata,
                                HYPRE_Int  ss_logging )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurSolverLogging(ilu_data) = ss_logging;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurSolverPrintLevel
 *
 * Set print level for Solver of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurSolverPrintLevel( void      *ilu_vdata,
                                   HYPRE_Int  ss_print_level )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurSolverPrintLevel(ilu_data) = ss_print_level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurSolverRelChange
 *
 * Set rel change (for GMRES) for Solver of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurSolverRelChange( void *ilu_vdata, HYPRE_Int ss_rel_change )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurGMRESRelChange(ilu_data) = ss_rel_change;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurPrecondILUType
 *
 * Set ILU type for Precond of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurPrecondILUType( void *ilu_vdata, HYPRE_Int sp_ilu_type )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurPrecondIluType(ilu_data) = sp_ilu_type;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurPrecondILULevelOfFill
 *
 * Set ILU level of fill for Precond of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurPrecondILULevelOfFill( void *ilu_vdata, HYPRE_Int sp_ilu_lfil )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurPrecondIluLfil(ilu_data) = sp_ilu_lfil;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurPrecondILUMaxNnzPerRow
 *
 * Set ILU max nonzeros per row for Precond of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurPrecondILUMaxNnzPerRow( void      *ilu_vdata,
                                         HYPRE_Int  sp_ilu_max_row_nnz )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data) = sp_ilu_max_row_nnz;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurPrecondILUDropThreshold
 *
 * Set ILU drop threshold for ILUT for Precond of Schur System
 * We don't want to influence the original ILU, so create new array if
 * not own data
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurPrecondILUDropThreshold( void       *ilu_vdata,
                                          HYPRE_Real  sp_ilu_droptol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   if (!(hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)))
   {
      hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 3, HYPRE_MEMORY_HOST);
   }
   hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[0]   = sp_ilu_droptol;
   hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[1]   = sp_ilu_droptol;
   hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[2]   = sp_ilu_droptol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurPrecondILUDropThresholdArray
 *
 * Set array of ILU drop threshold for ILUT for Precond of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurPrecondILUDropThresholdArray( void       *ilu_vdata,
                                               HYPRE_Real *sp_ilu_droptol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   if (!(hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)))
   {
      hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 3, HYPRE_MEMORY_HOST);
   }

   hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[0] = sp_ilu_droptol[0];
   hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[1] = sp_ilu_droptol[1];
   hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[2] = sp_ilu_droptol[2];

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurPrecondPrintLevel
 *
 * Set print level for Precond of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurPrecondPrintLevel( void      *ilu_vdata,
                                    HYPRE_Int  sp_print_level )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurPrecondPrintLevel(ilu_data) = sp_print_level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurPrecondMaxIter
 *
 * Set max number of iterations for Precond of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurPrecondMaxIter( void      *ilu_vdata,
                                 HYPRE_Int  sp_max_iter )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurPrecondMaxIter(ilu_data) = sp_max_iter;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurPrecondTriSolve
 *
 * Set triangular solver type for Precond of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurPrecondTriSolve( void      *ilu_vdata,
                                  HYPRE_Int  sp_tri_solve )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurPrecondTriSolve(ilu_data) = sp_tri_solve;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurPrecondLowerJacobiIters
 *
 * Set Lower Jacobi iterations for Precond of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurPrecondLowerJacobiIters( void      *ilu_vdata,
                                          HYPRE_Int  sp_lower_jacobi_iters )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurPrecondLowerJacobiIters(ilu_data) = sp_lower_jacobi_iters;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurPrecondUpperJacobiIters
 *
 * Set Upper Jacobi iterations for Precond of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurPrecondUpperJacobiIters( void      *ilu_vdata,
                                          HYPRE_Int  sp_upper_jacobi_iters )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurPrecondUpperJacobiIters(ilu_data) = sp_upper_jacobi_iters;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurPrecondTol
 *
 * Set convergence tolerance for Precond of Schur System
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurPrecondTol( void      *ilu_vdata,
                             HYPRE_Int  sp_tol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_ParILUDataSchurPrecondTol(ilu_data) = sp_tol;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurNSHDropThreshold
 *
 * Set tolorance for dropping in NSH for Schur System
 * We don't want to influence the original ILU, so create new array if
 * not own data
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurNSHDropThreshold( void       *ilu_vdata,
                                   HYPRE_Real  threshold )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   if (!(hypre_ParILUDataSchurNSHDroptol(ilu_data)))
   {
      hypre_ParILUDataSchurNSHDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 2, HYPRE_MEMORY_HOST);
   }

   hypre_ParILUDataSchurNSHDroptol(ilu_data)[0] = threshold;
   hypre_ParILUDataSchurNSHDroptol(ilu_data)[1] = threshold;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetSchurNSHDropThresholdArray
 *
 * Set tolorance array for NSH for Schur System
 *    - threshold[0] : threshold for Minimal Residual iteration (initial guess for NSH).
 *    - threshold[1] : threshold for Newton-Schulz-Hotelling iteration.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetSchurNSHDropThresholdArray( void       *ilu_vdata,
                                        HYPRE_Real *threshold )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;

   if (!(hypre_ParILUDataSchurNSHDroptol(ilu_data)))
   {
      hypre_ParILUDataSchurNSHDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 2, HYPRE_MEMORY_HOST);
   }

   hypre_ParILUDataSchurNSHDroptol(ilu_data)[0] = threshold[0];
   hypre_ParILUDataSchurNSHDroptol(ilu_data)[1] = threshold[1];

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUGetNumIterations
 *
 * Get number of iterations for ILU solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUGetNumIterations( void      *ilu_vdata,
                           HYPRE_Int *num_iterations )
{
   hypre_ParILUData  *ilu_data = (hypre_ParILUData*) ilu_vdata;

   if (!ilu_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *num_iterations = hypre_ParILUDataNumIterations(ilu_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUGetFinalRelativeResidualNorm
 *
 * Get residual norms for ILU solver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUGetFinalRelativeResidualNorm( void       *ilu_vdata,
                                       HYPRE_Real *res_norm )
{
   hypre_ParILUData  *ilu_data = (hypre_ParILUData*) ilu_vdata;

   if (!ilu_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   *res_norm = hypre_ParILUDataFinalRelResidualNorm(ilu_data);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUWriteSolverParams
 *
 * Print solver params
 *
 * TODO (VPM): check runtime switch to decide whether running on host or device
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUWriteSolverParams(void *ilu_vdata)
{
   hypre_ParILUData  *ilu_data = (hypre_ParILUData*) ilu_vdata;

   hypre_printf("ILU Setup parameters: \n");
   hypre_printf("ILU factorization type: %d : ", hypre_ParILUDataIluType(ilu_data));
   switch (hypre_ParILUDataIluType(ilu_data))
   {
      case 0:
#if defined(HYPRE_USING_GPU)
         if ( hypre_ParILUDataLfil(ilu_data) == 0 )
         {
            hypre_printf("Block Jacobi with GPU-accelerated ILU0 \n");
            hypre_printf("Operator Complexity (Fill factor) = %f \n",
                         hypre_ParILUDataOperatorComplexity(ilu_data));
         }
         else
#endif
         {
            hypre_printf("Block Jacobi with ILU(%d) \n", hypre_ParILUDataLfil(ilu_data));
            hypre_printf("Operator Complexity (Fill factor) = %f \n",
                         hypre_ParILUDataOperatorComplexity(ilu_data));
         }
         break;

      case 1:
         hypre_printf("Block Jacobi with ILUT \n");
         hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n",
                      hypre_ParILUDataDroptol(ilu_data)[0],
                      hypre_ParILUDataDroptol(ilu_data)[1],
                      hypre_ParILUDataDroptol(ilu_data)[2]);
         hypre_printf("Max nnz per row = %d \n", hypre_ParILUDataMaxRowNnz(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      hypre_ParILUDataOperatorComplexity(ilu_data));
         break;

      case 10:
#if defined(HYPRE_USING_GPU)
         if ( hypre_ParILUDataLfil(ilu_data) == 0 )
         {
            hypre_printf("ILU-GMRES with GPU-accelerated ILU0 \n");
            hypre_printf("Operator Complexity (Fill factor) = %f \n",
                         hypre_ParILUDataOperatorComplexity(ilu_data));
         }
         else
#endif
         {
            hypre_printf("ILU-GMRES with ILU(%d) \n", hypre_ParILUDataLfil(ilu_data));
            hypre_printf("Operator Complexity (Fill factor) = %f \n",
                         hypre_ParILUDataOperatorComplexity(ilu_data));
         }
         break;

      case 11:
         hypre_printf("ILU-GMRES with ILUT \n");
         hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n",
                      hypre_ParILUDataDroptol(ilu_data)[0],
                      hypre_ParILUDataDroptol(ilu_data)[1],
                      hypre_ParILUDataDroptol(ilu_data)[2]);
         hypre_printf("Max nnz per row = %d \n", hypre_ParILUDataMaxRowNnz(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      hypre_ParILUDataOperatorComplexity(ilu_data));
         break;

      case 20:
         hypre_printf("Newton-Schulz-Hotelling with ILU(%d) \n", hypre_ParILUDataLfil(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      hypre_ParILUDataOperatorComplexity(ilu_data));
         break;

      case 21:
         hypre_printf("Newton-Schulz-Hotelling with ILUT \n");
         hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n",
                      hypre_ParILUDataDroptol(ilu_data)[0],
                      hypre_ParILUDataDroptol(ilu_data)[1],
                      hypre_ParILUDataDroptol(ilu_data)[2]);
         hypre_printf("Max nnz per row = %d \n", hypre_ParILUDataMaxRowNnz(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      hypre_ParILUDataOperatorComplexity(ilu_data));
         break;

      case 30:
         hypre_printf("RAS with ILU(%d) \n", hypre_ParILUDataLfil(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      hypre_ParILUDataOperatorComplexity(ilu_data));
         break;

      case 31:
         hypre_printf("RAS with ILUT \n");
         hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n",
                      hypre_ParILUDataDroptol(ilu_data)[0],
                      hypre_ParILUDataDroptol(ilu_data)[1],
                      hypre_ParILUDataDroptol(ilu_data)[2]);
         hypre_printf("Max nnz per row = %d \n", hypre_ParILUDataMaxRowNnz(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      hypre_ParILUDataOperatorComplexity(ilu_data));
         break;

      case 40:
         hypre_printf("ddPQ-ILU-GMRES with ILU(%d) \n", hypre_ParILUDataLfil(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      hypre_ParILUDataOperatorComplexity(ilu_data));
         break;

      case 41:
         hypre_printf("ddPQ-ILU-GMRES with ILUT \n");
         hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n",
                      hypre_ParILUDataDroptol(ilu_data)[0],
                      hypre_ParILUDataDroptol(ilu_data)[1],
                      hypre_ParILUDataDroptol(ilu_data)[2]);
         hypre_printf("Max nnz per row = %d \n", hypre_ParILUDataMaxRowNnz(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      hypre_ParILUDataOperatorComplexity(ilu_data));
         break;

      case 50:
         hypre_printf("RAP-Modified-ILU with ILU(%d) \n", hypre_ParILUDataLfil(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n",
                      hypre_ParILUDataOperatorComplexity(ilu_data));
         break;

      default:
         hypre_printf("Unknown type \n");
         break;
   }

   hypre_printf("\n ILU Solver Parameters: \n");
   hypre_printf("Max number of iterations: %d\n", hypre_ParILUDataMaxIter(ilu_data));
   if (hypre_ParILUDataTriSolve(ilu_data))
   {
      hypre_printf("  Triangular solver type: exact (1)\n");
   }
   else
   {
      hypre_printf("  Triangular solver type: iterative (0)\n");
      hypre_printf(" Lower Jacobi Iterations: %d\n", hypre_ParILUDataLowerJacobiIters(ilu_data));
      hypre_printf(" Upper Jacobi Iterations: %d\n", hypre_ParILUDataUpperJacobiIters(ilu_data));
   }
   hypre_printf("      Stopping tolerance: %e\n", hypre_ParILUDataTol(ilu_data));

   return hypre_error_flag;
}

/******************************************************************************
 *
 * ILU helper functions
 *
 * TODO (VPM): move these to a new "par_ilu_utils.c" file
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_ILUMinHeapAddI
 *
 * Add an element to the heap
 * I means HYPRE_Int
 * R means HYPRE_Real
 * max/min heap
 * r means heap goes from 0 to -1, -2 instead of 0 1 2
 * Ii and Ri means orderd by value of heap, like iw for ILU
 * heap: array of that heap
 * len: the current length of the heap
 * WARNING: You should first put that element to the end of the heap
 *    and add the length of heap by one before call this function.
 * the reason is that we don't want to change something outside the
 *    heap, so left it to the user
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUMinHeapAddI(HYPRE_Int *heap, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p;

   len--; /* now len is the current index */
   while (len > 0)
   {
      /* get the parent index */
      p = (len - 1) / 2;
      if (heap[p] > heap[len])
      {
         /* this is smaller */
         hypre_swap(heap, p, len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUMinHeapAddIIIi
 *
 * See hypre_ILUMinHeapAddI for detail instructions
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUMinHeapAddIIIi(HYPRE_Int *heap, HYPRE_Int *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p;

   len--; /* now len is the current index */
   while (len > 0)
   {
      /* get the parent index */
      p = (len - 1) / 2;
      if (heap[p] > heap[len])
      {
         /* this is smaller */
         hypre_swap(Ii1, heap[p], heap[len]);
         hypre_swap2i(heap, I1, p, len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUMinHeapAddIRIi
 *
 * see hypre_ILUMinHeapAddI for detail instructions
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUMinHeapAddIRIi(HYPRE_Int *heap, HYPRE_Real *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p;

   len--; /* now len is the current index */
   while (len > 0)
   {
      /* get the parent index */
      p = (len - 1) / 2;
      if (heap[p] > heap[len])
      {
         /* this is smaller */
         hypre_swap(Ii1, heap[p], heap[len]);
         hypre_swap2(heap, I1, p, len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUMaxrHeapAddRabsI
 *
 * See hypre_ILUMinHeapAddI for detail instructions
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUMaxrHeapAddRabsI(HYPRE_Real *heap, HYPRE_Int *I1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p;
   len--;/* now len is the current index */
   while (len > 0)
   {
      /* get the parent index */
      p = (len - 1) / 2;
      if (hypre_abs(heap[-p]) < hypre_abs(heap[-len]))
      {
         /* this is smaller */
         hypre_swap2(I1, heap, -p, -len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUMinHeapRemoveI
 *
 * Swap the first element with the last element of the heap,
 *    reduce size by one, and maintain the heap structure
 * I means HYPRE_Int
 * R means HYPRE_Real
 * max/min heap
 * r means heap goes from 0 to -1, -2 instead of 0 1 2
 * Ii and Ri means orderd by value of heap, like iw for ILU
 * heap: aray of that heap
 * len: current length of the heap
 * WARNING: Remember to change the len yourself
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUMinHeapRemoveI(HYPRE_Int *heap, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p, l, r;

   len--; /* now len is the max index */

   /* swap the first element to last */
   hypre_swap(heap, 0, len);
   p = 0;
   l = 1;

   /* while I'm still in the heap */
   while (l < len)
   {
      r = 2 * p + 2;

      /* two childs, pick the smaller one */
      l = r >= len || heap[l] < heap[r] ? l : r;
      if (heap[l] < heap[p])
      {
         hypre_swap(heap, l, p);
         p = l;
         l = 2 * p + 1;
      }
      else
      {
         break;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUMinHeapRemoveIIIi
 *
 * See hypre_ILUMinHeapRemoveI for detail instructions
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUMinHeapRemoveIIIi(HYPRE_Int *heap, HYPRE_Int *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p, l, r;

   len--;/* now len is the max index */

   /* swap the first element to last */
   hypre_swap(Ii1, heap[0], heap[len]);
   hypre_swap2i(heap, I1, 0, len);
   p = 0;
   l = 1;

   /* while I'm still in the heap */
   while (l < len)
   {
      r = 2 * p + 2;

      /* two childs, pick the smaller one */
      l = r >= len || heap[l] < heap[r] ? l : r;
      if (heap[l] < heap[p])
      {
         hypre_swap(Ii1, heap[p], heap[l]);
         hypre_swap2i(heap, I1, l, p);
         p = l;
         l = 2 * p + 1;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUMinHeapRemoveIRIi
 *
 * See hypre_ILUMinHeapRemoveI for detail instructions
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUMinHeapRemoveIRIi(HYPRE_Int *heap, HYPRE_Real *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p, l, r;

   len--;/* now len is the max index */

   /* swap the first element to last */
   hypre_swap(Ii1, heap[0], heap[len]);
   hypre_swap2(heap, I1, 0, len);
   p = 0;
   l = 1;

   /* while I'm still in the heap */
   while (l < len)
   {
      r = 2 * p + 2;

      /* two childs, pick the smaller one */
      l = r >= len || heap[l] < heap[r] ? l : r;
      if (heap[l] < heap[p])
      {
         hypre_swap(Ii1, heap[p], heap[l]);
         hypre_swap2(heap, I1, l, p);
         p = l;
         l = 2 * p + 1;
      }
      else
      {
         break;
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUMaxrHeapRemoveRabsI
 *
 * See hypre_ILUMinHeapRemoveI for detail instructions
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUMaxrHeapRemoveRabsI(HYPRE_Real *heap, HYPRE_Int *I1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p, l, r;

   len--;/* now len is the max index */

   /* swap the first element to last */
   hypre_swap2(I1, heap, 0, -len);
   p = 0;
   l = 1;

   /* while I'm still in the heap */
   while (l < len)
   {
      r = 2 * p + 2;

      /* two childs, pick the smaller one */
      l = r >= len || hypre_abs(heap[-l]) > hypre_abs(heap[-r]) ? l : r;
      if (hypre_abs(heap[-l]) > hypre_abs(heap[-p]))
      {
         hypre_swap2(I1, heap, -l, -p);
         p = l;
         l = 2 * p + 1;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUMaxQSplitRabsI
 *
 * Split based on quick sort algorithm (avoid sorting the entire array)
 * find the largest k elements out of original array
 *
 * arrayR: input array for compare
 * arrayI: integer array bind with array
 * k: largest k elements
 * len: length of the array
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUMaxQSplitRabsI(HYPRE_Real *arrayR,
                        HYPRE_Int  *arrayI,
                        HYPRE_Int   left,
                        HYPRE_Int   bound,
                        HYPRE_Int   right)
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return hypre_error_flag;
   }

   hypre_swap2(arrayI, arrayR, left, (left + right) / 2);
   last = left;
   for (i = left + 1 ; i <= right ; i ++)
   {
      if (hypre_abs(arrayR[i]) > hypre_abs(arrayR[left]))
      {
         hypre_swap2(arrayI, arrayR, ++last, i);
      }
   }

   hypre_swap2(arrayI, arrayR, left, last);
   hypre_ILUMaxQSplitRabsI(arrayR, arrayI, left, bound, last - 1);
   if (bound > last)
   {
      hypre_ILUMaxQSplitRabsI(arrayR, arrayI, last + 1, bound, right);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUMaxRabs
 *
 * Helper function to search max value from a row
 * array: the array we work on
 * start: the start of the search range
 * end: the end of the search range
 * nLU: ignore rows (new row index) after nLU
 * rperm: reverse permutation array rperm[old] = new.
 *        if rperm set to NULL, ingore nLU and rperm
 * value: return the value ge get (absolute value)
 * index: return the index of that value, could be NULL which means not return
 * l1_norm: return the l1_norm of the array, could be NULL which means no return
 * nnz: return the number of nonzeros inside this array, could be NULL which means no return
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUMaxRabs(HYPRE_Real  *array_data,
                 HYPRE_Int   *array_j,
                 HYPRE_Int    start,
                 HYPRE_Int    end,
                 HYPRE_Int    nLU,
                 HYPRE_Int   *rperm,
                 HYPRE_Real  *value,
                 HYPRE_Int   *index,
                 HYPRE_Real  *l1_norm,
                 HYPRE_Int   *nnz)
{
   HYPRE_Int i, idx, col, nz;
   HYPRE_Real val, max_value, norm;

   nz = 0;
   norm = 0.0;
   max_value = -1.0;
   idx = -1;
   if (rperm)
   {
      /* apply rperm and nLU */
      for (i = start ; i < end ; i ++)
      {
         col = rperm[array_j[i]];
         if (col > nLU)
         {
            /* this old column is in new external part */
            continue;
         }
         nz ++;
         val = hypre_abs(array_data[i]);
         norm += val;
         if (max_value < val)
         {
            max_value = val;
            idx = i;
         }
      }
   }
   else
   {
      /* basic search */
      for (i = start ; i < end ; i ++)
      {
         val = hypre_abs(array_data[i]);
         norm += val;
         if (max_value < val)
         {
            max_value = val;
            idx = i;
         }
      }
      nz = end - start;
   }

   *value = max_value;
   if (index)
   {
      *index = idx;
   }
   if (l1_norm)
   {
      *l1_norm = norm;
   }
   if (nnz)
   {
      *nnz = nz;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUGetPermddPQPre
 *
 * Pre selection for ddPQ, this is the basic version considering row sparsity
 * n: size of matrix
 * nLU: size we consider ddPQ reorder, only first nLU*nLU block is considered
 * A_diag_i/j/data: information of A
 * tol: tol for ddPQ, normally between 0.1-0.3
 * *perm: current row order
 * *rperm: current column order
 * *pperm_pre: output ddPQ pre row roder
 * *qperm_pre: output ddPQ pre column order
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUGetPermddPQPre(HYPRE_Int   n,
                        HYPRE_Int   nLU,
                        HYPRE_Int  *A_diag_i,
                        HYPRE_Int  *A_diag_j,
                        HYPRE_Real *A_diag_data,
                        HYPRE_Real  tol,
                        HYPRE_Int  *perm,
                        HYPRE_Int  *rperm,
                        HYPRE_Int  *pperm_pre,
                        HYPRE_Int  *qperm_pre,
                        HYPRE_Int  *nB)
{
   HYPRE_UNUSED_VAR(n);

   HYPRE_Int   i, ii, nB_pre, k1, k2;
   HYPRE_Real  gtol, max_value, norm;

   HYPRE_Int   *jcol, *jnnz;
   HYPRE_Real  *weight;

   weight = hypre_TAlloc(HYPRE_Real, nLU + 1, HYPRE_MEMORY_HOST);
   jcol   = hypre_TAlloc(HYPRE_Int, nLU + 1, HYPRE_MEMORY_HOST);
   jnnz   = hypre_TAlloc(HYPRE_Int, nLU + 1, HYPRE_MEMORY_HOST);

   max_value = -1.0;

   /* first need to build gtol */
   for (ii = 0; ii < nLU; ii++)
   {
      /* find real row */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];

      /* find max|a| of that row and its index */
      hypre_ILUMaxRabs(A_diag_data, A_diag_j, k1, k2, nLU, rperm,
                       weight + ii, jcol + ii, &norm, jnnz + ii);
      weight[ii] /= norm;
      if (weight[ii] > max_value)
      {
         max_value = weight[ii];
      }
   }

   gtol = tol * max_value;

   /* second loop to pre select B */
   nB_pre = 0;
   for ( ii = 0 ; ii < nLU ; ii ++)
   {
      /* keep this row */
      if (weight[ii] > gtol)
      {
         weight[nB_pre] /= (HYPRE_Real)(jnnz[ii]);
         pperm_pre[nB_pre] = perm[ii];
         qperm_pre[nB_pre++] = A_diag_j[jcol[ii]];
      }
   }

   *nB = nB_pre;

   /* sort from small to large */
   hypre_qsort3(weight, pperm_pre, qperm_pre, 0, nB_pre - 1);

   hypre_TFree(weight, HYPRE_MEMORY_HOST);
   hypre_TFree(jcol, HYPRE_MEMORY_HOST);
   hypre_TFree(jnnz, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUGetPermddPQ
 *
 * Get ddPQ version perm array for ParCSR matrices. ddPQ is a two-side
 * permutation for diagonal dominance. Greedy matching selection
 *
 * Parameters:
 *   A: the input matrix
 *   pperm: row permutation (lives at memory_location_A)
 *   qperm: col permutation (lives at memory_location_A)
 *   nB: the size of B block
 *   nI: number of interial nodes
 *   tol: the dropping tolorance for ddPQ
 *   reordering_type: Type of reordering for the interior nodes.
 *
 * Currently only supports RCM reordering. Set to 0 for no reordering.
 *
 * TODO (VPM): Change permutation arrays types to hypre_IntArray
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUGetPermddPQ(hypre_ParCSRMatrix   *A,
                     HYPRE_Int           **io_pperm,
                     HYPRE_Int           **io_qperm,
                     HYPRE_Real            tol,
                     HYPRE_Int            *nB,
                     HYPRE_Int            *nI,
                     HYPRE_Int             reordering_type)
{
   /* data objects for A */
   hypre_CSRMatrix       *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int              n               = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_MemoryLocation   memory_location = hypre_CSRMatrixMemoryLocation(A_diag);

   hypre_CSRMatrix       *h_A_diag;
   HYPRE_Int             *A_diag_i;
   HYPRE_Int             *A_diag_j;
   HYPRE_Complex         *A_diag_data;

   /* Local variables */
   HYPRE_Int              i, nB_pre, irow, jcol, nLU;
   HYPRE_Int             *pperm, *qperm;
   HYPRE_Int             *new_pperm, *new_qperm;
   HYPRE_Int             *rpperm, *rqperm, *pperm_pre, *qperm_pre;
   HYPRE_MemoryLocation   memory_location_perm;

   /* 1: Setup and create memory */
   pperm  = NULL;
   qperm  = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   rpperm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   rqperm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);

   /* 2: Find interior nodes first */
   hypre_ILUGetInteriorExteriorPerm(A, HYPRE_MEMORY_HOST, &pperm, &nLU, 0);

   /* 3: Pre selection on interial nodes
    * this pre selection puts external nodes to the last
    * also provide candidate rows for B block
    */

   /* build reverse permutation array
    * rperm[old] = new
    */
   for (i = 0 ; i < n ; i ++)
   {
      rpperm[pperm[i]] = i;
   }

   /* build place holder for pre selection pairs */
   pperm_pre = hypre_TAlloc(HYPRE_Int, nLU, HYPRE_MEMORY_HOST);
   qperm_pre = hypre_TAlloc(HYPRE_Int, nLU, HYPRE_MEMORY_HOST);

   /* Set/Move A_diag to host memory */
   h_A_diag = (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE) ?
              hypre_CSRMatrixClone_v2(A_diag, 1, HYPRE_MEMORY_HOST) : A_diag;
   A_diag_i = hypre_CSRMatrixI(h_A_diag);
   A_diag_j = hypre_CSRMatrixJ(h_A_diag);
   A_diag_data = hypre_CSRMatrixData(h_A_diag);

   /* pre selection */
   hypre_ILUGetPermddPQPre(n, nLU, A_diag_i, A_diag_j, A_diag_data, tol,
                           pperm, rpperm, pperm_pre, qperm_pre, &nB_pre);

   /* 4: Build B block
    * Greedy selection
    */

   /* rperm[old] = new */
   for (i = 0 ; i < nLU ; i ++)
   {
      rpperm[pperm[i]] = -1;
   }

   hypre_TMemcpy(rqperm, rpperm, HYPRE_Int, n, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   hypre_TMemcpy(qperm, pperm, HYPRE_Int, n, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);

   /* we sort from small to large, so we need to go from back to start
    * we only need nB_pre to start the loop, after that we could use it for size of B
    */
   for (i = nB_pre - 1, nB_pre = 0 ; i >= 0 ; i --)
   {
      irow = pperm_pre[i];
      jcol = qperm_pre[i];

      /* this col is not yet taken */
      if (rqperm[jcol] < 0)
      {
         rpperm[irow] = nB_pre;
         rqperm[jcol] = nB_pre;
         pperm[nB_pre] = irow;
         qperm[nB_pre++] = jcol;
      }
   }

   /* 5: Complete the permutation
    * rperm[old] = new
    * those still mapped to a new index means not yet covered
    */
   nLU = nB_pre;
   for (i = 0 ; i < n ; i ++)
   {
      if (rpperm[i] < 0)
      {
         pperm[nB_pre++] = i;
      }
   }
   nB_pre = nLU;
   for (i = 0 ; i < n ; i ++)
   {
      if (rqperm[i] < 0)
      {
         qperm[nB_pre++] = i;
      }
   }

   /* Apply RCM reordering */
   if (reordering_type != 0)
   {
      hypre_ILULocalRCM(h_A_diag, 0, nLU, &pperm, &qperm, 0);
      memory_location_perm = memory_location;
   }
   else
   {
      memory_location_perm = HYPRE_MEMORY_HOST;
   }

   /* Move to device memory if needed */
   if (memory_location_perm != memory_location)
   {
      new_pperm = hypre_TAlloc(HYPRE_Int, n, memory_location);
      new_qperm = hypre_TAlloc(HYPRE_Int, n, memory_location);

      hypre_TMemcpy(new_pperm, pperm, HYPRE_Int, n,
                    memory_location, memory_location_perm);
      hypre_TMemcpy(new_qperm, qperm, HYPRE_Int, n,
                    memory_location, memory_location_perm);

      hypre_TFree(pperm, memory_location_perm);
      hypre_TFree(qperm, memory_location_perm);

      pperm = new_pperm;
      qperm = new_qperm;
   }

   /* Output pointers */
   *nI = nLU;
   *nB = nLU;
   *io_pperm = pperm;
   *io_qperm = qperm;

   /* Free memory */
   if (h_A_diag != A_diag)
   {
      hypre_CSRMatrixDestroy(h_A_diag);
   }
   hypre_TFree(rpperm, HYPRE_MEMORY_HOST);
   hypre_TFree(rqperm, HYPRE_MEMORY_HOST);
   hypre_TFree(pperm_pre, HYPRE_MEMORY_HOST);
   hypre_TFree(qperm_pre, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUGetInteriorExteriorPerm
 *
 * Get perm array from parcsr matrix based on diag and offdiag matrix
 * Just simply loop through the rows of offd of A, check for nonzero rows
 * Put interior nodes at the beginning
 *
 * Parameters:
 *   A: parcsr matrix
 *   perm: permutation array
 *   nLU: number of interial nodes
 *   reordering_type: Type of (additional) reordering for the interior nodes.
 *
 * Currently only supports RCM reordering. Set to 0 for no reordering.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUGetInteriorExteriorPerm(hypre_ParCSRMatrix   *A,
                                 HYPRE_MemoryLocation  memory_location,
                                 HYPRE_Int           **perm,
                                 HYPRE_Int            *nLU,
                                 HYPRE_Int             reordering_type)
{
   /* get basic information of A */
   HYPRE_Int              n        = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix       *A_diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix       *A_offd   = hypre_ParCSRMatrixOffd(A);
   hypre_ParCSRCommPkg   *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   HYPRE_MemoryLocation   A_memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   HYPRE_Int             *A_offd_i;
   HYPRE_Int              i, j, first, last, start, end;
   HYPRE_Int              num_sends, send_map_start, send_map_end, col;

   /* Local arrays */
   HYPRE_Int             *tperm   = hypre_TAlloc(HYPRE_Int, n, memory_location);
   HYPRE_Int             *h_tperm = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   HYPRE_Int             *marker  = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);

   /* Get comm_pkg, create one if not present */
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* Set A_offd_i on the host */
   if (hypre_GetActualMemLocation(A_memory_location) == hypre_MEMORY_DEVICE)
   {
      /* Move A_offd_i to host */
      A_offd_i = hypre_CTAlloc(HYPRE_Int, n + 1, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(A_offd_i,  hypre_CSRMatrixI(A_offd), HYPRE_Int, n + 1,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      A_offd_i = hypre_CSRMatrixI(A_offd);
   }

   /* Set initial interior/exterior pointers */
   first = 0;
   last  = n - 1;

   /* now directly take advantage of comm_pkg */
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   for (i = 0; i < num_sends; i++)
   {
      send_map_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      send_map_end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1);
      for (j = send_map_start; j < send_map_end; j++)
      {
         col = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         if (marker[col] == 0)
         {
            h_tperm[last--] = col;
            marker[col] = -1;
         }
      }
   }

   /* now deal with the row */
   for (i = 0; i < n; i++)
   {
      if (marker[i] == 0)
      {
         start = A_offd_i[i];
         end = A_offd_i[i + 1];
         if (start == end)
         {
            h_tperm[first++] = i;
         }
         else
         {
            h_tperm[last--] = i;
         }
      }
   }

   if (reordering_type != 0)
   {
      /* Apply RCM. Note: h_tperm lives at A_memory_location at output */
      hypre_ILULocalRCM(A_diag, 0, first, &h_tperm, &h_tperm, 1);

      /* Move permutation vector to final memory location */
      hypre_TMemcpy(tperm, h_tperm, HYPRE_Int, n, memory_location, A_memory_location);

      /* Free memory */
      hypre_TFree(h_tperm, A_memory_location);
   }
   else
   {
      /* Move permutation vector to final memory location */
      hypre_TMemcpy(tperm, h_tperm, HYPRE_Int, n, memory_location, HYPRE_MEMORY_HOST);

      /* Free memory */
      hypre_TFree(h_tperm, HYPRE_MEMORY_HOST);
   }

   /* Free memory */
   hypre_TFree(marker, HYPRE_MEMORY_HOST);
   if (A_offd_i != hypre_CSRMatrixI(A_offd))
   {
      hypre_TFree(A_offd_i, HYPRE_MEMORY_HOST);
   }

   /* Set output values */
   if ((*perm) != NULL)
   {
      hypre_TFree(*perm, memory_location);
   }
   *perm = tperm;
   *nLU = first;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUGetLocalPerm
 *
 * Get the (local) ordering of the diag (local) matrix (no permutation).
 * This is the permutation used for the block-jacobi case.
 *
 * Parameters:
 *   A: parcsr matrix
 *   perm: permutation array
 *   nLU: number of interior nodes
 *   reordering_type: Type of (additional) reordering for the nodes.
 *
 * Currently only supports RCM reordering. Set to 0 for no reordering.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUGetLocalPerm(hypre_ParCSRMatrix  *A,
                      HYPRE_Int          **perm_ptr,
                      HYPRE_Int           *nLU,
                      HYPRE_Int            reordering_type)
{
   /* get basic information of A */
   HYPRE_Int             num_rows = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix      *A_diag = hypre_ParCSRMatrixDiag(A);

   /* Local variables */
   HYPRE_Int            *perm = NULL;

   /* Compute local RCM ordering on the host */
   if (reordering_type != 0)
   {
      hypre_ILULocalRCM(A_diag, 0, num_rows, &perm, &perm, 1);
   }

   /* Set output pointers */
   *nLU = num_rows;
   *perm_ptr = perm;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUBuildRASExternalMatrix
 *
 * Build the expanded matrix for RAS-1
 * A: input ParCSR matrix
 * E_i, E_j, E_data: information for external matrix
 * rperm: reverse permutation to build real index, rperm[old] = new
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUBuildRASExternalMatrix(hypre_ParCSRMatrix  *A,
                                HYPRE_Int           *rperm,
                                HYPRE_Int          **E_i,
                                HYPRE_Int          **E_j,
                                HYPRE_Real         **E_data)
{
   /* data objects for communication */
   MPI_Comm                 comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int                my_id;

   /* data objects for A */
   hypre_CSRMatrix          *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix          *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_BigInt             *A_col_starts = hypre_ParCSRMatrixColStarts(A);
   HYPRE_BigInt             *A_offd_colmap = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int                *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int                *A_offd_i = hypre_CSRMatrixI(A_offd);

   /* data objects for external A matrix */
   // Need to check the new version of hypre_ParcsrGetExternalRows
   hypre_CSRMatrix          *A_ext = NULL;
   // # up to local offd cols, no need to be HYPRE_BigInt
   HYPRE_Int                *A_ext_i = NULL;
   // Return global index, HYPRE_BigInt required
   HYPRE_BigInt             *A_ext_j = NULL;
   HYPRE_Real               *A_ext_data = NULL;

   /* data objects for output */
   HYPRE_Int                 E_nnz;
   HYPRE_Int                *E_ext_i = NULL;
   // Local index, no need to use HYPRE_BigInt
   HYPRE_Int                *E_ext_j = NULL;
   HYPRE_Real               *E_ext_data = NULL;

   //guess non-zeros for E before start
   HYPRE_Int                 E_init_alloc;

   /* size */
   HYPRE_Int                 n = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int                 m = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int                 A_diag_nnz = A_diag_i[n];
   HYPRE_Int                 A_offd_nnz = A_offd_i[n];

   HYPRE_Int                 i, j, idx;
   HYPRE_BigInt              big_col;

   /* 1: Set up phase and get external rows
    * Use the HYPRE build-in function
    */

   /* MPI stuff */
   //hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* Param of hypre_ParcsrGetExternalRows:
    * hypre_ParCSRMatrix   *A          [in]  -> Input parcsr matrix.
    * HYPRE_Int            indies_len  [in]  -> Input length of indices_len array
    * HYPRE_Int            *indices    [in]  -> Input global indices of rows we want to get
    * hypre_CSRMatrix      **A_ext     [out] -> Return the external CSR matrix.
    * hypre_ParCSRCommPkg  commpkg_out [out] -> Return commpkg if set to a point. Use NULL here since we don't want it.
    */
   //   hypre_ParcsrGetExternalRows( A, m, A_offd_colmap, &A_ext, NULL );
   A_ext = hypre_ParCSRMatrixExtractBExt(A, A, 1);

   A_ext_i              = hypre_CSRMatrixI(A_ext);
   //This should be HYPRE_BigInt since this is global index, use big_j in csr */
   A_ext_j = hypre_CSRMatrixBigJ(A_ext);
   A_ext_data           = hypre_CSRMatrixData(A_ext);

   /* guess memory we need to allocate to E_j */
   E_init_alloc =  hypre_max( (HYPRE_Int) ( A_diag_nnz / (HYPRE_Real) n / (HYPRE_Real) n *
                                            (HYPRE_Real) m * (HYPRE_Real) m + A_offd_nnz), 1);

   /* Initial guess */
   E_ext_i     = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_HOST);
   E_ext_j     = hypre_TAlloc(HYPRE_Int, E_init_alloc, HYPRE_MEMORY_HOST);
   E_ext_data  = hypre_TAlloc(HYPRE_Real, E_init_alloc, HYPRE_MEMORY_HOST);

   /* 2: Discard unecessary cols
    * Search A_ext_j, discard those cols not belong to current proc
    * First check diag, and search in offd_col_map
    */

   E_nnz       = 0;
   E_ext_i[0]  = 0;

   for ( i = 0 ;  i < m ; i ++)
   {
      E_ext_i[i] = E_nnz;
      for ( j = A_ext_i[i] ; j < A_ext_i[i + 1] ; j ++)
      {
         big_col = A_ext_j[j];
         /* First check if that belongs to the diagonal part */
         if ( big_col >= A_col_starts[0] && big_col < A_col_starts[1] )
         {
            /* this is a diagonal entry, rperm (map old to new) and shift it */

            /* Note here, the result of big_col - A_col_starts[0] in no longer a HYPRE_BigInt */
            idx = (HYPRE_Int)(big_col - A_col_starts[0]);
            E_ext_j[E_nnz]       = rperm[idx];
            E_ext_data[E_nnz++]  = A_ext_data[j];
         }

         /* If not, apply binary search to check if is offdiagonal */
         else
         {
            /* Search, result is not HYPRE_BigInt */
            E_ext_j[E_nnz] = hypre_BigBinarySearch( A_offd_colmap, big_col, m);
            if ( E_ext_j[E_nnz] >= 0)
            {
               /* this is an offdiagonal entry */
               E_ext_j[E_nnz]      = E_ext_j[E_nnz] + n;
               E_ext_data[E_nnz++] = A_ext_data[j];
            }
            else
            {
               /* skip capacity check */
               continue;
            }
         }
         /* capacity check, allocate new memory when full */
         if (E_nnz >= E_init_alloc)
         {
            HYPRE_Int tmp;
            tmp = E_init_alloc;
            E_init_alloc   = (HYPRE_Int)(E_init_alloc * EXPAND_FACT + 1);
            E_ext_j        = hypre_TReAlloc_v2(E_ext_j, HYPRE_Int, tmp, HYPRE_Int,
                                               E_init_alloc, HYPRE_MEMORY_HOST);
            E_ext_data     = hypre_TReAlloc_v2(E_ext_data, HYPRE_Real, tmp, HYPRE_Real,
                                               E_init_alloc, HYPRE_MEMORY_HOST);
         }
      }
   }
   E_ext_i[m] = E_nnz;

   /* 3: Free and finish up
    * Free memory, set E_i, E_j and E_data
    */

   *E_i     = E_ext_i;
   *E_j     = E_ext_j;
   *E_data  = E_ext_data;

   hypre_CSRMatrixDestroy(A_ext);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSortOffdColmap
 *
 * This function sort offdiagonal map as well as J array for offdiagonal part
 * A: The input CSR matrix.
 *
 * TODO (VPM): This work should be done via hypre_ParCSRMatrixPermute. This
 * function needs to be implemented.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSortOffdColmap(hypre_ParCSRMatrix *A)
{
   hypre_CSRMatrix      *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int            *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   HYPRE_Int             A_offd_nnz      = hypre_CSRMatrixNumNonzeros(A_offd);
   HYPRE_Int             A_offd_num_cols = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_MemoryLocation  memory_location = hypre_CSRMatrixMemoryLocation(A_offd);
   HYPRE_BigInt         *col_map_offd    = hypre_ParCSRMatrixColMapOffd(A);

   HYPRE_Int            *h_A_offd_j;

   HYPRE_Int            *perm  = hypre_TAlloc(HYPRE_Int, A_offd_num_cols, HYPRE_MEMORY_HOST);
   HYPRE_Int            *rperm = hypre_TAlloc(HYPRE_Int, A_offd_num_cols, HYPRE_MEMORY_HOST);
   HYPRE_Int             i;

   /* Set/Move A_offd_j on the host */
   if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE)
   {
      h_A_offd_j = hypre_TAlloc(HYPRE_Int, A_offd_nnz, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(h_A_offd_j, A_offd_j, HYPRE_Int, A_offd_nnz,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      h_A_offd_j = A_offd_j;
   }

   for (i = 0; i < A_offd_num_cols; i++)
   {
      perm[i] = i;
   }

   hypre_BigQsort2i(col_map_offd, perm, 0, A_offd_num_cols - 1);

   for (i = 0; i < A_offd_num_cols; i++)
   {
      rperm[perm[i]] = i;
   }

   for (i = 0; i < A_offd_nnz; i++)
   {
      h_A_offd_j[i] = rperm[h_A_offd_j[i]];
   }

   if (h_A_offd_j != A_offd_j)
   {
      hypre_TMemcpy(A_offd_j, h_A_offd_j, HYPRE_Int, A_offd_nnz,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TFree(h_A_offd_j, HYPRE_MEMORY_HOST);
   }

   /* Free memory */
   hypre_TFree(perm, HYPRE_MEMORY_HOST);
   hypre_TFree(rperm, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMBuildFinalPerm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILULocalRCMBuildFinalPerm(HYPRE_Int   start,
                                HYPRE_Int   end,
                                HYPRE_Int  *G_perm,
                                HYPRE_Int  *perm,
                                HYPRE_Int  *qperm,
                                HYPRE_Int **permp,
                                HYPRE_Int **qpermp)
{
   /* update to new index */
   HYPRE_Int i = 0;
   HYPRE_Int num_nodes = end - start;
   HYPRE_Int *perm_temp = hypre_TAlloc(HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);

   for ( i = 0 ; i < num_nodes ; i ++)
   {
      perm_temp[i] = perm[i + start];
   }
   for ( i = 0 ; i < num_nodes ; i ++)
   {
      perm[i + start] = perm_temp[G_perm[i]];
   }
   if (perm != qperm)
   {
      for ( i = 0 ; i < num_nodes ; i ++)
      {
         perm_temp[i] = qperm[i + start];
      }
      for ( i = 0 ; i < num_nodes ; i ++)
      {
         qperm[i + start] = perm_temp[G_perm[i]];
      }
   }

   *permp   = perm;
   *qpermp  = qperm;

   hypre_TFree(perm_temp, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCM
 *
 * This function computes the RCM ordering of a sub matrix of
 * sparse matrix B = A(perm,perm)
 * For nonsymmetrix problem, is the RCM ordering of B + B'
 * A: The input CSR matrix
 * start:      the start position of the submatrix in B
 * end:        the end position of the submatrix in B ( exclude end, [start,end) )
 * permp:      pointer to the row permutation array such that B = A(perm, perm)
 *             point to NULL if you want to work directly on A
 *             on return, permp will point to the new permutation where
 *             in [start, end) the matrix will reordered. if *permp is not NULL,
 *             we assume that it lives on the host memory at input. At output,
 *             it lives in the same memory location as A.
 * qpermp:     pointer to the col permutation array such that B = A(perm, perm)
 *             point to NULL or equal to permp if you want symmetric order
 *             on return, qpermp will point to the new permutation where
 *             in [start, end) the matrix will reordered. if *qpermp is not NULL,
 *             we assume that it lives on the host memory at input. At output,
 *             it lives in the same memory location as A.
 * sym:        set to nonzero to work on A only(symmetric), otherwise A + A'.
 *             WARNING: if you use non-symmetric reordering, that is,
 *             different row and col reordering, the resulting A might be non-symmetric.
 *             Be careful if you are using non-symmetric reordering
 *
 * TODO (VPM): Implement RCM computation on the device.
 *             Use IntArray for perm.
 *             Move this function and internal RCM calls to parcsr_mv.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILULocalRCM(hypre_CSRMatrix *A,
                  HYPRE_Int        start,
                  HYPRE_Int        end,
                  HYPRE_Int      **permp,
                  HYPRE_Int      **qpermp,
                  HYPRE_Int        sym)
{
   /* Input variables */
   HYPRE_Int               num_nodes       = end - start;
   HYPRE_Int               n               = hypre_CSRMatrixNumRows(A);
   HYPRE_Int               ncol            = hypre_CSRMatrixNumCols(A);
   HYPRE_MemoryLocation    memory_location = hypre_CSRMatrixMemoryLocation(A);
   HYPRE_Int               A_nnz           = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int              *A_i;
   HYPRE_Int              *A_j;

   /* Local variables */
   hypre_CSRMatrix         *GT        = NULL;
   hypre_CSRMatrix         *GGT       = NULL;
   hypre_CSRMatrix         *G         = NULL;
   HYPRE_Int               *G_i       = NULL;
   HYPRE_Int               *G_j       = NULL;
   HYPRE_Int               *G_perm    = NULL;
   HYPRE_Int               *perm_temp = NULL;
   HYPRE_Int               *rqperm    = NULL;
   HYPRE_Int               *d_perm    = NULL;
   HYPRE_Int               *d_qperm   = NULL;
   HYPRE_Int               *perm      = *permp;
   HYPRE_Int               *qperm     = *qpermp;

   HYPRE_Int                perm_is_qperm;
   HYPRE_Int                i, j, row, col, r1, r2;
   HYPRE_Int                G_nnz, G_capacity;

   /* Set flag for computing row and column permutations (true) or only row permutation (false) */
   perm_is_qperm = (perm == qperm) ? 1 : 0;

   /* 1: Preprosessing
    * Check error in input, set some parameters
    */
   if (num_nodes <= 0)
   {
      /* don't do this if we are too small */
      return hypre_error_flag;
   }

   if (n != ncol || end > n || start < 0)
   {
      /* don't do this if the input has error */
      hypre_printf("Error input, abort RCM\n");
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("ILULocalRCM");

   /* create permutation array if we don't have one yet */
   if (!perm)
   {
      perm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
      for (i = 0; i < n; i++)
      {
         perm[i] = i;
      }
   }

   /* Check for symmetric reordering, then point qperm to row reordering */
   if (!qperm)
   {
      qperm = perm;
   }

   /* Compute reverse qperm ordering */
   rqperm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      rqperm[qperm[i]] = i;
   }

   /* Set/Move A_i and A_j to host */
   if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE)
   {
      A_i = hypre_TAlloc(HYPRE_Int, n + 1, HYPRE_MEMORY_HOST);
      A_j = hypre_TAlloc(HYPRE_Int, A_nnz, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(A_i, hypre_CSRMatrixI(A), HYPRE_Int, n + 1,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(A_j, hypre_CSRMatrixJ(A), HYPRE_Int, A_nnz,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }
   else
   {
      A_i = hypre_CSRMatrixI(A);
      A_j = hypre_CSRMatrixJ(A);
   }

   /* 2: Build Graph
    * Build Graph for RCM ordering
    */
   G_nnz = 0;
   G_capacity = hypre_max((A_nnz * n * n / num_nodes / num_nodes) - num_nodes, 1);
   G_i = hypre_TAlloc(HYPRE_Int, num_nodes + 1, HYPRE_MEMORY_HOST);
   G_j = hypre_TAlloc(HYPRE_Int, G_capacity, HYPRE_MEMORY_HOST);

   /* TODO (VPM): Extend hypre_CSRMatrixPermute to replace the block below */
   for (i = 0; i < num_nodes; i++)
   {
      G_i[i] = G_nnz;
      row = perm[i + start];
      r1 = A_i[row];
      r2 = A_i[row + 1];
      for (j = r1; j < r2; j ++)
      {
         col = rqperm[A_j[j]];
         if (col != row && col >= start && col < end)
         {
            /* this is an entry in G */
            G_j[G_nnz++] = col - start;
            if (G_nnz >= G_capacity)
            {
               HYPRE_Int tmp = G_capacity;
               G_capacity = (HYPRE_Int) (G_capacity * EXPAND_FACT + 1);
               G_j = hypre_TReAlloc_v2(G_j, HYPRE_Int, tmp, HYPRE_Int,
                                       G_capacity, HYPRE_MEMORY_HOST);
            }
         }
      }
   }
   G_i[num_nodes] = G_nnz;

   /* Free memory */
   if (A_i != hypre_CSRMatrixI(A))
   {
      hypre_TFree(A_i, HYPRE_MEMORY_HOST);
   }
   if (A_j != hypre_CSRMatrixJ(A))
   {
      hypre_TFree(A_j, HYPRE_MEMORY_HOST);
   }

   /* Create matrix G on the host */
   G = hypre_CSRMatrixCreate(num_nodes, num_nodes, G_nnz);
   hypre_CSRMatrixMemoryLocation(G) = HYPRE_MEMORY_HOST;
   hypre_CSRMatrixI(G) = G_i;
   hypre_CSRMatrixJ(G) = G_j;

   /* Check if G is not empty (no need to do any kind of RCM) */
   if (G_nnz > 0)
   {
      /* Sum G with G' if G is nonsymmetric */
      if (!sym)
      {
         hypre_CSRMatrixData(G) = hypre_CTAlloc(HYPRE_Complex, G_nnz, HYPRE_MEMORY_HOST);
         hypre_CSRMatrixTranspose(G, &GT, 1);
         GGT = hypre_CSRMatrixAdd(1.0, G, 1.0, GT);
         hypre_CSRMatrixDestroy(G);
         hypre_CSRMatrixDestroy(GT);
         G = GGT;
         GGT = NULL;
      }

      /* 3: Build RCM on the host */
      G_perm = hypre_TAlloc(HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
      hypre_ILULocalRCMOrder(G, G_perm);

      /* 4: Post processing
       * Free, set value, return
       */

      /* update to new index */
      perm_temp = hypre_TAlloc(HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(perm_temp, &perm[start], HYPRE_Int, num_nodes,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      for (i = 0; i < num_nodes; i++)
      {
         perm[i + start] = perm_temp[G_perm[i]];
      }

      if (!perm_is_qperm)
      {
         hypre_TMemcpy(perm_temp, &qperm[start], HYPRE_Int, num_nodes,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_nodes; i++)
         {
            qperm[i + start] = perm_temp[G_perm[i]];
         }
      }
   }

   /* Move to device memory if needed */
   if (memory_location == HYPRE_MEMORY_DEVICE)
   {
      d_perm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(d_perm, perm, HYPRE_Int, n,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TFree(perm, HYPRE_MEMORY_HOST);

      perm = d_perm;
      if (perm_is_qperm)
      {
         qperm = d_perm;
      }
      else
      {
         d_qperm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
         hypre_TMemcpy(d_qperm, qperm, HYPRE_Int, n,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         hypre_TFree(qperm, HYPRE_MEMORY_HOST);

         qperm = d_qperm;
      }
   }

   /* Set output pointers */
   *permp  = perm;
   *qpermp = qperm;

   /* Free memory */
   hypre_CSRMatrixDestroy(G);
   hypre_TFree(G_perm, HYPRE_MEMORY_HOST);
   hypre_TFree(perm_temp, HYPRE_MEMORY_HOST);
   hypre_TFree(rqperm, HYPRE_MEMORY_HOST);

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMMindegree
 *
 * This function finds the unvisited node with the minimum degree
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILULocalRCMMindegree(HYPRE_Int  n,
                           HYPRE_Int *degree,
                           HYPRE_Int *marker,
                           HYPRE_Int *rootp)
{
   HYPRE_Int i;
   HYPRE_Int min_degree = n + 1;
   HYPRE_Int root = 0;

   for (i = 0 ; i < n ; i ++)
   {
      if (marker[i] < 0)
      {
         if (degree[i] < min_degree)
         {
            root = i;
            min_degree = degree[i];
         }
      }
   }
   *rootp = root;
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMOrder
 *
 * This function actually does the RCM ordering of a symmetric CSR matrix (entire)
 * A: the csr matrix, A_data is not needed
 * perm: the permutation array, space should be allocated outside
 * This is pure host code.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILULocalRCMOrder( hypre_CSRMatrix *A, HYPRE_Int *perm)
{
   HYPRE_Int      i, root;
   HYPRE_Int      *degree     = NULL;
   HYPRE_Int      *marker     = NULL;
   HYPRE_Int      *A_i        = hypre_CSRMatrixI(A);
   HYPRE_Int      n           = hypre_CSRMatrixNumRows(A);
   HYPRE_Int      current_num;
   /* get the degree for each node */
   degree = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   marker = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   for (i = 0 ; i < n ; i ++)
   {
      degree[i] = A_i[i + 1] - A_i[i];
      marker[i] = -1;
   }

   /* start RCM loop */
   current_num = 0;
   while (current_num < n)
   {
      hypre_ILULocalRCMMindegree( n, degree, marker, &root);
      /* This is a new connect component */
      hypre_ILULocalRCMFindPPNode(A, &root, marker);

      /* Numbering of this component */
      hypre_ILULocalRCMNumbering(A, root, marker, perm, &current_num);
   }

   /* Free */
   hypre_TFree(degree, HYPRE_MEMORY_HOST);
   hypre_TFree(marker, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMFindPPNode
 *
 * This function find a pseudo-peripheral node start from root
 *   A: the csr matrix, A_data is not needed
 *   rootp: pointer to the root, on return will be a end of the pseudo-peripheral
 *   marker: the marker array for unvisited node
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILULocalRCMFindPPNode( hypre_CSRMatrix *A, HYPRE_Int *rootp, HYPRE_Int *marker)
{
   HYPRE_Int      i, r1, r2, row, min_degree, lev_degree, nlev, newnlev;

   HYPRE_Int      root           = *rootp;
   HYPRE_Int      n              = hypre_CSRMatrixNumRows(A);
   HYPRE_Int     *A_i            = hypre_CSRMatrixI(A);

   /* at most n levels */
   HYPRE_Int     *level_i        = hypre_TAlloc(HYPRE_Int, n + 1, HYPRE_MEMORY_HOST);
   HYPRE_Int     *level_j        = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);

   /* build initial level structure from root */
   hypre_ILULocalRCMBuildLevel(A, root, marker, level_i, level_j, &newnlev);

   nlev = newnlev - 1;
   while (nlev < newnlev)
   {
      nlev = newnlev;
      r1 =  level_i[nlev - 1];
      r2 =  level_i[nlev];
      min_degree = n;
      for (i = r1 ; i < r2 ; i ++)
      {
         /* select the last level, pick min-degree node */
         row = level_j[i];
         lev_degree = A_i[row + 1] - A_i[row];
         if (min_degree > lev_degree)
         {
            min_degree = lev_degree;
            root = row;
         }
      }
      hypre_ILULocalRCMBuildLevel( A, root, marker, level_i, level_j, &newnlev);
   }

   /* Set output pointers */
   *rootp = root;

   /* Free */
   hypre_TFree(level_i, HYPRE_MEMORY_HOST);
   hypre_TFree(level_j, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMBuildLevel
 *
 * This function build level structure start from root
 *   A: the csr matrix, A_data is not needed
 *   root: pointer to the root
 *   marker: the marker array for unvisited node
 *   level_i: points to the start/end of position on level_j, similar to CSR Matrix
 *   level_j: store node number on each level
 *   nlevp: return the number of level on this level structure
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILULocalRCMBuildLevel(hypre_CSRMatrix *A, HYPRE_Int root, HYPRE_Int *marker,
                            HYPRE_Int *level_i, HYPRE_Int *level_j, HYPRE_Int *nlevp)
{
   HYPRE_Int      i, j, l1, l2, l_current, r1, r2, rowi, rowj, nlev;
   HYPRE_Int      *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int      *A_j = hypre_CSRMatrixJ(A);

   /* set first level first */
   level_i[0] = 0;
   level_j[0] = root;
   marker[root] = 0;
   nlev = 1;
   l1 = 0;
   l2 = 1;
   l_current = l2;

   /* Explore nbhds of all nodes in current level */
   while (l2 > l1)
   {
      level_i[nlev++] = l2;
      /* loop through last level */
      for (i = l1 ; i < l2 ; i ++)
      {
         /* the node to explore */
         rowi = level_j[i];
         r1 = A_i[rowi];
         r2 = A_i[rowi + 1];
         for (j = r1 ; j < r2 ; j ++)
         {
            rowj = A_j[j];
            if ( marker[rowj] < 0 )
            {
               /* Aha, an unmarked row */
               marker[rowj] = 0;
               level_j[l_current++] = rowj;
            }
         }
      }
      l1 = l2;
      l2 = l_current;
   }

   /* after this we always have a "ghost" last level */
   nlev --;

   /* reset marker */
   for (i = 0 ; i < l2 ; i ++)
   {
      marker[level_j[i]] = -1;
   }

   *nlevp = nlev;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMNumbering
 *
 * This function generate numbering for a connect component
 *   A: the csr matrix, A_data is not needed
 *   root: pointer to the root
 *   marker: the marker array for unvisited node
 *   perm: permutation array
 *   current_nump: number of nodes already have a perm value
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILULocalRCMNumbering(hypre_CSRMatrix *A, HYPRE_Int root, HYPRE_Int *marker, HYPRE_Int *perm,
                           HYPRE_Int *current_nump)
{
   HYPRE_Int        i, j, l1, l2, r1, r2, rowi, rowj, row_start, row_end;
   HYPRE_Int        *A_i        = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j        = hypre_CSRMatrixJ(A);
   HYPRE_Int        current_num = *current_nump;


   marker[root]        = 0;
   l1                  = current_num;
   perm[current_num++] = root;
   l2                  = current_num;

   while (l2 > l1)
   {
      /* loop through all nodes is current level */
      for (i = l1 ; i < l2 ; i ++)
      {
         rowi = perm[i];
         r1 = A_i[rowi];
         r2 = A_i[rowi + 1];
         row_start = current_num;
         for (j = r1 ; j < r2 ; j ++)
         {
            rowj = A_j[j];
            if (marker[rowj] < 0)
            {
               /* save the degree in marker and add it to perm */
               marker[rowj] = A_i[rowj + 1] - A_i[rowj];
               perm[current_num++] = rowj;
            }
         }
         row_end = current_num;
         hypre_ILULocalRCMQsort(perm, row_start, row_end - 1, marker);
      }
      l1 = l2;
      l2 = current_num;
   }

   //reverse
   hypre_ILULocalRCMReverse(perm, *current_nump, current_num - 1);
   *current_nump = current_num;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMQsort
 *
 * This qsort is very specialized, not worth to put into utilities
 * Sort a part of array perm based on degree value (ascend)
 * That is, if degree[perm[i]] < degree[perm[j]], we should have i < j
 *   perm: the perm array
 *   start: start in perm
 *   end: end in perm
 *   degree: degree array
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILULocalRCMQsort(HYPRE_Int *perm, HYPRE_Int start, HYPRE_Int end, HYPRE_Int *degree)
{
   HYPRE_Int i, mid;
   if (start >= end)
   {
      return hypre_error_flag;
   }

   hypre_swap(perm, start, (start + end) / 2);
   mid = start;

   /* Loop to split */
   for (i = start + 1 ; i <= end ; i ++)
   {
      if (degree[perm[i]] < degree[perm[start]])
      {
         hypre_swap(perm, ++mid, i);
      }
   }
   hypre_swap(perm, start, mid);
   hypre_ILULocalRCMQsort(perm, mid + 1, end, degree);
   hypre_ILULocalRCMQsort(perm, start, mid - 1, degree);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMReverse
 *
 * Last step in RCM, reverse it
 * perm: perm array
 * srart: start position
 * end: end position
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILULocalRCMReverse(HYPRE_Int *perm, HYPRE_Int start, HYPRE_Int end)
{
   HYPRE_Int     i, j;
   HYPRE_Int     mid = (start + end + 1) / 2;

   for (i = start, j = end ; i < mid ; i ++, j--)
   {
      hypre_swap(perm, i, j);
   }
   return hypre_error_flag;
}

/* TODO (VPM): Change this block to another file? */
#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * hypre_ParILUSchurGMRESDummySolveDevice
 *
 * Unit GMRES preconditioner, just copy data from one slot to another
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILUSchurGMRESDummySolveDevice( void             *ilu_vdata,
                                        void             *ilu_vdata2,
                                        hypre_ParVector  *f,
                                        hypre_ParVector  *u )
{
   hypre_ParILUData    *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix  *S        = hypre_ParILUDataMatS(ilu_data);
   HYPRE_Int            n_local  = hypre_ParCSRMatrixNumRows(S);

   hypre_Vector        *u_local = hypre_ParVectorLocalVector(u);
   HYPRE_Complex       *u_data  = hypre_VectorData(u_local);

   hypre_Vector        *f_local = hypre_ParVectorLocalVector(f);
   HYPRE_Complex       *f_data  = hypre_VectorData(f_local);

   hypre_TMemcpy(u_data, f_data, HYPRE_Real, n_local, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUSchurGMRESCommInfoDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILUSchurGMRESCommInfoDevice(void       *ilu_vdata,
                                     HYPRE_Int  *my_id,
                                     HYPRE_Int  *num_procs)
{
   /* get comm info from ilu_data */
   hypre_ParILUData     *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix   *S        = hypre_ParILUDataMatS(ilu_data);
   MPI_Comm              comm     = hypre_ParCSRMatrixComm(S);

   hypre_MPI_Comm_size(comm, num_procs);
   hypre_MPI_Comm_rank(comm, my_id);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILURAPSchurGMRESSolveDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPSchurGMRESSolveDevice( void               *ilu_vdata,
                                      void               *ilu_vdata2,
                                      hypre_ParVector    *par_f,
                                      hypre_ParVector    *par_u )
{
   hypre_ParILUData        *ilu_data  = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix      *S         = hypre_ParILUDataMatS(ilu_data);
   hypre_CSRMatrix         *SLU       = hypre_ParCSRMatrixDiag(S);

   hypre_ParVector         *par_rhs   = hypre_ParILUDataRhs(ilu_data);
   hypre_Vector            *rhs       = hypre_ParVectorLocalVector(par_rhs);
   hypre_Vector            *f         = hypre_ParVectorLocalVector(par_f);
   hypre_Vector            *u         = hypre_ParVectorLocalVector(par_u);

   /* L solve */
   hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, SLU, NULL, f, rhs);

   /* U solve */
   hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, SLU, NULL, rhs, u);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILURAPSchurGMRESMatvecDevice
 *
 * Compute y = alpha * S * x + beta * y
 *
 * TODO (VPM): Unify this function with hypre_ParILURAPSchurGMRESMatvecHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPSchurGMRESMatvecDevice( void           *matvec_data,
                                       HYPRE_Complex   alpha,
                                       void           *ilu_vdata,
                                       void           *x,
                                       HYPRE_Complex   beta,
                                       void           *y )
{
   /* Get matrix information first */
   hypre_ParILUData       *ilu_data    = (hypre_ParILUData*) ilu_vdata;
   HYPRE_Int               test_opt    = hypre_ParILUDataTestOption(ilu_data);
   hypre_ParCSRMatrix     *Aperm       = hypre_ParILUDataAperm(ilu_data);
   HYPRE_Int               n           = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(Aperm));
   hypre_CSRMatrix        *EiU         = hypre_ParILUDataMatEDevice(ilu_data);
   hypre_CSRMatrix        *iLF         = hypre_ParILUDataMatFDevice(ilu_data);
   hypre_CSRMatrix        *BLU         = hypre_ParILUDataMatBILUDevice(ilu_data);
   hypre_CSRMatrix        *C           = hypre_ParILUDataMatSILUDevice(ilu_data);

   hypre_ParVector        *x_vec       = (hypre_ParVector *) x;
   hypre_Vector           *x_local     = hypre_ParVectorLocalVector(x_vec);
   HYPRE_Real             *x_data      = hypre_VectorData(x_local);
   hypre_ParVector        *xtemp       = hypre_ParILUDataUTemp(ilu_data);
   hypre_Vector           *xtemp_local = hypre_ParVectorLocalVector(xtemp);
   HYPRE_Real             *xtemp_data  = hypre_VectorData(xtemp_local);

   hypre_ParVector        *y_vec       = (hypre_ParVector *) y;
   hypre_Vector           *y_local     = hypre_ParVectorLocalVector(y_vec);
   hypre_ParVector        *ytemp       = hypre_ParILUDataYTemp(ilu_data);
   hypre_Vector           *ytemp_local = hypre_ParVectorLocalVector(ytemp);
   HYPRE_Real             *ytemp_data  = hypre_VectorData(ytemp_local);

   HYPRE_Int               nLU;
   HYPRE_Int               m;
   hypre_Vector           *xtemp_upper;
   hypre_Vector           *xtemp_lower;
   hypre_Vector           *ytemp_upper;
   hypre_Vector           *ytemp_lower;

   switch (test_opt)
   {
      case 1:
         /* S = R * A * P */
         nLU                               = hypre_CSRMatrixNumRows(BLU);
         m                                 = n - nLU;
         xtemp_upper                       = hypre_SeqVectorCreate(nLU);
         ytemp_upper                       = hypre_SeqVectorCreate(nLU);
         xtemp_lower                       = hypre_SeqVectorCreate(m);
         hypre_VectorOwnsData(xtemp_upper) = 0;
         hypre_VectorOwnsData(ytemp_upper) = 0;
         hypre_VectorOwnsData(xtemp_lower) = 0;
         hypre_VectorData(xtemp_upper)     = xtemp_data;
         hypre_VectorData(ytemp_upper)     = ytemp_data;
         hypre_VectorData(xtemp_lower)     = xtemp_data + nLU;

         hypre_SeqVectorInitialize(xtemp_upper);
         hypre_SeqVectorInitialize(ytemp_upper);
         hypre_SeqVectorInitialize(xtemp_lower);

         /* first step, compute P*x put in y */
         /* -Fx */
         hypre_CSRMatrixMatvec(-1.0, iLF, x_local, 0.0, ytemp_upper);

         /* -L^{-1}Fx */
         /* L solve */
         hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, BLU, NULL, ytemp_local, xtemp_local);

         /* -U{-1}L^{-1}Fx */
         /* U solve */
         hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, BLU, NULL, xtemp_local, ytemp_local);

         /* now copy data to y_lower */
         hypre_TMemcpy(ytemp_data + nLU, x_data, HYPRE_Real, m,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

         /* second step, compute A*P*x store in xtemp */
         hypre_ParCSRMatrixMatvec(1.0, Aperm, ytemp, 0.0, xtemp);

         /* third step, compute R*A*P*x */
         /* solve L^{-1} */
         /* L solve */
         hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, BLU, NULL, xtemp_local, ytemp_local);

         /* U^{-1}L^{-1} */
         /* U solve */
         hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, BLU, NULL, ytemp_local, xtemp_local);

         /* -EU^{-1}L^{-1} */
         hypre_CSRMatrixMatvec(-alpha, EiU, xtemp_upper, beta, y_local);

         /* I*lower-EU^{-1}L^{-1}*upper */
         hypre_SeqVectorAxpy(alpha, xtemp_lower, y_local);

         hypre_SeqVectorDestroy(xtemp_upper);
         hypre_SeqVectorDestroy(ytemp_upper);
         hypre_SeqVectorDestroy(xtemp_lower);
         break;

      case 2:
         /* S = C - EU^{-1} * L^{-1}F */
         nLU                               = hypre_CSRMatrixNumRows(C);
         xtemp_upper                       = hypre_SeqVectorCreate(nLU);
         hypre_VectorOwnsData(xtemp_upper) = 0;
         hypre_VectorData(xtemp_upper)     = xtemp_data;

         hypre_SeqVectorInitialize(xtemp_upper);

         /* first step, compute EB^{-1}F*x put in y */
         /* -L^{-1}Fx */
         hypre_CSRMatrixMatvec(-1.0, iLF, x_local, 0.0, xtemp_upper);

         /* - alpha EU^{-1}L^{-1}Fx + beta * y */
         hypre_CSRMatrixMatvec(alpha, EiU, xtemp_upper, beta, y_local);

         /* alpha * C - alpha EU^{-1}L^{-1}Fx + beta y */
         hypre_CSRMatrixMatvec(alpha, C, x_local, 1.0, y_local);
         hypre_SeqVectorDestroy(xtemp_upper);
         break;

      case 3:
         /* S = C - EU^{-1} * L^{-1}F */
         nLU                               = hypre_CSRMatrixNumRows(C);
         xtemp_upper                       = hypre_SeqVectorCreate(nLU);
         hypre_VectorOwnsData(xtemp_upper) = 0;
         hypre_VectorData(xtemp_upper)     = xtemp_data;
         hypre_SeqVectorInitialize(xtemp_upper);

         /* first step, compute EB^{-1}F*x put in y */
         /* -Fx */
         hypre_CSRMatrixMatvec(-1.0, iLF, x_local, 0.0, xtemp_upper);

         /* -L^{-1}Fx */
         /* L solve */
         hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, BLU, NULL, xtemp_local, ytemp_local);

         /* -U^{-1}L^{-1}Fx */
         /* U solve */
         hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, BLU, NULL, ytemp_local, xtemp_local);

         /* - alpha EU^{-1}L^{-1}Fx + beta * y */
         hypre_CSRMatrixMatvec(alpha, EiU, xtemp_upper, beta, y_local);

         /* alpha * C - alpha EU^{-1}L^{-1}Fx + beta y */
         hypre_CSRMatrixMatvec(alpha, C, x_local, 1.0, y_local);
         hypre_SeqVectorDestroy(xtemp_upper);
         break;

   case 0: default:
         /* S = R * A * P */
         nLU                               = hypre_CSRMatrixNumRows(BLU);
         m                                 = n - nLU;
         xtemp_upper                       = hypre_SeqVectorCreate(nLU);
         ytemp_upper                       = hypre_SeqVectorCreate(nLU);
         ytemp_lower                       = hypre_SeqVectorCreate(m);
         hypre_VectorOwnsData(xtemp_upper) = 0;
         hypre_VectorOwnsData(ytemp_upper) = 0;
         hypre_VectorOwnsData(ytemp_lower) = 0;
         hypre_VectorData(xtemp_upper)     = xtemp_data;
         hypre_VectorData(ytemp_upper)     = ytemp_data;
         hypre_VectorData(ytemp_lower)     = ytemp_data + nLU;

         hypre_SeqVectorInitialize(xtemp_upper);
         hypre_SeqVectorInitialize(ytemp_upper);
         hypre_SeqVectorInitialize(ytemp_lower);

         /* first step, compute P*x put in y */
         /* -L^{-1}Fx */
         hypre_CSRMatrixMatvec(-1.0, iLF, x_local, 0.0, xtemp_upper);

         /* -U{-1}L^{-1}Fx */
         /* U solve */
         hypre_CSRMatrixTriLowerUpperSolveDevice('U', 0, BLU, NULL, xtemp_local, ytemp_local);

         /* now copy data to y_lower */
         hypre_TMemcpy(ytemp_data + nLU, x_data, HYPRE_Real, m,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

         /* second step, compute A*P*x store in xtemp */
         hypre_ParCSRMatrixMatvec(1.0, Aperm, ytemp, 0.0, xtemp);

         /* third step, compute R*A*P*x */
         /* copy partial data in */
         hypre_TMemcpy(ytemp_data + nLU, xtemp_data + nLU, HYPRE_Real, m,
                       HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

         /* solve L^{-1} */
         /* L solve */
         hypre_CSRMatrixTriLowerUpperSolveDevice('L', 1, BLU, NULL, xtemp_local, ytemp_local);

         /* -EU^{-1}L^{-1} */
         hypre_CSRMatrixMatvec(-alpha, EiU, ytemp_upper, beta, y_local);
         hypre_SeqVectorAxpy(alpha, ytemp_lower, y_local);

         /* over */
         hypre_SeqVectorDestroy(xtemp_upper);
         hypre_SeqVectorDestroy(ytemp_upper);
         hypre_SeqVectorDestroy(ytemp_lower);
         break;
   } /* switch (test_opt) */

   return hypre_error_flag;
}

#endif /* if defined(HYPRE_USING_GPU) */

/*--------------------------------------------------------------------------
 * hypre_ParILURAPSchurGMRESSolveHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPSchurGMRESSolveHost( void               *ilu_vdata,
                                    void               *ilu_vdata2,
                                    hypre_ParVector    *f,
                                    hypre_ParVector    *u )
{
   HYPRE_UNUSED_VAR(ilu_vdata2);

   hypre_ParILUData        *ilu_data     = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix      *L            = hypre_ParILUDataMatLModified(ilu_data);
   hypre_CSRMatrix         *L_diag       = hypre_ParCSRMatrixDiag(L);
   HYPRE_Int               *L_diag_i     = hypre_CSRMatrixI(L_diag);
   HYPRE_Int               *L_diag_j     = hypre_CSRMatrixJ(L_diag);
   HYPRE_Real              *L_diag_data  = hypre_CSRMatrixData(L_diag);

   HYPRE_Real              *D            = hypre_ParILUDataMatDModified(ilu_data);

   hypre_ParCSRMatrix      *U            = hypre_ParILUDataMatUModified(ilu_data);
   hypre_CSRMatrix         *U_diag       = hypre_ParCSRMatrixDiag(U);
   HYPRE_Int               *U_diag_i     = hypre_CSRMatrixI(U_diag);
   HYPRE_Int               *U_diag_j     = hypre_CSRMatrixJ(U_diag);
   HYPRE_Real              *U_diag_data  = hypre_CSRMatrixData(U_diag);

   HYPRE_Int               n             = hypre_CSRMatrixNumRows(L_diag);
   HYPRE_Int               nLU           = hypre_ParILUDataNLU(ilu_data);
   HYPRE_Int               m             = n - nLU;

   hypre_Vector            *f_local      = hypre_ParVectorLocalVector(f);
   HYPRE_Real              *f_data       = hypre_VectorData(f_local);
   hypre_Vector            *u_local      = hypre_ParVectorLocalVector(u);
   HYPRE_Real              *u_data       = hypre_VectorData(u_local);
   hypre_ParVector         *utemp        = hypre_ParILUDataUTemp(ilu_data);
   hypre_Vector            *utemp_local  = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real              *utemp_data   = hypre_VectorData(utemp_local);
   HYPRE_Int               *u_end        = hypre_ParILUDataUEnd(ilu_data);

   HYPRE_Int                i, j, k1, k2, col;

   /* permuted L solve */
   for (i = 0 ; i < m ; i ++)
   {
      utemp_data[i] = f_data[i];
      k1 = u_end[i + nLU] ; k2 = L_diag_i[i + nLU + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = L_diag_j[j];
         utemp_data[i] -= L_diag_data[j] * utemp_data[col - nLU];
      }
   }

   /* U solve */
   for (i = m - 1 ; i >= 0 ; i --)
   {
      u_data[i] = utemp_data[i];
      k1 = U_diag_i[i + nLU] ; k2 = U_diag_i[i + 1 + nLU];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = U_diag_j[j];
         u_data[i] -= U_diag_data[j] * u_data[col - nLU];
      }
      u_data[i] *= D[i];
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILURAPSchurGMRESCommInfoHost
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPSchurGMRESCommInfoHost(void      *ilu_vdata,
                                      HYPRE_Int *my_id,
                                      HYPRE_Int *num_procs)
{
   /* get comm info from ilu_data */
   hypre_ParILUData    *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix  *A        = hypre_ParILUDataMatA(ilu_data);
   MPI_Comm             comm     = hypre_ParCSRMatrixComm(A);

   hypre_MPI_Comm_size(comm, num_procs);
   hypre_MPI_Comm_rank(comm, my_id);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILURAPSchurGMRESMatvecHost
 *
 * Compute y = alpha * S * x + beta * y
 *
 * TODO (VPM): Unify this function with hypre_ParILURAPSchurGMRESMatvecDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPSchurGMRESMatvecHost( void          *matvec_data,
                                     HYPRE_Complex  alpha,
                                     void          *ilu_vdata,
                                     void          *x,
                                     HYPRE_Complex  beta,
                                     void          *y )
{
   HYPRE_UNUSED_VAR(matvec_data);

   /* get matrix information first */
   hypre_ParILUData        *ilu_data            = (hypre_ParILUData*) ilu_vdata;

   /* only option 1, use W and Z */
   HYPRE_Int               *u_end               = hypre_ParILUDataUEnd(ilu_data);
   hypre_ParCSRMatrix      *A                   = hypre_ParILUDataMatA(ilu_data);
   hypre_ParCSRMatrix      *mL                  = hypre_ParILUDataMatLModified(ilu_data);
   HYPRE_Real              *mD                  = hypre_ParILUDataMatDModified(ilu_data);
   hypre_ParCSRMatrix      *mU                  = hypre_ParILUDataMatUModified(ilu_data);

   hypre_CSRMatrix         *mL_diag             = hypre_ParCSRMatrixDiag(mL);
   HYPRE_Int               *mL_diag_i           = hypre_CSRMatrixI(mL_diag);
   HYPRE_Int               *mL_diag_j           = hypre_CSRMatrixJ(mL_diag);
   HYPRE_Real              *mL_diag_data        = hypre_CSRMatrixData(mL_diag);

   hypre_CSRMatrix         *mU_diag             = hypre_ParCSRMatrixDiag(mU);
   HYPRE_Int               *mU_diag_i           = hypre_CSRMatrixI(mU_diag);
   HYPRE_Int               *mU_diag_j           = hypre_CSRMatrixJ(mU_diag);
   HYPRE_Real              *mU_diag_data        = hypre_CSRMatrixData(mU_diag);

   HYPRE_Int               *perm                = hypre_ParILUDataPerm(ilu_data);
   HYPRE_Int               n                    = hypre_ParCSRMatrixNumRows(A);
   HYPRE_Int               nLU                  = hypre_ParILUDataNLU(ilu_data);

   hypre_ParVector         *x_vec               = (hypre_ParVector *) x;
   hypre_Vector            *x_local             = hypre_ParVectorLocalVector(x_vec);
   HYPRE_Real              *x_data              = hypre_VectorData(x_local);
   hypre_ParVector         *y_vec               = (hypre_ParVector *) y;
   hypre_Vector            *y_local             = hypre_ParVectorLocalVector(y_vec);
   HYPRE_Real              *y_data              = hypre_VectorData(y_local);

   hypre_ParVector         *utemp               = hypre_ParILUDataUTemp(ilu_data);
   hypre_Vector            *utemp_local         = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real              *utemp_data          = hypre_VectorData(utemp_local);

   hypre_ParVector         *ftemp               = hypre_ParILUDataFTemp(ilu_data);
   hypre_Vector            *ftemp_local         = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real              *ftemp_data          = hypre_VectorData(ftemp_local);

   hypre_ParVector         *ytemp               = hypre_ParILUDataYTemp(ilu_data);
   hypre_Vector            *ytemp_local         = hypre_ParVectorLocalVector(ytemp);
   HYPRE_Real              *ytemp_data          = hypre_VectorData(ytemp_local);

   HYPRE_Int               i, j, k1, k2, col;
   HYPRE_Real              one  = 1.0;
   HYPRE_Real              zero = 0.0;

   /* S = R * A * P */
   /* matvec */
   /* first compute alpha * P * x
    * P = [ -U\inv U_12 ]
    *     [  I          ]
    */
   /* matvec */
   for (i = 0 ; i < nLU ; i ++)
   {
      ytemp_data[i] = 0.0;
      k1 = u_end[i] ; k2 = mU_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = mU_diag_j[j];
         ytemp_data[i] -= alpha * mU_diag_data[j] * x_data[col - nLU];
      }
   }
   /* U solve */
   for (i = nLU - 1 ; i >= 0 ; i --)
   {
      ftemp_data[perm[i]] = ytemp_data[i];
      k1 = mU_diag_i[i] ; k2 = u_end[i];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = mU_diag_j[j];
         ftemp_data[perm[i]] -= mU_diag_data[j] * ftemp_data[perm[col]];
      }
      ftemp_data[perm[i]] *= mD[i];
   }

   /* update with I */
   for (i = nLU ; i < n ; i ++)
   {
      ftemp_data[perm[i]] = alpha * x_data[i - nLU];
   }

   /* apply alpha*A*P*x */
   hypre_ParCSRMatrixMatvec(one, A, ftemp, zero, utemp);

   // R = [-L21 L\inv, I]

   /* first is L solve */
   for (i = 0 ; i < nLU ; i ++)
   {
      ytemp_data[i] = utemp_data[perm[i]];
      k1 = mL_diag_i[i] ; k2 = mL_diag_i[i + 1];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = mL_diag_j[j];
         ytemp_data[i] -= mL_diag_data[j] * ytemp_data[col];
      }
   }

   /* apply -W * utemp on this, and take care of the I part */
   for (i = nLU ; i < n ; i ++)
   {
      y_data[i - nLU] = beta * y_data[i - nLU] + utemp_data[perm[i]];
      k1 = mL_diag_i[i] ; k2 = u_end[i];
      for (j = k1 ; j < k2 ; j ++)
      {
         col = mL_diag_j[j];
         y_data[i - nLU] -= mL_diag_data[j] * ytemp_data[col];
      }
   }

   return hypre_error_flag;
}

/******************************************************************************
 *
 * NSH create and solve and help functions.
 *
 * TODO (VPM): Move NSH code to separate files?
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_NSHCreate
 *--------------------------------------------------------------------------*/

void *
hypre_NSHCreate( void )
{
   hypre_ParNSHData  *nsh_data;

   nsh_data = hypre_CTAlloc(hypre_ParNSHData,  1, HYPRE_MEMORY_HOST);

   /* general data */
   hypre_ParNSHDataMatA(nsh_data)                  = NULL;
   hypre_ParNSHDataMatM(nsh_data)                  = NULL;
   hypre_ParNSHDataF(nsh_data)                     = NULL;
   hypre_ParNSHDataU(nsh_data)                     = NULL;
   hypre_ParNSHDataResidual(nsh_data)              = NULL;
   hypre_ParNSHDataRelResNorms(nsh_data)           = NULL;
   hypre_ParNSHDataNumIterations(nsh_data)         = 0;
   hypre_ParNSHDataL1Norms(nsh_data)               = NULL;
   hypre_ParNSHDataFinalRelResidualNorm(nsh_data)  = 0.0;
   hypre_ParNSHDataTol(nsh_data)                   = 1e-09;
   hypre_ParNSHDataLogging(nsh_data)               = 2;
   hypre_ParNSHDataPrintLevel(nsh_data)            = 2;
   hypre_ParNSHDataMaxIter(nsh_data)               = 5;

   hypre_ParNSHDataOperatorComplexity(nsh_data)    = 0.0;
   hypre_ParNSHDataDroptol(nsh_data)               = hypre_TAlloc(HYPRE_Real, 2, HYPRE_MEMORY_HOST);
   hypre_ParNSHDataOwnDroptolData(nsh_data)        = 1;
   hypre_ParNSHDataDroptol(nsh_data)[0]            = 1.0e-02;/* droptol for MR */
   hypre_ParNSHDataDroptol(nsh_data)[1]            = 1.0e-02;/* droptol for NSH */
   hypre_ParNSHDataUTemp(nsh_data)                 = NULL;
   hypre_ParNSHDataFTemp(nsh_data)                 = NULL;

   /* MR data */
   hypre_ParNSHDataMRMaxIter(nsh_data)             = 2;
   hypre_ParNSHDataMRTol(nsh_data)                 = 1e-09;
   hypre_ParNSHDataMRMaxRowNnz(nsh_data)           = 800;
   hypre_ParNSHDataMRColVersion(nsh_data)          = 0;

   /* NSH data */
   hypre_ParNSHDataNSHMaxIter(nsh_data)            = 2;
   hypre_ParNSHDataNSHTol(nsh_data)                = 1e-09;
   hypre_ParNSHDataNSHMaxRowNnz(nsh_data)          = 1000;

   return (void *) nsh_data;
}

/*--------------------------------------------------------------------------
 * hypre_NSHDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHDestroy( void *data )
{
   hypre_ParNSHData * nsh_data = (hypre_ParNSHData*) data;

   /* residual */
   hypre_ParVectorDestroy( hypre_ParNSHDataResidual(nsh_data) );
   hypre_ParNSHDataResidual(nsh_data) = NULL;

   /* residual norms */
   hypre_TFree( hypre_ParNSHDataRelResNorms(nsh_data), HYPRE_MEMORY_HOST );
   hypre_ParNSHDataRelResNorms(nsh_data) = NULL;

   /* l1 norms */
   hypre_TFree( hypre_ParNSHDataL1Norms(nsh_data), HYPRE_MEMORY_HOST );
   hypre_ParNSHDataL1Norms(nsh_data) = NULL;

   /* temp arrays */
   hypre_ParVectorDestroy( hypre_ParNSHDataUTemp(nsh_data) );
   hypre_ParVectorDestroy( hypre_ParNSHDataFTemp(nsh_data) );
   hypre_ParNSHDataUTemp(nsh_data) = NULL;
   hypre_ParNSHDataFTemp(nsh_data) = NULL;

   /* approx inverse matrix */
   hypre_ParCSRMatrixDestroy( hypre_ParNSHDataMatM(nsh_data) );
   hypre_ParNSHDataMatM(nsh_data) = NULL;

   /* droptol array */
   if (hypre_ParNSHDataOwnDroptolData(nsh_data))
   {
      hypre_TFree(hypre_ParNSHDataDroptol(nsh_data), HYPRE_MEMORY_HOST);
      hypre_ParNSHDataOwnDroptolData(nsh_data) = 0;
      hypre_ParNSHDataDroptol(nsh_data) = NULL;
   }

   /* nsh data */
   hypre_TFree(nsh_data, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHWriteSolverParams
 *
 * Print solver params
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHWriteSolverParams( void *nsh_vdata )
{
   hypre_ParNSHData  *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_printf("Newton-Schulz-Hotelling Setup parameters: \n");
   hypre_printf("NSH max iterations = %d \n", hypre_ParNSHDataNSHMaxIter(nsh_data));
   hypre_printf("NSH drop tolerance = %e \n", hypre_ParNSHDataDroptol(nsh_data)[1]);
   hypre_printf("NSH max nnz per row = %d \n", hypre_ParNSHDataNSHMaxRowNnz(nsh_data));
   hypre_printf("MR max iterations = %d \n", hypre_ParNSHDataMRMaxIter(nsh_data));
   hypre_printf("MR drop tolerance = %e \n", hypre_ParNSHDataDroptol(nsh_data)[0]);
   hypre_printf("MR max nnz per row = %d \n", hypre_ParNSHDataMRMaxRowNnz(nsh_data));
   hypre_printf("Operator Complexity (Fill factor) = %f \n",
                hypre_ParNSHDataOperatorComplexity(nsh_data));
   hypre_printf("\n Newton-Schulz-Hotelling Solver Parameters: \n");
   hypre_printf("Max number of iterations: %d\n", hypre_ParNSHDataMaxIter(nsh_data));
   hypre_printf("Stopping tolerance: %e\n", hypre_ParNSHDataTol(nsh_data));

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetPrintLevel( void *nsh_vdata, HYPRE_Int print_level )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataPrintLevel(nsh_data) = print_level;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetLogging( void *nsh_vdata, HYPRE_Int logging )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataLogging(nsh_data) = logging;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetMaxIter( void *nsh_vdata, HYPRE_Int max_iter )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMaxIter(nsh_data) = max_iter;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetTol( void *nsh_vdata, HYPRE_Real tol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataTol(nsh_data) = tol;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetGlobalSolver
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetGlobalSolver( void *nsh_vdata, HYPRE_Int global_solver )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataGlobalSolver(nsh_data) = global_solver;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetDropThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetDropThreshold( void *nsh_vdata, HYPRE_Real droptol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataDroptol(nsh_data)[0] = droptol;
   hypre_ParNSHDataDroptol(nsh_data)[1] = droptol;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetDropThresholdArray
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetDropThresholdArray( void *nsh_vdata, HYPRE_Real *droptol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   if (hypre_ParNSHDataOwnDroptolData(nsh_data))
   {
      hypre_TFree(hypre_ParNSHDataDroptol(nsh_data), HYPRE_MEMORY_HOST);
      hypre_ParNSHDataOwnDroptolData(nsh_data) = 0;
   }
   hypre_ParNSHDataDroptol(nsh_data) = droptol;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetMRMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetMRMaxIter( void *nsh_vdata, HYPRE_Int mr_max_iter )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMRMaxIter(nsh_data) = mr_max_iter;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetMRTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetMRTol( void *nsh_vdata, HYPRE_Real mr_tol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMRTol(nsh_data) = mr_tol;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetMRMaxRowNnz
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetMRMaxRowNnz( void *nsh_vdata, HYPRE_Int mr_max_row_nnz )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMRMaxRowNnz(nsh_data) = mr_max_row_nnz;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetColVersion
 *
 * set MR version, column version or global version
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetColVersion( void *nsh_vdata, HYPRE_Int mr_col_version )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMRColVersion(nsh_data) = mr_col_version;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetNSHMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetNSHMaxIter( void *nsh_vdata, HYPRE_Int nsh_max_iter )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataNSHMaxIter(nsh_data) = nsh_max_iter;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetNSHTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetNSHTol( void *nsh_vdata, HYPRE_Real nsh_tol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataNSHTol(nsh_data) = nsh_tol;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetNSHMaxRowNnz
 *
 * Set NSH max nonzeros of a row
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetNSHMaxRowNnz( void *nsh_vdata, HYPRE_Int nsh_max_row_nnz )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataNSHMaxRowNnz(nsh_data) = nsh_max_row_nnz;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixNormFro
 *
 * Compute the F norm of CSR matrix
 * A: the target CSR matrix
 * norm_io: output
 *
 * TODO (VPM): Move this function to seq_mv
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixNormFro(hypre_CSRMatrix *A, HYPRE_Real *norm_io)
{
   HYPRE_Real norm = 0.0;
   HYPRE_Real *data = hypre_CSRMatrixData(A);
   HYPRE_Int i, k;
   k = hypre_CSRMatrixNumNonzeros(A);

   /* main loop */
   for (i = 0 ; i < k ; i ++)
   {
      norm += data[i] * data[i];
   }
   *norm_io = hypre_sqrt(norm);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixResNormFro
 *
 * Compute the norm of I-A where I is identity matrix and A is a CSR matrix
 * A: the target CSR matrix
 * norm_io: the output
 *
 * TODO (VPM): Move this function to seq_mv
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixResNormFro(hypre_CSRMatrix *A, HYPRE_Real *norm_io)
{
   HYPRE_Real        norm = 0.0, value;
   HYPRE_Int         i, j, k1, k2, n;
   HYPRE_Int         *idx  = hypre_CSRMatrixI(A);
   HYPRE_Int         *cols = hypre_CSRMatrixJ(A);
   HYPRE_Real        *data = hypre_CSRMatrixData(A);

   n = hypre_CSRMatrixNumRows(A);
   /* main loop to sum up data */
   for (i = 0 ; i < n ; i ++)
   {
      k1 = idx[i];
      k2 = idx[i + 1];
      /* check if we have diagonal in A */
      if (k2 > k1)
      {
         if (cols[k1] == i)
         {
            /* reduce 1 on diagonal */
            value = data[k1] - 1.0;
            norm += value * value;
         }
         else
         {
            /* we don't have diagonal in A, so we need to add 1 to norm */
            norm += 1.0;
            norm += data[k1] * data[k1];
         }
      }
      else
      {
         /* we don't have diagonal in A, so we need to add 1 to norm */
         norm += 1.0;
      }
      /* and the rest of the code */
      for (j = k1 + 1 ; j < k2 ; j ++)
      {
         norm += data[j] * data[j];
      }
   }
   *norm_io = hypre_sqrt(norm);
   return hypre_error_flag;
}


/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixNormFro
 *
 * Compute the F norm of ParCSR matrix
 * A: the target CSR matrix
 *
 * TODO (VPM): Move this function to parcsr_mv
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixNormFro(hypre_ParCSRMatrix *A, HYPRE_Real *norm_io)
{
   HYPRE_Real        local_norm = 0.0;
   HYPRE_Real        global_norm;
   MPI_Comm          comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix   *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix   *A_offd = hypre_ParCSRMatrixOffd(A);

   hypre_CSRMatrixNormFro(A_diag, &local_norm);
   /* use global_norm to store offd for now */
   hypre_CSRMatrixNormFro(A_offd, &global_norm);

   /* square and sum them */
   local_norm *= local_norm;
   local_norm += global_norm * global_norm;

   /* do communication to get global total sum */
   hypre_MPI_Allreduce(&local_norm, &global_norm, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);

   *norm_io = hypre_sqrt(global_norm);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixResNormFro
 *
 * Compute the F norm of ParCSR matrix
 * Norm of I-A
 * A: the target CSR matrix
 *
 * TODO (VPM): Move this function to parcsr_mv
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixResNormFro(hypre_ParCSRMatrix *A, HYPRE_Real *norm_io)
{
   HYPRE_Real        local_norm = 0.0;
   HYPRE_Real        global_norm;
   MPI_Comm          comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix   *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix   *A_offd = hypre_ParCSRMatrixOffd(A);

   /* compute I-A for diagonal */
   hypre_CSRMatrixResNormFro(A_diag, &local_norm);

   /* use global_norm to store offd for now */
   hypre_CSRMatrixNormFro(A_offd, &global_norm);

   /* square and sum them */
   local_norm *= local_norm;
   local_norm += global_norm * global_norm;

   /* do communication to get global total sum */
   hypre_MPI_Allreduce(&local_norm, &global_norm, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);

   *norm_io = hypre_sqrt(global_norm);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTrace
 *
 * Compute the trace of CSR matrix
 * A: the target CSR matrix
 * trace_io: the output trace
 *
 * TODO (VPM): Move this function to seq_mv
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixTrace(hypre_CSRMatrix *A, HYPRE_Real *trace_io)
{
   HYPRE_Real  trace = 0.0;
   HYPRE_Int   *idx = hypre_CSRMatrixI(A);
   HYPRE_Int   *cols = hypre_CSRMatrixJ(A);
   HYPRE_Real  *data = hypre_CSRMatrixData(A);
   HYPRE_Int i, k1, k2, n;

   n = hypre_CSRMatrixNumRows(A);
   for (i = 0 ; i < n ; i ++)
   {
      k1 = idx[i];
      k2 = idx[i + 1];
      if (cols[k1] == i && k2 > k1)
      {
         /* only add when diagonal is nonzero */
         trace += data[k1];
      }
   }

   *trace_io = trace;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixDropInplace
 *
 * Apply dropping to CSR matrix
 * A: the target CSR matrix
 * droptol: all entries have smaller absolute value than this will be dropped
 * max_row_nnz: max nonzeros allowed for each row, only largest max_row_nnz kept
 * we NEVER drop diagonal entry if exists
 *
 * TODO (VPM): Move this function to seq_mv
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixDropInplace(hypre_CSRMatrix *A, HYPRE_Real droptol, HYPRE_Int max_row_nnz)
{
   HYPRE_Int      i, j, k1, k2;
   HYPRE_Int      *idx, len, drop_len;
   HYPRE_Real     *data, value, itol, norm;

   /* info of matrix A */
   HYPRE_Int      n = hypre_CSRMatrixNumRows(A);
   HYPRE_Int      m = hypre_CSRMatrixNumCols(A);
   HYPRE_Int      *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int      *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Real     *A_data = hypre_CSRMatrixData(A);
   HYPRE_Real     nnzA = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(A);

   /* new data */
   HYPRE_Int      *new_i;
   HYPRE_Int      *new_j;
   HYPRE_Real     *new_data;

   /* memory */
   HYPRE_Int      capacity;
   HYPRE_Int      ctrA;

   /* setup */
   capacity = (HYPRE_Int)(nnzA * 0.3 + 1);
   ctrA = 0;
   new_i = hypre_TAlloc(HYPRE_Int, n + 1, memory_location);
   new_j = hypre_TAlloc(HYPRE_Int, capacity, memory_location);
   new_data = hypre_TAlloc(HYPRE_Real, capacity, memory_location);

   idx = hypre_TAlloc(HYPRE_Int, m, memory_location);
   data = hypre_TAlloc(HYPRE_Real, m, memory_location);

   /* start of main loop */
   new_i[0] = 0;
   for (i = 0 ; i < n ; i ++)
   {
      len = 0;
      k1 = A_i[i];
      k2 = A_i[i + 1];
      /* compute droptol for current row */
      norm = 0.0;
      for (j = k1 ; j < k2 ; j ++)
      {
         norm += hypre_abs(A_data[j]);
      }
      if (k2 > k1)
      {
         norm /= (HYPRE_Real)(k2 - k1);
      }
      itol = droptol * norm;
      /* we don't want to drop the diagonal entry, so use an if statement here */
      if (A_j[k1] == i)
      {
         /* we have diagonal entry, skip it */
         idx[len] = A_j[k1];
         data[len++] = A_data[k1];
         for (j = k1 + 1 ; j < k2 ; j ++)
         {
            value = A_data[j];
            if (hypre_abs(value) < itol)
            {
               /* skip small element */
               continue;
            }
            idx[len] = A_j[j];
            data[len++] = A_data[j];
         }

         /* now apply drop on length */
         if (len > max_row_nnz)
         {
            drop_len = max_row_nnz;
            hypre_ILUMaxQSplitRabsI( data + 1, idx + 1, 0, drop_len - 1, len - 2);
         }
         else
         {
            /* don't need to sort, we keep all of them */
            drop_len = len;
         }
         /* copy data */
         while (ctrA + drop_len > capacity)
         {
            HYPRE_Int tmp = capacity;
            capacity = (HYPRE_Int)(capacity * EXPAND_FACT + 1);
            new_j = hypre_TReAlloc_v2(new_j, HYPRE_Int, tmp,
                                      HYPRE_Int, capacity, memory_location);
            new_data = hypre_TReAlloc_v2(new_data, HYPRE_Real, tmp,
                                         HYPRE_Real, capacity, memory_location);
         }
         hypre_TMemcpy(new_j + ctrA, idx, HYPRE_Int, drop_len, memory_location, memory_location);
         hypre_TMemcpy(new_data + ctrA, data, HYPRE_Real, drop_len, memory_location,
                       memory_location);
         ctrA += drop_len;
         new_i[i + 1] = ctrA;
      }
      else
      {
         /* we don't have diagonal entry */
         for (j = k1 ; j < k2 ; j ++)
         {
            value = A_data[j];
            if (hypre_abs(value) < itol)
            {
               /* skip small element */
               continue;
            }
            idx[len] = A_j[j];
            data[len++] = A_data[j];
         }

         /* now apply drop on length */
         if (len > max_row_nnz)
         {
            drop_len = max_row_nnz;
            hypre_ILUMaxQSplitRabsI( data, idx, 0, drop_len, len - 1);
         }
         else
         {
            /* don't need to sort, we keep all of them */
            drop_len = len;
         }

         /* copy data */
         while (ctrA + drop_len > capacity)
         {
            HYPRE_Int tmp = capacity;
            capacity = (HYPRE_Int)(capacity * EXPAND_FACT + 1);
            new_j = hypre_TReAlloc_v2(new_j, HYPRE_Int, tmp,
                                      HYPRE_Int, capacity, memory_location);
            new_data = hypre_TReAlloc_v2(new_data, HYPRE_Real, tmp,
                                         HYPRE_Real, capacity, memory_location);
         }
         hypre_TMemcpy(new_j + ctrA, idx, HYPRE_Int, drop_len, memory_location, memory_location);
         hypre_TMemcpy(new_data + ctrA, data, HYPRE_Real, drop_len, memory_location,
                       memory_location);
         ctrA += drop_len;
         new_i[i + 1] = ctrA;
      }
   }/* end of main loop */
   /* destory data if A own them */
   if (hypre_CSRMatrixOwnsData(A))
   {
      hypre_TFree(A_i, memory_location);
      hypre_TFree(A_j, memory_location);
      hypre_TFree(A_data, memory_location);
   }

   hypre_CSRMatrixI(A) = new_i;
   hypre_CSRMatrixJ(A) = new_j;
   hypre_CSRMatrixData(A) = new_data;
   hypre_CSRMatrixNumNonzeros(A) = ctrA;
   hypre_CSRMatrixOwnsData(A) = 1;

   hypre_TFree(idx, memory_location);
   hypre_TFree(data, memory_location);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal
 *
 * Compute the inverse with MR of original CSR matrix
 * Global(not by each column) and out place version
 * A: the input matrix
 * M: the output matrix
 * droptol: the dropping tolorance
 * tol: when to stop the iteration
 * eps_tol: to avoid divide by 0
 * max_row_nnz: max number of nonzeros per row
 * max_iter: max number of iterations
 * print_level: the print level of this algorithm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal(hypre_CSRMatrix  *matA,
                                             hypre_CSRMatrix **M,
                                             HYPRE_Real        droptol,
                                             HYPRE_Real        tol,
                                             HYPRE_Real        eps_tol,
                                             HYPRE_Int         max_row_nnz,
                                             HYPRE_Int         max_iter,
                                             HYPRE_Int         print_level)
{
   /* matrix A */
   HYPRE_Int         *A_i = hypre_CSRMatrixI(matA);
   HYPRE_Int         *A_j = hypre_CSRMatrixJ(matA);
   HYPRE_Real        *A_data = hypre_CSRMatrixData(matA);
   HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(matA);

   /* complexity */
   HYPRE_Real        nnzA = hypre_CSRMatrixNumNonzeros(matA);
   HYPRE_Real        nnzM = 1.0;

   /* inverse matrix */
   hypre_CSRMatrix   *inM = *M;
   hypre_CSRMatrix   *matM;
   HYPRE_Int         *M_i;
   HYPRE_Int         *M_j;
   HYPRE_Real        *M_data;

   /* idendity matrix */
   hypre_CSRMatrix   *matI;
   HYPRE_Int         *I_i;
   HYPRE_Int         *I_j;
   HYPRE_Real        *I_data;

   /* helper matrices */
   hypre_CSRMatrix   *matR;
   hypre_CSRMatrix   *matR_temp;
   hypre_CSRMatrix   *matZ;
   hypre_CSRMatrix   *matC;
   hypre_CSRMatrix   *matW;

   HYPRE_Real        time_s = 0.0, time_e;
   HYPRE_Int         i, k1, k2;
   HYPRE_Real        value, trace1, trace2, alpha, r_norm = 0.0;

   HYPRE_Int         n = hypre_CSRMatrixNumRows(matA);

   /* create initial guess and matrix I */
   matM = hypre_CSRMatrixCreate(n, n, n);
   M_i = hypre_TAlloc(HYPRE_Int, n + 1, memory_location);
   M_j = hypre_TAlloc(HYPRE_Int, n, memory_location);
   M_data = hypre_TAlloc(HYPRE_Real, n, memory_location);

   matI = hypre_CSRMatrixCreate(n, n, n);
   I_i = hypre_TAlloc(HYPRE_Int, n + 1, memory_location);
   I_j = hypre_TAlloc(HYPRE_Int, n, memory_location);
   I_data = hypre_TAlloc(HYPRE_Real, n, memory_location);

   /* now loop to create initial guess */
   M_i[0] = 0;
   I_i[0] = 0;
   for (i = 0 ; i < n ; i ++)
   {
      M_i[i + 1] = i + 1;
      M_j[i] = i;
      k1 = A_i[i];
      k2 = A_i[i + 1];
      if (k2 > k1)
      {
         if (A_j[k1] == i)
         {
            value = A_data[k1];
            if (hypre_abs(value) < MAT_TOL)
            {
               value = 1.0;
            }
            M_data[i] = 1.0 / value;
         }
         else
         {
            M_data[i] = 1.0;
         }
      }
      else
      {
         M_data[i] = 1.0;
      }
      I_i[i + 1] = i + 1;
      I_j[i] = i;
      I_data[i] = 1.0;
   }

   hypre_CSRMatrixI(matM) = M_i;
   hypre_CSRMatrixJ(matM) = M_j;
   hypre_CSRMatrixData(matM) = M_data;
   hypre_CSRMatrixOwnsData(matM) = 1;

   hypre_CSRMatrixI(matI) = I_i;
   hypre_CSRMatrixJ(matI) = I_j;
   hypre_CSRMatrixData(matI) = I_data;
   hypre_CSRMatrixOwnsData(matI) = 1;

   /* now start the main loop */
   if (print_level > 1)
   {
      /* time the iteration */
      time_s = hypre_MPI_Wtime();
   }

   /* main loop */
   for (i = 0 ; i < max_iter ; i ++)
   {
      nnzM = hypre_CSRMatrixNumNonzeros(matM);
      /* R = I - AM */
      matR_temp = hypre_CSRMatrixMultiply(matA, matM);

      hypre_CSRMatrixScale(matR_temp, -1.0);

      matR = hypre_CSRMatrixAdd(1.0, matI, 1.0, matR_temp);
      hypre_CSRMatrixDestroy(matR_temp);

      /* r_norm */
      hypre_CSRMatrixNormFro(matR, &r_norm);
      if (r_norm < tol)
      {
         break;
      }

      /* Z = MR and dropping */
      matZ = hypre_CSRMatrixMultiply(matM, matR);
      //hypre_CSRMatrixNormFro(matZ, &z_norm);
      hypre_CSRMatrixDropInplace(matZ, droptol, max_row_nnz);

      /* C = A*Z */
      matC = hypre_CSRMatrixMultiply(matA, matZ);

      /* W = R' * C */
      hypre_CSRMatrixTranspose(matR, &matR_temp, 1);
      matW = hypre_CSRMatrixMultiply(matR_temp, matC);

      /* trace and alpha */
      hypre_CSRMatrixTrace(matW, &trace1);
      hypre_CSRMatrixNormFro(matC, &trace2);
      trace2 *= trace2;

      if (hypre_abs(trace2) < eps_tol)
      {
         break;
      }

      alpha = trace1 / trace2;

      /* M - M + alpha * Z */
      hypre_CSRMatrixScale(matZ, alpha);

      hypre_CSRMatrixDestroy(matR);
      matR = hypre_CSRMatrixAdd(1.0, matM, 1.0, matZ);
      hypre_CSRMatrixDestroy(matM);
      matM = matR;

      hypre_CSRMatrixDestroy(matZ);
      hypre_CSRMatrixDestroy(matW);
      hypre_CSRMatrixDestroy(matC);
      hypre_CSRMatrixDestroy(matR_temp);
   } /* end of main loop i for compute inverse matrix */

   /* time if we need to print */
   if (print_level > 1)
   {
      time_e = hypre_MPI_Wtime();
      if (i == 0)
      {
         i = 1;
      }
      hypre_printf("matrix size %5d\nfinal norm at loop %5d is %16.12f, time per iteration is %16.12f, complexity is %16.12f out of maximum %16.12f\n",
                   n, i, r_norm, (time_e - time_s) / i, nnzM / nnzA, n / nnzA * n);
   }

   hypre_CSRMatrixDestroy(matI);
   if (inM)
   {
      hypre_CSRMatrixDestroy(inM);
   }
   *M = matM;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUParCSRInverseNSH
 *
 * Compute inverse with NSH method
 * Use MR to get local initial guess
 * A: input matrix
 * M: output matrix
 * droptol: droptol array. droptol[0] for MR and droptol[1] for NSH.
 * mr_tol: tol for stop iteration for MR
 * nsh_tol: tol for stop iteration for NSH
 * esp_tol: tol for avoid divide by 0
 * mr_max_row_nnz: max number of nonzeros for MR
 * nsh_max_row_nnz: max number of nonzeros for NSH
 * mr_max_iter: max number of iterations for MR
 * nsh_max_iter: max number of iterations for NSH
 * mr_col_version: column version of global version
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUParCSRInverseNSH(hypre_ParCSRMatrix  *A,
                          hypre_ParCSRMatrix **M,
                          HYPRE_Real          *droptol,
                          HYPRE_Real           mr_tol,
                          HYPRE_Real           nsh_tol,
                          HYPRE_Real           eps_tol,
                          HYPRE_Int            mr_max_row_nnz,
                          HYPRE_Int            nsh_max_row_nnz,
                          HYPRE_Int            mr_max_iter,
                          HYPRE_Int            nsh_max_iter,
                          HYPRE_Int            mr_col_version,
                          HYPRE_Int            print_level)
{
   HYPRE_UNUSED_VAR(nsh_max_row_nnz);

   /* data slots for matrices */
   hypre_ParCSRMatrix      *matM = NULL;
   hypre_ParCSRMatrix      *inM = *M;
   hypre_ParCSRMatrix      *AM, *MAM;
   HYPRE_Real              norm, s_norm;
   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int               myid;
   HYPRE_MemoryLocation    memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   hypre_CSRMatrix         *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix         *M_diag = NULL;
   hypre_CSRMatrix         *M_offd;
   HYPRE_Int               *M_offd_i;

   HYPRE_Real              time_s, time_e;

   HYPRE_Int               n = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int               i;

   /* setup */
   hypre_MPI_Comm_rank(comm, &myid);

   M_offd_i = hypre_CTAlloc(HYPRE_Int, n + 1, memory_location);

   if (mr_col_version)
   {
      hypre_printf("Column version is not yet support, switch to global version\n");
   }

   /* call MR to build loacl initial matrix
    * droptol here should be larger
    * we want same number for MR and NSH to let user set them eaiser
    * but we don't want a too dense MR initial guess
    */
   hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal(A_diag, &M_diag, droptol[0] * 10.0, mr_tol, eps_tol,
                                                mr_max_row_nnz, mr_max_iter, print_level );

   /* create parCSR matM */
   matM = hypre_ParCSRMatrixCreate( comm,
                                    hypre_ParCSRMatrixGlobalNumRows(A),
                                    hypre_ParCSRMatrixGlobalNumRows(A),
                                    hypre_ParCSRMatrixRowStarts(A),
                                    hypre_ParCSRMatrixColStarts(A),
                                    0,
                                    hypre_CSRMatrixNumNonzeros(M_diag),
                                    0 );

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matM));
   hypre_ParCSRMatrixDiag(matM) = M_diag;

   M_offd = hypre_ParCSRMatrixOffd(matM);
   hypre_CSRMatrixI(M_offd) = M_offd_i;
   hypre_CSRMatrixNumRownnz(M_offd) = 0;
   hypre_CSRMatrixOwnsData(M_offd)  = 1;

   /* now start NSH
    * Mj+1 = 2Mj - MjAMj
    */

   AM = hypre_ParMatmul(A, matM);
   hypre_ParCSRMatrixResNormFro(AM, &norm);
   s_norm = norm;
   hypre_ParCSRMatrixDestroy(AM);
   if (print_level > 1)
   {
      if (myid == 0)
      {
         hypre_printf("before NSH the norm is %16.12f\n", norm);
      }
      time_s = hypre_MPI_Wtime();
   }

   for (i = 0 ; i < nsh_max_iter ; i ++)
   {
      /* compute XjAXj */
      AM = hypre_ParMatmul(A, matM);
      hypre_ParCSRMatrixResNormFro(AM, &norm);
      if (norm < nsh_tol)
      {
         break;
      }
      MAM = hypre_ParMatmul(matM, AM);
      hypre_ParCSRMatrixDestroy(AM);

      /* apply dropping */
      //hypre_ParCSRMatrixNormFro(MAM, &norm);
      /* drop small entries based on 2-norm */
      hypre_ParCSRMatrixDropSmallEntries(MAM, droptol[1], 2);

      /* update Mj+1 = 2Mj - MjAMj
       * the result holds it own start/end data!
       */
      hypre_ParCSRMatrixAdd(2.0, matM, -1.0, MAM, &AM);
      hypre_ParCSRMatrixDestroy(matM);
      matM = AM;

      /* destroy */
      hypre_ParCSRMatrixDestroy(MAM);
   }

   if (print_level > 1)
   {
      time_e = hypre_MPI_Wtime();
      /* at this point of time, norm has to be already computed */
      if (i == 0)
      {
         i = 1;
      }
      if (myid == 0)
      {
         hypre_printf("after %5d NSH iterations the norm is %16.12f, time per iteration is %16.12f\n", i,
                      norm, (time_e - time_s) / i);
      }
   }

   if (s_norm < norm)
   {
      /* the residual norm increase after NSH iteration, need to let user know */
      if (myid == 0)
      {
         hypre_printf("Warning: NSH divergence, probably bad approximate invese matrix.\n");
      }
   }

   if (inM)
   {
      hypre_ParCSRMatrixDestroy(inM);
   }
   *M = matM;

   return hypre_error_flag;
}
