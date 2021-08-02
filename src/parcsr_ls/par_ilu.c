/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
#include "_hypre_utilities.hpp"
#include "par_ilu.h"

/* Create */
void *
hypre_ILUCreate()
{
   hypre_ParILUData                       *ilu_data;

   ilu_data                               = hypre_CTAlloc(hypre_ParILUData,  1, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_CUDA
   hypre_ParILUDataMatLMatrixDescription(ilu_data) = NULL;
   hypre_ParILUDataMatUMatrixDescription(ilu_data) = NULL;
   hypre_ParILUDataMatBLILUSolveInfo(ilu_data) = NULL;
   hypre_ParILUDataMatBUILUSolveInfo(ilu_data) = NULL;
   hypre_ParILUDataMatSLILUSolveInfo(ilu_data) = NULL;
   hypre_ParILUDataMatSUILUSolveInfo(ilu_data) = NULL;
   hypre_ParILUDataILUSolveBuffer(ilu_data) = NULL;
   hypre_ParILUDataILUSolvePolicy(ilu_data) = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   hypre_ParILUDataAperm(ilu_data) = NULL;
   hypre_ParILUDataMatBILUDevice(ilu_data) = NULL;
   hypre_ParILUDataMatSILUDevice(ilu_data) = NULL;
   hypre_ParILUDataMatEDevice(ilu_data) = NULL;
   hypre_ParILUDataMatFDevice(ilu_data) = NULL;
   hypre_ParILUDataR(ilu_data) = NULL;
   hypre_ParILUDataP(ilu_data) = NULL;
   hypre_ParILUDataFTempUpper(ilu_data) = NULL;
   hypre_ParILUDataUTempLower(ilu_data) = NULL;
   hypre_ParILUDataMatAFakeDiagonal(ilu_data) = NULL;
#endif

   /* general data */
   hypre_ParILUDataGlobalSolver(ilu_data) = 0;
   hypre_ParILUDataMatA(ilu_data) = NULL;
   hypre_ParILUDataMatL(ilu_data) = NULL;
   hypre_ParILUDataMatD(ilu_data) = NULL;
   hypre_ParILUDataMatU(ilu_data) = NULL;
   hypre_ParILUDataMatS(ilu_data) = NULL;
   hypre_ParILUDataSchurSolver(ilu_data) = NULL;
   hypre_ParILUDataSchurPrecond(ilu_data) = NULL;
   hypre_ParILUDataRhs(ilu_data) = NULL;
   hypre_ParILUDataX(ilu_data) = NULL;

   hypre_ParILUDataDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real,3,HYPRE_MEMORY_HOST);
   hypre_ParILUDataDroptol(ilu_data)[0] = 1.0e-02;/* droptol for B */
   hypre_ParILUDataDroptol(ilu_data)[1] = 1.0e-02;/* droptol for E and F */
   hypre_ParILUDataDroptol(ilu_data)[2] = 1.0e-02;/* droptol for S */
   hypre_ParILUDataLfil(ilu_data) = 0;
   hypre_ParILUDataMaxRowNnz(ilu_data) = 1000;
   hypre_ParILUDataCFMarkerArray(ilu_data) = NULL;
   hypre_ParILUDataPerm(ilu_data) = NULL;
   hypre_ParILUDataQPerm(ilu_data) = NULL;
   hypre_ParILUDataTolDDPQ(ilu_data) = 1.0e-01;

   hypre_ParILUDataF(ilu_data) = NULL;
   hypre_ParILUDataU(ilu_data) = NULL;
   hypre_ParILUDataFTemp(ilu_data) = NULL;
   hypre_ParILUDataUTemp(ilu_data) = NULL;
   hypre_ParILUDataXTemp(ilu_data) = NULL;
   hypre_ParILUDataYTemp(ilu_data) = NULL;
   hypre_ParILUDataUExt(ilu_data) = NULL;
   hypre_ParILUDataFExt(ilu_data) = NULL;
   hypre_ParILUDataResidual(ilu_data) = NULL;
   hypre_ParILUDataRelResNorms(ilu_data) = NULL;

   hypre_ParILUDataNumIterations(ilu_data) = 0;

   hypre_ParILUDataMaxIter(ilu_data) = 20;
   hypre_ParILUDataTol(ilu_data) = 1.0e-7;

   hypre_ParILUDataLogging(ilu_data) = 0;
   hypre_ParILUDataPrintLevel(ilu_data) = 0;

   hypre_ParILUDataL1Norms(ilu_data) = NULL;

   hypre_ParILUDataOperatorComplexity(ilu_data) = 0.;

   hypre_ParILUDataIluType(ilu_data) = 0;
   hypre_ParILUDataNLU(ilu_data) = 0;
   hypre_ParILUDataNI(ilu_data) = 0;
   hypre_ParILUDataUEnd(ilu_data) = NULL;

   /* reordering_type default to use local RCM */
   hypre_ParILUDataReorderingType(ilu_data) = 1;

   /* see hypre_ILUSetType for more default values */
   hypre_ParILUDataTestOption(ilu_data) = 0;

   /* -> GENERAL-SLOTS */
   hypre_ParILUDataSchurSolverLogging(ilu_data) = 0;
   hypre_ParILUDataSchurSolverPrintLevel(ilu_data) = 0;

   /* -> SCHUR-GMRES */
   hypre_ParILUDataSchurGMRESKDim(ilu_data) = 5;
   hypre_ParILUDataSchurGMRESMaxIter(ilu_data) = 5;
   hypre_ParILUDataSchurGMRESTol(ilu_data) = 0.0;
   hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data) = 0.0;
   hypre_ParILUDataSchurGMRESRelChange(ilu_data) = 0;

   /* schur precond data */
   hypre_ParILUDataSchurPrecondIluType(ilu_data) = 0;
   hypre_ParILUDataSchurPrecondIluLfil(ilu_data) = 0;
   hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data) = 100;
   hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = NULL;/* this is not the default option, set it only when switched to */
   hypre_ParILUDataSchurPrecondPrintLevel(ilu_data) = 0;
   hypre_ParILUDataSchurPrecondMaxIter(ilu_data) = 1;
   hypre_ParILUDataSchurPrecondTol(ilu_data) = 0.0;

   /* -> SCHUR-NSH */
   hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data) = 5;
   hypre_ParILUDataSchurNSHSolveTol(ilu_data) = 0.0;
   hypre_ParILUDataSchurNSHDroptol(ilu_data) = NULL;/* this is not the default option, set it only when switched to */

   hypre_ParILUDataSchurNSHMaxNumIter(ilu_data) = 2;
   hypre_ParILUDataSchurNSHMaxRowNnz(ilu_data) = 1000;
   hypre_ParILUDataSchurNSHTol(ilu_data) = 1e-09;

   hypre_ParILUDataSchurMRMaxIter(ilu_data) = 2;
   hypre_ParILUDataSchurMRColVersion(ilu_data) = 0;
   hypre_ParILUDataSchurMRMaxRowNnz(ilu_data) = 200;
   hypre_ParILUDataSchurMRTol(ilu_data) = 1e-09;

   return (void *)                        ilu_data;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* Destroy */
HYPRE_Int
hypre_ILUDestroy( void *data )
{
   hypre_ParILUData * ilu_data = (hypre_ParILUData*) data;

#ifdef HYPRE_USING_CUDA
   if(hypre_ParILUDataILUSolveBuffer(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataILUSolveBuffer(ilu_data), HYPRE_MEMORY_DEVICE);
      hypre_ParILUDataILUSolveBuffer(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatLMatrixDescription(ilu_data))
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyMatDescr(hypre_ParILUDataMatLMatrixDescription(ilu_data))) );
      hypre_ParILUDataMatLMatrixDescription(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatUMatrixDescription(ilu_data))
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyMatDescr(hypre_ParILUDataMatUMatrixDescription(ilu_data))) );
      hypre_ParILUDataMatUMatrixDescription(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatALILUSolveInfo(ilu_data))
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(hypre_ParILUDataMatALILUSolveInfo(ilu_data))) );
      hypre_ParILUDataMatALILUSolveInfo(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatAUILUSolveInfo(ilu_data))
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(hypre_ParILUDataMatAUILUSolveInfo(ilu_data))) );
      hypre_ParILUDataMatAUILUSolveInfo(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatBLILUSolveInfo(ilu_data))
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(hypre_ParILUDataMatBLILUSolveInfo(ilu_data))) );
      hypre_ParILUDataMatBLILUSolveInfo(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatBUILUSolveInfo(ilu_data))
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(hypre_ParILUDataMatBUILUSolveInfo(ilu_data))) );
      hypre_ParILUDataMatBUILUSolveInfo(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatSLILUSolveInfo(ilu_data))
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(hypre_ParILUDataMatSLILUSolveInfo(ilu_data))) );
      hypre_ParILUDataMatSLILUSolveInfo(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatSUILUSolveInfo(ilu_data))
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(hypre_ParILUDataMatSUILUSolveInfo(ilu_data))) );
      hypre_ParILUDataMatSUILUSolveInfo(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatAILUDevice(ilu_data))
   {
      hypre_CSRMatrixDestroy( hypre_ParILUDataMatAILUDevice(ilu_data) );
      hypre_ParILUDataMatAILUDevice(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatBILUDevice(ilu_data))
   {
      hypre_CSRMatrixDestroy( hypre_ParILUDataMatBILUDevice(ilu_data) );
      hypre_ParILUDataMatBILUDevice(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatSILUDevice(ilu_data))
   {
      hypre_CSRMatrixDestroy( hypre_ParILUDataMatSILUDevice(ilu_data) );
      hypre_ParILUDataMatSILUDevice(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatEDevice(ilu_data))
   {
      hypre_CSRMatrixDestroy( hypre_ParILUDataMatEDevice(ilu_data) );
      hypre_ParILUDataMatEDevice(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatFDevice(ilu_data))
   {
      hypre_CSRMatrixDestroy( hypre_ParILUDataMatFDevice(ilu_data) );
      hypre_ParILUDataMatFDevice(ilu_data) = NULL;
   }
   if(hypre_ParILUDataAperm(ilu_data))
   {
      hypre_ParCSRMatrixDestroy( hypre_ParILUDataAperm(ilu_data) );
      hypre_ParILUDataAperm(ilu_data) = NULL;
   }
   if(hypre_ParILUDataR(ilu_data))
   {
      hypre_ParCSRMatrixDestroy( hypre_ParILUDataR(ilu_data) );
      hypre_ParILUDataR(ilu_data) = NULL;
   }
   if(hypre_ParILUDataP(ilu_data))
   {
      hypre_ParCSRMatrixDestroy( hypre_ParILUDataP(ilu_data) );
      hypre_ParILUDataP(ilu_data) = NULL;
   }
   if(hypre_ParILUDataFTempUpper(ilu_data))
   {
      hypre_SeqVectorDestroy( hypre_ParILUDataFTempUpper(ilu_data) );
      hypre_ParILUDataFTempUpper(ilu_data) = NULL;
   }
   if(hypre_ParILUDataUTempLower(ilu_data))
   {
      hypre_SeqVectorDestroy( hypre_ParILUDataUTempLower(ilu_data) );
      hypre_ParILUDataUTempLower(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatAFakeDiagonal(ilu_data))
   {
      hypre_TFree( hypre_ParILUDataMatAFakeDiagonal(ilu_data), HYPRE_MEMORY_DEVICE);
      hypre_ParILUDataMatAFakeDiagonal(ilu_data) = NULL;
   }
#endif

   /* final residual vector */
   if(hypre_ParILUDataResidual(ilu_data))
   {
      hypre_ParVectorDestroy( hypre_ParILUDataResidual(ilu_data) );
      hypre_ParILUDataResidual(ilu_data) = NULL;
   }
   if(hypre_ParILUDataRelResNorms(ilu_data))
   {
      hypre_TFree( hypre_ParILUDataRelResNorms(ilu_data) , HYPRE_MEMORY_HOST);
      hypre_ParILUDataRelResNorms(ilu_data) = NULL;
   }
   /* temp vectors for solve phase */
   if(hypre_ParILUDataUTemp(ilu_data))
   {
      hypre_ParVectorDestroy( hypre_ParILUDataUTemp(ilu_data) );
      hypre_ParILUDataUTemp(ilu_data) = NULL;
   }
   if(hypre_ParILUDataFTemp(ilu_data))
   {
      hypre_ParVectorDestroy( hypre_ParILUDataFTemp(ilu_data) );
      hypre_ParILUDataFTemp(ilu_data) = NULL;
   }
   if(hypre_ParILUDataXTemp(ilu_data))
   {
      hypre_ParVectorDestroy( hypre_ParILUDataXTemp(ilu_data) );
      hypre_ParILUDataXTemp(ilu_data) = NULL;
   }
   if(hypre_ParILUDataYTemp(ilu_data))
   {
      hypre_ParVectorDestroy( hypre_ParILUDataYTemp(ilu_data) );
      hypre_ParILUDataYTemp(ilu_data) = NULL;
   }
   if(hypre_ParILUDataUExt(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataUExt(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataUExt(ilu_data) = NULL;
   }
   if(hypre_ParILUDataFExt(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataFExt(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataFExt(ilu_data) = NULL;
   }
   if(hypre_ParILUDataRhs(ilu_data))
   {
      hypre_ParVectorDestroy( hypre_ParILUDataRhs(ilu_data) );
      hypre_ParILUDataRhs(ilu_data) = NULL;
   }
   if(hypre_ParILUDataX(ilu_data))
   {
      hypre_ParVectorDestroy( hypre_ParILUDataX(ilu_data) );
      hypre_ParILUDataX(ilu_data) = NULL;
   }
   /* l1_norms */
   if(hypre_ParILUDataL1Norms(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataL1Norms(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataL1Norms(ilu_data) = NULL;
   }

   /* u_end */
   if(hypre_ParILUDataUEnd(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataUEnd(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataUEnd(ilu_data) = NULL;
   }

   /* Factors */
   if(hypre_ParILUDataMatL(ilu_data))
   {
      hypre_ParCSRMatrixDestroy(hypre_ParILUDataMatL(ilu_data));
      hypre_ParILUDataMatL(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatU(ilu_data))
   {
      hypre_ParCSRMatrixDestroy(hypre_ParILUDataMatU(ilu_data));
      hypre_ParILUDataMatU(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatD(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataMatD(ilu_data), HYPRE_MEMORY_DEVICE);
      hypre_ParILUDataMatD(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatLModified(ilu_data))
   {
      hypre_ParCSRMatrixDestroy(hypre_ParILUDataMatLModified(ilu_data));
      hypre_ParILUDataMatLModified(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatUModified(ilu_data))
   {
      hypre_ParCSRMatrixDestroy(hypre_ParILUDataMatUModified(ilu_data));
      hypre_ParILUDataMatUModified(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatDModified(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataMatDModified(ilu_data), HYPRE_MEMORY_DEVICE);
      hypre_ParILUDataMatDModified(ilu_data) = NULL;
   }
   if(hypre_ParILUDataMatS(ilu_data))
   {
      hypre_ParCSRMatrixDestroy(hypre_ParILUDataMatS(ilu_data));
      hypre_ParILUDataMatS(ilu_data) = NULL;
   }
   if(hypre_ParILUDataSchurSolver(ilu_data))
   {
      switch(hypre_ParILUDataIluType(ilu_data)){
      case 10: case 11: case 40: case 41: case 50:
         HYPRE_ParCSRGMRESDestroy(hypre_ParILUDataSchurSolver(ilu_data)); //GMRES for Schur
         break;
      case 20: case 21:
         hypre_NSHDestroy(hypre_ParILUDataSchurSolver(ilu_data));//NSH for Schur
         break;
      default:
         break;
      }
      hypre_ParILUDataSchurSolver(ilu_data) = NULL;
   }
   if(hypre_ParILUDataSchurPrecond(ilu_data))
   {
      switch(hypre_ParILUDataIluType(ilu_data)){
      case 10: case 11: case 40: case 41:
#ifdef HYPRE_USING_CUDA
         if(hypre_ParILUDataIluType(ilu_data) != 10 && hypre_ParILUDataIluType(ilu_data) != 11)
         {
#endif
            HYPRE_ILUDestroy(hypre_ParILUDataSchurPrecond(ilu_data)); //ILU as precond for Schur
#ifdef HYPRE_USING_CUDA
         }
#endif
         break;
      default:
         break;
        }
      hypre_ParILUDataSchurPrecond(ilu_data) = NULL;
   }
   /* CF marker array */
   if(hypre_ParILUDataCFMarkerArray(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataCFMarkerArray(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataCFMarkerArray(ilu_data) = NULL;
   }
   /* permutation array */
   if(hypre_ParILUDataPerm(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataPerm(ilu_data), HYPRE_MEMORY_DEVICE);
      hypre_ParILUDataPerm(ilu_data) = NULL;
   }
   if(hypre_ParILUDataQPerm(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataQPerm(ilu_data), HYPRE_MEMORY_DEVICE);
      hypre_ParILUDataQPerm(ilu_data) = NULL;
   }
   /* droptol array */
   if(hypre_ParILUDataDroptol(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataDroptol(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataDroptol(ilu_data) = NULL;
   }
   if(hypre_ParILUDataSchurPrecondIluDroptol(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataSchurPrecondIluDroptol(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = NULL;
   }
   if(hypre_ParILUDataSchurNSHDroptol(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataSchurNSHDroptol(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataSchurNSHDroptol(ilu_data) = NULL;
   }
   /* ilu data */
   hypre_TFree(ilu_data, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* set fill level (for ilu(k)) */
HYPRE_Int
hypre_ILUSetLevelOfFill( void *ilu_vdata, HYPRE_Int lfil )
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataLfil(ilu_data) = lfil;
   return hypre_error_flag;
}
/* set max non-zeros per row in factors (for ilut) */
HYPRE_Int
hypre_ILUSetMaxNnzPerRow( void *ilu_vdata, HYPRE_Int nzmax )
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataMaxRowNnz(ilu_data) = nzmax;
   return hypre_error_flag;
}
/* set threshold for dropping in LU factors (for ilut) */
HYPRE_Int
hypre_ILUSetDropThreshold( void *ilu_vdata, HYPRE_Real threshold )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   if(!(hypre_ParILUDataDroptol(ilu_data)))
   {
      hypre_ParILUDataDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 3, HYPRE_MEMORY_HOST);
   }
   hypre_ParILUDataDroptol(ilu_data)[0] = threshold;
   hypre_ParILUDataDroptol(ilu_data)[1] = threshold;
   hypre_ParILUDataDroptol(ilu_data)[2] = threshold;
   return hypre_error_flag;
}
/* set array of threshold for dropping in LU factors (for ilut) */
HYPRE_Int
hypre_ILUSetDropThresholdArray( void *ilu_vdata, HYPRE_Real *threshold )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   if(!(hypre_ParILUDataDroptol(ilu_data)))
   {
      hypre_ParILUDataDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 3, HYPRE_MEMORY_HOST);
   }
   hypre_TMemcpy( hypre_ParILUDataDroptol(ilu_data), threshold, HYPRE_Real, 3, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   return hypre_error_flag;
}
/* set ILU factorization type */
HYPRE_Int
hypre_ILUSetType( void *ilu_vdata, HYPRE_Int ilu_type )
{
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;

   /* destroy schur solver and/or preconditioner if already have one */
   if(hypre_ParILUDataSchurSolver(ilu_data))
   {
      switch(hypre_ParILUDataIluType(ilu_data)){
      case 10: case 11: case 40: case 41: case 50:
         HYPRE_ParCSRGMRESDestroy(hypre_ParILUDataSchurSolver(ilu_data)); //GMRES for Schur
         break;
      case 20: case 21:
         hypre_NSHDestroy(hypre_ParILUDataSchurSolver(ilu_data));//NSH for Schur
         break;
      default:
         break;
      }
      hypre_ParILUDataSchurSolver(ilu_data) = NULL;
   }
   if(hypre_ParILUDataSchurPrecond(ilu_data))
   {
      switch(hypre_ParILUDataIluType(ilu_data)){
      case 10: case 11: case 40: case 41:
#ifdef HYPRE_USING_CUDA
         if(hypre_ParILUDataIluType(ilu_data) != 10 && hypre_ParILUDataIluType(ilu_data) != 11)
         {
#endif
            HYPRE_ILUDestroy(hypre_ParILUDataSchurPrecond(ilu_data)); //ILU as precond for Schur
#ifdef HYPRE_USING_CUDA
         }
#endif
         break;
      default:
         break;
        }
      hypre_ParILUDataSchurPrecond(ilu_data) = NULL;
   }

   hypre_ParILUDataIluType(ilu_data) = ilu_type;
   /* reset default value, not a large cost
    * assume we won't change back from
    */
   switch(ilu_type)
   {
      /* NSH type */
      case 20: case 21:
      {
         /* only set value when user has not assiged value before */
         if(!(hypre_ParILUDataSchurNSHDroptol(ilu_data)))
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
         if(!(hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)))
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
/* Set max number of iterations for ILU solver */
HYPRE_Int
hypre_ILUSetMaxIter( void *ilu_vdata, HYPRE_Int max_iter )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataMaxIter(ilu_data) = max_iter;
   return hypre_error_flag;
}
/* Set convergence tolerance for ILU solver */
HYPRE_Int
hypre_ILUSetTol( void *ilu_vdata, HYPRE_Real tol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataTol(ilu_data) = tol;
   return hypre_error_flag;
}
/* Set print level for ilu solver */
HYPRE_Int
hypre_ILUSetPrintLevel( void *ilu_vdata, HYPRE_Int print_level )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataPrintLevel(ilu_data) = print_level;
   return hypre_error_flag;
}
/* Set print level for ilu solver */
HYPRE_Int
hypre_ILUSetLogging( void *ilu_vdata, HYPRE_Int logging )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataLogging(ilu_data) = logging;
   return hypre_error_flag;
}
/* Set type of reordering for local matrix */
HYPRE_Int
hypre_ILUSetLocalReordering( void *ilu_vdata, HYPRE_Int ordering_type )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataReorderingType(ilu_data) = ordering_type;
   return hypre_error_flag;
}

/* Set KDim (for GMRES) for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverKDIM( void *ilu_vdata, HYPRE_Int ss_kDim )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataSchurGMRESKDim(ilu_data) = ss_kDim;
   return hypre_error_flag;
}
/* Set max iteration for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverMaxIter( void *ilu_vdata, HYPRE_Int ss_max_iter )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   /* for the GMRES solve, the max iter is same as kdim by default */
   hypre_ParILUDataSchurGMRESKDim(ilu_data) = ss_max_iter;
   hypre_ParILUDataSchurGMRESMaxIter(ilu_data) = ss_max_iter;

   /* also set this value for NSH solve */
   hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data) = ss_max_iter;

   return hypre_error_flag;
}
/* Set convergence tolerance for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverTol( void *ilu_vdata, HYPRE_Real ss_tol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataSchurGMRESTol(ilu_data) = ss_tol;
   return hypre_error_flag;
}
/* Set absolute tolerance for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverAbsoluteTol( void *ilu_vdata, HYPRE_Real ss_absolute_tol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data) = ss_absolute_tol;
   return hypre_error_flag;
}
/* Set logging for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverLogging( void *ilu_vdata, HYPRE_Int ss_logging )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataSchurSolverLogging(ilu_data) = ss_logging;
   return hypre_error_flag;
}
/* Set print level for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverPrintLevel( void *ilu_vdata, HYPRE_Int ss_print_level )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataSchurSolverPrintLevel(ilu_data) = ss_print_level;
   return hypre_error_flag;
}
/* Set rel change (for GMRES) for Solver of Schur System */
HYPRE_Int
hypre_ILUSetSchurSolverRelChange( void *ilu_vdata, HYPRE_Int ss_rel_change )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataSchurGMRESRelChange(ilu_data) = ss_rel_change;
   return hypre_error_flag;
}
/* Set ILU type for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondILUType( void *ilu_vdata, HYPRE_Int sp_ilu_type )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataSchurPrecondIluType(ilu_data) = sp_ilu_type;
   return hypre_error_flag;
}
/* Set ILU level of fill for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondILULevelOfFill( void *ilu_vdata, HYPRE_Int sp_ilu_lfil )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataSchurPrecondIluLfil(ilu_data) = sp_ilu_lfil;
   return hypre_error_flag;
}
/* Set ILU max nonzeros per row for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondILUMaxNnzPerRow( void *ilu_vdata, HYPRE_Int sp_ilu_max_row_nnz )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data) = sp_ilu_max_row_nnz;
   return hypre_error_flag;
}
/* Set ILU drop threshold for ILUT for Precond of Schur System
 * We don't want to influence the original ILU, so create new array if not own data
 */
HYPRE_Int
hypre_ILUSetSchurPrecondILUDropThreshold( void *ilu_vdata, HYPRE_Real sp_ilu_droptol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   if(!(hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)))
   {
      hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 3, HYPRE_MEMORY_HOST);
   }
   hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[0]   = sp_ilu_droptol;
   hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[1]   = sp_ilu_droptol;
   hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)[2]   = sp_ilu_droptol;
   return hypre_error_flag;
}
/* Set array of ILU drop threshold for ILUT for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondILUDropThresholdArray( void *ilu_vdata, HYPRE_Real *sp_ilu_droptol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   if(!(hypre_ParILUDataSchurPrecondIluDroptol(ilu_data)))
   {
      hypre_ParILUDataSchurPrecondIluDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 3, HYPRE_MEMORY_HOST);
   }
   hypre_TMemcpy( hypre_ParILUDataSchurPrecondIluDroptol(ilu_data), sp_ilu_droptol, HYPRE_Real, 3, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   return hypre_error_flag;
}
/* Set print level for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondPrintLevel( void *ilu_vdata, HYPRE_Int sp_print_level )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataSchurPrecondPrintLevel(ilu_data) = sp_print_level;
   return hypre_error_flag;
}
/* Set max number of iterations for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondMaxIter( void *ilu_vdata, HYPRE_Int sp_max_iter )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataSchurPrecondMaxIter(ilu_data) = sp_max_iter;
   return hypre_error_flag;
}
/* Set onvergence tolerance for Precond of Schur System */
HYPRE_Int
hypre_ILUSetSchurPrecondTol( void *ilu_vdata, HYPRE_Int sp_tol )
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUDataSchurPrecondTol(ilu_data) = sp_tol;
   return hypre_error_flag;
}
/* Set tolorance for dropping in NSH for Schur System
 * We don't want to influence the original ILU, so create new array if not own data
 */
HYPRE_Int
hypre_ILUSetSchurNSHDropThreshold( void *ilu_vdata, HYPRE_Real threshold)
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   if(!(hypre_ParILUDataSchurNSHDroptol(ilu_data)))
   {
      hypre_ParILUDataSchurNSHDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 2, HYPRE_MEMORY_HOST);
   }
   hypre_ParILUDataSchurNSHDroptol(ilu_data)[0]           = threshold;
   hypre_ParILUDataSchurNSHDroptol(ilu_data)[1]           = threshold;
   return hypre_error_flag;
}
/* Set tolorance array for NSH for Schur System
 *    - threshold[0] : threshold for Minimal Residual iteration (initial guess for NSH).
 *    - threshold[1] : threshold for Newton–Schulz–Hotelling iteration.
*/
HYPRE_Int
hypre_ILUSetSchurNSHDropThresholdArray( void *ilu_vdata, HYPRE_Real *threshold)
{
   hypre_ParILUData   *ilu_data = (hypre_ParILUData*) ilu_vdata;
   if(!(hypre_ParILUDataSchurNSHDroptol(ilu_data)))
   {
      hypre_ParILUDataSchurNSHDroptol(ilu_data) = hypre_TAlloc(HYPRE_Real, 2, HYPRE_MEMORY_HOST);
   }
   hypre_TMemcpy( hypre_ParILUDataSchurNSHDroptol(ilu_data), threshold, HYPRE_Real, 2, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   return hypre_error_flag;
}

/* Get number of iterations for ILU solver */
HYPRE_Int
hypre_ILUGetNumIterations( void *ilu_vdata, HYPRE_Int *num_iterations )
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
/* Get residual norms for ILU solver */
HYPRE_Int
hypre_ILUGetFinalRelativeResidualNorm( void *ilu_vdata, HYPRE_Real *res_norm )
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
/*
 * Quicksort of the elements in a from low to high.
 * The elements in b are permuted according to the sorted a.
 * The elements in iw are permuted reverse according to the sorted a as it's index
 *   ie, iw[a1] and iw[a2] will be switched if a1 and a2 are switched
 * lo and hi are the extents of the region of the array a, that is to be sorted.
*/
/*
HYPRE_Int
hypre_quickSortIR (HYPRE_Int *a, HYPRE_Real *b, HYPRE_Int *iw, const HYPRE_Int lo, const HYPRE_Int hi)
{
   HYPRE_Int i=lo, j=hi;
   HYPRE_Int v;
   HYPRE_Int mid = (lo+hi)>>1;
   HYPRE_Int x=ceil(a[mid]);
   HYPRE_Real q;
   //  partition
   do
   {
      while (a[i]<x) i++;
      while (a[j]>x) j--;
      if (i<=j)
      {
          v=a[i]; a[i]=a[j]; a[j]=v;
          q=b[i]; b[i]=b[j]; b[j]=q;
          v=iw[a[i]];iw[a[i]]=iw[a[j]];iw[a[j]]=v;
          i++; j--;
      }
   } while (i<=j);
   //  recursion
   if (lo<j) hypre_quickSortIR(a, b, iw, lo, j);
   if (i<hi) hypre_quickSortIR(a, b, iw, i, hi);

   return hypre_error_flag;
}
*/
/* Print solver params */
HYPRE_Int
hypre_ILUWriteSolverParams(void *ilu_vdata)
{
   hypre_ParILUData  *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_printf("ILU Setup parameters: \n");
   hypre_printf("ILU factorization type: %d : ", hypre_ParILUDataIluType(ilu_data));
   switch(hypre_ParILUDataIluType(ilu_data)){
      case 0:
#ifdef HYPRE_USING_CUDA
            if( hypre_ParILUDataLfil(ilu_data) == 0 )
            {
               hypre_printf("Block Jacobi with GPU-accelerated ILU0 \n");
               hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParILUDataOperatorComplexity(ilu_data));
            }
            else
            {
#endif
               hypre_printf("Block Jacobi with ILU(%d) \n", hypre_ParILUDataLfil(ilu_data));
               hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParILUDataOperatorComplexity(ilu_data));
#ifdef HYPRE_USING_CUDA
            }
#endif
         break;
      case 1:
            hypre_printf("Block Jacobi with ILUT \n");
            hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n", hypre_ParILUDataDroptol(ilu_data)[0],hypre_ParILUDataDroptol(ilu_data)[1],hypre_ParILUDataDroptol(ilu_data)[2]);
            hypre_printf("Max nnz per row = %d \n", hypre_ParILUDataMaxRowNnz(ilu_data));
            hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 10:
#ifdef HYPRE_USING_CUDA
            if( hypre_ParILUDataLfil(ilu_data) == 0 )
            {
               hypre_printf("ILU-GMRES with GPU-accelerated ILU0 \n");
               hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParILUDataOperatorComplexity(ilu_data));
            }
            else
            {
#endif
               hypre_printf("ILU-GMRES with ILU(%d) \n", hypre_ParILUDataLfil(ilu_data));
               hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParILUDataOperatorComplexity(ilu_data));
#ifdef HYPRE_USING_CUDA
            }
#endif
         break;
      case 11:
         hypre_printf("ILU-GMRES with ILUT \n");
         hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n", hypre_ParILUDataDroptol(ilu_data)[0],hypre_ParILUDataDroptol(ilu_data)[1],hypre_ParILUDataDroptol(ilu_data)[2]);
         hypre_printf("Max nnz per row = %d \n", hypre_ParILUDataMaxRowNnz(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 20:
         hypre_printf("Newton-Schulz-Hotelling with ILU(%d) \n", hypre_ParILUDataLfil(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 21:
         hypre_printf("Newton-Schulz-Hotelling with ILUT \n");
         hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n", hypre_ParILUDataDroptol(ilu_data)[0],hypre_ParILUDataDroptol(ilu_data)[1],hypre_ParILUDataDroptol(ilu_data)[2]);
         hypre_printf("Max nnz per row = %d \n", hypre_ParILUDataMaxRowNnz(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 30:
         hypre_printf("RAS with ILU(%d) \n", hypre_ParILUDataLfil(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 31:
         hypre_printf("RAS with ILUT \n");
         hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n", hypre_ParILUDataDroptol(ilu_data)[0],hypre_ParILUDataDroptol(ilu_data)[1],hypre_ParILUDataDroptol(ilu_data)[2]);
         hypre_printf("Max nnz per row = %d \n", hypre_ParILUDataMaxRowNnz(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 40:
         hypre_printf("ddPQ-ILU-GMRES with ILU(%d) \n", hypre_ParILUDataLfil(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 41:
         hypre_printf("ddPQ-ILU-GMRES with ILUT \n");
         hypre_printf("drop tolerance for B = %e, E&F = %e, S = %e \n", hypre_ParILUDataDroptol(ilu_data)[0],hypre_ParILUDataDroptol(ilu_data)[1],hypre_ParILUDataDroptol(ilu_data)[2]);
         hypre_printf("Max nnz per row = %d \n", hypre_ParILUDataMaxRowNnz(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      case 50:
         hypre_printf("RAP-Modified-ILU with ILU(%d) \n", hypre_ParILUDataLfil(ilu_data));
         hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParILUDataOperatorComplexity(ilu_data));
         break;
      default: hypre_printf("Unknown type \n");
               break;
   }

   hypre_printf("\n ILU Solver Parameters: \n");
   hypre_printf("Max number of iterations: %d\n", hypre_ParILUDataMaxIter(ilu_data));
   hypre_printf("Stopping tolerance: %e\n", hypre_ParILUDataTol(ilu_data));

   return hypre_error_flag;
}

/* helper functions */
/*
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
 */
HYPRE_Int
hypre_ILUMinHeapAddI(HYPRE_Int *heap, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p;
   len--;/* now len is the current index */
   while(len > 0)
   {
      /* get the parent index */
      p = (len-1)/2;
      if(heap[p] > heap[len])
      {
         /* this is smaller */
         hypre_swap(heap,p,len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/* see hypre_ILUMinHeapAddI for detail instructions */
HYPRE_Int
hypre_ILUMinHeapAddIIIi(HYPRE_Int *heap, HYPRE_Int *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p;
   len--;/* now len is the current index */
   while(len > 0)
   {
      /* get the parent index */
      p = (len-1)/2;
      if(heap[p] > heap[len])
      {
         /* this is smaller */
         hypre_swap(Ii1,heap[p],heap[len]);
         hypre_swap2i(heap,I1,p,len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/* see hypre_ILUMinHeapAddI for detail instructions */
HYPRE_Int
hypre_ILUMinHeapAddIRIi(HYPRE_Int *heap, HYPRE_Real *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p;
   len--;/* now len is the current index */
   while(len > 0)
   {
      /* get the parent index */
      p = (len-1)/2;
      if(heap[p] > heap[len])
      {
         /* this is smaller */
         hypre_swap(Ii1,heap[p],heap[len]);
         hypre_swap2(heap,I1,p,len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/* see hypre_ILUMinHeapAddI for detail instructions */
HYPRE_Int
hypre_ILUMaxHeapAddRabsIIi(HYPRE_Real *heap, HYPRE_Int *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p;
   len--;/* now len is the current index */
   while(len > 0)
   {
      /* get the parent index */
      p = (len-1)/2;
      if(hypre_abs(heap[p]) < hypre_abs(heap[len]))
      {
         /* this is smaller */
         hypre_swap(Ii1,heap[p],heap[len]);
         hypre_swap2(I1,heap,p,len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/* see hypre_ILUMinHeapAddI for detail instructions */
HYPRE_Int
hypre_ILUMaxrHeapAddRabsI(HYPRE_Real *heap, HYPRE_Int *I1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p;
   len--;/* now len is the current index */
   while(len > 0)
   {
      /* get the parent index */
      p = (len-1)/2;
      if(hypre_abs(heap[-p]) < hypre_abs(heap[-len]))
      {
         /* this is smaller */
         hypre_swap2(I1,heap,-p,-len);
         len = p;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/*
 * Swap the first element with the last element of the heap,
 *    reduce size by one, and maintain the heap structure
 * I means HYPRE_Int
 * R means HYPRE_Real
 * max/min heap
 * r means heap goes from 0 to -1, -2 instead of 0 1 2
 * Ii and Ri means orderd by value of heap, like iw for ILU
 * heap: aray of that heap
 * len: current length of the heap
 * WARNING: Remember to change the len youself
 */
HYPRE_Int
hypre_ILUMinHeapRemoveI(HYPRE_Int *heap, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p,l,r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   hypre_swap(heap,0,len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while(l < len)
   {
      r = 2*p+2;
      /* two childs, pick the smaller one */
      l = r >= len || heap[l]<heap[r] ? l : r;
      if(heap[l]<heap[p])
      {
         hypre_swap(heap,l,p);
         p = l;
         l = 2*p+1;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/* see hypre_ILUMinHeapRemoveI for detail instructions */
HYPRE_Int
hypre_ILUMinHeapRemoveIIIi(HYPRE_Int *heap, HYPRE_Int *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p,l,r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   hypre_swap(Ii1,heap[0],heap[len]);
   hypre_swap2i(heap,I1,0,len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while(l < len)
   {
      r = 2*p+2;
      /* two childs, pick the smaller one */
      l = r >= len || heap[l]<heap[r] ? l : r;
      if(heap[l]<heap[p])
      {
         hypre_swap(Ii1,heap[p],heap[l]);
         hypre_swap2i(heap,I1,l,p);
         p = l;
         l = 2*p+1;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/* see hypre_ILUMinHeapRemoveI for detail instructions */
HYPRE_Int
hypre_ILUMinHeapRemoveIRIi(HYPRE_Int *heap, HYPRE_Real *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p,l,r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   hypre_swap(Ii1,heap[0],heap[len]);
   hypre_swap2(heap,I1,0,len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while(l < len)
   {
      r = 2*p+2;
      /* two childs, pick the smaller one */
      l = r >= len || heap[l]<heap[r] ? l : r;
      if(heap[l]<heap[p])
      {
         hypre_swap(Ii1,heap[p],heap[l]);
         hypre_swap2(heap,I1,l,p);
         p = l;
         l = 2*p+1;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/* see hypre_ILUMinHeapRemoveI for detail instructions */
HYPRE_Int
hypre_ILUMaxHeapRemoveRabsIIi(HYPRE_Real *heap, HYPRE_Int *I1, HYPRE_Int *Ii1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p,l,r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   hypre_swap(Ii1,heap[0],heap[len]);
   hypre_swap2(I1,heap,0,len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while(l < len)
   {
      r = 2*p+2;
      /* two childs, pick the smaller one */
      l = r >= len || hypre_abs(heap[l])>hypre_abs(heap[r]) ? l : r;
      if(hypre_abs(heap[l])>hypre_abs(heap[p]))
      {
         hypre_swap(Ii1,heap[p],heap[l]);
         hypre_swap2(I1,heap,l,p);
         p = l;
         l = 2*p+1;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/* see hypre_ILUMinHeapRemoveI for detail instructions */
HYPRE_Int
hypre_ILUMaxrHeapRemoveRabsI(HYPRE_Real *heap, HYPRE_Int *I1, HYPRE_Int len)
{
   /* parent, left, right */
   HYPRE_Int p,l,r;
   len--;/* now len is the max index */
   /* swap the first element to last */
   hypre_swap2(I1,heap,0,-len);
   p = 0;
   l = 1;
   /* while I'm still in the heap */
   while(l < len)
   {
      r = 2*p+2;
      /* two childs, pick the smaller one */
      l = r >= len || hypre_abs(heap[-l])>hypre_abs(heap[-r]) ? l : r;
      if(hypre_abs(heap[-l])>hypre_abs(heap[-p]))
      {
         hypre_swap2(I1,heap,-l,-p);
         p = l;
         l = 2*p+1;
      }
      else
      {
         break;
      }
   }
   return hypre_error_flag;
}

/* Split based on quick sort algorithm (avoid sorting the entire array)
 * find the largest k elements out of original array
 * array: input array for compare
 * I: integer array bind with array
 * k: largest k elements
 * len: length of the array
 */
HYPRE_Int
hypre_ILUMaxQSplitRabsI(HYPRE_Real *array, HYPRE_Int *I, HYPRE_Int left, HYPRE_Int bound, HYPRE_Int right)
{
   HYPRE_Int i, last;
   if (left >= right)
   {
      return hypre_error_flag;
   }
   hypre_swap2(I,array,left,(left+right)/2);
   last = left;
   for(i = left + 1 ; i <= right ; i ++)
   {
      if(hypre_abs(array[i]) > hypre_abs(array[left]))
      {
         hypre_swap2(I,array,++last,i);
      }
   }
   hypre_swap2(I,array,left,last);
   hypre_ILUMaxQSplitRabsI(array,I,left,bound,last-1);
   if(bound > last)
   {
       hypre_ILUMaxQSplitRabsI(array,I,last+1,bound,right);
   }

   return hypre_error_flag;
}

/* Helper function to search max value from a row
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
 */
HYPRE_Int
hypre_ILUMaxRabs(HYPRE_Real *array_data, HYPRE_Int *array_j, HYPRE_Int start, HYPRE_Int end, HYPRE_Int nLU, HYPRE_Int *rperm, HYPRE_Real *value, HYPRE_Int *index, HYPRE_Real *l1_norm, HYPRE_Int *nnz)
{
   HYPRE_Int i, idx, col;
   HYPRE_Real val, max_value, norm, nz;

   nz = 0;
   norm = 0.0;
   max_value = -1.0;
   idx = -1;
   if(rperm)
   {
      /* apply rperm and nLU */
      for(i = start ; i < end ; i ++)
      {
         col = rperm[array_j[i]];
         if(col > nLU)
         {
            /* this old column is in new external part */
            continue;
         }
         nz ++;
         val = hypre_abs(array_data[i]);
         norm += val;
         if(max_value < val)
         {
            max_value = val;
            idx = i;
         }
      }
   }
   else
   {
      /* basic search */
      for(i = start ; i < end ; i ++)
      {
         val = hypre_abs(array_data[i]);
         norm += val;
         if(max_value < val)
         {
            max_value = val;
            idx = i;
         }
      }
      nz = end - start;
   }

   *value = max_value;
   if(index)
   {
      *index = idx;
   }
   if(l1_norm)
   {
      *l1_norm = norm;
   }
   if(nnz)
   {
      *nnz = nz;
   }

   return hypre_error_flag;
}

/* Pre selection for ddPQ, this is the basic version considering row sparsity
 * n: size of matrix
 * nLU: size we consider ddPQ reorder, only first nLU*nLU block is considered
 * A_diag_i/j/data: information of A
 * tol: tol for ddPQ, normally between 0.1-0.3
 * *perm: current row order
 * *rperm: current column order
 * *pperm_pre: output ddPQ pre row roder
 * *qperm_pre: output ddPQ pre column order
 */
HYPRE_Int
hypre_ILUGetPermddPQPre(HYPRE_Int n, HYPRE_Int nLU, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j, HYPRE_Real *A_diag_data, HYPRE_Real tol, HYPRE_Int *perm, HYPRE_Int *rperm,
                        HYPRE_Int *pperm_pre, HYPRE_Int *qperm_pre, HYPRE_Int *nB)
{
   HYPRE_Int   i, ii, nB_pre, k1, k2;
   HYPRE_Real  gtol, max_value, norm;

   HYPRE_Int   *jcol, *jnnz;
   HYPRE_Real  *weight;

   weight      = hypre_TAlloc(HYPRE_Real, nLU + 1, HYPRE_MEMORY_HOST);
   jcol        = hypre_TAlloc(HYPRE_Int, nLU + 1, HYPRE_MEMORY_HOST);
   jnnz        = hypre_TAlloc(HYPRE_Int, nLU + 1, HYPRE_MEMORY_HOST);

   max_value   = -1.0;
   /* first need to build gtol */
   for( ii = 0 ; ii < nLU ; ii ++)
   {
      /* find real row */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      /* find max|a| of that row and its index */
      hypre_ILUMaxRabs(A_diag_data, A_diag_j, k1, k2, nLU, rperm, weight + ii, jcol + ii, &norm, jnnz + ii);
      weight[ii] /= norm;
      if(weight[ii] > max_value)
      {
         max_value = weight[ii];
      }
   }

   gtol = tol * max_value;

   /* second loop to pre select B */
   nB_pre = 0;
   for( ii = 0 ; ii < nLU ; ii ++)
   {
      /* keep this row */
      if(weight[ii] > gtol)
      {
         weight[nB_pre] /= (HYPRE_Real)(jnnz[ii]);
         pperm_pre[nB_pre] = perm[ii];
         qperm_pre[nB_pre++] = A_diag_j[jcol[ii]];
      }
   }

   *nB = nB_pre;

   /* sort from small to large */
   hypre_qsort3(weight, pperm_pre, qperm_pre, 0, nB_pre-1);

   hypre_TFree(weight, HYPRE_MEMORY_HOST);
   hypre_TFree(jcol, HYPRE_MEMORY_HOST);
   hypre_TFree(jnnz, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/* Get ddPQ version perm array for ParCSR
 * Greedy matching selection
 * ddPQ is a two-side permutation for diagonal dominance
 * A: the input matrix
 * pperm: row permutation
 * qperm: col permutation
 * nB: the size of B block
 * nI: number of interial nodes
 * tol: the dropping tolorance for ddPQ
 * reordering_type: Type of reordering for the interior nodes.
 * Currently only supports RCM reordering. Set to 0 for no reordering.
 */

HYPRE_Int
hypre_ILUGetPermddPQ(hypre_ParCSRMatrix *A, HYPRE_Int **io_pperm, HYPRE_Int **io_qperm, HYPRE_Real tol, HYPRE_Int *nB, HYPRE_Int *nI, HYPRE_Int reordering_type)
{
   HYPRE_Int         i, nB_pre, irow, jcol, nLU;
   HYPRE_Int         *pperm, *qperm;
   HYPRE_Int         *rpperm, *rqperm, *pperm_pre, *qperm_pre;

   /* data objects for A */
   hypre_CSRMatrix   *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int         *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int         *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real        *A_diag_data = hypre_CSRMatrixData(A_diag);

   /* problem size */
   HYPRE_Int         n = hypre_CSRMatrixNumRows(A_diag);

   /* 1: Setup and create memory
    */

   pperm             = NULL;
   qperm             = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
   rpperm            = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   rqperm            = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);

   /* 2: Find interior nodes first
    */
   hypre_ILUGetInteriorExteriorPerm( A, &pperm, &nLU, 0);
   *nI = nLU;

   /* 3: Pre selection on interial nodes
    * this pre selection puts external nodes to the last
    * also provide candidate rows for B block
    */

   /* build reverse permutation array
    * rperm[old] = new
    */
   for(i = 0 ; i < n ; i ++)
   {
      rpperm[pperm[i]] = i;
   }

   /* build place holder for pre selection pairs */
   pperm_pre = hypre_TAlloc(HYPRE_Int, nLU, HYPRE_MEMORY_HOST);
   qperm_pre = hypre_TAlloc(HYPRE_Int, nLU, HYPRE_MEMORY_HOST);

   /* pre selection */
   hypre_ILUGetPermddPQPre(n, nLU, A_diag_i, A_diag_j, A_diag_data, tol, pperm, rpperm, pperm_pre, qperm_pre, &nB_pre);

   /* 4: Build B block
    * Greedy selection
    */

   /* rperm[old] = new */
   for(i = 0 ; i < nLU ; i ++)
   {
      rpperm[pperm[i]] = -1;
   }

   hypre_TMemcpy( rqperm, rpperm, HYPRE_Int, n, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
   hypre_TMemcpy( qperm, pperm, HYPRE_Int, n, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);

   /* we sort from small to large, so we need to go from back to start
    * we only need nB_pre to start the loop, after that we could use it for size of B
    */
   for(i = nB_pre-1, nB_pre = 0 ; i >=0 ; i --)
   {
      irow = pperm_pre[i];
      jcol = qperm_pre[i];

      /* this col is not yet taken */
      if(rqperm[jcol] < 0)
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
   for(i = 0 ; i < n ; i ++)
   {
      if(rpperm[i] < 0)
      {
         pperm[nB_pre++] = i;
      }
   }
   nB_pre = nLU;
   for(i = 0 ; i < n ; i ++)
   {
      if(rqperm[i] < 0)
      {
         qperm[nB_pre++] = i;
      }
   }

   /* Finishing up and free
    */

   switch(reordering_type)
   {
      case 0:
         /* no RCM in this case */
         break;
      case 1:
         /* RCM */
         hypre_ILULocalRCM( hypre_ParCSRMatrixDiag(A), 0, nLU, &pperm, &qperm, 0);
         break;
      default:
         /* RCM */
         hypre_ILULocalRCM( hypre_ParCSRMatrixDiag(A), 0, nLU, &pperm, &qperm, 0);
         break;
   }

   *nB = nLU;
   *io_pperm = pperm;
   *io_qperm = qperm;

   hypre_TFree( rpperm, HYPRE_MEMORY_HOST);
   hypre_TFree( rqperm, HYPRE_MEMORY_HOST);
   hypre_TFree( pperm_pre, HYPRE_MEMORY_HOST);
   hypre_TFree( qperm_pre, HYPRE_MEMORY_HOST);
   return hypre_error_flag;
}

/*
 * Get perm array from parcsr matrix based on diag and offdiag matrix
 * Just simply loop through the rows of offd of A, check for nonzero rows
 * Put interior nodes at the beginning
 * A: parcsr matrix
 * perm: permutation array
 * nLU: number of interial nodes
 * reordering_type: Type of (additional) reordering for the interior nodes.
 * Currently only supports RCM reordering. Set to 0 for no reordering.
 */
HYPRE_Int
hypre_ILUGetInteriorExteriorPerm(hypre_ParCSRMatrix *A, HYPRE_Int **perm, HYPRE_Int *nLU, HYPRE_Int reordering_type)
{
   /* get basic information of A */
   HYPRE_Int            n = hypre_ParCSRMatrixNumRows(A);
   HYPRE_Int            i, j, first, last, start, end;
   HYPRE_Int            num_sends, send_map_start, send_map_end, col;
   hypre_CSRMatrix      *A_offd;
   HYPRE_Int            *A_offd_i;
   A_offd               = hypre_ParCSRMatrixOffd(A);
   A_offd_i             = hypre_CSRMatrixI(A_offd);
   first                = 0;
   last                 = n - 1;
   HYPRE_Int            *temp_perm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
   HYPRE_Int            *marker = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);

   /* first get col nonzero from com_pkg */
   /* get comm_pkg, craete one if we not yet have one */
   hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* now directly take adavantage of comm_pkg */
   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   for( i = 0 ; i < num_sends ; i ++ )
   {
      send_map_start = hypre_ParCSRCommPkgSendMapStart(comm_pkg,i);
      send_map_end = hypre_ParCSRCommPkgSendMapStart(comm_pkg,i+1);
      for ( j = send_map_start ; j < send_map_end ; j ++)
      {
         col = hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j);
         if(marker[col] == 0)
         {
            temp_perm[last--] = col;
            marker[col] = -1;
         }
      }
   }

   /* now deal with the row */
   for( i = 0 ; i < n ; i ++)
   {
      if(marker[i] == 0)
      {
         start = A_offd_i[i];
         end = A_offd_i[i+1];
         if(start == end)
         {
            temp_perm[first++] = i;
         }
         else
         {
            temp_perm[last--] = i;
         }
      }
   }
   switch(reordering_type)
   {
      case 0:
         /* no RCM in this case */
         break;
      case 1:
         /* RCM */
         hypre_ILULocalRCM( hypre_ParCSRMatrixDiag(A), 0, first, &temp_perm, &temp_perm, 1);
         break;
      default:
         /* RCM */
         hypre_ILULocalRCM( hypre_ParCSRMatrixDiag(A), 0, first, &temp_perm, &temp_perm, 1);
         break;
   }

   /* set out values */
   *nLU = first;
   if((*perm) != NULL) hypre_TFree(*perm,HYPRE_MEMORY_DEVICE);
   *perm = temp_perm;

   hypre_TFree(marker, HYPRE_MEMORY_HOST);
   return hypre_error_flag;
}

/*
 * Get the (local) ordering of the diag (local) matrix (no permutation). This is the permutation used for the block-jacobi case
 * A: parcsr matrix
 * perm: permutation array
 * nLU: number of interior nodes
 * reordering_type: Type of (additional) reordering for the nodes.
 * Currently only supports RCM reordering. Set to 0 for no reordering.
 */
HYPRE_Int
hypre_ILUGetLocalPerm(hypre_ParCSRMatrix *A, HYPRE_Int **perm, HYPRE_Int *nLU, HYPRE_Int reordering_type)
{
   /* get basic information of A */
   HYPRE_Int            n = hypre_ParCSRMatrixNumRows(A);
   HYPRE_Int            i;
   HYPRE_Int            *temp_perm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);

   /* set perm array */
   for( i = 0 ; i < n ; i ++ )
   {
      temp_perm[i] = i;
   }
   switch(reordering_type)
   {
      case 0:
         /* no RCM in this case */
         break;
      case 1:
         /* RCM */
         hypre_ILULocalRCM( hypre_ParCSRMatrixDiag(A), 0, n, &temp_perm, &temp_perm, 1);
         break;
      default:
         /* RCM */
         hypre_ILULocalRCM( hypre_ParCSRMatrixDiag(A), 0, n, &temp_perm, &temp_perm, 1);
         break;
   }
   *nLU = n;
   if((*perm) != NULL) hypre_TFree(*perm,HYPRE_MEMORY_DEVICE);
   *perm = temp_perm;

   return hypre_error_flag;
}

#if 0
/* Build the expanded matrix for RAS-1
 * A: input ParCSR matrix
 * E_i, E_j, E_data: information for external matrix
 * rperm: reverse permutation to build real index, rperm[old] = new
 *
 * NOTE: Modified to avoid communicating BigInt arrays - DOK
 */
HYPRE_Int
hypre_ILUBuildRASExternalMatrix(hypre_ParCSRMatrix *A, HYPRE_Int *rperm, HYPRE_Int **E_i, HYPRE_Int **E_j, HYPRE_Real **E_data)
{
   HYPRE_Int                i, i1, i2, j, jj, k, row, k1, k2, k3, lend, leno, col, l1, l2;
   HYPRE_BigInt    big_col;

   /* data objects for communication */
   MPI_Comm                 comm = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg      *comm_pkg;
   hypre_ParCSRCommPkg      *comm_pkg_tmp;
   hypre_ParCSRCommHandle   *comm_handle_count;
   hypre_ParCSRCommHandle   *comm_handle_marker;
   hypre_ParCSRCommHandle   *comm_handle_j;
   hypre_ParCSRCommHandle   *comm_handle_data;
   HYPRE_BigInt                *col_starts;
   HYPRE_Int                total_rows;
   HYPRE_Int                num_sends;
   HYPRE_Int                num_recvs;
   HYPRE_Int                begin, end;
   HYPRE_Int                my_id,num_procs,proc_id;

   /* data objects for buffers in communication */
   HYPRE_Int                *send_map;
   HYPRE_Int                *send_count = NULL,*send_disp = NULL;
   HYPRE_Int                *send_count_offd = NULL;
   HYPRE_Int                *recv_count = NULL,*recv_disp = NULL,*recv_marker = NULL;
   HYPRE_Int                *send_buf_int = NULL;
   HYPRE_Int                *recv_buf_int = NULL;
   HYPRE_Real               *send_buf_real = NULL, *recv_buf_real = NULL;
   HYPRE_Int                *send_disp_comm = NULL, *recv_disp_comm = NULL;

   /* data objects for A */
   hypre_CSRMatrix          *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix          *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_BigInt             *A_col_starts = hypre_ParCSRMatrixColStarts(A);
   HYPRE_BigInt             *A_offd_colmap = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Real               *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int                *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int                *A_diag_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int                *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int                *A_offd_j = hypre_CSRMatrixJ(A_offd);
   HYPRE_Real               *A_offd_data = hypre_CSRMatrixData(A_offd);

   /* size */
   HYPRE_Int                n = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int                m = hypre_CSRMatrixNumCols(A_offd);

   /* 1: setup part
    * allocate memory and setup working array
    */

   /* MPI stuff */
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* now check communication package */
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   /* create if not yet built */
   if(!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* get communication information */
   send_map          = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
   num_sends         = hypre_ParCSRCommPkgNumSends(comm_pkg);
   send_disp_comm    = hypre_TAlloc(HYPRE_Int, num_sends + 1, HYPRE_MEMORY_HOST);
   begin             = hypre_ParCSRCommPkgSendMapStart(comm_pkg,0);
   end               = hypre_ParCSRCommPkgSendMapStart(comm_pkg,num_sends);
   total_rows        = end - begin;
   num_recvs         = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
   recv_disp_comm    = hypre_TAlloc(HYPRE_Int, num_recvs + 1, HYPRE_MEMORY_HOST);

   /* create buffers */
   send_count        = hypre_TAlloc(HYPRE_Int, total_rows, HYPRE_MEMORY_HOST);
   send_disp         = hypre_TAlloc(HYPRE_Int, total_rows + 1, HYPRE_MEMORY_HOST);
   send_count_offd   = hypre_CTAlloc(HYPRE_Int, total_rows, HYPRE_MEMORY_HOST);
   recv_count        = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_HOST);
   recv_marker       = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_HOST);
   recv_disp         = hypre_TAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_HOST);

   /* 2: communication part 1 to get amount of send and recv */

   /* first we need to know the global start */
   col_starts        = hypre_TAlloc(HYPRE_BigInt, num_procs + 1, HYPRE_MEMORY_HOST);
   hypre_MPI_Allgather(A_col_starts+1,1,HYPRE_MPI_BIG_INT,col_starts+1,1,HYPRE_MPI_BIG_INT,comm);
   col_starts[0]     = 0;

   send_disp[0]      = 0;
   send_disp_comm[0] = 0;
   /* now loop to know how many to send per row */
   for( i = 0 ; i < num_sends ; i ++ )
   {
      /* update disp for comm package */
      send_disp_comm[i+1] = send_disp_comm[i];
      /* get the proc we are sending to */
      proc_id = hypre_ParCSRCommPkgSendProc(comm_pkg,i);
      /* set start end of this proc */
      l1 = hypre_ParCSRCommPkgSendMapStart(comm_pkg,i);
      l2 = hypre_ParCSRCommPkgSendMapStart(comm_pkg,i + 1);
      /* loop through rows we need to send */
      for( j = l1 ; j < l2 ; j ++ )
      {
         /* reset length */
         leno = lend = 0;
         /* we need to send out this row */
         row = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);

         /* check how many we need to send from diagonal first */
         k1 = A_diag_i[row], k2 = A_diag_i[row+1];
         for( k = k1 ; k < k2 ; k ++ )
         {
            col = A_diag_j[k];
            if(hypre_BinarySearch(send_map+l1,col,l2-l1) >=0 )
            {
               lend++;
            }
         }

         /* check how many we need to send from offdiagonal */
         k1 = A_offd_i[row], k2 = A_offd_i[row+1];
         for( k = k1 ; k < k2 ; k ++ )
         {
            /* get real column number of this offdiagonal column */
            big_col = A_offd_colmap[A_offd_j[k]];
            if(big_col >= col_starts[proc_id] && big_col < col_starts[proc_id+1])
            {
               /* this column is in diagonal range of proc_id
                * everything in diagonal range need to be in the factorization
                */
               leno++;
            }
         }
         send_count_offd[j]   = leno;
         send_count[j]        = leno + lend;
         send_disp[j+1]       = send_disp[j] + send_count[j];
         send_disp_comm[i+1] += send_count[j];
      }
   }

   /* 3: new communication to know how many we need to receive for each external row
    * main communication, 11 is integer
    */
   comm_handle_count    = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_count, recv_count);
   comm_handle_marker   = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_count_offd, recv_marker);
   hypre_ParCSRCommHandleDestroy(comm_handle_count);
   hypre_ParCSRCommHandleDestroy(comm_handle_marker);

   recv_disp[0] = 0;
   recv_disp_comm[0] = 0;
   /* now build the recv disp array */
   for(i = 0 ; i < num_recvs ; i ++)
   {
      recv_disp_comm[i+1] = recv_disp_comm[i];
      k1 = hypre_ParCSRCommPkgRecvVecStart( comm_pkg, i );
      k2 = hypre_ParCSRCommPkgRecvVecStart( comm_pkg, i + 1 );
      for(j = k1 ; j < k2 ; j ++)
      {
         recv_disp[j+1] = recv_disp[j] + recv_count[j];
         recv_disp_comm[i+1] += recv_count[j];
      }
   }

   /* 4: ready to start real communication
    * now we know how many we need to send out, create send/recv buffers
    */
   send_buf_int   = hypre_TAlloc(HYPRE_Int, send_disp[total_rows], HYPRE_MEMORY_HOST);
   send_buf_real  = hypre_TAlloc(HYPRE_Real, send_disp[total_rows], HYPRE_MEMORY_HOST);
   recv_buf_int   = hypre_TAlloc(HYPRE_Int, recv_disp[m], HYPRE_MEMORY_HOST);
   recv_buf_real  = hypre_TAlloc(HYPRE_Real, recv_disp[m], HYPRE_MEMORY_HOST);

   /* fill send buffer */
   for( i = 0 ; i < num_sends ; i ++ )
   {
      /* get the proc we are sending to */
      proc_id = hypre_ParCSRCommPkgSendProc(comm_pkg,i);
      /* set start end of this proc */
      l1 = hypre_ParCSRCommPkgSendMapStart(comm_pkg,i);
      l2 = hypre_ParCSRCommPkgSendMapStart(comm_pkg,i + 1);
      /* loop through rows we need to apply communication */
      for( j = l1 ; j < l2 ; j ++ )
      {
         /* reset length
          * one remark here, the diagonal we send becomes
          *    off diagonal part for reciver
          */
         leno = send_disp[j];
         lend = leno + send_count_offd[j];
         /* we need to send out this row */
         row = hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j);

         /* fill diagonal first */
         k1 = A_diag_i[row], k2 = A_diag_i[row+1];
         for( k = k1 ; k < k2 ; k ++ )
         {
            col = A_diag_j[k];
            if(hypre_BinarySearch(send_map+l1,col,l2-l1) >=0)
            {
               send_buf_real[lend] = A_diag_data[k];
               /* the diag part becomes offd for recv part, so update index
                * set up to global index
                * set it to be negative
                */
               send_buf_int[lend++] = col;// + col_starts[my_id];
            }
         }

         /* fill offdiagonal */
         k1 = A_offd_i[row], k2 = A_offd_i[row+1];
         for( k = k1 ; k < k2 ; k ++ )
         {
            /* get real column number of this offdiagonal column */
            big_col = A_offd_colmap[A_offd_j[k]];
            if(big_col >= col_starts[proc_id] && big_col < col_starts[proc_id+1])
            {
               /* this column is in diagonal range of proc_id
                * everything in diagonal range need to be in the factorization
                */
               send_buf_real[leno] = A_offd_data[k];
               /* the offd part becomes diagonal for recv part, so update index */
               send_buf_int[leno++] = (HYPRE_Int)(big_col - col_starts[proc_id]);
            }
         }
      }
   }

   /* now build new comm_pkg for this communication */
   comm_pkg_tmp = hypre_CTAlloc(hypre_ParCSRCommPkg, 1, HYPRE_MEMORY_HOST);
   hypre_ParCSRCommPkgComm         (comm_pkg_tmp) = comm;
   hypre_ParCSRCommPkgNumSends     (comm_pkg_tmp) = num_sends;
   hypre_ParCSRCommPkgSendProcs    (comm_pkg_tmp) = hypre_ParCSRCommPkgSendProcs(comm_pkg);
   hypre_ParCSRCommPkgSendMapStarts(comm_pkg_tmp) = send_disp_comm;
   hypre_ParCSRCommPkgNumRecvs     (comm_pkg_tmp) = num_recvs;
   hypre_ParCSRCommPkgRecvProcs    (comm_pkg_tmp) = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
   hypre_ParCSRCommPkgRecvVecStarts(comm_pkg_tmp) = recv_disp_comm;

   /* communication */
   comm_handle_j = hypre_ParCSRCommHandleCreate(11, comm_pkg_tmp, send_buf_int, recv_buf_int);
   comm_handle_data = hypre_ParCSRCommHandleCreate(1, comm_pkg_tmp, send_buf_real, recv_buf_real);
   hypre_ParCSRCommHandleDestroy(comm_handle_j);
   hypre_ParCSRCommHandleDestroy(comm_handle_data);

   /* Update the index to be real index */
   /* Dealing with diagonal part */
   for(i = 0 ; i < m ; i++ )
   {
      k1 = recv_disp[i];
      k2 = recv_disp[i] + recv_marker[i];
      k3 = recv_disp[i+1];
      for(j = k1 ; j < k2 ; j ++ )
      {
         recv_buf_int[j] = rperm[recv_buf_int[j]];
      }
   }

   /* Dealing with off-diagonal part */
   for(i = 0 ; i < num_recvs ; i ++)
   {
      proc_id = hypre_ParCSRCommPkgRecvProc( comm_pkg_tmp, i);
      i1 = hypre_ParCSRCommPkgRecvVecStart( comm_pkg_tmp, i );
      i2 = hypre_ParCSRCommPkgRecvVecStart( comm_pkg_tmp, i + 1 );
      for(j = i1 ; j < i2 ; j++)
      {
         k1 = recv_disp[j] + recv_marker[j];
         k2 = recv_disp[j+1];

         for(jj = k1 ; jj < k2 ; jj++)
         {
            /* Correct index to get actual global index */
            big_col = recv_buf_int[jj] + col_starts[proc_id];
            recv_buf_int[jj] = hypre_BigBinarySearch( A_offd_colmap, big_col, m) + n;
         }
      }
   }

   /* Assign data */
   *E_i     = recv_disp;
   *E_j     = recv_buf_int;
   *E_data  = recv_buf_real;

   /* 5: finish and free
    */

   hypre_TFree(send_disp_comm, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_disp_comm, HYPRE_MEMORY_HOST);
   hypre_TFree(comm_pkg_tmp, HYPRE_MEMORY_HOST);
   hypre_TFree(col_starts, HYPRE_MEMORY_HOST);
   hypre_TFree(send_count, HYPRE_MEMORY_HOST);
   hypre_TFree(send_disp, HYPRE_MEMORY_HOST);
   hypre_TFree(send_count_offd, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_count, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buf_int, HYPRE_MEMORY_HOST);
   hypre_TFree(send_buf_real, HYPRE_MEMORY_HOST);
   hypre_TFree(recv_marker, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
#else
/* Build the expanded matrix for RAS-1
 * A: input ParCSR matrix
 * E_i, E_j, E_data: information for external matrix
 * rperm: reverse permutation to build real index, rperm[old] = new
 */
HYPRE_Int
hypre_ILUBuildRASExternalMatrix(hypre_ParCSRMatrix *A, HYPRE_Int *rperm, HYPRE_Int **E_i, HYPRE_Int **E_j, HYPRE_Real **E_data)
{
   HYPRE_Int                i, j, idx;
   HYPRE_BigInt   big_col;

   /* data objects for communication */
   MPI_Comm                 comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int                my_id;

   /* data objects for A */
   hypre_CSRMatrix          *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix          *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_BigInt   *A_col_starts = hypre_ParCSRMatrixColStarts(A);
   HYPRE_BigInt   *A_offd_colmap = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int                *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int                *A_offd_i = hypre_CSRMatrixI(A_offd);

   /* data objects for external A matrix */
   // Need to check the new version of hypre_ParcsrGetExternalRows
   hypre_CSRMatrix          *A_ext = NULL;
   // # up to local offd cols, no need to be HYPRE_BigInt
   HYPRE_Int                *A_ext_i = NULL;
   // Return global index, HYPRE_BigInt required
   HYPRE_BigInt   *A_ext_j = NULL;
   HYPRE_Real               *A_ext_data = NULL;

   /* data objects for output */
   HYPRE_Int                E_nnz;
   HYPRE_Int                *E_ext_i = NULL;
   // Local index, no need to use HYPRE_BigInt
   HYPRE_Int                *E_ext_j = NULL;
   HYPRE_Real               *E_ext_data = NULL;

   //guess non-zeros for E before start
   HYPRE_Int                E_init_alloc;

   /* size */
   HYPRE_Int                n = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int                m = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int                A_diag_nnz = A_diag_i[n];
   HYPRE_Int                A_offd_nnz = A_offd_i[n];

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
   E_init_alloc =  hypre_max( (HYPRE_Int) ( A_diag_nnz / (HYPRE_Real) n / (HYPRE_Real) n * (HYPRE_Real) m * (HYPRE_Real) m + A_offd_nnz), 1);

   /* Initial guess */
   E_ext_i     = hypre_TAlloc(HYPRE_Int, m + 1 , HYPRE_MEMORY_HOST);
   E_ext_j     = hypre_TAlloc(HYPRE_Int, E_init_alloc , HYPRE_MEMORY_HOST);
   E_ext_data  = hypre_TAlloc(HYPRE_Real, E_init_alloc , HYPRE_MEMORY_HOST);

   /* 2: Discard unecessary cols
    * Search A_ext_j, discard those cols not belong to current proc
    * First check diag, and search in offd_col_map
    */

   E_nnz       = 0;
   E_ext_i[0]  = 0;

   for( i = 0 ;  i < m ; i ++)
   {
      E_ext_i[i] = E_nnz;
      for( j = A_ext_i[i] ; j < A_ext_i[i+1] ; j ++)
      {
         big_col = A_ext_j[j];
         /* First check if that belongs to the diagonal part */
         if( big_col >= A_col_starts[0] && big_col < A_col_starts[1] )
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
            if( E_ext_j[E_nnz] >= 0)
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
         if(E_nnz >= E_init_alloc)
         {
            HYPRE_Int tmp;
            tmp = E_init_alloc;
            E_init_alloc   = E_init_alloc * EXPAND_FACT + 1;
            E_ext_j        = hypre_TReAlloc_v2(E_ext_j, HYPRE_Int, tmp, HYPRE_Int, E_init_alloc, HYPRE_MEMORY_HOST);
            E_ext_data     = hypre_TReAlloc_v2(E_ext_data, HYPRE_Real, tmp, HYPRE_Real, E_init_alloc, HYPRE_MEMORY_HOST);
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
#endif

/* This function sort offdiagonal map as well as J array for offdiagonal part
 * A: The input CSR matrix
 */
HYPRE_Int
hypre_ILUSortOffdColmap(hypre_ParCSRMatrix *A)
{
   HYPRE_Int i;
   hypre_CSRMatrix *A_offd    = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   HYPRE_BigInt *A_offd_colmap   = hypre_ParCSRMatrixColMapOffd(A);
   HYPRE_Int len              = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int nnz              = hypre_CSRMatrixNumNonzeros(A_offd);
   HYPRE_Int *perm            = hypre_TAlloc(HYPRE_Int,len,HYPRE_MEMORY_HOST);
   HYPRE_Int *rperm           = hypre_TAlloc(HYPRE_Int,len,HYPRE_MEMORY_HOST);

   for(i = 0 ; i < len ; i ++)
   {
      perm[i] = i;
   }

   hypre_BigQsort2i(A_offd_colmap,perm,0,len-1);

   for(i = 0 ; i < len ; i ++)
   {
      rperm[perm[i]] = i;
   }

   for(i = 0 ; i < nnz ; i ++)
   {
      A_offd_j[i] = rperm[A_offd_j[i]];
   }

   hypre_TFree(perm,HYPRE_MEMORY_HOST);
   hypre_TFree(rperm,HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCM
 *--------------------------------------------------------------------------*/

/* This function computes the RCM ordering of a sub matrix of
 * sparse matrix B = A(perm,perm)
 * For nonsymmetrix problem, is the RCM ordering of B + B'
 * A: The input CSR matrix
 * start:      the start position of the submatrix in B
 * end:        the end position of the submatrix in B ( exclude end, [start,end) )
 * permp:      pointer to the row permutation array such that B = A(perm, perm)
 *             point to NULL if you want to work directly on A
 *             on return, permp will point to the new permutation where
 *             in [start, end) the matrix will reordered
 * qpermp:     pointer to the col permutation array such that B = A(perm, perm)
 *             point to NULL or equal to permp if you want symmetric order
 *             on return, qpermp will point to the new permutation where
 *             in [start, end) the matrix will reordered
 * sym:        set to nonzero to work on A only(symmetric), otherwise A + A'.
 *             WARNING: if you use non-symmetric reordering, that is,
 *             different row and col reordering, the resulting A might be non-symmetric.
 *             Be careful if you are using non-symmetric reordering
 */
HYPRE_Int
hypre_ILULocalRCM( hypre_CSRMatrix *A, HYPRE_Int start, HYPRE_Int end,
                     HYPRE_Int **permp, HYPRE_Int **qpermp, HYPRE_Int sym)
{
   HYPRE_Int               i, j, row, col, r1, r2;

   HYPRE_Int               num_nodes      = end - start;
   HYPRE_Int               n              = hypre_CSRMatrixNumRows(A);
   HYPRE_Int               ncol           = hypre_CSRMatrixNumCols(A);
   HYPRE_Int               *A_i           = hypre_CSRMatrixI(A);
   HYPRE_Int               *A_j           = hypre_CSRMatrixJ(A);
   hypre_CSRMatrix         *GT            = NULL;
   hypre_CSRMatrix         *GGT           = NULL;
   //    HYPRE_Int               *AAT_i         = NULL;
   //    HYPRE_Int               *AAT_j         = NULL;
   HYPRE_Int               A_nnz          = hypre_CSRMatrixNumNonzeros(A);
   hypre_CSRMatrix         *G             = NULL;
   HYPRE_Int               *G_i           = NULL;
   HYPRE_Int               *G_j           = NULL;
   HYPRE_Real              *G_data           = NULL;
   HYPRE_Int               *G_perm        = NULL;
   HYPRE_Int               G_nnz;
   HYPRE_Int               G_capacity;
   HYPRE_Int               *perm_temp     = NULL;
   HYPRE_Int               *perm          = *permp;
   HYPRE_Int               *qperm         = *qpermp;
   HYPRE_Int               *rqperm        = NULL;

   /* 1: Preprosessing
    * Check error in input, set some parameters
    */
   if(num_nodes <= 0)
   {
      /* don't do this if we are too small */
      return hypre_error_flag;
   }
   if(n!=ncol || end > n || start < 0)
   {
      /* don't do this if the input has error */
      hypre_printf("Error input, abort RCM\n");
      return hypre_error_flag;
   }
   if(!perm)
   {
      /* create permutation array if we don't have one yet */
      perm = hypre_TAlloc( HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
      for(i = 0 ; i < n ; i ++)
      {
         perm[i] = i;
      }
   }
   if(!qperm)
   {
      /* symmetric reordering, just point it to row reordering */
      qperm = perm;
   }
   rqperm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   for(i = 0 ; i < n ; i ++)
   {
      rqperm[qperm[i]] = i;
   }
   /* 2: Build Graph
    * Build Graph for RCM ordering
    */
   G = hypre_CSRMatrixCreate(num_nodes, num_nodes, 0);
   hypre_CSRMatrixInitialize(G);
   hypre_CSRMatrixSetDataOwner(G, 1);
   G_i = hypre_CSRMatrixI(G);
   if(sym)
   {
      /* Directly use A */
      G_nnz = 0;
      G_capacity = hypre_max(A_nnz * n * n / num_nodes / num_nodes - num_nodes, 1);
      G_j = hypre_TAlloc(HYPRE_Int, G_capacity, HYPRE_MEMORY_DEVICE);
      for(i = 0 ; i < num_nodes ; i ++)
      {
         G_i[i] = G_nnz;
         row = perm[i + start];
         r1 = A_i[row];
         r2 = A_i[row+1];
         for(j = r1 ; j < r2 ; j ++)
         {
            col = rqperm[A_j[j]];
            if(col != row && col >= start && col < end)
            {
               /* this is an entry in G */
               G_j[G_nnz++] = col - start;
               if(G_nnz >= G_capacity)
               {
                  HYPRE_Int tmp = G_capacity;
                  G_capacity = G_capacity * EXPAND_FACT + 1;
                  G_j = hypre_TReAlloc_v2(G_j, HYPRE_Int, tmp, HYPRE_Int, G_capacity, HYPRE_MEMORY_DEVICE);
               }
            }
         }
      }
      G_i[num_nodes] = G_nnz;
      if(G_nnz == 0)
      {
         //G has only diagonal, no need to do any kind of RCM
         hypre_TFree(G_j, HYPRE_MEMORY_DEVICE);
         hypre_TFree(rqperm, HYPRE_MEMORY_HOST);
         *permp   = perm;
         *qpermp  = qperm;
         hypre_CSRMatrixDestroy(G);
         return hypre_error_flag;
      }
      hypre_CSRMatrixJ(G) = G_j;
      hypre_CSRMatrixNumNonzeros(G) = G_nnz;
   }
   else
   {
      /* Use A + A' */
      G_nnz = 0;
      G_capacity = hypre_max(A_nnz * n * n / num_nodes / num_nodes - num_nodes, 1);
      G_j = hypre_TAlloc(HYPRE_Int, G_capacity, HYPRE_MEMORY_DEVICE);
      for(i = 0 ; i < num_nodes ; i ++)
      {
         G_i[i] = G_nnz;
         row = perm[i + start];
         r1 = A_i[row];
         r2 = A_i[row+1];
         for(j = r1 ; j < r2 ; j ++)
         {
            col = rqperm[A_j[j]];
            if(col != row && col >= start && col < end)
            {
               /* this is an entry in G */
               G_j[G_nnz++] = col - start;
               if(G_nnz >= G_capacity)
               {
                  HYPRE_Int tmp = G_capacity;
                  G_capacity = G_capacity * EXPAND_FACT + 1;
                  G_j = hypre_TReAlloc_v2(G_j, HYPRE_Int, tmp, HYPRE_Int, G_capacity, HYPRE_MEMORY_DEVICE);
               }
            }
         }
      }
      G_i[num_nodes] = G_nnz;
      if(G_nnz == 0)
      {
         //G has only diagonal, no need to do any kind of RCM
         hypre_TFree(G_j, HYPRE_MEMORY_DEVICE);
         hypre_TFree(rqperm, HYPRE_MEMORY_HOST);
         *permp   = perm;
         *qpermp  = qperm;
         hypre_CSRMatrixDestroy(G);
         return hypre_error_flag;
      }
      hypre_CSRMatrixJ(G) = G_j;
      G_data = hypre_CTAlloc(HYPRE_Real, G_nnz, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixData(G) = G_data;
      hypre_CSRMatrixNumNonzeros(G) = G_nnz;

      /* now sum G with G' */
      hypre_CSRMatrixTranspose(G, &GT, 1);
      GGT = hypre_CSRMatrixAdd(1.0, G, 1.0, GT);
      hypre_CSRMatrixDestroy(G);
      hypre_CSRMatrixDestroy(GT);
      G = GGT;
      GGT = NULL;
   }

   /* 3: Build Graph
    * Build RCM
    */
   /* no need to be shared, but perm should be shared */
   G_perm = hypre_TAlloc(HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
   hypre_ILULocalRCMOrder( G, G_perm);

   /* 4: Post processing
    * Free, set value, return
    */

   /* update to new index */
   perm_temp = hypre_TAlloc(HYPRE_Int, num_nodes, HYPRE_MEMORY_HOST);
   for( i = 0 ; i < num_nodes ; i ++)
   {
      perm_temp[i] = perm[i + start];
   }
   for( i = 0 ; i < num_nodes ; i ++)
   {
      perm[i+start] = perm_temp[G_perm[i]];
   }
   if(perm != qperm)
   {
      for( i = 0 ; i < num_nodes ; i ++)
      {
         perm_temp[i] = qperm[i + start];
      }
      for( i = 0 ; i < num_nodes ; i ++)
      {
         qperm[i+start] = perm_temp[G_perm[i]];
      }
   }
   *permp   = perm;
   *qpermp  = qperm;
   hypre_CSRMatrixDestroy(G);

   hypre_TFree(G_perm, HYPRE_MEMORY_HOST);
   hypre_TFree(perm_temp, HYPRE_MEMORY_HOST);
   hypre_TFree(rqperm, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMMindegree
 *--------------------------------------------------------------------------*/

/* This function finds the unvisited node with the minimum degree
 */
HYPRE_Int
hypre_ILULocalRCMMindegree(HYPRE_Int n, HYPRE_Int *degree, HYPRE_Int *marker, HYPRE_Int *rootp)
{
    HYPRE_Int i;
    HYPRE_Int min_degree = n+1;
    HYPRE_Int root = 0;
    for(i = 0 ; i < n ; i ++)
    {
        if(marker[i] < 0)
        {
            if(degree[i] < min_degree)
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
 *--------------------------------------------------------------------------*/

/* This function actually does the RCM ordering of a symmetric csr matrix (entire)
 * A: the csr matrix, A_data is not needed
 * perm: the permutation array, space should be allocated outside
 */
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
   for(i = 0 ; i < n ; i ++)
   {
      degree[i] = A_i[i+1] - A_i[i];
      marker[i] = -1;
   }

   /* start RCM loop */
   current_num = 0;
   while(current_num < n)
   {
      hypre_ILULocalRCMMindegree( n, degree, marker, &root);
      /* This is a new connect component */
      hypre_ILULocalRCMFindPPNode(A, &root, marker);

      /* Numbering of this component */
      hypre_ILULocalRCMNumbering(A, root, marker, perm, &current_num);
   }

   /* free */
   hypre_TFree(degree, HYPRE_MEMORY_HOST);
   hypre_TFree(marker, HYPRE_MEMORY_HOST);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMFindPPNode
 *--------------------------------------------------------------------------*/

/* This function find a pseudo-peripheral node start from root
 * A: the csr matrix, A_data is not needed
 * rootp: pointer to the root, on return will be a end of the pseudo-peripheral
 * marker: the marker array for unvisited node
 */
HYPRE_Int
hypre_ILULocalRCMFindPPNode( hypre_CSRMatrix *A, HYPRE_Int *rootp, HYPRE_Int *marker)
{
   HYPRE_Int      i, r1, r2, row, min_degree, lev_degree, nlev, newnlev;

   HYPRE_Int      root           = *rootp;
   HYPRE_Int      n              = hypre_CSRMatrixNumRows(A);
   HYPRE_Int      *A_i           = hypre_CSRMatrixI(A);
   /* at most n levels */
   HYPRE_Int      *level_i       = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_HOST);
   HYPRE_Int      *level_j       = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);

   /* build initial level structure from root */
   hypre_ILULocalRCMBuildLevel( A, root, marker, level_i, level_j, &newnlev);

   nlev  = newnlev - 1;
   while(nlev < newnlev)
   {
      nlev = newnlev;
      r1 =  level_i[nlev-1];
      r2 =  level_i[nlev];
      min_degree = n;
      for(i = r1 ; i < r2 ; i ++)
      {
         /* select the last level, pick min-degree node */
         row = level_j[i];
         lev_degree = A_i[row+1] - A_i[row];
         if(min_degree > lev_degree)
         {
            min_degree = lev_degree;
            root = row;
         }
      }
      hypre_ILULocalRCMBuildLevel( A, root, marker, level_i, level_j, &newnlev);
   }

   *rootp = root;
   /* free */
   hypre_TFree(level_i, HYPRE_MEMORY_HOST);
   hypre_TFree(level_j, HYPRE_MEMORY_HOST);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMBuildLevel
 *--------------------------------------------------------------------------*/

/* This function build level structure start from root
 * A: the csr matrix, A_data is not needed
 * root: pointer to the root
 * marker: the marker array for unvisited node
 * level_i: points to the start/end of position on level_j, similar to CSR Matrix
 * level_j: store node number on each level
 * nlevp: return the number of level on this level structure
 */
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

   //explore nbhds of all nodes in current level
   while(l2 > l1)
   {
      level_i[nlev++] = l2;
      /* loop through last level */
      for(i = l1 ; i < l2 ; i ++)
      {
         /* the node to explore */
         rowi = level_j[i];
         r1 = A_i[rowi];
         r2 = A_i[rowi + 1];
         for(j = r1 ; j < r2 ; j ++)
         {
            rowj = A_j[j];
            if( marker[rowj] < 0 )
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
   for(i = 0 ; i < l2 ; i ++)
   {
      marker[level_j[i]] = -1;
   }

   *nlevp = nlev;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMNumbering
 *--------------------------------------------------------------------------*/

/* This function generate numbering for a connect component
 * A: the csr matrix, A_data is not needed
 * root: pointer to the root
 * marker: the marker array for unvisited node
 * perm: permutation array
 * current_nump: number of nodes already have a perm value
 */

HYPRE_Int
hypre_ILULocalRCMNumbering(hypre_CSRMatrix *A, HYPRE_Int root, HYPRE_Int *marker, HYPRE_Int *perm, HYPRE_Int *current_nump)
{
    HYPRE_Int        i, j, l1, l2, r1, r2, rowi, rowj, row_start, row_end;
    HYPRE_Int        *A_i        = hypre_CSRMatrixI(A);
    HYPRE_Int        *A_j        = hypre_CSRMatrixJ(A);
    HYPRE_Int        current_num = *current_nump;


    marker[root]        = 0;
    l1                  = current_num;
    perm[current_num++] = root;
    l2                  = current_num;

    while(l2 > l1)
    {
       /* loop through all nodes is current level */
       for(i = l1 ; i < l2 ; i ++)
       {
          rowi = perm[i];
          r1 = A_i[rowi];
          r2 = A_i[rowi+1];
          row_start = current_num;
          for(j = r1 ; j < r2 ; j ++)
          {
             rowj = A_j[j];
             if(marker[rowj] < 0)
             {
                /* save the degree in marker and add it to perm */
                marker[rowj] = A_i[rowj+1] - A_i[rowj];
                perm[current_num++] = rowj;
             }
          }
          row_end = current_num;
          hypre_ILULocalRCMQsort(perm, row_start, row_end-1, marker);
       }
       l1 = l2;
       l2 = current_num;
    }

    //reverse
    hypre_ILULocalRCMReverse(perm, *current_nump, current_num-1);
    *current_nump = current_num;
    return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMQsort
 *--------------------------------------------------------------------------*/

/* This qsort is very specialized, not worth to put into utilities
 * Sort a part of array perm based on degree value (ascend)
 * That is, if degree[perm[i]] < degree[perm[j]], we should have i < j
 * perm: the perm array
 * start: start in perm
 * end: end in perm
 * degree: degree array
 */

HYPRE_Int
hypre_ILULocalRCMQsort(HYPRE_Int *perm, HYPRE_Int start, HYPRE_Int end, HYPRE_Int *degree)
{

    HYPRE_Int i, mid;
    if(start >= end)
    {
        return hypre_error_flag;
    }

    hypre_swap(perm, start, (start + end) / 2);
    mid = start;
    //loop to split
    for(i = start + 1 ; i <= end ; i ++)
    {
        if(degree[perm[i]] < degree[perm[start]])
        {
            hypre_swap(perm, ++mid, i);
        }
    }
    hypre_swap(perm, start, mid);
    hypre_ILULocalRCMQsort(perm, mid+1, end, degree);
    hypre_ILULocalRCMQsort(perm, start, mid-1, degree);
    return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILULocalRCMReverse
 *--------------------------------------------------------------------------*/

/* Last step in RCM, reverse it
 * perm: perm array
 * srart: start position
 * end: end position
 */

HYPRE_Int
hypre_ILULocalRCMReverse(HYPRE_Int *perm, HYPRE_Int start, HYPRE_Int end)
{
    HYPRE_Int     i, j;
    HYPRE_Int     mid = (start + end + 1) / 2;

    for(i = start, j = end ; i < mid ; i ++, j--)
    {
        hypre_swap(perm, i, j);
    }
   return hypre_error_flag;
}

#ifdef HYPRE_USING_CUDA

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESDummySetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILUCusparseSchurGMRESDummySetup(void *a, void *b, void *c, void *d)
{
   /* Null GMRES setup, does nothing */
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESDummySolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILUCusparseSchurGMRESDummySolve( void               *ilu_vdata,
                  void               *ilu_vdata2,
                  hypre_ParVector    *f,
                  hypre_ParVector    *u )
{
   /* Unit GMRES preconditioner, just copy data from one slot to another */
   hypre_ParILUData *ilu_data                = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix *A                     = hypre_ParILUDataMatS(ilu_data);
   hypre_CSRMatrix         *A_diag           = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int               n_local           = hypre_CSRMatrixNumRows(A_diag);

   hypre_Vector            *u_local          = hypre_ParVectorLocalVector(u);
   HYPRE_Real              *u_data           = hypre_VectorData(u_local);

   hypre_Vector            *f_local          = hypre_ParVectorLocalVector(f);
   HYPRE_Real              *f_data           = hypre_VectorData(f_local);

   cudaDeviceSynchronize();
   hypre_TMemcpy(u_data, f_data, HYPRE_Real, n_local, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESCommInfo
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILUCusparseSchurGMRESCommInfo( void *ilu_vdata, HYPRE_Int *my_id, HYPRE_Int *num_procs)
{
   /* get comm info from ilu_data */
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix *A = hypre_ParILUDataMatS(ilu_data);
   MPI_Comm comm = hypre_ParCSRMatrixComm ( A );
   hypre_MPI_Comm_size(comm,num_procs);
   hypre_MPI_Comm_rank(comm,my_id);
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_ParILUCusparseSchurGMRESMatvecCreate( void   *ilu_vdata,
                             void   *x )
{
   /* Null matvec create */
   void *matvec_data;
   matvec_data = NULL;
   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILUCusparseSchurGMRESMatvec( void   *matvec_data,
                       HYPRE_Complex  alpha,
                       void   *ilu_vdata,
                       void   *x,
                       HYPRE_Complex  beta,
                       void   *y           )
{
   /* Slightly different, for this new matvec, the diagonal of the original matrix
    * is the LU factorization. Thus, the matvec is done in an different way
    * |IS_1 E_12 E_13|
    * |E_21 IS_2 E_23| = S
    * |E_31 E_32 IS_3|
    *
    * |IS_1          |
    * |     IS_2     | = M
    * |          IS_3|
    *
    * Solve Sy = g is just M^{-1}S = M^{-1}g
    *
    * |      I       IS_1^{-1}E_12 IS_1^{-1}E_13|
    * |IS_2^{-1}E_21       I       IS_2^{-1}E_23| = M^{-1}S
    * |IS_3^{-1}E_31 IS_3^{-1}E_32       I      |
    *
    * */

   /* get matrix information first */
   hypre_ParILUData *ilu_data                   = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix *A                        = hypre_ParILUDataMatS(ilu_data);

   /* fist step, apply matvec on empty diagonal slot */
   hypre_CSRMatrix   *A_diag                    = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int         *A_diag_i                  = hypre_CSRMatrixI(A_diag);
   HYPRE_Int         *A_diag_j                  = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real        *A_diag_data               = hypre_CSRMatrixData(A_diag);
   HYPRE_Int         A_diag_n                   = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int         A_diag_nnz                 = A_diag_i[A_diag_n];
   HYPRE_Int         *A_diag_fake_i             = hypre_ParILUDataMatAFakeDiagonal(ilu_data);

   cusparseMatDescr_t      matL_des             = hypre_ParILUDataMatLMatrixDescription(ilu_data);
   cusparseMatDescr_t      matU_des             = hypre_ParILUDataMatUMatrixDescription(ilu_data);
   void                    *ilu_solve_buffer    = hypre_ParILUDataILUSolveBuffer(ilu_data);//device memory
   cusparseSolvePolicy_t   ilu_solve_policy     = hypre_ParILUDataILUSolvePolicy(ilu_data);
   csrsv2Info_t            matSL_info           = hypre_ParILUDataMatSLILUSolveInfo(ilu_data);
   csrsv2Info_t            matSU_info           = hypre_ParILUDataMatSUILUSolveInfo(ilu_data);

   HYPRE_Int               isDoublePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int               isSinglePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   hypre_assert(isDoublePrecision || isSinglePrecision);

   hypre_ParVector         *xtemp               = hypre_ParILUDataXTemp(ilu_data);
   hypre_Vector            *xtemp_local         = hypre_ParVectorLocalVector(xtemp);
   HYPRE_Real              *xtemp_data          = hypre_VectorData(xtemp_local);
   hypre_ParVector         *ytemp               = hypre_ParILUDataYTemp(ilu_data);
   hypre_Vector            *ytemp_local         = hypre_ParVectorLocalVector(ytemp);
   HYPRE_Real              *ytemp_data          = hypre_VectorData(ytemp_local);
   HYPRE_Real              zero                 = 0.0;
   HYPRE_Real              one                  = 1.0;

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());

   /* Matvec with
    *         |  O  E_12 E_13|
    * alpha * |E_21   O  E_23|
    *         |E_31 E_32   O |
    * store in xtemp
    */
   hypre_CSRMatrixI(A_diag)                     = A_diag_fake_i;
   hypre_ParCSRMatrixMatvec( alpha, (hypre_ParCSRMatrix *) A, (hypre_ParVector *) x, zero, xtemp );
   hypre_CSRMatrixI(A_diag)                     = A_diag_i;

   /* Compute U^{-1}*L^{-1}*(A_offd * x)
    * Or in another word, matvec with
    *         |      O       IS_1^{-1}E_12 IS_1^{-1}E_13|
    * alpha * |IS_2^{-1}E_21       O       IS_2^{-1}E_23|
    *         |IS_3^{-1}E_31 IS_3^{-1}E_32       O      |
    * store in xtemp
    */
   if( A_diag_n > 0 )
   {
      if(isDoublePrecision)
      {
         /* L solve - Forward solve */
         HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   A_diag_n, A_diag_nnz, (hypre_double *) &one, matL_des,
                                                   (hypre_double *) A_diag_data, A_diag_i, A_diag_j, matSL_info,
                                                   (hypre_double *) xtemp_data, (hypre_double *) ytemp_data, ilu_solve_policy, ilu_solve_buffer));

         /* U solve - Backward substitution */
         HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   A_diag_n, A_diag_nnz, (hypre_double *) &one, matU_des,
                                                   (hypre_double *) A_diag_data, A_diag_i, A_diag_j, matSU_info,
                                                   (hypre_double *) ytemp_data, (hypre_double *) xtemp_data, ilu_solve_policy, ilu_solve_buffer));
      }
      else if(isSinglePrecision)
      {
         /* L solve - Forward solve */
         HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   A_diag_n, A_diag_nnz, (float *) &one, matL_des,
                                                   (float *) A_diag_data, A_diag_i, A_diag_j, matSL_info,
                                                   (float *) xtemp_data, (float *) ytemp_data, ilu_solve_policy, ilu_solve_buffer));

         /* U solve - Backward substitution */
         HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   A_diag_n, A_diag_nnz, (float *) &one, matU_des,
                                                   (float *) A_diag_data, A_diag_i, A_diag_j, matSU_info,
                                                   (float *) ytemp_data, (float *) xtemp_data, ilu_solve_policy, ilu_solve_buffer));
      }
   }

   /* now add the original x onto it */
   hypre_ParVectorAxpy( alpha, (hypre_ParVector *) x, (hypre_ParVector *) xtemp);

   /* finall, add that into y and get final result */
   hypre_ParVectorScale( beta, (hypre_ParVector *) y );
   hypre_ParVectorAxpy( one, xtemp, (hypre_ParVector *) y);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESMatvecDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILUCusparseSchurGMRESMatvecDestroy( void *matvec_data )
{
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESDummySetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPSchurGMRESDummySetup(void *a, void *b, void *c, void *d)
{
   /* Null GMRES setup, does nothing */
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESDummySolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPSchurGMRESSolve( void               *ilu_vdata,
                  void               *ilu_vdata2,
                  hypre_ParVector    *f,
                  hypre_ParVector    *u )
{
   /* Unit GMRES preconditioner, just copy data from one slot to another */
   hypre_ParILUData        *ilu_data            = (hypre_ParILUData*) ilu_vdata;
   //hypre_ParCSRMatrix      *Aperm               = hypre_ParILUDataAperm(ilu_data);

   cusparseHandle_t        handle               = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t      matL_des             = hypre_ParILUDataMatLMatrixDescription(ilu_data);
   cusparseMatDescr_t      matU_des             = hypre_ParILUDataMatUMatrixDescription(ilu_data);
   void                    *ilu_solve_buffer    = hypre_ParILUDataILUSolveBuffer(ilu_data);//device memory
   cusparseSolvePolicy_t   ilu_solve_policy     = hypre_ParILUDataILUSolvePolicy(ilu_data);

   HYPRE_Int               isDoublePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int               isSinglePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   hypre_assert(isDoublePrecision || isSinglePrecision);

   hypre_ParCSRMatrix      *S                   = hypre_ParILUDataMatS(ilu_data);
   hypre_CSRMatrix         *SLU                 = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int               *SLU_i               = hypre_CSRMatrixI(SLU);
   HYPRE_Int               *SLU_j               = hypre_CSRMatrixJ(SLU);
   HYPRE_Real              *SLU_data            = hypre_CSRMatrixData(SLU);
   HYPRE_Int               SLU_nnz              = hypre_CSRMatrixNumNonzeros(SLU);

   csrsv2Info_t            matSL_info           = hypre_ParILUDataMatSLILUSolveInfo(ilu_data);
   csrsv2Info_t            matSU_info           = hypre_ParILUDataMatSUILUSolveInfo(ilu_data);

   //HYPRE_Int               n                    = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(Aperm));
   HYPRE_Int               m                    = hypre_CSRMatrixNumRows(SLU);
   //HYPRE_Int               nLU                  = n - m;

   hypre_ParVector         *f_vec               = (hypre_ParVector *) f;
   hypre_Vector            *f_local             = hypre_ParVectorLocalVector(f_vec);
   HYPRE_Real              *f_data              = hypre_VectorData(f_local);
   hypre_ParVector         *u_vec               = (hypre_ParVector *) u;
   hypre_Vector            *u_local             = hypre_ParVectorLocalVector(u_vec);
   HYPRE_Real              *u_data              = hypre_VectorData(u_local);
   hypre_ParVector         *rhs                 = hypre_ParILUDataRhs(ilu_data);
   hypre_Vector            *rhs_local           = hypre_ParVectorLocalVector(rhs);
   HYPRE_Real              *rhs_data            = hypre_VectorData(rhs_local);

   //HYPRE_Real zero = 0.0;
   HYPRE_Real one = 1.0;
   //HYPRE_Real mone = -1.0;

   if(m > 0)
   {
      if(isDoublePrecision)
      {
         /* L solve */
         HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   m, SLU_nnz, (hypre_double *) &one, matL_des,
                                                   (hypre_double *) SLU_data, SLU_i, SLU_j, matSL_info,
                                                   (hypre_double *) f_data, (hypre_double *) rhs_data, ilu_solve_policy, ilu_solve_buffer));
         /* U solve */
         HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   m, SLU_nnz, (hypre_double *) &one, matU_des,
                                                   (hypre_double *) SLU_data, SLU_i, SLU_j, matSU_info,
                                                   (hypre_double *) rhs_data, (hypre_double *) u_data, ilu_solve_policy, ilu_solve_buffer));
      }
      else if(isSinglePrecision)
      {
         /* L solve */
         HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   m, SLU_nnz, (float *) &one, matL_des,
                                                   (float *) SLU_data, SLU_i, SLU_j, matSL_info,
                                                   (float *) f_data, (float *) rhs_data, ilu_solve_policy, ilu_solve_buffer));
         /* U solve */
         HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   m, SLU_nnz, (float *) &one, matU_des,
                                                   (float *) SLU_data, SLU_i, SLU_j, matSU_info,
                                                   (float *) rhs_data, (float *) u_data, ilu_solve_policy, ilu_solve_buffer));
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_ParILURAPSchurGMRESMatvecCreate( void   *ilu_vdata,
                             void   *x )
{
   /* Null matvec create */
   void *matvec_data;
   matvec_data = NULL;
   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPSchurGMRESMatvec( void   *matvec_data,
                       HYPRE_Complex  alpha,
                       void   *ilu_vdata,
                       void   *x,
                       HYPRE_Complex  beta,
                       void   *y           )
{
   /* Compute y = alpha * S * x + beta * y
    * */

   /* get matrix information first */
   hypre_ParILUData        *ilu_data            = (hypre_ParILUData*) ilu_vdata;

   HYPRE_Int               test_opt             = hypre_ParILUDataTestOption(ilu_data);

   switch(test_opt)
   {
      case 1:
      {
         /* S = R * A * P */
         hypre_ParCSRMatrix      *Aperm               = hypre_ParILUDataAperm(ilu_data);

         cusparseHandle_t        handle               = hypre_HandleCusparseHandle(hypre_handle());
         cusparseMatDescr_t      matL_des             = hypre_ParILUDataMatLMatrixDescription(ilu_data);
         cusparseMatDescr_t      matU_des             = hypre_ParILUDataMatUMatrixDescription(ilu_data);
         void                    *ilu_solve_buffer    = hypre_ParILUDataILUSolveBuffer(ilu_data);//device memory
         cusparseSolvePolicy_t   ilu_solve_policy     = hypre_ParILUDataILUSolvePolicy(ilu_data);

         HYPRE_Int               isDoublePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double);
         HYPRE_Int               isSinglePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

         hypre_assert(isDoublePrecision || isSinglePrecision);

         hypre_CSRMatrix         *EiU                 = hypre_ParILUDataMatEDevice(ilu_data);
         hypre_CSRMatrix         *iLF                 = hypre_ParILUDataMatFDevice(ilu_data);
         hypre_CSRMatrix         *BLU                 = hypre_ParILUDataMatBILUDevice(ilu_data);
         HYPRE_Int               *BLU_i               = hypre_CSRMatrixI(BLU);
         HYPRE_Int               *BLU_j               = hypre_CSRMatrixJ(BLU);
         HYPRE_Real              *BLU_data            = hypre_CSRMatrixData(BLU);
         HYPRE_Int               BLU_nnz              = hypre_CSRMatrixNumNonzeros(BLU);

         csrsv2Info_t            matBL_info           = hypre_ParILUDataMatBLILUSolveInfo(ilu_data);
         csrsv2Info_t            matBU_info           = hypre_ParILUDataMatBUILUSolveInfo(ilu_data);

         HYPRE_Int               n                    = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(Aperm));
         HYPRE_Int               nLU                  = hypre_CSRMatrixNumRows(BLU);
         HYPRE_Int               m                    = n - nLU;

         hypre_ParVector         *x_vec               = (hypre_ParVector *) x;
         hypre_Vector            *x_local             = hypre_ParVectorLocalVector(x_vec);
         HYPRE_Real              *x_data              = hypre_VectorData(x_local);
         hypre_ParVector         *y_vec               = (hypre_ParVector *) y;
         hypre_Vector            *y_local             = hypre_ParVectorLocalVector(y_vec);
         //HYPRE_Real              *y_data              = hypre_VectorData(y_local);
         hypre_ParVector         *xtemp               = hypre_ParILUDataUTemp(ilu_data);
         hypre_Vector            *xtemp_local         = hypre_ParVectorLocalVector(xtemp);
         HYPRE_Real              *xtemp_data          = hypre_VectorData(xtemp_local);
         hypre_ParVector         *ytemp               = hypre_ParILUDataYTemp(ilu_data);
         hypre_Vector            *ytemp_local         = hypre_ParVectorLocalVector(ytemp);
         HYPRE_Real              *ytemp_data          = hypre_VectorData(ytemp_local);

         hypre_Vector *xtemp_upper           = hypre_SeqVectorCreate(nLU);
         hypre_Vector *ytemp_upper           = hypre_SeqVectorCreate(nLU);
         hypre_Vector *xtemp_lower           = hypre_SeqVectorCreate(m);
         hypre_VectorOwnsData(xtemp_upper)   = 0;
         hypre_VectorOwnsData(ytemp_upper)   = 0;
         hypre_VectorOwnsData(xtemp_lower)   = 0;
         hypre_VectorData(xtemp_upper)       = xtemp_data;
         hypre_VectorData(ytemp_upper)       = ytemp_data;
         hypre_VectorData(xtemp_lower)       = xtemp_data+nLU;
         hypre_SeqVectorInitialize(xtemp_upper);
         hypre_SeqVectorInitialize(ytemp_upper);
         hypre_SeqVectorInitialize(xtemp_lower);

         HYPRE_Real zero = 0.0;
         HYPRE_Real one = 1.0;
         HYPRE_Real mone = -1.0;

         /* first step, compute P*x put in y */
         /* -Fx */
         hypre_CSRMatrixMatvec(mone, iLF, x_local, zero, ytemp_upper);
         /* -L^{-1}Fx */
         if(isDoublePrecision)
         {
            /* L solve */
            HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      nLU, BLU_nnz, (hypre_double *) &one, matL_des,
                                                      (hypre_double *) BLU_data, BLU_i, BLU_j, matBL_info,
                                                      (hypre_double *) ytemp_data, (hypre_double *) xtemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         else if(isSinglePrecision)
         {
            /* L solve */
            HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      nLU, BLU_nnz, (float *) &one, matL_des,
                                                      (float *) BLU_data, BLU_i, BLU_j, matBL_info,
                                                      (float *) ytemp_data, (float *) xtemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         /* -U{-1}L^{-1}Fx */
         if(isDoublePrecision)
         {
            /* U solve */
            HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      nLU, BLU_nnz, (hypre_double *) &one, matU_des,
                                                      (hypre_double *) BLU_data, BLU_i, BLU_j, matBU_info,
                                                      (hypre_double *) xtemp_data, (hypre_double *) ytemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         else if(isSinglePrecision)
         {
            /* U solve */
            HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      nLU, BLU_nnz, (float *) &one, matU_des,
                                                      (float *) BLU_data, BLU_i, BLU_j, matBU_info,
                                                      (float *) xtemp_data, (float *) ytemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         /* now copy data to y_lower */
         hypre_TMemcpy( ytemp_data + nLU, x_data, HYPRE_Real, m, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

         /* second step, compute A*P*x store in xtemp */
         hypre_ParCSRMatrixMatvec( one, Aperm, ytemp, zero, xtemp);

         /* third step, compute R*A*P*x */
         /* solve L^{-1} */
         if(isDoublePrecision)
         {
            /* L solve */
            HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      nLU, BLU_nnz, (hypre_double *) &one, matL_des,
                                                      (hypre_double *) BLU_data, BLU_i, BLU_j, matBL_info,
                                                      (hypre_double *) xtemp_data, (hypre_double *) ytemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         else if(isSinglePrecision)
         {
            /* L solve */
            HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      nLU, BLU_nnz, (float *) &one, matL_des,
                                                      (float *) BLU_data, BLU_i, BLU_j, matBL_info,
                                                      (float *) xtemp_data, (float *) ytemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         /* U^{-1}L^{-1} */
         if(isDoublePrecision)
         {
            /* U solve */
            HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      nLU, BLU_nnz, (hypre_double *) &one, matU_des,
                                                      (hypre_double *) BLU_data, BLU_i, BLU_j, matBU_info,
                                                      (hypre_double *) ytemp_data, (hypre_double *) xtemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         else if(isSinglePrecision)
         {
            /* U solve */
            HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      nLU, BLU_nnz, (float *) &one, matU_des,
                                                      (float *) BLU_data, BLU_i, BLU_j, matBU_info,
                                                      (float *) ytemp_data, (float *) xtemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         /* -EU^{-1}L^{-1} */
         hypre_CSRMatrixMatvec(mone * alpha, EiU, xtemp_upper, beta, y_local);
         /* I*lower-EU^{-1}L^{-1}*upper */
         hypre_SeqVectorAxpy(alpha, xtemp_lower, y_local);

         /* over */
         hypre_SeqVectorDestroy(xtemp_upper);
         hypre_SeqVectorDestroy(ytemp_upper);
         hypre_SeqVectorDestroy(xtemp_lower);
      }
      break;
      case 2:
      {
         /* S = C - EU^{-1} * L^{-1}F */

         //hypre_ParCSRMatrix      *Aperm               = hypre_ParILUDataAperm(ilu_data);

         //cusparseHandle_t        handle               = hypre_HandleCusparseHandle(hypre_handle());
         //cusparseMatDescr_t      matL_des             = hypre_ParILUDataMatLMatrixDescription(ilu_data);
         //cusparseMatDescr_t      matU_des             = hypre_ParILUDataMatUMatrixDescription(ilu_data);
         //void                    *ilu_solve_buffer    = hypre_ParILUDataILUSolveBuffer(ilu_data);//device memory
         //cusparseSolvePolicy_t   ilu_solve_policy     = hypre_ParILUDataILUSolvePolicy(ilu_data);

         //HYPRE_Int               isDoublePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double);
         //HYPRE_Int               isSinglePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

         //hypre_assert(isDoublePrecision || isSinglePrecision);

         hypre_CSRMatrix         *EiU                 = hypre_ParILUDataMatEDevice(ilu_data);
         hypre_CSRMatrix         *iLF                 = hypre_ParILUDataMatFDevice(ilu_data);
         hypre_CSRMatrix         *C                   = hypre_ParILUDataMatSILUDevice(ilu_data);

         //HYPRE_Int               n                    = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(Aperm));
         HYPRE_Int               nLU                  = hypre_CSRMatrixNumRows(C);
         //HYPRE_Int               m                    = n - nLU;

         hypre_ParVector         *x_vec               = (hypre_ParVector *) x;
         hypre_Vector            *x_local             = hypre_ParVectorLocalVector(x_vec);
         //HYPRE_Real              *x_data              = hypre_VectorData(x_local);
         hypre_ParVector         *y_vec               = (hypre_ParVector *) y;
         hypre_Vector            *y_local             = hypre_ParVectorLocalVector(y_vec);
         //HYPRE_Real              *y_data              = hypre_VectorData(y_local);
         hypre_ParVector         *xtemp               = hypre_ParILUDataUTemp(ilu_data);
         hypre_Vector            *xtemp_local         = hypre_ParVectorLocalVector(xtemp);
         HYPRE_Real              *xtemp_data          = hypre_VectorData(xtemp_local);

         hypre_Vector *xtemp_upper           = hypre_SeqVectorCreate(nLU);
         hypre_VectorOwnsData(xtemp_upper)   = 0;
         hypre_VectorData(xtemp_upper)       = xtemp_data;
         hypre_SeqVectorInitialize(xtemp_upper);

         HYPRE_Real zero = 0.0;
         HYPRE_Real one = 1.0;
         HYPRE_Real mone = -1.0;

         /* first step, compute EB^{-1}F*x put in y */
         /* -L^{-1}Fx */
         hypre_CSRMatrixMatvec(mone, iLF, x_local, zero, xtemp_upper);
         /* - alpha EU^{-1}L^{-1}Fx + beta * y */
         hypre_CSRMatrixMatvec( alpha, EiU, xtemp_upper, beta, y_local);
         /* alpha * C - alpha EU^{-1}L^{-1}Fx + beta y */
         hypre_CSRMatrixMatvec( alpha, C, x_local, one, y_local);

         /* over */
         hypre_SeqVectorDestroy(xtemp_upper);
      }
      break;
      case 3:
      {
         /* S = C - EU^{-1} * L^{-1}F */

         //hypre_ParCSRMatrix      *Aperm               = hypre_ParILUDataAperm(ilu_data);

         cusparseHandle_t        handle               = hypre_HandleCusparseHandle(hypre_handle());
         cusparseMatDescr_t      matL_des             = hypre_ParILUDataMatLMatrixDescription(ilu_data);
         cusparseMatDescr_t      matU_des             = hypre_ParILUDataMatUMatrixDescription(ilu_data);
         void                    *ilu_solve_buffer    = hypre_ParILUDataILUSolveBuffer(ilu_data);//device memory
         cusparseSolvePolicy_t   ilu_solve_policy     = hypre_ParILUDataILUSolvePolicy(ilu_data);

         HYPRE_Int               isDoublePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double);
         HYPRE_Int               isSinglePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

         hypre_assert(isDoublePrecision || isSinglePrecision);

         hypre_CSRMatrix         *EiU                 = hypre_ParILUDataMatEDevice(ilu_data);
         hypre_CSRMatrix         *iLF                 = hypre_ParILUDataMatFDevice(ilu_data);
         hypre_CSRMatrix         *C                   = hypre_ParILUDataMatSILUDevice(ilu_data);
         hypre_CSRMatrix         *BLU                 = hypre_ParILUDataMatBILUDevice(ilu_data);
         HYPRE_Int               *BLU_i               = hypre_CSRMatrixI(BLU);
         HYPRE_Int               *BLU_j               = hypre_CSRMatrixJ(BLU);
         HYPRE_Real              *BLU_data            = hypre_CSRMatrixData(BLU);
         HYPRE_Int               BLU_nnz              = hypre_CSRMatrixNumNonzeros(BLU);

         csrsv2Info_t            matBL_info           = hypre_ParILUDataMatBLILUSolveInfo(ilu_data);
         csrsv2Info_t            matBU_info           = hypre_ParILUDataMatBUILUSolveInfo(ilu_data);

         //HYPRE_Int               n                    = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(Aperm));
         HYPRE_Int               nLU                  = hypre_CSRMatrixNumRows(C);
         //HYPRE_Int               m                    = n - nLU;

         hypre_ParVector         *x_vec               = (hypre_ParVector *) x;
         hypre_Vector            *x_local             = hypre_ParVectorLocalVector(x_vec);
         //HYPRE_Real              *x_data              = hypre_VectorData(x_local);
         hypre_ParVector         *y_vec               = (hypre_ParVector *) y;
         hypre_Vector            *y_local             = hypre_ParVectorLocalVector(y_vec);
         //HYPRE_Real              *y_data              = hypre_VectorData(y_local);
         hypre_ParVector         *xtemp               = hypre_ParILUDataUTemp(ilu_data);
         hypre_Vector            *xtemp_local         = hypre_ParVectorLocalVector(xtemp);
         HYPRE_Real              *xtemp_data          = hypre_VectorData(xtemp_local);
         hypre_ParVector         *ytemp               = hypre_ParILUDataYTemp(ilu_data);
         hypre_Vector            *ytemp_local         = hypre_ParVectorLocalVector(ytemp);
         HYPRE_Real              *ytemp_data          = hypre_VectorData(ytemp_local);

         hypre_Vector *xtemp_upper           = hypre_SeqVectorCreate(nLU);
         hypre_VectorOwnsData(xtemp_upper)   = 0;
         hypre_VectorData(xtemp_upper)       = xtemp_data;
         hypre_SeqVectorInitialize(xtemp_upper);

         HYPRE_Real zero = 0.0;
         HYPRE_Real one = 1.0;
         HYPRE_Real mone = -1.0;

         /* first step, compute EB^{-1}F*x put in y */
         /* -Fx */
         hypre_CSRMatrixMatvec(mone, iLF, x_local, zero, xtemp_upper);
         /* -L^{-1}Fx */
         if(isDoublePrecision)
         {
            /* L solve */
            HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      nLU, BLU_nnz, (hypre_double *) &one, matL_des,
                                                      (hypre_double *) BLU_data, BLU_i, BLU_j, matBL_info,
                                                      (hypre_double *) xtemp_data, (hypre_double *) ytemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         else if(isSinglePrecision)
         {
            /* L solve */
            HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      nLU, BLU_nnz, (float *) &one, matL_des,
                                                      (float *) BLU_data, BLU_i, BLU_j, matBL_info,
                                                      (float *) xtemp_data, (float *) ytemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         /* -U^{-1}L^{-1}Fx */
         if(isDoublePrecision)
         {
            /* U solve */
            HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      nLU, BLU_nnz, (hypre_double *) &one, matU_des,
                                                      (hypre_double *) BLU_data, BLU_i, BLU_j, matBU_info,
                                                      (hypre_double *) ytemp_data, (hypre_double *) xtemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         else if(isSinglePrecision)
         {
            /* U solve */
            HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      nLU, BLU_nnz, (float *) &one, matU_des,
                                                      (float *) BLU_data, BLU_i, BLU_j, matBU_info,
                                                      (float *) ytemp_data, (float *) xtemp_data, ilu_solve_policy, ilu_solve_buffer));
         }
         /* - alpha EU^{-1}L^{-1}Fx + beta * y */
         hypre_CSRMatrixMatvec( alpha, EiU, xtemp_upper, beta, y_local);
         /* alpha * C - alpha EU^{-1}L^{-1}Fx + beta y */
         hypre_CSRMatrixMatvec( alpha, C, x_local, one, y_local);

         /* over */
         hypre_SeqVectorDestroy(xtemp_upper);
      }
      break;
      case 0: default:
      {
         /* S = R * A * P */

         hypre_ParCSRMatrix      *Aperm               = hypre_ParILUDataAperm(ilu_data);

         cusparseHandle_t        handle               = hypre_HandleCusparseHandle(hypre_handle());
         cusparseMatDescr_t      matL_des             = hypre_ParILUDataMatLMatrixDescription(ilu_data);
         cusparseMatDescr_t      matU_des             = hypre_ParILUDataMatUMatrixDescription(ilu_data);
         void                    *ilu_solve_buffer    = hypre_ParILUDataILUSolveBuffer(ilu_data);//device memory
         cusparseSolvePolicy_t   ilu_solve_policy     = hypre_ParILUDataILUSolvePolicy(ilu_data);

         HYPRE_Int               isDoublePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double);
         HYPRE_Int               isSinglePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

         hypre_assert(isDoublePrecision || isSinglePrecision);

         hypre_CSRMatrix         *EiU                 = hypre_ParILUDataMatEDevice(ilu_data);
         hypre_CSRMatrix         *iLF                 = hypre_ParILUDataMatFDevice(ilu_data);
         hypre_CSRMatrix         *BLU                 = hypre_ParILUDataMatBILUDevice(ilu_data);
         HYPRE_Int               *BLU_i               = hypre_CSRMatrixI(BLU);
         HYPRE_Int               *BLU_j               = hypre_CSRMatrixJ(BLU);
         HYPRE_Real              *BLU_data            = hypre_CSRMatrixData(BLU);
         HYPRE_Int               BLU_nnz              = hypre_CSRMatrixNumNonzeros(BLU);

         csrsv2Info_t            matBL_info           = hypre_ParILUDataMatBLILUSolveInfo(ilu_data);
         csrsv2Info_t            matBU_info           = hypre_ParILUDataMatBUILUSolveInfo(ilu_data);

         HYPRE_Int               n                    = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(Aperm));
         HYPRE_Int               nLU                  = hypre_CSRMatrixNumRows(BLU);
         HYPRE_Int               m                    = n - nLU;

         hypre_ParVector         *x_vec               = (hypre_ParVector *) x;
         hypre_Vector            *x_local             = hypre_ParVectorLocalVector(x_vec);
         HYPRE_Real              *x_data              = hypre_VectorData(x_local);
         hypre_ParVector         *y_vec               = (hypre_ParVector *) y;
         hypre_Vector            *y_local             = hypre_ParVectorLocalVector(y_vec);
         //HYPRE_Real              *y_data              = hypre_VectorData(y_local);
         hypre_ParVector         *xtemp               = hypre_ParILUDataUTemp(ilu_data);
         hypre_Vector            *xtemp_local         = hypre_ParVectorLocalVector(xtemp);
         HYPRE_Real              *xtemp_data          = hypre_VectorData(xtemp_local);
         hypre_ParVector         *ytemp               = hypre_ParILUDataYTemp(ilu_data);
         hypre_Vector            *ytemp_local         = hypre_ParVectorLocalVector(ytemp);
         HYPRE_Real              *ytemp_data          = hypre_VectorData(ytemp_local);

         hypre_Vector *xtemp_upper           = hypre_SeqVectorCreate(nLU);
         hypre_Vector *ytemp_upper           = hypre_SeqVectorCreate(nLU);
         hypre_Vector *ytemp_lower           = hypre_SeqVectorCreate(m);
         hypre_VectorOwnsData(xtemp_upper)   = 0;
         hypre_VectorOwnsData(ytemp_upper)   = 0;
         hypre_VectorOwnsData(ytemp_lower)   = 0;
         hypre_VectorData(xtemp_upper)       = xtemp_data;
         hypre_VectorData(ytemp_upper)       = ytemp_data;
         hypre_VectorData(ytemp_lower)       = ytemp_data+nLU;
         hypre_SeqVectorInitialize(xtemp_upper);
         hypre_SeqVectorInitialize(ytemp_upper);
         hypre_SeqVectorInitialize(ytemp_lower);

         HYPRE_Real zero = 0.0;
         HYPRE_Real one = 1.0;
         HYPRE_Real mone = -1.0;

         /* first step, compute P*x put in y */
         /* -L^{-1}Fx */
         hypre_CSRMatrixMatvec(mone, iLF, x_local, zero, xtemp_upper);

         /* -U{-1}L^{-1}Fx */
         if(nLU > 0)
         {
            if(isDoublePrecision)
            {
               /* U solve */
               HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         nLU, BLU_nnz, (hypre_double *) &one, matU_des,
                                                         (hypre_double *) BLU_data, BLU_i, BLU_j, matBU_info,
                                                         (hypre_double *) xtemp_data, (hypre_double *) ytemp_data, ilu_solve_policy, ilu_solve_buffer));
            }
            else if(isSinglePrecision)
            {
               /* U solve */
               HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         nLU, BLU_nnz, (float *) &one, matU_des,
                                                         (float *) BLU_data, BLU_i, BLU_j, matBU_info,
                                                         (float *) xtemp_data, (float *) ytemp_data, ilu_solve_policy, ilu_solve_buffer));
            }
         }
         /* now copy data to y_lower */
         hypre_TMemcpy( ytemp_data + nLU, x_data, HYPRE_Real, m, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

         /* second step, compute A*P*x store in xtemp */
         hypre_ParCSRMatrixMatvec( one, Aperm, ytemp, zero, xtemp);

         /* third step, compute R*A*P*x */
         /* copy partial data in */
         hypre_TMemcpy( ytemp_data+nLU, xtemp_data + nLU, HYPRE_Real, m, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

         /* solve L^{-1} */
         if(nLU > 0)
         {
            if(isDoublePrecision)
            {
               /* L solve */
               HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         nLU, BLU_nnz, (hypre_double *) &one, matL_des,
                                                         (hypre_double *) BLU_data, BLU_i, BLU_j, matBL_info,
                                                         (hypre_double *) xtemp_data, (hypre_double *) ytemp_data, ilu_solve_policy, ilu_solve_buffer));
            }
            else if(isSinglePrecision)
            {
               /* L solve */
               HYPRE_CUSPARSE_CALL(cusparseScsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         nLU, BLU_nnz, (float *) &one, matL_des,
                                                         (float *) BLU_data, BLU_i, BLU_j, matBL_info,
                                                         (float *) xtemp_data, (float *) ytemp_data, ilu_solve_policy, ilu_solve_buffer));
            }
         }
         /* -EU^{-1}L^{-1} */
         hypre_CSRMatrixMatvec(mone * alpha, EiU, ytemp_upper, beta, y_local);
         hypre_SeqVectorAxpy(alpha, ytemp_lower, y_local);

         /* over */
         hypre_SeqVectorDestroy(xtemp_upper);
         hypre_SeqVectorDestroy(ytemp_upper);
         hypre_SeqVectorDestroy(ytemp_lower);
      }
      break;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESMatvecDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPSchurGMRESMatvecDestroy( void *matvec_data )
{
   return 0;
}

#else

HYPRE_Int
hypre_ParILURAPSchurGMRESDummySetupH(void *a, void *b, void *c, void *d)
{
   /* Null GMRES setup, does nothing */
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESDummySolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPSchurGMRESSolveH( void               *ilu_vdata,
                  void               *ilu_vdata2,
                  hypre_ParVector    *f,
                  hypre_ParVector    *u )
{
   /* Unit GMRES preconditioner, just copy data from one slot to another */
   hypre_ParILUData        *ilu_data            = (hypre_ParILUData*) ilu_vdata;

   HYPRE_Int               i, j, k1, k2, col;

   hypre_ParCSRMatrix      *L                   = hypre_ParILUDataMatLModified(ilu_data);
   hypre_CSRMatrix         *L_diag              = hypre_ParCSRMatrixDiag(L);
   HYPRE_Int               *L_diag_i            = hypre_CSRMatrixI(L_diag);
   HYPRE_Int               *L_diag_j            = hypre_CSRMatrixJ(L_diag);
   HYPRE_Real              *L_diag_data         = hypre_CSRMatrixData(L_diag);
   HYPRE_Real              *D                   = hypre_ParILUDataMatDModified(ilu_data);
   hypre_ParCSRMatrix      *U                   = hypre_ParILUDataMatUModified(ilu_data);
   hypre_CSRMatrix         *U_diag              = hypre_ParCSRMatrixDiag(U);
   HYPRE_Int               *U_diag_i            = hypre_CSRMatrixI(U_diag);
   HYPRE_Int               *U_diag_j            = hypre_CSRMatrixJ(U_diag);
   HYPRE_Real              *U_diag_data         = hypre_CSRMatrixData(U_diag);

   HYPRE_Int               n                    = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(L));
   HYPRE_Int               nLU                  = hypre_ParILUDataNLU(ilu_data);
   HYPRE_Int               m                    = n - nLU;

   hypre_Vector            *f_local             = hypre_ParVectorLocalVector(f);
   HYPRE_Real              *f_data              = hypre_VectorData(f_local);
   hypre_Vector            *u_local             = hypre_ParVectorLocalVector(u);
   HYPRE_Real              *u_data              = hypre_VectorData(u_local);

   hypre_ParVector         *utemp               = hypre_ParILUDataUTemp(ilu_data);
   hypre_Vector            *utemp_local         = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real              *utemp_data          = hypre_VectorData(utemp_local);

   HYPRE_Int               *u_end               = hypre_ParILUDataUEnd(ilu_data);

   /* permuted L solve */
   for(i = 0 ; i < m ; i ++)
   {
      utemp_data[i] = f_data[i];
      k1 = u_end[i+nLU] ; k2 = L_diag_i[i+nLU+1];
      for(j = k1 ; j < k2 ; j ++)
      {
         col = L_diag_j[j];
         utemp_data[i] -= L_diag_data[j] * utemp_data[col-nLU];
      }
   }

   /* U solve */
   for(i = m-1 ; i >= 0 ; i --)
   {
      u_data[i] = utemp_data[i];
      k1 = U_diag_i[i+nLU] ; k2 = U_diag_i[i+1+nLU];
      for(j = k1 ; j < k2 ; j ++)
      {
         col = U_diag_j[j];
         u_data[i] -= U_diag_data[j] * u_data[col-nLU];
      }
      u_data[i] *= D[i];
   }


   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESCommInfo
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPSchurGMRESCommInfoH( void *ilu_vdata, HYPRE_Int *my_id, HYPRE_Int *num_procs)
{
   /* get comm info from ilu_data */
   hypre_ParILUData *ilu_data = (hypre_ParILUData*) ilu_vdata;
   hypre_ParCSRMatrix *A = hypre_ParILUDataMatA(ilu_data);
   MPI_Comm comm = hypre_ParCSRMatrixComm ( A );
   hypre_MPI_Comm_size(comm,num_procs);
   hypre_MPI_Comm_rank(comm,my_id);
   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESMatvecCreate
 *--------------------------------------------------------------------------*/

void *
hypre_ParILURAPSchurGMRESMatvecCreateH( void   *ilu_vdata,
                             void   *x )
{
   /* Null matvec create */
   void *matvec_data;
   matvec_data = NULL;
   return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESMatvec
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPSchurGMRESMatvecH( void   *matvec_data,
                       HYPRE_Complex  alpha,
                       void   *ilu_vdata,
                       void   *x,
                       HYPRE_Complex  beta,
                       void   *y           )
{
   /* Compute y = alpha * S * x + beta * y
    * */
   HYPRE_Int               i, j, k1, k2, col;
   HYPRE_Real              one = 1.0;
   HYPRE_Real              zero = 0.0;

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

   hypre_ParVector   *utemp       = hypre_ParILUDataUTemp(ilu_data);
   hypre_Vector      *utemp_local = hypre_ParVectorLocalVector(utemp);
   HYPRE_Real        *utemp_data  = hypre_VectorData(utemp_local);

   hypre_ParVector   *ftemp       = hypre_ParILUDataFTemp(ilu_data);
   hypre_Vector      *ftemp_local = hypre_ParVectorLocalVector(ftemp);
   HYPRE_Real        *ftemp_data  = hypre_VectorData(ftemp_local);

   hypre_ParVector   *ytemp       = hypre_ParILUDataYTemp(ilu_data);
   hypre_Vector      *ytemp_local = hypre_ParVectorLocalVector(ytemp);
   HYPRE_Real        *ytemp_data  = hypre_VectorData(ytemp_local);

   /* S = R * A * P */
   /* matvec */
   /* first compute alpha * P * x
    * P = [ -U\inv U_12 ]
    *     [  I          ]
    */
   /* matvec */
   for(i = 0 ; i < nLU ; i ++)
   {
      ytemp_data[i] = 0.0;
      k1 = u_end[i] ; k2 = mU_diag_i[i+1];
      for(j = k1 ; j < k2 ; j ++)
      {
         col = mU_diag_j[j];
         ytemp_data[i] -= alpha * mU_diag_data[j] * x_data[col-nLU];
      }
   }
   /* U solve */
   for(i = nLU-1 ; i >= 0 ; i --)
   {
      ftemp_data[perm[i]] = ytemp_data[i];
      k1 = mU_diag_i[i] ; k2 = u_end[i];
      for(j = k1 ; j < k2 ; j ++)
      {
         col = mU_diag_j[j];
         ftemp_data[perm[i]] -= mU_diag_data[j] * ftemp_data[perm[col]];
      }
      ftemp_data[perm[i]] *= mD[i];
   }

   /* update with I */
   for(i = nLU ; i < n ; i ++)
   {
      ftemp_data[perm[i]] = alpha * x_data[i-nLU];
   }

   /* apply alpha*A*P*x */
   hypre_ParCSRMatrixMatvec(one, A, ftemp, zero, utemp);

   // R = [-L21 L\inv, I]

   /* first is L solve */
   for(i = 0 ; i < nLU ; i ++)
   {
      ytemp_data[i] = utemp_data[perm[i]];
      k1 = mL_diag_i[i] ; k2 = mL_diag_i[i+1];
      for(j = k1 ; j < k2 ; j ++)
      {
         col = mL_diag_j[j];
         ytemp_data[i] -= mL_diag_data[j] * ytemp_data[col];
      }
   }

   /* apply -W * utemp on this, and take care of the I part */
   for(i = nLU ; i < n ; i ++)
   {
      y_data[i - nLU] = beta * y_data[i - nLU] + utemp_data[perm[i]];
      k1 = mL_diag_i[i] ; k2 = u_end[i];
      for(j = k1 ; j < k2 ; j ++)
      {
         col = mL_diag_j[j];
         y_data[i - nLU] -= mL_diag_data[j] * ytemp_data[col];
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUCusparseSchurGMRESMatvecDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPSchurGMRESMatvecDestroyH( void *matvec_data )
{
   return 0;
}

#endif

/* NSH create and solve and help functions */

/* Create */
void *
hypre_NSHCreate()
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
   hypre_ParNSHDataDroptol(nsh_data)               = hypre_TAlloc(HYPRE_Real,2,HYPRE_MEMORY_HOST);
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

/* Destroy */
HYPRE_Int
hypre_NSHDestroy( void *data )
{
   hypre_ParNSHData * nsh_data = (hypre_ParNSHData*) data;

   /* residual */
   if(hypre_ParNSHDataResidual(nsh_data))
   {
      hypre_ParVectorDestroy( hypre_ParNSHDataResidual(nsh_data) );
      hypre_ParNSHDataResidual(nsh_data) = NULL;
   }

   /* residual norms */
   if(hypre_ParNSHDataRelResNorms(nsh_data))
   {
      hypre_TFree( hypre_ParNSHDataRelResNorms(nsh_data), HYPRE_MEMORY_HOST );
      hypre_ParNSHDataRelResNorms(nsh_data) = NULL;
   }

   /* l1 norms */
   if(hypre_ParNSHDataL1Norms(nsh_data))
   {
      hypre_TFree( hypre_ParNSHDataL1Norms(nsh_data), HYPRE_MEMORY_HOST );
      hypre_ParNSHDataL1Norms(nsh_data) = NULL;
   }

   /* temp arrays */
   if(hypre_ParNSHDataUTemp(nsh_data))
   {
      hypre_ParVectorDestroy( hypre_ParNSHDataUTemp(nsh_data) );
      hypre_ParNSHDataUTemp(nsh_data) = NULL;
   }
   if(hypre_ParNSHDataFTemp(nsh_data))
   {
      hypre_ParVectorDestroy( hypre_ParNSHDataFTemp(nsh_data) );
      hypre_ParNSHDataFTemp(nsh_data) = NULL;
   }

   /* approx inverse matrix */
   if(hypre_ParNSHDataMatM(nsh_data))
   {
      hypre_ParCSRMatrixDestroy( hypre_ParNSHDataMatM(nsh_data) );
      hypre_ParNSHDataMatM(nsh_data) = NULL;
   }

   /* droptol array */
  if(hypre_ParNSHDataOwnDroptolData(nsh_data))
  {
     hypre_TFree(hypre_ParNSHDataDroptol(nsh_data), HYPRE_MEMORY_HOST);
     hypre_ParNSHDataOwnDroptolData(nsh_data) = 0;
     hypre_ParNSHDataDroptol(nsh_data) = NULL;
  }

   /* nsh data */
   hypre_TFree(nsh_data, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/* Print solver params */
HYPRE_Int
hypre_NSHWriteSolverParams(void *nsh_vdata)
{
   hypre_ParNSHData  *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_printf("Newton–Schulz–Hotelling Setup parameters: \n");
   hypre_printf("NSH max iterations = %d \n", hypre_ParNSHDataNSHMaxIter(nsh_data));
   hypre_printf("NSH drop tolerance = %e \n", hypre_ParNSHDataDroptol(nsh_data)[1]);
   hypre_printf("NSH max nnz per row = %d \n", hypre_ParNSHDataNSHMaxRowNnz(nsh_data));
   hypre_printf("MR max iterations = %d \n", hypre_ParNSHDataMRMaxIter(nsh_data));
   hypre_printf("MR drop tolerance = %e \n", hypre_ParNSHDataDroptol(nsh_data)[0]);
   hypre_printf("MR max nnz per row = %d \n", hypre_ParNSHDataMRMaxRowNnz(nsh_data));
   hypre_printf("Operator Complexity (Fill factor) = %f \n", hypre_ParNSHDataOperatorComplexity(nsh_data));
   hypre_printf("\n Newton–Schulz–Hotelling Solver Parameters: \n");
   hypre_printf("Max number of iterations: %d\n", hypre_ParNSHDataMaxIter(nsh_data));
   hypre_printf("Stopping tolerance: %e\n", hypre_ParNSHDataTol(nsh_data));

   return hypre_error_flag;
}

/* set print level */
HYPRE_Int
hypre_NSHSetPrintLevel( void *nsh_vdata, HYPRE_Int print_level )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataPrintLevel(nsh_data) = print_level;
   return hypre_error_flag;
}
/* set logging level */
HYPRE_Int
hypre_NSHSetLogging( void *nsh_vdata, HYPRE_Int logging )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataLogging(nsh_data) = logging;
   return hypre_error_flag;
}
/* set max iteration */
HYPRE_Int
hypre_NSHSetMaxIter( void *nsh_vdata, HYPRE_Int max_iter )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMaxIter(nsh_data) = max_iter;
   return hypre_error_flag;
}
/* set solver iteration tol */
HYPRE_Int
hypre_NSHSetTol( void *nsh_vdata, HYPRE_Real tol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataTol(nsh_data) = tol;
   return hypre_error_flag;
}
/* set global solver */
HYPRE_Int
hypre_NSHSetGlobalSolver( void *nsh_vdata, HYPRE_Int global_solver )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataGlobalSolver(nsh_data) = global_solver;
   return hypre_error_flag;
}
/* set all droptols */
HYPRE_Int
hypre_NSHSetDropThreshold( void *nsh_vdata, HYPRE_Real droptol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataDroptol(nsh_data)[0] = droptol;
   hypre_ParNSHDataDroptol(nsh_data)[1] = droptol;
   return hypre_error_flag;
}
/* set array of droptols */
HYPRE_Int
hypre_NSHSetDropThresholdArray( void *nsh_vdata, HYPRE_Real *droptol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   if(hypre_ParNSHDataOwnDroptolData(nsh_data))
   {
      hypre_TFree(hypre_ParNSHDataDroptol(nsh_data),HYPRE_MEMORY_HOST);
      hypre_ParNSHDataOwnDroptolData(nsh_data) = 0;
   }
   hypre_ParNSHDataDroptol(nsh_data) = droptol;
   return hypre_error_flag;
}
/* set MR max iter */
HYPRE_Int
hypre_NSHSetMRMaxIter( void *nsh_vdata, HYPRE_Int mr_max_iter )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMRMaxIter(nsh_data) = mr_max_iter;
   return hypre_error_flag;
}
/* set MR tol */
HYPRE_Int
hypre_NSHSetMRTol( void *nsh_vdata, HYPRE_Real mr_tol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMRTol(nsh_data) = mr_tol;
   return hypre_error_flag;
}
/* set MR max nonzeros of a row */
HYPRE_Int
hypre_NSHSetMRMaxRowNnz( void *nsh_vdata, HYPRE_Int mr_max_row_nnz )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMRMaxRowNnz(nsh_data) = mr_max_row_nnz;
   return hypre_error_flag;
}
/* set MR version, column version or global version */
HYPRE_Int
hypre_NSHSetColVersion( void *nsh_vdata, HYPRE_Int mr_col_version )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataMRColVersion(nsh_data) = mr_col_version;
   return hypre_error_flag;
}
/* set NSH max iter */
HYPRE_Int
hypre_NSHSetNSHMaxIter( void *nsh_vdata, HYPRE_Int nsh_max_iter )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataNSHMaxIter(nsh_data) = nsh_max_iter;
   return hypre_error_flag;
}
/* set NSH tol */
HYPRE_Int
hypre_NSHSetNSHTol( void *nsh_vdata, HYPRE_Real nsh_tol )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataNSHTol(nsh_data) = nsh_tol;
   return hypre_error_flag;
}
/* set NSH max nonzeros of a row */
HYPRE_Int
hypre_NSHSetNSHMaxRowNnz( void *nsh_vdata, HYPRE_Int nsh_max_row_nnz )
{
   hypre_ParNSHData   *nsh_data = (hypre_ParNSHData*) nsh_vdata;
   hypre_ParNSHDataNSHMaxRowNnz(nsh_data) = nsh_max_row_nnz;
   return hypre_error_flag;
}


/* Compute the F norm of CSR matrix
 * A: the target CSR matrix
 * norm_io: output
 */
HYPRE_Int
hypre_CSRMatrixNormFro(hypre_CSRMatrix *A, HYPRE_Real *norm_io)
{
   HYPRE_Real norm = 0.0;
   HYPRE_Real *data = hypre_CSRMatrixData(A);
   HYPRE_Int i,k;
   k = hypre_CSRMatrixNumNonzeros(A);
   /* main loop */
   for(i = 0 ; i < k ; i ++)
   {
      norm += data[i] * data[i];
   }
   *norm_io = sqrt(norm);
   return hypre_error_flag;

}

/* Compute the norm of I-A where I is identity matrix and A is a CSR matrix
 * A: the target CSR matrix
 * norm_io: the output
 */
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
   for(i = 0 ; i < n ; i ++)
   {
      k1 = idx[i];
      k2 = idx[i+1];
      /* check if we have diagonal in A */
      if(k2 > k1)
      {
         if(cols[k1] == i)
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
      for(j = k1 + 1 ; j < k2 ; j ++)
      {
         norm += data[j] * data[j];
      }
   }
   *norm_io = sqrt(norm);
   return hypre_error_flag;
}

/* Compute the F norm of ParCSR matrix
 * A: the target CSR matrix
 */
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
   local_norm += global_norm*global_norm;

   /* do communication to get global total sum */
   hypre_MPI_Allreduce(&local_norm, &global_norm, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);

   *norm_io = sqrt(global_norm);
   return hypre_error_flag;

}

/* Compute the F norm of ParCSR matrix
 * Norm of I-A
 * A: the target CSR matrix
 */
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
   local_norm += global_norm*global_norm;

   /* do communication to get global total sum */
   hypre_MPI_Allreduce(&local_norm, &global_norm, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);

   *norm_io = sqrt(global_norm);
   return hypre_error_flag;

}

/* Compute the trace of CSR matrix
 * A: the target CSR matrix
 * trace_io: the output trace
 */
HYPRE_Int
hypre_CSRMatrixTrace(hypre_CSRMatrix *A, HYPRE_Real *trace_io)
{
   HYPRE_Real  trace = 0.0;
   HYPRE_Int   *idx = hypre_CSRMatrixI(A);
   HYPRE_Int   *cols = hypre_CSRMatrixJ(A);
   HYPRE_Real  *data = hypre_CSRMatrixData(A);
   HYPRE_Int i,k1,k2,n;
   n = hypre_CSRMatrixNumRows(A);
   for(i = 0 ; i < n ; i ++)
   {
      k1 = idx[i];
      k2 = idx[i+1];
      if(cols[k1] == i && k2 > k1)
      {
         /* only add when diagonal is nonzero */
         trace += data[k1];
      }
   }

   *trace_io = trace;
   return hypre_error_flag;

}

/* Apply dropping to CSR matrix
 * A: the target CSR matrix
 * droptol: all entries have smaller absolute value than this will be dropped
 * max_row_nnz: max nonzeros allowed for each row, only largest max_row_nnz kept
 * we NEVER drop diagonal entry if exists
 */
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

   /* new data */
   HYPRE_Int      *new_i;
   HYPRE_Int      *new_j;
   HYPRE_Real     *new_data;

   /* memory */
   HYPRE_Int      capacity;
   HYPRE_Int      ctrA;

   /* setup */
   capacity = nnzA*0.3+1;
   ctrA = 0;
   new_i = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_DEVICE);
   new_j = hypre_TAlloc(HYPRE_Int, capacity, HYPRE_MEMORY_DEVICE);
   new_data = hypre_TAlloc(HYPRE_Real, capacity, HYPRE_MEMORY_DEVICE);

   idx = hypre_TAlloc(HYPRE_Int, m, HYPRE_MEMORY_DEVICE);
   data = hypre_TAlloc(HYPRE_Real, m, HYPRE_MEMORY_DEVICE);

   /* start of main loop */
   new_i[0] = 0;
   for(i = 0 ; i < n ; i ++)
   {
      len = 0;
      k1 = A_i[i];
      k2 = A_i[i+1];
      /* compute droptol for current row */
      norm = 0.0;
      for(j = k1 ; j < k2 ; j ++)
      {
         norm += hypre_abs(A_data[j]);
      }
      if(k2 > k1)
      {
         norm /= (HYPRE_Real)(k2 - k1);
      }
      itol = droptol * norm;
      /* we don't want to drop the diagonal entry, so use an if statement here */
      if(A_j[k1] == i)
      {
         /* we have diagonal entry, skip it */
         idx[len] = A_j[k1];
         data[len++] = A_data[k1];
         for(j = k1 + 1 ; j < k2 ; j ++)
         {
            value = A_data[j];
            if(hypre_abs(value) < itol)
            {
               /* skip small element */
               continue;
            }
            idx[len] = A_j[j];
            data[len++] = A_data[j];
         }

         /* now apply drop on length */
         if(len > max_row_nnz)
         {
            drop_len = max_row_nnz;
            hypre_ILUMaxQSplitRabsI( data + 1, idx + 1, 0, drop_len - 1 , len - 2);
         }
         else
         {
            /* don't need to sort, we keep all of them */
            drop_len = len;
         }
         /* copy data */
         while(ctrA + drop_len > capacity)
         {
            HYPRE_Int tmp = capacity;
            capacity = capacity * EXPAND_FACT + 1;
            new_j = hypre_TReAlloc_v2(new_j, HYPRE_Int, tmp, HYPRE_Int, capacity, HYPRE_MEMORY_DEVICE);
            new_data = hypre_TReAlloc_v2(new_data, HYPRE_Real, tmp, HYPRE_Real, capacity, HYPRE_MEMORY_DEVICE);
         }
         hypre_TMemcpy( new_j + ctrA, idx,HYPRE_Int, drop_len, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         hypre_TMemcpy( new_data + ctrA, data,HYPRE_Real, drop_len, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         ctrA += drop_len;
         new_i[i+1] = ctrA;
      }
      else
      {
         /* we don't have diagonal entry */
         for(j = k1 ; j < k2 ; j ++)
         {
            value = A_data[j];
            if(hypre_abs(value)<itol)
            {
               /* skip small element */
               continue;
            }
            idx[len] = A_j[j];
            data[len++] = A_data[j];
         }

         /* now apply drop on length */
         if(len > max_row_nnz)
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
         while(ctrA + drop_len > capacity)
         {
            HYPRE_Int tmp = capacity;
            capacity = capacity * EXPAND_FACT + 1;
            new_j = hypre_TReAlloc_v2(new_j, HYPRE_Int, tmp, HYPRE_Int, capacity, HYPRE_MEMORY_DEVICE);
            new_data = hypre_TReAlloc_v2(new_data, HYPRE_Real, tmp, HYPRE_Real, capacity, HYPRE_MEMORY_DEVICE);
         }
         hypre_TMemcpy( new_j + ctrA, idx,HYPRE_Int, drop_len, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         hypre_TMemcpy( new_data + ctrA, data,HYPRE_Real, drop_len, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
         ctrA += drop_len;
         new_i[i+1] = ctrA;
      }
   }/* end of main loop */
   /* destory data if A own them */
   if(hypre_CSRMatrixOwnsData(A))
   {
      hypre_TFree(A_i, HYPRE_MEMORY_DEVICE);
      hypre_TFree(A_j, HYPRE_MEMORY_DEVICE);
      hypre_TFree(A_data, HYPRE_MEMORY_DEVICE);
   }

   hypre_CSRMatrixI(A) = new_i;
   hypre_CSRMatrixJ(A) = new_j;
   hypre_CSRMatrixData(A) = new_data;
   hypre_CSRMatrixNumNonzeros(A) = ctrA;
   hypre_CSRMatrixOwnsData(A) = 1;

   hypre_TFree(idx, HYPRE_MEMORY_DEVICE);
   hypre_TFree(data, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

/* Compute the inverse with MR of original CSR matrix
 * Global(not by each column) and out place version
 * A: the input matrix
 * M: the output matrix
 * droptol: the dropping tolorance
 * tol: when to stop the iteration
 * eps_tol: to avoid divide by 0
 * max_row_nnz: max number of nonzeros per row
 * max_iter: max number of iterations
 * print_level: the print level of this algorithm
 */
HYPRE_Int
hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal(hypre_CSRMatrix *matA, hypre_CSRMatrix **M, HYPRE_Real droptol,
                                               HYPRE_Real tol, HYPRE_Real eps_tol, HYPRE_Int max_row_nnz, HYPRE_Int max_iter,
                                               HYPRE_Int print_level )
{
   HYPRE_Int         i, k1, k2;
   HYPRE_Real        value, trace1, trace2, alpha, r_norm;

   /* martix A */
   HYPRE_Int         *A_i = hypre_CSRMatrixI(matA);
   HYPRE_Int         *A_j = hypre_CSRMatrixJ(matA);
   HYPRE_Real        *A_data = hypre_CSRMatrixData(matA);

   /* complexity */
   HYPRE_Real        nnzA = hypre_CSRMatrixNumNonzeros(matA);
   HYPRE_Real        nnzM;

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

   HYPRE_Real        time_s, time_e;

   HYPRE_Int         n = hypre_CSRMatrixNumRows(matA);

   /* create initial guess and matrix I */
   matM = hypre_CSRMatrixCreate(n,n,n);
   M_i = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_DEVICE);
   M_j = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
   M_data = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_DEVICE);

   matI = hypre_CSRMatrixCreate(n,n,n);
   I_i = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_DEVICE);
   I_j = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
   I_data = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_DEVICE);

   /* now loop to create initial guess */
   M_i[0] = 0;
   I_i[0] = 0;
   for(i = 0 ; i < n ; i ++)
   {
      M_i[i+1] = i+1;
      M_j[i] = i;
      k1 = A_i[i];
      k2 = A_i[i+1];
      if(k2 > k1)
      {
         if(A_j[k1] == i)
         {
            value = A_data[k1];
            if(hypre_abs(value) < MAT_TOL)
            {
               value = 1.0;
            }
            M_data[i] = 1.0/value;
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
      I_i[i+1] = i+1;
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
   if(print_level > 1)
   {
      /* time the iteration */
      time_s = hypre_MPI_Wtime();
   }

   /* main loop */
   for(i = 0 ; i < max_iter ; i ++)
   {
      nnzM = hypre_CSRMatrixNumNonzeros(matM);
      /* R = I - AM */
      matR_temp = hypre_CSRMatrixMultiply(matA,matM);

      hypre_CSRMatrixScale(matR_temp, -1.0);

      matR = hypre_CSRMatrixAdd(1.0, matI, 1.0, matR_temp);
      hypre_CSRMatrixDestroy(matR_temp);

      /* r_norm */
      hypre_CSRMatrixNormFro(matR, &r_norm);
      if(r_norm < tol)
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
      hypre_CSRMatrixTranspose(matR,&matR_temp,1);
      matW = hypre_CSRMatrixMultiply(matR_temp,matC);

      /* trace and alpha */
      hypre_CSRMatrixTrace(matW, &trace1);
      hypre_CSRMatrixNormFro(matC, &trace2);
      trace2 *= trace2;

      if(hypre_abs(trace2) < eps_tol)
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

   }/* end of main loop i for compute inverse matrix */

   /* time if we need to print */
   if(print_level > 1)
   {
      time_e = hypre_MPI_Wtime();
      if(i == 0)
      {
         i = 1;
      }
      hypre_printf("matrix size %5d\nfinal norm at loop %5d is %16.12f, time per iteration is %16.12f, complexity is %16.12f out of maximum %16.12f\n",n,i,r_norm, (time_e-time_s)/i, nnzM/nnzA, n/nnzA*n);
   }

   hypre_CSRMatrixDestroy(matI);
   if(inM)
   {
      hypre_CSRMatrixDestroy(inM);
   }
   *M = matM;

   return hypre_error_flag;

}

/* Compute inverse with NSH method
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
 */
HYPRE_Int
hypre_ILUParCSRInverseNSH(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **M, HYPRE_Real *droptol, HYPRE_Real mr_tol,
                            HYPRE_Real nsh_tol, HYPRE_Real eps_tol, HYPRE_Int mr_max_row_nnz, HYPRE_Int nsh_max_row_nnz,
                            HYPRE_Int mr_max_iter, HYPRE_Int nsh_max_iter, HYPRE_Int mr_col_version,
                            HYPRE_Int print_level)
{
   HYPRE_Int               i;

   /* data slots for matrices */
   hypre_ParCSRMatrix      *matM = NULL;
   hypre_ParCSRMatrix      *inM = *M;
   hypre_ParCSRMatrix      *AM,*MAM;
   HYPRE_Real              norm, s_norm;
   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int               myid;


   hypre_CSRMatrix         *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix         *M_diag = NULL;
   hypre_CSRMatrix         *M_offd;
   HYPRE_Int               *M_offd_i;

   HYPRE_Real              time_s, time_e;

   HYPRE_Int               n = hypre_CSRMatrixNumRows(A_diag);

   /* setup */
   hypre_MPI_Comm_rank(comm, &myid);

   M_offd_i = hypre_CTAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_DEVICE);

   if(mr_col_version)
   {
      hypre_printf("Column version is not yet support, switch to global version\n");
   }

   /* call MR to build loacl initial matrix
    * droptol here should be larger
    * we want same number for MR and NSH to let user set them eaiser
    * but we don't want a too dense MR initial guess
    */
   hypre_ILUCSRMatrixInverseSelfPrecondMRGlobal(A_diag, &M_diag, droptol[0] * 10.0, mr_tol, eps_tol, mr_max_row_nnz, mr_max_iter, print_level );

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
   if(print_level > 1)
   {
      if(myid == 0)
      {
         hypre_printf("before NSH the norm is %16.12f\n", norm);
      }
      time_s = hypre_MPI_Wtime();
   }

   for(i = 0 ; i < nsh_max_iter ; i ++)
   {
      /* compute XjAXj */
      AM = hypre_ParMatmul(A, matM);
      hypre_ParCSRMatrixResNormFro(AM, &norm);
      if(norm < nsh_tol)
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

   if(print_level > 1)
   {
      time_e = hypre_MPI_Wtime();
      /* at this point of time, norm has to be already computed */
      if(i == 0)
      {
         i = 1;
      }
      if(myid == 0)
      {
         hypre_printf("after %5d NSH iterations the norm is %16.12f, time per iteration is %16.12f\n", i, norm, (time_e-time_s)/i);
      }
   }

   if(s_norm < norm)
   {
      /* the residual norm increase after NSH iteration, need to let user know */
      if(myid == 0)
      {
         hypre_printf("Warning: NSH divergence, probably bad approximate invese matrix.\n");
      }
   }

   if(inM)
   {
      hypre_ParCSRMatrixDestroy(inM);
   }
   *M = matM;

   return hypre_error_flag;
}
