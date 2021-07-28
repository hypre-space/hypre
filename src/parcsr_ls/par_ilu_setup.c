/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"
#include "par_ilu.h"
#include "seq_mv.hpp"

/* Setup ILU data */
HYPRE_Int
hypre_ILUSetup( void               *ilu_vdata,
                hypre_ParCSRMatrix *A,
                hypre_ParVector    *f,
                hypre_ParVector    *u )
{
   MPI_Comm             comm                 = hypre_ParCSRMatrixComm(A);
   hypre_ParILUData     *ilu_data            = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUData     *schur_precond_ilu;
   hypre_ParNSHData     *schur_solver_nsh;

   HYPRE_Int            i;
   // HYPRE_Int            num_threads;
   // HYPRE_Int            debug_flag           = 0;

   /* pointers to ilu data */
   HYPRE_Int            logging              = hypre_ParILUDataLogging(ilu_data);
   HYPRE_Int            print_level          = hypre_ParILUDataPrintLevel(ilu_data);
   HYPRE_Int            ilu_type             = hypre_ParILUDataIluType(ilu_data);
   HYPRE_Int            nLU                  = hypre_ParILUDataNLU(ilu_data);
   HYPRE_Int            nI                   = hypre_ParILUDataNI(ilu_data);
   HYPRE_Int            fill_level           = hypre_ParILUDataLfil(ilu_data);
   HYPRE_Int            max_row_elmts        = hypre_ParILUDataMaxRowNnz(ilu_data);
   HYPRE_Real           *droptol             = hypre_ParILUDataDroptol(ilu_data);
   HYPRE_Int            *CF_marker_array     = hypre_ParILUDataCFMarkerArray(ilu_data);
   HYPRE_Int            *perm                = hypre_ParILUDataPerm(ilu_data);
   HYPRE_Int            *qperm               = hypre_ParILUDataQPerm(ilu_data);
   HYPRE_Real           tol_ddPQ             = hypre_ParILUDataTolDDPQ(ilu_data);

#ifdef HYPRE_USING_CUDA
   /* pointers to cusparse data, note that they are not NULL only when needed */
   cusparseMatDescr_t      matL_des          = hypre_ParILUDataMatLMatrixDescription(ilu_data);
   cusparseMatDescr_t      matU_des          = hypre_ParILUDataMatUMatrixDescription(ilu_data);
   void                    *ilu_solve_buffer = hypre_ParILUDataILUSolveBuffer(ilu_data);//device memory
   cusparseSolvePolicy_t   ilu_solve_policy  = hypre_ParILUDataILUSolvePolicy(ilu_data);
   hypre_ParCSRMatrix      *Aperm            = hypre_ParILUDataAperm(ilu_data);
   hypre_ParCSRMatrix      *R                = hypre_ParILUDataR(ilu_data);
   hypre_ParCSRMatrix      *P                = hypre_ParILUDataP(ilu_data);
   hypre_CSRMatrix         *matALU_d         = hypre_ParILUDataMatAILUDevice(ilu_data);
   hypre_CSRMatrix         *matBLU_d         = hypre_ParILUDataMatBILUDevice(ilu_data);
   hypre_CSRMatrix         *matSLU_d         = hypre_ParILUDataMatSILUDevice(ilu_data);
   hypre_CSRMatrix         *matE_d           = hypre_ParILUDataMatEDevice(ilu_data);
   hypre_CSRMatrix         *matF_d           = hypre_ParILUDataMatFDevice(ilu_data);
   csrsv2Info_t            matAL_info        = hypre_ParILUDataMatALILUSolveInfo(ilu_data);
   csrsv2Info_t            matAU_info        = hypre_ParILUDataMatAUILUSolveInfo(ilu_data);
   csrsv2Info_t            matBL_info        = hypre_ParILUDataMatBLILUSolveInfo(ilu_data);
   csrsv2Info_t            matBU_info        = hypre_ParILUDataMatBUILUSolveInfo(ilu_data);
   csrsv2Info_t            matSL_info        = hypre_ParILUDataMatSLILUSolveInfo(ilu_data);
   csrsv2Info_t            matSU_info        = hypre_ParILUDataMatSUILUSolveInfo(ilu_data);
   HYPRE_Int               *A_diag_fake      = hypre_ParILUDataMatAFakeDiagonal(ilu_data);
   hypre_Vector            *Ftemp_upper      = NULL;
   hypre_Vector            *Utemp_lower      = NULL;
#endif

   hypre_ParCSRMatrix   *matA                = hypre_ParILUDataMatA(ilu_data);
   hypre_ParCSRMatrix   *matL                = hypre_ParILUDataMatL(ilu_data);
   HYPRE_Real           *matD                = hypre_ParILUDataMatD(ilu_data);
   hypre_ParCSRMatrix   *matU                = hypre_ParILUDataMatU(ilu_data);
   hypre_ParCSRMatrix   *matmL               = hypre_ParILUDataMatLModified(ilu_data);
   HYPRE_Real           *matmD               = hypre_ParILUDataMatDModified(ilu_data);
   hypre_ParCSRMatrix   *matmU               = hypre_ParILUDataMatUModified(ilu_data);
   hypre_ParCSRMatrix   *matS                = hypre_ParILUDataMatS(ilu_data);
//   hypre_ParCSRMatrix   *matM                = NULL;
//   HYPRE_Int            nnzG;/* g stands for global */
   HYPRE_Real           nnzS;/* total nnz in S */
   HYPRE_Int            nnzS_offd;
   HYPRE_Int            size_C/* total size of coarse grid */;

   HYPRE_Int            n                    = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   //   HYPRE_Int            m;/* m = n-LU */
   /* reordering option */
   HYPRE_Int            reordering_type = hypre_ParILUDataReorderingType(ilu_data);
   HYPRE_Int            num_procs,  my_id;

   hypre_ParVector      *Utemp               = NULL;
   hypre_ParVector      *Ftemp               = NULL;
   hypre_ParVector      *Xtemp               = NULL;
   hypre_ParVector      *Ytemp               = NULL;
   HYPRE_Real           *uext                = NULL;
   HYPRE_Real           *fext                = NULL;
   hypre_ParVector      *rhs                 = NULL;
   hypre_ParVector      *x                   = NULL;
   hypre_ParVector      *F_array             = hypre_ParILUDataF(ilu_data);
   hypre_ParVector      *U_array             = hypre_ParILUDataU(ilu_data);
   hypre_ParVector      *residual            = hypre_ParILUDataResidual(ilu_data);
   HYPRE_Real           *rel_res_norms       = hypre_ParILUDataRelResNorms(ilu_data);

   /* might need for Schur Complement */
   HYPRE_Int            *u_end               = NULL;
   HYPRE_Solver         schur_solver         = NULL;
   HYPRE_Solver         schur_precond        = NULL;
   HYPRE_Solver         schur_precond_gotten = NULL;

   /* help to build external */
   hypre_ParCSRCommPkg  *comm_pkg;
   HYPRE_Int            buffer_size;
   HYPRE_Int            send_size;
   HYPRE_Int            recv_size;
#ifdef HYPRE_USING_CUDA
   HYPRE_Int            test_opt;
#endif
   /* ----- begin -----*/
   HYPRE_ANNOTATE_FUNC_BEGIN;

   //num_threads = hypre_NumThreads();

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

#ifdef HYPRE_USING_CUDA
   /* create cuda and cusparse information when needed */
   /* Use most of them from global information */
   /* set matrix L descripter, L is a lower triangular matrix with unit diagonal entries */
   if (!matL_des)
   {
      HYPRE_CUSPARSE_CALL(cusparseCreateMatDescr(&(hypre_ParILUDataMatLMatrixDescription(ilu_data))));
      matL_des = hypre_ParILUDataMatLMatrixDescription(ilu_data);
      HYPRE_CUSPARSE_CALL(cusparseSetMatIndexBase(matL_des, CUSPARSE_INDEX_BASE_ZERO));
      HYPRE_CUSPARSE_CALL(cusparseSetMatType(matL_des, CUSPARSE_MATRIX_TYPE_GENERAL));
      HYPRE_CUSPARSE_CALL(cusparseSetMatFillMode(matL_des, CUSPARSE_FILL_MODE_LOWER));
      HYPRE_CUSPARSE_CALL(cusparseSetMatDiagType(matL_des, CUSPARSE_DIAG_TYPE_UNIT));
   }
   /* set matrix U descripter, U is a upper triangular matrix with non-unit diagonal entries */
   if (!matU_des)
   {
      HYPRE_CUSPARSE_CALL(cusparseCreateMatDescr(&(hypre_ParILUDataMatUMatrixDescription(ilu_data))));
      matU_des = hypre_ParILUDataMatUMatrixDescription(ilu_data);
      HYPRE_CUSPARSE_CALL(cusparseSetMatIndexBase(matU_des, CUSPARSE_INDEX_BASE_ZERO));
      HYPRE_CUSPARSE_CALL(cusparseSetMatType(matU_des, CUSPARSE_MATRIX_TYPE_GENERAL));
      HYPRE_CUSPARSE_CALL(cusparseSetMatFillMode(matU_des, CUSPARSE_FILL_MODE_UPPER));
      HYPRE_CUSPARSE_CALL(cusparseSetMatDiagType(matU_des, CUSPARSE_DIAG_TYPE_NON_UNIT));
   }
   if (!matAL_info)
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(hypre_ParILUDataMatALILUSolveInfo(ilu_data))) );
      matAL_info = NULL;
   }
   if (!matAU_info)
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(hypre_ParILUDataMatAUILUSolveInfo(ilu_data))) );
      matAU_info = NULL;
   }
   if (!matBL_info)
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(hypre_ParILUDataMatBLILUSolveInfo(ilu_data))) );
      matBL_info = NULL;
   }
   if (!matBU_info)
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(hypre_ParILUDataMatBUILUSolveInfo(ilu_data))) );
      matBU_info = NULL;
   }
   if (!matSL_info)
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(hypre_ParILUDataMatSLILUSolveInfo(ilu_data))) );
      matSL_info = NULL;
   }
   if (!matSU_info)
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info(hypre_ParILUDataMatSUILUSolveInfo(ilu_data))) );
      matSU_info = NULL;
   }
   if (ilu_solve_buffer)
   {
      hypre_TFree(ilu_solve_buffer, HYPRE_MEMORY_DEVICE);
      ilu_solve_buffer = NULL;
   }
   if (matALU_d)
   {
      hypre_CSRMatrixDestroy( matALU_d );
      matALU_d = NULL;
   }
   if (matSLU_d)
   {
      hypre_CSRMatrixDestroy( matSLU_d );
      matSLU_d = NULL;
   }
   if (matBLU_d)
   {
      hypre_CSRMatrixDestroy( matBLU_d );
      matBLU_d = NULL;
   }
   if (matE_d)
   {
      hypre_CSRMatrixDestroy( matE_d );
      matE_d = NULL;
   }
   if (matF_d)
   {
      hypre_CSRMatrixDestroy( matF_d );
      matF_d = NULL;
   }
   if (Aperm)
   {
      hypre_ParCSRMatrixDestroy( Aperm );
      Aperm = NULL;
   }
   if (R)
   {
      hypre_ParCSRMatrixDestroy( R );
      R = NULL;
   }
   if (P)
   {
      hypre_ParCSRMatrixDestroy( P );
      P = NULL;
   }
   if (hypre_ParILUDataXTemp(ilu_data))
   {
      hypre_ParVectorDestroy(hypre_ParILUDataXTemp(ilu_data));
      hypre_ParILUDataXTemp(ilu_data) = NULL;
   }
   if (hypre_ParILUDataYTemp(ilu_data))
   {
      hypre_ParVectorDestroy(hypre_ParILUDataYTemp(ilu_data));
      hypre_ParILUDataYTemp(ilu_data) = NULL;
   }
   if (hypre_ParILUDataFTempUpper(ilu_data))
   {
      hypre_SeqVectorDestroy(hypre_ParILUDataFTempUpper(ilu_data));
      hypre_ParILUDataFTempUpper(ilu_data) = NULL;
   }
   if (hypre_ParILUDataUTempLower(ilu_data))
   {
      hypre_SeqVectorDestroy(hypre_ParILUDataUTempLower(ilu_data));
      hypre_ParILUDataUTempLower(ilu_data) = NULL;
   }
   if (hypre_ParILUDataMatAFakeDiagonal(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataMatAFakeDiagonal(ilu_data), HYPRE_MEMORY_DEVICE);
      hypre_ParILUDataMatAFakeDiagonal(ilu_data) = NULL;
   }
#endif

   /* Free Previously allocated data, if any not destroyed */
   if (matL)
   {
      hypre_ParCSRMatrixDestroy(matL);
      matL = NULL;
   }
   if (matU)
   {
      hypre_ParCSRMatrixDestroy(matU);
      matU = NULL;
   }
   if (matmL)
   {
       hypre_ParCSRMatrixDestroy(matmL);
       matmL = NULL;
   }
   if (matmU)
   {
      hypre_ParCSRMatrixDestroy(matmU);
      matmU = NULL;
   }
   if (matS)
   {
      hypre_ParCSRMatrixDestroy(matS);
      matS = NULL;
   }
   if (matD)
   {
      hypre_TFree(matD, HYPRE_MEMORY_DEVICE);
      matD = NULL;
   }
   if (matmD)
   {
      hypre_TFree(matmD, HYPRE_MEMORY_DEVICE);
      matmD = NULL;
   }
   if (CF_marker_array)
   {
      hypre_TFree(CF_marker_array, HYPRE_MEMORY_HOST);
      CF_marker_array = NULL;
   }


   /* clear old l1_norm data, if created */
   if (hypre_ParILUDataL1Norms(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataL1Norms(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataL1Norms(ilu_data) = NULL;
   }

   /* setup temporary storage
    * first check is they've already here
    */
   if (hypre_ParILUDataUTemp(ilu_data))
   {
      hypre_ParVectorDestroy(hypre_ParILUDataUTemp(ilu_data));
      hypre_ParILUDataUTemp(ilu_data) = NULL;
   }
   if (hypre_ParILUDataFTemp(ilu_data))
   {
      hypre_ParVectorDestroy(hypre_ParILUDataFTemp(ilu_data));
      hypre_ParILUDataFTemp(ilu_data) = NULL;
   }
   if (hypre_ParILUDataUExt(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataUExt(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataUExt(ilu_data) = NULL;
   }
   if ( hypre_ParILUDataFExt(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataFExt(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataFExt(ilu_data) = NULL;
   }
   if ( hypre_ParILUDataUEnd(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataUEnd(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataUEnd(ilu_data) = NULL;
   }
   if (hypre_ParILUDataRhs(ilu_data))
   {
      hypre_ParVectorDestroy(hypre_ParILUDataRhs(ilu_data));
      hypre_ParILUDataRhs(ilu_data) = NULL;
   }
   if (hypre_ParILUDataX(ilu_data))
   {
      hypre_ParVectorDestroy(hypre_ParILUDataX(ilu_data));
      hypre_ParILUDataX(ilu_data) = NULL;
   }
   if (hypre_ParILUDataResidual(ilu_data))
   {
      hypre_ParVectorDestroy(hypre_ParILUDataResidual(ilu_data));
      hypre_ParILUDataResidual(ilu_data) = NULL;
   }
   if (hypre_ParILUDataRelResNorms(ilu_data))
   {
      hypre_TFree(hypre_ParILUDataRelResNorms(ilu_data), HYPRE_MEMORY_HOST);
      hypre_ParILUDataRelResNorms(ilu_data) = NULL;
   }
   if (hypre_ParILUDataSchurSolver(ilu_data))
   {
      switch(ilu_type){
         case 10: case 11: case 40: case 41: case 50:
            HYPRE_ParCSRGMRESDestroy(hypre_ParILUDataSchurSolver(ilu_data)); //GMRES for Schur
            break;
         case 20: case 21:
            hypre_NSHDestroy(hypre_ParILUDataSchurSolver(ilu_data)); //NSH for Schur
            break;
         default:
            break;
      }
      (hypre_ParILUDataSchurSolver(ilu_data)) = NULL;
   }
   if (hypre_ParILUDataSchurPrecond(ilu_data))
   {
      switch(ilu_type){
         case 10: case 11: case 40: case 41:
#ifdef HYPRE_USING_CUDA
         if (hypre_ParILUDataIluType(ilu_data) != 10 &&
            hypre_ParILUDataIluType(ilu_data) != 11)
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
      (hypre_ParILUDataSchurPrecond(ilu_data)) = NULL;
   }
   /* start to create working vectors */
   Utemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Utemp);
   hypre_ParILUDataUTemp(ilu_data) = Utemp;

   Ftemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Ftemp);
   hypre_ParILUDataFTemp(ilu_data) = Ftemp;
   /* set matrix, solution and rhs pointers */
   matA = A;
   F_array = f;
   U_array = u;

   // create perm arary if necessary
   if (perm == NULL)
   {
      switch(ilu_type)
      {
         case 10: case 11: case 20: case 21: case 30: case 31: case 50:/* symmetric */
            hypre_ILUGetInteriorExteriorPerm(matA, &perm, &nLU, reordering_type);
            break;
         case 40: case 41:/* ddPQ */
            hypre_ILUGetPermddPQ(matA, &perm, &qperm, tol_ddPQ, &nLU, &nI, reordering_type);
            break;
         case 0: case 1:
            hypre_ILUGetLocalPerm(matA, &perm, &nLU, reordering_type);
            break;
         default:
            hypre_ILUGetLocalPerm(matA, &perm, &nLU, reordering_type);
            break;
      }
   }
   //   m = n - nLU;
   /* factorization */
   switch(ilu_type)
   {
      case 0:
#ifdef HYPRE_USING_CUDA
               /* only apply the setup of ILU0 with cusparse */
               if (fill_level == 0)
               {
                  hypre_ILUSetupILU0Device(matA, perm, perm, n, n, matL_des, matU_des, ilu_solve_policy, &ilu_solve_buffer,
                                                         &matBL_info, &matBU_info, &matSL_info, &matSU_info, &matBLU_d, &matS,
                                                         &matE_d, &matF_d, &A_diag_fake);//BJ + cusparse_ilu0()
               }
               else
               {
                  hypre_ILUSetupILUKDevice(matA, fill_level, perm, perm, n, n, matL_des, matU_des, ilu_solve_policy, &ilu_solve_buffer,
                                                         &matBL_info, &matBU_info, &matSL_info, &matSU_info, &matBLU_d, &matS,
                                                         &matE_d, &matF_d, &A_diag_fake);//BJ + hypre_iluk(), setup the device solve
               }
#else
               hypre_ILUSetupILUK(matA, fill_level, perm, perm, n, n, &matL, &matD, &matU, &matS, &u_end); //BJ + hypre_iluk()
#endif
               break;
      case 1:
#ifdef HYPRE_USING_CUDA
               hypre_ILUSetupILUTDevice(matA, max_row_elmts, droptol, perm, perm, n, n, matL_des, matU_des, ilu_solve_policy, &ilu_solve_buffer,
                                                         &matBL_info, &matBU_info, &matSL_info, &matSU_info, &matBLU_d, &matS,
                                                         &matE_d, &matF_d, &A_diag_fake);//BJ + hypre_ilut(), setup the device solve
#else
               hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, perm, n, n, &matL, &matD, &matU, &matS, &u_end); //BJ + hypre_ilut()
#endif
               break;
      case 10:
#ifdef HYPRE_USING_CUDA
               if (fill_level == 0)
               {
                  /* Only support ILU0 */
                  hypre_ILUSetupILU0Device(matA, perm, perm, n, nLU, matL_des, matU_des, ilu_solve_policy, &ilu_solve_buffer,
                                                         &matBL_info, &matBU_info, &matSL_info, &matSU_info, &matBLU_d, &matS,
                                                         &matE_d, &matF_d, &A_diag_fake);//BJ + cusparse_ilu0()
               }
               else
               {
                  hypre_ILUSetupILUKDevice(matA, fill_level, perm, perm, n, nLU, matL_des, matU_des, ilu_solve_policy, &ilu_solve_buffer,
                                                         &matBL_info, &matBU_info, &matSL_info, &matSU_info, &matBLU_d, &matS,
                                                         &matE_d, &matF_d, &A_diag_fake);//BJ + cusparse_ilu0()
               }
#else
               hypre_ILUSetupILUK(matA, fill_level, perm, perm, nLU, nLU, &matL, &matD, &matU, &matS, &u_end); //GMRES + hypre_iluk()
#endif
               break;
      case 11:
#ifdef HYPRE_USING_CUDA
               hypre_ILUSetupILUTDevice(matA, max_row_elmts, droptol, perm, perm, n, nLU, matL_des, matU_des, ilu_solve_policy, &ilu_solve_buffer,
                                                         &matBL_info, &matBU_info, &matSL_info, &matSU_info, &matBLU_d, &matS,
                                                         &matE_d, &matF_d, &A_diag_fake);//BJ + cusparse_ilu0()
#else
               hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, perm, nLU, nLU, &matL, &matD, &matU, &matS, &u_end); //GMRES + hypre_ilut()
#endif
               break;
      case 20: hypre_ILUSetupILUK(matA, fill_level, perm, perm, nLU, nLU, &matL, &matD, &matU, &matS, &u_end); //Newton Schulz Hotelling + hypre_iluk()
               break;
      case 21: hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, perm, nLU, nLU, &matL, &matD, &matU, &matS, &u_end); //Newton Schulz Hotelling + hypre_ilut()
               break;
      case 30: hypre_ILUSetupILUKRAS(matA, fill_level, perm, nLU, &matL, &matD, &matU); //RAS + hypre_iluk()
               break;
      case 31: hypre_ILUSetupILUTRAS(matA, max_row_elmts, droptol, perm, nLU, &matL, &matD, &matU); //RAS + hypre_ilut()
               break;
      case 40: hypre_ILUSetupILUK(matA, fill_level, perm, qperm, nLU, nI, &matL, &matD, &matU, &matS, &u_end); //ddPQ + GMRES + hypre_iluk()
               break;
      case 41: hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, qperm, nLU, nI, &matL, &matD, &matU, &matS, &u_end); //ddPQ + GMRES + hypre_ilut()
               break;
      case 50:
#ifdef HYPRE_USING_CUDA
               test_opt = hypre_ParILUDataTestOption(ilu_data);
               hypre_ILUSetupRAPILU0Device(matA, perm, n, nLU, matL_des, matU_des, ilu_solve_policy,
                              &ilu_solve_buffer, &matAL_info, &matAU_info, &matBL_info, &matBU_info, &matSL_info, &matSU_info,
                              &Aperm, &matS, &matALU_d, &matBLU_d, &matSLU_d, &matE_d, &matF_d, test_opt); //RAP + hypre_modified_ilu0
#else
               hypre_ILUSetupRAPILU0(matA, perm, n, nLU, &matL, &matD, &matU, &matmL, &matmD, &matmU, &u_end); //RAP + hypre_modified_ilu0
#endif
               break;
      default:
#ifdef HYPRE_USING_CUDA
               hypre_ILUSetupILU0Device(matA, perm, perm, n, n, matL_des, matU_des, ilu_solve_policy, &ilu_solve_buffer,
                                                      &matBL_info, &matBU_info, &matSL_info, &matSU_info, &matBLU_d, &matS,
                                                      &matE_d, &matF_d, &A_diag_fake);//BJ + cusparse_ilu0()
#else
               hypre_ILUSetupILU0(matA, perm, perm, n, n, &matL, &matD, &matU, &matS, &u_end);//BJ + hypre_ilu0()
#endif
               break;
   }
   /* setup Schur solver */
   switch(ilu_type)
   {
      case 10: case 11:
         if (matS)
         {
#ifdef HYPRE_USING_CUDA
            /* create working vectors */

            Xtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matS),
                                   hypre_ParCSRMatrixGlobalNumRows(matS),
                                   hypre_ParCSRMatrixRowStarts(matS));
            hypre_ParVectorInitialize(Xtemp);

            Ytemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matS),
                                   hypre_ParCSRMatrixGlobalNumRows(matS),
                                   hypre_ParCSRMatrixRowStarts(matS));
            hypre_ParVectorInitialize(Ytemp);

            Ftemp_upper = hypre_SeqVectorCreate(nLU);
            hypre_VectorOwnsData(Ftemp_upper)   = 0;
            hypre_VectorData(Ftemp_upper)       = hypre_VectorData(hypre_ParVectorLocalVector(Ftemp));
            hypre_SeqVectorInitialize(Ftemp_upper);

            Utemp_lower = hypre_SeqVectorCreate(n - nLU);
            hypre_VectorOwnsData(Utemp_lower)   = 0;
            hypre_VectorData(Utemp_lower)       = hypre_VectorData(hypre_ParVectorLocalVector(Utemp)) + nLU;
            hypre_SeqVectorInitialize(Utemp_lower);

            /* create GMRES */
//            HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

            hypre_GMRESFunctions * gmres_functions;

            gmres_functions =
               hypre_GMRESFunctionsCreate(
                  hypre_CAlloc,
                  hypre_ParKrylovFree,
                  hypre_ParILUCusparseSchurGMRESCommInfo, //parCSR A -> ilu_data
                  hypre_ParKrylovCreateVector,
                  hypre_ParKrylovCreateVectorArray,
                  hypre_ParKrylovDestroyVector,
                  hypre_ParILUCusparseSchurGMRESMatvecCreate, //parCSR A -- inactive
                  hypre_ParILUCusparseSchurGMRESMatvec, //parCSR A -> ilu_data
                  hypre_ParILUCusparseSchurGMRESMatvecDestroy, //parCSR A -- inactive
                  hypre_ParKrylovInnerProd,
                  hypre_ParKrylovCopyVector,
                  hypre_ParKrylovClearVector,
                  hypre_ParKrylovScaleVector,
                  hypre_ParKrylovAxpy,
                  hypre_ParKrylovIdentitySetup, //parCSR A -- inactive
                  hypre_ParKrylovIdentity ); //parCSR A -- inactive
            schur_solver = ( (HYPRE_Solver) hypre_GMRESCreate( gmres_functions ) );

            /* setup GMRES parameters */
            HYPRE_GMRESSetKDim            (schur_solver, hypre_ParILUDataSchurGMRESKDim(ilu_data));
            HYPRE_GMRESSetMaxIter         (schur_solver, hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
            HYPRE_GMRESSetTol             (schur_solver, hypre_ParILUDataSchurGMRESTol(ilu_data));
            HYPRE_GMRESSetAbsoluteTol     (schur_solver, hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
            HYPRE_GMRESSetLogging         (schur_solver, hypre_ParILUDataSchurSolverLogging(ilu_data));
            HYPRE_GMRESSetPrintLevel      (schur_solver, hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
            HYPRE_GMRESSetRelChange       (schur_solver, hypre_ParILUDataSchurGMRESRelChange(ilu_data));

            /* setup preconditioner parameters */
            /* create Unit precond */
            schur_precond = (HYPRE_Solver) ilu_vdata;
            /* add preconditioner to solver */
            HYPRE_GMRESSetPrecond(schur_solver,
                     (HYPRE_PtrToSolverFcn) hypre_ParILUCusparseSchurGMRESDummySolve,
                     (HYPRE_PtrToSolverFcn) hypre_ParILUCusparseSchurGMRESDummySetup,
                                          schur_precond);
            HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);
            if (schur_precond_gotten != (schur_precond))
            {
               hypre_printf("Schur complement got bad precond\n");
               return(-1);
            }

            /* need to create working vector rhs and x for Schur System */
            rhs = hypre_ParVectorCreate(comm,
                                    hypre_ParCSRMatrixGlobalNumRows(matS),
                                    hypre_ParCSRMatrixRowStarts(matS));
            hypre_ParVectorInitialize(rhs);
            x = hypre_ParVectorCreate(comm,
                                    hypre_ParCSRMatrixGlobalNumRows(matS),
                                    hypre_ParCSRMatrixRowStarts(matS));
            hypre_ParVectorInitialize(x);

            /* setup solver */
            HYPRE_GMRESSetup(schur_solver,(HYPRE_Matrix)ilu_vdata,(HYPRE_Vector)rhs,(HYPRE_Vector)x);

            /* solve for right-hand-side consists of only 1 */
            hypre_Vector      *rhs_local = hypre_ParVectorLocalVector(rhs);
            //HYPRE_Real        *Xtemp_data  = hypre_VectorData(Xtemp_local);
            hypre_SeqVectorSetConstantValues(rhs_local, 1.0);

            /* update ilu_data */
            hypre_ParILUDataSchurSolver   (ilu_data) = schur_solver;
            hypre_ParILUDataSchurPrecond  (ilu_data) = schur_precond;
            hypre_ParILUDataRhs           (ilu_data) = rhs;
            hypre_ParILUDataX             (ilu_data) = x;
#else
            /* setup GMRES parameters */
            HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

            HYPRE_GMRESSetKDim            (schur_solver, hypre_ParILUDataSchurGMRESKDim(ilu_data));
            HYPRE_GMRESSetMaxIter         (schur_solver, hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
            HYPRE_GMRESSetTol             (schur_solver, hypre_ParILUDataSchurGMRESTol(ilu_data));
            HYPRE_GMRESSetAbsoluteTol     (schur_solver, hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
            HYPRE_GMRESSetLogging         (schur_solver, hypre_ParILUDataSchurSolverLogging(ilu_data));
            HYPRE_GMRESSetPrintLevel      (schur_solver, hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
            HYPRE_GMRESSetRelChange       (schur_solver, hypre_ParILUDataSchurGMRESRelChange(ilu_data));

            /* setup preconditioner parameters */
            /* create precond, the default is ILU0 */
            HYPRE_ILUCreate               (&schur_precond);
            HYPRE_ILUSetType              (schur_precond, hypre_ParILUDataSchurPrecondIluType(ilu_data));
            HYPRE_ILUSetLevelOfFill       (schur_precond, hypre_ParILUDataSchurPrecondIluLfil(ilu_data));
            HYPRE_ILUSetMaxNnzPerRow      (schur_precond, hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data));
            HYPRE_ILUSetDropThresholdArray(schur_precond, hypre_ParILUDataSchurPrecondIluDroptol(ilu_data));
            HYPRE_ILUSetPrintLevel        (schur_precond, hypre_ParILUDataSchurPrecondPrintLevel(ilu_data));
            HYPRE_ILUSetMaxIter           (schur_precond, hypre_ParILUDataSchurPrecondMaxIter(ilu_data));
            HYPRE_ILUSetTol               (schur_precond, hypre_ParILUDataSchurPrecondTol(ilu_data));

            /* add preconditioner to solver */
            HYPRE_GMRESSetPrecond(schur_solver,
                     (HYPRE_PtrToSolverFcn) HYPRE_ILUSolve,
                     (HYPRE_PtrToSolverFcn) HYPRE_ILUSetup,
                                          schur_precond);
            HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);
            if (schur_precond_gotten != (schur_precond))
            {
               hypre_printf("Schur complement got bad precond\n");
               HYPRE_ANNOTATE_FUNC_END;

               return(-1);
            }

            /* need to create working vector rhs and x for Schur System */
            rhs = hypre_ParVectorCreate(comm,
                                    hypre_ParCSRMatrixGlobalNumRows(matS),
                                    hypre_ParCSRMatrixRowStarts(matS));
            hypre_ParVectorInitialize(rhs);
            x = hypre_ParVectorCreate(comm,
                                    hypre_ParCSRMatrixGlobalNumRows(matS),
                                    hypre_ParCSRMatrixRowStarts(matS));
            hypre_ParVectorInitialize(x);

            /* setup solver */
            HYPRE_GMRESSetup(schur_solver,(HYPRE_Matrix)matS,(HYPRE_Vector)rhs,(HYPRE_Vector)x);

            /* update ilu_data */
            hypre_ParILUDataSchurSolver   (ilu_data) = schur_solver;
            hypre_ParILUDataSchurPrecond  (ilu_data) = schur_precond;
            hypre_ParILUDataRhs           (ilu_data) = rhs;
            hypre_ParILUDataX             (ilu_data) = x;
#endif
         }
         break;
      case 20: case 21:
         if (matS)
         {
            /* approximate inverse preconditioner */
            schur_solver = (HYPRE_Solver)hypre_NSHCreate();

            /* set NSH parameters */
            hypre_NSHSetMaxIter           (schur_solver, hypre_ParILUDataSchurNSHSolveMaxIter(ilu_data));
            hypre_NSHSetTol               (schur_solver, hypre_ParILUDataSchurNSHSolveTol(ilu_data));
            hypre_NSHSetLogging           (schur_solver, hypre_ParILUDataSchurSolverLogging(ilu_data));
            hypre_NSHSetPrintLevel        (schur_solver, hypre_ParILUDataSchurSolverPrintLevel(ilu_data));
            hypre_NSHSetDropThresholdArray(schur_solver, hypre_ParILUDataSchurNSHDroptol(ilu_data));

            hypre_NSHSetNSHMaxIter        (schur_solver, hypre_ParILUDataSchurNSHMaxNumIter(ilu_data));
            hypre_NSHSetNSHMaxRowNnz      (schur_solver, hypre_ParILUDataSchurNSHMaxRowNnz(ilu_data));
            hypre_NSHSetNSHTol            (schur_solver, hypre_ParILUDataSchurNSHTol(ilu_data));

            hypre_NSHSetMRMaxIter         (schur_solver, hypre_ParILUDataSchurMRMaxIter(ilu_data));
            hypre_NSHSetMRMaxRowNnz       (schur_solver, hypre_ParILUDataSchurMRMaxRowNnz(ilu_data));
            hypre_NSHSetMRTol             (schur_solver, hypre_ParILUDataSchurMRTol(ilu_data));
            hypre_NSHSetColVersion        (schur_solver, hypre_ParILUDataSchurMRColVersion(ilu_data));

            /* need to create working vector rhs and x for Schur System */
            rhs = hypre_ParVectorCreate(comm,
                  hypre_ParCSRMatrixGlobalNumRows(matS),
                  hypre_ParCSRMatrixRowStarts(matS));
            hypre_ParVectorInitialize(rhs);
            x = hypre_ParVectorCreate(comm,
                  hypre_ParCSRMatrixGlobalNumRows(matS),
                  hypre_ParCSRMatrixRowStarts(matS));
            hypre_ParVectorInitialize(x);

            /* setup solver */
            hypre_NSHSetup(schur_solver,matS,rhs,x);

            hypre_ParILUDataSchurSolver(ilu_data) = schur_solver;
            hypre_ParILUDataRhs        (ilu_data) = rhs;
            hypre_ParILUDataX          (ilu_data) = x;
         }
         break;
      case 30 : case 31:
         /* now check communication package */
         comm_pkg = hypre_ParCSRMatrixCommPkg(matA);
         /* create if not yet built */
         if (!comm_pkg)
         {
            hypre_MatvecCommPkgCreate(matA);
            comm_pkg = hypre_ParCSRMatrixCommPkg(matA);
         }
         /* create uext and fext */
         send_size =  hypre_ParCSRCommPkgSendMapStart(comm_pkg,hypre_ParCSRCommPkgNumSends(comm_pkg))
            - hypre_ParCSRCommPkgSendMapStart(comm_pkg,0);
         recv_size = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(matA));
         buffer_size = send_size > recv_size ? send_size : recv_size;
         fext = hypre_TAlloc(HYPRE_Real,buffer_size,HYPRE_MEMORY_HOST);
         uext = hypre_TAlloc(HYPRE_Real,buffer_size,HYPRE_MEMORY_HOST);
         break;
      case 40: case 41:
         if (matS)
         {
            /* setup GMRES parameters */
            HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

            HYPRE_GMRESSetKDim            (schur_solver, hypre_ParILUDataSchurGMRESKDim(ilu_data));
            HYPRE_GMRESSetMaxIter         (schur_solver, hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
            HYPRE_GMRESSetTol             (schur_solver, hypre_ParILUDataSchurGMRESTol(ilu_data));
            HYPRE_GMRESSetAbsoluteTol     (schur_solver, hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
            HYPRE_GMRESSetLogging         (schur_solver, hypre_ParILUDataSchurSolverLogging(ilu_data));
            HYPRE_GMRESSetPrintLevel      (schur_solver, hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
            HYPRE_GMRESSetRelChange       (schur_solver, hypre_ParILUDataSchurGMRESRelChange(ilu_data));

            /* setup preconditioner parameters */
            /* create precond, the default is ILU0 */
            HYPRE_ILUCreate               (&schur_precond);
            HYPRE_ILUSetType              (schur_precond, hypre_ParILUDataSchurPrecondIluType(ilu_data));
            HYPRE_ILUSetLevelOfFill       (schur_precond, hypre_ParILUDataSchurPrecondIluLfil(ilu_data));
            HYPRE_ILUSetMaxNnzPerRow      (schur_precond, hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data));
            HYPRE_ILUSetDropThresholdArray(schur_precond, hypre_ParILUDataSchurPrecondIluDroptol(ilu_data));
            HYPRE_ILUSetPrintLevel        (schur_precond, hypre_ParILUDataSchurPrecondPrintLevel(ilu_data));
            HYPRE_ILUSetMaxIter           (schur_precond, hypre_ParILUDataSchurPrecondMaxIter(ilu_data));
            HYPRE_ILUSetTol               (schur_precond, hypre_ParILUDataSchurPrecondTol(ilu_data));

            /* add preconditioner to solver */
            HYPRE_GMRESSetPrecond(schur_solver,
                     (HYPRE_PtrToSolverFcn) HYPRE_ILUSolve,
                     (HYPRE_PtrToSolverFcn) HYPRE_ILUSetup,
                                          schur_precond);
            HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);
            if (schur_precond_gotten != (schur_precond))
            {
               hypre_printf("Schur complement got bad precond\n");
               return(-1);
            }

            /* need to create working vector rhs and x for Schur System */
            rhs = hypre_ParVectorCreate(comm,
                                    hypre_ParCSRMatrixGlobalNumRows(matS),
                                    hypre_ParCSRMatrixRowStarts(matS));
            hypre_ParVectorInitialize(rhs);
            x = hypre_ParVectorCreate(comm,
                                    hypre_ParCSRMatrixGlobalNumRows(matS),
                                    hypre_ParCSRMatrixRowStarts(matS));
            hypre_ParVectorInitialize(x);

            /* setup solver */
            HYPRE_GMRESSetup(schur_solver,(HYPRE_Matrix)matS,(HYPRE_Vector)rhs,(HYPRE_Vector)x);

            /* update ilu_data */
            hypre_ParILUDataSchurSolver   (ilu_data) = schur_solver;
            hypre_ParILUDataSchurPrecond  (ilu_data) = schur_precond;
            hypre_ParILUDataRhs           (ilu_data) = rhs;
            hypre_ParILUDataX             (ilu_data) = x;
         }
         break;
      case 50:
      {
#ifdef HYPRE_USING_CUDA
         if (matS)
         {
            /* create working vectors */
            Xtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
                                   hypre_ParCSRMatrixGlobalNumRows(matA),
                                   hypre_ParCSRMatrixRowStarts(matA));
            hypre_ParVectorInitialize(Xtemp);

            Ytemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
                                   hypre_ParCSRMatrixGlobalNumRows(matA),
                                   hypre_ParCSRMatrixRowStarts(matA));
            hypre_ParVectorInitialize(Ytemp);

            Ftemp_upper = hypre_SeqVectorCreate(nLU);
            hypre_VectorOwnsData(Ftemp_upper)   = 0;
            hypre_VectorData(Ftemp_upper)       = hypre_VectorData(hypre_ParVectorLocalVector(Ftemp));
            hypre_SeqVectorInitialize(Ftemp_upper);

            Utemp_lower = hypre_SeqVectorCreate(n - nLU);
            hypre_VectorOwnsData(Utemp_lower)   = 0;
            hypre_VectorData(Utemp_lower)       = hypre_VectorData(hypre_ParVectorLocalVector(Utemp)) + nLU;
            hypre_SeqVectorInitialize(Utemp_lower);

            /* create GMRES */
//            HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

            hypre_GMRESFunctions * gmres_functions;

            gmres_functions =
               hypre_GMRESFunctionsCreate(
                  hypre_CAlloc,
                  hypre_ParKrylovFree,
                  hypre_ParILUCusparseSchurGMRESCommInfo, //parCSR A -> ilu_data
                  hypre_ParKrylovCreateVector,
                  hypre_ParKrylovCreateVectorArray,
                  hypre_ParKrylovDestroyVector,
                  hypre_ParILURAPSchurGMRESMatvecCreate, //parCSR A -- inactive
                  hypre_ParILURAPSchurGMRESMatvec, //parCSR A -> ilu_data
                  hypre_ParILURAPSchurGMRESMatvecDestroy, //parCSR A -- inactive
                  hypre_ParKrylovInnerProd,
                  hypre_ParKrylovCopyVector,
                  hypre_ParKrylovClearVector,
                  hypre_ParKrylovScaleVector,
                  hypre_ParKrylovAxpy,
                  hypre_ParKrylovIdentitySetup, //parCSR A -- inactive
                  hypre_ParKrylovIdentity ); //parCSR A -- inactive
            schur_solver = ( (HYPRE_Solver) hypre_GMRESCreate( gmres_functions ) );

            /* setup GMRES parameters */
            /* at least should apply 1 solve */
            if (hypre_ParILUDataSchurGMRESKDim(ilu_data) == 0)
            {
               hypre_ParILUDataSchurGMRESKDim(ilu_data) ++;
            }
            HYPRE_GMRESSetKDim            (schur_solver, hypre_ParILUDataSchurGMRESKDim(ilu_data));
            HYPRE_GMRESSetMaxIter         (schur_solver, hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
            HYPRE_GMRESSetTol             (schur_solver, hypre_ParILUDataSchurGMRESTol(ilu_data));
            HYPRE_GMRESSetAbsoluteTol     (schur_solver, hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
            HYPRE_GMRESSetLogging         (schur_solver, hypre_ParILUDataSchurSolverLogging(ilu_data));
            HYPRE_GMRESSetPrintLevel      (schur_solver, hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
            HYPRE_GMRESSetRelChange       (schur_solver, hypre_ParILUDataSchurGMRESRelChange(ilu_data));

            /* setup preconditioner parameters */
            /* create Schur precond */
            schur_precond = (HYPRE_Solver) ilu_vdata;
            /* add preconditioner to solver */
            HYPRE_GMRESSetPrecond(schur_solver,
                     (HYPRE_PtrToSolverFcn) hypre_ParILURAPSchurGMRESSolve,
                     //(HYPRE_PtrToSolverFcn) hypre_ParILUCusparseSchurGMRESDummySolve,
                     (HYPRE_PtrToSolverFcn) hypre_ParILURAPSchurGMRESDummySetup,
                                          schur_precond);
            HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);
            if (schur_precond_gotten != (schur_precond))
            {
               hypre_printf("Schur complement got bad precond\n");
               return(-1);
            }

            /* need to create working vector rhs and x for Schur System */
            rhs = hypre_ParVectorCreate(comm,
                                    hypre_ParCSRMatrixGlobalNumRows(matS),
                                    hypre_ParCSRMatrixRowStarts(matS));
            hypre_ParVectorInitialize(rhs);
            x = hypre_ParVectorCreate(comm,
                                    hypre_ParCSRMatrixGlobalNumRows(matS),
                                    hypre_ParCSRMatrixRowStarts(matS));
            hypre_ParVectorInitialize(x);

            /* setup solver */
            HYPRE_GMRESSetup(schur_solver,(HYPRE_Matrix)ilu_vdata,(HYPRE_Vector)rhs,(HYPRE_Vector)x);

            /* solve for right-hand-side consists of only 1 */
            //hypre_Vector      *rhs_local = hypre_ParVectorLocalVector(rhs);
            //HYPRE_Real        *Xtemp_data  = hypre_VectorData(Xtemp_local);
            //hypre_SeqVectorSetConstantValues(rhs_local, 1.0);

            /* update ilu_data */
            hypre_ParILUDataSchurSolver   (ilu_data) = schur_solver;
            hypre_ParILUDataSchurPrecond  (ilu_data) = schur_precond;
            hypre_ParILUDataRhs           (ilu_data) = rhs;
            hypre_ParILUDataX             (ilu_data) = x;
         }
#else
         /* need to create working vector rhs and x for Schur System */
         HYPRE_Int      m = n - nLU;
         HYPRE_BigInt   S_total_rows, S_row_starts[2];
         HYPRE_BigInt   big_m = (HYPRE_BigInt)m;
         hypre_MPI_Allreduce( &big_m, &S_total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

         if ( S_total_rows > 0 )
         {
            /* create working vectors */
            Xtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
                                      hypre_ParCSRMatrixGlobalNumRows(matA),
                                      hypre_ParCSRMatrixRowStarts(matA));
            hypre_ParVectorInitialize(Xtemp);

            Ytemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
                                      hypre_ParCSRMatrixGlobalNumRows(matA),
                                      hypre_ParCSRMatrixRowStarts(matA));
            hypre_ParVectorInitialize(Ytemp);

            /* only do so when we hae the Schur Complement */
            {
               HYPRE_BigInt global_start;
               hypre_MPI_Scan( &big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
               S_row_starts[0] = global_start - m;
               S_row_starts[1] = global_start;
            }

            rhs = hypre_ParVectorCreate(comm,
                                    S_total_rows,
                                    S_row_starts);
            hypre_ParVectorInitialize(rhs);

            x = hypre_ParVectorCreate(comm,
                                    S_total_rows,
                                    S_row_starts);
            hypre_ParVectorInitialize(x);

            /* add when necessary */
            /* create GMRES */
//            HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

            hypre_GMRESFunctions * gmres_functions;

            gmres_functions =
                  hypre_GMRESFunctionsCreate(
                     hypre_CAlloc,
                     hypre_ParKrylovFree,
                     hypre_ParILURAPSchurGMRESCommInfoH, //parCSR A -> ilu_data
                     hypre_ParKrylovCreateVector,
                     hypre_ParKrylovCreateVectorArray,
                     hypre_ParKrylovDestroyVector,
                     hypre_ParILURAPSchurGMRESMatvecCreateH, //parCSR A -- inactive
                     hypre_ParILURAPSchurGMRESMatvecH, //parCSR A -> ilu_data
                     hypre_ParILURAPSchurGMRESMatvecDestroyH, //parCSR A -- inactive
                     hypre_ParKrylovInnerProd,
                     hypre_ParKrylovCopyVector,
                     hypre_ParKrylovClearVector,
                     hypre_ParKrylovScaleVector,
                     hypre_ParKrylovAxpy,
                     hypre_ParKrylovIdentitySetup, //parCSR A -- inactive
                     hypre_ParKrylovIdentity ); //parCSR A -- inactive
            schur_solver = ( (HYPRE_Solver) hypre_GMRESCreate( gmres_functions ) );

            /* setup GMRES parameters */
            /* at least should apply 1 solve */
            if (hypre_ParILUDataSchurGMRESKDim(ilu_data) == 0)
            {
               hypre_ParILUDataSchurGMRESKDim(ilu_data) ++;
            }
            HYPRE_GMRESSetKDim            (schur_solver, hypre_ParILUDataSchurGMRESKDim(ilu_data));
            HYPRE_GMRESSetMaxIter         (schur_solver, hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
            HYPRE_GMRESSetTol             (schur_solver, hypre_ParILUDataSchurGMRESTol(ilu_data));
            HYPRE_GMRESSetAbsoluteTol     (schur_solver, hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
            HYPRE_GMRESSetLogging         (schur_solver, hypre_ParILUDataSchurSolverLogging(ilu_data));
            HYPRE_GMRESSetPrintLevel      (schur_solver, hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
            HYPRE_GMRESSetRelChange       (schur_solver, hypre_ParILUDataSchurGMRESRelChange(ilu_data));

            /* setup preconditioner parameters */
            /* create Schur precond */
            schur_precond = (HYPRE_Solver) ilu_vdata;
            /* add preconditioner to solver */
            HYPRE_GMRESSetPrecond(schur_solver,
                     (HYPRE_PtrToSolverFcn) hypre_ParILURAPSchurGMRESSolveH,
                     //(HYPRE_PtrToSolverFcn) hypre_ParILUCusparseSchurGMRESDummySolve,
                     (HYPRE_PtrToSolverFcn) hypre_ParILURAPSchurGMRESDummySetupH,
                                          schur_precond);
            HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);
            if (schur_precond_gotten != (schur_precond))
            {
               hypre_printf("Schur complement got bad precond\n");
               return(-1);
            }

            /* setup solver */
            HYPRE_GMRESSetup(schur_solver,(HYPRE_Matrix)ilu_vdata,(HYPRE_Vector)rhs,(HYPRE_Vector)x);

            /* solve for right-hand-side consists of only 1 */
            //hypre_Vector      *rhs_local = hypre_ParVectorLocalVector(rhs);
            //HYPRE_Real        *Xtemp_data  = hypre_VectorData(Xtemp_local);
            //hypre_SeqVectorSetConstantValues(rhs_local, 1.0);
         }
         /* update ilu_data */
         hypre_ParILUDataSchurSolver   (ilu_data) = schur_solver;
         hypre_ParILUDataSchurPrecond  (ilu_data) = schur_precond;
         hypre_ParILUDataRhs           (ilu_data) = rhs;
         hypre_ParILUDataX             (ilu_data) = x;

#endif
         break;
      }
      default:
         break;
   }
   /* set pointers to ilu data */
#ifdef HYPRE_USING_CUDA
   /* set cusparse pointers */
   //hypre_ParILUDataILUSolveBuffer(ilu_data)  = ilu_solve_buffer;
   hypre_ParILUDataMatAILUDevice(ilu_data)      = matALU_d;
   hypre_ParILUDataMatBILUDevice(ilu_data)      = matBLU_d;
   hypre_ParILUDataMatSILUDevice(ilu_data)      = matSLU_d;
   hypre_ParILUDataMatEDevice(ilu_data)         = matE_d;
   hypre_ParILUDataMatFDevice(ilu_data)         = matF_d;
   hypre_ParILUDataILUSolveBuffer(ilu_data)     = ilu_solve_buffer;
   hypre_ParILUDataMatALILUSolveInfo(ilu_data)  = matAL_info;
   hypre_ParILUDataMatAUILUSolveInfo(ilu_data)  = matAU_info;
   hypre_ParILUDataMatBLILUSolveInfo(ilu_data)  = matBL_info;
   hypre_ParILUDataMatBUILUSolveInfo(ilu_data)  = matBU_info;
   hypre_ParILUDataMatSLILUSolveInfo(ilu_data)  = matSL_info;
   hypre_ParILUDataMatSUILUSolveInfo(ilu_data)  = matSU_info;
   hypre_ParILUDataAperm(ilu_data)              = Aperm;
   hypre_ParILUDataR(ilu_data)                  = R;
   hypre_ParILUDataP(ilu_data)                  = P;
   hypre_ParILUDataFTempUpper(ilu_data)         = Ftemp_upper;
   hypre_ParILUDataUTempLower(ilu_data)         = Utemp_lower;
   hypre_ParILUDataMatAFakeDiagonal(ilu_data)   = A_diag_fake;
#endif
   hypre_ParILUDataMatA(ilu_data)               = matA;
   hypre_ParILUDataXTemp(ilu_data)              = Xtemp;
   hypre_ParILUDataYTemp(ilu_data)              = Ytemp;
   hypre_ParILUDataF(ilu_data)                  = F_array;
   hypre_ParILUDataU(ilu_data)                  = U_array;
   hypre_ParILUDataMatL(ilu_data)               = matL;
   hypre_ParILUDataMatD(ilu_data)               = matD;
   hypre_ParILUDataMatU(ilu_data)               = matU;
   hypre_ParILUDataMatLModified(ilu_data)       = matmL;
   hypre_ParILUDataMatDModified(ilu_data)       = matmD;
   hypre_ParILUDataMatUModified(ilu_data)       = matmU;
   hypre_ParILUDataMatS(ilu_data)               = matS;
   hypre_ParILUDataCFMarkerArray(ilu_data)      = CF_marker_array;
   hypre_ParILUDataPerm(ilu_data)               = perm;
   hypre_ParILUDataQPerm(ilu_data)              = qperm;
   hypre_ParILUDataNLU(ilu_data)                = nLU;
   hypre_ParILUDataNI(ilu_data)                 = nI;
   hypre_ParILUDataUEnd(ilu_data)               = u_end;
   hypre_ParILUDataUExt(ilu_data)               = uext;
   hypre_ParILUDataFExt(ilu_data)               = fext;

   /* compute operator complexity */
   hypre_ParCSRMatrixSetDNumNonzeros(matA);
   nnzS = 0.0;
   /* size_C is the size of global coarse grid, upper left part */
   size_C = hypre_ParCSRMatrixGlobalNumRows(matA);
   /* switch to compute complexity */

#ifdef HYPRE_USING_CUDA
   HYPRE_Int nnzBEF = 0;
   HYPRE_Int nnzG;/* Global nnz */
   if (ilu_type == 0 && fill_level == 0)
   {
      /* The nnz is for sure 1.0 in this case */
      hypre_ParILUDataOperatorComplexity(ilu_data) =  1.0;
   }
   else if (ilu_type == 10 && fill_level == 0)
   {
      /* The nnz is the sum of different parts */
      if (matBLU_d)
      {
         nnzBEF  += hypre_CSRMatrixNumNonzeros(matBLU_d);
      }
      if (matE_d)
      {
         nnzBEF  += hypre_CSRMatrixNumNonzeros(matE_d);
      }
      if (matF_d)
      {
         nnzBEF  += hypre_CSRMatrixNumNonzeros(matF_d);
      }
      hypre_MPI_Allreduce(&nnzBEF, &nnzG, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
      if (matS)
      {
         hypre_ParCSRMatrixSetDNumNonzeros(matS);
         nnzS = hypre_ParCSRMatrixDNumNonzeros(matS);
         /* if we have Schur system need to reduce it from size_C */
      }
      hypre_ParILUDataOperatorComplexity(ilu_data) =  ((HYPRE_Real)nnzG + nnzS) /
                                           hypre_ParCSRMatrixDNumNonzeros(matA);
   }
   else if (ilu_type == 50)
   {
      hypre_ParILUDataOperatorComplexity(ilu_data) =  1.0;
   }
   else if (ilu_type == 0 || ilu_type == 1 || ilu_type == 10 || ilu_type == 11)
   {
      if (matBLU_d)
      {
         nnzBEF  += hypre_CSRMatrixNumNonzeros(matBLU_d);
      }
      if (matE_d)
      {
         nnzBEF  += hypre_CSRMatrixNumNonzeros(matE_d);
      }
      if (matF_d)
      {
         nnzBEF  += hypre_CSRMatrixNumNonzeros(matF_d);
      }
      hypre_MPI_Allreduce(&nnzBEF, &nnzG, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
      if (matS)
      {
         hypre_ParCSRMatrixSetDNumNonzeros(matS);
         nnzS = hypre_ParCSRMatrixDNumNonzeros(matS);
         /* if we have Schur system need to reduce it from size_C */
      }
      hypre_ParILUDataOperatorComplexity(ilu_data) =  ((HYPRE_Real)nnzG + nnzS) /
                                           hypre_ParCSRMatrixDNumNonzeros(matA);
   }
   else
   {
#endif
      if (matS)
      {
         hypre_ParCSRMatrixSetDNumNonzeros(matS);
         nnzS = hypre_ParCSRMatrixDNumNonzeros(matS);
         /* if we have Schur system need to reduce it from size_C */
         size_C -= hypre_ParCSRMatrixGlobalNumRows(matS);
         switch(ilu_type)
         {
            case 10: case 11: case 40: case 41: case 50:
               /* now we need to compute the preconditioner */
               schur_precond_ilu = (hypre_ParILUData*) (hypre_ParILUDataSchurPrecond(ilu_data));
               /* borrow i for local nnz of S */
               i = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(matS));
               hypre_MPI_Allreduce(&i, &nnzS_offd, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
               nnzS = nnzS * hypre_ParILUDataOperatorComplexity(schur_precond_ilu) +nnzS_offd;
               break;
            case 20: case 21:
               schur_solver_nsh = (hypre_ParNSHData*) hypre_ParILUDataSchurSolver(ilu_data);
               nnzS = nnzS * (hypre_ParNSHDataOperatorComplexity(schur_solver_nsh));
               break;
            default:
               break;
         }
      }

      hypre_ParILUDataOperatorComplexity(ilu_data) =  ((HYPRE_Real)size_C + nnzS +
                                          hypre_ParCSRMatrixDNumNonzeros(matL) +
                                          hypre_ParCSRMatrixDNumNonzeros(matU))/
                                          hypre_ParCSRMatrixDNumNonzeros(matA);
#ifdef HYPRE_USING_CUDA
   }
#endif
   if ((my_id == 0) && (print_level > 0))
   {
      hypre_printf("ILU SETUP: operator complexity = %f  \n", hypre_ParILUDataOperatorComplexity(ilu_data));
   }

   if ( logging > 1 ) {
      residual =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
               hypre_ParCSRMatrixGlobalNumRows(matA),
               hypre_ParCSRMatrixRowStarts(matA) );
      hypre_ParVectorInitialize(residual);
      hypre_ParILUDataResidual(ilu_data) = residual;
   }
   else{
      hypre_ParILUDataResidual(ilu_data) = NULL;
   }
   rel_res_norms = hypre_CTAlloc(HYPRE_Real, hypre_ParILUDataMaxIter(ilu_data), HYPRE_MEMORY_HOST);
   hypre_ParILUDataRelResNorms(ilu_data) = rel_res_norms;
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

#ifdef HYPRE_USING_CUDA

/* Extract submatrix from diagonal part of A into a new CSRMatrix without sort rows
 * WARNING: We don't put diagonal to the first entry of each row since this function is now for cuSparse only
 * A = input matrix
 * perm = permutation array indicating ordering of rows. Perm could come from a
 *    CF_marker array or a reordering routine.
 * rqperm = reverse permutation array indicating ordering of columns
 * A_diagp = pointer to the output diagonal matrix.
 */
HYPRE_Int
hypre_ParILUCusparseExtractDiagonalCSR( hypre_ParCSRMatrix *A,
                                        HYPRE_Int          *perm,
                                        HYPRE_Int          *rqperm,
                                        hypre_CSRMatrix   **A_diagp )
{
   /* Get necessary slots */
   hypre_CSRMatrix     *A_diag         = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int           *A_diag_i       = hypre_CSRMatrixI(A_diag);
   HYPRE_Int           *A_diag_j       = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real          *A_diag_data    = hypre_CSRMatrixData(A_diag);
   HYPRE_Int            n              = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int            nnz_A_diag     = A_diag_i[n];

   HYPRE_Int            i, j, current_idx;

   /* No schur complement makes everything easy :) */
   hypre_CSRMatrix  *B              = NULL;
   B                                = hypre_CSRMatrixCreate(n, n, nnz_A_diag);
   hypre_CSRMatrixInitialize(B);
   HYPRE_Int        *B_i            = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j            = hypre_CSRMatrixJ(B);
   HYPRE_Real       *B_data         = hypre_CSRMatrixData(B);

   /* Copy everything in with permutation */
   current_idx = 0;
   for ( i = 0; i < n; i++ )
   {
      B_i[i] = current_idx;
      for (j = A_diag_i[perm[i]] ; j < A_diag_i[perm[i]+1] ; j ++)
      {
         B_j[current_idx] = rqperm[A_diag_j[j]];
         B_data[current_idx++] = A_diag_data[j];
      }
   }
   B_i[n] = current_idx;

   hypre_assert(current_idx == nnz_A_diag);
   *A_diagp = B;

   return hypre_error_flag;
}

/* Extract submatrix from diagonal part of A into a
 * | B F |
 * | E C |
 * Struct in order to do ILU with cusparse.
 * WARNING: Cusparse requires each row been sorted by column
 *          This function only works when rows are sorted!.
 * A = input matrix
 * perm = permutation array indicating ordering of rows. Perm could come from a
 *    CF_marker array or a reordering routine.
 * qperm = permutation array indicating ordering of columns
 * Bp = pointer to the output B matrix.
 * Cp = pointer to the output C matrix.
 * Ep = pointer to the output E matrix.
 * Fp = pointer to the output F matrix.
 */
HYPRE_Int
hypre_ParILUCusparseILUExtractEBFC(hypre_CSRMatrix *A_diag, HYPRE_Int nLU, hypre_CSRMatrix **Bp, hypre_CSRMatrix **Cp, hypre_CSRMatrix **Ep, hypre_CSRMatrix **Fp)
{
   /* Get necessary slots */
   HYPRE_Int           *A_diag_i       = hypre_CSRMatrixI(A_diag);
   HYPRE_Int           *A_diag_j       = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real          *A_diag_data    = hypre_CSRMatrixData(A_diag);
   HYPRE_Int            n              = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int            nnz_A_diag     = A_diag_i[n];

   HYPRE_Int            i, j, row, col;

   hypre_assert(nLU >= 0 && nLU <= n);

   if (nLU == n)
   {
      /* No schur complement makes everything easy :) */
      hypre_CSRMatrix  *B              = NULL;
      hypre_CSRMatrix  *C              = NULL;
      hypre_CSRMatrix  *E              = NULL;
      hypre_CSRMatrix  *F              = NULL;
      B                                = hypre_CSRMatrixCreate(n, n, nnz_A_diag);
      hypre_CSRMatrixInitialize(B);
      hypre_CSRMatrixCopy(A_diag, B, 1);
      C                                = hypre_CSRMatrixCreate(0, 0, 0);
      hypre_CSRMatrixInitialize(C);
      E                                = hypre_CSRMatrixCreate(0, 0, 0);
      hypre_CSRMatrixInitialize(E);
      F                                = hypre_CSRMatrixCreate(0, 0, 0);
      hypre_CSRMatrixInitialize(F);
      *Bp = B;
      *Cp = C;
      *Ep = E;
      *Fp = F;
   }
   else if (nLU ==0)
   {
      /* All schur complement also makes everything easy :) */
      hypre_CSRMatrix  *B              = NULL;
      hypre_CSRMatrix  *C              = NULL;
      hypre_CSRMatrix  *E              = NULL;
      hypre_CSRMatrix  *F              = NULL;
      C                                = hypre_CSRMatrixCreate(n, n, nnz_A_diag);
      hypre_CSRMatrixInitialize(C);
      hypre_CSRMatrixCopy(A_diag, C, 1);
      B                                = hypre_CSRMatrixCreate(0, 0, 0);
      hypre_CSRMatrixInitialize(B);
      E                                = hypre_CSRMatrixCreate(0, 0, 0);
      hypre_CSRMatrixInitialize(E);
      F                                = hypre_CSRMatrixCreate(0, 0, 0);
      hypre_CSRMatrixInitialize(F);
      *Bp = B;
      *Cp = C;
      *Ep = E;
      *Fp = F;
   }
   else
   {
      /* Has schur complement :( */
      HYPRE_Int         m              = n - nLU;
      hypre_CSRMatrix  *B              = NULL;
      hypre_CSRMatrix  *C              = NULL;
      hypre_CSRMatrix  *E              = NULL;
      hypre_CSRMatrix  *F              = NULL;
      HYPRE_Int         capacity_B;
      HYPRE_Int         capacity_E;
      HYPRE_Int         capacity_F;
      HYPRE_Int         capacity_C;
      HYPRE_Int         ctrB;
      HYPRE_Int         ctrC;
      HYPRE_Int         ctrE;
      HYPRE_Int         ctrF;

      HYPRE_Int        *B_i            = NULL;
      HYPRE_Int        *C_i            = NULL;
      HYPRE_Int        *E_i            = NULL;
      HYPRE_Int        *F_i            = NULL;
      HYPRE_Int        *B_j            = NULL;
      HYPRE_Int        *C_j            = NULL;
      HYPRE_Int        *E_j            = NULL;
      HYPRE_Int        *F_j            = NULL;
      HYPRE_Real       *B_data         = NULL;
      HYPRE_Real       *C_data         = NULL;
      HYPRE_Real       *E_data         = NULL;
      HYPRE_Real       *F_data         = NULL;

      /* Create CSRMatrices */
      B                                = hypre_CSRMatrixCreate(nLU, nLU, 0);
      hypre_CSRMatrixInitialize(B);
      C                                = hypre_CSRMatrixCreate(m, m, 0);
      hypre_CSRMatrixInitialize(C);
      E                                = hypre_CSRMatrixCreate(m, nLU, 0);
      hypre_CSRMatrixInitialize(E);
      F                                = hypre_CSRMatrixCreate(nLU, m, 0);
      hypre_CSRMatrixInitialize(F);

      /* Estimate # of nonzeros */
      capacity_B                       = nLU + ceil(nnz_A_diag * 1.0 * nLU / n * nLU / n);
      capacity_C                       = m + ceil(nnz_A_diag * 1.0 * m / n * m / n);
      capacity_E                       = hypre_min(m, nLU) + ceil(nnz_A_diag * 1.0 * nLU / n * m / n);
      capacity_F                       = capacity_E;

      /* Allocate memory */
      B_i                              = hypre_CSRMatrixI(B);
      B_j                              = hypre_CTAlloc(HYPRE_Int, capacity_B, HYPRE_MEMORY_DEVICE);
      B_data                           = hypre_CTAlloc(HYPRE_Real, capacity_B, HYPRE_MEMORY_DEVICE);
      C_i                              = hypre_CSRMatrixI(C);
      C_j                              = hypre_CTAlloc(HYPRE_Int, capacity_C, HYPRE_MEMORY_DEVICE);
      C_data                           = hypre_CTAlloc(HYPRE_Real, capacity_C, HYPRE_MEMORY_DEVICE);
      E_i                              = hypre_CSRMatrixI(E);
      E_j                              = hypre_CTAlloc(HYPRE_Int, capacity_E, HYPRE_MEMORY_DEVICE);
      E_data                           = hypre_CTAlloc(HYPRE_Real, capacity_E, HYPRE_MEMORY_DEVICE);
      F_i                              = hypre_CSRMatrixI(F);
      F_j                              = hypre_CTAlloc(HYPRE_Int, capacity_F, HYPRE_MEMORY_DEVICE);
      F_data                           = hypre_CTAlloc(HYPRE_Real, capacity_F, HYPRE_MEMORY_DEVICE);
      ctrB                             = 0;
      ctrC                             = 0;
      ctrE                             = 0;
      ctrF                             = 0;

      /* Loop to copy data */
      /* B and F first */
      for (i = 0; i < nLU; i++)
      {
         B_i[i]   = ctrB;
         F_i[i]   = ctrF;
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (col >= nLU)
            {
               break;
            }
            B_j[ctrB] = col;
            B_data[ctrB++] = A_diag_data[j];
            /* check capacity */
            if (ctrB >= capacity_B)
            {
               HYPRE_Int tmp;
               tmp = capacity_B;
               capacity_B = capacity_B * EXPAND_FACT + 1;
               B_j = hypre_TReAlloc_v2(B_j, HYPRE_Int, tmp, HYPRE_Int, capacity_B, HYPRE_MEMORY_DEVICE);
               B_data = hypre_TReAlloc_v2(B_data, HYPRE_Real, tmp, HYPRE_Real, capacity_B, HYPRE_MEMORY_DEVICE);
            }
         }
         for (; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            col = col - nLU;
            F_j[ctrF] = col;
            F_data[ctrF++] = A_diag_data[j];
            if (ctrF >= capacity_F)
            {
               HYPRE_Int tmp;
               tmp = capacity_F;
               capacity_F = capacity_F * EXPAND_FACT + 1;
               F_j = hypre_TReAlloc_v2(F_j, HYPRE_Int, tmp, HYPRE_Int, capacity_F, HYPRE_MEMORY_DEVICE);
               F_data = hypre_TReAlloc_v2(F_data, HYPRE_Real, tmp, HYPRE_Real, capacity_F, HYPRE_MEMORY_DEVICE);
            }
         }
      }
      B_i[nLU] = ctrB;
      F_i[nLU] = ctrF;

      /* E and C afterward */
      for (i = nLU; i < n; i++)
      {
         row = i - nLU;
         E_i[row] = ctrE;
         C_i[row] = ctrC;
         for (j = A_diag_i[i]; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            if (col >= nLU)
            {
               break;
            }
            E_j[ctrE] = col;
            E_data[ctrE++] = A_diag_data[j];
            /* check capacity */
            if (ctrE >= capacity_E)
            {
               HYPRE_Int tmp;
               tmp = capacity_E;
               capacity_E = capacity_E * EXPAND_FACT + 1;
               E_j = hypre_TReAlloc_v2(E_j, HYPRE_Int, tmp, HYPRE_Int, capacity_E, HYPRE_MEMORY_DEVICE);
               E_data = hypre_TReAlloc_v2(E_data, HYPRE_Real, tmp, HYPRE_Real, capacity_E, HYPRE_MEMORY_DEVICE);
            }
         }
         for (; j < A_diag_i[i+1]; j++)
         {
            col = A_diag_j[j];
            col = col - nLU;
            C_j[ctrC] = col;
            C_data[ctrC++] = A_diag_data[j];
            if (ctrC >= capacity_C)
            {
               HYPRE_Int tmp;
               tmp = capacity_C;
               capacity_C = capacity_C * EXPAND_FACT + 1;
               C_j = hypre_TReAlloc_v2(C_j, HYPRE_Int, tmp, HYPRE_Int, capacity_C, HYPRE_MEMORY_DEVICE);
               C_data = hypre_TReAlloc_v2(C_data, HYPRE_Real, tmp, HYPRE_Real, capacity_C, HYPRE_MEMORY_DEVICE);
            }
         }
      }
      E_i[m] = ctrE;
      C_i[m] = ctrC;

      hypre_assert((ctrB+ctrC+ctrE+ctrF) == nnz_A_diag);

      /* Create CSRMatrices */
      hypre_CSRMatrixJ(B)              = B_j;
      hypre_CSRMatrixData(B)           = B_data;
      hypre_CSRMatrixNumNonzeros(B)    = ctrB;
      hypre_CSRMatrixSetDataOwner(B, 1);
      *Bp                              = B;

      hypre_CSRMatrixJ(C)              = C_j;
      hypre_CSRMatrixData(C)           = C_data;
      hypre_CSRMatrixNumNonzeros(C)    = ctrC;
      hypre_CSRMatrixSetDataOwner(C, 1);
      *Cp                              = C;

      hypre_CSRMatrixJ(E)              = E_j;
      hypre_CSRMatrixData(E)           = E_data;
      hypre_CSRMatrixNumNonzeros(E)    = ctrE;
      hypre_CSRMatrixSetDataOwner(E, 1);
      *Ep                              = E;

      hypre_CSRMatrixJ(F)              = F_j;
      hypre_CSRMatrixData(F)           = F_data;
      hypre_CSRMatrixNumNonzeros(F)    = ctrF;
      hypre_CSRMatrixSetDataOwner(F, 1);
      *Fp                              = F;
   }

   return hypre_error_flag;
}

/* Wrapper for ILU0 with cusparse on a matrix, csr sort was done in this function */
HYPRE_Int
HYPRE_ILUSetupCusparseCSRILU0(hypre_CSRMatrix *A, cusparseSolvePolicy_t ilu_solve_policy)
{

   /* data objects for A */
   HYPRE_Int               n                    = hypre_CSRMatrixNumRows(A);
   HYPRE_Int               m                    = hypre_CSRMatrixNumCols(A);

   hypre_assert(n == m);

   HYPRE_Real              *A_data              = hypre_CSRMatrixData(A);
   HYPRE_Int               *A_i                 = hypre_CSRMatrixI(A);
   HYPRE_Int               *A_j                 = hypre_CSRMatrixJ(A);
   HYPRE_Int               nnz_A                = hypre_CSRMatrixNumNonzeros(A);

   /* pointers to cusparse data */
   csrilu02Info_t          matA_info            = NULL;

   /* variables and working arrays used during the ilu */
   HYPRE_Int               zero_pivot;
   HYPRE_Int               matA_buffersize;
   void                    *matA_buffer         = NULL;

   HYPRE_Int               isDoublePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int               isSinglePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t descr = hypre_CSRMatrixGPUMatDescr(A);

   hypre_assert(isDoublePrecision || isSinglePrecision);

   /* 1. Sort columns inside each row first, we can't assume that's sorted */
   hypre_SortCSRCusparse(n, m, nnz_A, descr, A_i, A_j, A_data);

   /* 2. Create info for ilu setup and solve */
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrilu02Info(&matA_info));

   /* 3. Get working array size */
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrilu02_bufferSize(handle, n, nnz_A, descr,
                                                         (hypre_double *) A_data, A_i, A_j,
                                                         matA_info, &matA_buffersize));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrilu02_bufferSize(handle, n, nnz_A, descr,
                                                         (float *) A_data, A_i, A_j,
                                                         matA_info, &matA_buffersize));
   }
   /* 4. Create working array, since they won't be visited by host, allocate on device */
   matA_buffer                                  = hypre_MAlloc(matA_buffersize, HYPRE_MEMORY_DEVICE);

   /* 5. Now perform the analysis */
   /* 5-1. Analysis */
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrilu02_analysis(handle, n, nnz_A, descr,
                                                      (hypre_double *) A_data, A_i, A_j,
                                                      matA_info, ilu_solve_policy, matA_buffer));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrilu02_analysis(handle, n, nnz_A, descr,
                                                      (float *) A_data, A_i, A_j,
                                                      matA_info, ilu_solve_policy, matA_buffer));
   }
   /* 5-2. Check for zero pivot */
   HYPRE_CUSPARSE_CALL(cusparseXcsrilu02_zeroPivot(handle, matA_info, &zero_pivot));

   /* 6. Apply the factorization */
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrilu02(handle, n, nnz_A, descr,
                                             (hypre_double *) A_data, A_i, A_j,
                                             matA_info, ilu_solve_policy, matA_buffer));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrilu02(handle, n, nnz_A, descr,
                                             (float *) A_data, A_i, A_j,
                                             matA_info, ilu_solve_policy, matA_buffer));
   }

   /* Check for zero pivot */
   HYPRE_CUSPARSE_CALL(cusparseXcsrilu02_zeroPivot(handle, matA_info, &zero_pivot));

   /* Done with factorization, finishing up */
   hypre_TFree(matA_buffer, HYPRE_MEMORY_DEVICE);
   HYPRE_CUSPARSE_CALL(cusparseDestroyCsrilu02Info(matA_info));

   return hypre_error_flag;
}

/* Wrapper for ILU0 solve analysis phase with cusparse on a matrix */
HYPRE_Int
HYPRE_ILUSetupCusparseCSRILU0SetupSolve(hypre_CSRMatrix *A, cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des,
                              cusparseSolvePolicy_t ilu_solve_policy, csrsv2Info_t *matL_infop, csrsv2Info_t *matU_infop,
                              HYPRE_Int *buffer_sizep, void **bufferp)
{
   if (!A)
   {
      /* return if A is NULL */
      *matL_infop    = NULL;
      *matU_infop    = NULL;
      *buffer_sizep  = 0;
      *bufferp       = NULL;
      return hypre_error_flag;
   }

   /* data objects for A */
   HYPRE_Int               n                    = hypre_CSRMatrixNumRows(A);
   HYPRE_Int               m                    = hypre_CSRMatrixNumCols(A);

   hypre_assert(n == m);

   if (n == 0)
   {
      /* return if A is 0 by 0 */
      *matL_infop    = NULL;
      *matU_infop    = NULL;
      *buffer_sizep  = 0;
      *bufferp       = NULL;
      return hypre_error_flag;
   }

   HYPRE_Real              *A_data              = hypre_CSRMatrixData(A);
   HYPRE_Int               *A_i                 = hypre_CSRMatrixI(A);
   HYPRE_Int               *A_j                 = hypre_CSRMatrixJ(A);
   HYPRE_Int               nnz_A                = A_i[n];

   /* pointers to cusparse data */
   csrsv2Info_t            matL_info            = *matL_infop;
   csrsv2Info_t            matU_info            = *matU_infop;

   /* clear data if already exists */
   if (matL_info)
   {
      HYPRE_CUSPARSE_CALL( cusparseDestroyCsrsv2Info(matL_info) );
      matL_info = NULL;
   }
   if (matU_info)
   {
      HYPRE_CUSPARSE_CALL( cusparseDestroyCsrsv2Info(matU_info) );
      matU_info = NULL;
   }

   /* variables and working arrays used during the ilu */
   HYPRE_Int               matL_buffersize;
   HYPRE_Int               matU_buffersize;
   HYPRE_Int               solve_buffersize;
   HYPRE_Int               solve_oldbuffersize  = *buffer_sizep;
   void                    *solve_buffer        = *bufferp;

   HYPRE_Int               isDoublePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int               isSinglePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   hypre_assert(isDoublePrecision || isSinglePrecision);

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());

   /* 1. Create info for ilu setup and solve */
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrsv2Info(&(matL_info)));
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrsv2Info(&(matU_info)));

   /* 2. Get working array size */
   if (isDoublePrecision)
   {

      HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz_A,
                                                      matL_des, (hypre_double *) A_data, A_i, A_j,
                                                      matL_info, &matL_buffersize));

      HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz_A,
                                                      matU_des, (hypre_double *) A_data, A_i, A_j,
                                                      matU_info, &matU_buffersize));
   }
   else if (isSinglePrecision)
   {

      HYPRE_CUSPARSE_CALL(cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz_A,
                                                      matL_des, (float *) A_data, A_i, A_j,
                                                      matL_info, &matL_buffersize));

      HYPRE_CUSPARSE_CALL(cusparseScsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz_A,
                                                      matU_des, (float *) A_data, A_i, A_j,
                                                      matU_info, &matU_buffersize));
   }
   solve_buffersize = hypre_max( matL_buffersize, matU_buffersize );
   /* 3. Create working array, since they won't be visited by host, allocate on device */
   if (solve_buffersize > solve_oldbuffersize)
   {
      if (solve_buffer)
      {
         solve_buffer                           = hypre_ReAlloc_v2(solve_buffer, solve_oldbuffersize, solve_buffersize, HYPRE_MEMORY_DEVICE);
      }
      else
      {
         solve_buffer                           = hypre_MAlloc(solve_buffersize, HYPRE_MEMORY_DEVICE);
      }
   }

   /* 4. Now perform the analysis */
   if (isDoublePrecision)
   {

      HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      n, nnz_A, matL_des,
                                                      (hypre_double *) A_data, A_i, A_j,
                                                      matL_info, ilu_solve_policy, solve_buffer));

      HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      n, nnz_A, matU_des,
                                                      (hypre_double *) A_data, A_i, A_j,
                                                      matU_info, ilu_solve_policy, solve_buffer));
   }
   else if (isSinglePrecision)
   {

      HYPRE_CUSPARSE_CALL(cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      n, nnz_A, matL_des,
                                                      (float *) A_data, A_i, A_j,
                                                      matL_info, ilu_solve_policy, solve_buffer));

      HYPRE_CUSPARSE_CALL(cusparseScsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      n, nnz_A, matU_des,
                                                      (float *) A_data, A_i, A_j,
                                                      matU_info, ilu_solve_policy, solve_buffer));
   }

   /* Done with analysis, finishing up */
   /* Set return value */
   *matL_infop    = matL_info;
   *matU_infop    = matU_info;
   *buffer_sizep  = solve_buffersize;
   *bufferp       = solve_buffer;

   return hypre_error_flag;
}

/* ILU(0) (GPU)
 * A = input matrix
 * perm = permutation array indicating ordering of rows. Perm could come from a
 *    CF_marker array or a reordering routine.
 * qperm = permutation array indicating ordering of columns
 * nI = number of interial unknowns
 * nLU = size of incomplete factorization, nLU should obey nLU <= nI.
 *    Schur complement is formed if nLU < n
 * Lptr, Dptr, Uptr, Sptr = L, D, U, S factors. Note that with CUDA, Dptr and Uptr are unused
 * xtempp, ytempp = helper vector used in 2-level solve.
 * A_fake_diagp = fake diagonal for matvec
 * will form global Schur Matrix if nLU < n
 */
HYPRE_Int
hypre_ILUSetupILU0Device(hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int *qperm, HYPRE_Int n, HYPRE_Int nLU,
                           cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des, cusparseSolvePolicy_t ilu_solve_policy,
                           void **bufferp, csrsv2Info_t *matBL_infop, csrsv2Info_t *matBU_infop,
                           csrsv2Info_t *matSL_infop, csrsv2Info_t *matSU_infop,
                           hypre_CSRMatrix **BLUptr, hypre_ParCSRMatrix **matSptr, hypre_CSRMatrix **Eptr, hypre_CSRMatrix **Fptr,
                           HYPRE_Int **A_fake_diag_ip)
{
   /* GPU-accelerated ILU0 with cusparse */
   HYPRE_Int               i, j, k1, k2, k3, col;

   /* communication stuffs for S */
   MPI_Comm                comm                 = hypre_ParCSRMatrixComm(A);

   HYPRE_Int               my_id, num_procs;
   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   hypre_ParCSRCommPkg     *comm_pkg;
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int               num_sends, begin, end;
   HYPRE_BigInt            *send_buf            = NULL;
   HYPRE_Int               *rperm               = NULL;
   HYPRE_Int               *rqperm              = NULL;

   hypre_ParCSRMatrix      *matS                = NULL;
   hypre_CSRMatrix         *A_diag              = NULL;
   HYPRE_Int               *A_fake_diag_i       = NULL;
   hypre_CSRMatrix         *A_offd              = NULL;
   HYPRE_Int               *A_offd_i            = NULL;
   HYPRE_Int               *A_offd_j            = NULL;
   HYPRE_Real              *A_offd_data         = NULL;
   hypre_CSRMatrix         *SLU                 = NULL;
   /* pointers to cusparse data */
   csrsv2Info_t            matBL_info           = NULL;
   csrsv2Info_t            matBU_info           = NULL;
   csrsv2Info_t            matSL_info           = NULL;
   csrsv2Info_t            matSU_info           = NULL;

   HYPRE_Int               buffer_size          = 0;
   void                    *buffer              = NULL;

   /* variables for matS */
   HYPRE_Int               m                    = n - nLU;
   HYPRE_Int               nI                   = nLU;//use default
   HYPRE_Int               e                    = 0;
   HYPRE_Int               m_e                  = m;
   HYPRE_BigInt            total_rows;
   HYPRE_BigInt            col_starts[2];
   HYPRE_Int               *S_diag_i            = NULL;
   HYPRE_Int               S_diag_nnz;
   hypre_CSRMatrix         *S_offd              = NULL;
   HYPRE_Int               *S_offd_i            = NULL;
   HYPRE_Int               *S_offd_j            = NULL;
   HYPRE_Real              *S_offd_data         = NULL;
   HYPRE_BigInt            *S_offd_colmap       = NULL;
   HYPRE_Int               S_offd_nnz;
   HYPRE_Int               S_offd_ncols;

   /* set data slots */
   A_offd                                       = hypre_ParCSRMatrixOffd(A);
   A_offd_i                                     = hypre_CSRMatrixI(A_offd);
   A_offd_j                                     = hypre_CSRMatrixJ(A_offd);
   A_offd_data                                  = hypre_CSRMatrixData(A_offd);

   /* unfortunately we need to build the reverse permutation array */
   rperm                                        = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   rqperm                                       = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      rperm[perm[i]] = i;
      rqperm[qperm[i]] = i;
   }

   /* Only call ILU when we really have a matrix on this processor */
   if (n > 0)
   {
      /* Copy diagonal matrix into a new place with permutation
       * That is, A_diag = A_diag(perm,qperm);
       */
      hypre_ParILUCusparseExtractDiagonalCSR(A, perm, rqperm, &A_diag);

      /* Apply ILU factorization to the entile A_diag */
      HYPRE_ILUSetupCusparseCSRILU0(A_diag, ilu_solve_policy);

      /* | L \ U (B) L^{-1}F  |
       * | EU^{-1}   L \ U (S)|
       * Extract submatrix L_B U_B, L_S U_S, EU_B^{-1}, L_B^{-1}F
       * Note that in this function after ILU, all rows are sorted
       * in a way different than HYPRE. Diagonal is not listed in the front
       */
      hypre_ParILUCusparseILUExtractEBFC(A_diag, nLU, BLUptr, &SLU, Eptr, Fptr);
   }
   else
   {
      *BLUptr = NULL;
      *Eptr = NULL;
      *Fptr = NULL;
      SLU = NULL;
   }

   /* create B */
   /* only analyse when nacessary */
   if ( nLU > 0 )
   {
      /* Analysis of BILU */
      HYPRE_ILUSetupCusparseCSRILU0SetupSolve(*BLUptr, matL_des, matU_des,
                                 ilu_solve_policy, &matBL_info, &matBU_info,
                                 &buffer_size, &buffer);
   }

   HYPRE_BigInt big_m = (HYPRE_BigInt)m;
   hypre_MPI_Allreduce(&big_m, &total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
   /* only form when total_rows > 0 */
   if ( total_rows > 0 )
   {
      /* now create S */
      /* need to get new column start */
      {
         HYPRE_BigInt global_start;
         hypre_MPI_Scan( &big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
         col_starts[0] = global_start - m;
         col_starts[1] = global_start;
      }

      A_fake_diag_i = hypre_CTAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
      if (SLU)
      {
         /* Analysis of SILU */
         HYPRE_ILUSetupCusparseCSRILU0SetupSolve(SLU, matL_des, matU_des,
                                       ilu_solve_policy, &matSL_info, &matSU_info,
                                       &buffer_size, &buffer);
      }
      else
      {
         SLU = hypre_CSRMatrixCreate(0,0,0);
         hypre_CSRMatrixInitialize(SLU);
      }
      S_diag_i = hypre_CSRMatrixI(SLU);
      S_diag_nnz = S_diag_i[m];
      /* Build ParCSRMatrix matS
       * For example when np == 3 the new matrix takes the following form
       * |IS_1 E_12 E_13|
       * |E_21 IS_2 E_22| = S
       * |E_31 E_32 IS_3|
       * In which IS_i is the cusparse ILU factorization of S_i in one matrix
       * */

      /* We did nothing to A_offd, so all the data kept, just reorder them
       * The create function takes comm, global num rows/cols,
       *    row/col start, num cols offd, nnz diag, nnz offd
       */
      S_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);
      S_offd_ncols = hypre_CSRMatrixNumCols(A_offd);

      matS = hypre_ParCSRMatrixCreate( comm,
                           total_rows,
                           total_rows,
                           col_starts,
                           col_starts,
                           S_offd_ncols,
                           S_diag_nnz,
                           S_offd_nnz);

      /* first put diagonal data in */
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matS));
      hypre_ParCSRMatrixDiag(matS) = SLU;

      /* now start to construct offdiag of S */
      S_offd = hypre_ParCSRMatrixOffd(matS);
      S_offd_i = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_colmap = hypre_CTAlloc(HYPRE_BigInt, S_offd_ncols, HYPRE_MEMORY_HOST);

      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i[i] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col+1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i+1+e] = k3;
      }

      /* give I, J, DATA to S_offd */
      hypre_CSRMatrixI(S_offd) = S_offd_i;
      hypre_CSRMatrixJ(S_offd) = S_offd_j;
      hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      /* setup comm_pkg if not yet built */
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      }
      /* get total num of send */
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg,0);
      end = hypre_ParCSRCommPkgSendMapStart(comm_pkg,num_sends);
      send_buf = hypre_TAlloc(HYPRE_BigInt, end - begin, HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for (i = begin; i < end; i++)
      {
         send_buf[i-begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)] - nLU + col_starts[0];
      }

      /* main communication */
      comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf, S_offd_colmap);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;

      hypre_ILUSortOffdColmap(matS);

      /* free */
      hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
   } /* end of forming S */

   *matSptr       = matS;
   *bufferp       = buffer;
   *matBL_infop   = matBL_info;
   *matBU_infop   = matBU_info;
   *matSL_infop   = matSL_info;
   *matSU_infop   = matSU_info;
   *A_fake_diag_ip= A_fake_diag_i;

   /* Destroy the bridge after acrossing the river */
   hypre_CSRMatrixDestroy(A_diag);
   hypre_TFree(rperm, HYPRE_MEMORY_HOST);
   hypre_TFree(rqperm, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

HYPRE_Int
hypre_ILUSetupILUKDevice(hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Int *perm, HYPRE_Int *qperm, HYPRE_Int n, HYPRE_Int nLU,
                           cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des, cusparseSolvePolicy_t ilu_solve_policy,
                           void **bufferp, csrsv2Info_t *matBL_infop, csrsv2Info_t *matBU_infop,
                           csrsv2Info_t *matSL_infop, csrsv2Info_t *matSU_infop,
                           hypre_CSRMatrix **BLUptr, hypre_ParCSRMatrix **matSptr, hypre_CSRMatrix **Eptr, hypre_CSRMatrix **Fptr,
                           HYPRE_Int **A_fake_diag_ip)
{
   /* GPU-accelerated ILU0 with cusparse */
   HYPRE_Int               i, j, k1, k2, k3, col;

   /* communication stuffs for S */
   MPI_Comm                comm                 = hypre_ParCSRMatrixComm(A);

   HYPRE_Int               my_id, num_procs;
   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   hypre_ParCSRCommPkg     *comm_pkg;
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int               num_sends, begin, end;
   HYPRE_BigInt            *send_buf            = NULL;
   HYPRE_Int               *rperm               = NULL;
   HYPRE_Int               *rqperm              = NULL;

   hypre_ParCSRMatrix      *Apq                 = NULL;
   hypre_ParCSRMatrix      *ALU                 = NULL;

   hypre_ParCSRMatrix      *matS                = NULL;
   hypre_CSRMatrix         *A_diag              = NULL;
   HYPRE_Int               *A_fake_diag_i       = NULL;
   hypre_CSRMatrix         *A_offd              = NULL;
   HYPRE_Int               *A_offd_i            = NULL;
   HYPRE_Int               *A_offd_j            = NULL;
   HYPRE_Real              *A_offd_data         = NULL;
   hypre_CSRMatrix         *SLU                 = NULL;
   /* pointers to cusparse data */
   csrsv2Info_t            matBL_info           = NULL;
   csrsv2Info_t            matBU_info           = NULL;
   csrsv2Info_t            matSL_info           = NULL;
   csrsv2Info_t            matSU_info           = NULL;

   HYPRE_Int               buffer_size          = 0;
   void                    *buffer              = NULL;

   /* variables for matS */
   HYPRE_Int               m                    = n - nLU;
   HYPRE_Int               nI                   = nLU;//use default
   HYPRE_Int               e                    = 0;
   HYPRE_Int               m_e                  = m;
   HYPRE_BigInt            total_rows;
   HYPRE_BigInt            col_starts[2];
   HYPRE_Int               *S_diag_i            = NULL;
   HYPRE_Int               S_diag_nnz;
   hypre_CSRMatrix         *S_offd              = NULL;
   HYPRE_Int               *S_offd_i            = NULL;
   HYPRE_Int               *S_offd_j            = NULL;
   HYPRE_Real              *S_offd_data         = NULL;
   HYPRE_BigInt            *S_offd_colmap       = NULL;
   HYPRE_Int               S_offd_nnz;
   HYPRE_Int               S_offd_ncols;

   /* set data slots */
   A_offd                                       = hypre_ParCSRMatrixOffd(A);
   A_offd_i                                     = hypre_CSRMatrixI(A_offd);
   A_offd_j                                     = hypre_CSRMatrixJ(A_offd);
   A_offd_data                                  = hypre_CSRMatrixData(A_offd);

   hypre_ParCSRMatrix      *parL = NULL;
   hypre_ParCSRMatrix      *parU = NULL;
   hypre_ParCSRMatrix      *parS = NULL;
   HYPRE_Real              *parD = NULL;
   HYPRE_Int               *uend = NULL;

   /* unfortunately we need to build the reverse permutation array */
   rperm                                        = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   rqperm                                       = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      rperm[perm[i]] = i;
      rqperm[qperm[i]] = i;
   }

   /* Only call ILU when we really have a matrix on this processor */
   if (n > 0)
   {
      /* Copy diagonal matrix into a new place with permutation
       * That is, A_diag = A_diag(perm,qperm);
       */
      hypre_ParILURAPReorder( A, perm, rqperm, &Apq);

      /* Apply ILU factorization to the entile A_diag */
      hypre_ILUSetupILUK(Apq, lfil, NULL, NULL, n, n, &parL, &parD, &parU, &parS, &uend);

      if (uend)
      {
         hypre_TFree(uend, HYPRE_MEMORY_HOST);
      }

      if (parS)
      {
         hypre_ParCSRMatrixDestroy(parS);
      }

      /* | L \ U (B) L^{-1}F  |
       * | EU^{-1}   L \ U (S)|
       * Extract submatrix L_B U_B, L_S U_S, EU_B^{-1}, L_B^{-1}F
       * Note that in this function after ILU, all rows are sorted
       * in a way different than HYPRE. Diagonal is not listed in the front
       */
      hypre_ILUSetupLDUtoCusparse( parL, parD, parU, &ALU);

      if (parL)
      {
         hypre_ParCSRMatrixDestroy(parL);
      }
      if (parD)
      {
         hypre_TFree(parD, HYPRE_MEMORY_DEVICE);
      }
      if (parU)
      {
         hypre_ParCSRMatrixDestroy(parU);
      }

      A_diag = hypre_ParCSRMatrixDiag(ALU);

      hypre_ParILUCusparseILUExtractEBFC(A_diag, nLU, BLUptr, &SLU, Eptr, Fptr);

      if (Apq)
      {
         hypre_ParCSRMatrixDestroy(Apq);
      }

   }
   else
   {
      *BLUptr = NULL;
      *Eptr = NULL;
      *Fptr = NULL;
      SLU = NULL;
   }

   /* create B */
   /* only analyse when nacessary */
   if ( nLU > 0 )
   {
      /* Analysis of BILU */
      HYPRE_ILUSetupCusparseCSRILU0SetupSolve(*BLUptr, matL_des, matU_des,
                                 ilu_solve_policy, &matBL_info, &matBU_info,
                                 &buffer_size, &buffer);
   }

   HYPRE_BigInt big_m = (HYPRE_BigInt)m;
   hypre_MPI_Allreduce(&big_m, &total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
   /* only form when total_rows > 0 */
   if ( total_rows > 0 )
   {
      /* now create S */
      /* need to get new column start */
      {
         HYPRE_BigInt global_start;
         hypre_MPI_Scan( &big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
         col_starts[0] = global_start - m;
         col_starts[1] = global_start;
      }

      A_fake_diag_i = hypre_CTAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
      if (SLU)
      {
         /* Analysis of SILU */
         HYPRE_ILUSetupCusparseCSRILU0SetupSolve(SLU, matL_des, matU_des,
                                       ilu_solve_policy, &matSL_info, &matSU_info,
                                       &buffer_size, &buffer);
      }
      else
      {
         SLU = hypre_CSRMatrixCreate(0,0,0);
         hypre_CSRMatrixInitialize(SLU);
      }
      S_diag_i = hypre_CSRMatrixI(SLU);
      S_diag_nnz = S_diag_i[m];
      /* Build ParCSRMatrix matS
       * For example when np == 3 the new matrix takes the following form
       * |IS_1 E_12 E_13|
       * |E_21 IS_2 E_22| = S
       * |E_31 E_32 IS_3|
       * In which IS_i is the cusparse ILU factorization of S_i in one matrix
       * */

      /* We did nothing to A_offd, so all the data kept, just reorder them
       * The create function takes comm, global num rows/cols,
       *    row/col start, num cols offd, nnz diag, nnz offd
       */
      S_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);
      S_offd_ncols = hypre_CSRMatrixNumCols(A_offd);

      matS = hypre_ParCSRMatrixCreate( comm,
                           total_rows,
                           total_rows,
                           col_starts,
                           col_starts,
                           S_offd_ncols,
                           S_diag_nnz,
                           S_offd_nnz);

      /* first put diagonal data in */
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matS));
      hypre_ParCSRMatrixDiag(matS) = SLU;

      /* now start to construct offdiag of S */
      S_offd = hypre_ParCSRMatrixOffd(matS);
      S_offd_i = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_colmap = hypre_CTAlloc(HYPRE_BigInt, S_offd_ncols, HYPRE_MEMORY_HOST);

      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i[i] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col+1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i+1+e] = k3;
      }

      /* give I, J, DATA to S_offd */
      hypre_CSRMatrixI(S_offd) = S_offd_i;
      hypre_CSRMatrixJ(S_offd) = S_offd_j;
      hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      /* setup comm_pkg if not yet built */
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      }
      /* get total num of send */
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg,0);
      end = hypre_ParCSRCommPkgSendMapStart(comm_pkg,num_sends);
      send_buf = hypre_TAlloc(HYPRE_BigInt, end - begin, HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for (i = begin; i < end; i++)
      {
         send_buf[i-begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)] - nLU + col_starts[0];
      }

      /* main communication */
      comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf, S_offd_colmap);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;

      hypre_ILUSortOffdColmap(matS);

      /* free */
      hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
   } /* end of forming S */

   *matSptr       = matS;
   *bufferp       = buffer;
   *matBL_infop   = matBL_info;
   *matBU_infop   = matBU_info;
   *matSL_infop   = matSL_info;
   *matSU_infop   = matSU_info;
   *A_fake_diag_ip= A_fake_diag_i;

   /* Destroy the bridge after acrossing the river */
   hypre_CSRMatrixDestroy(A_diag);
   hypre_TFree(rperm, HYPRE_MEMORY_HOST);
   hypre_TFree(rqperm, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}


HYPRE_Int
hypre_ILUSetupILUTDevice(hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Real *tol, HYPRE_Int *perm, HYPRE_Int *qperm, HYPRE_Int n, HYPRE_Int nLU,
                           cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des, cusparseSolvePolicy_t ilu_solve_policy,
                           void **bufferp, csrsv2Info_t *matBL_infop, csrsv2Info_t *matBU_infop,
                           csrsv2Info_t *matSL_infop, csrsv2Info_t *matSU_infop,
                           hypre_CSRMatrix **BLUptr, hypre_ParCSRMatrix **matSptr, hypre_CSRMatrix **Eptr, hypre_CSRMatrix **Fptr,
                           HYPRE_Int **A_fake_diag_ip)
{
   /* GPU-accelerated ILU0 with cusparse */
   HYPRE_Int               i, j, k1, k2, k3, col;

   /* communication stuffs for S */
   MPI_Comm                comm                 = hypre_ParCSRMatrixComm(A);

   HYPRE_Int               my_id, num_procs;
   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   hypre_ParCSRCommPkg     *comm_pkg;
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int               num_sends, begin, end;
   HYPRE_BigInt            *send_buf            = NULL;
   HYPRE_Int               *rperm               = NULL;
   HYPRE_Int               *rqperm              = NULL;

   hypre_ParCSRMatrix      *Apq                 = NULL;
   hypre_ParCSRMatrix      *ALU                 = NULL;

   hypre_ParCSRMatrix      *matS                = NULL;
   hypre_CSRMatrix         *A_diag              = NULL;
   HYPRE_Int               *A_fake_diag_i       = NULL;
   hypre_CSRMatrix         *A_offd              = NULL;
   HYPRE_Int               *A_offd_i            = NULL;
   HYPRE_Int               *A_offd_j            = NULL;
   HYPRE_Real              *A_offd_data         = NULL;
   hypre_CSRMatrix         *SLU                 = NULL;
   /* pointers to cusparse data */
   csrsv2Info_t            matBL_info           = NULL;
   csrsv2Info_t            matBU_info           = NULL;
   csrsv2Info_t            matSL_info           = NULL;
   csrsv2Info_t            matSU_info           = NULL;

   HYPRE_Int               buffer_size          = 0;
   void                    *buffer              = NULL;

   /* variables for matS */
   HYPRE_Int               m                    = n - nLU;
   HYPRE_Int               nI                   = nLU;//use default
   HYPRE_Int               e                    = 0;
   HYPRE_Int               m_e                  = m;
   HYPRE_BigInt            total_rows;
   HYPRE_BigInt            col_starts[2];
   HYPRE_Int               *S_diag_i            = NULL;
   HYPRE_Int               S_diag_nnz;
   hypre_CSRMatrix         *S_offd              = NULL;
   HYPRE_Int               *S_offd_i            = NULL;
   HYPRE_Int               *S_offd_j            = NULL;
   HYPRE_Real              *S_offd_data         = NULL;
   HYPRE_BigInt            *S_offd_colmap       = NULL;
   HYPRE_Int               S_offd_nnz;
   HYPRE_Int               S_offd_ncols;

   /* set data slots */
   A_offd                                       = hypre_ParCSRMatrixOffd(A);
   A_offd_i                                     = hypre_CSRMatrixI(A_offd);
   A_offd_j                                     = hypre_CSRMatrixJ(A_offd);
   A_offd_data                                  = hypre_CSRMatrixData(A_offd);

   hypre_ParCSRMatrix      *parL = NULL;
   hypre_ParCSRMatrix      *parU = NULL;
   hypre_ParCSRMatrix      *parS = NULL;
   HYPRE_Real              *parD = NULL;
   HYPRE_Int               *uend = NULL;

   /* unfortunately we need to build the reverse permutation array */
   rperm                                        = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   rqperm                                       = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      rperm[perm[i]] = i;
      rqperm[qperm[i]] = i;
   }

   /* Only call ILU when we really have a matrix on this processor */
   if (n > 0)
   {
      /* Copy diagonal matrix into a new place with permutation
       * That is, A_diag = A_diag(perm,qperm);
       */
      hypre_ParILURAPReorder( A, perm, rqperm, &Apq);

      /* Apply ILU factorization to the entile A_diag */
      hypre_ILUSetupILUT(Apq, lfil, tol, NULL, NULL, n, n, &parL, &parD, &parU, &parS, &uend);

      if (uend)
      {
         hypre_TFree(uend, HYPRE_MEMORY_HOST);
      }

      if (parS)
      {
         hypre_ParCSRMatrixDestroy(parS);
      }

      /* | L \ U (B) L^{-1}F  |
       * | EU^{-1}   L \ U (S)|
       * Extract submatrix L_B U_B, L_S U_S, EU_B^{-1}, L_B^{-1}F
       * Note that in this function after ILU, all rows are sorted
       * in a way different than HYPRE. Diagonal is not listed in the front
       */
      hypre_ILUSetupLDUtoCusparse( parL, parD, parU, &ALU);

      if (parL)
      {
         hypre_ParCSRMatrixDestroy(parL);
      }
      if (parD)
      {
         hypre_TFree(parD, HYPRE_MEMORY_DEVICE);
      }
      if (parU)
      {
         hypre_ParCSRMatrixDestroy(parU);
      }

      A_diag = hypre_ParCSRMatrixDiag(ALU);

      hypre_ParILUCusparseILUExtractEBFC(A_diag, nLU, BLUptr, &SLU, Eptr, Fptr);

      if (Apq)
      {
         hypre_ParCSRMatrixDestroy(Apq);
      }

   }
   else
   {
      *BLUptr = NULL;
      *Eptr = NULL;
      *Fptr = NULL;
      SLU = NULL;
   }

   /* create B */
   /* only analyse when nacessary */
   if ( nLU > 0 )
   {
      /* Analysis of BILU */
      HYPRE_ILUSetupCusparseCSRILU0SetupSolve(*BLUptr, matL_des, matU_des,
                                 ilu_solve_policy, &matBL_info, &matBU_info,
                                 &buffer_size, &buffer);
   }

   HYPRE_BigInt big_m = (HYPRE_BigInt)m;
   hypre_MPI_Allreduce(&big_m, &total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
   /* only form when total_rows > 0 */
   if ( total_rows > 0 )
   {
      /* now create S */
      /* need to get new column start */
      {
         HYPRE_BigInt global_start;
         hypre_MPI_Scan( &big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
         col_starts[0] = global_start - m;
         col_starts[1] = global_start;
      }

      A_fake_diag_i = hypre_CTAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_DEVICE);
      if (SLU)
      {
         /* Analysis of SILU */
         HYPRE_ILUSetupCusparseCSRILU0SetupSolve(SLU, matL_des, matU_des,
                                       ilu_solve_policy, &matSL_info, &matSU_info,
                                       &buffer_size, &buffer);
      }
      else
      {
         SLU = hypre_CSRMatrixCreate(0,0,0);
         hypre_CSRMatrixInitialize(SLU);
      }
      S_diag_i = hypre_CSRMatrixI(SLU);
      S_diag_nnz = S_diag_i[m];
      /* Build ParCSRMatrix matS
       * For example when np == 3 the new matrix takes the following form
       * |IS_1 E_12 E_13|
       * |E_21 IS_2 E_22| = S
       * |E_31 E_32 IS_3|
       * In which IS_i is the cusparse ILU factorization of S_i in one matrix
       * */

      /* We did nothing to A_offd, so all the data kept, just reorder them
       * The create function takes comm, global num rows/cols,
       *    row/col start, num cols offd, nnz diag, nnz offd
       */
      S_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);
      S_offd_ncols = hypre_CSRMatrixNumCols(A_offd);

      matS = hypre_ParCSRMatrixCreate( comm,
                           total_rows,
                           total_rows,
                           col_starts,
                           col_starts,
                           S_offd_ncols,
                           S_diag_nnz,
                           S_offd_nnz);

      /* first put diagonal data in */
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matS));
      hypre_ParCSRMatrixDiag(matS) = SLU;

      /* now start to construct offdiag of S */
      S_offd = hypre_ParCSRMatrixOffd(matS);
      S_offd_i = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_colmap = hypre_CTAlloc(HYPRE_BigInt, S_offd_ncols, HYPRE_MEMORY_HOST);

      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i[i] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col+1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i+1+e] = k3;
      }

      /* give I, J, DATA to S_offd */
      hypre_CSRMatrixI(S_offd) = S_offd_i;
      hypre_CSRMatrixJ(S_offd) = S_offd_j;
      hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      /* setup comm_pkg if not yet built */
      if (!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      }
      /* get total num of send */
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg,0);
      end = hypre_ParCSRCommPkgSendMapStart(comm_pkg,num_sends);
      send_buf = hypre_TAlloc(HYPRE_BigInt, end - begin, HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for (i = begin; i < end; i++)
      {
         send_buf[i-begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)] - nLU + col_starts[0];
      }

      /* main communication */
      comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf, S_offd_colmap);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;

      hypre_ILUSortOffdColmap(matS);

      /* free */
      hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
   } /* end of forming S */

   *matSptr       = matS;
   *bufferp       = buffer;
   *matBL_infop   = matBL_info;
   *matBU_infop   = matBU_info;
   *matSL_infop   = matSL_info;
   *matSU_infop   = matSU_info;
   *A_fake_diag_ip= A_fake_diag_i;

   /* Destroy the bridge after acrossing the river */
   hypre_CSRMatrixDestroy(A_diag);
   hypre_TFree(rperm, HYPRE_MEMORY_HOST);
   hypre_TFree(rqperm, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/* Reorder matrix A based on local permutation (combine local permutation into global permutation)
 * WARNING: We don't put diagonal to the first entry of each row
 * A = input matrix
 * perm = permutation array indicating ordering of rows. Perm could come from a
 *    CF_marker array or a reordering routine.
 * rqperm = reverse permutation array indicating ordering of columns
 * A_pq = pointer to the output par CSR matrix.
 */
HYPRE_Int
hypre_ParILURAPReorder(hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int *rqperm, hypre_ParCSRMatrix **A_pq)
{
   /* Get necessary slots */
   hypre_CSRMatrix     *A_diag         = hypre_ParCSRMatrixDiag(A);
   //HYPRE_Int           *A_diag_i       = hypre_CSRMatrixI(A_diag);
   //HYPRE_Int           *A_diag_j       = hypre_CSRMatrixJ(A_diag);
   //HYPRE_Real          *A_diag_data    = hypre_CSRMatrixData(A_diag);
   HYPRE_Int            n              = hypre_CSRMatrixNumRows(A_diag);
   //HYPRE_Int            nnz_A_diag     = A_diag_i[n];

   //HYPRE_Int            i, j, current_idx;
   HYPRE_Int            i;

   /* MPI */
   MPI_Comm             comm                 = hypre_ParCSRMatrixComm(A);
   HYPRE_Int            num_procs,  my_id;

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   /* Create permutation matrices P = I(perm,:) and Q(rqperm,:), such that Apq = PAQ */
   hypre_ParCSRMatrix *P, *Q, *PAQ, *PA;

   hypre_CSRMatrix *P_diag, *Q_diag;
   hypre_CSRMatrix *P_offd, *Q_offd;

   P = hypre_ParCSRMatrixCreate( comm,
                           hypre_ParCSRMatrixGlobalNumRows(A),
                           hypre_ParCSRMatrixGlobalNumRows(A),
                           hypre_ParCSRMatrixRowStarts(A),
                           hypre_ParCSRMatrixColStarts(A),
                           0,
                           n,
                           0);

   Q = hypre_ParCSRMatrixCreate( comm,
                           hypre_ParCSRMatrixGlobalNumRows(A),
                           hypre_ParCSRMatrixGlobalNumRows(A),
                           hypre_ParCSRMatrixRowStarts(A),
                           hypre_ParCSRMatrixColStarts(A),
                           0,
                           n,
                           0);

   P_diag = hypre_ParCSRMatrixDiag(P);
   Q_diag = hypre_ParCSRMatrixDiag(Q);
   P_offd = hypre_ParCSRMatrixOffd(P);
   Q_offd = hypre_ParCSRMatrixOffd(Q);

   HYPRE_Int   *P_diag_i, *P_diag_j, *Q_diag_i, *Q_diag_j;
   HYPRE_Real  *P_diag_data, *Q_diag_data;
   HYPRE_Int   *P_offd_i, *Q_offd_i;

   P_diag_i = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_DEVICE);
   P_diag_j = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
   P_diag_data = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_DEVICE);

   Q_diag_i = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_DEVICE);
   Q_diag_j = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
   Q_diag_data = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_DEVICE);

   /* fill data, openmp should be availiable here */
   for (i = 0; i < n; i++)
   {
      P_diag_i[i] = i;
      P_diag_j[i] = perm[i];
      P_diag_data[i] = 1.0;

      Q_diag_i[i] = i;
      Q_diag_j[i] = rqperm[i];
      Q_diag_data[i] = 1.0;

   }
   P_diag_i[n] = n;
   Q_diag_i[n] = n;

   /* give I, J, DATA */
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   hypre_CSRMatrixData(P_diag) = P_diag_data;

   hypre_CSRMatrixI(Q_diag) = Q_diag_i;
   hypre_CSRMatrixJ(Q_diag) = Q_diag_j;
   hypre_CSRMatrixData(Q_diag) = Q_diag_data;

   P_offd_i = hypre_CTAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_DEVICE);
   Q_offd_i = hypre_CTAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_DEVICE);

   hypre_CSRMatrixI(P_offd) = P_offd_i;
   hypre_CSRMatrixI(Q_offd) = Q_offd_i;

   /* Update A */
   PA = hypre_ParCSRMatMat(P, A);
   PAQ = hypre_ParCSRMatMat(PA, Q);
   //PAQ = hypre_ParCSRMatrixRAPKT(P, A, Q, 0);

   /* free and return */
   hypre_ParCSRMatrixDestroy(P);
   hypre_ParCSRMatrixDestroy(Q);

   *A_pq = PAQ;

   return hypre_error_flag;
}

/* Convert the L, D, U style to the cusparse style
 * Assume the diagonal of L and U are the ilu factorization, directly combine them
 */
HYPRE_Int
hypre_ParILURAPBuildRP(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix *BLUm, hypre_ParCSRMatrix* E, hypre_ParCSRMatrix *F,
                        cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des, hypre_ParCSRMatrix **Rp, hypre_ParCSRMatrix **Pp)
{
   /* declare variables */
   HYPRE_Int            j, row, col;
   HYPRE_Real           val;
   hypre_ParCSRMatrix   *R, *P;
   hypre_CSRMatrix      *R_diag, *P_diag;

   hypre_CSRMatrix      *BLUm_diag           = hypre_ParCSRMatrixDiag(BLUm);
   HYPRE_Int            *BLUm_diag_i         = hypre_CSRMatrixI(BLUm_diag);
   HYPRE_Int            *BLUm_diag_j         = hypre_CSRMatrixJ(BLUm_diag);
   HYPRE_Real           *BLUm_diag_data      = hypre_CSRMatrixData(BLUm_diag);

   hypre_CSRMatrix      *E_diag              = hypre_ParCSRMatrixDiag(E);
   HYPRE_Int            *E_diag_i            = hypre_CSRMatrixI(E_diag);
   HYPRE_Int            *E_diag_j            = hypre_CSRMatrixJ(E_diag);
   HYPRE_Real           *E_diag_data         = hypre_CSRMatrixData(E_diag);
   hypre_CSRMatrix      *F_diag              = hypre_ParCSRMatrixDiag(F);
   HYPRE_Int            *F_diag_i            = hypre_CSRMatrixI(F_diag);
   HYPRE_Int            *F_diag_j            = hypre_CSRMatrixJ(F_diag);
   HYPRE_Real           *F_diag_data         = hypre_CSRMatrixData(F_diag);

   HYPRE_Int            n                    = hypre_CSRMatrixNumRows(F_diag);
   HYPRE_Int            m                    = hypre_CSRMatrixNumCols(F_diag);

   HYPRE_Int            nnz_BLUm             = BLUm_diag_i[n];

   /* MPI */
   MPI_Comm             comm                 = hypre_ParCSRMatrixComm(A);
   HYPRE_Int            num_procs,  my_id;

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   /* cusparse */
   HYPRE_Int               isDoublePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int               isSinglePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   hypre_assert(isDoublePrecision || isSinglePrecision);

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());

   /* compute P = -UB\(LB\F)
    * op(A) * op(X) = \alpha op(B)
    * first iLF = LB\F -> LB*iLF = F
    */

   HYPRE_Int               algo = 0;
   HYPRE_Real              alpha = 1.0;
   HYPRE_Real              *rhs;
   cusparseSolvePolicy_t   policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
   size_t                  buffer_size, buffer_size_old;
   void                    *buffer;
   csrsm2Info_t            malL_info = NULL;
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrsm2Info(&malL_info));

   rhs = hypre_CTAlloc(HYPRE_Real, m * n, HYPRE_MEMORY_DEVICE);

   /* fill data, note that rhs is in Fortan style (col first)
    * oprating by col is slow, but
    */
   for (row = 0; row < n; row++)
   {
      for (j = F_diag_i[row]; j < F_diag_i[row+1]; j++)
      {
         col = F_diag_j[j];
         *(rhs + col*n + row) = F_diag_data[j];
      }
   }

   /* check buffer size and create buffer */

   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrsm2_bufferSizeExt( handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         n, m, nnz_BLUm, (hypre_double *)&alpha, matL_des, (hypre_double *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (hypre_double *)rhs, n, malL_info, policy, &buffer_size));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrsm2_bufferSizeExt( handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         n, m, nnz_BLUm, (float *)&alpha, matL_des, (float *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (float *)rhs, n, malL_info, policy, &buffer_size));
   }

   buffer = hypre_MAlloc(buffer_size, HYPRE_MEMORY_DEVICE);

   /* analysis */

   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrsm2_analysis( handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         n, m, nnz_BLUm, (hypre_double *)&alpha, matL_des, (hypre_double *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (hypre_double *)rhs, n, malL_info, policy, buffer));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrsm2_analysis( handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         n, m, nnz_BLUm, (float *)&alpha, matL_des, (float *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (float *)rhs, n, malL_info, policy, buffer));
   }

   /* solve phase */
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrsm2_solve( handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         n, m, nnz_BLUm, (hypre_double *)&alpha, matL_des, (hypre_double *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (hypre_double *)rhs, n, malL_info, policy, buffer));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrsm2_solve( handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         n, m, nnz_BLUm, (float *)&alpha, matL_des, (float *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (float *)rhs, n, malL_info, policy, buffer));
   }
   /* now P = -UB\(LB\F) -> UB*P = -(LB\F)
    */
   alpha = -1.0;
   csrsm2Info_t            malU_info = NULL;
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrsm2Info(&malU_info));

   buffer_size_old = buffer_size;

   /* check buffer size and create buffer */

   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrsm2_bufferSizeExt( handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         n, m, nnz_BLUm, (hypre_double *)&alpha, matU_des, (hypre_double *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (hypre_double *)rhs, n, malU_info, policy, &buffer_size));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrsm2_bufferSizeExt( handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         n, m, nnz_BLUm, (float *)&alpha, matU_des, (float *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (float *)rhs, n, malU_info, policy, &buffer_size));
   }

   if (buffer_size > buffer_size_old)
   {
      buffer = hypre_ReAlloc_v2(buffer, buffer_size_old, buffer_size, HYPRE_MEMORY_DEVICE);
      buffer_size_old = buffer_size;
   }

   /* analysis */

   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrsm2_analysis( handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         n, m, nnz_BLUm, (hypre_double *)&alpha, matU_des, (hypre_double *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (hypre_double *)rhs, n, malU_info, policy, buffer));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrsm2_analysis( handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         n, m, nnz_BLUm, (float *)&alpha, matU_des, (float *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (float *)rhs, n, malU_info, policy, buffer));
   }

   /* solve phase */
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrsm2_solve( handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         n, m, nnz_BLUm, (hypre_double *)&alpha, matU_des, (hypre_double *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (hypre_double *)rhs, n, malU_info, policy, buffer));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrsm2_solve( handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         n, m, nnz_BLUm, (float *)&alpha, matU_des, (float *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (float *)rhs, n, malU_info, policy, buffer));
   }
   /* wait till GPU done to copy data */
   cudaDeviceSynchronize();
   /* now form P, (n + m) * m */
   HYPRE_Real           drop_tol = 1e-06;
   HYPRE_Int            ctrP = 0;
   HYPRE_Int            *P_diag_i;
   HYPRE_Int            *P_offd_i;
   HYPRE_Int            *P_diag_j;
   HYPRE_Real           *P_diag_data;

   HYPRE_Int             capacity_P = nnz_BLUm + m;

   P_diag_i       = hypre_TAlloc(HYPRE_Int, n+m+1, HYPRE_MEMORY_DEVICE);
   P_offd_i       = hypre_CTAlloc(HYPRE_Int, n+m+1, HYPRE_MEMORY_DEVICE);
   P_diag_j       = hypre_TAlloc(HYPRE_Int, capacity_P, HYPRE_MEMORY_DEVICE);
   P_diag_data    = hypre_TAlloc(HYPRE_Real, capacity_P, HYPRE_MEMORY_DEVICE);

   for (row = 0; row < n; row++)
   {
      P_diag_i[row] = ctrP;
      for (col = 0; col < m; col++)
      {
         val = *(rhs + col*n + row);
         if (hypre_abs(val) > drop_tol)
         {
            if (ctrP >= capacity_P)
            {
               HYPRE_Int tmp;
               tmp = capacity_P;
               capacity_P = capacity_P * EXPAND_FACT;
               P_diag_j       = hypre_TReAlloc_v2(P_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_P, HYPRE_MEMORY_DEVICE);
               P_diag_data    = hypre_TReAlloc_v2(P_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_P, HYPRE_MEMORY_DEVICE);
            }
            P_diag_j[ctrP] = col;
            P_diag_data[ctrP++] = val;
         }
      }
   }

   if (ctrP + m >= capacity_P)
   {
      HYPRE_Int tmp;
      tmp = capacity_P;
      capacity_P = ctrP + m;
      P_diag_j       = hypre_TReAlloc_v2(P_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_P, HYPRE_MEMORY_DEVICE);
      P_diag_data    = hypre_TReAlloc_v2(P_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_P, HYPRE_MEMORY_DEVICE);
   }

   for (row = 0; row < m; row++)
   {
      P_diag_i[row+n] = ctrP;
      P_diag_j[ctrP] = row;
      P_diag_data[ctrP++] = 1.0;
   }

   P_diag_i[m+n] = ctrP;

   /* now start to form R = - (E / UB ) / LB
    * first EiUB = E / UB -> UB'*EiUB'=E'
    */
   alpha = 1.0;
   csrsm2Info_t            malU_info2 = NULL;
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrsm2Info(&malU_info2));

   /* fill data, note that rhs is in Fortan style (col first)
    * oprating by col is slow, but
    */

   hypre_TFree(rhs, HYPRE_MEMORY_DEVICE);
   rhs = hypre_CTAlloc(HYPRE_Real, m * n, HYPRE_MEMORY_DEVICE);

   for (row = 0; row < m; row++)
   {
      for (j = E_diag_i[row]; j < E_diag_i[row+1]; j++)
      {
         col = E_diag_j[j];
         *(rhs + col*m + row) = E_diag_data[j];
      }
   }

   /* check buffer size and create buffer */

   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrsm2_bufferSizeExt( handle, algo, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                                         n, m, nnz_BLUm, (hypre_double *)&alpha, matU_des, (hypre_double *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (hypre_double *)rhs, m, malU_info2, policy, &buffer_size));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrsm2_bufferSizeExt( handle, algo, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                                         n, m, nnz_BLUm, (float *)&alpha, matU_des, (float *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (float *)rhs, m, malU_info2, policy, &buffer_size));
   }

   if (buffer_size > buffer_size_old)
   {
      buffer = hypre_ReAlloc_v2(buffer, buffer_size_old, buffer_size, HYPRE_MEMORY_DEVICE);
      buffer_size_old = buffer_size;
   }

   /* analysis */

   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrsm2_analysis( handle, algo, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                                         n, m, nnz_BLUm, (hypre_double *)&alpha, matU_des, (hypre_double *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (hypre_double *)rhs, m, malU_info2, policy, buffer));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrsm2_analysis( handle, algo, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                                      n, m, nnz_BLUm, (float *)&alpha, matU_des, (float *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (float *)rhs, m, malU_info2, policy, buffer));
   }

   /* solve phase */
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrsm2_solve( handle, algo, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                                         n, m, nnz_BLUm, (hypre_double *)&alpha, matU_des, (hypre_double *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (hypre_double *)rhs, m, malU_info2, policy, buffer));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrsm2_solve( handle, algo, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                                         n, m, nnz_BLUm, (float *)&alpha, matU_des, (float *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (float *)rhs, m, malU_info2, policy, buffer));
   }

   /* R = - (EiUB ) / LB -> LB'R' = -EiUB'
    */
   alpha = -1.0;
   csrsm2Info_t            malL_info2 = NULL;
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrsm2Info(&malL_info2));

   /* check buffer size and create buffer */

   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrsm2_bufferSizeExt( handle, algo, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                                         n, m, nnz_BLUm, (hypre_double *)&alpha, matL_des, (hypre_double *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (hypre_double *)rhs, m, malL_info2, policy, &buffer_size));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrsm2_bufferSizeExt( handle, algo, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                                         n, m, nnz_BLUm, (float *)&alpha, matL_des, (float *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (float *)rhs, m, malL_info2, policy, &buffer_size));
   }

   if (buffer_size > buffer_size_old)
   {
      buffer = hypre_ReAlloc_v2(buffer, buffer_size_old, buffer_size, HYPRE_MEMORY_DEVICE);
      buffer_size_old = buffer_size;
   }

   /* analysis */

   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrsm2_analysis( handle, algo, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                                         n, m, nnz_BLUm, (hypre_double *)&alpha, matL_des, (hypre_double *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (hypre_double *)rhs, m, malL_info2, policy, buffer));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrsm2_analysis( handle, algo, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                                      n, m, nnz_BLUm, (float *)&alpha, matL_des, (float *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (float *)rhs, m, malL_info2, policy, buffer));
   }

   /* solve phase */
   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrsm2_solve( handle, algo, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                                         n, m, nnz_BLUm, (hypre_double *)&alpha, matL_des, (hypre_double *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (hypre_double *)rhs, m, malL_info2, policy, buffer));
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrsm2_solve( handle, algo, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                                         n, m, nnz_BLUm, (float *)&alpha, matL_des, (float *)BLUm_diag_data, BLUm_diag_i, BLUm_diag_j, (float *)rhs, m, malL_info2, policy, buffer));
   }
   cudaDeviceSynchronize();
   /* now form R, m * (n + m) */
   HYPRE_Int            ctrR = 0;
   HYPRE_Int            *R_diag_i;
   HYPRE_Int            *R_offd_i;
   HYPRE_Int            *R_diag_j;
   HYPRE_Real           *R_diag_data;

   HYPRE_Int       capacity_R = nnz_BLUm + m;
   R_diag_i       = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);
   R_offd_i       = hypre_CTAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);
   R_diag_j       = hypre_TAlloc(HYPRE_Int, capacity_R, HYPRE_MEMORY_DEVICE);
   R_diag_data    = hypre_TAlloc(HYPRE_Real, capacity_R, HYPRE_MEMORY_DEVICE);

   for (row = 0; row < m; row++)
   {
      R_diag_i[row] = ctrR;
      for (col = 0; col < n; col++)
      {
         val = *(rhs + col*m + row);
         if (hypre_abs(val) > drop_tol)
         {
            if (ctrR >= capacity_R)
            {
               HYPRE_Int tmp;
               tmp = capacity_R;
               capacity_R = capacity_R * EXPAND_FACT;
               R_diag_j       = hypre_TReAlloc_v2(R_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_R, HYPRE_MEMORY_DEVICE);
               R_diag_data    = hypre_TReAlloc_v2(R_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_R, HYPRE_MEMORY_DEVICE);
            }
            R_diag_j[ctrR] = col;
            R_diag_data[ctrR++] = val;
         }
      }
      if (ctrR >= capacity_R)
      {
         HYPRE_Int tmp;
         tmp = capacity_R;
         capacity_R = capacity_R * EXPAND_FACT;
         R_diag_j       = hypre_TReAlloc_v2(R_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_R, HYPRE_MEMORY_DEVICE);
         R_diag_data    = hypre_TReAlloc_v2(R_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_R, HYPRE_MEMORY_DEVICE);
      }
      R_diag_j[ctrR] = n + row;
      R_diag_data[ctrR++] = 1.0;
   }

   R_diag_i[m] = ctrR;

   hypre_TFree(buffer, HYPRE_MEMORY_DEVICE);

   /* create ParCSR matrices */

   R = hypre_ParCSRMatrixCreate( hypre_ParCSRMatrixComm(A),
                        hypre_ParCSRMatrixGlobalNumRows(E),
                        hypre_ParCSRMatrixGlobalNumCols(A),
                        hypre_ParCSRMatrixRowStarts(E),
                        hypre_ParCSRMatrixColStarts(A),
                        0,
                        ctrR,
                        0);

   P = hypre_ParCSRMatrixCreate( hypre_ParCSRMatrixComm(A),
                        hypre_ParCSRMatrixGlobalNumRows(A),
                        hypre_ParCSRMatrixGlobalNumCols(F),
                        hypre_ParCSRMatrixRowStarts(A),
                        hypre_ParCSRMatrixColStarts(F),
                        0,
                        ctrP,
                        0);

   /* Assign value to diagonal data */

   R_diag = hypre_ParCSRMatrixDiag(R);
   hypre_CSRMatrixI(R_diag) = R_diag_i;
   hypre_CSRMatrixJ(R_diag) = R_diag_j;
   hypre_CSRMatrixData(R_diag) = R_diag_data;
   hypre_CSRMatrixSetDataOwner(R_diag, 1);

   P_diag = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrixI(P_diag) = P_diag_i;
   hypre_CSRMatrixJ(P_diag) = P_diag_j;
   hypre_CSRMatrixData(P_diag) = P_diag_data;
   hypre_CSRMatrixSetDataOwner(P_diag, 1);

   /* Assign value to off diagonal data */

   R_diag = hypre_ParCSRMatrixOffd(R);
   hypre_CSRMatrixI(R_diag) = R_offd_i;
   P_diag = hypre_ParCSRMatrixOffd(P);
   hypre_CSRMatrixI(P_diag) = P_offd_i;

   *Rp = R;
   *Pp = P;

   HYPRE_CUSPARSE_CALL(cusparseDestroyCsrsm2Info(malL_info));
   HYPRE_CUSPARSE_CALL(cusparseDestroyCsrsm2Info(malU_info));
   HYPRE_CUSPARSE_CALL(cusparseDestroyCsrsm2Info(malL_info2));
   HYPRE_CUSPARSE_CALL(cusparseDestroyCsrsm2Info(malU_info2));

   return hypre_error_flag;
}

/* Convert the L, D, U style to the cusparse style
 * Assume the diagonal of L and U are the ilu factorization, directly combine them
 */
HYPRE_Int
hypre_ILUSetupLDUtoCusparse(hypre_ParCSRMatrix *L, HYPRE_Real *D, hypre_ParCSRMatrix *U, hypre_ParCSRMatrix **LDUp)
{
   /* data slots */
   HYPRE_Int            i, j, pos;

   hypre_CSRMatrix      *L_diag        = hypre_ParCSRMatrixDiag(L);
   hypre_CSRMatrix      *U_diag        = hypre_ParCSRMatrixDiag(U);
   HYPRE_Int            *L_diag_i      = hypre_CSRMatrixI(L_diag);
   HYPRE_Int            *L_diag_j      = hypre_CSRMatrixJ(L_diag);
   HYPRE_Real           *L_diag_data   = hypre_CSRMatrixData(L_diag);
   HYPRE_Int            *U_diag_i      = hypre_CSRMatrixI(U_diag);
   HYPRE_Int            *U_diag_j      = hypre_CSRMatrixJ(U_diag);
   HYPRE_Real           *U_diag_data   = hypre_CSRMatrixData(U_diag);
   HYPRE_Int            n              = hypre_ParCSRMatrixNumRows(L);
   HYPRE_Int            nnz_L          = L_diag_i[n];
   HYPRE_Int            nnz_U          = U_diag_i[n];
   HYPRE_Int            nnz_LDU        = n + nnz_L + nnz_U;

   hypre_ParCSRMatrix   *LDU;
   hypre_CSRMatrix      *LDU_diag;
   HYPRE_Int            *LDU_diag_i;
   HYPRE_Int            *LDU_diag_j;
   HYPRE_Real           *LDU_diag_data;

   /* MPI */
   MPI_Comm             comm                 = hypre_ParCSRMatrixComm(L);
   HYPRE_Int            num_procs,  my_id;

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);


   /* cuda data slot */

   /* create matrix */

   LDU = hypre_ParCSRMatrixCreate(  comm,
                                    hypre_ParCSRMatrixGlobalNumRows(L),
                                    hypre_ParCSRMatrixGlobalNumRows(L),
                                    hypre_ParCSRMatrixRowStarts(L),
                                    hypre_ParCSRMatrixColStarts(L),
                                    0,
                                    nnz_LDU,
                                    0);

   LDU_diag = hypre_ParCSRMatrixDiag(LDU);
   LDU_diag_i = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_DEVICE);
   LDU_diag_j = hypre_TAlloc(HYPRE_Int, nnz_LDU, HYPRE_MEMORY_DEVICE);
   LDU_diag_data = hypre_TAlloc(HYPRE_Real, nnz_LDU, HYPRE_MEMORY_DEVICE);

   pos = 0;

   for (i = 1; i <= n; i++)
   {
      LDU_diag_i[i-1] = pos;
      for (j = L_diag_i[i-1]; j < L_diag_i[i]; j++)
      {
         LDU_diag_j[pos] = L_diag_j[j];
         LDU_diag_data[pos++] = L_diag_data[j];
      }
      LDU_diag_j[pos] = i-1;
      LDU_diag_data[pos++] = 1.0/D[i-1];
      for (j = U_diag_i[i-1]; j < U_diag_i[i]; j++)
      {
         LDU_diag_j[pos] = U_diag_j[j];
         LDU_diag_data[pos++] = U_diag_data[j];
      }
   }
   LDU_diag_i[n] = pos;

   hypre_CSRMatrixI(LDU_diag)    = LDU_diag_i;
   hypre_CSRMatrixJ(LDU_diag)    = LDU_diag_j;
   hypre_CSRMatrixData(LDU_diag) = LDU_diag_data;

   /* now sort */
   hypre_CSRMatrixSortRow(LDU_diag);
   hypre_ParCSRMatrixDiag(LDU) = LDU_diag;

   *LDUp = LDU;

   return hypre_error_flag;
}

/* Apply the (modified) ILU factorization to the diagonal block of A only.
 * A: matrix
 * ALUp: pointer to the result, factorization stroed on the diagonal
 * modified: set to 0 to use classical ILU0
 */
HYPRE_Int
hypre_ILUSetupRAPMILU0(hypre_ParCSRMatrix *A, hypre_ParCSRMatrix **ALUp, HYPRE_Int modified)
{
   HYPRE_Int            n              = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   /* Get necessary slots */
   hypre_ParCSRMatrix   *L, *U, *S, *ALU;
   HYPRE_Real           *D;
   HYPRE_Int            *u_end;

   /* u_end is the end position of the upper triangular part (if we need E and F implicitly), not used here */
   hypre_ILUSetupMILU0( A, NULL, NULL, n, n, &L, &D, &U, &S, &u_end, modified);
   hypre_TFree(u_end, HYPRE_MEMORY_HOST);

   hypre_ILUSetupLDUtoCusparse(L, D, U, &ALU);

   if (L)
   {
      hypre_ParCSRMatrixDestroy(L);
   }
   if (D)
   {
      hypre_TFree(D, HYPRE_MEMORY_DEVICE);
   }
   if (U)
   {
      hypre_ParCSRMatrixDestroy(U);
   }

   *ALUp = ALU;

   return hypre_error_flag;
}

/* Modified ILU(0) with RAP like solve
 * A = input matrix
 * Not explicitly forming the matrix, the previous version was abondoned
 */
HYPRE_Int
hypre_ILUSetupRAPILU0Device(hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int n, HYPRE_Int nLU,
                           cusparseMatDescr_t matL_des, cusparseMatDescr_t matU_des, cusparseSolvePolicy_t ilu_solve_policy,
                           void **bufferp, csrsv2Info_t *matAL_infop, csrsv2Info_t *matAU_infop,
                           csrsv2Info_t *matBL_infop, csrsv2Info_t *matBU_infop,
                           csrsv2Info_t *matSL_infop, csrsv2Info_t *matSU_infop,
                           hypre_ParCSRMatrix **Apermptr, hypre_ParCSRMatrix **matSptr, hypre_CSRMatrix **ALUptr, hypre_CSRMatrix **BLUptr, hypre_CSRMatrix **CLUptr,
                           hypre_CSRMatrix **Eptr, hypre_CSRMatrix **Fptr, HYPRE_Int test_opt)
{

   /* params */
   MPI_Comm             comm           = hypre_ParCSRMatrixComm(A);
   HYPRE_Int            *rperm         = NULL;

   csrsv2Info_t         matAL_info     = NULL;
   csrsv2Info_t         matAU_info     = NULL;
   csrsv2Info_t         matBL_info     = NULL;
   csrsv2Info_t         matBU_info     = NULL;
   csrsv2Info_t         matSL_info     = NULL;
   csrsv2Info_t         matSU_info     = NULL;

   HYPRE_Int            buffer_size    = 0;
   void                 *buffer        = NULL;

   //hypre_CSRMatrix      *A_diag        = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int            m              = n - nLU;

   //printf("Size of local Schur: %d\n",m);

   HYPRE_Int            i;

   /* MPI */
   HYPRE_Int            num_procs,  my_id;
   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   /* Matrix Structure */
   hypre_ParCSRMatrix   *Apq, *ALU, *ALUm, *S;
   hypre_CSRMatrix      *Amd, *Ad, *SLU, *Apq_diag;

   rperm                               = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);

   for(i = 0; i < n; i++)
   {
      rperm[perm[i]] = i;
   }

   /* first we need to compute the ILU0 factorization of B */

   /* Copy diagonal matrix into a new place with permutation
    * That is, Apq = A(perm,qperm);
    */
   hypre_ParILURAPReorder(A, perm, rperm, &Apq);

   /* do the full ILU0 and modified ILU0 */
   hypre_ILUSetupRAPMILU0(Apq, &ALU, 0);
   hypre_ILUSetupRAPMILU0(Apq, &ALUm, 1);

   hypre_CSRMatrix *dB, *dS, *dE, *dF;

   /* get modified and extract LU factorization */
   Amd = hypre_ParCSRMatrixDiag(ALUm);
   Ad = hypre_ParCSRMatrixDiag(ALU);
   switch(test_opt)
   {
      case 1:
         {
            /* RAP where we save E and F */
            Apq_diag = hypre_ParCSRMatrixDiag(Apq);
            hypre_CSRMatrixSortRow(Apq_diag);
            hypre_ParILUCusparseILUExtractEBFC(Apq_diag, nLU, &dB, &dS, Eptr, Fptr);
            /* get modified ILU of B */
            hypre_ParILUCusparseILUExtractEBFC(Amd, nLU, BLUptr, &SLU, &dE, &dF);
            hypre_CSRMatrixDestroy(dB);
            hypre_CSRMatrixDestroy(dS);
            hypre_CSRMatrixDestroy(dE);
            hypre_CSRMatrixDestroy(dF);
         }
         break;
      case 2:
         {
            /* C-EB^{-1}F where we save EU^{-1}, L^{-1}F as sparse matrices */
            Apq_diag = hypre_ParCSRMatrixDiag(Apq);
            hypre_CSRMatrixSortRow(Apq_diag);
            hypre_ParILUCusparseILUExtractEBFC(Apq_diag, nLU, &dB, CLUptr, &dE, &dF);
            /* get modified ILU of B */
            hypre_ParILUCusparseILUExtractEBFC(Amd, nLU, BLUptr, &SLU, Eptr, Fptr);
            hypre_CSRMatrixDestroy(dB);
            hypre_CSRMatrixDestroy(dE);
            hypre_CSRMatrixDestroy(dF);
         }
         break;
      case 3:
         {
            /* C-EB^{-1}F where we save E and F */
            Apq_diag = hypre_ParCSRMatrixDiag(Apq);
            hypre_CSRMatrixSortRow(Apq_diag);
            hypre_ParILUCusparseILUExtractEBFC(Apq_diag, nLU, &dB, CLUptr, Eptr, Fptr);
            /* get modified ILU of B */
            hypre_ParILUCusparseILUExtractEBFC(Amd, nLU, BLUptr, &SLU, &dE, &dF);
            hypre_CSRMatrixDestroy(dB);
            hypre_CSRMatrixDestroy(dE);
            hypre_CSRMatrixDestroy(dF);
         }
         break;
      case 4:
         {
            /* RAP where we save EU^{-1}, L^{-1}F as sparse matrices */
            hypre_ParILUCusparseILUExtractEBFC(Ad, nLU, BLUptr, &SLU, Eptr, Fptr);
         }
         break;
      case 0: default:
         {
            /* RAP where we save EU^{-1}, L^{-1}F as sparse matrices */
            hypre_ParILUCusparseILUExtractEBFC(Amd, nLU, BLUptr, &SLU, Eptr, Fptr);
         }
         break;
   }

   *ALUptr = hypre_ParCSRMatrixDiag(ALU);
   /* Analysis of BILU */
   HYPRE_ILUSetupCusparseCSRILU0SetupSolve(*ALUptr, matL_des, matU_des,
                           ilu_solve_policy, &matAL_info, &matAU_info,
                           &buffer_size, &buffer);

   /* Analysis of BILU */
   HYPRE_ILUSetupCusparseCSRILU0SetupSolve(*BLUptr, matL_des, matU_des,
                           ilu_solve_policy, &matBL_info, &matBU_info,
                           &buffer_size, &buffer);

   /* Analysis of SILU */
   HYPRE_ILUSetupCusparseCSRILU0SetupSolve(SLU, matL_des, matU_des,
                           ilu_solve_policy, &matSL_info, &matSU_info,
                           &buffer_size, &buffer);

   /* start forming parCSR matrix S */

   HYPRE_BigInt   S_total_rows, *S_row_starts;
   HYPRE_BigInt   big_m = (HYPRE_BigInt)m;
   hypre_MPI_Allreduce( &big_m, &S_total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

   if (S_total_rows>0)
   {
      {
         HYPRE_BigInt global_start;
         S_row_starts = hypre_CTAlloc(HYPRE_BigInt,2,HYPRE_MEMORY_HOST);
         hypre_MPI_Scan( &big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
         S_row_starts[0] = global_start - m;
         S_row_starts[1] = global_start;
      }

      S_row_starts = hypre_CTAlloc(HYPRE_BigInt, 2, HYPRE_MEMORY_HOST);
      S_row_starts[1] = S_total_rows;
      S_row_starts[0] = S_total_rows - m;
      hypre_MPI_Allreduce(&m, &S_total_rows, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
      S = hypre_ParCSRMatrixCreate( hypre_ParCSRMatrixComm(A),
                           S_total_rows,
                           S_total_rows,
                           S_row_starts,
                           S_row_starts,
                           0,
                           0,
                           0);

      /* memroy leak here */
      hypre_ParCSRMatrixDiag(S) = SLU;

      /* free memory */
      hypre_TFree(S_row_starts, HYPRE_MEMORY_HOST);
   }

   *matSptr       = S;
   *Apermptr      = Apq;
   *bufferp       = buffer;
   *matAL_infop   = matAL_info;
   *matAU_infop   = matAU_info;
   *matBL_infop   = matBL_info;
   *matBU_infop   = matBU_info;
   *matSL_infop   = matSL_info;
   *matSU_infop   = matSU_info;

   return hypre_error_flag;
}

#endif

/* Modified ILU(0) with RAP like solve
 * A = input matrix
 * Not explicitly forming the matrix
 */
HYPRE_Int
hypre_ILUSetupRAPILU0(hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int n, HYPRE_Int nLU,
                           hypre_ParCSRMatrix **Lptr, HYPRE_Real **Dptr, hypre_ParCSRMatrix **Uptr,
                           hypre_ParCSRMatrix **mLptr, HYPRE_Real **mDptr, hypre_ParCSRMatrix **mUptr, HYPRE_Int **u_end)
{
   HYPRE_Int            i;
   hypre_ParCSRMatrix   *S_temp = NULL;
   HYPRE_Int            *u_temp = NULL;

   /* standard ILU0 factorization */
   hypre_ILUSetupMILU0(A, perm, perm, n, n, Lptr, Dptr, Uptr, &S_temp, &u_temp, 0);
   if (S_temp)
   {
      hypre_ParCSRMatrixDestroy(S_temp);
   }
   if (u_temp)
   {
      hypre_Free( u_temp, HYPRE_MEMORY_HOST);
   }
   /* modified ILU0 factorization */
   hypre_ILUSetupMILU0(A, perm, perm, n, n, mLptr, mDptr, mUptr, &S_temp, &u_temp, 1);
   if (S_temp)
   {
      hypre_ParCSRMatrixDestroy(S_temp);
   }
   if (u_temp)
   {
      hypre_Free( u_temp, HYPRE_MEMORY_HOST);
   }

   /* pointer to the start location */
   HYPRE_Int *u_end_array;
   u_end_array = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);

   hypre_CSRMatrix   *U_diag = hypre_ParCSRMatrixDiag(*Uptr);
   HYPRE_Int         *U_diag_i = hypre_CSRMatrixI(U_diag);
   HYPRE_Int         *U_diag_j = hypre_CSRMatrixJ(U_diag);
   HYPRE_Real        *U_diag_data = hypre_CSRMatrixData(U_diag);
   hypre_CSRMatrix   *mU_diag = hypre_ParCSRMatrixDiag(*mUptr);
   HYPRE_Int         *mU_diag_i = hypre_CSRMatrixI(mU_diag);
   HYPRE_Int         *mU_diag_j = hypre_CSRMatrixJ(mU_diag);
   HYPRE_Real        *mU_diag_data = hypre_CSRMatrixData(mU_diag);

   // first sort the Upper part U
   for (i = 0; i < nLU; i++)
   {
      hypre_qsort1(U_diag_j,U_diag_data,U_diag_i[i],U_diag_i[i+1]-1);
      hypre_qsort1(mU_diag_j,mU_diag_data,mU_diag_i[i],mU_diag_i[i+1]-1);
      hypre_BinarySearch2(U_diag_j,nLU,U_diag_i[i],U_diag_i[i+1]-1,u_end_array + i);
   }

   hypre_CSRMatrix   *L_diag = hypre_ParCSRMatrixDiag(*Lptr);
   HYPRE_Int         *L_diag_i = hypre_CSRMatrixI(L_diag);
   HYPRE_Int         *L_diag_j = hypre_CSRMatrixJ(L_diag);
   HYPRE_Real        *L_diag_data = hypre_CSRMatrixData(L_diag);
   hypre_CSRMatrix   *mL_diag = hypre_ParCSRMatrixDiag(*mLptr);
   HYPRE_Int         *mL_diag_i = hypre_CSRMatrixI(mL_diag);
   HYPRE_Int         *mL_diag_j = hypre_CSRMatrixJ(mL_diag);
   HYPRE_Real        *mL_diag_data = hypre_CSRMatrixData(mL_diag);

   // now sort the Lower part L
   for (i = nLU; i < n; i++)
   {
      hypre_qsort1(L_diag_j,L_diag_data,L_diag_i[i],L_diag_i[i+1]-1);
      hypre_qsort1(mL_diag_j,mL_diag_data,mL_diag_i[i],mL_diag_i[i+1]-1);
      hypre_BinarySearch2(L_diag_j, nLU, L_diag_i[i], L_diag_i[i+1]-1, u_end_array + i);
   }

   *u_end = u_end_array;

   return hypre_error_flag;
}

/* ILU(0)
 * A = input matrix
 * perm = permutation array indicating ordering of rows. Perm could come from a
 *    CF_marker array or a reordering routine. When set to NULL, indentity permutation is used.
 * qperm = permutation array indicating ordering of columns. When set to NULL, indentity permutation is used.
 * nI = number of interial unknowns
 * nLU = size of incomplete factorization, nLU should obey nLU <= nI.
 *    Schur complement is formed if nLU < n
 * Lptr, Dptr, Uptr, Sptr = L, D, U, S factors.
 * will form global Schur Matrix if nLU < n
 */
HYPRE_Int
hypre_ILUSetupILU0(hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int *qperm, HYPRE_Int nLU, HYPRE_Int nI,
      hypre_ParCSRMatrix **Lptr, HYPRE_Real** Dptr, hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr, HYPRE_Int **u_end)
{
   return hypre_ILUSetupMILU0( A, perm, qperm, nLU, nI, Lptr, Dptr, Uptr, Sptr, u_end, 0);
}

/* (modified) ILU(0)
 * A = input matrix
 * perm = permutation array indicating ordering of rows. Perm could come from a
 *    CF_marker array or a reordering routine. When set to NULL, indentity permutation is used.
 * qperm = permutation array indicating ordering of columns When set to NULL, identity permutation is used.
 * nI = number of interior unknowns
 * nLU = size of incomplete factorization, nLU should obey nLU <= nI.
 *    Schur complement is formed if nLU < n
 * Lptr, Dptr, Uptr, Sptr = L, D, U, S factors.
 * modified set to 0 to use classical ILU
 * will form global Schur Matrix if nLU < n
 */
HYPRE_Int
hypre_ILUSetupMILU0(hypre_ParCSRMatrix *A, HYPRE_Int *permp, HYPRE_Int *qpermp, HYPRE_Int nLU, HYPRE_Int nI,
      hypre_ParCSRMatrix **Lptr, HYPRE_Real** Dptr, hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr, HYPRE_Int **u_end,
      HYPRE_Int modified)
{
   HYPRE_Int                i, ii, j, k, k1, k2, k3, ctrU, ctrL, ctrS, lenl, lenu, jpiv, col, jpos;
   HYPRE_Int                *iw, *iL, *iU;
   HYPRE_Real               dd, t, dpiv, lxu, *wU, *wL;
   HYPRE_Real               drop;

   /* communication stuffs for S */
   MPI_Comm                 comm             = hypre_ParCSRMatrixComm(A);
   HYPRE_Int                S_offd_nnz, S_offd_ncols;
   hypre_ParCSRCommPkg      *comm_pkg;
   hypre_ParCSRCommHandle   *comm_handle;
   HYPRE_Int                num_sends, begin, end;
   HYPRE_BigInt                *send_buf        = NULL;
   HYPRE_Int                num_procs, my_id;

   /* data objects for A */
   hypre_CSRMatrix          *A_diag          = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix          *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real               *A_diag_data     = hypre_CSRMatrixData(A_diag);
   HYPRE_Int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Int                *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real               *A_offd_data     = hypre_CSRMatrixData(A_offd);
   HYPRE_Int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Int                *A_offd_j        = hypre_CSRMatrixJ(A_offd);

   /* size of problem and schur system */
   HYPRE_Int                n                = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int                m                = n - nLU;
   HYPRE_Int                e                = nI - nLU;
   HYPRE_Int                m_e              = n - nI;
   HYPRE_Real               local_nnz, total_nnz;
   HYPRE_Int                *u_end_array;

   /* data objects for L, D, U */
   hypre_ParCSRMatrix       *matL;
   hypre_ParCSRMatrix       *matU;
   hypre_CSRMatrix          *L_diag;
   hypre_CSRMatrix          *U_diag;
   HYPRE_Real               *D_data;
   HYPRE_Real               *L_diag_data;
   HYPRE_Int                *L_diag_i;
   HYPRE_Int                *L_diag_j;
   HYPRE_Real               *U_diag_data;
   HYPRE_Int                *U_diag_i;
   HYPRE_Int                *U_diag_j;

   /* data objects for S */
   hypre_ParCSRMatrix       *matS = NULL;
   hypre_CSRMatrix          *S_diag;
   hypre_CSRMatrix          *S_offd;
   HYPRE_Real               *S_diag_data     = NULL;
   HYPRE_Int                *S_diag_i        = NULL;
   HYPRE_Int                *S_diag_j        = NULL;
   HYPRE_Int                *S_offd_i        = NULL;
   HYPRE_Int                *S_offd_j        = NULL;
   HYPRE_BigInt             *S_offd_colmap   = NULL;
   HYPRE_Real               *S_offd_data;
   HYPRE_BigInt             col_starts[2];
   HYPRE_BigInt             total_rows;

   /* memory management */
   HYPRE_Int                initial_alloc    = 0;
   HYPRE_Int                capacity_L;
   HYPRE_Int                capacity_U;
   HYPRE_Int                capacity_S       = 0;
   HYPRE_Int                nnz_A            = A_diag_i[n];

   /* reverse permutation array */
   HYPRE_Int                *rperm;
   HYPRE_Int                *perm, *qperm;

   /* start setup
    * get communication stuffs first
    */
   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   /* setup if not yet built */
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* check for correctness */
   if (nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU out of range.\n");
   }
   if (e < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU should not exceed nI.\n");
   }

   /* Allocate memory for u_end array */
   u_end_array    = hypre_TAlloc(HYPRE_Int, nLU, HYPRE_MEMORY_HOST);

   /* Allocate memory for L,D,U,S factors */
   if (n > 0)
   {
      initial_alloc  = nLU + ceil((nnz_A / 2.0)*nLU/n);
      capacity_S     = m + ceil((nnz_A / 2.0)*m/n);
   }
   capacity_L     = initial_alloc;
   capacity_U     = initial_alloc;

   D_data         = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_DEVICE);
   L_diag_i       = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_DEVICE);
   L_diag_j       = hypre_TAlloc(HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
   L_diag_data    = hypre_TAlloc(HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
   U_diag_i       = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_DEVICE);
   U_diag_j       = hypre_TAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
   U_diag_data    = hypre_TAlloc(HYPRE_Real, capacity_U, HYPRE_MEMORY_DEVICE);
   S_diag_i       = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);
   S_diag_j       = hypre_TAlloc(HYPRE_Int, capacity_S, HYPRE_MEMORY_DEVICE);
   S_diag_data    = hypre_TAlloc(HYPRE_Real, capacity_S, HYPRE_MEMORY_DEVICE);

   /* allocate working arrays */
   iw             = hypre_TAlloc(HYPRE_Int, 3*n, HYPRE_MEMORY_HOST);
   iL             = iw+n;
   rperm          = iw + 2*n;
   wL             = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);

   ctrU        = ctrL        = ctrS        = 0;
   L_diag_i[0] = U_diag_i[0] = S_diag_i[0] = 0;
   /* set marker array iw to -1 */
   for (i = 0; i < n; i++)
   {
      iw[i] = -1;
   }

   /* get reverse permutation (rperm).
    * create permutation if they are null
    * rperm holds the reordered indexes.
    * rperm only used for column
    */

   if (!permp)
   {
      perm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
      for (i = 0; i < n; i++)
      {
         perm[i] = i;
      }
   }
   else
   {
      perm = permp;
   }

   if (!qpermp)
   {
      qperm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
      for (i = 0; i < n; i++)
      {
         qperm[i] = i;
      }
   }
   else
   {
      qperm = qpermp;
   }

   for (i = 0; i < n; i++)
   {
      rperm[qperm[i]] = i;
   }

   /*---------  Begin Factorization. Work in permuted space  ----*/
   for (ii = 0; ii < nLU; ii++)
   {
      // get row i
      i = perm[ii];
      // get extents of row i
      k1=A_diag_i[i];
      k2=A_diag_i[i+1];
      // track the drop
      drop = 0.0;

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL+ii;
      wU = wL+ii;
      /*--------------------  diagonal entry */
      dd = 0.0;
      lenl  = lenu = 0;
      iw[ii] = ii;
      /*-------------------- scan & unwrap column */
      for (j = k1; j < k2; j++)
      {
         col = rperm[A_diag_j[j]];
         t = A_diag_data[j];
         if ( col < ii )
         {
            iw[col] = lenl;
            iL[lenl] = col;
            wL[lenl++] = t;
         }
         else if (col > ii)
         {
            iw[col] = lenu;
            iU[lenu] = col;
            wU[lenu++] = t;
         }
         else
         {
            dd=t;
         }
      }

      /* eliminate row */
      /*-------------------------------------------------------------------------
       *  In order to do the elimination in the correct order we must select the
       *  smallest column index among iL[k], k = j, j+1, ..., lenl-1. For ILU(0),
       *  no new fill-ins are expect, so we can pre-sort iL and wL prior to the
       *  entering the elimination loop.
       *-----------------------------------------------------------------------*/
      //      hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      hypre_qsort3ir(iL, wL, iw, 0, (lenl-1));
      for (j = 0; j < lenl; j++)
      {
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;

         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv+1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if (jpos < 0)
            {
               drop = drop - U_diag_data[k] * dpiv;
               continue;
            }

            lxu = - U_diag_data[k] * dpiv;
            if (col < ii)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if (col > ii)
            {
               /* dealing with U part */
               wU[jpos] += lxu;
            }
            else
            {
               /* diagonal update */
               dd += lxu;
            }
         }
      }
      /* modify when necessary */
      if (modified)
      {
         dd = dd + drop;
      }

      /* restore iw (only need to restore diagonal and U part */
      iw[ii] = -1;
      for (j = 0; j < lenu; j++)
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      if (lenl > 0)
      {
         while ((ctrL+lenl) > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = capacity_L * EXPAND_FACT + 1;
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
         }
         //hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         //hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(&L_diag_j[ctrL], iL, HYPRE_Int, lenl,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(&L_diag_data[ctrL], wL, HYPRE_Real, lenl,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      }
      L_diag_i[ii+1] = (ctrL+=lenl);

      /* diagonal part (we store the inverse) */
      if (fabs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;
      }
      D_data[ii] = 1./dd;

      /* U part */
      /* Check that memory is sufficient */
      if (lenu > 0)
      {
         while ((ctrU+lenu) > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = capacity_U * EXPAND_FACT + 1;
            U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
            U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_U, HYPRE_MEMORY_DEVICE);
         }
         //hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         //hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(&U_diag_j[ctrU], iU, HYPRE_Int, lenu,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(&U_diag_data[ctrU], wU, HYPRE_Real, lenu,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      }
      U_diag_i[ii+1] = (ctrU+=lenu);

      /* check and build u_end array */
      if (m > 0)
      {
         hypre_qsort1(U_diag_j,U_diag_data,U_diag_i[ii],U_diag_i[ii+1]-1);
         hypre_BinarySearch2(U_diag_j,nLU,U_diag_i[ii],U_diag_i[ii+1]-1,u_end_array + ii);
      }
      else
      {
         /* Everything is in U */
         u_end_array[ii] = ctrU;
      }

   }

   /*---------  Begin Factorization in Schur Complement part  ----*/
   for (ii = nLU; ii < n; ii++)
   {
      // get row i
      i = perm[ii];
      // get extents of row i
      k1=A_diag_i[i];
      k2=A_diag_i[i+1];
      drop = 0.0;

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL+nLU + 1;
      wU = wL+nLU + 1;
      /*--------------------  diagonal entry */
      dd = 0.0;
      lenl  = lenu = 0;
      iw[ii] = nLU;
      /*-------------------- scan & unwrap column */
      for (j = k1; j < k2; j++)
      {
         col = rperm[A_diag_j[j]];
         t = A_diag_data[j];
         if ( col < nLU )
         {
            iw[col] = lenl;
            iL[lenl] = col;
            wL[lenl++] = t;
         }
         else if (col != ii)
         {
            iw[col] = lenu;
            iU[lenu] = col;
            wU[lenu++] = t;
         }
         else
         {
            dd=t;
         }
      }

      /* eliminate row */
      /*-------------------------------------------------------------------------
       *  In order to do the elimination in the correct order we must select the
       *  smallest column index among iL[k], k = j, j+1, ..., lenl-1. For ILU(0),
       *  no new fill-ins are expect, so we can pre-sort iL and wL prior to the
       *  entering the elimination loop.
       *-----------------------------------------------------------------------*/
      //      hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      hypre_qsort3ir(iL, wL, iw, 0, (lenl-1));
      for (j = 0; j < lenl; j++)
      {
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;

         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv+1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if (jpos < 0)
            {
               drop = drop - U_diag_data[k] * dpiv;
               continue;
            }

            lxu = - U_diag_data[k] * dpiv;
            if (col < nLU)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if (col != ii)
            {
               /* dealing with U part */
               wU[jpos] += lxu;
            }
            else
            {
               /* diagonal update */
               dd += lxu;
            }
         }
      }
      if (modified)
      {
         dd = dd + drop;
      }
      /* restore iw (only need to restore diagonal and U part */
      iw[ii] = -1;
      for (j = 0; j < lenu; j++)
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      if (lenl > 0)
      {
         while ((ctrL+lenl) > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = capacity_L * EXPAND_FACT + 1;
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
         }
         //hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         //hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(&L_diag_j[ctrL], iL, HYPRE_Int, lenl,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(&L_diag_data[ctrL], wL, HYPRE_Real, lenl,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      }
      L_diag_i[ii+1] = (ctrL+=lenl);

      /* S part */
      /* Check that memory is sufficient */
      while ((ctrS+lenu+1) > capacity_S)
      {
         HYPRE_Int tmp = capacity_S;
         capacity_S = capacity_S * EXPAND_FACT + 1;
         S_diag_j = hypre_TReAlloc_v2(S_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_S, HYPRE_MEMORY_DEVICE);
         S_diag_data = hypre_TReAlloc_v2(S_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_S, HYPRE_MEMORY_DEVICE);
      }
      /* remember S in under a new index system! */
      S_diag_j[ctrS] = ii - nLU;
      S_diag_data[ctrS] = dd;
      for (j = 0; j < lenu; j++)
      {
         S_diag_j[ctrS+1+j] = iU[j] - nLU;
      }
      //hypre_TMemcpy(S_diag_data+ctrS+1, wU, HYPRE_Real, lenu, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(S_diag_data+ctrS+1, wU, HYPRE_Real, lenu,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      S_diag_i[ii-nLU+1] = ctrS+=(lenu+1);
   }
   /* Assemble LDUS matrices */
   /* zero out unfactored rows for U and D */
   for (k = nLU; k < n; k++)
   {
      U_diag_i[k+1] = ctrU;
      D_data[k] = 1.;
   }

   /* First create Schur complement if necessary
    * Check if we need to create Schur complement
    */
   HYPRE_BigInt big_m = (HYPRE_BigInt)m;
   hypre_MPI_Allreduce(&big_m, &total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
   /* only form when total_rows > 0 */
   if ( total_rows > 0 )
   {
      /* now create S */
      /* need to get new column start */
      {
         HYPRE_BigInt global_start;
         hypre_MPI_Scan( &big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
         col_starts[0] = global_start - m;
         col_starts[1] = global_start;
      }

      /* We did nothing to A_offd, so all the data kept, just reorder them
       * The create function takes comm, global num rows/cols,
       *    row/col start, num cols offd, nnz diag, nnz offd
       */
      S_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);
      S_offd_ncols = hypre_CSRMatrixNumCols(A_offd);

      matS = hypre_ParCSRMatrixCreate( comm,
            total_rows,
            total_rows,
            col_starts,
            col_starts,
            S_offd_ncols,
            ctrS,
            S_offd_nnz);

      /* first put diagonal data in */
      S_diag = hypre_ParCSRMatrixDiag(matS);

      hypre_CSRMatrixI(S_diag) = S_diag_i;
      hypre_CSRMatrixData(S_diag) = S_diag_data;
      hypre_CSRMatrixJ(S_diag) = S_diag_j;

      /* now start to construct offdiag of S */
      S_offd = hypre_ParCSRMatrixOffd(matS);
      S_offd_i = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_colmap = hypre_CTAlloc(HYPRE_BigInt, S_offd_ncols, HYPRE_MEMORY_HOST);

      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i[i] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col+1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i+1+e] = k3;
      }

      /* give I, J, DATA to S_offd */
      hypre_CSRMatrixI(S_offd) = S_offd_i;
      hypre_CSRMatrixJ(S_offd) = S_offd_j;
      hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */

      /* get total num of send */
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg,0);
      end = hypre_ParCSRCommPkgSendMapStart(comm_pkg,num_sends);
      send_buf = hypre_TAlloc(HYPRE_BigInt, end - begin, HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for (i = begin; i < end; i++)
      {
         send_buf[i-begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)] - nLU + col_starts[0];
      }
      /* main communication */
      comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf, S_offd_colmap);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;

      hypre_ILUSortOffdColmap(matS);

      /* free */
      hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
   } /* end of forming S */

   /* create S finished */

   matL = hypre_ParCSRMatrixCreate( comm,
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixRowStarts(A),
         hypre_ParCSRMatrixColStarts(A),
         0,
         ctrL,
         0 );

   L_diag = hypre_ParCSRMatrixDiag(matL);
   hypre_CSRMatrixI(L_diag) = L_diag_i;
   if (ctrL)
   {
      hypre_CSRMatrixData(L_diag) = L_diag_data;
      hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we've allocated some memory, so free if not used */
      hypre_TFree(L_diag_j,HYPRE_MEMORY_DEVICE);
      hypre_TFree(L_diag_data,HYPRE_MEMORY_DEVICE);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) ctrL;
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

   matU = hypre_ParCSRMatrixCreate( comm,
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixRowStarts(A),
         hypre_ParCSRMatrixColStarts(A),
         0,
         ctrU,
         0 );

   U_diag = hypre_ParCSRMatrixDiag(matU);
   hypre_CSRMatrixI(U_diag) = U_diag_i;
   if (ctrU)
   {
      hypre_CSRMatrixData(U_diag) = U_diag_data;
      hypre_CSRMatrixJ(U_diag) = U_diag_j;
   }
   else
   {
      /* we've allocated some memory, so free if not used */
      hypre_TFree(U_diag_j,HYPRE_MEMORY_DEVICE);
      hypre_TFree(U_diag_data,HYPRE_MEMORY_DEVICE);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) ctrU;
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;
   /* free memory */
   hypre_TFree(wL,HYPRE_MEMORY_HOST);
   hypre_TFree(iw,HYPRE_MEMORY_HOST);
   if (!matS)
   {
      /* we allocate some memory for S, need to free if unused */
      hypre_TFree(S_diag_i,HYPRE_MEMORY_DEVICE);
   }

   if (!permp)
   {
      hypre_TFree(perm, HYPRE_MEMORY_DEVICE);
   }
   if (!qpermp)
   {
      hypre_TFree(qperm, HYPRE_MEMORY_DEVICE);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;
   *Sptr = matS;
   *u_end = u_end_array;

   return hypre_error_flag;
}

/* ILU(k) symbolic factorization
 * n = total rows of input
 * lfil = level of fill-in, the k in ILU(k)
 * perm = permutation array indicating ordering of factorization. Perm could come from a
 * rperm = reverse permutation array, used here to avoid duplicate memory allocation
 * iw = working array, used here to avoid duplicate memory allocation
 * nLU = size of computed LDU factorization.
 * A/L/U/S_diag_i = the I slot of A, L, U and S
 * A/L/U/S_diag_j = the J slot of A, L, U and S
 * will form global Schur Matrix if nLU < n
 */
HYPRE_Int
hypre_ILUSetupILUKSymbolic(HYPRE_Int n, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j, HYPRE_Int lfil, HYPRE_Int *perm,
      HYPRE_Int *rperm,   HYPRE_Int *iw,   HYPRE_Int nLU, HYPRE_Int *L_diag_i, HYPRE_Int *U_diag_i,
      HYPRE_Int *S_diag_i, HYPRE_Int **L_diag_j, HYPRE_Int **U_diag_j, HYPRE_Int **S_diag_j, HYPRE_Int **u_end)
{
   /*
    * 1: Setup and create buffers
    * A_diag_*: tempory pointer for the diagonal matrix of A and its '*' slot
    * ii: outer loop from 0 to nLU - 1
    * i: the real col number in diag inside the outer loop
    * iw:  working array store the reverse of active col number
    * iL: working array store the active col number
    * iLev: working array store the active level of current row
    * lenl/u: current position in iw and so
    * ctrL/U/S: global position in J
    */

   HYPRE_Int         *temp_L_diag_j, *temp_U_diag_j, *temp_S_diag_j = NULL, *u_levels;
   HYPRE_Int         *iL, *iLev;
   HYPRE_Int         ii, i, j, k, ku, lena, lenl, lenu, lenh, ilev, lev, col, icol;
   HYPRE_Int         m = n - nLU;
   HYPRE_Int         *u_end_array;

   /* memory management */
   HYPRE_Int         ctrL;
   HYPRE_Int         ctrU;
   HYPRE_Int         ctrS;
   HYPRE_Int         capacity_L;
   HYPRE_Int         capacity_U;
   HYPRE_Int         capacity_S;
   HYPRE_Int         initial_alloc = 0;
   HYPRE_Int         nnz_A;

   /* set iL and iLev to right place in iw array */
   iL                = iw + n;
   iLev              = iw + 2*n;

   /* setup initial memory used */
   nnz_A             = A_diag_i[n];
   if (n > 0)
   {
      initial_alloc     = nLU + ceil((nnz_A / 2.0) * nLU / n);
   }
   capacity_L        = initial_alloc;
   capacity_U        = initial_alloc;

   /* allocate other memory for L and U struct */
   temp_L_diag_j     = hypre_CTAlloc(HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
   temp_U_diag_j     = hypre_CTAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);

   if (m > 0)
   {
      capacity_S     = m + ceil(nnz_A / 2.0 * m / n);
      temp_S_diag_j  = hypre_CTAlloc(HYPRE_Int, capacity_S, HYPRE_MEMORY_DEVICE);
   }

   u_end_array       = hypre_TAlloc(HYPRE_Int, nLU, HYPRE_MEMORY_HOST);
   u_levels          = hypre_CTAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
   ctrL = ctrU = ctrS = 0;

   /* set initial value for working array */
   for (ii = 0 ; ii < n; ii++)
   {
      iw[ii] = -1;
   }

   /*
    * 2: Start of main loop
    * those in iL are NEW col index (after permutation)
    */
   for (ii = 0; ii < nLU; ii++)
   {
      i = perm[ii];
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = ii;
      lena = A_diag_i[i+1];
      /* put those already inside original pattern, and set their level to 0 */
      for (j = A_diag_i[i]; j < lena; j++)
      {
         /* get the neworder of that col */
         col = rperm[A_diag_j[j]];
         if (col < ii)
         {
            /*
             * this is an entry in L
             * we maintain a heap structure for L part
             */
            iL[lenh] = col;
            iLev[lenh] = 0;
            iw[col] = lenh++;
            /*now miantian a heap structure*/
            hypre_ILUMinHeapAddIIIi(iL,iLev,iw,lenh);
         }
         else if (col > ii)
         {
            /* this is an entry in U */
            iL[lenu] = col;
            iLev[lenu] = 0;
            iw[col] = lenu++;
         }
      }/* end of j loop for adding pattern in original matrix */

      /*
       * search lower part of current row and update pattern based on level
       */
      while (lenh > 0)
      {
         /*
          * k is now the new col index after permutation
          * the first element of the heap is the smallest
          */
         k = iL[0];
         ilev = iLev[0];
         /*
          * we now need to maintain the heap structure
          */
         hypre_ILUMinHeapRemoveIIIi(iL,iLev,iw,lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k]=-1;
         hypre_swap2i(iL,iLev,ii-lenl,lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k+1];
         for (j = U_diag_i[k]; j < ku; j++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if (lev > lfil)
            {
               continue;
            }
            if (icol < 0)
            {
               /* not yet in */
               if (col < ii)
               {
                  /*
                   * if we add to the left L, we need to maintian the
                   *    heap structure
                   */
                  iL[lenh] = col;
                  iLev[lenh] = lev;
                  iw[col] = lenh++;
                  /*swap it with the element right after the heap*/

                  /* maintain the heap */
                  hypre_ILUMinHeapAddIIIi(iL,iLev,iw,lenh);
               }
               else if (col > ii)
               {
                  iL[lenu] = col;
                  iLev[lenu] = lev;
                  iw[col] = lenu++;
               }
            }
            else
            {
               iLev[icol] = hypre_min(lev, iLev[icol]);
            }
         }/* end of loop j for level update */
      }/* end of while loop for iith row */

      /* now update everything, indices, levels and so */
      L_diag_i[ii+1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = capacity_L * EXPAND_FACT + 1;
            temp_L_diag_j = hypre_TReAlloc_v2(temp_L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL+j] = iL[ii-j-1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii+1] = U_diag_i[ii] + k;
      if (k > 0)
      {
         /* check if memory is enough */
         while (ctrU + k > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = capacity_U * EXPAND_FACT + 1;
            temp_U_diag_j = hypre_TReAlloc_v2(temp_U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
            u_levels = hypre_TReAlloc_v2(u_levels, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
         }
         //hypre_TMemcpy(temp_U_diag_j+ctrU,iL+ii,HYPRE_Int,k,HYPRE_MEMORY_DEVICE,HYPRE_MEMORY_HOST);
         hypre_TMemcpy(temp_U_diag_j+ctrU, iL+ii, HYPRE_Int, k,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(u_levels+ctrU, iLev+ii, HYPRE_Int, k,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         ctrU += k;
      }
      if (m > 0)
      {
         hypre_qsort2i(temp_U_diag_j,u_levels,U_diag_i[ii],U_diag_i[ii+1]-1);
         hypre_BinarySearch2(temp_U_diag_j,nLU,U_diag_i[ii],U_diag_i[ii+1]-1,u_end_array + ii);
      }
      else
      {
         /* Everything is in U */
         u_end_array[ii] = ctrU;
      }

      /* reset iw */
      for(j = ii; j < lenu; j++)
      {
         iw[iL[j]] = -1;
      }

   }/* end of main loop ii from 0 to nLU-1 */

   /* another loop to set EU^-1 and Schur complement */
   for (ii = nLU; ii < n; ii++)
   {
      i = perm[ii];
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = nLU;/* now this stores S, start from nLU */
      lena = A_diag_i[i+1];
      /* put those already inside original pattern, and set their level to 0 */
      for (j = A_diag_i[i]; j < lena; j++)
      {
         /* get the neworder of that col */
         col = rperm[A_diag_j[j]];
         if (col < nLU)
         {
            /*
             * this is an entry in L
             * we maintain a heap structure for L part
             */
            iL[lenh] = col;
            iLev[lenh] = 0;
            iw[col] = lenh++;
            /*now miantian a heap structure*/
            hypre_ILUMinHeapAddIIIi(iL,iLev,iw,lenh);
         }
         else if (col != ii) /* we for sure to add ii, avoid duplicate */
         {
            /* this is an entry in S */
            iL[lenu] = col;
            iLev[lenu] = 0;
            iw[col] = lenu++;
         }
      }/* end of j loop for adding pattern in original matrix */

      /*
       * search lower part of current row and update pattern based on level
       */
      while (lenh > 0)
      {
         /*
          * k is now the new col index after permutation
          * the first element of the heap is the smallest
          */
         k = iL[0];
         ilev = iLev[0];
         /*
          * we now need to maintain the heap structure
          */
         hypre_ILUMinHeapRemoveIIIi(iL,iLev,iw,lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k]=-1;
         hypre_swap2i(iL,iLev,nLU-lenl,lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k+1];
         for (j = U_diag_i[k]; j < ku; j++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if (lev > lfil)
            {
               continue;
            }
            if (icol < 0)
            {
               /* not yet in */
               if (col < nLU)
               {
                  /*
                   * if we add to the left L, we need to maintian the
                   *    heap structure
                   */
                  iL[lenh] = col;
                  iLev[lenh] = lev;
                  iw[col] = lenh++;
                  /*swap it with the element right after the heap*/

                  /* maintain the heap */
                  hypre_ILUMinHeapAddIIIi(iL,iLev,iw,lenh);
               }
               else if (col != ii)
               {
                  /* S part */
                  iL[lenu] = col;
                  iLev[lenu] = lev;
                  iw[col] = lenu++;
               }
            }
            else
            {
               iLev[icol] = hypre_min(lev, iLev[icol]);
            }
         }/* end of loop j for level update */
      }/* end of while loop for iith row */

      /* now update everything, indices, levels and so */
      L_diag_i[ii+1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = capacity_L * EXPAND_FACT + 1;
            temp_L_diag_j = hypre_TReAlloc_v2(temp_L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j ++)
         {
            temp_L_diag_j[ctrL+j] = iL[nLU-j-1];
         }
         ctrL += lenl;
      }
      k = lenu - nLU + 1;
      /* check if memory is enough */
      while (ctrS + k > capacity_S)
      {
         HYPRE_Int tmp = capacity_S;
         capacity_S = capacity_S * EXPAND_FACT + 1;
         temp_S_diag_j = hypre_TReAlloc_v2(temp_S_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_S, HYPRE_MEMORY_DEVICE);
      }
      temp_S_diag_j[ctrS] = ii;/* must have diagonal */
      //hypre_TMemcpy(temp_S_diag_j+ctrS+1,iL+nLU,HYPRE_Int,k-1,HYPRE_MEMORY_DEVICE,HYPRE_MEMORY_HOST);
      hypre_TMemcpy(temp_S_diag_j+ctrS+1, iL+nLU, HYPRE_Int, k-1,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      ctrS += k;
      S_diag_i[ii-nLU+1] = ctrS;

      /* reset iw */
      for (j = nLU; j < lenu; j++)
      {
         iw[iL[j]] = -1;
      }

   }/* end of main loop ii from nLU to n-1 */

   /*
    * 3: Update the struct for L, U and S
    */
   for (k = nLU; k < n; k++)
   {
      U_diag_i[k+1] = U_diag_i[nLU];
   }
   /*
    * 4: Finishing up and free memory
    */
   hypre_TFree(u_levels,HYPRE_MEMORY_HOST);

   *L_diag_j = temp_L_diag_j;
   *U_diag_j = temp_U_diag_j;
   *S_diag_j = temp_S_diag_j;
   *u_end = u_end_array;

   return hypre_error_flag;
}

/* ILU(k)
 * A: input matrix
 * lfil: level of fill-in, the k in ILU(k)
 * permp: permutation array indicating ordering of factorization. Perm could come from a
 *    CF_marker: array or a reordering routine.
 * qpermp: column permutation array.
 * nLU: size of computed LDU factorization.
 * nI: number of interial unknowns, nI should obey nI >= nLU
 * Lptr, Dptr, Uptr: L, D, U factors.
 * Sprt: Schur Complement, if no Schur Complement is needed it will be set to NULL
 */
HYPRE_Int
hypre_ILUSetupILUK(hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Int *permp, HYPRE_Int *qpermp, HYPRE_Int nLU, HYPRE_Int nI,
      hypre_ParCSRMatrix **Lptr, HYPRE_Real** Dptr, hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr, HYPRE_Int **u_end)
{
   /*
    * 1: Setup and create buffers
    * matL/U: the ParCSR matrix for L and U
    * L/U_diag: the diagonal csr matrix of matL/U
    * A_diag_*: tempory pointer for the diagonal matrix of A and its '*' slot
    * ii = outer loop from 0 to nLU - 1
    * i = the real col number in diag inside the outer loop
    * iw =  working array store the reverse of active col number
    * iL = working array store the active col number
    */

   /* call ILU0 if lfil is 0 */
   if (lfil == 0)
   {
      return hypre_ILUSetupILU0( A, permp, qpermp, nLU, nI, Lptr, Dptr, Uptr, Sptr, u_end);
   }
   HYPRE_Real              local_nnz, total_nnz;
   HYPRE_Int               i, ii, j, k, k1, k2, k3, kl, ku, jpiv, col, icol;
   HYPRE_Int               *iw;
   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int            num_procs,  my_id;

   /* data objects for A */
   hypre_CSRMatrix         *A_diag        = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix         *A_offd        = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real              *A_diag_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int               *A_diag_i      = hypre_CSRMatrixI(A_diag);
   HYPRE_Int               *A_diag_j      = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real              *A_offd_data   = hypre_CSRMatrixData(A_offd);
   HYPRE_Int               *A_offd_i      = hypre_CSRMatrixI(A_offd);
   HYPRE_Int               *A_offd_j      = hypre_CSRMatrixJ(A_offd);

   /* data objects for L, D, U */
   hypre_ParCSRMatrix      *matL;
   hypre_ParCSRMatrix      *matU;
   hypre_CSRMatrix         *L_diag;
   hypre_CSRMatrix         *U_diag;
   HYPRE_Real              *D_data;
   HYPRE_Real              *L_diag_data   = NULL;
   HYPRE_Int               *L_diag_i;
   HYPRE_Int               *L_diag_j      = NULL;
   HYPRE_Real              *U_diag_data   = NULL;
   HYPRE_Int               *U_diag_i;
   HYPRE_Int               *U_diag_j      = NULL;

   /* data objects for S */
   hypre_ParCSRMatrix      *matS          = NULL;
   hypre_CSRMatrix         *S_diag;
   hypre_CSRMatrix         *S_offd;
   HYPRE_Real              *S_diag_data   = NULL;
   HYPRE_Int               *S_diag_i      = NULL;
   HYPRE_Int               *S_diag_j      = NULL;
   HYPRE_Int               *S_offd_i      = NULL;
   HYPRE_Int               *S_offd_j      = NULL;
   HYPRE_BigInt            *S_offd_colmap = NULL;
   HYPRE_Real              *S_offd_data;
   HYPRE_Int               S_offd_nnz, S_offd_ncols;
   HYPRE_BigInt            col_starts[2];
   HYPRE_BigInt            total_rows;

   /* communication */
   hypre_ParCSRCommPkg     *comm_pkg;
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_BigInt            *send_buf      = NULL;

   /* problem size */
   HYPRE_Int               n;
   HYPRE_Int               m;
   HYPRE_Int               e;
   HYPRE_Int               m_e;

   /* reverse permutation array */
   HYPRE_Int               *rperm;
   HYPRE_Int               *perm, *qperm;

   /* start setup */
   /* check input and get problem size */
   n =  hypre_CSRMatrixNumRows(A_diag);
   if (nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU out of range.\n");
   }
   m = n - nLU;
   e = nI - nLU;
   m_e = n - nI;
   if (e < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU should not exceed nI.\n");
   }

   /* Init I array anyway. S's might be freed later */
   D_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_DEVICE);
   L_diag_i = hypre_CTAlloc(HYPRE_Int, (n+1), HYPRE_MEMORY_DEVICE);
   U_diag_i = hypre_CTAlloc(HYPRE_Int, (n+1), HYPRE_MEMORY_DEVICE);
   S_diag_i = hypre_CTAlloc(HYPRE_Int, (m+1), HYPRE_MEMORY_DEVICE);

   /* set Comm_Pkg if not yet built */
   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /*
    * 2: Symbolic factorization
    * setup iw and rperm first
    */
   /* allocate work arrays */
   iw = hypre_CTAlloc(HYPRE_Int, 4*n, HYPRE_MEMORY_HOST);
   rperm = iw + 3*n;
   L_diag_i[0] = U_diag_i[0] = S_diag_i[0] = 0;
   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
    */

   if (!permp)
   {
      perm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
      for (i = 0; i < n; i++)
      {
         perm[i] = i;
      }
   }
   else
   {
      perm = permp;
   }

   if (!qpermp)
   {
      qperm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
      for (i = 0; i < n; i++)
      {
         qperm[i] = i;
      }
   }
   else
   {
      qperm = qpermp;
   }

   for (i = 0; i < n; i++)
   {
      rperm[qperm[i]] = i;
   }

   /* do symbolic factorization */
   hypre_ILUSetupILUKSymbolic(n, A_diag_i, A_diag_j, lfil, perm, rperm, iw,
         nLU, L_diag_i, U_diag_i, S_diag_i, &L_diag_j, &U_diag_j, &S_diag_j, u_end);

   /*
    * after this, we have our I,J for L, U and S ready, and L sorted
    * iw are still -1 after symbolic factorization
    * now setup helper array here
    */
   if (L_diag_i[n])
   {
      L_diag_data = hypre_CTAlloc(HYPRE_Real, L_diag_i[n], HYPRE_MEMORY_DEVICE);
   }
   if (U_diag_i[n])
   {
      U_diag_data = hypre_CTAlloc(HYPRE_Real, U_diag_i[n], HYPRE_MEMORY_DEVICE);
   }
   if (S_diag_i[m])
   {
      S_diag_data = hypre_CTAlloc(HYPRE_Real, S_diag_i[m], HYPRE_MEMORY_DEVICE);
   }

   /*
    * 3: Begin real factorization
    * we already have L and U structure ready, so no extra working array needed
    */
   /* first loop for upper part */
   for (ii = 0; ii < nLU; ii++)
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii+1];
      ku = U_diag_i[ii+1];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      /* set up working arrays */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      D_data[ii] = 0.0;
      iw[ii] = ii;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = j;
      }
      /* copy data from A into L, D and U */
      for (j = k1; j < k2; j++)
      {
         /* compute everything in new index */
         col = rperm[A_diag_j[j]];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if (col < ii)
         {
            L_diag_data[icol] = A_diag_data[j];
         }
         else if (col == ii)
         {
            D_data[ii] = A_diag_data[j];
         }
         else
         {
            U_diag_data[icol] = A_diag_data[j];
         }
      }
      /* elimination */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv+1];

         for (k = U_diag_i[jpiv]; k < ku; k++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if (icol < 0)
            {
               /* not in partern */
               continue;
            }
            if (col < ii)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
            else if (col == ii)
            {
               /* diag part */
               D_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
            else
            {
               /* U part */
               U_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
         }
      }
      /* reset working array */
      ku = U_diag_i[ii+1];
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      iw[ii] = -1;
      for (j = U_diag_i[ii]; j < ku ; j++)
      {
         col = U_diag_j[j];
         iw[col] = -1;
      }

      /* diagonal part (we store the inverse) */
      if (fabs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1e-06;
      }
      D_data[ii] = 1./ D_data[ii];
   }

   /* Now lower part for Schur complement */
   for (ii = nLU; ii < n; ii++)
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii+1];
      ku = S_diag_i[ii - nLU +1];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      /* set up working arrays */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      for (j = S_diag_i[ii - nLU]; j < ku; j++)
      {
         col = S_diag_j[j];
         iw[col] = j;
      }
      /* copy data from A into L, and S */
      for (j = k1; j < k2; j++)
      {
         /* compute everything in new index */
         col = rperm[A_diag_j[j]];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if (col < nLU)
         {
            L_diag_data[icol] = A_diag_data[j];
         }
         else
         {
            S_diag_data[icol] = A_diag_data[j];
         }
      }
      /* elimination */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv+1];
         for (k = U_diag_i[jpiv]; k < ku; k++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if (icol < 0)
            {
               /* not in partern */
               continue;
            }
            if (col < nLU)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
            else
            {
               /* S part */
               S_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
         }
      }
      /* reset working array */
      for (j = L_diag_i[ii]; j < kl ; j++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      ku = S_diag_i[ii-nLU+1];
      for (j = S_diag_i[ii-nLU]; j < ku; j++)
      {
         col = S_diag_j[j];
         iw[col] = -1;
         /* remember to update index, S is smaller! */
         S_diag_j[j]-=nLU;
      }
   }

   /*
    * 4: Finishing up and free
    */

   /* First create Schur complement if necessary
    * Check if we need to create Schur complement
    */
   HYPRE_BigInt big_m = (HYPRE_BigInt)m;
   hypre_MPI_Allreduce(&big_m, &total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
   /* only form when total_rows > 0 */
   if ( total_rows > 0 )
   {
      /* now create S */
      /* need to get new column start */
      {
         HYPRE_BigInt global_start;
         hypre_MPI_Scan( &big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
         col_starts[0] = global_start - m;
         col_starts[1] = global_start;
      }

      /* We did nothing to A_offd, so all the data kept, just reorder them
       * The create function takes comm, global num rows/cols,
       *    row/col start, num cols offd, nnz diag, nnz offd
       */
      S_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);
      S_offd_ncols = hypre_CSRMatrixNumCols(A_offd);

      matS = hypre_ParCSRMatrixCreate( comm,
            total_rows,
            total_rows,
            col_starts,
            col_starts,
            S_offd_ncols,
            S_diag_i[m],
            S_offd_nnz);

      /* first put diagonal data in */
      S_diag = hypre_ParCSRMatrixDiag(matS);

      hypre_CSRMatrixI(S_diag) = S_diag_i;
      hypre_CSRMatrixData(S_diag) = S_diag_data;
      hypre_CSRMatrixJ(S_diag) = S_diag_j;

      /* now start to construct offdiag of S */
      S_offd = hypre_ParCSRMatrixOffd(matS);
      S_offd_i = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_colmap = hypre_CTAlloc(HYPRE_BigInt, S_offd_ncols, HYPRE_MEMORY_HOST);

      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i[i+1] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col+1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i+e+1] = k3;
      }

      /* give I, J, DATA to S_offd */
      hypre_CSRMatrixI(S_offd) = S_offd_i;
      hypre_CSRMatrixJ(S_offd) = S_offd_j;
      hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */

      /* get total num of send */
      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg,0);
      HYPRE_Int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg,num_sends);
      send_buf = hypre_TAlloc(HYPRE_BigInt, end - begin, HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for (i = begin; i < end; i++)
      {
         send_buf[i-begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)] - nLU + col_starts[0];
      }

      /* main communication */
      comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf, S_offd_colmap);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;

      hypre_ILUSortOffdColmap(matS);

      /* free */
      hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
   } /* end of forming S */

   /* Assemble LDU matrices */
   /* zero out unfactored rows */
   for (k = nLU; k < n; k++)
   {
      D_data[k] = 1.;
   }

   matL = hypre_ParCSRMatrixCreate( comm,
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixRowStarts(A),
         hypre_ParCSRMatrixColStarts(A),
         0 /* num_cols_offd */,
         L_diag_i[n],
         0 /* num_nonzeros_offd */);

   L_diag = hypre_ParCSRMatrixDiag(matL);
   hypre_CSRMatrixI(L_diag) = L_diag_i;
   if (L_diag_i[n]>0)
   {
      hypre_CSRMatrixData(L_diag) = L_diag_data;
      hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we allocated some initial length, so free them */
      hypre_TFree(L_diag_j, HYPRE_MEMORY_DEVICE);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (L_diag_i[n]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

   matU = hypre_ParCSRMatrixCreate( comm,
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixRowStarts(A),
         hypre_ParCSRMatrixColStarts(A),
         0,
         U_diag_i[n],
         0 );

   U_diag = hypre_ParCSRMatrixDiag(matU);
   hypre_CSRMatrixI(U_diag) = U_diag_i;
   if (U_diag_i[n]>0)
   {
      hypre_CSRMatrixData(U_diag) = U_diag_data;
      hypre_CSRMatrixJ(U_diag) = U_diag_j;
   }
   else
   {
      /* we allocated some initial length, so free them */
      hypre_TFree(U_diag_j, HYPRE_MEMORY_DEVICE);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (U_diag_i[n]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

   /* free */
   hypre_TFree(iw,HYPRE_MEMORY_HOST);
   if (!matS)
   {
      /* we allocate some memory for S, need to free if unused */
      hypre_TFree(S_diag_i,HYPRE_MEMORY_DEVICE);
   }

   if (!permp)
   {
      hypre_TFree(perm, HYPRE_MEMORY_DEVICE);
   }

   if (!qpermp)
   {
      hypre_TFree(qperm, HYPRE_MEMORY_DEVICE);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;
   *Sptr = matS;

   return hypre_error_flag;
}

/* ILUT
 * A: input matrix
 * lfil: maximum nnz per row in L and U
 * tol: droptol array in ILUT
 *    tol[0]: matrix B
 *    tol[1]: matrix E and F
 *    tol[2]: matrix S
 * perm: permutation array indicating ordering of factorization. Perm could come from a
 *    CF_marker: array or a reordering routine.
 * qperm: permutation array for column
 * nLU: size of computed LDU factorization. If nLU < n, Schur compelemnt will be formed
 * nI: number of interial unknowns. nLU should obey nLU <= nI.
 * Lptr, Dptr, Uptr: L, D, U factors.
 * Sptr: Schur complement
 *
 * Keep the largest lfil entries that is greater than some tol relative
 *    to the input tol and the norm of that row in both L and U
 */
HYPRE_Int
hypre_ILUSetupILUT(hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Real *tol,
      HYPRE_Int *permp, HYPRE_Int *qpermp, HYPRE_Int nLU, HYPRE_Int nI, hypre_ParCSRMatrix **Lptr,
      HYPRE_Real** Dptr, hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr, HYPRE_Int **u_end)
{
   /*
    * 1: Setup and create buffers
    * matL/U: the ParCSR matrix for L and U
    * L/U_diag: the diagonal csr matrix of matL/U
    * A_diag_*: tempory pointer for the diagonal matrix of A and its '*' slot
    * ii = outer loop from 0 to nLU - 1
    * i = the real col number in diag inside the outer loop
    * iw =  working array store the reverse of active col number
    * iL = working array store the active col number
    */
   HYPRE_Real               local_nnz, total_nnz;
   HYPRE_Int                i, ii, j, k, k1, k2, k3, kl, ku, col, icol, lenl, lenu, lenhu, lenhlr, lenhll, jpos, jrow;
   HYPRE_Real               inorm, itolb, itolef, itols, dpiv, lxu;
   HYPRE_Int                *iw,*iL;
   HYPRE_Real               *w;

   /* memory management */
   HYPRE_Int                ctrL;
   HYPRE_Int                ctrU;
   HYPRE_Int                initial_alloc = 0;
   HYPRE_Int                capacity_L;
   HYPRE_Int                capacity_U;
   HYPRE_Int                ctrS;
   HYPRE_Int                capacity_S;
   HYPRE_Int                nnz_A;

   /* communication stuffs for S */
   MPI_Comm                 comm             = hypre_ParCSRMatrixComm(A);
   HYPRE_Int                S_offd_nnz, S_offd_ncols;
   hypre_ParCSRCommPkg      *comm_pkg;
   hypre_ParCSRCommHandle   *comm_handle;
   HYPRE_Int                num_procs, my_id;
   HYPRE_BigInt             col_starts[2];
   HYPRE_BigInt             total_rows;
   HYPRE_Int                num_sends;
   HYPRE_Int                begin, end;

   /* data objects for A */
   hypre_CSRMatrix          *A_diag          = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix          *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real               *A_diag_data     = hypre_CSRMatrixData(A_diag);
   HYPRE_Int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Int                *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Int                *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   HYPRE_Real               *A_offd_data     = hypre_CSRMatrixData(A_offd);

   /* data objects for L, D, U */
   hypre_ParCSRMatrix       *matL;
   hypre_ParCSRMatrix       *matU;
   hypre_CSRMatrix          *L_diag;
   hypre_CSRMatrix          *U_diag;
   HYPRE_Real               *D_data;
   HYPRE_Real               *L_diag_data     = NULL;
   HYPRE_Int                *L_diag_i;
   HYPRE_Int                *L_diag_j        = NULL;
   HYPRE_Real               *U_diag_data     = NULL;
   HYPRE_Int                *U_diag_i;
   HYPRE_Int                *U_diag_j        = NULL;

   /* data objects for S */
   hypre_ParCSRMatrix       *matS            = NULL;
   hypre_CSRMatrix          *S_diag;
   hypre_CSRMatrix          *S_offd;
   HYPRE_Real               *S_diag_data     = NULL;
   HYPRE_Int                *S_diag_i        = NULL;
   HYPRE_Int                *S_diag_j        = NULL;
   HYPRE_Int                *S_offd_i        = NULL;
   HYPRE_Int                *S_offd_j        = NULL;
   HYPRE_BigInt                *S_offd_colmap   = NULL;
   HYPRE_Real               *S_offd_data;
   HYPRE_BigInt                *send_buf        = NULL;
   HYPRE_Int                *u_end_array;

   /* reverse permutation */
   HYPRE_Int                *rperm;
   HYPRE_Int                *perm, *qperm;

   /* problem size
    * m is n - nLU, num of rows of local Schur system
    * m_e is the size of interface nodes
    * e is the number of interial rows in local Schur Complement
    */
   HYPRE_Int                n;
   HYPRE_Int                m;
   HYPRE_Int                e;
   HYPRE_Int                m_e;

   /* start setup
    * check input first
    */
   n = hypre_CSRMatrixNumRows(A_diag);
   if (nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU out of range.\n");
   }
   m = n - nLU;
   e = nI - nLU;
   m_e = n - nI;
   if (e < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU should not exceed nI.\n");
   }

   u_end_array = hypre_TAlloc(HYPRE_Int, nLU, HYPRE_MEMORY_HOST);

   /* start set up
    * setup communication stuffs first
    */
   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   /* create if not yet built */
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* setup initial memory, in ILUT, just guess with max nnz per row */
   nnz_A = A_diag_i[nLU];
   if (n > 0)
   {
      initial_alloc = hypre_min(nLU + ceil((nnz_A / 2.0) * nLU / n), nLU * lfil);
   }
   capacity_L = initial_alloc;
   capacity_U = initial_alloc;

   D_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_DEVICE);
   L_diag_i = hypre_CTAlloc(HYPRE_Int, (n+1), HYPRE_MEMORY_DEVICE);
   U_diag_i = hypre_CTAlloc(HYPRE_Int, (n+1), HYPRE_MEMORY_DEVICE);

   L_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
   U_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
   L_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
   U_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_U, HYPRE_MEMORY_DEVICE);

   ctrL = ctrU = 0;

   ctrS = 0;
   S_diag_i = hypre_CTAlloc(HYPRE_Int, (m + 1), HYPRE_MEMORY_DEVICE);
   S_diag_i[0] = 0;
   /* only setup S part when n > nLU */
   if (m > 0)
   {
      capacity_S = hypre_min(m + ceil((nnz_A / 2.0) * m / n), m * lfil);
      S_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_S, HYPRE_MEMORY_DEVICE);
      S_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_S, HYPRE_MEMORY_DEVICE);
   }

   /* setting up working array */
   iw = hypre_CTAlloc(HYPRE_Int,3*n,HYPRE_MEMORY_HOST);
   iL = iw + n;
   w = hypre_CTAlloc(HYPRE_Real,n,HYPRE_MEMORY_HOST);
   for (i = 0; i < n; i++)
   {
      iw[i] = -1;
   }
   L_diag_i[0] = U_diag_i[0] = 0;
   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
    * rperm[old] -> new
    * perm[new]  -> old
    */
   rperm = iw + 2*n;

   if (!permp)
   {
      perm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
      for (i = 0; i < n; i++)
      {
         perm[i] = i;
      }
   }
   else
   {
      perm = permp;
   }

   if (!qpermp)
   {
      qperm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_DEVICE);
      for (i = 0; i < n; i++)
      {
         qperm[i] = i;
      }
   }
   else
   {
      qperm = qpermp;
   }

   for (i = 0; i < n; i++)
   {
      rperm[perm[i]] = i;
   }
   /*
    * 2: Main loop of elimination
    * maintain two heaps
    * |----->*********<-----|-----*********|
    * |col heap***value heap|value in U****|
    */

   /* main outer loop for upper part */
   for (ii = 0; ii < nLU; ii++)
   {
      /* get real row with perm */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      kl = ii-1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += fabs(A_diag_data[j]);
      }
      if (inorm == .0)
      {
         hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: ILUT with zero row.\n");
      }
      inorm /= (HYPRE_Real)(k2-k1);
      /* set the scaled tol for that row */
      itolb = tol[0] * inorm;
      itolef = tol[1] * inorm;

      /* reset displacement */
      lenhll = lenhlr = lenu = 0;
      w[ii] = 0.0;
      iw[ii] = ii;
      /* copy in data from A */
      for (j = k1; j < k2; j++)
      {
         /* get now col number */
         col = rperm[A_diag_j[j]];
         if (col < ii)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = A_diag_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
         }
         else if (col == ii)
         {
            w[ii] = A_diag_data[j];
         }
         else
         {
            lenu++;
            jpos = lenu + ii;
            iL[jpos] = col;
            w[jpos] = A_diag_data[j];
            iw[col] = jpos;
         }
      }

      /*
       * main elimination
       * need to maintain 2 heaps for L, one heap for col and one heaps for value
       * maintian an array for U, and do qsplit with quick sort after that
       * while the heap of col is greater than zero
       */
      while (lenhll > 0)
      {

         /* get the next row from top of the heap */
         jrow = iL[0];
         dpiv = w[0] * D_data[jrow];
         w[0] = dpiv;
         /* now remove it from the top of the heap */
         hypre_ILUMinHeapRemoveIRIi(iL,w,iw,lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;
         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         hypre_swap2(iL,w,lenhll,kl-lenhlr);
         lenhlr++;
         hypre_ILUMaxrHeapAddRabsI(w+kl,iL+kl,lenhlr);
         /* loop for elimination */
         ku = U_diag_i[jrow+1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv*U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col < nLU && fabs(lxu) < itolb) || (col >= nLU && fabs(lxu) < itolef)))
            {
               continue;
            }
            if (icol == -1)
            {
               if (col < ii)
               {
                  /* L part
                   * not already in L part
                   * put it to the end of heap
                   * might overwrite some small entries, no issue
                   */
                  iL[lenhll] = col;
                  w[lenhll] = lxu;
                  iw[col] = lenhll++;
                  /* add to heap, by col number */
                  hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
               }
               else if (col == ii)
               {
                  w[ii] += lxu;
               }
               else
               {
                  /*
                   * not already in U part
                   * put is to the end of heap
                   */
                  lenu++;
                  jpos = lenu + ii;
                  iL[jpos] = col;
                  w[jpos] = lxu;
                  iw[col] = jpos;
               }
            }
            else
            {
               w[icol] += lxu;
            }
         }
      }/* while loop for the elimination of current row */

      if (fabs(w[ii]) < MAT_TOL)
      {
         w[ii]=1e-06;
      }
      D_data[ii] = 1./w[ii];
      iw[ii] = -1;

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii+1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = capacity_L * EXPAND_FACT + 1;
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
         }
         ctrL += lenl;
         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            hypre_ILUMaxrHeapRemoveRabsI(w+kl,iL+kl,lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only U part
       */
      ku = lenu+ii;
      for (j = ii + 1; j <= ku; j++)
      {
         iw[iL[j]] = -1;
      }

      if (lenu < lfil)
      {
         /* we simply keep all of the data, no need to sort */
         lenhu = lenu;
      }
      else
      {
         /* need to sort the first small(hopefully) part of it */
         lenhu = lfil;
         /* quick split, only sort the first small part of the array */
         hypre_ILUMaxQSplitRabsI(w,iL,ii+1,ii+lenhu,ii+lenu);
      }

      U_diag_i[ii+1] = U_diag_i[ii] + lenhu;
      if (lenhu > 0)
      {
         /* test if memory is enough */
         while (ctrU + lenhu > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = capacity_U * EXPAND_FACT + 1;
            U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
            U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_U, HYPRE_MEMORY_DEVICE);
         }
         ctrU += lenhu;
         /* copy large data in */
         for (j = U_diag_i[ii]; j < ctrU; j++)
         {
            jpos = ii+1+j-U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
      /* check and build u_end array */
      if (m > 0)
      {
         hypre_qsort1(U_diag_j,U_diag_data,U_diag_i[ii],U_diag_i[ii+1]-1);
         hypre_BinarySearch2(U_diag_j,nLU,U_diag_i[ii],U_diag_i[ii+1]-1,u_end_array + ii);
      }
      else
      {
         /* Everything is in U */
         u_end_array[ii] = ctrU;
      }
   }/* end of ii loop from 0 to nLU-1 */


   /* now main loop for Schur comlement part */
   for (ii = nLU; ii < n; ii++)
   {
      /* get real row with perm */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      kl = nLU-1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += fabs(A_diag_data[j]);
      }
      if (inorm == .0)
      {
         hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: ILUT with zero row.\n");
      }
      inorm /= (HYPRE_Real)(k2-k1);
      /* set the scaled tol for that row */
      itols = tol[2] * inorm;
      itolef = tol[1] * inorm;

      /* reset displacement */
      lenhll = lenhlr = lenu = 0;
      /* copy in data from A */
      for (j = k1; j < k2; j++)
      {
         /* get now col number */
         col = rperm[A_diag_j[j]];
         if (col < nLU)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = A_diag_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
         }
         else if (col == ii)
         {
            /* the diagonla entry of S */
            iL[nLU] = col;
            w[nLU] = A_diag_data[j];
            iw[col] = nLU;
         }
         else
         {
            /* S part of it */
            lenu++;
            jpos = lenu + nLU;
            iL[jpos] = col;
            w[jpos] = A_diag_data[j];
            iw[col] = jpos;
         }
      }

      /*
       * main elimination
       * need to maintain 2 heaps for L, one heap for col and one heaps for value
       * maintian an array for S, and do qsplit with quick sort after that
       * while the heap of col is greater than zero
       */
      while (lenhll > 0)
      {
         /* get the next row from top of the heap */
         jrow = iL[0];
         dpiv = w[0] * D_data[jrow];
         w[0] = dpiv;
         /* now remove it from the top of the heap */
         hypre_ILUMinHeapRemoveIRIi(iL,w,iw,lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;
         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         hypre_swap2(iL,w,lenhll,kl-lenhlr);
         lenhlr++;
         hypre_ILUMaxrHeapAddRabsI(w+kl,iL+kl,lenhlr);
         /* loop for elimination */
         ku = U_diag_i[jrow+1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv*U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col < nLU && fabs(lxu) < itolef) || ( col >= nLU && fabs(lxu) < itols )))
            {
               continue;
            }
            if (icol == -1)
            {
               if (col < nLU)
               {
                  /* L part
                   * not already in L part
                   * put it to the end of heap
                   * might overwrite some small entries, no issue
                   */
                  iL[lenhll] = col;
                  w[lenhll] = lxu;
                  iw[col] = lenhll++;
                  /* add to heap, by col number */
                  hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
               }
               else if (col == ii)
               {
                  /* the diagonla entry of S */
                  iL[nLU] = col;
                  w[nLU] = A_diag_data[j];
                  iw[col] = nLU;
               }
               else
               {
                  /*
                   * not already in S part
                   * put is to the end of heap
                   */
                  lenu++;
                  jpos = lenu + nLU;
                  iL[jpos] = col;
                  w[jpos] = lxu;
                  iw[col] = jpos;
               }
            }
            else
            {
               w[icol] += lxu;
            }
         }
      }/* while loop for the elimination of current row */

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii+1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = capacity_L * EXPAND_FACT + 1;
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
         }
         ctrL += lenl;
         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j ++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            hypre_ILUMaxrHeapRemoveRabsI(w+kl,iL+kl,lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only S part
       */
      ku = lenu+nLU;
      for (j = nLU; j <= ku; j++)
      {
         iw[iL[j]] = -1;
      }

      /* no dropping at this point of time for S */
      //lenhu = lenu < lfil ? lenu : lfil;
      lenhu = lenu;
      /* quick split, only sort the first small part of the array */
      hypre_ILUMaxQSplitRabsI(w,iL,nLU+1,nLU+lenhu,nLU+lenu);
      /* we have diagonal in S anyway */
      /* test if memory is enough */
      while (ctrS + lenhu + 1 > capacity_S)
      {
         HYPRE_Int tmp = capacity_S;
         capacity_S = capacity_S * EXPAND_FACT + 1;
         S_diag_j = hypre_TReAlloc_v2(S_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_S, HYPRE_MEMORY_DEVICE);
         S_diag_data = hypre_TReAlloc_v2(S_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_S, HYPRE_MEMORY_DEVICE);
      }

      ctrS += (lenhu+1);
      S_diag_i[ii-nLU+1] = ctrS;

      /* copy large data in, diagonal first */
      S_diag_j[S_diag_i[ii-nLU]] = iL[nLU]-nLU;
      S_diag_data[S_diag_i[ii-nLU]] = w[nLU];
      for (j = S_diag_i[ii-nLU] + 1; j < ctrS; j++)
      {
         jpos = nLU+j-S_diag_i[ii-nLU];
         S_diag_j[j] = iL[jpos]-nLU;
         S_diag_data[j] = w[jpos];
      }
   }/* end of ii loop from nLU to n-1 */

   /*
    * 3: Finishing up and free
    */

   /* First create Schur complement if necessary
    * Check if we need to create Schur complement
    */
   HYPRE_BigInt big_m = (HYPRE_BigInt)m;
   hypre_MPI_Allreduce(&big_m, &total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
   /* only form when total_rows > 0 */
   if ( total_rows > 0 )
   {
      /* now create S */
      /* need to get new column start */
      {
         HYPRE_BigInt global_start;
         hypre_MPI_Scan( &big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
         col_starts[0] = global_start - m;
         col_starts[1] = global_start;
      }
      /* We did nothing to A_offd, so all the data kept, just reorder them
       * The create function takes comm, global num rows/cols,
       *    row/col start, num cols offd, nnz diag, nnz offd
       */
      S_offd_nnz = hypre_CSRMatrixNumNonzeros(A_offd);
      S_offd_ncols = hypre_CSRMatrixNumCols(A_offd);

      matS = hypre_ParCSRMatrixCreate( comm,
            total_rows,
            total_rows,
            col_starts,
            col_starts,
            S_offd_ncols,
            S_diag_i[m],
            S_offd_nnz);

      /* first put diagonal data in */
      S_diag = hypre_ParCSRMatrixDiag(matS);

      hypre_CSRMatrixI(S_diag) = S_diag_i;
      hypre_CSRMatrixData(S_diag) = S_diag_data;
      hypre_CSRMatrixJ(S_diag) = S_diag_j;

      /* now start to construct offdiag of S */
      S_offd = hypre_ParCSRMatrixOffd(matS);
      S_offd_i = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_DEVICE);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_DEVICE);
      S_offd_colmap = hypre_CTAlloc(HYPRE_BigInt, S_offd_ncols, HYPRE_MEMORY_HOST);

      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i[i] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col+1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i+e+1] = k3;
      }

      /* give I, J, DATA to S_offd */
      hypre_CSRMatrixI(S_offd) = S_offd_i;
      hypre_CSRMatrixJ(S_offd) = S_offd_j;
      hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */

      /* get total num of send */
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg,0);
      end = hypre_ParCSRCommPkgSendMapStart(comm_pkg,num_sends);
      send_buf = hypre_TAlloc(HYPRE_BigInt, end - begin, HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for (i = begin; i < end; i++)
      {
         send_buf[i-begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)] - nLU + col_starts[0];
      }

      /* main communication */
      comm_handle = hypre_ParCSRCommHandleCreate(21, comm_pkg, send_buf, S_offd_colmap);
      /* need this to synchronize, Isend & Irecv used in above functions */
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;

      hypre_ILUSortOffdColmap(matS);

      /* free */
      hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
   } /* end of forming S */

   /* now start to construct L and U */
   for (k = nLU; k < n; k++)
   {
      /* set U after nLU to be 0, and diag to be one */
      U_diag_i[k+1] = U_diag_i[nLU];
      D_data[k] = 1.;
   }

   /* create parcsr matrix */
   matL = hypre_ParCSRMatrixCreate( comm,
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixRowStarts(A),
         hypre_ParCSRMatrixColStarts(A),
         0,
         L_diag_i[n],
         0 );

   L_diag = hypre_ParCSRMatrixDiag(matL);
   hypre_CSRMatrixI(L_diag) = L_diag_i;
   if (L_diag_i[n] > 0)
   {
      hypre_CSRMatrixData(L_diag) = L_diag_data;
      hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we initialized some anyway, so remove if unused */
      hypre_TFree(L_diag_j,HYPRE_MEMORY_DEVICE);
      hypre_TFree(L_diag_data,HYPRE_MEMORY_DEVICE);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (L_diag_i[n]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

   matU = hypre_ParCSRMatrixCreate( comm,
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixRowStarts(A),
         hypre_ParCSRMatrixColStarts(A),
         0,
         U_diag_i[n],
         0 );

   U_diag = hypre_ParCSRMatrixDiag(matU);
   hypre_CSRMatrixI(U_diag) = U_diag_i;
   if (U_diag_i[n] > 0)
   {
      hypre_CSRMatrixData(U_diag) = U_diag_data;
      hypre_CSRMatrixJ(U_diag) = U_diag_j;
   }
   else
   {
      /* we initialized some anyway, so remove if unused */
      hypre_TFree(U_diag_j,HYPRE_MEMORY_DEVICE);
      hypre_TFree(U_diag_data,HYPRE_MEMORY_DEVICE);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (U_diag_i[n]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

   /* free working array */
   hypre_TFree(iw,HYPRE_MEMORY_HOST);
   hypre_TFree(w,HYPRE_MEMORY_HOST);

   if (!matS)
   {
      hypre_TFree(S_diag_i,HYPRE_MEMORY_DEVICE);
   }

   if (!permp)
   {
      hypre_TFree(perm, HYPRE_MEMORY_DEVICE);
   }

   if (!qpermp)
   {
      hypre_TFree(qperm, HYPRE_MEMORY_DEVICE);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;
   *Sptr = matS;
   *u_end = u_end_array;

   return hypre_error_flag;
}


/* NSH setup */
/* Setup NSH data */
HYPRE_Int
hypre_NSHSetup( void               *nsh_vdata,
                hypre_ParCSRMatrix *A,
                hypre_ParVector    *f,
                hypre_ParVector    *u )
{
   MPI_Comm             comm              = hypre_ParCSRMatrixComm(A);
   hypre_ParNSHData     *nsh_data         = (hypre_ParNSHData*) nsh_vdata;

   //   HYPRE_Int            i;
   // HYPRE_Int            num_threads;
   // HYPRE_Int            debug_flag = 0;

   /* pointers to NSH data */
   HYPRE_Int            logging           = hypre_ParNSHDataLogging(nsh_data);
   HYPRE_Int            print_level       = hypre_ParNSHDataPrintLevel(nsh_data);

   hypre_ParCSRMatrix   *matA             = hypre_ParNSHDataMatA(nsh_data);
   hypre_ParCSRMatrix   *matM             = hypre_ParNSHDataMatM(nsh_data);

   //   HYPRE_Int            n                 = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int            num_procs,  my_id;

   hypre_ParVector      *Utemp;
   hypre_ParVector      *Ftemp;
   hypre_ParVector      *F_array          = hypre_ParNSHDataF(nsh_data);
   hypre_ParVector      *U_array          = hypre_ParNSHDataU(nsh_data);
   hypre_ParVector      *residual         = hypre_ParNSHDataResidual(nsh_data);
   HYPRE_Real           *rel_res_norms    = hypre_ParNSHDataRelResNorms(nsh_data);

   /* solver setting */
   HYPRE_Real           *droptol          = hypre_ParNSHDataDroptol(nsh_data);
   HYPRE_Real           mr_tol            = hypre_ParNSHDataMRTol(nsh_data);
   HYPRE_Int            mr_max_row_nnz    = hypre_ParNSHDataMRMaxRowNnz(nsh_data);
   HYPRE_Int            mr_max_iter       = hypre_ParNSHDataMRMaxIter(nsh_data);
   HYPRE_Int            mr_col_version    = hypre_ParNSHDataMRColVersion(nsh_data);
   HYPRE_Real           nsh_tol           = hypre_ParNSHDataNSHTol(nsh_data);
   HYPRE_Int            nsh_max_row_nnz   = hypre_ParNSHDataNSHMaxRowNnz(nsh_data);
   HYPRE_Int            nsh_max_iter      = hypre_ParNSHDataNSHMaxIter(nsh_data);

   /* ----- begin -----*/

   //num_threads = hypre_NumThreads();

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);

   /* Free Previously allocated data, if any not destroyed */
   if (matM)
   {
      hypre_TFree(matM, HYPRE_MEMORY_HOST);
      matM = NULL;
   }

   /* clear old l1_norm data, if created */
   if (hypre_ParNSHDataL1Norms(nsh_data))
   {
      hypre_TFree(hypre_ParNSHDataL1Norms(nsh_data), HYPRE_MEMORY_HOST);
      hypre_ParNSHDataL1Norms(nsh_data) = NULL;
   }

   /* setup temporary storage
    * first check is they've already here
    */
   if (hypre_ParNSHDataUTemp(nsh_data))
   {
      hypre_ParVectorDestroy(hypre_ParNSHDataUTemp(nsh_data));
      hypre_ParNSHDataUTemp(nsh_data) = NULL;
   }
   if (hypre_ParNSHDataFTemp(nsh_data))
   {
      hypre_ParVectorDestroy(hypre_ParNSHDataFTemp(nsh_data));
      hypre_ParNSHDataFTemp(nsh_data) = NULL;
   }
   if (hypre_ParNSHDataResidual(nsh_data))
   {
      hypre_ParVectorDestroy(hypre_ParNSHDataResidual(nsh_data));
      hypre_ParNSHDataResidual(nsh_data) = NULL;
   }
   if (hypre_ParNSHDataRelResNorms(nsh_data))
   {
      hypre_TFree(hypre_ParNSHDataRelResNorms(nsh_data), HYPRE_MEMORY_HOST);
      hypre_ParNSHDataRelResNorms(nsh_data) = NULL;
   }

   /* start to create working vectors */
   Utemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Utemp);
   hypre_ParNSHDataUTemp(nsh_data) = Utemp;

   Ftemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
         hypre_ParCSRMatrixGlobalNumRows(A),
         hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Ftemp);
   hypre_ParNSHDataFTemp(nsh_data) = Ftemp;
   /* set matrix, solution and rhs pointers */
   matA = A;
   F_array = f;
   U_array = u;

   /* NSH compute approximate inverse, see par_ilu.c */
   hypre_ILUParCSRInverseNSH(matA, &matM, droptol, mr_tol, nsh_tol, DIVIDE_TOL, mr_max_row_nnz,
         nsh_max_row_nnz, mr_max_iter, nsh_max_iter, mr_col_version, print_level);

   /* set pointers to NSH data */
   hypre_ParNSHDataMatA(nsh_data) = matA;
   hypre_ParNSHDataF(nsh_data) = F_array;
   hypre_ParNSHDataU(nsh_data) = U_array;
   hypre_ParNSHDataMatM(nsh_data) = matM;

   /* compute operator complexity */
   hypre_ParCSRMatrixSetDNumNonzeros(matA);
   hypre_ParCSRMatrixSetDNumNonzeros(matM);
   /* compute complexity */
   hypre_ParNSHDataOperatorComplexity(nsh_data) =  hypre_ParCSRMatrixDNumNonzeros(matM)/hypre_ParCSRMatrixDNumNonzeros(matA);
   if (my_id == 0)
   {
      hypre_printf("NSH SETUP: operator complexity = %f  \n", hypre_ParNSHDataOperatorComplexity(nsh_data));
   }

   if ( logging > 1 ) {
      residual =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
               hypre_ParCSRMatrixGlobalNumRows(matA),
               hypre_ParCSRMatrixRowStarts(matA) );
      hypre_ParVectorInitialize(residual);
      hypre_ParNSHDataResidual(nsh_data)= residual;
   }
   else{
      hypre_ParNSHDataResidual(nsh_data) = NULL;
   }
   rel_res_norms = hypre_CTAlloc(HYPRE_Real, hypre_ParNSHDataMaxIter(nsh_data), HYPRE_MEMORY_HOST);
   hypre_ParNSHDataRelResNorms(nsh_data) = rel_res_norms;

   return hypre_error_flag;
}


/* ILU(0) for RAS, has some external rows
 * A = input matrix
 * perm = permutation array indicating ordering of factorization. Perm could come from a
 *    CF_marker array or a reordering routine.
 * nLU = size of computed LDU factorization.
 * Lptr, Dptr, Uptr, Sptr = L, D, U, S factors.
 * will form global Schur Matrix if nLU < n
 */
HYPRE_Int
hypre_ILUSetupILU0RAS(hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int nLU,
      hypre_ParCSRMatrix **Lptr, HYPRE_Real** Dptr, hypre_ParCSRMatrix **Uptr)
{
   HYPRE_Int                i, ii, j, k, k1, k2, ctrU, ctrL, lenl, lenu, jpiv, col, jpos;
   HYPRE_Int                *iw, *iL, *iU;
   HYPRE_Real               dd, t, dpiv, lxu, *wU, *wL;

   /* communication stuffs for S */
   MPI_Comm                 comm          = hypre_ParCSRMatrixComm(A);
   HYPRE_Int                num_procs;
   //   HYPRE_Int                S_offd_nnz, S_offd_ncols;
   hypre_ParCSRCommPkg      *comm_pkg;
   //   hypre_ParCSRCommHandle   *comm_handle;
   //   HYPRE_Int                num_sends, begin, end;
   //   HYPRE_Int                *send_buf     = NULL;

   /* data objects for A */
   hypre_CSRMatrix          *A_diag       = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix          *A_offd       = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real               *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int                *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int                *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real               *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int                *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Int                *A_offd_j     = hypre_CSRMatrixJ(A_offd);

   /* size of problem and external matrix */
   HYPRE_Int                n             =  hypre_CSRMatrixNumRows(A_diag);
   //   HYPRE_Int                m             = n - nLU;
   HYPRE_Int                ext           = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int                total_rows    = n + ext;
   HYPRE_BigInt             col_starts[2];
   HYPRE_BigInt             global_num_rows;
   HYPRE_Real               local_nnz, total_nnz;

   /* data objects for L, D, U */
   hypre_ParCSRMatrix       *matL;
   hypre_ParCSRMatrix       *matU;
   hypre_CSRMatrix          *L_diag;
   hypre_CSRMatrix          *U_diag;
   HYPRE_Real               *D_data;
   HYPRE_Real               *L_diag_data;
   HYPRE_Int                *L_diag_i;
   HYPRE_Int                *L_diag_j;
   HYPRE_Real               *U_diag_data;
   HYPRE_Int                *U_diag_i;
   HYPRE_Int                *U_diag_j;

   /* data objects for E, external matrix */
   HYPRE_Int                *E_i;
   HYPRE_Int                *E_j;
   HYPRE_Real               *E_data;

   /* memory management */
   HYPRE_Int                initial_alloc = 0;
   HYPRE_Int                capacity_L;
   HYPRE_Int                capacity_U;
   HYPRE_Int                nnz_A = A_diag_i[n];

   /* reverse permutation array */
   HYPRE_Int                *rperm;
   /* the original permutation array */
   HYPRE_Int                *perm_old;

   /* start setup
    * get communication stuffs first
    */
   hypre_MPI_Comm_size(comm,&num_procs);
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   /* setup if not yet built */
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* check for correctness */
   if (nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU out of range.\n");
   }

   /* Allocate memory for L,D,U,S factors */
   if (n > 0)
   {
      initial_alloc = (n + ext) + ceil((nnz_A / 2.0)*total_rows/n);
   }
   capacity_L = initial_alloc;
   capacity_U = initial_alloc;

   D_data      = hypre_TAlloc(HYPRE_Real, total_rows, HYPRE_MEMORY_DEVICE);
   L_diag_i    = hypre_TAlloc(HYPRE_Int, total_rows+1, HYPRE_MEMORY_DEVICE);
   L_diag_j    = hypre_TAlloc(HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
   L_diag_data = hypre_TAlloc(HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
   U_diag_i    = hypre_TAlloc(HYPRE_Int, total_rows+1, HYPRE_MEMORY_DEVICE);
   U_diag_j    = hypre_TAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
   U_diag_data = hypre_TAlloc(HYPRE_Real, capacity_U, HYPRE_MEMORY_DEVICE);

   /* allocate working arrays */
   iw          = hypre_TAlloc(HYPRE_Int, 4*total_rows, HYPRE_MEMORY_HOST);
   iL          = iw+total_rows;
   rperm       = iw + 2 * total_rows;
   perm_old    = perm;
   perm        = iw + 3 * total_rows;
   wL          = hypre_TAlloc(HYPRE_Real, total_rows, HYPRE_MEMORY_HOST);
   ctrU = ctrL = 0;
   L_diag_i[0] = U_diag_i[0] = 0;
   /* set marker array iw to -1 */
   for (i = 0; i < total_rows; i++)
   {
      iw[i] = -1;
   }

   /* expand perm to suit extra data, remember to free */
   for (i = 0; i < n; i++)
   {
      perm[i] = perm_old[i];
   }
   for (i = n; i < total_rows; i++)
   {
      perm[i] = i;
   }

   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
    */
   for (i = 0; i < total_rows; i++)
   {
      rperm[perm[i]] = i;
   }

   /* get external rows */
   hypre_ILUBuildRASExternalMatrix(A, rperm, &E_i, &E_j, &E_data);

   /*---------  Begin Factorization. Work in permuted space  ----
    * this is the first part, without offd
    */
   for (ii = 0; ii < nLU; ii++)
   {
      // get row i
      i = perm[ii];
      // get extents of row i
      k1=A_diag_i[i];
      k2=A_diag_i[i+1];

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL+ii;
      wU = wL+ii;
      /*--------------------  diagonal entry */
      dd = 0.0;
      lenl  = lenu = 0;
      iw[ii] = ii;
      /*-------------------- scan & unwrap column */
      for (j = k1; j < k2; j++)
      {
         col = rperm[A_diag_j[j]];
         t = A_diag_data[j];
         if ( col < ii )
         {
            iw[col] = lenl;
            iL[lenl] = col;
            wL[lenl++] = t;
         }
         else if (col > ii)
         {
            iw[col] = lenu;
            iU[lenu] = col;
            wU[lenu++] = t;
         }
         else
         {
            dd=t;
         }
      }

      /* eliminate row */
      /*-------------------------------------------------------------------------
       *  In order to do the elimination in the correct order we must select the
       *  smallest column index among iL[k], k = j, j+1, ..., lenl-1. For ILU(0),
       *  no new fill-ins are expect, so we can pre-sort iL and wL prior to the
       *  entering the elimination loop.
       *-----------------------------------------------------------------------*/
      //      hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      hypre_qsort3ir(iL, wL, iw, 0, (lenl-1));
      for (j = 0; j < lenl; j++)
      {
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;

         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv+1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if (jpos < 0)
            {
               continue;
            }

            lxu = - U_diag_data[k] * dpiv;
            if (col < ii)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if (col > ii)
            {
               /* dealing with U part */
               wU[jpos] += lxu;
            }
            else
            {
               /* diagonal update */
               dd += lxu;
            }
         }
      }
      /* restore iw (only need to restore diagonal and U part */
      iw[ii] = -1;
      for (j = 0; j < lenu; j++)
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      while ((ctrL+lenl) > capacity_L)
      {
         HYPRE_Int tmp = capacity_L;
         capacity_L = capacity_L * EXPAND_FACT + 1;
         L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
         L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
      }
      //hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      //hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      L_diag_i[ii+1] = (ctrL+=lenl);

      /* diagonal part (we store the inverse) */
      if (fabs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;
      }
      D_data[ii] = 1./dd;

      /* U part */
      /* Check that memory is sufficient */
      while ((ctrU+lenu) > capacity_U)
      {
         HYPRE_Int tmp = capacity_U;
         capacity_U = capacity_U * EXPAND_FACT + 1;
         U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
         U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_U, HYPRE_MEMORY_DEVICE);
      }
      //hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      //hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      U_diag_i[ii+1] = (ctrU+=lenu);
   }

   /*---------  Begin Factorization in lower part  ----
    * here we need to get off diagonals in
    */
   for (ii = nLU; ii < n; ii++)
   {
      // get row i
      i = perm[ii];
      // get extents of row i
      k1=A_diag_i[i];
      k2=A_diag_i[i+1];

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL+ii;
      wU = wL+ii;
      /*--------------------  diagonal entry */
      dd = 0.0;
      lenl  = lenu = 0;
      iw[ii] = ii;
      /*-------------------- scan & unwrap column */
      for (j = k1; j < k2; j++)
      {
         col = rperm[A_diag_j[j]];
         t = A_diag_data[j];
         if (col < ii)
         {
            iw[col] = lenl;
            iL[lenl] = col;
            wL[lenl++] = t;
         }
         else if (col > ii)
         {
            iw[col] = lenu;
            iU[lenu] = col;
            wU[lenu++] = t;
         }
         else
         {
            dd=t;
         }
      }

      /*------------------ sjcan offd*/
      k1=A_offd_i[i];
      k2=A_offd_i[i+1];
      for (j = k1; j < k2; j++)
      {
         /* add offd to U part, all offd are U for this part */
         col = A_offd_j[j] + n;
         t = A_offd_data[j];
         iw[col] = lenu;
         iU[lenu] = col;
         wU[lenu++] = t;
      }

      /* eliminate row */
      /*-------------------------------------------------------------------------
       *  In order to do the elimination in the correct order we must select the
       *  smallest column index among iL[k], k = j, j+1, ..., lenl-1. For ILU(0),
       *  no new fill-ins are expect, so we can pre-sort iL and wL prior to the
       *  entering the elimination loop.
       *-----------------------------------------------------------------------*/
      //      hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      hypre_qsort3ir(iL, wL, iw, 0, (lenl-1));
      for (j = 0; j < lenl; j++)
      {
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;

         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv+1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if (jpos < 0)
            {
               continue;
            }

            lxu = - U_diag_data[k] * dpiv;
            if (col < ii)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if (col > ii)
            {
               /* dealing with U part */
               wU[jpos] += lxu;
            }
            else
            {
               /* diagonal update */
               dd += lxu;
            }
         }
      }
      /* restore iw (only need to restore diagonal and U part */
      iw[ii] = -1;
      for (j = 0; j < lenu; j++)
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      while ((ctrL+lenl) > capacity_L)
      {
         HYPRE_Int tmp = capacity_L;
         capacity_L = capacity_L * EXPAND_FACT + 1;
         L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
         L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
      }
      //hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      //hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      L_diag_i[ii+1] = (ctrL+=lenl);

      /* diagonal part (we store the inverse) */
      if (fabs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;
      }
      D_data[ii] = 1./dd;

      /* U part */
      /* Check that memory is sufficient */
      while ((ctrU+lenu) > capacity_U)
      {
         HYPRE_Int tmp = capacity_U;
         capacity_U = capacity_U * EXPAND_FACT + 1;
         U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
         U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_U, HYPRE_MEMORY_DEVICE);
      }
      //hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      //hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      U_diag_i[ii+1] = (ctrU+=lenu);
   }

   /*---------  Begin Factorization in external part  ----
    * here we need to get off diagonals in
    */
   for (ii = n ; ii < total_rows ; ii++)
   {
      // get row i
      i = ii-n;
      // get extents of row i
      k1=E_i[i];
      k2=E_i[i+1];

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL+ii;
      wU = wL+ii;
      /*--------------------  diagonal entry */
      dd = 0.0;
      lenl  = lenu = 0;
      iw[ii] = ii;
      /*-------------------- scan & unwrap column */
      for (j = k1; j < k2; j++)
      {
         col = rperm[E_j[j]];
         t = E_data[j];
         if (col < ii)
         {
            iw[col] = lenl;
            iL[lenl] = col;
            wL[lenl++] = t;
         }
         else if (col > ii)
         {
            iw[col] = lenu;
            iU[lenu] = col;
            wU[lenu++] = t;
         }
         else
         {
            dd=t;
         }
      }

      /* eliminate row */
      /*-------------------------------------------------------------------------
       *  In order to do the elimination in the correct order we must select the
       *  smallest column index among iL[k], k = j, j+1, ..., lenl-1. For ILU(0),
       *  no new fill-ins are expect, so we can pre-sort iL and wL prior to the
       *  entering the elimination loop.
       *-----------------------------------------------------------------------*/
      //      hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      hypre_qsort3ir(iL, wL, iw, 0, (lenl-1));
      for (j = 0; j < lenl; j++)
      {
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;

         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv+1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if (jpos < 0)
            {
               continue;
            }

            lxu = - U_diag_data[k] * dpiv;
            if (col < ii)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if (col > ii)
            {
               /* dealing with U part */
               wU[jpos] += lxu;
            }
            else
            {
               /* diagonal update */
               dd += lxu;
            }
         }
      }
      /* restore iw (only need to restore diagonal and U part */
      iw[ii] = -1;
      for (j = 0; j < lenu; j++)
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      while ((ctrL+lenl) > capacity_L)
      {
         HYPRE_Int tmp = capacity_L;
         capacity_L = capacity_L * EXPAND_FACT + 1;
         L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
         L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
      }
      //hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      //hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      L_diag_i[ii+1] = (ctrL+=lenl);

      /* diagonal part (we store the inverse) */
      if (fabs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;
      }
      D_data[ii] = 1./dd;

      /* U part */
      /* Check that memory is sufficient */
      while ((ctrU+lenu) > capacity_U)
      {
         HYPRE_Int tmp = capacity_U;
         capacity_U = capacity_U * EXPAND_FACT + 1;
         U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
         U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_U, HYPRE_MEMORY_DEVICE);
      }
      //hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      //hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu,
                    HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      U_diag_i[ii+1] = (ctrU+=lenu);
   }

   HYPRE_BigInt big_total_rows = (HYPRE_BigInt)total_rows;
   hypre_MPI_Allreduce(&big_total_rows, &global_num_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

   /* need to get new column start */
   {
      HYPRE_BigInt global_start;
      hypre_MPI_Scan( &big_total_rows, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
      col_starts[0] = global_start - total_rows;
      col_starts[1] = global_start;
   }

   matL = hypre_ParCSRMatrixCreate( comm,
         global_num_rows,
         global_num_rows,
         col_starts,
         col_starts,
         0,
         ctrL,
         0 );

   L_diag = hypre_ParCSRMatrixDiag(matL);
   hypre_CSRMatrixI(L_diag) = L_diag_i;
   if (ctrL)
   {
      hypre_CSRMatrixData(L_diag) = L_diag_data;
      hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we've allocated some memory, so free if not used */
      hypre_TFree(L_diag_j,HYPRE_MEMORY_DEVICE);
      hypre_TFree(L_diag_data,HYPRE_MEMORY_DEVICE);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) ctrL;
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

   matU = hypre_ParCSRMatrixCreate( comm,
         global_num_rows,
         global_num_rows,
         col_starts,
         col_starts,
         0,
         ctrU,
         0 );

   U_diag = hypre_ParCSRMatrixDiag(matU);
   hypre_CSRMatrixI(U_diag) = U_diag_i;
   if (ctrU)
   {
      hypre_CSRMatrixData(U_diag) = U_diag_data;
      hypre_CSRMatrixJ(U_diag) = U_diag_j;
   }
   else
   {
      /* we've allocated some memory, so free if not used */
      hypre_TFree(U_diag_j,HYPRE_MEMORY_DEVICE);
      hypre_TFree(U_diag_data,HYPRE_MEMORY_DEVICE);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) ctrU;
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;
   /* free memory */
   hypre_TFree(wL,HYPRE_MEMORY_HOST);
   hypre_TFree(iw,HYPRE_MEMORY_HOST);

   /* free external data */
   if (E_i)
   {
      hypre_TFree(E_i, HYPRE_MEMORY_HOST);
   }
   if (E_j)
   {
      hypre_TFree(E_j, HYPRE_MEMORY_HOST);
      hypre_TFree(E_data, HYPRE_MEMORY_HOST);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;

   return hypre_error_flag;
}



/* ILU(k) symbolic factorization for RAS
 * n = total rows of input
 * lfil = level of fill-in, the k in ILU(k)
 * perm = permutation array indicating ordering of factorization. Perm could come from a
 * rperm = reverse permutation array, used here to avoid duplicate memory allocation
 * iw = working array, used here to avoid duplicate memory allocation
 * nLU = size of computed LDU factorization.
 * A/L/U/E_i = the I slot of A, L, U and E
 * A/L/U/E_j = the J slot of A, L, U and E
 * will form global Schur Matrix if nLU < n
 */
HYPRE_Int
hypre_ILUSetupILUKRASSymbolic(HYPRE_Int n, HYPRE_Int *A_diag_i, HYPRE_Int *A_diag_j, HYPRE_Int *A_offd_i, HYPRE_Int *A_offd_j,
                              HYPRE_Int *E_i, HYPRE_Int *E_j, HYPRE_Int ext,
                              HYPRE_Int lfil, HYPRE_Int *perm,
                              HYPRE_Int *rperm,   HYPRE_Int *iw,   HYPRE_Int nLU,
                              HYPRE_Int *L_diag_i, HYPRE_Int *U_diag_i,
                              HYPRE_Int **L_diag_j, HYPRE_Int **U_diag_j)
{
   /*
    * 1: Setup and create buffers
    * A_diag_*: tempory pointer for the diagonal matrix of A and its '*' slot
    * ii: outer loop from 0 to nLU - 1
    * i: the real col number in diag inside the outer loop
    * iw:  working array store the reverse of active col number
    * iL: working array store the active col number
    * iLev: working array store the active level of current row
    * lenl/u: current position in iw and so
    * ctrL/U/S: global position in J
    */

   HYPRE_Int      *temp_L_diag_j, *temp_U_diag_j, *u_levels;
   HYPRE_Int      *iL, *iLev;
   HYPRE_Int      ii, i, j, k, ku, lena, lenl, lenu, lenh, ilev, lev, col, icol;
   //   HYPRE_Int      m = n - nLU;
   HYPRE_Int      total_rows = ext + n;

   /* memory management */
   HYPRE_Int      ctrL;
   HYPRE_Int      ctrU;
   HYPRE_Int      capacity_L;
   HYPRE_Int      capacity_U;
   HYPRE_Int      initial_alloc = 0;
   HYPRE_Int      nnz_A;

   /* set iL and iLev to right place in iw array */
   iL             = iw + total_rows;
   iLev           = iw + 2*total_rows;

   /* setup initial memory used */
   nnz_A          = A_diag_i[n];
   if (n > 0)
   {
      initial_alloc  = (n + ext) + ceil((nnz_A / 2.0) * total_rows / n);
   }
   capacity_L     = initial_alloc;
   capacity_U     = initial_alloc;

   /* allocate other memory for L and U struct */
   temp_L_diag_j  = hypre_CTAlloc(HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
   temp_U_diag_j  = hypre_CTAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);

   u_levels       = hypre_CTAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
   ctrL = ctrU = 0;

   /* set initial value for working array */
   for (ii = 0; ii < total_rows; ii++)
   {
      iw[ii] = -1;
   }

   /*
    * 2: Start of main loop
    * those in iL are NEW col index (after permutation)
    */
   for (ii = 0; ii < nLU; ii++)
   {
      i = perm[ii];
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = ii;
      lena = A_diag_i[i+1];
      /* put those already inside original pattern, and set their level to 0 */
      for (j = A_diag_i[i]; j < lena; j++)
      {
         /* get the neworder of that col */
         col = rperm[A_diag_j[j]];
         if (col < ii)
         {
            /*
             * this is an entry in L
             * we maintain a heap structure for L part
             */
            iL[lenh] = col;
            iLev[lenh] = 0;
            iw[col] = lenh++;
            /*now miantian a heap structure*/
            hypre_ILUMinHeapAddIIIi(iL,iLev,iw,lenh);
         }
         else if (col > ii)
         {
            /* this is an entry in U */
            iL[lenu] = col;
            iLev[lenu] = 0;
            iw[col] = lenu++;
         }
      }/* end of j loop for adding pattern in original matrix */

      /*
       * search lower part of current row and update pattern based on level
       */
      while (lenh > 0)
      {
         /*
          * k is now the new col index after permutation
          * the first element of the heap is the smallest
          */
         k = iL[0];
         ilev = iLev[0];
         /*
          * we now need to maintain the heap structure
          */
         hypre_ILUMinHeapRemoveIIIi(iL,iLev,iw,lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k]=-1;
         hypre_swap2i(iL,iLev,ii-lenl,lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k+1];
         for (j = U_diag_i[k]; j < ku; j++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if (lev > lfil)
            {
               continue;
            }
            if (icol < 0)
            {
               /* not yet in */
               if (col < ii)
               {
                  /*
                   * if we add to the left L, we need to maintian the
                   *    heap structure
                   */
                  iL[lenh] = col;
                  iLev[lenh] = lev;
                  iw[col] = lenh++;
                  /*swap it with the element right after the heap*/

                  /* maintain the heap */
                  hypre_ILUMinHeapAddIIIi(iL,iLev,iw,lenh);
               }
               else if (col > ii)
               {
                  iL[lenu] = col;
                  iLev[lenu] = lev;
                  iw[col] = lenu++;
               }
            }
            else
            {
               iLev[icol] = hypre_min(lev, iLev[icol]);
            }
         }/* end of loop j for level update */
      }/* end of while loop for iith row */

      /* now update everything, indices, levels and so */
      L_diag_i[ii+1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = capacity_L * EXPAND_FACT + 1;
            temp_L_diag_j = hypre_TReAlloc_v2(temp_L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL+j] = iL[ii-j-1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii+1] = U_diag_i[ii] + k;
      if (k > 0)
      {
         /* check if memory is enough */
         while (ctrU + k > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = capacity_U * EXPAND_FACT + 1;
            temp_U_diag_j = hypre_TReAlloc_v2(temp_U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
            u_levels = hypre_TReAlloc_v2(u_levels, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
         }
         //hypre_TMemcpy(temp_U_diag_j+ctrU,iL+ii,HYPRE_Int,k,HYPRE_MEMORY_DEVICE,HYPRE_MEMORY_HOST);
         hypre_TMemcpy(temp_U_diag_j+ctrU, iL+ii, HYPRE_Int, k,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(u_levels+ctrU, iLev+ii, HYPRE_Int, k,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         ctrU += k;
      }

      /* reset iw */
      for (j = ii; j < lenu; j++)
      {
         iw[iL[j]] = -1;
      }

   }/* end of main loop ii from 0 to nLU-1 */

   /*
    * Offd part
    */
   for (ii = nLU; ii < n; ii++)
   {
      i = perm[ii];
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = ii;
      lena = A_diag_i[i+1];
      /* put those already inside original pattern, and set their level to 0 */
      for (j = A_diag_i[i]; j < lena; j++)
      {
         /* get the neworder of that col */
         col = rperm[A_diag_j[j]];
         if (col < ii)
         {
            /*
             * this is an entry in L
             * we maintain a heap structure for L part
             */
            iL[lenh] = col;
            iLev[lenh] = 0;
            iw[col] = lenh++;
            /*now miantian a heap structure*/
            hypre_ILUMinHeapAddIIIi(iL,iLev,iw,lenh);
         }
         else if (col > ii)
         {
            /* this is an entry in U */
            iL[lenu] = col;
            iLev[lenu] = 0;
            iw[col] = lenu++;
         }
      }/* end of j loop for adding pattern in original matrix */

      /* put those already inside offd pattern in, and set their level to 0 */
      lena = A_offd_i[i+1];
      for (j = A_offd_i[i]; j < lena; j++)
      {
         /* the offd cols are in order */
         col = A_offd_j[j] + n;
         /* col for sure to be greater than ii */
         iL[lenu] = col;
         iLev[lenu] = 0;
         iw[col] = lenu++;
      }

      /*
       * search lower part of current row and update pattern based on level
       */
      while (lenh > 0)
      {
         /*
          * k is now the new col index after permutation
          * the first element of the heap is the smallest
          */
         k = iL[0];
         ilev = iLev[0];
         /*
          * we now need to maintain the heap structure
          */
         hypre_ILUMinHeapRemoveIIIi(iL,iLev,iw,lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k]=-1;
         hypre_swap2i(iL,iLev,ii-lenl,lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k+1];
         for (j = U_diag_i[k]; j < ku; j++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if (lev > lfil)
            {
               continue;
            }
            if (icol < 0)
            {
               /* not yet in */
               if (col < ii)
               {
                  /*
                   * if we add to the left L, we need to maintian the
                   *    heap structure
                   */
                  iL[lenh] = col;
                  iLev[lenh] = lev;
                  iw[col] = lenh++;
                  /*swap it with the element right after the heap*/

                  /* maintain the heap */
                  hypre_ILUMinHeapAddIIIi(iL,iLev,iw,lenh);
               }
               else if (col > ii)
               {
                  iL[lenu] = col;
                  iLev[lenu] = lev;
                  iw[col] = lenu++;
               }
            }
            else
            {
               iLev[icol] = hypre_min(lev, iLev[icol]);
            }
         }/* end of loop j for level update */
      }/* end of while loop for iith row */

      /* now update everything, indices, levels and so */
      L_diag_i[ii+1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = capacity_L * EXPAND_FACT + 1;
            temp_L_diag_j = hypre_TReAlloc_v2(temp_L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL+j] = iL[ii-j-1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii+1] = U_diag_i[ii] + k;
      if (k > 0)
      {
         /* check if memory is enough */
         while (ctrU + k > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = capacity_U * EXPAND_FACT + 1;
            temp_U_diag_j = hypre_TReAlloc_v2(temp_U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
            u_levels = hypre_TReAlloc_v2(u_levels, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
         }
         //hypre_TMemcpy(temp_U_diag_j+ctrU,iL+ii,HYPRE_Int,k,HYPRE_MEMORY_DEVICE,HYPRE_MEMORY_HOST);
         hypre_TMemcpy(temp_U_diag_j+ctrU, iL+ii, HYPRE_Int, k,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(u_levels+ctrU, iLev+ii, HYPRE_Int, k,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         ctrU += k;
      }

      /* reset iw */
      for (j = ii; j < lenu; j++)
      {
         iw[iL[j]] = -1;
      }
   } /* end of main loop ii from nLU to n */

   /* external part matrix */
   for (ii = n ; ii < total_rows ; ii ++)
   {
      i = ii - n;
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = ii;
      lena = E_i[i+1];
      /* put those already inside original pattern, and set their level to 0 */
      for (j = E_i[i]; j < lena; j++)
      {
         /* get the neworder of that col */
         col = E_j[j];
         if (col < ii)
         {
            /*
             * this is an entry in L
             * we maintain a heap structure for L part
             */
            iL[lenh] = col;
            iLev[lenh] = 0;
            iw[col] = lenh++;
            /*now miantian a heap structure*/
            hypre_ILUMinHeapAddIIIi(iL,iLev,iw,lenh);
         }
         else if (col > ii)
         {
            /* this is an entry in U */
            iL[lenu] = col;
            iLev[lenu] = 0;
            iw[col] = lenu++;
         }
      }/* end of j loop for adding pattern in original matrix */

      /*
       * search lower part of current row and update pattern based on level
       */
      while (lenh > 0)
      {
         /*
          * k is now the new col index after permutation
          * the first element of the heap is the smallest
          */
         k = iL[0];
         ilev = iLev[0];
         /*
          * we now need to maintain the heap structure
          */
         hypre_ILUMinHeapRemoveIIIi(iL,iLev,iw,lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k]=-1;
         hypre_swap2i(iL,iLev,ii-lenl,lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k+1];
         for (j = U_diag_i[k]; j < ku; j++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if (lev > lfil)
            {
               continue;
            }
            if (icol < 0)
            {
               /* not yet in */
               if (col < ii)
               {
                  /*
                   * if we add to the left L, we need to maintian the
                   *    heap structure
                   */
                  iL[lenh] = col;
                  iLev[lenh] = lev;
                  iw[col] = lenh++;
                  /*swap it with the element right after the heap*/

                  /* maintain the heap */
                  hypre_ILUMinHeapAddIIIi(iL,iLev,iw,lenh);
               }
               else if (col > ii)
               {
                  iL[lenu] = col;
                  iLev[lenu] = lev;
                  iw[col] = lenu++;
               }
            }
            else
            {
               iLev[icol] = hypre_min(lev, iLev[icol]);
            }
         }/* end of loop j for level update */
      }/* end of while loop for iith row */

      /* now update everything, indices, levels and so */
      L_diag_i[ii+1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = capacity_L * EXPAND_FACT + 1;
            temp_L_diag_j = hypre_TReAlloc_v2(temp_L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL+j] = iL[ii-j-1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii+1] = U_diag_i[ii] + k;
      if (k > 0)
      {
         /* check if memory is enough */
         while (ctrU + k > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = capacity_U * EXPAND_FACT + 1;
            temp_U_diag_j = hypre_TReAlloc_v2(temp_U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
            u_levels = hypre_TReAlloc_v2(u_levels, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
         }
         //hypre_TMemcpy(temp_U_diag_j+ctrU,iL+ii,HYPRE_Int,k,HYPRE_MEMORY_DEVICE,HYPRE_MEMORY_HOST);
         hypre_TMemcpy(temp_U_diag_j+ctrU, iL+ii, HYPRE_Int, k,
                       HYPRE_MEMORY_HOST,HYPRE_MEMORY_HOST);
         hypre_TMemcpy(u_levels+ctrU, iLev+ii, HYPRE_Int, k,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         ctrU += k;
      }

      /* reset iw */
      for (j = ii; j < lenu; j++)
      {
         iw[iL[j]] = -1;
      }

   }/* end of main loop ii from n to total_rows */

   /*
    * 3: Finishing up and free memory
    */
   hypre_TFree(u_levels,HYPRE_MEMORY_HOST);

   *L_diag_j = temp_L_diag_j;
   *U_diag_j = temp_U_diag_j;

   return hypre_error_flag;
}

/* ILU(k) for RAS
 * A: input matrix
 * lfil: level of fill-in, the k in ILU(k)
 * perm: permutation array indicating ordering of factorization. Perm could come from a
 * CF_marker: array or a reordering routine.
 * nLU: size of computed LDU factorization.
 * Lptr, Dptr, Uptr: L, D, U factors.
 */
HYPRE_Int
hypre_ILUSetupILUKRAS(hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Int *perm, HYPRE_Int nLU,
      hypre_ParCSRMatrix **Lptr, HYPRE_Real** Dptr, hypre_ParCSRMatrix **Uptr)
{
   /*
    * 1: Setup and create buffers
    * matL/U: the ParCSR matrix for L and U
    * L/U_diag: the diagonal csr matrix of matL/U
    * A_diag_*: tempory pointer for the diagonal matrix of A and its '*' slot
    * ii = outer loop from 0 to nLU - 1
    * i = the real col number in diag inside the outer loop
    * iw =  working array store the reverse of active col number
    * iL = working array store the active col number
    */

   /* call ILU0 if lfil is 0 */
   if (lfil == 0)
   {
      return hypre_ILUSetupILU0RAS(A,perm,nLU,Lptr,Dptr,Uptr);
   }
   HYPRE_Int               i, ii, j, k, k1, k2, kl, ku, jpiv, col, icol;
   HYPRE_Int               *iw;
   MPI_Comm                comm           = hypre_ParCSRMatrixComm(A);
   HYPRE_Int               num_procs;

   /* data objects for A */
   hypre_CSRMatrix         *A_diag        = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix         *A_offd        = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real              *A_diag_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int               *A_diag_i      = hypre_CSRMatrixI(A_diag);
   HYPRE_Int               *A_diag_j      = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real              *A_offd_data   = hypre_CSRMatrixData(A_offd);
   HYPRE_Int               *A_offd_i      = hypre_CSRMatrixI(A_offd);
   HYPRE_Int               *A_offd_j      = hypre_CSRMatrixJ(A_offd);

   /* data objects for L, D, U */
   hypre_ParCSRMatrix      *matL;
   hypre_ParCSRMatrix      *matU;
   hypre_CSRMatrix         *L_diag;
   hypre_CSRMatrix         *U_diag;
   HYPRE_Real              *D_data;
   HYPRE_Real              *L_diag_data   = NULL;
   HYPRE_Int               *L_diag_i;
   HYPRE_Int               *L_diag_j      = NULL;
   HYPRE_Real              *U_diag_data   = NULL;
   HYPRE_Int               *U_diag_i;
   HYPRE_Int               *U_diag_j      = NULL;

   /* size of problem and external matrix */
   HYPRE_Int               n              = hypre_CSRMatrixNumRows(A_diag);
   //   HYPRE_Int               m              = n - nLU;
   HYPRE_Int               ext            = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int               total_rows     = n + ext;
   HYPRE_BigInt            global_num_rows;
   HYPRE_BigInt            col_starts[2];
   HYPRE_Real              local_nnz, total_nnz;

   /* data objects for E, external matrix */
   HYPRE_Int               *E_i;
   HYPRE_Int               *E_j;
   HYPRE_Real              *E_data;

   /* communication */
   hypre_ParCSRCommPkg     *comm_pkg;
   hypre_MPI_Comm_size(comm, &num_procs);
   //   hypre_ParCSRCommHandle  *comm_handle;
   //   HYPRE_Int               *send_buf      = NULL;

   /* reverse permutation array */
   HYPRE_Int               *rperm;
   /* temp array for old permutation */
   HYPRE_Int               *perm_old;

   /* start setup */
   /* check input and get problem size */
   n =  hypre_CSRMatrixNumRows(A_diag);
   if (nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU out of range.\n");
   }

   /* Init I array anyway. S's might be freed later */
   D_data   = hypre_CTAlloc(HYPRE_Real, total_rows, HYPRE_MEMORY_DEVICE);
   L_diag_i = hypre_CTAlloc(HYPRE_Int, (total_rows+1), HYPRE_MEMORY_DEVICE);
   U_diag_i = hypre_CTAlloc(HYPRE_Int, (total_rows+1), HYPRE_MEMORY_DEVICE);

   /* set Comm_Pkg if not yet built */
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /*
    * 2: Symbolic factorization
    * setup iw and rperm first
    */
   /* allocate work arrays */
   iw          = hypre_CTAlloc(HYPRE_Int, 5*total_rows, HYPRE_MEMORY_HOST);
   rperm       = iw + 3*total_rows;
   perm_old    = perm;
   perm        = iw + 4*total_rows;
   L_diag_i[0] = U_diag_i[0] = 0;
   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
    */
   for (i = 0; i < n; i++)
   {
      perm[i] = perm_old[i];
   }
   for (i = n; i < total_rows; i++)
   {
      perm[i] = i;
   }
   for (i = 0; i < total_rows; i++)
   {
      rperm[perm[i]] = i;
   }

   /* get external rows */
   hypre_ILUBuildRASExternalMatrix(A,rperm,&E_i,&E_j,&E_data);
   /* do symbolic factorization */
   hypre_ILUSetupILUKRASSymbolic(n, A_diag_i, A_diag_j, A_offd_i, A_offd_j, E_i, E_j, ext, lfil, perm, rperm, iw,
         nLU, L_diag_i, U_diag_i, &L_diag_j, &U_diag_j);

   /*
    * after this, we have our I,J for L, U and S ready, and L sorted
    * iw are still -1 after symbolic factorization
    * now setup helper array here
    */
   if (L_diag_i[total_rows])
   {
      L_diag_data = hypre_CTAlloc(HYPRE_Real, L_diag_i[total_rows], HYPRE_MEMORY_DEVICE);
   }
   if (U_diag_i[total_rows])
   {
      U_diag_data = hypre_CTAlloc(HYPRE_Real, U_diag_i[total_rows], HYPRE_MEMORY_DEVICE);
   }

   /*
    * 3: Begin real factorization
    * we already have L and U structure ready, so no extra working array needed
    */
   /* first loop for upper part */
   for (ii = 0; ii < nLU; ii++)
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii+1];
      ku = U_diag_i[ii+1];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      /* set up working arrays */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      D_data[ii] = 0.0;
      iw[ii] = ii;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = j;
      }
      /* copy data from A into L, D and U */
      for (j = k1; j < k2; j++)
      {
         /* compute everything in new index */
         col = rperm[A_diag_j[j]];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if (col < ii)
         {
            L_diag_data[icol] = A_diag_data[j];
         }
         else if (col == ii)
         {
            D_data[ii] = A_diag_data[j];
         }
         else
         {
            U_diag_data[icol] = A_diag_data[j];
         }
      }
      /* elimination */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv+1];

         for (k = U_diag_i[jpiv]; k < ku; k++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if (icol < 0)
            {
               /* not in partern */
               continue;
            }
            if (col < ii)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
            else if (col == ii)
            {
               /* diag part */
               D_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
            else
            {
               /* U part */
               U_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
         }
      }
      /* reset working array */
      ku = U_diag_i[ii+1];
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      iw[ii] = -1;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = -1;
      }

      /* diagonal part (we store the inverse) */
      if (fabs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1e-06;
      }
      D_data[ii] = 1./ D_data[ii];

   }/* end of loop for upper part */

   /* first loop for upper part */
   for (ii = nLU; ii < n; ii++)
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii+1];
      ku = U_diag_i[ii+1];
      /* set up working arrays */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      D_data[ii] = 0.0;
      iw[ii] = ii;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = j;
      }
      /* copy data from A into L, D and U */
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      for (j = k1; j < k2; j++)
      {
         /* compute everything in new index */
         col = rperm[A_diag_j[j]];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if (col < ii)
         {
            L_diag_data[icol] = A_diag_data[j];
         }
         else if (col == ii)
         {
            D_data[ii] = A_diag_data[j];
         }
         else
         {
            U_diag_data[icol] = A_diag_data[j];
         }
      }
      /* copy data from A_offd into L, D and U */
      k1 = A_offd_i[i];
      k2 = A_offd_i[i+1];
      for (j = k1; j < k2; j++)
      {
         /* compute everything in new index */
         col = A_offd_j[j] + n;
         icol = iw[col];
         U_diag_data[icol] = A_offd_data[j];
      }
      /* elimination */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv+1];

         for (k = U_diag_i[jpiv]; k < ku; k++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if (icol < 0)
            {
               /* not in partern */
               continue;
            }
            if (col < ii)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
            else if (col == ii)
            {
               /* diag part */
               D_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
            else
            {
               /* U part */
               U_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
         }
      }
      /* reset working array */
      ku = U_diag_i[ii+1];
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      iw[ii] = -1;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = -1;
      }

      /* diagonal part (we store the inverse) */
      if (fabs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1e-06;
      }
      D_data[ii] = 1./ D_data[ii];

   }/* end of loop for lower part */

   /* last loop through external */
   for (ii = n; ii < total_rows; ii++)
   {
      // get row i
      i = ii - n;
      kl = L_diag_i[ii+1];
      ku = U_diag_i[ii+1];
      k1 = E_i[i];
      k2 = E_i[i+1];
      /* set up working arrays */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      D_data[ii] = 0.0;
      iw[ii] = ii;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = j;
      }
      /* copy data from E into L, D and U */
      for (j = k1; j < k2; j++)
      {
         /* compute everything in new index */
         col = E_j[j];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if (col < ii)
         {
            L_diag_data[icol] = E_data[j];
         }
         else if (col == ii)
         {
            D_data[ii] = E_data[j];
         }
         else
         {
            U_diag_data[icol] = E_data[j];
         }
      }
      /* elimination */
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv+1];

         for (k = U_diag_i[jpiv]; k < ku; k++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if (icol < 0)
            {
               /* not in partern */
               continue;
            }
            if (col < ii)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
            else if (col == ii)
            {
               /* diag part */
               D_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
            else
            {
               /* U part */
               U_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
         }
      }
      /* reset working array */
      ku = U_diag_i[ii+1];
      for (j = L_diag_i[ii]; j < kl; j++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      iw[ii] = -1;
      for (j = U_diag_i[ii]; j < ku; j++)
      {
         col = U_diag_j[j];
         iw[col] = -1;
      }

      /* diagonal part (we store the inverse) */
      if (fabs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1e-06;
      }
      D_data[ii] = 1./ D_data[ii];

   }/* end of loop for external loop */

   /*
    * 4: Finishing up and free
    */
   HYPRE_BigInt big_total_rows = (HYPRE_BigInt)total_rows;
   hypre_MPI_Allreduce( &big_total_rows, &global_num_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
   /* need to get new column start */
   {
      HYPRE_BigInt global_start;
      hypre_MPI_Scan( &big_total_rows, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
      col_starts[0] = global_start - total_rows;
      col_starts[1] = global_start;
   }
   /* Assemble LDU matrices */
   matL = hypre_ParCSRMatrixCreate( comm,
         global_num_rows,
         global_num_rows,
         col_starts,
         col_starts,
         0 /* num_cols_offd */,
         L_diag_i[total_rows],
         0 /* num_nonzeros_offd */);

   L_diag = hypre_ParCSRMatrixDiag(matL);
   hypre_CSRMatrixI(L_diag) = L_diag_i;
   if (L_diag_i[total_rows]>0)
   {
      hypre_CSRMatrixData(L_diag) = L_diag_data;
      hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we allocated some initial length, so free them */
      hypre_TFree(L_diag_j, HYPRE_MEMORY_DEVICE);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (L_diag_i[total_rows]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

   matU = hypre_ParCSRMatrixCreate( comm,
         global_num_rows,
         global_num_rows,
         col_starts,
         col_starts,
         0,
         U_diag_i[total_rows],
         0 );

   U_diag = hypre_ParCSRMatrixDiag(matU);
   hypre_CSRMatrixI(U_diag) = U_diag_i;
   if (U_diag_i[n]>0)
   {
      hypre_CSRMatrixData(U_diag) = U_diag_data;
      hypre_CSRMatrixJ(U_diag) = U_diag_j;
   }
   else
   {
      /* we allocated some initial length, so free them */
      hypre_TFree(U_diag_j, HYPRE_MEMORY_DEVICE);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (U_diag_i[total_rows]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

   /* free */
   hypre_TFree(iw,HYPRE_MEMORY_HOST);

   /* free external data */
   if (E_i)
   {
      hypre_TFree(E_i, HYPRE_MEMORY_HOST);
   }
   if (E_j)
   {
      hypre_TFree(E_j, HYPRE_MEMORY_HOST);
      hypre_TFree(E_data, HYPRE_MEMORY_HOST);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;

   return hypre_error_flag;
}

/* ILUT for RAS
 * A: input matrix
 * lfil: level of fill-in, the k in ILU(k)
 * tol: droptol array in ILUT
 *    tol[0]: matrix B
 *    tol[1]: matrix E and F
 *    tol[2]: matrix S
 * perm: permutation array indicating ordering of factorization. Perm could come from a
 * CF_marker: array or a reordering routine.
 * nLU: size of computed LDU factorization. If nLU < n, Schur compelemnt will be formed
 * Lptr, Dptr, Uptr: L, D, U factors.
 * Sptr: Schur complement
 *
 * Keep the largest lfil entries that is greater than some tol relative
 *    to the input tol and the norm of that row in both L and U
 */
HYPRE_Int
hypre_ILUSetupILUTRAS(hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Real *tol,
      HYPRE_Int *perm, HYPRE_Int nLU, hypre_ParCSRMatrix **Lptr,
      HYPRE_Real** Dptr, hypre_ParCSRMatrix **Uptr)
{
   /*
    * 1: Setup and create buffers
    * matL/U: the ParCSR matrix for L and U
    * L/U_diag: the diagonal csr matrix of matL/U
    * A_diag_*: tempory pointer for the diagonal matrix of A and its '*' slot
    * ii = outer loop from 0 to nLU - 1
    * i = the real col number in diag inside the outer loop
    * iw =  working array store the reverse of active col number
    * iL = working array store the active col number
    */
   HYPRE_Real               local_nnz, total_nnz;
   HYPRE_Int                i, ii, j, k1, k2, k12, k22, kl, ku, col, icol, lenl, lenu, lenhu, lenhlr, lenhll, jpos, jrow;
   HYPRE_Real               inorm, itolb, itolef, dpiv, lxu;
   HYPRE_Int                *iw,*iL;
   HYPRE_Real               *w;

   /* memory management */
   HYPRE_Int                ctrL;
   HYPRE_Int                ctrU;
   HYPRE_Int                initial_alloc = 0;
   HYPRE_Int                capacity_L;
   HYPRE_Int                capacity_U;
   HYPRE_Int                nnz_A;

   /* communication stuffs for S */
   MPI_Comm                 comm          = hypre_ParCSRMatrixComm(A);
   HYPRE_Int                num_procs;
   hypre_ParCSRCommPkg      *comm_pkg;
   //   hypre_ParCSRCommHandle   *comm_handle;
   HYPRE_BigInt             col_starts[2];
   //   HYPRE_Int                num_sends;
   //   HYPRE_Int                begin, end;

   /* data objects for A */
   hypre_CSRMatrix          *A_diag       = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix          *A_offd       = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real               *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int                *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int                *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int                *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Int                *A_offd_j     = hypre_CSRMatrixJ(A_offd);
   HYPRE_Real               *A_offd_data  = hypre_CSRMatrixData(A_offd);

   /* data objects for L, D, U */
   hypre_ParCSRMatrix       *matL;
   hypre_ParCSRMatrix       *matU;
   hypre_CSRMatrix          *L_diag;
   hypre_CSRMatrix          *U_diag;
   HYPRE_Real               *D_data;
   HYPRE_Real               *L_diag_data  = NULL;
   HYPRE_Int                *L_diag_i;
   HYPRE_Int                *L_diag_j     = NULL;
   HYPRE_Real               *U_diag_data  = NULL;
   HYPRE_Int                *U_diag_i;
   HYPRE_Int                *U_diag_j     = NULL;

   /* size of problem and external matrix */
   HYPRE_Int                n             = hypre_CSRMatrixNumRows(A_diag);
   //   HYPRE_Int                m             = n - nLU;
   HYPRE_Int                ext           = hypre_CSRMatrixNumCols(A_offd);
   HYPRE_Int                total_rows    = n + ext;
   HYPRE_BigInt              global_num_rows;

   /* data objects for E, external matrix */
   HYPRE_Int                *E_i;
   HYPRE_Int                *E_j;
   HYPRE_Real               *E_data;

   /* reverse permutation */
   HYPRE_Int                *rperm;
   /* old permutation */
   HYPRE_Int                *perm_old;

   /* start setup
    * check input first
    */
   n = hypre_CSRMatrixNumRows(A_diag);
   if (nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU out of range.\n");
   }

   /* start set up
    * setup communication stuffs first
    */
   hypre_MPI_Comm_size(comm, &num_procs);
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   /* create if not yet built */
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* setup initial memory */
   nnz_A = A_diag_i[nLU];
   if (n > 0)
   {
      initial_alloc = nLU + ceil(nnz_A / 2.0);
   }
   capacity_L = initial_alloc;
   capacity_U = initial_alloc;

   D_data = hypre_CTAlloc(HYPRE_Real, total_rows, HYPRE_MEMORY_DEVICE);
   L_diag_i = hypre_CTAlloc(HYPRE_Int, (total_rows+1), HYPRE_MEMORY_DEVICE);
   U_diag_i = hypre_CTAlloc(HYPRE_Int, (total_rows+1), HYPRE_MEMORY_DEVICE);

   L_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
   U_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
   L_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
   U_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_U, HYPRE_MEMORY_DEVICE);

   ctrL = ctrU = 0;

   /* setting up working array */
   iw = hypre_CTAlloc(HYPRE_Int,4*total_rows,HYPRE_MEMORY_HOST);
   iL = iw + total_rows;
   w = hypre_CTAlloc(HYPRE_Real,total_rows,HYPRE_MEMORY_HOST);
   for (i = 0; i < total_rows; i++)
   {
      iw[i] = -1;
   }
   L_diag_i[0] = U_diag_i[0] = 0;
   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
    * rperm[old] -> new
    * perm[new]  -> old
    */
   rperm = iw + 2*total_rows;
   perm_old = perm;
   perm = iw + 3*total_rows;
   for (i = 0; i < n; i++)
   {
      perm[i] = perm_old[i];
   }
   for (i = n; i < total_rows; i++)
   {
      perm[i] = i;
   }
   for (i = 0; i < total_rows; i++)
   {
      rperm[perm[i]] = i;
   }
   /* get external matrix */
   hypre_ILUBuildRASExternalMatrix(A,rperm,&E_i,&E_j,&E_data);

   /*
    * 2: Main loop of elimination
    * maintain two heaps
    * |----->*********<-----|-----*********|
    * |col heap***value heap|value in U****|
    */

   /* main outer loop for upper part */
   for (ii = 0 ; ii < nLU; ii++)
   {
      /* get real row with perm */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      kl = ii-1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += fabs(A_diag_data[j]);
      }
      if (inorm == .0)
      {
         hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: ILUT with zero row.\n");
      }
      inorm /= (HYPRE_Real)(k2-k1);
      /* set the scaled tol for that row */
      itolb = tol[0] * inorm;
      itolef = tol[1] * inorm;

      /* reset displacement */
      lenhll = lenhlr = lenu = 0;
      w[ii] = 0.0;
      iw[ii] = ii;
      /* copy in data from A */
      for (j = k1; j < k2; j++)
      {
         /* get now col number */
         col = rperm[A_diag_j[j]];
         if (col < ii)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = A_diag_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
         }
         else if (col == ii)
         {
            w[ii] = A_diag_data[j];
         }
         else
         {
            lenu++;
            jpos = lenu + ii;
            iL[jpos] = col;
            w[jpos] = A_diag_data[j];
            iw[col] = jpos;
         }
      }

      /*
       * main elimination
       * need to maintain 2 heaps for L, one heap for col and one heaps for value
       * maintian an array for U, and do qsplit with quick sort after that
       * while the heap of col is greater than zero
       */
      while (lenhll > 0)
      {

         /* get the next row from top of the heap */
         jrow = iL[0];
         dpiv = w[0] * D_data[jrow];
         w[0] = dpiv;
         /* now remove it from the top of the heap */
         hypre_ILUMinHeapRemoveIRIi(iL,w,iw,lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;
         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         hypre_swap2(iL,w,lenhll,kl-lenhlr);
         lenhlr++;
         hypre_ILUMaxrHeapAddRabsI(w+kl,iL+kl,lenhlr);
         /* loop for elimination */
         ku = U_diag_i[jrow+1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv*U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col < nLU && fabs(lxu) < itolb) || (col >= nLU && fabs(lxu) < itolef)))
            {
               continue;
            }
            if (icol == -1)
            {
               if (col < ii)
               {
                  /* L part
                   * not already in L part
                   * put it to the end of heap
                   * might overwrite some small entries, no issue
                   */
                  iL[lenhll] = col;
                  w[lenhll] = lxu;
                  iw[col] = lenhll++;
                  /* add to heap, by col number */
                  hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
               }
               else if (col == ii)
               {
                  w[ii] += lxu;
               }
               else
               {
                  /*
                   * not already in U part
                   * put is to the end of heap
                   */
                  lenu++;
                  jpos = lenu + ii;
                  iL[jpos] = col;
                  w[jpos] = lxu;
                  iw[col] = jpos;
               }
            }
            else
            {
               w[icol] += lxu;
            }
         }
      }/* while loop for the elimination of current row */

      if (fabs(w[ii]) < MAT_TOL)
      {
         w[ii]=1e-06;
      }
      D_data[ii] = 1./w[ii];
      iw[ii] = -1;

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii+1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = capacity_L * EXPAND_FACT + 1;
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
         }
         ctrL += lenl;
         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            hypre_ILUMaxrHeapRemoveRabsI(w+kl,iL+kl,lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only U part
       */
      ku = lenu+ii;
      for (j = ii + 1; j <= ku; j++)
      {
         iw[iL[j]] = -1;
      }

      if (lenu < lfil)
      {
         /* we simply keep all of the data, no need to sort */
         lenhu = lenu;
      }
      else
      {
         /* need to sort the first small(hopefully) part of it */
         lenhu = lfil;
         /* quick split, only sort the first small part of the array */
         hypre_ILUMaxQSplitRabsI(w,iL,ii+1,ii+lenhu,ii+lenu);
      }

      U_diag_i[ii+1] = U_diag_i[ii] + lenhu;
      if (lenhu > 0)
      {
         /* test if memory is enough */
         while (ctrU + lenhu > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = capacity_U * EXPAND_FACT + 1;
            U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
            U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_U, HYPRE_MEMORY_DEVICE);
         }
         ctrU += lenhu;
         /* copy large data in */
         for (j = U_diag_i[ii]; j < ctrU; j++)
         {
            jpos = ii+1+j-U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
   }/* end of ii loop from 0 to nLU-1 */

   /* second outer loop for lower part */
   for (ii = nLU; ii < n; ii++)
   {
      /* get real row with perm */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      k12 = A_offd_i[i];
      k22 = A_offd_i[i+1];
      kl = ii-1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += fabs(A_diag_data[j]);
      }
      for (j = k12; j < k22; j++)
      {
         inorm += fabs(A_offd_data[j]);
      }
      if (inorm == .0)
      {
         hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: ILUT with zero row.\n");
      }
      inorm /= (HYPRE_Real)(k2+k22-k1-k12);
      /* set the scaled tol for that row */
      itolb = tol[0] * inorm;
      itolef = tol[1] * inorm;

      /* reset displacement */
      lenhll = lenhlr = lenu = 0;
      w[ii] = 0.0;
      iw[ii] = ii;
      /* copy in data from A_diag */
      for (j = k1; j < k2; j++)
      {
         /* get now col number */
         col = rperm[A_diag_j[j]];
         if (col < ii)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = A_diag_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
         }
         else if (col == ii)
         {
            w[ii] = A_diag_data[j];
         }
         else
         {
            lenu++;
            jpos = lenu + ii;
            iL[jpos] = col;
            w[jpos] = A_diag_data[j];
            iw[col] = jpos;
         }
      }
      /* copy in data from A_offd */
      for (j = k12; j < k22; j++)
      {
         /* get now col number */
         col = A_offd_j[j] + n;
         /* all should greater than ii in lower part */
         lenu++;
         jpos = lenu + ii;
         iL[jpos] = col;
         w[jpos] = A_offd_data[j];
         iw[col] = jpos;
      }

      /*
       * main elimination
       * need to maintain 2 heaps for L, one heap for col and one heaps for value
       * maintian an array for U, and do qsplit with quick sort after that
       * while the heap of col is greater than zero
       */
      while (lenhll > 0)
      {

         /* get the next row from top of the heap */
         jrow = iL[0];
         dpiv = w[0] * D_data[jrow];
         w[0] = dpiv;
         /* now remove it from the top of the heap */
         hypre_ILUMinHeapRemoveIRIi(iL,w,iw,lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;
         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         hypre_swap2(iL,w,lenhll,kl-lenhlr);
         lenhlr++;
         hypre_ILUMaxrHeapAddRabsI(w+kl,iL+kl,lenhlr);
         /* loop for elimination */
         ku = U_diag_i[jrow+1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv*U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col < nLU && fabs(lxu) < itolb) || (col >= nLU && fabs(lxu) < itolef)))
            {
               continue;
            }
            if (icol == -1)
            {
               if (col < ii)
               {
                  /* L part
                   * not already in L part
                   * put it to the end of heap
                   * might overwrite some small entries, no issue
                   */
                  iL[lenhll] = col;
                  w[lenhll] = lxu;
                  iw[col] = lenhll++;
                  /* add to heap, by col number */
                  hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
               }
               else if (col == ii)
               {
                  w[ii] += lxu;
               }
               else
               {
                  /*
                   * not already in U part
                   * put is to the end of heap
                   */
                  lenu++;
                  jpos = lenu + ii;
                  iL[jpos] = col;
                  w[jpos] = lxu;
                  iw[col] = jpos;
               }
            }
            else
            {
               w[icol] += lxu;
            }
         }
      }/* while loop for the elimination of current row */

      if (fabs(w[ii]) < MAT_TOL)
      {
         w[ii]=1e-06;
      }
      D_data[ii] = 1./w[ii];
      iw[ii] = -1;

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii+1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = capacity_L * EXPAND_FACT + 1;
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
         }
         ctrL += lenl;
         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            hypre_ILUMaxrHeapRemoveRabsI(w+kl,iL+kl,lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only U part
       */
      ku = lenu+ii;
      for (j = ii + 1; j <= ku; j++)
      {
         iw[iL[j]] = -1;
      }

      if (lenu < lfil)
      {
         /* we simply keep all of the data, no need to sort */
         lenhu = lenu;
      }
      else
      {
         /* need to sort the first small(hopefully) part of it */
         lenhu = lfil;
         /* quick split, only sort the first small part of the array */
         hypre_ILUMaxQSplitRabsI(w,iL,ii+1,ii+lenhu,ii+lenu);
      }

      U_diag_i[ii+1] = U_diag_i[ii] + lenhu;
      if (lenhu > 0)
      {
         /* test if memory is enough */
         while (ctrU + lenhu > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = capacity_U * EXPAND_FACT + 1;
            U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
            U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_U, HYPRE_MEMORY_DEVICE);
         }
         ctrU += lenhu;
         /* copy large data in */
         for (j = U_diag_i[ii]; j < ctrU; j++)
         {
            jpos = ii+1+j-U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
   }/* end of ii loop from nLU to n */


   /* main outer loop for upper part */
   for (ii = n; ii < total_rows; ii++)
   {
      /* get real row with perm */
      i = ii-n;
      k1 = E_i[i];
      k2 = E_i[i+1];
      kl = ii-1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += fabs(E_data[j]);
      }
      if (inorm == .0)
      {
         hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: ILUT with zero row.\n");
      }
      inorm /= (HYPRE_Real)(k2-k1);
      /* set the scaled tol for that row */
      itolb = tol[0] * inorm;
      itolef = tol[1] * inorm;

      /* reset displacement */
      lenhll = lenhlr = lenu = 0;
      w[ii] = 0.0;
      iw[ii] = ii;
      /* copy in data from A */
      for (j = k1; j < k2; j++)
      {
         /* get now col number */
         col = rperm[E_j[j]];
         if (col < ii)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = E_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
         }
         else if (col == ii)
         {
            w[ii] = E_data[j];
         }
         else
         {
            lenu++;
            jpos = lenu + ii;
            iL[jpos] = col;
            w[jpos] = E_data[j];
            iw[col] = jpos;
         }
      }

      /*
       * main elimination
       * need to maintain 2 heaps for L, one heap for col and one heaps for value
       * maintian an array for U, and do qsplit with quick sort after that
       * while the heap of col is greater than zero
       */
      while (lenhll > 0)
      {

         /* get the next row from top of the heap */
         jrow = iL[0];
         dpiv = w[0] * D_data[jrow];
         w[0] = dpiv;
         /* now remove it from the top of the heap */
         hypre_ILUMinHeapRemoveIRIi(iL,w,iw,lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;
         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         hypre_swap2(iL,w,lenhll,kl-lenhlr);
         lenhlr++;
         hypre_ILUMaxrHeapAddRabsI(w+kl,iL+kl,lenhlr);
         /* loop for elimination */
         ku = U_diag_i[jrow+1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv*U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col < nLU && fabs(lxu) < itolb) || (col >= nLU && fabs(lxu) < itolef)))
            {
               continue;
            }
            if (icol == -1)
            {
               if (col < ii)
               {
                  /* L part
                   * not already in L part
                   * put it to the end of heap
                   * might overwrite some small entries, no issue
                   */
                  iL[lenhll] = col;
                  w[lenhll] = lxu;
                  iw[col] = lenhll++;
                  /* add to heap, by col number */
                  hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
               }
               else if (col == ii)
               {
                  w[ii] += lxu;
               }
               else
               {
                  /*
                   * not already in U part
                   * put is to the end of heap
                   */
                  lenu++;
                  jpos = lenu + ii;
                  iL[jpos] = col;
                  w[jpos] = lxu;
                  iw[col] = jpos;
               }
            }
            else
            {
               w[icol] += lxu;
            }
         }
      }/* while loop for the elimination of current row */

      if (fabs(w[ii]) < MAT_TOL)
      {
         w[ii]=1e-06;
      }
      D_data[ii] = 1./w[ii];
      iw[ii] = -1;

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii+1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = capacity_L * EXPAND_FACT + 1;
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L, HYPRE_MEMORY_DEVICE);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_L, HYPRE_MEMORY_DEVICE);
         }
         ctrL += lenl;
         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            hypre_ILUMaxrHeapRemoveRabsI(w+kl,iL+kl,lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only U part
       */
      ku = lenu+ii;
      for (j = ii + 1; j <= ku; j++)
      {
         iw[iL[j]] = -1;
      }

      if (lenu < lfil)
      {
         /* we simply keep all of the data, no need to sort */
         lenhu = lenu;
      }
      else
      {
         /* need to sort the first small(hopefully) part of it */
         lenhu = lfil;
         /* quick split, only sort the first small part of the array */
         hypre_ILUMaxQSplitRabsI(w,iL,ii+1,ii+lenhu,ii+lenu);
      }

      U_diag_i[ii+1] = U_diag_i[ii] + lenhu;
      if (lenhu > 0)
      {
         /* test if memory is enough */
         while (ctrU + lenhu > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = capacity_U * EXPAND_FACT + 1;
            U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_DEVICE);
            U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp, HYPRE_Real, capacity_U, HYPRE_MEMORY_DEVICE);
         }
         ctrU += lenhu;
         /* copy large data in */
         for (j = U_diag_i[ii]; j < ctrU; j++)
         {
            jpos = ii+1+j-U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
   }/* end of ii loop from nLU to total_rows */

   /*
    * 3: Finishing up and free
    */
   HYPRE_BigInt big_total_rows = (HYPRE_BigInt)total_rows;
   hypre_MPI_Allreduce( &big_total_rows, &global_num_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
   /* need to get new column start */
   {
      HYPRE_BigInt global_start;
      hypre_MPI_Scan( &big_total_rows, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
      col_starts[0] = global_start - total_rows;
      col_starts[1] = global_start;
   }

   /* create parcsr matrix */
   matL = hypre_ParCSRMatrixCreate( comm,
         global_num_rows,
         global_num_rows,
         col_starts,
         col_starts,
         0,
         L_diag_i[total_rows],
         0 );

   L_diag = hypre_ParCSRMatrixDiag(matL);
   hypre_CSRMatrixI(L_diag) = L_diag_i;
   if (L_diag_i[total_rows] > 0)
   {
      hypre_CSRMatrixData(L_diag) = L_diag_data;
      hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we initialized some anyway, so remove if unused */
      hypre_TFree(L_diag_j,HYPRE_MEMORY_DEVICE);
      hypre_TFree(L_diag_data,HYPRE_MEMORY_DEVICE);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (L_diag_i[total_rows]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matL) = total_nnz;

   matU = hypre_ParCSRMatrixCreate( comm,
         global_num_rows,
         global_num_rows,
         col_starts,
         col_starts,
         0,
         U_diag_i[total_rows],
         0 );

   U_diag = hypre_ParCSRMatrixDiag(matU);
   hypre_CSRMatrixI(U_diag) = U_diag_i;
   if (U_diag_i[total_rows] > 0)
   {
      hypre_CSRMatrixData(U_diag) = U_diag_data;
      hypre_CSRMatrixJ(U_diag) = U_diag_j;
   }
   else
   {
      /* we initialized some anyway, so remove if unused */
      hypre_TFree(U_diag_j,HYPRE_MEMORY_DEVICE);
      hypre_TFree(U_diag_data,HYPRE_MEMORY_DEVICE);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (U_diag_i[total_rows]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

   /* free working array */
   hypre_TFree(iw,HYPRE_MEMORY_HOST);
   hypre_TFree(w,HYPRE_MEMORY_HOST);

   /* free external data */
   if (E_i)
   {
      hypre_TFree(E_i, HYPRE_MEMORY_HOST);
   }
   if (E_j)
   {
      hypre_TFree(E_j, HYPRE_MEMORY_HOST);
      hypre_TFree(E_data, HYPRE_MEMORY_HOST);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;

   return hypre_error_flag;
}
