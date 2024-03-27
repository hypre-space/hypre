/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_ILUSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetup( void               *ilu_vdata,
                hypre_ParCSRMatrix *A,
                hypre_ParVector    *f,
                hypre_ParVector    *u )
{
   MPI_Comm              comm                = hypre_ParCSRMatrixComm(A);
   HYPRE_MemoryLocation  memory_location     = hypre_ParCSRMatrixMemoryLocation(A);
   hypre_ParILUData     *ilu_data            = (hypre_ParILUData*) ilu_vdata;
   hypre_ParILUData     *schur_precond_ilu;
   hypre_ParNSHData     *schur_solver_nsh;

   /* Pointers to ilu data */
   HYPRE_Int             logging             = hypre_ParILUDataLogging(ilu_data);
   HYPRE_Int             print_level         = hypre_ParILUDataPrintLevel(ilu_data);
   HYPRE_Int             ilu_type            = hypre_ParILUDataIluType(ilu_data);
   HYPRE_Int             nLU                 = hypre_ParILUDataNLU(ilu_data);
   HYPRE_Int             nI                  = hypre_ParILUDataNI(ilu_data);
   HYPRE_Int             fill_level          = hypre_ParILUDataLfil(ilu_data);
   HYPRE_Int             max_row_elmts       = hypre_ParILUDataMaxRowNnz(ilu_data);
   HYPRE_Real           *droptol             = hypre_ParILUDataDroptol(ilu_data);
   HYPRE_Int            *CF_marker_array     = hypre_ParILUDataCFMarkerArray(ilu_data);
   HYPRE_Int            *perm                = hypre_ParILUDataPerm(ilu_data);
   HYPRE_Int            *qperm               = hypre_ParILUDataQPerm(ilu_data);
   HYPRE_Real            tol_ddPQ            = hypre_ParILUDataTolDDPQ(ilu_data);

   /* Pointers to device data, note that they are not NULL only when needed */
#if defined(HYPRE_USING_GPU)
   HYPRE_Int             test_opt            = hypre_ParILUDataTestOption(ilu_data);
   hypre_ParCSRMatrix   *Aperm               = hypre_ParILUDataAperm(ilu_data);
   hypre_ParCSRMatrix   *R                   = hypre_ParILUDataR(ilu_data);
   hypre_ParCSRMatrix   *P                   = hypre_ParILUDataP(ilu_data);
   hypre_CSRMatrix      *matALU_d            = hypre_ParILUDataMatAILUDevice(ilu_data);
   hypre_CSRMatrix      *matBLU_d            = hypre_ParILUDataMatBILUDevice(ilu_data);
   hypre_CSRMatrix      *matSLU_d            = hypre_ParILUDataMatSILUDevice(ilu_data);
   hypre_CSRMatrix      *matE_d              = hypre_ParILUDataMatEDevice(ilu_data);
   hypre_CSRMatrix      *matF_d              = hypre_ParILUDataMatFDevice(ilu_data);
   hypre_Vector         *Ftemp_upper         = NULL;
   hypre_Vector         *Utemp_lower         = NULL;
   hypre_Vector         *Adiag_diag          = NULL;
   hypre_Vector         *Sdiag_diag          = NULL;
#endif

   hypre_ParCSRMatrix   *matA                = hypre_ParILUDataMatA(ilu_data);
   hypre_ParCSRMatrix   *matL                = hypre_ParILUDataMatL(ilu_data);
   HYPRE_Real           *matD                = hypre_ParILUDataMatD(ilu_data);
   hypre_ParCSRMatrix   *matU                = hypre_ParILUDataMatU(ilu_data);
   hypre_ParCSRMatrix   *matmL               = hypre_ParILUDataMatLModified(ilu_data);
   HYPRE_Real           *matmD               = hypre_ParILUDataMatDModified(ilu_data);
   hypre_ParCSRMatrix   *matmU               = hypre_ParILUDataMatUModified(ilu_data);
   hypre_ParCSRMatrix   *matS                = hypre_ParILUDataMatS(ilu_data);
   HYPRE_Int             n                   = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   HYPRE_Int             reordering_type     = hypre_ParILUDataReorderingType(ilu_data);
   HYPRE_Real            nnzS;  /* Total nnz in S */
   HYPRE_Real            nnzS_offd_local;
   HYPRE_Real            nnzS_offd;
   HYPRE_Int             size_C /* Total size of coarse grid */;

   hypre_ParVector      *Utemp               = NULL;
   hypre_ParVector      *Ftemp               = NULL;
   hypre_ParVector      *Xtemp               = NULL;
   hypre_ParVector      *Ytemp               = NULL;
   hypre_ParVector      *Ztemp               = NULL;
   HYPRE_Real           *uext                = NULL;
   HYPRE_Real           *fext                = NULL;
   hypre_ParVector      *rhs                 = NULL;
   hypre_ParVector      *x                   = NULL;

   /* TODO (VPM): Change F_array and U_array variable names */
   hypre_ParVector      *F_array             = hypre_ParILUDataF(ilu_data);
   hypre_ParVector      *U_array             = hypre_ParILUDataU(ilu_data);
   hypre_ParVector      *residual            = hypre_ParILUDataResidual(ilu_data);
   HYPRE_Real           *rel_res_norms       = hypre_ParILUDataRelResNorms(ilu_data);

   /* might need for Schur Complement */
   HYPRE_Int            *u_end                = NULL;
   HYPRE_Solver          schur_solver         = NULL;
   HYPRE_Solver          schur_precond        = NULL;
   HYPRE_Solver          schur_precond_gotten = NULL;

   /* Whether or not to use exact (direct) triangular solves */
   HYPRE_Int             tri_solve            = hypre_ParILUDataTriSolve(ilu_data);

   /* help to build external */
   hypre_ParCSRCommPkg  *comm_pkg;
   HYPRE_Int             buffer_size;
   HYPRE_Int             num_sends;
   HYPRE_Int             send_size;
   HYPRE_Int             recv_size;
   HYPRE_Int             num_procs, my_id;

#if defined (HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(hypre_ParCSRMatrixMemoryLocation(A));

   /* TODO (VPM): Placeholder check to avoid -Wunused-variable warning. Remove this! */
   if (exec != HYPRE_EXEC_DEVICE && exec != HYPRE_EXEC_HOST)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Need to run either on host or device!");
      return hypre_error_flag;
   }
#endif

   /* Sanity checks */
#if defined(HYPRE_USING_CUDA) && !defined(HYPRE_USING_CUSPARSE)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ILU CUDA build requires cuSPARSE!");
      return hypre_error_flag;
   }
#elif defined(HYPRE_USING_HIP) && !defined(HYPRE_USING_ROCSPARSE)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ILU HIP build requires rocSPARSE!");
      return hypre_error_flag;
   }
#elif defined(HYPRE_USING_SYCL) && !defined(HYPRE_USING_ONEMKLSPARSE)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "ILU SYCL build requires oneMKLSparse!");
      return hypre_error_flag;
   }
#endif

   /* ----- begin -----*/
   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("hypre_ILUSetup");

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

#if defined(HYPRE_USING_GPU)
   hypre_CSRMatrixDestroy(matALU_d); matALU_d = NULL;
   hypre_CSRMatrixDestroy(matSLU_d); matSLU_d = NULL;
   hypre_CSRMatrixDestroy(matBLU_d); matBLU_d = NULL;
   hypre_CSRMatrixDestroy(matE_d);   matE_d   = NULL;
   hypre_CSRMatrixDestroy(matF_d);   matF_d   = NULL;
   hypre_ParCSRMatrixDestroy(Aperm); Aperm    = NULL;
   hypre_ParCSRMatrixDestroy(R);     R        = NULL;
   hypre_ParCSRMatrixDestroy(P);     P        = NULL;

   hypre_SeqVectorDestroy(hypre_ParILUDataFTempUpper(ilu_data));
   hypre_SeqVectorDestroy(hypre_ParILUDataUTempLower(ilu_data));
   hypre_ParVectorDestroy(hypre_ParILUDataXTemp(ilu_data));
   hypre_ParVectorDestroy(hypre_ParILUDataYTemp(ilu_data));
   hypre_ParVectorDestroy(hypre_ParILUDataZTemp(ilu_data));
   hypre_SeqVectorDestroy(hypre_ParILUDataADiagDiag(ilu_data));
   hypre_SeqVectorDestroy(hypre_ParILUDataSDiagDiag(ilu_data));

   hypre_ParILUDataFTempUpper(ilu_data) = NULL;
   hypre_ParILUDataUTempLower(ilu_data) = NULL;
   hypre_ParILUDataXTemp(ilu_data)      = NULL;
   hypre_ParILUDataYTemp(ilu_data)      = NULL;
   hypre_ParILUDataZTemp(ilu_data)      = NULL;
   hypre_ParILUDataADiagDiag(ilu_data)  = NULL;
   hypre_ParILUDataSDiagDiag(ilu_data)  = NULL;
#endif

   /* Free previously allocated data, if any not destroyed */
   hypre_ParCSRMatrixDestroy(matL);  matL  = NULL;
   hypre_ParCSRMatrixDestroy(matU);  matU  = NULL;
   hypre_ParCSRMatrixDestroy(matmL); matmL = NULL;
   hypre_ParCSRMatrixDestroy(matmU); matmU = NULL;
   hypre_ParCSRMatrixDestroy(matS);  matS  = NULL;

   hypre_TFree(matD, HYPRE_MEMORY_DEVICE);
   hypre_TFree(matmD, HYPRE_MEMORY_DEVICE);
   hypre_TFree(CF_marker_array, HYPRE_MEMORY_HOST);

   /* clear old l1_norm data, if created */
   hypre_TFree(hypre_ParILUDataL1Norms(ilu_data), HYPRE_MEMORY_HOST);

   /* setup temporary storage
    * first check is they've already here
    */
   hypre_ParVectorDestroy(hypre_ParILUDataUTemp(ilu_data));
   hypre_ParVectorDestroy(hypre_ParILUDataFTemp(ilu_data));
   hypre_ParVectorDestroy(hypre_ParILUDataRhs(ilu_data));
   hypre_ParVectorDestroy(hypre_ParILUDataX(ilu_data));
   hypre_ParVectorDestroy(hypre_ParILUDataResidual(ilu_data));
   hypre_TFree(hypre_ParILUDataUExt(ilu_data), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParILUDataFExt(ilu_data), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParILUDataUEnd(ilu_data), HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParILUDataRelResNorms(ilu_data), HYPRE_MEMORY_HOST);

   hypre_ParILUDataUTemp(ilu_data) = NULL;
   hypre_ParILUDataFTemp(ilu_data) = NULL;
   hypre_ParILUDataRhs(ilu_data) = NULL;
   hypre_ParILUDataX(ilu_data) = NULL;
   hypre_ParILUDataResidual(ilu_data) = NULL;

   if (hypre_ParILUDataSchurSolver(ilu_data))
   {
      switch (ilu_type)
      {
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
      HYPRE_ILUDestroy(hypre_ParILUDataSchurPrecond(ilu_data));
      hypre_ParILUDataSchurPrecond(ilu_data) = NULL;
   }

   /* Create work vectors */
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
   matA    = A;
   F_array = f;
   U_array = u;

   /* Create perm array if necessary */
   if (!perm)
   {
      switch (ilu_type)
      {
         case 10: case 11: case 20: case 21: case 30: case 31: case 50:
            /* symmetric */
            hypre_ILUGetInteriorExteriorPerm(matA, memory_location, &perm, &nLU, reordering_type);
            break;

         case 40: case 41:
            /* ddPQ */
            hypre_ILUGetPermddPQ(matA, &perm, &qperm, tol_ddPQ, &nLU, &nI, reordering_type);
            break;

         case 0: case 1:
         default:
            /* RCM or none */
            hypre_ILUGetLocalPerm(matA, &perm, &nLU, reordering_type);
            break;
      }
   }

   /* Factorization */
   switch (ilu_type)
   {
      case 0: /* BJ + hypre_iluk() */
#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_ILUSetupDevice(ilu_data, matA, perm, perm, n, n,
                                 &matBLU_d, &matS, &matE_d, &matF_d);
         }
         else
#endif
         {
            hypre_ILUSetupILUK(matA, fill_level, perm, perm, n, n,
                               &matL, &matD, &matU, &matS, &u_end);
         }
         break;

      case 1: /* BJ + hypre_ilut() */
#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_ILUSetupDevice(ilu_data, matA, perm, perm, n, n,
                                 &matBLU_d, &matS, &matE_d, &matF_d);
         }
         else
#endif
         {
            hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, perm, n, n,
                               &matL, &matD, &matU, &matS, &u_end);
         }
         break;

      case 10: /* GMRES + hypre_iluk() */
#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_ILUSetupDevice(ilu_data, matA, perm, perm, n, nLU,
                                 &matBLU_d, &matS, &matE_d, &matF_d);
         }
         else
#endif
         {
            hypre_ILUSetupILUK(matA, fill_level, perm, perm, nLU, nLU,
                               &matL, &matD, &matU, &matS, &u_end);
         }
         break;

      case 11: /* GMRES + hypre_ilut() */
#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_ILUSetupDevice(ilu_data, matA, perm, perm, n, nLU,
                                 &matBLU_d, &matS, &matE_d, &matF_d);
         }
         else
#endif
         {

            hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, perm, nLU, nLU,
                               &matL, &matD, &matU, &matS, &u_end);
         }
         break;

      case 20: /* Newton Schulz Hotelling + hypre_iluk() */
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "NSH+ILUK setup on device runs requires unified memory!");
            return hypre_error_flag;
         }
#endif

         hypre_ILUSetupILUK(matA, fill_level, perm, perm, nLU, nLU,
                            &matL, &matD, &matU, &matS, &u_end);
         break;

      case 21: /* Newton Schulz Hotelling + hypre_ilut() */
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "NSH+ILUT setup on device runs requires unified memory!");
            return hypre_error_flag;
         }
#endif

         hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, perm, nLU, nLU,
                            &matL, &matD, &matU, &matS, &u_end);
         break;

      case 30: /* RAS + hypre_iluk() */
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "RAS+ILUK setup on device runs requires unified memory!");
            return hypre_error_flag;
         }
#endif

         hypre_ILUSetupILUKRAS(matA, fill_level, perm, nLU,
                               &matL, &matD, &matU);
         break;

      case 31: /* RAS + hypre_ilut() */
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "RAS+ILUT setup on device runs requires unified memory!");
            return hypre_error_flag;
         }
#endif

         hypre_ILUSetupILUTRAS(matA, max_row_elmts, droptol,
                               perm, nLU, &matL, &matD, &matU);
         break;

      case 40: /* ddPQ + GMRES + hypre_iluk() */
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "ddPQ+GMRES+ILUK setup on device runs requires unified memory!");
            return hypre_error_flag;
         }
#endif

         hypre_ILUSetupILUK(matA, fill_level, perm, qperm, nLU, nI,
                            &matL, &matD, &matU, &matS, &u_end);
         break;

      case 41: /* ddPQ + GMRES + hypre_ilut() */
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "ddPQ+GMRES+ILUT setup on device runs requires unified memory!");
            return hypre_error_flag;
         }
#endif

         hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, qperm, nLU, nI,
                            &matL, &matD, &matU, &matS, &u_end);
         break;

      case 50: /* RAP + hypre_modified_ilu0 */
#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
#if !defined(HYPRE_USING_UNIFIED_MEMORY)
            hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                              "GMRES+ILU0-RAP setup on device runs requires unified memory!");
            return hypre_error_flag;
#endif

            hypre_ILUSetupRAPILU0Device(matA, perm, n, nLU,
                                        &Aperm, &matS, &matALU_d, &matBLU_d,
                                        &matSLU_d, &matE_d, &matF_d, test_opt);
         }
         else
#endif
         {
            hypre_ILUSetupRAPILU0(matA, perm, n, nLU, &matL, &matD, &matU,
                                  &matmL, &matmD, &matmU, &u_end);
         }
         break;

      default: /* BJ + device_ilu0() */
#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypre_ILUSetupDevice(ilu_data, matA, perm, perm, n, n,
                                 &matBLU_d, &matS, &matE_d, &matF_d);
         }
         else
#endif
         {
            hypre_ILUSetupILU0(matA, perm, perm, n, n, &matL,
                               &matD, &matU, &matS, &u_end);
         }
         break;
   }

   /* Create additional temporary vector for iterative triangular solve */
   if (!tri_solve)
   {
      Ztemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                    hypre_ParCSRMatrixGlobalNumRows(A),
                                    hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize(Ztemp);
   }

   /* setup Schur solver - TODO (VPM): merge host and device paths below */
   switch (ilu_type)
   {
      case 0: case 1:
      default:
         break;

      case 10: case 11:
         if (matS)
         {
            /* Create work vectors */
#if defined(HYPRE_USING_GPU)
            if (exec == HYPRE_EXEC_DEVICE)
            {
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
               hypre_VectorData(Ftemp_upper)       = hypre_VectorData(
                                                        hypre_ParVectorLocalVector(Ftemp));
               hypre_SeqVectorInitialize(Ftemp_upper);

               Utemp_lower = hypre_SeqVectorCreate(n - nLU);
               hypre_VectorOwnsData(Utemp_lower)   = 0;
               hypre_VectorData(Utemp_lower)       = hypre_VectorData(
                                                        hypre_ParVectorLocalVector(Utemp)) + nLU;
               hypre_SeqVectorInitialize(Utemp_lower);

               /* create GMRES */
               //            HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

               hypre_GMRESFunctions * gmres_functions;

               gmres_functions =
                  hypre_GMRESFunctionsCreate(
                     hypre_ParKrylovCAlloc,
                     hypre_ParKrylovFree,
                     hypre_ParILUSchurGMRESCommInfoDevice, //parCSR A -> ilu_data
                     hypre_ParKrylovCreateVector,
                     hypre_ParKrylovCreateVectorArray,
                     hypre_ParKrylovDestroyVector,
                     hypre_ParKrylovMatvecCreate, //parCSR A -- inactive
                     ((tri_solve == 1) ?
                      hypre_ParILUSchurGMRESMatvecDevice :
                      hypre_ParILUSchurGMRESMatvecJacIterDevice), //parCSR A -> ilu_data
                     hypre_ParKrylovMatvecDestroy, //parCSR A -- inactive
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
               HYPRE_GMRESSetMaxIter         (schur_solver,
                                              hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
               HYPRE_GMRESSetTol             (schur_solver, hypre_ParILUDataSchurGMRESTol(ilu_data));
               HYPRE_GMRESSetAbsoluteTol     (schur_solver, hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
               HYPRE_GMRESSetLogging         (schur_solver, hypre_ParILUDataSchurSolverLogging(ilu_data));
               HYPRE_GMRESSetPrintLevel      (schur_solver,
                                              hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
               HYPRE_GMRESSetRelChange       (schur_solver, hypre_ParILUDataSchurGMRESRelChange(ilu_data));

               /* setup preconditioner parameters */
               /* create Unit precond */
               schur_precond = (HYPRE_Solver) ilu_vdata;

               /* add preconditioner to solver */
               HYPRE_GMRESSetPrecond(schur_solver,
                                     (HYPRE_PtrToSolverFcn) hypre_ParILUSchurGMRESDummySolveDevice,
                                     (HYPRE_PtrToSolverFcn) hypre_ParKrylovIdentitySetup,
                                     schur_precond);

               HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);
               if (schur_precond_gotten != (schur_precond))
               {
                  hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Schur complement got bad precond!");
                  hypre_GpuProfilingPopRange();
                  HYPRE_ANNOTATE_FUNC_END;

                  return hypre_error_flag;
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
               HYPRE_GMRESSetup(schur_solver,
                                (HYPRE_Matrix) ilu_vdata,
                                (HYPRE_Vector) rhs,
                                (HYPRE_Vector) x);

               /* solve for right-hand-side consists of only 1 */
               hypre_Vector      *rhs_local = hypre_ParVectorLocalVector(rhs);
               //HYPRE_Real        *Xtemp_data  = hypre_VectorData(Xtemp_local);
               hypre_SeqVectorSetConstantValues(rhs_local, 1.0);

               /* update ilu_data */
               hypre_ParILUDataSchurSolver   (ilu_data) = schur_solver;
               hypre_ParILUDataSchurPrecond  (ilu_data) = schur_precond;
               hypre_ParILUDataRhs           (ilu_data) = rhs;
               hypre_ParILUDataX             (ilu_data) = x;
            }
            else
#endif
            {
               /* setup GMRES parameters */
               HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

               HYPRE_GMRESSetKDim            (schur_solver, hypre_ParILUDataSchurGMRESKDim(ilu_data));
               HYPRE_GMRESSetMaxIter         (schur_solver,
                                              hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
               HYPRE_GMRESSetTol             (schur_solver, hypre_ParILUDataSchurGMRESTol(ilu_data));
               HYPRE_GMRESSetAbsoluteTol     (schur_solver, hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
               HYPRE_GMRESSetLogging         (schur_solver, hypre_ParILUDataSchurSolverLogging(ilu_data));
               HYPRE_GMRESSetPrintLevel      (schur_solver,
                                              hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
               HYPRE_GMRESSetRelChange       (schur_solver, hypre_ParILUDataSchurGMRESRelChange(ilu_data));

               /* setup preconditioner parameters */
               /* create precond, the default is ILU0 */
               HYPRE_ILUCreate               (&schur_precond);
               HYPRE_ILUSetType              (schur_precond, hypre_ParILUDataSchurPrecondIluType(ilu_data));
               HYPRE_ILUSetLevelOfFill       (schur_precond, hypre_ParILUDataSchurPrecondIluLfil(ilu_data));
               HYPRE_ILUSetMaxNnzPerRow      (schur_precond, hypre_ParILUDataSchurPrecondIluMaxRowNnz(ilu_data));
               HYPRE_ILUSetDropThresholdArray(schur_precond, hypre_ParILUDataSchurPrecondIluDroptol(ilu_data));
               HYPRE_ILUSetPrintLevel        (schur_precond, hypre_ParILUDataSchurPrecondPrintLevel(ilu_data));
               HYPRE_ILUSetTriSolve          (schur_precond, hypre_ParILUDataSchurPrecondTriSolve(ilu_data));
               HYPRE_ILUSetMaxIter           (schur_precond, hypre_ParILUDataSchurPrecondMaxIter(ilu_data));
               HYPRE_ILUSetLowerJacobiIters  (schur_precond,
                                              hypre_ParILUDataSchurPrecondLowerJacobiIters(ilu_data));
               HYPRE_ILUSetUpperJacobiIters  (schur_precond,
                                              hypre_ParILUDataSchurPrecondUpperJacobiIters(ilu_data));
               HYPRE_ILUSetTol               (schur_precond, hypre_ParILUDataSchurPrecondTol(ilu_data));

               /* add preconditioner to solver */
               HYPRE_GMRESSetPrecond(schur_solver,
                                     (HYPRE_PtrToSolverFcn) HYPRE_ILUSolve,
                                     (HYPRE_PtrToSolverFcn) HYPRE_ILUSetup,
                                     schur_precond);

               HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);
               if (schur_precond_gotten != (schur_precond))
               {
                  hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Schur complement got bad precond!");
                  hypre_GpuProfilingPopRange();
                  HYPRE_ANNOTATE_FUNC_END;

                  return hypre_error_flag;
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
               HYPRE_GMRESSetup(schur_solver,
                                (HYPRE_Matrix) matS,
                                (HYPRE_Vector) rhs,
                                (HYPRE_Vector) x);

               /* update ilu_data */
               hypre_ParILUDataSchurSolver   (ilu_data) = schur_solver;
               hypre_ParILUDataSchurPrecond  (ilu_data) = schur_precond;
               hypre_ParILUDataRhs           (ilu_data) = rhs;
               hypre_ParILUDataX             (ilu_data) = x;
            }
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
            hypre_NSHSetup(schur_solver, matS, rhs, x);

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
         num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
         send_size = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends) -
                     hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
         recv_size = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(matA));
         buffer_size = send_size > recv_size ? send_size : recv_size;

         /* TODO (VPM): Check these memory locations */
         fext = hypre_TAlloc(HYPRE_Real, buffer_size, HYPRE_MEMORY_HOST);
         uext = hypre_TAlloc(HYPRE_Real, buffer_size, HYPRE_MEMORY_HOST);
         break;

      case 40: case 41:
         if (matS)
         {
            /* setup GMRES parameters */
            HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

            HYPRE_GMRESSetKDim            (schur_solver, hypre_ParILUDataSchurGMRESKDim(ilu_data));
            HYPRE_GMRESSetMaxIter         (schur_solver,
                                           hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
            HYPRE_GMRESSetTol             (schur_solver, hypre_ParILUDataSchurGMRESTol(ilu_data));
            HYPRE_GMRESSetAbsoluteTol     (schur_solver, hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
            HYPRE_GMRESSetLogging         (schur_solver, hypre_ParILUDataSchurSolverLogging(ilu_data));
            HYPRE_GMRESSetPrintLevel      (schur_solver,
                                           hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
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
            HYPRE_ILUSetTriSolve          (schur_precond, hypre_ParILUDataSchurPrecondTriSolve(ilu_data));
            HYPRE_ILUSetLowerJacobiIters  (schur_precond,
                                           hypre_ParILUDataSchurPrecondLowerJacobiIters(ilu_data));
            HYPRE_ILUSetUpperJacobiIters  (schur_precond,
                                           hypre_ParILUDataSchurPrecondUpperJacobiIters(ilu_data));
            HYPRE_ILUSetTol               (schur_precond, hypre_ParILUDataSchurPrecondTol(ilu_data));

            /* add preconditioner to solver */
            HYPRE_GMRESSetPrecond(schur_solver,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ILUSolve,
                                  (HYPRE_PtrToSolverFcn) HYPRE_ILUSetup,
                                  schur_precond);

            HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);
            if (schur_precond_gotten != (schur_precond))
            {
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Schur complement got bad precond!");
               hypre_GpuProfilingPopRange();
               HYPRE_ANNOTATE_FUNC_END;

               return hypre_error_flag;
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
            HYPRE_GMRESSetup(schur_solver,
                             (HYPRE_Matrix) matS,
                             (HYPRE_Vector) rhs,
                             (HYPRE_Vector) x);

            /* update ilu_data */
            hypre_ParILUDataSchurSolver   (ilu_data) = schur_solver;
            hypre_ParILUDataSchurPrecond  (ilu_data) = schur_precond;
            hypre_ParILUDataRhs           (ilu_data) = rhs;
            hypre_ParILUDataX             (ilu_data) = x;
         }
         break;

      case 50:
#if defined(HYPRE_USING_GPU)
         if (matS && exec == HYPRE_EXEC_DEVICE)
         {
            Xtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
                                          hypre_ParCSRMatrixGlobalNumRows(matA),
                                          hypre_ParCSRMatrixRowStarts(matA));
            hypre_ParVectorInitialize(Xtemp);

            Ytemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
                                          hypre_ParCSRMatrixGlobalNumRows(matA),
                                          hypre_ParCSRMatrixRowStarts(matA));
            hypre_ParVectorInitialize(Ytemp);

            Ftemp_upper = hypre_SeqVectorCreate(nLU);
            hypre_VectorOwnsData(Ftemp_upper) = 0;
            hypre_VectorData(Ftemp_upper) = hypre_VectorData(hypre_ParVectorLocalVector(Ftemp));
            hypre_SeqVectorInitialize(Ftemp_upper);

            Utemp_lower = hypre_SeqVectorCreate(n - nLU);
            hypre_VectorOwnsData(Utemp_lower) = 0;
            hypre_VectorData(Utemp_lower) = nLU +
                                            hypre_VectorData(hypre_ParVectorLocalVector(Utemp));
            hypre_SeqVectorInitialize(Utemp_lower);

            /* create GMRES */
            //            HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

            hypre_GMRESFunctions * gmres_functions;

            gmres_functions =
               hypre_GMRESFunctionsCreate(
                  hypre_ParKrylovCAlloc,
                  hypre_ParKrylovFree,
                  hypre_ParILUSchurGMRESCommInfoDevice, //parCSR A -> ilu_data
                  hypre_ParKrylovCreateVector,
                  hypre_ParKrylovCreateVectorArray,
                  hypre_ParKrylovDestroyVector,
                  hypre_ParKrylovMatvecCreate, //parCSR A -- inactive
                  hypre_ParILURAPSchurGMRESMatvecDevice, //parCSR A -> ilu_data
                  hypre_ParKrylovMatvecDestroy, //parCSR A -- inactive
                  hypre_ParKrylovInnerProd,
                  hypre_ParKrylovCopyVector,
                  hypre_ParKrylovClearVector,
                  hypre_ParKrylovScaleVector,
                  hypre_ParKrylovAxpy,
                  hypre_ParKrylovIdentitySetup, //parCSR A -- inactive
                  hypre_ParKrylovIdentity ); //parCSR A -- inactive
            schur_solver = (HYPRE_Solver) hypre_GMRESCreate(gmres_functions);

            /* setup GMRES parameters */
            /* at least should apply 1 solve */
            if (hypre_ParILUDataSchurGMRESKDim(ilu_data) == 0)
            {
               hypre_ParILUDataSchurGMRESKDim(ilu_data)++;
            }
            HYPRE_GMRESSetKDim            (schur_solver, hypre_ParILUDataSchurGMRESKDim(ilu_data));
            HYPRE_GMRESSetMaxIter         (schur_solver,
                                           hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
            HYPRE_GMRESSetTol             (schur_solver, hypre_ParILUDataSchurGMRESTol(ilu_data));
            HYPRE_GMRESSetAbsoluteTol     (schur_solver, hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
            HYPRE_GMRESSetLogging         (schur_solver, hypre_ParILUDataSchurSolverLogging(ilu_data));
            HYPRE_GMRESSetPrintLevel      (schur_solver,
                                           hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
            HYPRE_GMRESSetRelChange       (schur_solver, hypre_ParILUDataSchurGMRESRelChange(ilu_data));

            /* setup preconditioner parameters */
            /* create Schur precond */
            schur_precond = (HYPRE_Solver) ilu_vdata;

            /* add preconditioner to solver */
            HYPRE_GMRESSetPrecond(schur_solver,
                                  (HYPRE_PtrToSolverFcn) hypre_ParILURAPSchurGMRESSolveDevice,
                                  (HYPRE_PtrToSolverFcn) hypre_ParKrylovIdentitySetup,
                                  schur_precond);
            HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);

            if (schur_precond_gotten != (schur_precond))
            {
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Schur complement got bad precond!");
               hypre_GpuProfilingPopRange();
               HYPRE_ANNOTATE_FUNC_END;

               return hypre_error_flag;
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
            HYPRE_GMRESSetup(schur_solver,
                             (HYPRE_Matrix) ilu_vdata,
                             (HYPRE_Vector) rhs,
                             (HYPRE_Vector) x);

            /* solve for right-hand-side consists of only 1 */
            //hypre_Vector      *rhs_local = hypre_ParVectorLocalVector(rhs);
            //HYPRE_Real        *Xtemp_data  = hypre_VectorData(Xtemp_local);
            //hypre_SeqVectorSetConstantValues(rhs_local, 1.0);

            /* Update ilu_data */
            hypre_ParILUDataSchurSolver(ilu_data)  = schur_solver;
            hypre_ParILUDataSchurPrecond(ilu_data) = schur_precond;
            hypre_ParILUDataRhs(ilu_data)          = rhs;
            hypre_ParILUDataX(ilu_data)            = x;
         }
         else
#endif
         {
            /* Need to create working vector rhs and x for Schur System */
            HYPRE_Int      m = n - nLU;
            HYPRE_BigInt   global_start, S_total_rows, S_row_starts[2];
            HYPRE_BigInt   big_m = (HYPRE_BigInt) m;

            hypre_MPI_Allreduce(&big_m, &S_total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

            if (S_total_rows > 0)
            {
               Xtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
                                             hypre_ParCSRMatrixGlobalNumRows(matA),
                                             hypre_ParCSRMatrixRowStarts(matA));
               hypre_ParVectorInitialize(Xtemp);

               Ytemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
                                             hypre_ParCSRMatrixGlobalNumRows(matA),
                                             hypre_ParCSRMatrixRowStarts(matA));
               hypre_ParVectorInitialize(Ytemp);

               hypre_MPI_Scan(&big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
               S_row_starts[0] = global_start - big_m;
               S_row_starts[1] = global_start;

               rhs = hypre_ParVectorCreate(comm,
                                           S_total_rows,
                                           S_row_starts);
               hypre_ParVectorInitialize(rhs);

               x = hypre_ParVectorCreate(comm,
                                         S_total_rows,
                                         S_row_starts);
               hypre_ParVectorInitialize(x);

               /* create GMRES */
               //            HYPRE_ParCSRGMRESCreate(comm, &schur_solver);

               hypre_GMRESFunctions * gmres_functions;

               gmres_functions =
                  hypre_GMRESFunctionsCreate(
                     hypre_ParKrylovCAlloc,
                     hypre_ParKrylovFree,
                     hypre_ParILURAPSchurGMRESCommInfoHost, //parCSR A -> ilu_data
                     hypre_ParKrylovCreateVector,
                     hypre_ParKrylovCreateVectorArray,
                     hypre_ParKrylovDestroyVector,
                     hypre_ParKrylovMatvecCreate, //parCSR A -- inactive
                     hypre_ParILURAPSchurGMRESMatvecHost, //parCSR A -> ilu_data
                     hypre_ParKrylovMatvecDestroy, //parCSR A -- inactive
                     hypre_ParKrylovInnerProd,
                     hypre_ParKrylovCopyVector,
                     hypre_ParKrylovClearVector,
                     hypre_ParKrylovScaleVector,
                     hypre_ParKrylovAxpy,
                     hypre_ParKrylovIdentitySetup, //parCSR A -- inactive
                     hypre_ParKrylovIdentity ); //parCSR A -- inactive
               schur_solver = (HYPRE_Solver) hypre_GMRESCreate(gmres_functions);

               /* setup GMRES parameters */
               /* at least should apply 1 solve */
               if (hypre_ParILUDataSchurGMRESKDim(ilu_data) == 0)
               {
                  hypre_ParILUDataSchurGMRESKDim(ilu_data)++;
               }
               HYPRE_GMRESSetKDim            (schur_solver, hypre_ParILUDataSchurGMRESKDim(ilu_data));
               HYPRE_GMRESSetMaxIter         (schur_solver,
                                              hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
               HYPRE_GMRESSetTol             (schur_solver, hypre_ParILUDataSchurGMRESTol(ilu_data));
               HYPRE_GMRESSetAbsoluteTol     (schur_solver, hypre_ParILUDataSchurGMRESAbsoluteTol(ilu_data));
               HYPRE_GMRESSetLogging         (schur_solver, hypre_ParILUDataSchurSolverLogging(ilu_data));
               HYPRE_GMRESSetPrintLevel      (schur_solver,
                                              hypre_ParILUDataSchurSolverPrintLevel(ilu_data));/* set to zero now, don't print */
               HYPRE_GMRESSetRelChange       (schur_solver, hypre_ParILUDataSchurGMRESRelChange(ilu_data));

               /* setup preconditioner parameters */
               /* create Schur precond */
               schur_precond = (HYPRE_Solver) ilu_vdata;

               /* add preconditioner to solver */
               HYPRE_GMRESSetPrecond(schur_solver,
                                     (HYPRE_PtrToSolverFcn) hypre_ParILURAPSchurGMRESSolveHost,
                                     (HYPRE_PtrToSolverFcn) hypre_ParKrylovIdentitySetup,
                                     schur_precond);
               HYPRE_GMRESGetPrecond(schur_solver, &schur_precond_gotten);
               if (schur_precond_gotten != (schur_precond))
               {
                  hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Schur complement got bad precond!");
                  hypre_GpuProfilingPopRange();
                  HYPRE_ANNOTATE_FUNC_END;

                  return hypre_error_flag;
               }

               /* setup solver */
               HYPRE_GMRESSetup(schur_solver,
                                (HYPRE_Matrix) ilu_vdata,
                                (HYPRE_Vector) rhs,
                                (HYPRE_Vector) x);

               /* solve for right-hand-side consists of only 1 */
               //hypre_Vector      *rhs_local = hypre_ParVectorLocalVector(rhs);
               //HYPRE_Real        *Xtemp_data  = hypre_VectorData(Xtemp_local);
               //hypre_SeqVectorSetConstantValues(rhs_local, 1.0);
            } /* if (S_total_rows > 0) */

            /* Update ilu_data */
            hypre_ParILUDataSchurSolver(ilu_data)  = schur_solver;
            hypre_ParILUDataSchurPrecond(ilu_data) = schur_precond;
            hypre_ParILUDataRhs(ilu_data)          = rhs;
            hypre_ParILUDataX(ilu_data)            = x;
         }
         break;
   }

   /* set pointers to ilu data */
   /* set device data pointers */
#if defined(HYPRE_USING_GPU)
   hypre_ParILUDataMatAILUDevice(ilu_data) = matALU_d;
   hypre_ParILUDataMatBILUDevice(ilu_data) = matBLU_d;
   hypre_ParILUDataMatSILUDevice(ilu_data) = matSLU_d;
   hypre_ParILUDataMatEDevice(ilu_data)    = matE_d;
   hypre_ParILUDataMatFDevice(ilu_data)    = matF_d;
   hypre_ParILUDataAperm(ilu_data)         = Aperm;
   hypre_ParILUDataR(ilu_data)             = R;
   hypre_ParILUDataP(ilu_data)             = P;
   hypre_ParILUDataFTempUpper(ilu_data)    = Ftemp_upper;
   hypre_ParILUDataUTempLower(ilu_data)    = Utemp_lower;
   hypre_ParILUDataADiagDiag(ilu_data)     = Adiag_diag;
   hypre_ParILUDataSDiagDiag(ilu_data)     = Sdiag_diag;
#endif

   /* Set pointers to ilu data */
   hypre_ParILUDataMatA(ilu_data)          = matA;
   hypre_ParILUDataXTemp(ilu_data)         = Xtemp;
   hypre_ParILUDataYTemp(ilu_data)         = Ytemp;
   hypre_ParILUDataZTemp(ilu_data)         = Ztemp;
   hypre_ParILUDataF(ilu_data)             = F_array;
   hypre_ParILUDataU(ilu_data)             = U_array;
   hypre_ParILUDataMatL(ilu_data)          = matL;
   hypre_ParILUDataMatD(ilu_data)          = matD;
   hypre_ParILUDataMatU(ilu_data)          = matU;
   hypre_ParILUDataMatLModified(ilu_data)  = matmL;
   hypre_ParILUDataMatDModified(ilu_data)  = matmD;
   hypre_ParILUDataMatUModified(ilu_data)  = matmU;
   hypre_ParILUDataMatS(ilu_data)          = matS;
   hypre_ParILUDataCFMarkerArray(ilu_data) = CF_marker_array;
   hypre_ParILUDataPerm(ilu_data)          = perm;
   hypre_ParILUDataQPerm(ilu_data)         = qperm;
   hypre_ParILUDataNLU(ilu_data)           = nLU;
   hypre_ParILUDataNI(ilu_data)            = nI;
   hypre_ParILUDataUEnd(ilu_data)          = u_end;
   hypre_ParILUDataUExt(ilu_data)          = uext;
   hypre_ParILUDataFExt(ilu_data)          = fext;

   /* compute operator complexity */
   hypre_ParCSRMatrixSetDNumNonzeros(matA);
   nnzS = 0.0;

   /* size_C is the size of global coarse grid, upper left part */
   size_C = hypre_ParCSRMatrixGlobalNumRows(matA);

   /* switch to compute complexity */
#if defined(HYPRE_USING_GPU)
   HYPRE_Int nnzBEF = 0;
   HYPRE_Int nnzG; /* Global nnz */

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
#endif
   {
      if (matS)
      {
         hypre_ParCSRMatrixSetDNumNonzeros(matS);
         nnzS = hypre_ParCSRMatrixDNumNonzeros(matS);

         /* If we have Schur system need to reduce it from size_C */
         size_C -= hypre_ParCSRMatrixGlobalNumRows(matS);
         switch (ilu_type)
         {
            case 10: case 11: case 40: case 41: case 50:
               /* Now we need to compute the preconditioner */
               schur_precond_ilu = (hypre_ParILUData*) (hypre_ParILUDataSchurPrecond(ilu_data));

               /* borrow i for local nnz of S */
               nnzS_offd_local = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(matS));
               hypre_MPI_Allreduce(&nnzS_offd_local, &nnzS_offd, 1, HYPRE_MPI_REAL,
                                   hypre_MPI_SUM, comm);
               nnzS = nnzS * hypre_ParILUDataOperatorComplexity(schur_precond_ilu) + nnzS_offd;
               break;

            case 20: case 21:
               schur_solver_nsh = (hypre_ParNSHData*) hypre_ParILUDataSchurSolver(ilu_data);
               nnzS *= hypre_ParNSHDataOperatorComplexity(schur_solver_nsh);
               break;

            default:
               break;
         }
      }

      hypre_ParILUDataOperatorComplexity(ilu_data) = ((HYPRE_Real)size_C + nnzS +
                                                      hypre_ParCSRMatrixDNumNonzeros(matL) +
                                                      hypre_ParCSRMatrixDNumNonzeros(matU)) /
                                                     hypre_ParCSRMatrixDNumNonzeros(matA);
   }

   /* TODO (VPM): Move ILU statistics printout to its own function */
   if ((my_id == 0) && (print_level > 0))
   {
      hypre_printf("ILU SETUP: operator complexity = %f  \n",
                   hypre_ParILUDataOperatorComplexity(ilu_data));
      if (hypre_ParILUDataTriSolve(ilu_data))
      {
         hypre_printf("ILU SOLVE: using direct triangular solves\n",
                      hypre_ParILUDataOperatorComplexity(ilu_data));
      }
      else
      {
         hypre_printf("ILU SOLVE: using iterative triangular solves\n",
                      hypre_ParILUDataOperatorComplexity(ilu_data));
      }

#if defined (HYPRE_USING_ROCSPARSE)
      HYPRE_Int i;

      if (hypre_ParILUDataIterativeSetupType(ilu_data))
      {
         hypre_printf("ILU: iterative setup type = %d\n",
                      hypre_ParILUDataIterativeSetupType(ilu_data));
         hypre_printf("ILU: iterative setup option = %d\n",
                      hypre_ParILUDataIterativeSetupOption(ilu_data));
         if (hypre_ParILUDataIterativeSetupOption(ilu_data) & 0x2)
         {
            /* This path enables termination based on stopping tolerance */
            hypre_printf("ILU: iterative setup tolerance = %g\n",
                         hypre_ParILUDataIterativeSetupTolerance(ilu_data));
         }
         else
         {
            /* This path enables termination based on number of iterations */
            hypre_printf("ILU: iterative setup max. iters = %d\n",
                         hypre_ParILUDataIterativeSetupMaxIter(ilu_data));
         }

         /* TODO (VPM): Add min, max, avg statistics across ranks */
         hypre_printf("ILU: iterative setup num. iters at rank 0 = %d\n",
                      hypre_ParILUDataIterativeSetupNumIter(ilu_data));

         /* Show convergence history */
         if (hypre_ParILUDataIterativeSetupOption(ilu_data) & 0x10)
         {
            hypre_printf("ILU: iterative setup convergence history at rank 0:\n");
            hypre_printf("%8s", "iter");
            if (hypre_ParILUDataIterativeSetupOption(ilu_data) & 0x08)
            {
               hypre_printf(" %14s %14s", "residual", "rate");
            }
            if (hypre_ParILUDataIterativeSetupOption(ilu_data) & 0x04)
            {
               hypre_printf(" %14s %14s", "correction", "rate");
            }
            hypre_printf("\n");
            printf("%8d", 0);
            if (hypre_ParILUDataIterativeSetupOption(ilu_data) & 0x08)
            {
               hypre_printf(" %14.5e %14.5e",
                            hypre_ParILUDataIterSetupResidualNorm(ilu_data, 0), 1.0);
            }
            if (hypre_ParILUDataIterativeSetupOption(ilu_data) & 0x04)
            {
               hypre_printf(" %14.5e %14.5e",
                            hypre_ParILUDataIterSetupCorrectionNorm(ilu_data, 0), 1.0);
            }
            hypre_printf("\n");

            for (i = 1; i < hypre_ParILUDataIterativeSetupNumIter(ilu_data); i++)
            {
               printf("%8d", i);
               if (hypre_ParILUDataIterativeSetupOption(ilu_data) & 0x08)
               {
                  hypre_printf(" %14.5e %14.5e",
                               hypre_ParILUDataIterSetupResidualNorm(ilu_data, i),
                               hypre_ParILUDataIterSetupResidualNorm(ilu_data, i) /
                               hypre_ParILUDataIterSetupResidualNorm(ilu_data, i - 1));
               }
               if (hypre_ParILUDataIterativeSetupOption(ilu_data) & 0x04)
               {
                  hypre_printf(" %14.5e %14.5e",
                               hypre_ParILUDataIterSetupCorrectionNorm(ilu_data, i),
                               hypre_ParILUDataIterSetupCorrectionNorm(ilu_data, i) /
                               hypre_ParILUDataIterSetupCorrectionNorm(ilu_data, i - 1));
               }
               hypre_printf("\n");
            }
         }
      }
#endif
   }

   if (logging > 1)
   {
      residual =
         hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
                               hypre_ParCSRMatrixGlobalNumRows(matA),
                               hypre_ParCSRMatrixRowStarts(matA) );
      hypre_ParVectorInitialize(residual);
      hypre_ParILUDataResidual(ilu_data) = residual;
   }
   else
   {
      hypre_ParILUDataResidual(ilu_data) = NULL;
   }
   rel_res_norms = hypre_CTAlloc(HYPRE_Real,
                                 hypre_ParILUDataMaxIter(ilu_data),
                                 HYPRE_MEMORY_HOST);
   hypre_ParILUDataRelResNorms(ilu_data) = rel_res_norms;

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILUExtractEBFC
 *
 * Extract submatrix from diagonal part of A into
 *    | B F |
 *    | E C |
 *
 * A = input matrix
 * perm = permutation array indicating ordering of rows. Perm could come from a
 *    CF_marker array or a reordering routine.
 * qperm = permutation array indicating ordering of columns
 * Bp = pointer to the output B matrix.
 * Cp = pointer to the output C matrix.
 * Ep = pointer to the output E matrix.
 * Fp = pointer to the output F matrix.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILUExtractEBFC(hypre_CSRMatrix   *A_diag,
                        HYPRE_Int          nLU,
                        hypre_CSRMatrix  **Bp,
                        hypre_CSRMatrix  **Cp,
                        hypre_CSRMatrix  **Ep,
                        hypre_CSRMatrix  **Fp)
{
   /* Get necessary slots */
   HYPRE_Int            n                = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int            nnz_A_diag       = hypre_CSRMatrixNumNonzeros(A_diag);
   HYPRE_MemoryLocation memory_location  = hypre_CSRMatrixMemoryLocation(A_diag);

   hypre_CSRMatrix     *B = NULL;
   hypre_CSRMatrix     *C = NULL;
   hypre_CSRMatrix     *E = NULL;
   hypre_CSRMatrix     *F = NULL;
   HYPRE_Int            i, j, row, col;

   hypre_assert(nLU >= 0 && nLU <= n);

   if (nLU == n)
   {
      /* No Schur complement */
      B = hypre_CSRMatrixCreate(n, n, nnz_A_diag);
      C = hypre_CSRMatrixCreate(0, 0, 0);
      E = hypre_CSRMatrixCreate(0, 0, 0);
      F = hypre_CSRMatrixCreate(0, 0, 0);

      hypre_CSRMatrixInitialize_v2(B, 0, memory_location);
      hypre_CSRMatrixInitialize_v2(C, 0, memory_location);
      hypre_CSRMatrixInitialize_v2(E, 0, memory_location);
      hypre_CSRMatrixInitialize_v2(F, 0, memory_location);

      hypre_CSRMatrixCopy(A_diag, B, 1);
   }
   else if (nLU == 0)
   {
      /* All Schur complement */
      C = hypre_CSRMatrixCreate(n, n, nnz_A_diag);
      B = hypre_CSRMatrixCreate(0, 0, 0);
      E = hypre_CSRMatrixCreate(0, 0, 0);
      F = hypre_CSRMatrixCreate(0, 0, 0);

      hypre_CSRMatrixInitialize_v2(C, 0, memory_location);
      hypre_CSRMatrixInitialize_v2(B, 0, memory_location);
      hypre_CSRMatrixInitialize_v2(E, 0, memory_location);
      hypre_CSRMatrixInitialize_v2(F, 0, memory_location);

      hypre_CSRMatrixCopy(A_diag, C, 1);
   }
   else
   {
      /* Has schur complement */
      HYPRE_Int         m = n - nLU;
      HYPRE_Int         capacity_B;
      HYPRE_Int         capacity_E;
      HYPRE_Int         capacity_F;
      HYPRE_Int         capacity_C;
      HYPRE_Int         ctrB;
      HYPRE_Int         ctrC;
      HYPRE_Int         ctrE;
      HYPRE_Int         ctrF;

      HYPRE_Int        *B_i    = NULL;
      HYPRE_Int        *C_i    = NULL;
      HYPRE_Int        *E_i    = NULL;
      HYPRE_Int        *F_i    = NULL;
      HYPRE_Int        *B_j    = NULL;
      HYPRE_Int        *C_j    = NULL;
      HYPRE_Int        *E_j    = NULL;
      HYPRE_Int        *F_j    = NULL;
      HYPRE_Complex    *B_data = NULL;
      HYPRE_Complex    *C_data = NULL;
      HYPRE_Complex    *E_data = NULL;
      HYPRE_Complex    *F_data = NULL;

      hypre_CSRMatrix  *h_A_diag;
      HYPRE_Int        *A_diag_i;
      HYPRE_Int        *A_diag_j;
      HYPRE_Complex    *A_diag_data;

      /* Create/Get host pointer for A_diag */
      h_A_diag = (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE) ?
                 hypre_CSRMatrixClone_v2(A_diag, 1, HYPRE_MEMORY_HOST) : A_diag;
      A_diag_i = hypre_CSRMatrixI(h_A_diag);
      A_diag_j = hypre_CSRMatrixJ(h_A_diag);
      A_diag_data = hypre_CSRMatrixData(h_A_diag);

      /* Estimate # of nonzeros */
      capacity_B = (HYPRE_Int) (nLU + hypre_ceil(nnz_A_diag * 1.0 * nLU / n * nLU / n));
      capacity_C = (HYPRE_Int) (m + hypre_ceil(nnz_A_diag * 1.0 * m / n * m / n));
      capacity_E = (HYPRE_Int) (hypre_min(m, nLU) + hypre_ceil(nnz_A_diag * 1.0 * nLU / n * m / n));
      capacity_F = capacity_E;

      /* Create CSRMatrices */
      B = hypre_CSRMatrixCreate(nLU, nLU, capacity_B);
      C = hypre_CSRMatrixCreate(m, m, capacity_C);
      E = hypre_CSRMatrixCreate(m, nLU, capacity_E);
      F = hypre_CSRMatrixCreate(nLU, m, capacity_F);

      /* Initialize matrices on the host */
      hypre_CSRMatrixInitialize_v2(B, 0, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixInitialize_v2(C, 0, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixInitialize_v2(E, 0, HYPRE_MEMORY_HOST);
      hypre_CSRMatrixInitialize_v2(F, 0, HYPRE_MEMORY_HOST);

      /* Access pointers */
      B_i    = hypre_CSRMatrixI(B);
      B_j    = hypre_CSRMatrixJ(B);
      B_data = hypre_CSRMatrixData(B);

      C_i    = hypre_CSRMatrixI(C);
      C_j    = hypre_CSRMatrixJ(C);
      C_data = hypre_CSRMatrixData(C);

      E_i    = hypre_CSRMatrixI(E);
      E_j    = hypre_CSRMatrixJ(E);
      E_data = hypre_CSRMatrixData(E);

      F_i    = hypre_CSRMatrixI(F);
      F_j    = hypre_CSRMatrixJ(F);
      F_data = hypre_CSRMatrixData(F);

      ctrB = ctrC = ctrE = ctrF = 0;

      /* Loop to copy data */
      /* B and F first */
      for (i = 0; i < nLU; i++)
      {
         B_i[i] = ctrB;
         F_i[i] = ctrF;
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
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
               capacity_B = (HYPRE_Int)(capacity_B * EXPAND_FACT + 1);
               B_j = hypre_TReAlloc_v2(B_j, HYPRE_Int, tmp, HYPRE_Int,
                                       capacity_B, HYPRE_MEMORY_HOST);
               B_data = hypre_TReAlloc_v2(B_data, HYPRE_Complex, tmp, HYPRE_Complex,
                                          capacity_B, HYPRE_MEMORY_HOST);
            }
         }
         for (; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            col = col - nLU;
            F_j[ctrF] = col;
            F_data[ctrF++] = A_diag_data[j];
            if (ctrF >= capacity_F)
            {
               HYPRE_Int tmp;
               tmp = capacity_F;
               capacity_F = (HYPRE_Int)(capacity_F * EXPAND_FACT + 1);
               F_j = hypre_TReAlloc_v2(F_j, HYPRE_Int, tmp, HYPRE_Int,
                                       capacity_F, HYPRE_MEMORY_HOST);
               F_data = hypre_TReAlloc_v2(F_data, HYPRE_Complex, tmp, HYPRE_Complex,
                                          capacity_F, HYPRE_MEMORY_HOST);
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
         for (j = A_diag_i[i]; j < A_diag_i[i + 1]; j++)
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
               capacity_E = (HYPRE_Int)(capacity_E * EXPAND_FACT + 1);
               E_j = hypre_TReAlloc_v2(E_j, HYPRE_Int, tmp, HYPRE_Int,
                                       capacity_E, HYPRE_MEMORY_HOST);
               E_data = hypre_TReAlloc_v2(E_data, HYPRE_Complex, tmp, HYPRE_Complex,
                                          capacity_E, HYPRE_MEMORY_HOST);
            }
         }
         for (; j < A_diag_i[i + 1]; j++)
         {
            col = A_diag_j[j];
            col = col - nLU;
            C_j[ctrC] = col;
            C_data[ctrC++] = A_diag_data[j];
            if (ctrC >= capacity_C)
            {
               HYPRE_Int tmp;
               tmp = capacity_C;
               capacity_C = (HYPRE_Int)(capacity_C * EXPAND_FACT + 1);
               C_j = hypre_TReAlloc_v2(C_j, HYPRE_Int, tmp, HYPRE_Int,
                                       capacity_C, HYPRE_MEMORY_HOST);
               C_data = hypre_TReAlloc_v2(C_data, HYPRE_Complex, tmp, HYPRE_Complex,
                                          capacity_C, HYPRE_MEMORY_HOST);
            }
         }
      }
      E_i[m] = ctrE;
      C_i[m] = ctrC;

      hypre_assert((ctrB + ctrC + ctrE + ctrF) == nnz_A_diag);

      /* Update pointers */
      hypre_CSRMatrixJ(B)           = B_j;
      hypre_CSRMatrixData(B)        = B_data;
      hypre_CSRMatrixNumNonzeros(B) = ctrB;

      hypre_CSRMatrixJ(C)           = C_j;
      hypre_CSRMatrixData(C)        = C_data;
      hypre_CSRMatrixNumNonzeros(C) = ctrC;

      hypre_CSRMatrixJ(E)           = E_j;
      hypre_CSRMatrixData(E)        = E_data;
      hypre_CSRMatrixNumNonzeros(E) = ctrE;

      hypre_CSRMatrixJ(F)           = F_j;
      hypre_CSRMatrixData(F)        = F_data;
      hypre_CSRMatrixNumNonzeros(F) = ctrF;

      /* Migrate to final memory location */
      hypre_CSRMatrixMigrate(B, memory_location);
      hypre_CSRMatrixMigrate(C, memory_location);
      hypre_CSRMatrixMigrate(E, memory_location);
      hypre_CSRMatrixMigrate(F, memory_location);

      /* Free memory */
      if (h_A_diag != A_diag)
      {
         hypre_CSRMatrixDestroy(h_A_diag);
      }
   }

   /* Set output pointers */
   *Bp = B;
   *Cp = C;
   *Ep = E;
   *Fp = F;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParILURAPReorder
 *
 * Reorder matrix A based on local permutation, i.e., combine local
 * permutation into global permutation)
 *
 * WARNING: We don't put diagonal to the first entry of each row
 *
 * A = input matrix
 * perm = permutation array indicating ordering of rows.
 *        Perm could come from a CF_marker array or a reordering routine.
 * rqperm = reverse permutation array indicating ordering of columns
 * A_pq = pointer to the output par CSR matrix.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParILURAPReorder(hypre_ParCSRMatrix  *A,
                       HYPRE_Int           *perm,
                       HYPRE_Int           *rqperm,
                       hypre_ParCSRMatrix **A_pq)
{
   /* Get necessary slots */
   MPI_Comm             comm            = hypre_ParCSRMatrixComm(A);
   hypre_CSRMatrix     *A_diag          = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int            n               = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_MemoryLocation memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   /* Permutation matrices */
   hypre_ParCSRMatrix  *P, *Q, *PAQ, *PA;
   hypre_CSRMatrix     *P_diag, *Q_diag;
   HYPRE_Int           *P_diag_i, *P_diag_j, *Q_diag_i, *Q_diag_j;
   HYPRE_Complex       *P_diag_data, *Q_diag_data;
   HYPRE_Int           *h_perm, *h_rqperm;

   /* Local variables */
   HYPRE_Int            i;

   /* Trivial case */
   if (!perm && !rqperm)
   {
      *A_pq = hypre_ParCSRMatrixClone(A, 1);

      return hypre_error_flag;
   }
   else if (!perm && rqperm)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG, "(!perm && rqperm) should not be possible!");
   }
   else if (perm && !rqperm)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG, "(perm && !rqperm) should not be possible!");
   }

   /* Create permutation matrices P = I(perm,:) and Q(rqperm,:), such that Apq = PAQ */
   P = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A),
                                hypre_ParCSRMatrixColStarts(A),
                                0,
                                n,
                                0);

   Q = hypre_ParCSRMatrixCreate(comm,
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A),
                                hypre_ParCSRMatrixColStarts(A),
                                0,
                                n,
                                0);

   hypre_ParCSRMatrixInitialize_v2(P, HYPRE_MEMORY_HOST);
   hypre_ParCSRMatrixInitialize_v2(Q, HYPRE_MEMORY_HOST);

   P_diag      = hypre_ParCSRMatrixDiag(P);
   Q_diag      = hypre_ParCSRMatrixDiag(Q);

   P_diag_i    = hypre_CSRMatrixI(P_diag);
   P_diag_j    = hypre_CSRMatrixJ(P_diag);
   P_diag_data = hypre_CSRMatrixData(P_diag);

   Q_diag_i    = hypre_CSRMatrixI(Q_diag);
   Q_diag_j    = hypre_CSRMatrixJ(Q_diag);
   Q_diag_data = hypre_CSRMatrixData(Q_diag);

   /* Set/Move permutation vectors on host */
   if (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE)
   {
      h_perm   = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
      h_rqperm = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);

      hypre_TMemcpy(h_perm,   perm,   HYPRE_Int, n, HYPRE_MEMORY_HOST, memory_location);
      hypre_TMemcpy(h_rqperm, rqperm, HYPRE_Int, n, HYPRE_MEMORY_HOST, memory_location);
   }
   else
   {
      h_perm   = perm;
      h_rqperm = rqperm;
   }

   /* Fill data */
#if defined(HYPRE_USING_OPENMP)
   #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
   for (i = 0; i < n; i++)
   {
      P_diag_i[i] = i;
      P_diag_j[i] = h_perm[i];
      P_diag_data[i] = 1.0;

      Q_diag_i[i] = i;
      Q_diag_j[i] = h_rqperm[i];
      Q_diag_data[i] = 1.0;
   }
   P_diag_i[n] = n;
   Q_diag_i[n] = n;

   /* Move to final memory location */
   hypre_ParCSRMatrixMigrate(P, memory_location);
   hypre_ParCSRMatrixMigrate(Q, memory_location);

   /* Update A */
   PA  = hypre_ParCSRMatMat(P, A);
   PAQ = hypre_ParCSRMatMat(PA, Q);
   //PAQ = hypre_ParCSRMatrixRAPKT(P, A, Q, 0);

   /* free and return */
   hypre_ParCSRMatrixDestroy(P);
   hypre_ParCSRMatrixDestroy(Q);
   hypre_ParCSRMatrixDestroy(PA);
   if (h_perm != perm)
   {
      hypre_TFree(h_perm, HYPRE_MEMORY_HOST);
   }
   if (h_rqperm != rqperm)
   {
      hypre_TFree(h_rqperm, HYPRE_MEMORY_HOST);
   }

   *A_pq = PAQ;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetupLDUtoCusparse
 *
 * Convert the L, D, U style to the cusparse style
 * Assume the diagonal of L and U are the ilu factorization, directly combine them
 *
 * TODO (VPM): Check this function's name
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupLDUtoCusparse(hypre_ParCSRMatrix  *L,
                            HYPRE_Real          *D,
                            hypre_ParCSRMatrix  *U,
                            hypre_ParCSRMatrix **LDUp)
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

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);


   /* cuda data slot */

   /* create matrix */

   LDU = hypre_ParCSRMatrixCreate(comm,
                                  hypre_ParCSRMatrixGlobalNumRows(L),
                                  hypre_ParCSRMatrixGlobalNumRows(L),
                                  hypre_ParCSRMatrixRowStarts(L),
                                  hypre_ParCSRMatrixColStarts(L),
                                  0,
                                  nnz_LDU,
                                  0);

   LDU_diag = hypre_ParCSRMatrixDiag(LDU);
   LDU_diag_i = hypre_TAlloc(HYPRE_Int, n + 1, HYPRE_MEMORY_DEVICE);
   LDU_diag_j = hypre_TAlloc(HYPRE_Int, nnz_LDU, HYPRE_MEMORY_DEVICE);
   LDU_diag_data = hypre_TAlloc(HYPRE_Real, nnz_LDU, HYPRE_MEMORY_DEVICE);

   pos = 0;

   for (i = 1; i <= n; i++)
   {
      LDU_diag_i[i - 1] = pos;
      for (j = L_diag_i[i - 1]; j < L_diag_i[i]; j++)
      {
         LDU_diag_j[pos] = L_diag_j[j];
         LDU_diag_data[pos++] = L_diag_data[j];
      }
      LDU_diag_j[pos] = i - 1;
      LDU_diag_data[pos++] = 1.0 / D[i - 1];
      for (j = U_diag_i[i - 1]; j < U_diag_i[i]; j++)
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
#if defined(HYPRE_USING_GPU)
   hypre_CSRMatrixSortRow(LDU_diag);
#endif
   hypre_ParCSRMatrixDiag(LDU) = LDU_diag;

   *LDUp = LDU;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetupRAPMILU0
 *
 * Apply the (modified) ILU factorization to the diagonal block of A only.
 *
 * A: matrix
 * ALUp: pointer to the result, factorization stroed on the diagonal
 * modified: set to 0 to use classical ILU0
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupRAPMILU0(hypre_ParCSRMatrix  *A,
                       hypre_ParCSRMatrix **ALUp,
                       HYPRE_Int            modified)
{
   HYPRE_Int             n = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

   /* Get necessary slots */
   hypre_ParCSRMatrix   *L, *U, *S, *ALU;
   HYPRE_Real           *D;
   HYPRE_Int            *u_end;

   /* u_end is the end position of the upper triangular part
     (if we need E and F implicitly), not used here */
   hypre_ILUSetupMILU0(A, NULL, NULL, n, n, &L, &D, &U, &S, &u_end, modified);
   hypre_TFree(u_end, HYPRE_MEMORY_HOST);

   /* TODO (VPM): Change this function's name */
   hypre_ILUSetupLDUtoCusparse(L, D, U, &ALU);

   /* Free memory */
   hypre_ParCSRMatrixDestroy(L);
   hypre_TFree(D, HYPRE_MEMORY_DEVICE);
   hypre_ParCSRMatrixDestroy(U);

   *ALUp = ALU;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetupRAPILU0Device
 *
 * Modified ILU(0) with RAP like solve
 * A = input matrix
 *
 * TODO (VPM): Move this function to par_setup_device.c?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupRAPILU0Device(hypre_ParCSRMatrix  *A,
                            HYPRE_Int           *perm,
                            HYPRE_Int            n,
                            HYPRE_Int            nLU,
                            hypre_ParCSRMatrix **Apermptr,
                            hypre_ParCSRMatrix **matSptr,
                            hypre_CSRMatrix    **ALUptr,
                            hypre_CSRMatrix    **BLUptr,
                            hypre_CSRMatrix    **CLUptr,
                            hypre_CSRMatrix    **Eptr,
                            hypre_CSRMatrix    **Fptr,
                            HYPRE_Int            test_opt)
{
   MPI_Comm             comm          = hypre_ParCSRMatrixComm(A);
   HYPRE_Int           *rperm         = NULL;
   HYPRE_Int            m             = n - nLU;
   HYPRE_Int            i;
   HYPRE_Int            num_procs,  my_id;

   /* Matrix Structure */
   hypre_ParCSRMatrix   *Apq, *ALU, *ALUm, *S;
   hypre_CSRMatrix      *Amd, *Ad, *SLU, *Apq_diag;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   rperm = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);

   for (i = 0; i < n; i++)
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
   Ad  = hypre_ParCSRMatrixDiag(ALU);
   switch (test_opt)
   {
      case 1:
      {
         /* RAP where we save E and F */
         Apq_diag = hypre_ParCSRMatrixDiag(Apq);
#if defined(HYPRE_USING_GPU)
         hypre_CSRMatrixSortRow(Apq_diag);
#endif
         hypre_ParILUExtractEBFC(Apq_diag, nLU, &dB, &dS, Eptr, Fptr);

         /* get modified ILU of B */
         hypre_ParILUExtractEBFC(Amd, nLU, BLUptr, &SLU, &dE, &dF);
         hypre_CSRMatrixDestroy(dB);
         hypre_CSRMatrixDestroy(dS);
         hypre_CSRMatrixDestroy(dE);
         hypre_CSRMatrixDestroy(dF);

         break;
      }

      case 2:
      {
         /* C-EB^{-1}F where we save EU^{-1}, L^{-1}F as sparse matrices */
         Apq_diag = hypre_ParCSRMatrixDiag(Apq);
#if defined(HYPRE_USING_GPU)
         hypre_CSRMatrixSortRow(Apq_diag);
#endif
         hypre_ParILUExtractEBFC(Apq_diag, nLU, &dB, CLUptr, &dE, &dF);

         /* get modified ILU of B */
         hypre_ParILUExtractEBFC(Amd, nLU, BLUptr, &SLU, Eptr, Fptr);
         hypre_CSRMatrixDestroy(dB);
         hypre_CSRMatrixDestroy(dE);
         hypre_CSRMatrixDestroy(dF);

         break;
      }

      case 3:
      {
         /* C-EB^{-1}F where we save E and F */
         Apq_diag = hypre_ParCSRMatrixDiag(Apq);
#if defined(HYPRE_USING_GPU)
         hypre_CSRMatrixSortRow(Apq_diag);
#endif
         hypre_ParILUExtractEBFC(Apq_diag, nLU, &dB, CLUptr, Eptr, Fptr);

         /* get modified ILU of B */
         hypre_ParILUExtractEBFC(Amd, nLU, BLUptr, &SLU, &dE, &dF);
         hypre_CSRMatrixDestroy(dB);
         hypre_CSRMatrixDestroy(dE);
         hypre_CSRMatrixDestroy(dF);

         break;
      }

      case 4:
      {
         /* RAP where we save EU^{-1}, L^{-1}F as sparse matrices */
         hypre_ParILUExtractEBFC(Ad, nLU, BLUptr, &SLU, Eptr, Fptr);

         break;
      }

      case 0:
      default:
      {
         /* RAP where we save EU^{-1}, L^{-1}F as sparse matrices */
         hypre_ParILUExtractEBFC(Amd, nLU, BLUptr, &SLU, Eptr, Fptr);

         break;
      }
   }

   *ALUptr = hypre_ParCSRMatrixDiag(ALU);

   hypre_ParCSRMatrixDiag(ALU) = NULL; /* not a good practice to manipulate parcsr's csr */
   hypre_ParCSRMatrixDestroy(ALU);
   hypre_ParCSRMatrixDestroy(ALUm);

   /* start forming parCSR matrix S */

   HYPRE_BigInt   S_total_rows, S_row_starts[2];
   HYPRE_BigInt   big_m = (HYPRE_BigInt)m;
   hypre_MPI_Allreduce(&big_m, &S_total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

   if (S_total_rows > 0)
   {
      {
         HYPRE_BigInt global_start;
         hypre_MPI_Scan(&big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
         S_row_starts[0] = global_start - big_m;
         S_row_starts[1] = global_start;
      }

      S = hypre_ParCSRMatrixCreate( hypre_ParCSRMatrixComm(A),
                                    S_total_rows,
                                    S_total_rows,
                                    S_row_starts,
                                    S_row_starts,
                                    0,
                                    0,
                                    0);

      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(S));
      hypre_ParCSRMatrixDiag(S) = SLU;
   }
   else
   {
      S = NULL;
      hypre_CSRMatrixDestroy(SLU);
   }

   *matSptr  = S;
   *Apermptr = Apq;

   hypre_TFree(rperm, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetupRAPILU0
 *
 * Modified ILU(0) with RAP like solve
 *
 * A = input matrix
 * Not explicitly forming the matrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupRAPILU0(hypre_ParCSRMatrix  *A,
                      HYPRE_Int           *perm,
                      HYPRE_Int            n,
                      HYPRE_Int            nLU,
                      hypre_ParCSRMatrix **Lptr,
                      HYPRE_Real         **Dptr,
                      hypre_ParCSRMatrix **Uptr,
                      hypre_ParCSRMatrix **mLptr,
                      HYPRE_Real         **mDptr,
                      hypre_ParCSRMatrix **mUptr,
                      HYPRE_Int          **u_end)
{
   hypre_ParCSRMatrix   *S_temp = NULL;
   HYPRE_Int            *u_temp = NULL;

   HYPRE_Int            *u_end_array;

   hypre_CSRMatrix      *L_diag, *U_diag;
   HYPRE_Int            *L_diag_i, *U_diag_i;
   HYPRE_Int            *L_diag_j, *U_diag_j;
   HYPRE_Complex        *L_diag_data, *U_diag_data;

   hypre_CSRMatrix      *mL_diag, *mU_diag;
   HYPRE_Int            *mL_diag_i, *mU_diag_i;
   HYPRE_Int            *mL_diag_j, *mU_diag_j;
   HYPRE_Complex        *mL_diag_data, *mU_diag_data;

   HYPRE_Int            i;

   /* Standard ILU0 factorization */
   hypre_ILUSetupMILU0(A, perm, perm, n, n, Lptr, Dptr, Uptr, &S_temp, &u_temp, 0);

   /* Free memory */
   hypre_ParCSRMatrixDestroy(S_temp);
   hypre_TFree(u_temp, HYPRE_MEMORY_HOST);

   /* Modified ILU0 factorization */
   hypre_ILUSetupMILU0(A, perm, perm, n, n, mLptr, mDptr, mUptr, &S_temp, &u_temp, 1);

   /* Free memory */
   hypre_ParCSRMatrixDestroy(S_temp);
   hypre_TFree(u_temp, HYPRE_MEMORY_HOST);

   /* Pointer to the start location */
   u_end_array  = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   U_diag       = hypre_ParCSRMatrixDiag(*Uptr);
   U_diag_i     = hypre_CSRMatrixI(U_diag);
   U_diag_j     = hypre_CSRMatrixJ(U_diag);
   U_diag_data  = hypre_CSRMatrixData(U_diag);
   mU_diag      = hypre_ParCSRMatrixDiag(*mUptr);
   mU_diag_i    = hypre_CSRMatrixI(mU_diag);
   mU_diag_j    = hypre_CSRMatrixJ(mU_diag);
   mU_diag_data = hypre_CSRMatrixData(mU_diag);

   /* first sort the Upper part U */
   for (i = 0; i < nLU; i++)
   {
      hypre_qsort1(U_diag_j, U_diag_data, U_diag_i[i], U_diag_i[i + 1] - 1);
      hypre_qsort1(mU_diag_j, mU_diag_data, mU_diag_i[i], mU_diag_i[i + 1] - 1);
      hypre_BinarySearch2(U_diag_j, nLU, U_diag_i[i], U_diag_i[i + 1] - 1, u_end_array + i);
   }

   L_diag       = hypre_ParCSRMatrixDiag(*Lptr);
   L_diag_i     = hypre_CSRMatrixI(L_diag);
   L_diag_j     = hypre_CSRMatrixJ(L_diag);
   L_diag_data  = hypre_CSRMatrixData(L_diag);
   mL_diag      = hypre_ParCSRMatrixDiag(*mLptr);
   mL_diag_i    = hypre_CSRMatrixI(mL_diag);
   mL_diag_j    = hypre_CSRMatrixJ(mL_diag);
   mL_diag_data = hypre_CSRMatrixData(mL_diag);

   /* now sort the Lower part L */
   for (i = nLU; i < n; i++)
   {
      hypre_qsort1(L_diag_j, L_diag_data, L_diag_i[i], L_diag_i[i + 1] - 1);
      hypre_qsort1(mL_diag_j, mL_diag_data, mL_diag_i[i], mL_diag_i[i + 1] - 1);
      hypre_BinarySearch2(L_diag_j, nLU, L_diag_i[i], L_diag_i[i + 1] - 1, u_end_array + i);
   }

   *u_end = u_end_array;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetupILU0
 *
 * Setup ILU(0)
 *
 * A = input matrix
 * perm = permutation array indicating ordering of rows.
 *        Perm could come from a CF_marker array or a reordering routine.
 *         When set to NULL, identity permutation is used.
 * qperm = permutation array indicating ordering of columns.
 *         When set to NULL, identity permutation is used.
 * nI = number of interial unknowns
 * nLU = size of incomplete factorization, nLU should obey nLU <= nI.
 *       Schur complement is formed if nLU < n
 * Lptr, Dptr, Uptr, Sptr = L, D, U, S factors.
 * will form global Schur Matrix if nLU < n
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupILU0(hypre_ParCSRMatrix  *A,
                   HYPRE_Int           *perm,
                   HYPRE_Int           *qperm,
                   HYPRE_Int            nLU,
                   HYPRE_Int            nI,
                   hypre_ParCSRMatrix **Lptr,
                   HYPRE_Real         **Dptr,
                   hypre_ParCSRMatrix **Uptr,
                   hypre_ParCSRMatrix **Sptr,
                   HYPRE_Int          **u_end)
{
   return hypre_ILUSetupMILU0(A, perm, qperm, nLU, nI, Lptr, Dptr, Uptr, Sptr, u_end, 0);
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetupILU0
 *
 * Setup modified ILU(0)
 *
 * A = input matrix
 * perm = permutation array indicating ordering of rows.
 *        Perm could come from a CF_marker array or a reordering routine.
 *        When set to NULL, indentity permutation is used.
 * qperm = permutation array indicating ordering of columns.
 *         When set to NULL, identity permutation is used.
 * nI = number of interior unknowns
 * nLU = size of incomplete factorization, nLU should obey nLU <= nI.
 *       Schur complement is formed if nLU < n
 * Lptr, Dptr, Uptr, Sptr = L, D, U, S factors.
 * modified set to 0 to use classical ILU
 * will form global Schur Matrix if nLU < n
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupMILU0(hypre_ParCSRMatrix  *A,
                    HYPRE_Int           *permp,
                    HYPRE_Int           *qpermp,
                    HYPRE_Int            nLU,
                    HYPRE_Int            nI,
                    hypre_ParCSRMatrix **Lptr,
                    HYPRE_Real         **Dptr,
                    hypre_ParCSRMatrix **Uptr,
                    hypre_ParCSRMatrix **Sptr,
                    HYPRE_Int          **u_end,
                    HYPRE_Int            modified)
{
   HYPRE_Int                i, ii, j, k, k1, k2, k3, ctrU, ctrL, ctrS;
   HYPRE_Int                lenl, lenu, jpiv, col, jpos;
   HYPRE_Int                *iw, *iL, *iU;
   HYPRE_Real               dd, t, dpiv, lxu, *wU, *wL;
   HYPRE_Real               drop;

   /* communication stuffs for S */
   MPI_Comm                  comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int                 S_offd_nnz, S_offd_ncols;
   hypre_ParCSRCommPkg      *comm_pkg;
   hypre_ParCSRCommHandle   *comm_handle;
   HYPRE_Int                 num_sends, begin, end;
   HYPRE_BigInt             *send_buf        = NULL;
   HYPRE_Int                 num_procs, my_id;

   /* data objects for A */
   hypre_CSRMatrix          *A_diag          = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix          *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real               *A_diag_data     = hypre_CSRMatrixData(A_diag);
   HYPRE_Int                *A_diag_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Int                *A_diag_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real               *A_offd_data     = hypre_CSRMatrixData(A_offd);
   HYPRE_Int                *A_offd_i        = hypre_CSRMatrixI(A_offd);
   HYPRE_Int                *A_offd_j        = hypre_CSRMatrixJ(A_offd);
   HYPRE_MemoryLocation      memory_location = hypre_ParCSRMatrixMemoryLocation(A);

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
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
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
      hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: nLU out of range.\n");
   }
   if (e < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: nLU should not exceed nI.\n");
   }

   /* Allocate memory for u_end array */
   u_end_array    = hypre_TAlloc(HYPRE_Int, nLU, HYPRE_MEMORY_HOST);

   /* Allocate memory for L,D,U,S factors */
   if (n > 0)
   {
      initial_alloc  = (HYPRE_Int)(nLU + hypre_ceil((nnz_A / 2.0) * nLU / n));
      capacity_S     = (HYPRE_Int)(m + hypre_ceil((nnz_A / 2.0) * m / n));
   }
   capacity_L     = initial_alloc;
   capacity_U     = initial_alloc;

   D_data         = hypre_TAlloc(HYPRE_Real, n, memory_location);
   L_diag_i       = hypre_TAlloc(HYPRE_Int, n + 1, memory_location);
   L_diag_j       = hypre_TAlloc(HYPRE_Int, capacity_L, memory_location);
   L_diag_data    = hypre_TAlloc(HYPRE_Real, capacity_L, memory_location);
   U_diag_i       = hypre_TAlloc(HYPRE_Int, n + 1, memory_location);
   U_diag_j       = hypre_TAlloc(HYPRE_Int, capacity_U, memory_location);
   U_diag_data    = hypre_TAlloc(HYPRE_Real, capacity_U, memory_location);
   S_diag_i       = hypre_TAlloc(HYPRE_Int, m + 1, memory_location);
   S_diag_j       = hypre_TAlloc(HYPRE_Int, capacity_S, memory_location);
   S_diag_data    = hypre_TAlloc(HYPRE_Real, capacity_S, memory_location);

   /* allocate working arrays */
   iw             = hypre_TAlloc(HYPRE_Int, 3 * n, HYPRE_MEMORY_HOST);
   iL             = iw + n;
   rperm          = iw + 2 * n;
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
      perm = hypre_TAlloc(HYPRE_Int, n, memory_location);
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
      qperm = hypre_TAlloc(HYPRE_Int, n, memory_location);
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
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
      // track the drop
      drop = 0.0;

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL + ii;
      wU = wL + ii;
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
            dd = t;
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
      hypre_qsort3ir(iL, wL, iw, 0, (lenl - 1));
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
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv + 1]; k++)
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
         while ((ctrL + lenl) > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int,
                                         capacity_L, memory_location);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real,
                                            capacity_L, memory_location);
         }
         hypre_TMemcpy(&L_diag_j[ctrL], iL, HYPRE_Int, lenl,
                       memory_location, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(&L_diag_data[ctrL], wL, HYPRE_Real, lenl,
                       memory_location, HYPRE_MEMORY_HOST);
      }
      L_diag_i[ii + 1] = (ctrL += lenl);

      /* diagonal part (we store the inverse) */
      if (hypre_abs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;
      }
      D_data[ii] = 1. / dd;

      /* U part */
      /* Check that memory is sufficient */
      if (lenu > 0)
      {
         while ((ctrU + lenu) > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = (HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp, HYPRE_Int,
                                         capacity_U, memory_location);
            U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp, HYPRE_Real,
                                            capacity_U, memory_location);
         }
         hypre_TMemcpy(&U_diag_j[ctrU], iU, HYPRE_Int, lenu,
                       memory_location, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(&U_diag_data[ctrU], wU, HYPRE_Real, lenu,
                       memory_location, HYPRE_MEMORY_HOST);
      }
      U_diag_i[ii + 1] = (ctrU += lenu);

      /* check and build u_end array */
      if (m > 0)
      {
         hypre_qsort1(U_diag_j, U_diag_data, U_diag_i[ii], U_diag_i[ii + 1] - 1);
         hypre_BinarySearch2(U_diag_j, nLU, U_diag_i[ii], U_diag_i[ii + 1] - 1, u_end_array + ii);
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
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
      drop = 0.0;

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL + nLU + 1;
      wU = wL + nLU + 1;
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
            dd = t;
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
      hypre_qsort3ir(iL, wL, iw, 0, (lenl - 1));
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
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv + 1]; k++)
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
         while ((ctrL + lenl) > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int,
                                         capacity_L, memory_location);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real,
                                            capacity_L, memory_location);
         }
         hypre_TMemcpy(&L_diag_j[ctrL], iL, HYPRE_Int, lenl,
                       memory_location, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(&L_diag_data[ctrL], wL, HYPRE_Real, lenl,
                       memory_location, HYPRE_MEMORY_HOST);
      }
      L_diag_i[ii + 1] = (ctrL += lenl);

      /* S part */
      /* Check that memory is sufficient */
      while ((ctrS + lenu + 1) > capacity_S)
      {
         HYPRE_Int tmp = capacity_S;
         capacity_S = (HYPRE_Int)(capacity_S * EXPAND_FACT + 1);
         S_diag_j = hypre_TReAlloc_v2(S_diag_j, HYPRE_Int, tmp, HYPRE_Int,
                                      capacity_S, memory_location);
         S_diag_data = hypre_TReAlloc_v2(S_diag_data, HYPRE_Real, tmp, HYPRE_Real,
                                         capacity_S, memory_location);
      }
      /* remember S in under a new index system! */
      S_diag_j[ctrS] = ii - nLU;
      S_diag_data[ctrS] = dd;
      for (j = 0; j < lenu; j++)
      {
         S_diag_j[ctrS + 1 + j] = iU[j] - nLU;
      }
      //hypre_TMemcpy(S_diag_data+ctrS+1, wU, HYPRE_Real, lenu, memory_location, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(S_diag_data + ctrS + 1, wU, HYPRE_Real, lenu,
                    memory_location, HYPRE_MEMORY_HOST);
      S_diag_i[ii - nLU + 1] = ctrS += (lenu + 1);
   }
   /* Assemble LDUS matrices */
   /* zero out unfactored rows for U and D */
   for (k = nLU; k < n; k++)
   {
      U_diag_i[k + 1] = ctrU;
      D_data[k] = 1.;
   }

   /* First create Schur complement if necessary
    * Check if we need to create Schur complement
    */
   HYPRE_BigInt big_m = (HYPRE_BigInt)m;
   hypre_MPI_Allreduce(&big_m, &total_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

   /* only form when total_rows > 0 */
   if (total_rows > 0)
   {
      /* now create S */
      /* need to get new column start */
      {
         HYPRE_BigInt global_start;
         hypre_MPI_Scan(&big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
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
      S_offd_i = hypre_TAlloc(HYPRE_Int, m + 1, memory_location);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, memory_location);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, memory_location);
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
         k2 = A_offd_i[col + 1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i + 1 + e] = k3;
      }

      /* give I, J, DATA to S_offd */
      hypre_CSRMatrixI(S_offd) = S_offd_i;
      hypre_CSRMatrixJ(S_offd) = S_offd_j;
      hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */

      /* get total num of send */
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
      end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      send_buf = hypre_TAlloc(HYPRE_BigInt, end - begin, HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for (i = begin; i < end; i++)
      {
         send_buf[i - begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)] -
                               nLU + col_starts[0];
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
      hypre_TFree(L_diag_j, memory_location);
      hypre_TFree(L_diag_data, memory_location);
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
      hypre_TFree(U_diag_j, memory_location);
      hypre_TFree(U_diag_data, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) ctrU;
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;
   /* free memory */
   hypre_TFree(wL, HYPRE_MEMORY_HOST);
   hypre_TFree(iw, HYPRE_MEMORY_HOST);
   if (!matS)
   {
      /* we allocate some memory for S, need to free if unused */
      hypre_TFree(S_diag_i, memory_location);
   }

   if (!permp)
   {
      hypre_TFree(perm, memory_location);
   }
   if (!qpermp)
   {
      hypre_TFree(qperm, memory_location);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;
   *Sptr = matS;
   *u_end = u_end_array;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetupILUKSymbolic
 *
 * Setup ILU(k) symbolic factorization
 *
 * n = total rows of input
 * lfil = level of fill-in, the k in ILU(k)
 * perm = permutation array indicating ordering of factorization.
 * rperm = reverse permutation array, used here to avoid duplicate memory allocation
 * iw = working array, used here to avoid duplicate memory allocation
 * nLU = size of computed LDU factorization.
 * A/L/U/S_diag_i = the I slot of A, L, U and S
 * A/L/U/S_diag_j = the J slot of A, L, U and S
 *
 * Will form global Schur Matrix if nLU < n
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupILUKSymbolic(HYPRE_Int   n,
                           HYPRE_Int  *A_diag_i,
                           HYPRE_Int  *A_diag_j,
                           HYPRE_Int   lfil,
                           HYPRE_Int  *perm,
                           HYPRE_Int  *rperm,
                           HYPRE_Int  *iw,
                           HYPRE_Int   nLU,
                           HYPRE_Int  *L_diag_i,
                           HYPRE_Int  *U_diag_i,
                           HYPRE_Int  *S_diag_i,
                           HYPRE_Int **L_diag_j,
                           HYPRE_Int **U_diag_j,
                           HYPRE_Int **S_diag_j,
                           HYPRE_Int **u_end)
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
   HYPRE_Int         capacity_S = 0;
   HYPRE_Int         initial_alloc = 0;
   HYPRE_Int         nnz_A;
   HYPRE_MemoryLocation memory_location;

   /* Get default memory location */
   HYPRE_GetMemoryLocation(&memory_location);

   /* set iL and iLev to right place in iw array */
   iL                = iw + n;
   iLev              = iw + 2 * n;

   /* setup initial memory used */
   nnz_A             = A_diag_i[n];
   if (n > 0)
   {
      initial_alloc     = (HYPRE_Int)(nLU + hypre_ceil((nnz_A / 2.0) * nLU / n));
   }
   capacity_L        = initial_alloc;
   capacity_U        = initial_alloc;

   /* allocate other memory for L and U struct */
   temp_L_diag_j     = hypre_CTAlloc(HYPRE_Int, capacity_L, memory_location);
   temp_U_diag_j     = hypre_CTAlloc(HYPRE_Int, capacity_U, memory_location);

   if (m > 0)
   {
      capacity_S     = (HYPRE_Int)(m + hypre_ceil(nnz_A / 2.0 * m / n));
      temp_S_diag_j  = hypre_CTAlloc(HYPRE_Int, capacity_S, memory_location);
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
      lena = A_diag_i[i + 1];
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
            hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
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
         hypre_ILUMinHeapRemoveIIIi(iL, iLev, iw, lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k] = -1;
         hypre_swap2i(iL, iLev, ii - lenl, lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k + 1];
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
                  hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
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
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            temp_L_diag_j = hypre_TReAlloc_v2(temp_L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L,
                                              memory_location);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL + j] = iL[ii - j - 1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii + 1] = U_diag_i[ii] + k;
      if (k > 0)
      {
         /* check if memory is enough */
         while (ctrU + k > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = (HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            temp_U_diag_j = hypre_TReAlloc_v2(temp_U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U,
                                              memory_location);
            u_levels = hypre_TReAlloc_v2(u_levels, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
         }
         //hypre_TMemcpy(temp_U_diag_j+ctrU,iL+ii,HYPRE_Int,k,memory_location,HYPRE_MEMORY_HOST);
         hypre_TMemcpy(temp_U_diag_j + ctrU, iL + ii, HYPRE_Int, k,
                       memory_location, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(u_levels + ctrU, iLev + ii, HYPRE_Int, k,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         ctrU += k;
      }
      if (m > 0)
      {
         hypre_qsort2i(temp_U_diag_j, u_levels, U_diag_i[ii], U_diag_i[ii + 1] - 1);
         hypre_BinarySearch2(temp_U_diag_j, nLU, U_diag_i[ii], U_diag_i[ii + 1] - 1, u_end_array + ii);
      }
      else
      {
         /* Everything is in U */
         u_end_array[ii] = ctrU;
      }

      /* reset iw */
      for (j = ii; j < lenu; j++)
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
      lena = A_diag_i[i + 1];
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
            hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
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
         hypre_ILUMinHeapRemoveIIIi(iL, iLev, iw, lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k] = -1;
         hypre_swap2i(iL, iLev, nLU - lenl, lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k + 1];
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
                  hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
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
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            temp_L_diag_j = hypre_TReAlloc_v2(temp_L_diag_j, HYPRE_Int, tmp, HYPRE_Int,
                                              capacity_L, memory_location);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL + j] = iL[nLU - j - 1];
         }
         ctrL += lenl;
      }
      k = lenu - nLU + 1;
      /* check if memory is enough */
      while (ctrS + k > capacity_S)
      {
         HYPRE_Int tmp = capacity_S;
         capacity_S = (HYPRE_Int)(capacity_S * EXPAND_FACT + 1);
         temp_S_diag_j = hypre_TReAlloc_v2(temp_S_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_S,
                                           memory_location);
      }
      temp_S_diag_j[ctrS] = ii;/* must have diagonal */
      //hypre_TMemcpy(temp_S_diag_j+ctrS+1,iL+nLU,HYPRE_Int,k-1,memory_location,HYPRE_MEMORY_HOST);
      hypre_TMemcpy(temp_S_diag_j + ctrS + 1, iL + nLU, HYPRE_Int, k - 1,
                    memory_location, HYPRE_MEMORY_HOST);
      ctrS += k;
      S_diag_i[ii - nLU + 1] = ctrS;

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
      U_diag_i[k + 1] = U_diag_i[nLU];
   }
   /*
    * 4: Finishing up and free memory
    */
   hypre_TFree(u_levels, HYPRE_MEMORY_HOST);

   *L_diag_j = temp_L_diag_j;
   *U_diag_j = temp_U_diag_j;
   *S_diag_j = temp_S_diag_j;
   *u_end = u_end_array;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetupILUK
 *
 * Setup ILU(k) numeric factorization
 *
 * A: input matrix
 * lfil: level of fill-in, the k in ILU(k)
 * permp: permutation array indicating ordering of factorization.
 *        Perm could come from a CF_marker array or a reordering routine.
 * qpermp: column permutation array.
 * nLU: size of computed LDU factorization.
 * nI: number of interial unknowns, nI should obey nI >= nLU
 * Lptr, Dptr, Uptr: L, D, U factors.
 * Sprt: Schur Complement, if no Schur Complement, it will be set to NULL
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupILUK(hypre_ParCSRMatrix  *A,
                   HYPRE_Int            lfil,
                   HYPRE_Int           *permp,
                   HYPRE_Int           *qpermp,
                   HYPRE_Int            nLU,
                   HYPRE_Int            nI,
                   hypre_ParCSRMatrix **Lptr,
                   HYPRE_Real         **Dptr,
                   hypre_ParCSRMatrix **Uptr,
                   hypre_ParCSRMatrix **Sptr,
                   HYPRE_Int          **u_end)
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
   HYPRE_Int               num_procs,  my_id;

   /* data objects for A */
   hypre_CSRMatrix         *A_diag        = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix         *A_offd        = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real              *A_diag_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int               *A_diag_i      = hypre_CSRMatrixI(A_diag);
   HYPRE_Int               *A_diag_j      = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real              *A_offd_data   = hypre_CSRMatrixData(A_offd);
   HYPRE_Int               *A_offd_i      = hypre_CSRMatrixI(A_offd);
   HYPRE_Int               *A_offd_j      = hypre_CSRMatrixJ(A_offd);
   HYPRE_MemoryLocation     memory_location = hypre_ParCSRMatrixMemoryLocation(A);

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
      hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: nLU out of range.\n");
   }
   m = n - nLU;
   e = nI - nLU;
   m_e = n - nI;
   if (e < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: nLU should not exceed nI.\n");
   }

   /* Init I array anyway. S's might be freed later */
   D_data = hypre_CTAlloc(HYPRE_Real, n, memory_location);
   L_diag_i = hypre_CTAlloc(HYPRE_Int, (n + 1), memory_location);
   U_diag_i = hypre_CTAlloc(HYPRE_Int, (n + 1), memory_location);
   S_diag_i = hypre_CTAlloc(HYPRE_Int, (m + 1), memory_location);

   /* set Comm_Pkg if not yet built */
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
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
   iw = hypre_CTAlloc(HYPRE_Int, 4 * n, HYPRE_MEMORY_HOST);
   rperm = iw + 3 * n;
   L_diag_i[0] = U_diag_i[0] = S_diag_i[0] = 0;
   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
    */

   if (!permp)
   {
      perm = hypre_TAlloc(HYPRE_Int, n, memory_location);
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
      qperm = hypre_TAlloc(HYPRE_Int, n, memory_location);
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
      L_diag_data = hypre_CTAlloc(HYPRE_Real, L_diag_i[n], memory_location);
   }
   if (U_diag_i[n])
   {
      U_diag_data = hypre_CTAlloc(HYPRE_Real, U_diag_i[n], memory_location);
   }
   if (S_diag_i[m])
   {
      S_diag_data = hypre_CTAlloc(HYPRE_Real, S_diag_i[m], memory_location);
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
      kl = L_diag_i[ii + 1];
      ku = U_diag_i[ii + 1];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
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
         ku = U_diag_i[jpiv + 1];

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
               L_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else if (col == ii)
            {
               /* diag part */
               D_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else
            {
               /* U part */
               U_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
         }
      }
      /* reset working array */
      ku = U_diag_i[ii + 1];
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
      if (hypre_abs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / D_data[ii];
   }

   /* Now lower part for Schur complement */
   for (ii = nLU; ii < n; ii++)
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii + 1];
      ku = S_diag_i[ii - nLU + 1];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
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
         ku = U_diag_i[jpiv + 1];
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
               L_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else
            {
               /* S part */
               S_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
         }
      }
      /* reset working array */
      for (j = L_diag_i[ii]; j < kl ; j++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      ku = S_diag_i[ii - nLU + 1];
      for (j = S_diag_i[ii - nLU]; j < ku; j++)
      {
         col = S_diag_j[j];
         iw[col] = -1;
         /* remember to update index, S is smaller! */
         S_diag_j[j] -= nLU;
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
         hypre_MPI_Scan(&big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
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
      S_offd_i = hypre_TAlloc(HYPRE_Int, m + 1, memory_location);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, memory_location);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, memory_location);
      S_offd_colmap = hypre_CTAlloc(HYPRE_BigInt, S_offd_ncols, HYPRE_MEMORY_HOST);

      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for (i = 1; i <= e; i++)
      {
         S_offd_i[i + 1] = k3;
      }
      for (i = 0; i < m_e; i++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col + 1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i + e + 1] = k3;
      }

      /* give I, J, DATA to S_offd */
      hypre_CSRMatrixI(S_offd) = S_offd_i;
      hypre_CSRMatrixJ(S_offd) = S_offd_j;
      hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */

      /* get total num of send */
      HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
      HYPRE_Int end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      send_buf = hypre_TAlloc(HYPRE_BigInt, end - begin, HYPRE_MEMORY_HOST);

      /* copy new index into send_buf */
      for (i = begin; i < end; i++)
      {
         send_buf[i - begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)] - nLU + col_starts[0];
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
   if (L_diag_i[n] > 0)
   {
      hypre_CSRMatrixData(L_diag) = L_diag_data;
      hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we allocated some initial length, so free them */
      hypre_TFree(L_diag_j, memory_location);
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
      /* we allocated some initial length, so free them */
      hypre_TFree(U_diag_j, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (U_diag_i[n]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

   /* free */
   hypre_TFree(iw, HYPRE_MEMORY_HOST);
   if (!matS)
   {
      /* we allocate some memory for S, need to free if unused */
      hypre_TFree(S_diag_i, memory_location);
   }

   if (!permp)
   {
      hypre_TFree(perm, memory_location);
   }

   if (!qpermp)
   {
      hypre_TFree(qperm, memory_location);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;
   *Sptr = matS;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetupILUT
 *
 * Setup ILU(t) numeric factorization
 *
 * A: input matrix
 * lfil: maximum nnz per row in L and U
 * tol: droptol array in ILUT
 *    tol[0]: matrix B
 *    tol[1]: matrix E and F
 *    tol[2]: matrix S
 * perm: permutation array indicating ordering of factorization.
 *       Perm could come from a CF_marker array or a reordering routine.
 * qperm: permutation array for column
 * nLU: size of computed LDU factorization.
 *      If nLU < n, Schur complement will be formed
 * nI: number of interial unknowns. nLU should obey nLU <= nI.
 * Lptr, Dptr, Uptr: L, D, U factors.
 * Sptr: Schur complement
 *
 * Keep the largest lfil entries that is greater than some tol relative
 *    to the input tol and the norm of that row in both L and U
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupILUT(hypre_ParCSRMatrix  *A,
                   HYPRE_Int            lfil,
                   HYPRE_Real          *tol,
                   HYPRE_Int           *permp,
                   HYPRE_Int           *qpermp,
                   HYPRE_Int            nLU,
                   HYPRE_Int            nI,
                   hypre_ParCSRMatrix **Lptr,
                   HYPRE_Real         **Dptr,
                   hypre_ParCSRMatrix **Uptr,
                   hypre_ParCSRMatrix **Sptr,
                   HYPRE_Int          **u_end)
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
   HYPRE_Int                i, ii, j, k, k1, k2, k3, kl, ku, col, icol, lenl, lenu, lenhu, lenhlr,
                            lenhll, jpos, jrow;
   HYPRE_Real               inorm, itolb, itolef, itols, dpiv, lxu;
   HYPRE_Int                *iw, *iL;
   HYPRE_Real               *w;

   /* memory management */
   HYPRE_Int                ctrL;
   HYPRE_Int                ctrU;
   HYPRE_Int                initial_alloc = 0;
   HYPRE_Int                capacity_L;
   HYPRE_Int                capacity_U;
   HYPRE_Int                ctrS;
   HYPRE_Int                capacity_S = 0;
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
   HYPRE_MemoryLocation      memory_location = hypre_ParCSRMatrixMemoryLocation(A);

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
      hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: nLU out of range.\n");
   }
   m = n - nLU;
   e = nI - nLU;
   m_e = n - nI;
   if (e < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: nLU should not exceed nI.\n");
   }

   u_end_array = hypre_TAlloc(HYPRE_Int, nLU, HYPRE_MEMORY_HOST);

   /* start set up
    * setup communication stuffs first
    */
   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
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
      initial_alloc = (HYPRE_Int)(hypre_min(nLU + hypre_ceil((nnz_A / 2.0) * nLU / n),
                                            nLU * lfil));
   }
   capacity_L = initial_alloc;
   capacity_U = initial_alloc;

   D_data = hypre_CTAlloc(HYPRE_Real, n, memory_location);
   L_diag_i = hypre_CTAlloc(HYPRE_Int, (n + 1), memory_location);
   U_diag_i = hypre_CTAlloc(HYPRE_Int, (n + 1), memory_location);

   L_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_L, memory_location);
   U_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_U, memory_location);
   L_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_L, memory_location);
   U_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_U, memory_location);

   ctrL = ctrU = 0;

   ctrS = 0;
   S_diag_i = hypre_CTAlloc(HYPRE_Int, (m + 1), memory_location);
   S_diag_i[0] = 0;

   /* only setup S part when n > nLU */
   if (m > 0)
   {
      capacity_S = (HYPRE_Int)(hypre_min(m + hypre_ceil((nnz_A / 2.0) * m / n), m * lfil));
      S_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_S, memory_location);
      S_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_S, memory_location);
   }

   /* setting up working array */
   iw = hypre_CTAlloc(HYPRE_Int, 3 * n, HYPRE_MEMORY_HOST);
   iL = iw + n;
   w = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
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
   rperm = iw + 2 * n;

   if (!permp)
   {
      perm = hypre_TAlloc(HYPRE_Int, n, memory_location);
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
      qperm = hypre_TAlloc(HYPRE_Int, n, memory_location);
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
      k2 = A_diag_i[i + 1];
      kl = ii - 1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += hypre_abs(A_diag_data[j]);
      }
      if (inorm == .0)
      {
         hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: ILUT with zero row.\n");
      }
      inorm /= (HYPRE_Real)(k2 - k1);
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
            hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
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
         hypre_ILUMinHeapRemoveIRIi(iL, w, iw, lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;
         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         hypre_swap2(iL, w, lenhll, kl - lenhlr);
         lenhlr++;
         hypre_ILUMaxrHeapAddRabsI(w + kl, iL + kl, lenhlr);
         /* loop for elimination */
         ku = U_diag_i[jrow + 1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv * U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col < nLU && hypre_abs(lxu) < itolb) || (col >= nLU && hypre_abs(lxu) < itolef)))
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
                  hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
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

      if (hypre_abs(w[ii]) < MAT_TOL)
      {
         w[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / w[ii];
      iw[ii] = -1;

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int,
                                         capacity_L, memory_location);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real,
                                            capacity_L, memory_location);
         }
         ctrL += lenl;

         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            hypre_ILUMaxrHeapRemoveRabsI(w + kl, iL + kl, lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only U part
       */
      ku = lenu + ii;
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
         hypre_ILUMaxQSplitRabsI(w, iL, ii + 1, ii + lenhu, ii + lenu);
      }

      U_diag_i[ii + 1] = U_diag_i[ii] + lenhu;
      if (lenhu > 0)
      {
         /* test if memory is enough */
         while (ctrU + lenhu > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = (HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp, HYPRE_Int,
                                         capacity_U, memory_location);
            U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp, HYPRE_Real,
                                            capacity_U, memory_location);
         }
         ctrU += lenhu;
         /* copy large data in */
         for (j = U_diag_i[ii]; j < ctrU; j++)
         {
            jpos = ii + 1 + j - U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
      /* check and build u_end array */
      if (m > 0)
      {
         hypre_qsort1(U_diag_j, U_diag_data, U_diag_i[ii], U_diag_i[ii + 1] - 1);
         hypre_BinarySearch2(U_diag_j, nLU, U_diag_i[ii], U_diag_i[ii + 1] - 1, u_end_array + ii);
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
      k2 = A_diag_i[i + 1];
      kl = nLU - 1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += hypre_abs(A_diag_data[j]);
      }
      if (inorm == .0)
      {
         hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: ILUT with zero row.\n");
      }
      inorm /= (HYPRE_Real)(k2 - k1);
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
            hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
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
         hypre_ILUMinHeapRemoveIRIi(iL, w, iw, lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;
         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         hypre_swap2(iL, w, lenhll, kl - lenhlr);
         lenhlr++;
         hypre_ILUMaxrHeapAddRabsI(w + kl, iL + kl, lenhlr);
         /* loop for elimination */
         ku = U_diag_i[jrow + 1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv * U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col < nLU  && hypre_abs(lxu) < itolef) ||
                 (col >= nLU && hypre_abs(lxu) < itols )))
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
                  hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
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
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp, HYPRE_Int,
                                         capacity_L, memory_location);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp, HYPRE_Real,
                                            capacity_L, memory_location);
         }
         ctrL += lenl;

         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            hypre_ILUMaxrHeapRemoveRabsI(w + kl, iL + kl, lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only S part
       */
      ku = lenu + nLU;
      for (j = nLU; j <= ku; j++)
      {
         iw[iL[j]] = -1;
      }

      /* no dropping at this point of time for S */
      //lenhu = lenu < lfil ? lenu : lfil;
      lenhu = lenu;
      /* quick split, only sort the first small part of the array */
      hypre_ILUMaxQSplitRabsI(w, iL, nLU + 1, nLU + lenhu, nLU + lenu);
      /* we have diagonal in S anyway */
      /* test if memory is enough */
      while (ctrS + lenhu + 1 > capacity_S)
      {
         HYPRE_Int tmp = capacity_S;
         capacity_S = (HYPRE_Int)(capacity_S * EXPAND_FACT + 1);
         S_diag_j = hypre_TReAlloc_v2(S_diag_j, HYPRE_Int, tmp,
                                      HYPRE_Int, capacity_S, memory_location);
         S_diag_data = hypre_TReAlloc_v2(S_diag_data, HYPRE_Real, tmp,
                                         HYPRE_Real, capacity_S, memory_location);
      }

      ctrS += (lenhu + 1);
      S_diag_i[ii - nLU + 1] = ctrS;

      /* copy large data in, diagonal first */
      S_diag_j[S_diag_i[ii - nLU]] = iL[nLU] - nLU;
      S_diag_data[S_diag_i[ii - nLU]] = w[nLU];
      for (j = S_diag_i[ii - nLU] + 1; j < ctrS; j++)
      {
         jpos = nLU + j - S_diag_i[ii - nLU];
         S_diag_j[j] = iL[jpos] - nLU;
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
         hypre_MPI_Scan(&big_m, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
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
      S_offd_i = hypre_TAlloc(HYPRE_Int, m + 1, memory_location);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, memory_location);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, memory_location);
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
         k2 = A_offd_i[col + 1];
         for (j = k1; j < k2; j++)
         {
            S_offd_j[k3] = A_offd_j[j];
            S_offd_data[k3++] = A_offd_data[j];
         }
         S_offd_i[i + e + 1] = k3;
      }

      /* give I, J, DATA to S_offd */
      hypre_CSRMatrixI(S_offd) = S_offd_i;
      hypre_CSRMatrixJ(S_offd) = S_offd_j;
      hypre_CSRMatrixData(S_offd) = S_offd_data;

      /* now we need to update S_offd_colmap */

      /* get total num of send */
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
      end = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
      send_buf = hypre_TAlloc(HYPRE_BigInt, end - begin, HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for (i = begin; i < end; i++)
      {
         send_buf[i - begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, i)] - nLU + col_starts[0];
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
      U_diag_i[k + 1] = U_diag_i[nLU];
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
      hypre_TFree(L_diag_j, memory_location);
      hypre_TFree(L_diag_data, memory_location);
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
      hypre_TFree(U_diag_j, memory_location);
      hypre_TFree(U_diag_data, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (U_diag_i[n]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

   /* free working array */
   hypre_TFree(iw, HYPRE_MEMORY_HOST);
   hypre_TFree(w, HYPRE_MEMORY_HOST);

   if (!matS)
   {
      hypre_TFree(S_diag_i, memory_location);
   }

   if (!permp)
   {
      hypre_TFree(perm, memory_location);
   }

   if (!qpermp)
   {
      hypre_TFree(qperm, memory_location);
   }

   /* set matrix pointers */
   *Lptr = matL;
   *Dptr = D_data;
   *Uptr = matU;
   *Sptr = matS;
   *u_end = u_end_array;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_NSHSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_NSHSetup( void               *nsh_vdata,
                hypre_ParCSRMatrix *A,
                hypre_ParVector    *f,
                hypre_ParVector    *u )
{
   MPI_Comm             comm              = hypre_ParCSRMatrixComm(A);
   hypre_ParNSHData     *nsh_data         = (hypre_ParNSHData*) nsh_vdata;

   /* Pointers to NSH data */
   HYPRE_Int             logging          = hypre_ParNSHDataLogging(nsh_data);
   HYPRE_Int             print_level      = hypre_ParNSHDataPrintLevel(nsh_data);
   hypre_ParCSRMatrix   *matA             = hypre_ParNSHDataMatA(nsh_data);
   hypre_ParCSRMatrix   *matM             = hypre_ParNSHDataMatM(nsh_data);
   hypre_ParVector      *Utemp;
   hypre_ParVector      *Ftemp;
   hypre_ParVector      *F_array          = hypre_ParNSHDataF(nsh_data);
   hypre_ParVector      *U_array          = hypre_ParNSHDataU(nsh_data);
   hypre_ParVector      *residual         = hypre_ParNSHDataResidual(nsh_data);
   HYPRE_Real           *rel_res_norms    = hypre_ParNSHDataRelResNorms(nsh_data);

   /* Solver setting */
   HYPRE_Real           *droptol          = hypre_ParNSHDataDroptol(nsh_data);
   HYPRE_Real            mr_tol           = hypre_ParNSHDataMRTol(nsh_data);
   HYPRE_Int             mr_max_row_nnz   = hypre_ParNSHDataMRMaxRowNnz(nsh_data);
   HYPRE_Int             mr_max_iter      = hypre_ParNSHDataMRMaxIter(nsh_data);
   HYPRE_Int             mr_col_version   = hypre_ParNSHDataMRColVersion(nsh_data);
   HYPRE_Real            nsh_tol          = hypre_ParNSHDataNSHTol(nsh_data);
   HYPRE_Int             nsh_max_row_nnz  = hypre_ParNSHDataNSHMaxRowNnz(nsh_data);
   HYPRE_Int             nsh_max_iter     = hypre_ParNSHDataNSHMaxIter(nsh_data);
   HYPRE_Int             num_procs,  my_id;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* Free Previously allocated data, if any not destroyed */
   hypre_TFree(matM, HYPRE_MEMORY_HOST);
   hypre_TFree(hypre_ParNSHDataL1Norms(nsh_data), HYPRE_MEMORY_HOST);
   hypre_ParVectorDestroy(hypre_ParNSHDataUTemp(nsh_data));
   hypre_ParVectorDestroy(hypre_ParNSHDataFTemp(nsh_data));
   hypre_ParVectorDestroy(hypre_ParNSHDataResidual(nsh_data));
   hypre_TFree(hypre_ParNSHDataRelResNorms(nsh_data), HYPRE_MEMORY_HOST);

   matM = NULL;
   hypre_ParNSHDataL1Norms(nsh_data)     = NULL;
   hypre_ParNSHDataUTemp(nsh_data)       = NULL;
   hypre_ParNSHDataFTemp(nsh_data)       = NULL;
   hypre_ParNSHDataResidual(nsh_data)    = NULL;
   hypre_ParNSHDataRelResNorms(nsh_data) = NULL;

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

   /* Set matrix, solution and rhs pointers */
   matA = A;
   F_array = f;
   U_array = u;

   /* NSH compute approximate inverse, see par_ilu.c */
   hypre_ILUParCSRInverseNSH(matA, &matM, droptol, mr_tol, nsh_tol, HYPRE_REAL_MIN,
                             mr_max_row_nnz, nsh_max_row_nnz, mr_max_iter, nsh_max_iter,
                             mr_col_version, print_level);

   /* Set pointers to NSH data */
   hypre_ParNSHDataMatA(nsh_data) = matA;
   hypre_ParNSHDataF(nsh_data)    = F_array;
   hypre_ParNSHDataU(nsh_data)    = U_array;
   hypre_ParNSHDataMatM(nsh_data) = matM;

   /* Compute operator complexity */
   hypre_ParCSRMatrixSetDNumNonzeros(matA);
   hypre_ParCSRMatrixSetDNumNonzeros(matM);

   /* Compute complexity */
   hypre_ParNSHDataOperatorComplexity(nsh_data) = hypre_ParCSRMatrixDNumNonzeros(matM) /
                                                  hypre_ParCSRMatrixDNumNonzeros(matA);
   if (my_id == 0 && print_level > 0)
   {
      hypre_printf("NSH SETUP: operator complexity = %f  \n",
                   hypre_ParNSHDataOperatorComplexity(nsh_data));
   }

   if (logging > 1)
   {
      residual = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
                                       hypre_ParCSRMatrixGlobalNumRows(matA),
                                       hypre_ParCSRMatrixRowStarts(matA));
      hypre_ParVectorInitialize(residual);
      hypre_ParNSHDataResidual(nsh_data) = residual;
   }
   else
   {
      hypre_ParNSHDataResidual(nsh_data) = NULL;
   }

   rel_res_norms = hypre_CTAlloc(HYPRE_Real, hypre_ParNSHDataMaxIter(nsh_data),
                                 HYPRE_MEMORY_HOST);
   hypre_ParNSHDataRelResNorms(nsh_data) = rel_res_norms;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetupILU0RAS
 *
 * ILU(0) for RAS, has some external rows
 *
 * A = input matrix
 * perm = permutation array indicating ordering of factorization.
 *        Perm could come from a CF_marker array or a reordering routine.
 * nLU = size of computed LDU factorization.
 * Lptr, Dptr, Uptr, Sptr = L, D, U, S factors.
 * will form global Schur Matrix if nLU < n
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupILU0RAS(hypre_ParCSRMatrix  *A,
                      HYPRE_Int           *perm,
                      HYPRE_Int            nLU,
                      hypre_ParCSRMatrix **Lptr,
                      HYPRE_Real         **Dptr,
                      hypre_ParCSRMatrix **Uptr)
{
   /* communication stuffs for S */
   MPI_Comm                 comm          = hypre_ParCSRMatrixComm(A);
   HYPRE_Int                num_procs;
   hypre_ParCSRCommPkg      *comm_pkg;

   /* data objects for A */
   hypre_CSRMatrix          *A_diag       = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix          *A_offd       = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real               *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int                *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int                *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real               *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_Int                *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Int                *A_offd_j     = hypre_CSRMatrixJ(A_offd);
   HYPRE_MemoryLocation      memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   /* size of problem and external matrix */
   HYPRE_Int                n             =  hypre_CSRMatrixNumRows(A_diag);
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

   HYPRE_Int                i, ii, j, k, k1, k2, ctrU, ctrL, lenl, lenu, jpiv, col, jpos;
   HYPRE_Int                *iw, *iL, *iU;
   HYPRE_Real               dd, t, dpiv, lxu, *wU, *wL;

   /* start setup
    * get communication stuffs first
    */
   hypre_MPI_Comm_size(comm, &num_procs);
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);

   /* Setup if not yet built */
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   /* check for correctness */
   if (nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: nLU out of range.\n");
   }

   /* Allocate memory for L,D,U,S factors */
   if (n > 0)
   {
      initial_alloc = (HYPRE_Int)((n + ext) + hypre_ceil((nnz_A / 2.0) * total_rows / n));
   }
   capacity_L = initial_alloc;
   capacity_U = initial_alloc;

   D_data      = hypre_TAlloc(HYPRE_Real, total_rows, memory_location);
   L_diag_i    = hypre_TAlloc(HYPRE_Int, total_rows + 1, memory_location);
   L_diag_j    = hypre_TAlloc(HYPRE_Int, capacity_L, memory_location);
   L_diag_data = hypre_TAlloc(HYPRE_Real, capacity_L, memory_location);
   U_diag_i    = hypre_TAlloc(HYPRE_Int, total_rows + 1, memory_location);
   U_diag_j    = hypre_TAlloc(HYPRE_Int, capacity_U, memory_location);
   U_diag_data = hypre_TAlloc(HYPRE_Real, capacity_U, memory_location);

   /* allocate working arrays */
   iw          = hypre_TAlloc(HYPRE_Int, 4 * total_rows, HYPRE_MEMORY_HOST);
   iL          = iw + total_rows;
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
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL + ii;
      wU = wL + ii;
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
            dd = t;
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
      hypre_qsort3ir(iL, wL, iw, 0, (lenl - 1));
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
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv + 1]; k++)
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
      while ((ctrL + lenl) > capacity_L)
      {
         HYPRE_Int tmp = capacity_L;
         capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
         L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp,
                                      HYPRE_Int, capacity_L, memory_location);
         L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp,
                                         HYPRE_Real, capacity_L, memory_location);
      }
      hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, memory_location, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, memory_location, HYPRE_MEMORY_HOST);
      L_diag_i[ii + 1] = (ctrL += lenl);

      /* diagonal part (we store the inverse) */
      if (hypre_abs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;
      }
      D_data[ii] = 1. / dd;

      /* U part */
      /* Check that memory is sufficient */
      while ((ctrU + lenu) > capacity_U)
      {
         HYPRE_Int tmp = capacity_U;
         capacity_U = (HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
         U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp,
                                      HYPRE_Int, capacity_U, memory_location);
         U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp,
                                         HYPRE_Real, capacity_U, memory_location);
      }
      hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu, memory_location, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu, memory_location, HYPRE_MEMORY_HOST);
      U_diag_i[ii + 1] = (ctrU += lenu);
   }

   /*---------  Begin Factorization in lower part  ----
    * here we need to get off diagonals in
    */
   for (ii = nLU; ii < n; ii++)
   {
      // get row i
      i = perm[ii];
      // get extents of row i
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL + ii;
      wU = wL + ii;
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
            dd = t;
         }
      }

      /*------------------ sjcan offd*/
      k1 = A_offd_i[i];
      k2 = A_offd_i[i + 1];
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
      hypre_qsort3ir(iL, wL, iw, 0, (lenl - 1));
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
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv + 1]; k++)
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
      while ((ctrL + lenl) > capacity_L)
      {
         HYPRE_Int tmp = capacity_L;
         capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
         L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp,
                                      HYPRE_Int, capacity_L, memory_location);
         L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp,
                                         HYPRE_Real, capacity_L, memory_location);
      }
      hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, memory_location, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, memory_location, HYPRE_MEMORY_HOST);
      L_diag_i[ii + 1] = (ctrL += lenl);

      /* diagonal part (we store the inverse) */
      if (hypre_abs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;
      }
      D_data[ii] = 1. / dd;

      /* U part */
      /* Check that memory is sufficient */
      while ((ctrU + lenu) > capacity_U)
      {
         HYPRE_Int tmp = capacity_U;
         capacity_U = (HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
         U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp,
                                      HYPRE_Int, capacity_U, memory_location);
         U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp,
                                         HYPRE_Real, capacity_U, memory_location);
      }
      hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu, memory_location, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu, memory_location, HYPRE_MEMORY_HOST);
      U_diag_i[ii + 1] = (ctrU += lenu);
   }

   /*---------  Begin Factorization in external part  ----
    * here we need to get off diagonals in
    */
   for (ii = n ; ii < total_rows ; ii++)
   {
      // get row i
      i = ii - n;
      // get extents of row i
      k1 = E_i[i];
      k2 = E_i[i + 1];

      /*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL + ii;
      wU = wL + ii;
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
            dd = t;
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
      hypre_qsort3ir(iL, wL, iw, 0, (lenl - 1));
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
         for (k = U_diag_i[jpiv]; k < U_diag_i[jpiv + 1]; k++)
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
      while ((ctrL + lenl) > capacity_L)
      {
         HYPRE_Int tmp = capacity_L;
         capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
         L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp,
                                      HYPRE_Int, capacity_L, memory_location);
         L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp,
                                         HYPRE_Real, capacity_L, memory_location);
      }
      hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, memory_location, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, memory_location, HYPRE_MEMORY_HOST);
      L_diag_i[ii + 1] = (ctrL += lenl);

      /* diagonal part (we store the inverse) */
      if (hypre_abs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;
      }
      D_data[ii] = 1. / dd;

      /* U part */
      /* Check that memory is sufficient */
      while ((ctrU + lenu) > capacity_U)
      {
         HYPRE_Int tmp = capacity_U;
         capacity_U = (HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
         U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp,
                                      HYPRE_Int, capacity_U, memory_location);
         U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp,
                                         HYPRE_Real, capacity_U, memory_location);
      }
      hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu, memory_location, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu, memory_location, HYPRE_MEMORY_HOST);
      U_diag_i[ii + 1] = (ctrU += lenu);
   }

   HYPRE_BigInt big_total_rows = (HYPRE_BigInt)total_rows;
   hypre_MPI_Allreduce(&big_total_rows, &global_num_rows, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

   /* need to get new column start */
   {
      HYPRE_BigInt global_start;
      hypre_MPI_Scan(&big_total_rows, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
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
      hypre_TFree(L_diag_j, memory_location);
      hypre_TFree(L_diag_data, memory_location);
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
      hypre_TFree(U_diag_j, memory_location);
      hypre_TFree(U_diag_data, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) ctrU;
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;
   /* free memory */
   hypre_TFree(wL, HYPRE_MEMORY_HOST);
   hypre_TFree(iw, HYPRE_MEMORY_HOST);

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

/*--------------------------------------------------------------------------
 * hypre_ILUSetupILUKRASSymbolic
 *
 * ILU(k) symbolic factorization for RAS
 *
 * n = total rows of input
 * lfil = level of fill-in, the k in ILU(k)
 * perm = permutation array indicating ordering of factorization.
 * rperm = reverse permutation array, used here to avoid duplicate memory allocation
 * iw = working array, used here to avoid duplicate memory allocation
 * nLU = size of computed LDU factorization.
 * A/L/U/E_i = the I slot of A, L, U and E
 * A/L/U/E_j = the J slot of A, L, U and E
 *
 * Will form global Schur Matrix if nLU < n
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupILUKRASSymbolic(HYPRE_Int   n,
                              HYPRE_Int  *A_diag_i,
                              HYPRE_Int  *A_diag_j,
                              HYPRE_Int  *A_offd_i,
                              HYPRE_Int  *A_offd_j,
                              HYPRE_Int  *E_i,
                              HYPRE_Int  *E_j,
                              HYPRE_Int   ext,
                              HYPRE_Int   lfil,
                              HYPRE_Int  *perm,
                              HYPRE_Int  *rperm,
                              HYPRE_Int  *iw,
                              HYPRE_Int   nLU,
                              HYPRE_Int  *L_diag_i,
                              HYPRE_Int  *U_diag_i,
                              HYPRE_Int **L_diag_j,
                              HYPRE_Int **U_diag_j)
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
   HYPRE_MemoryLocation memory_location;

   /* Get default memory location */
   HYPRE_GetMemoryLocation(&memory_location);

   /* set iL and iLev to right place in iw array */
   iL             = iw + total_rows;
   iLev           = iw + 2 * total_rows;

   /* setup initial memory used */
   nnz_A          = A_diag_i[n];
   if (n > 0)
   {
      initial_alloc  = (HYPRE_Int)((n + ext) + hypre_ceil((nnz_A / 2.0) * total_rows / n));
   }
   capacity_L     = initial_alloc;
   capacity_U     = initial_alloc;

   /* allocate other memory for L and U struct */
   temp_L_diag_j  = hypre_CTAlloc(HYPRE_Int, capacity_L, memory_location);
   temp_U_diag_j  = hypre_CTAlloc(HYPRE_Int, capacity_U, memory_location);

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
      lena = A_diag_i[i + 1];
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
            hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
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
         hypre_ILUMinHeapRemoveIIIi(iL, iLev, iw, lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k] = -1;
         hypre_swap2i(iL, iLev, ii - lenl, lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k + 1];
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
                  hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
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
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            temp_L_diag_j = hypre_TReAlloc_v2(temp_L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L,
                                              memory_location);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL + j] = iL[ii - j - 1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii + 1] = U_diag_i[ii] + k;
      if (k > 0)
      {
         /* check if memory is enough */
         while (ctrU + k > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = (HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            temp_U_diag_j = hypre_TReAlloc_v2(temp_U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U,
                                              memory_location);
            u_levels = hypre_TReAlloc_v2(u_levels, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
         }
         //hypre_TMemcpy(temp_U_diag_j+ctrU,iL+ii,HYPRE_Int,k,memory_location,HYPRE_MEMORY_HOST);
         hypre_TMemcpy(temp_U_diag_j + ctrU, iL + ii, HYPRE_Int, k,
                       memory_location, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(u_levels + ctrU, iLev + ii, HYPRE_Int, k,
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
      lena = A_diag_i[i + 1];
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
            hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
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
      lena = A_offd_i[i + 1];
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
         hypre_ILUMinHeapRemoveIIIi(iL, iLev, iw, lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k] = -1;
         hypre_swap2i(iL, iLev, ii - lenl, lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k + 1];
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
                  hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
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
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            temp_L_diag_j = hypre_TReAlloc_v2(temp_L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L,
                                              memory_location);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL + j] = iL[ii - j - 1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii + 1] = U_diag_i[ii] + k;
      if (k > 0)
      {
         /* check if memory is enough */
         while (ctrU + k > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = (HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            temp_U_diag_j = hypre_TReAlloc_v2(temp_U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U,
                                              memory_location);
            u_levels = hypre_TReAlloc_v2(u_levels, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
         }
         hypre_TMemcpy(temp_U_diag_j + ctrU, iL + ii, HYPRE_Int, k, memory_location, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(u_levels + ctrU, iLev + ii, HYPRE_Int, k, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         ctrU += k;
      }

      /* reset iw */
      for (j = ii; j < lenu; j++)
      {
         iw[iL[j]] = -1;
      }
   } /* end of main loop ii from nLU to n */

   /* external part matrix */
   for (ii = n; ii < total_rows; ii++)
   {
      i = ii - n;
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = ii;
      lena = E_i[i + 1];
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
            hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
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
         hypre_ILUMinHeapRemoveIIIi(iL, iLev, iw, lenh);
         lenh--;
         /* copy to the end of array */
         lenl++;
         /* reset iw for that, not using anymore */
         iw[k] = -1;
         hypre_swap2i(iL, iLev, ii - lenl, lenh);
         /*
          * now the elimination on current row could start.
          * eliminate row k (new index) from current row
          */
         ku = U_diag_i[k + 1];
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
                  hypre_ILUMinHeapAddIIIi(iL, iLev, iw, lenh);
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
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* check if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            temp_L_diag_j = hypre_TReAlloc_v2(temp_L_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_L,
                                              memory_location);
         }
         /* now copy L data, reverse order */
         for (j = 0; j < lenl; j++)
         {
            temp_L_diag_j[ctrL + j] = iL[ii - j - 1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii + 1] = U_diag_i[ii] + k;
      if (k > 0)
      {
         /* check if memory is enough */
         while (ctrU + k > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = (HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            temp_U_diag_j = hypre_TReAlloc_v2(temp_U_diag_j, HYPRE_Int, tmp, HYPRE_Int, capacity_U,
                                              memory_location);
            u_levels = hypre_TReAlloc_v2(u_levels, HYPRE_Int, tmp, HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
         }
         hypre_TMemcpy(temp_U_diag_j + ctrU, iL + ii, HYPRE_Int, k, memory_location, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(u_levels + ctrU, iLev + ii, HYPRE_Int, k, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
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
   hypre_TFree(u_levels, HYPRE_MEMORY_HOST);

   *L_diag_j = temp_L_diag_j;
   *U_diag_j = temp_U_diag_j;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ILUSetupILUKRAS
 *
 * ILU(k) numeric factorization for RAS
 *
 * A: input matrix
 * lfil: level of fill-in, the k in ILU(k)
 * perm: permutation array indicating ordering of factorization.
 *       Perm could come from a CF_marker array or a reordering routine.
 * nLU: size of computed LDU factorization.
 * Lptr, Dptr, Uptr: L, D, U factors.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupILUKRAS(hypre_ParCSRMatrix  *A,
                      HYPRE_Int            lfil,
                      HYPRE_Int           *perm,
                      HYPRE_Int            nLU,
                      hypre_ParCSRMatrix **Lptr,
                      HYPRE_Real         **Dptr,
                      hypre_ParCSRMatrix **Uptr)
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
      return hypre_ILUSetupILU0RAS(A, perm, nLU, Lptr, Dptr, Uptr);
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
   HYPRE_MemoryLocation     memory_location = hypre_ParCSRMatrixMemoryLocation(A);

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

   /* reverse permutation array */
   HYPRE_Int               *rperm;
   /* temp array for old permutation */
   HYPRE_Int               *perm_old;

   /* start setup */
   /* check input and get problem size */
   n =  hypre_CSRMatrixNumRows(A_diag);
   if (nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: nLU out of range.\n");
   }

   /* Init I array anyway. S's might be freed later */
   D_data   = hypre_CTAlloc(HYPRE_Real, total_rows, memory_location);
   L_diag_i = hypre_CTAlloc(HYPRE_Int, (total_rows + 1), memory_location);
   U_diag_i = hypre_CTAlloc(HYPRE_Int, (total_rows + 1), memory_location);

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
   iw          = hypre_CTAlloc(HYPRE_Int, 5 * total_rows, HYPRE_MEMORY_HOST);
   rperm       = iw + 3 * total_rows;
   perm_old    = perm;
   perm        = iw + 4 * total_rows;
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
   hypre_ILUBuildRASExternalMatrix(A, rperm, &E_i, &E_j, &E_data);
   /* do symbolic factorization */
   hypre_ILUSetupILUKRASSymbolic(n, A_diag_i, A_diag_j, A_offd_i, A_offd_j, E_i, E_j, ext, lfil, perm,
                                 rperm, iw,
                                 nLU, L_diag_i, U_diag_i, &L_diag_j, &U_diag_j);

   /*
    * after this, we have our I,J for L, U and S ready, and L sorted
    * iw are still -1 after symbolic factorization
    * now setup helper array here
    */
   if (L_diag_i[total_rows])
   {
      L_diag_data = hypre_CTAlloc(HYPRE_Real, L_diag_i[total_rows], memory_location);
   }
   if (U_diag_i[total_rows])
   {
      U_diag_data = hypre_CTAlloc(HYPRE_Real, U_diag_i[total_rows], memory_location);
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
      kl = L_diag_i[ii + 1];
      ku = U_diag_i[ii + 1];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i + 1];
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
         ku = U_diag_i[jpiv + 1];

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
               L_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else if (col == ii)
            {
               /* diag part */
               D_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else
            {
               /* U part */
               U_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
         }
      }
      /* reset working array */
      ku = U_diag_i[ii + 1];
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
      if (hypre_abs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / D_data[ii];

   }/* end of loop for upper part */

   /* first loop for upper part */
   for (ii = nLU; ii < n; ii++)
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii + 1];
      ku = U_diag_i[ii + 1];
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
      k2 = A_diag_i[i + 1];
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
      k2 = A_offd_i[i + 1];
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
         ku = U_diag_i[jpiv + 1];

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
               L_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else if (col == ii)
            {
               /* diag part */
               D_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else
            {
               /* U part */
               U_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
         }
      }
      /* reset working array */
      ku = U_diag_i[ii + 1];
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
      if (hypre_abs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / D_data[ii];

   }/* end of loop for lower part */

   /* last loop through external */
   for (ii = n; ii < total_rows; ii++)
   {
      // get row i
      i = ii - n;
      kl = L_diag_i[ii + 1];
      ku = U_diag_i[ii + 1];
      k1 = E_i[i];
      k2 = E_i[i + 1];
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
         ku = U_diag_i[jpiv + 1];

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
               L_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else if (col == ii)
            {
               /* diag part */
               D_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
            else
            {
               /* U part */
               U_diag_data[icol] -= L_diag_data[j] * U_diag_data[k];
            }
         }
      }
      /* reset working array */
      ku = U_diag_i[ii + 1];
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
      if (hypre_abs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / D_data[ii];

   }/* end of loop for external loop */

   /*
    * 4: Finishing up and free
    */
   HYPRE_BigInt big_total_rows = (HYPRE_BigInt)total_rows;
   hypre_MPI_Allreduce(&big_total_rows, &global_num_rows, 1, HYPRE_MPI_BIG_INT,
                       hypre_MPI_SUM, comm);
   /* need to get new column start */
   {
      HYPRE_BigInt global_start;
      hypre_MPI_Scan(&big_total_rows, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
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
   if (L_diag_i[total_rows] > 0)
   {
      hypre_CSRMatrixData(L_diag) = L_diag_data;
      hypre_CSRMatrixJ(L_diag) = L_diag_j;
   }
   else
   {
      /* we allocated some initial length, so free them */
      hypre_TFree(L_diag_j, memory_location);
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
   if (U_diag_i[n] > 0)
   {
      hypre_CSRMatrixData(U_diag) = U_diag_data;
      hypre_CSRMatrixJ(U_diag) = U_diag_j;
   }
   else
   {
      /* we allocated some initial length, so free them */
      hypre_TFree(U_diag_j, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (U_diag_i[total_rows]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

   /* free */
   hypre_TFree(iw, HYPRE_MEMORY_HOST);

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

/*--------------------------------------------------------------------------
 * hypre_ILUSetupILUTRAS
 *
 * ILUT for RAS
 *
 * A: input matrix
 * lfil: level of fill-in, the k in ILU(k)
 * tol: droptol array in ILUT
 *    tol[0]: matrix B
 *    tol[1]: matrix E and F
 *    tol[2]: matrix S
 * perm: permutation array indicating ordering of factorization.
 *       Perm could come from a CF_marker: array or a reordering routine.
 * nLU: size of computed LDU factorization. If nLU < n, Schur compelemnt will be formed
 * Lptr, Dptr, Uptr: L, D, U factors.
 * Sptr: Schur complement
 *
 * Keep the largest lfil entries that is greater than some tol relative
 *    to the input tol and the norm of that row in both L and U
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ILUSetupILUTRAS(hypre_ParCSRMatrix  *A,
                      HYPRE_Int            lfil,
                      HYPRE_Real          *tol,
                      HYPRE_Int           *perm,
                      HYPRE_Int            nLU,
                      hypre_ParCSRMatrix **Lptr,
                      HYPRE_Real         **Dptr,
                      hypre_ParCSRMatrix **Uptr)
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
   HYPRE_Int                i, ii, j, k1, k2, k12, k22, kl, ku, col, icol, lenl, lenu, lenhu, lenhlr,
                            lenhll, jpos, jrow;
   HYPRE_Real               inorm, itolb, itolef, dpiv, lxu;
   HYPRE_Int                *iw, *iL;
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
   HYPRE_BigInt             col_starts[2];

   /* data objects for A */
   hypre_CSRMatrix          *A_diag       = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix          *A_offd       = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real               *A_diag_data  = hypre_CSRMatrixData(A_diag);
   HYPRE_Int                *A_diag_i     = hypre_CSRMatrixI(A_diag);
   HYPRE_Int                *A_diag_j     = hypre_CSRMatrixJ(A_diag);
   HYPRE_Int                *A_offd_i     = hypre_CSRMatrixI(A_offd);
   HYPRE_Int                *A_offd_j     = hypre_CSRMatrixJ(A_offd);
   HYPRE_Real               *A_offd_data  = hypre_CSRMatrixData(A_offd);
   HYPRE_MemoryLocation      memory_location = hypre_ParCSRMatrixMemoryLocation(A);

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
      hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: nLU out of range.\n");
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
      initial_alloc = (HYPRE_Int)(nLU + hypre_ceil((HYPRE_Real)(nnz_A / 2.0)));
   }
   capacity_L = initial_alloc;
   capacity_U = initial_alloc;

   D_data = hypre_CTAlloc(HYPRE_Real, total_rows, memory_location);
   L_diag_i = hypre_CTAlloc(HYPRE_Int, (total_rows + 1), memory_location);
   U_diag_i = hypre_CTAlloc(HYPRE_Int, (total_rows + 1), memory_location);

   L_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_L, memory_location);
   U_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_U, memory_location);
   L_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_L, memory_location);
   U_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_U, memory_location);

   ctrL = ctrU = 0;

   /* setting up working array */
   iw = hypre_CTAlloc(HYPRE_Int, 4 * total_rows, HYPRE_MEMORY_HOST);
   iL = iw + total_rows;
   w = hypre_CTAlloc(HYPRE_Real, total_rows, HYPRE_MEMORY_HOST);
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
   rperm = iw + 2 * total_rows;
   perm_old = perm;
   perm = iw + 3 * total_rows;
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
   hypre_ILUBuildRASExternalMatrix(A, rperm, &E_i, &E_j, &E_data);

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
      k2 = A_diag_i[i + 1];
      kl = ii - 1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += hypre_abs(A_diag_data[j]);
      }
      if (inorm == .0)
      {
         hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: ILUT with zero row.\n");
      }
      inorm /= (HYPRE_Real)(k2 - k1);
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
            hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
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
         hypre_ILUMinHeapRemoveIRIi(iL, w, iw, lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;

         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         hypre_swap2(iL, w, lenhll, kl - lenhlr);
         lenhlr++;
         hypre_ILUMaxrHeapAddRabsI(w + kl, iL + kl, lenhlr);

         /* loop for elimination */
         ku = U_diag_i[jrow + 1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col  = U_diag_j[j];
            icol = iw[col];
            lxu  = - dpiv * U_diag_data[j];

            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col <  nLU && hypre_abs(lxu) < itolb) ||
                 (col >= nLU && hypre_abs(lxu) < itolef)))
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
                  hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
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

      if (hypre_abs(w[ii]) < MAT_TOL)
      {
         w[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / w[ii];
      iw[ii] = -1;

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;

            capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp,
                                         HYPRE_Int, capacity_L, memory_location);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp,
                                            HYPRE_Real, capacity_L, memory_location);
         }
         ctrL += lenl;
         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            hypre_ILUMaxrHeapRemoveRabsI(w + kl, iL + kl, lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only U part
       */
      ku = lenu + ii;
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
         hypre_ILUMaxQSplitRabsI(w, iL, ii + 1, ii + lenhu, ii + lenu);
      }

      U_diag_i[ii + 1] = U_diag_i[ii] + lenhu;
      if (lenhu > 0)
      {
         /* test if memory is enough */
         while (ctrU + lenhu > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = (HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp,
                                         HYPRE_Int, capacity_U, memory_location);
            U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp,
                                            HYPRE_Real, capacity_U, memory_location);
         }
         ctrU += lenhu;
         /* copy large data in */
         for (j = U_diag_i[ii]; j < ctrU; j++)
         {
            jpos = ii + 1 + j - U_diag_i[ii];
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
      k2 = A_diag_i[i + 1];
      k12 = A_offd_i[i];
      k22 = A_offd_i[i + 1];
      kl = ii - 1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += hypre_abs(A_diag_data[j]);
      }
      for (j = k12; j < k22; j++)
      {
         inorm += hypre_abs(A_offd_data[j]);
      }
      if (inorm == .0)
      {
         hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: ILUT with zero row.\n");
      }
      inorm /= (HYPRE_Real)(k2 + k22 - k1 - k12);
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
            hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
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
         hypre_ILUMinHeapRemoveIRIi(iL, w, iw, lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;
         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         hypre_swap2(iL, w, lenhll, kl - lenhlr);
         lenhlr++;
         hypre_ILUMaxrHeapAddRabsI(w + kl, iL + kl, lenhlr);
         /* loop for elimination */
         ku = U_diag_i[jrow + 1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv * U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col < nLU && hypre_abs(lxu) < itolb) || (col >= nLU && hypre_abs(lxu) < itolef)))
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
                  hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
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

      if (hypre_abs(w[ii]) < MAT_TOL)
      {
         w[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / w[ii];
      iw[ii] = -1;

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp,
                                         HYPRE_Int, capacity_L, memory_location);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp,
                                            HYPRE_Real, capacity_L, memory_location);
         }
         ctrL += lenl;
         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            hypre_ILUMaxrHeapRemoveRabsI(w + kl, iL + kl, lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only U part
       */
      ku = lenu + ii;
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
         hypre_ILUMaxQSplitRabsI(w, iL, ii + 1, ii + lenhu, ii + lenu);
      }

      U_diag_i[ii + 1] = U_diag_i[ii] + lenhu;
      if (lenhu > 0)
      {
         /* test if memory is enough */
         while (ctrU + lenhu > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = (HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp,
                                         HYPRE_Int, capacity_U, memory_location);
            U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp,
                                            HYPRE_Real, capacity_U, memory_location);
         }
         ctrU += lenhu;
         /* copy large data in */
         for (j = U_diag_i[ii]; j < ctrU; j++)
         {
            jpos = ii + 1 + j - U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
   }/* end of ii loop from nLU to n */


   /* main outer loop for upper part */
   for (ii = n; ii < total_rows; ii++)
   {
      /* get real row with perm */
      i = ii - n;
      k1 = E_i[i];
      k2 = E_i[i + 1];
      kl = ii - 1;
      /* reset row norm of ith row */
      inorm = .0;
      for (j = k1; j < k2; j++)
      {
         inorm += hypre_abs(E_data[j]);
      }
      if (inorm == .0)
      {
         hypre_error_w_msg(HYPRE_ERROR_ARG, "WARNING: ILUT with zero row.\n");
      }
      inorm /= (HYPRE_Real)(k2 - k1);
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
            hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
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
         hypre_ILUMinHeapRemoveIRIi(iL, w, iw, lenhll);
         lenhll--;
         /*
          * reset the drop part to -1
          * we don't need this iw anymore
          */
         iw[jrow] = -1;
         /* need to keep this one, move to the end of the heap */
         /* no longer need to maintain iw */
         hypre_swap2(iL, w, lenhll, kl - lenhlr);
         lenhlr++;
         hypre_ILUMaxrHeapAddRabsI(w + kl, iL + kl, lenhlr);
         /* loop for elimination */
         ku = U_diag_i[jrow + 1];
         for (j = U_diag_i[jrow]; j < ku; j++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv * U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if ((icol == -1) &&
                ((col < nLU && hypre_abs(lxu) < itolb) || (col >= nLU && hypre_abs(lxu) < itolef)))
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
                  hypre_ILUMinHeapAddIRIi(iL, w, iw, lenhll);
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

      if (hypre_abs(w[ii]) < MAT_TOL)
      {
         w[ii] = 1.0e-06;
      }
      D_data[ii] = 1. / w[ii];
      iw[ii] = -1;

      /*
       * now pick up the largest lfil from L
       * L part is guarantee to be larger than itol
       */

      lenl = lenhlr < lfil ? lenhlr : lfil;
      L_diag_i[ii + 1] = L_diag_i[ii] + lenl;
      if (lenl > 0)
      {
         /* test if memory is enough */
         while (ctrL + lenl > capacity_L)
         {
            HYPRE_Int tmp = capacity_L;
            capacity_L = (HYPRE_Int)(capacity_L * EXPAND_FACT + 1);
            L_diag_j = hypre_TReAlloc_v2(L_diag_j, HYPRE_Int, tmp,
                                         HYPRE_Int, capacity_L, memory_location);
            L_diag_data = hypre_TReAlloc_v2(L_diag_data, HYPRE_Real, tmp,
                                            HYPRE_Real, capacity_L, memory_location);
         }
         ctrL += lenl;
         /* copy large data in */
         for (j = L_diag_i[ii]; j < ctrL; j++)
         {
            L_diag_j[j] = iL[kl];
            L_diag_data[j] = w[kl];
            hypre_ILUMaxrHeapRemoveRabsI(w + kl, iL + kl, lenhlr);
            lenhlr--;
         }
      }
      /*
       * now reset working array
       * L part already reset when move out of heap, only U part
       */
      ku = lenu + ii;
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
         hypre_ILUMaxQSplitRabsI(w, iL, ii + 1, ii + lenhu, ii + lenu);
      }

      U_diag_i[ii + 1] = U_diag_i[ii] + lenhu;
      if (lenhu > 0)
      {
         /* test if memory is enough */
         while (ctrU + lenhu > capacity_U)
         {
            HYPRE_Int tmp = capacity_U;
            capacity_U = (HYPRE_Int)(capacity_U * EXPAND_FACT + 1);
            U_diag_j = hypre_TReAlloc_v2(U_diag_j, HYPRE_Int, tmp,
                                         HYPRE_Int, capacity_U, memory_location);
            U_diag_data = hypre_TReAlloc_v2(U_diag_data, HYPRE_Real, tmp,
                                            HYPRE_Real, capacity_U, memory_location);
         }
         ctrU += lenhu;
         /* copy large data in */
         for (j = U_diag_i[ii]; j < ctrU; j++)
         {
            jpos = ii + 1 + j - U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
   }/* end of ii loop from nLU to total_rows */

   /*
    * 3: Finishing up and free
    */
   HYPRE_BigInt big_total_rows = (HYPRE_BigInt)total_rows;
   hypre_MPI_Allreduce(&big_total_rows, &global_num_rows, 1, HYPRE_MPI_BIG_INT,
                       hypre_MPI_SUM, comm);
   /* need to get new column start */
   {
      HYPRE_BigInt global_start;
      hypre_MPI_Scan(&big_total_rows, &global_start, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
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
      hypre_TFree(L_diag_j, memory_location);
      hypre_TFree(L_diag_data, memory_location);
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
      hypre_TFree(U_diag_j, memory_location);
      hypre_TFree(U_diag_data, memory_location);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (U_diag_i[total_rows]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;

   /* free working array */
   hypre_TFree(iw, HYPRE_MEMORY_HOST);
   hypre_TFree(w, HYPRE_MEMORY_HOST);

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
