/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/
#include "_hypre_parcsr_ls.h"
#include "par_ilu.h"


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
   hypre_CSRMatrix         *matBLU_d         = hypre_ParILUDataMatBILUDevice(ilu_data);
   hypre_CSRMatrix         *matE_d           = hypre_ParILUDataMatEDevice(ilu_data);
   hypre_CSRMatrix         *matF_d           = hypre_ParILUDataMatFDevice(ilu_data);
   csrsv2Info_t            matBL_info        = hypre_ParILUDataMatBLILUSolveInfo(ilu_data);
   csrsv2Info_t            matBU_info        = hypre_ParILUDataMatBUILUSolveInfo(ilu_data);
   csrsv2Info_t            matSL_info        = hypre_ParILUDataMatSLILUSolveInfo(ilu_data);
   csrsv2Info_t            matSU_info        = hypre_ParILUDataMatSUILUSolveInfo(ilu_data);
   HYPRE_Int               *A_diag_fake      = hypre_ParILUDataMatAFakeDiagonal(ilu_data);
   hypre_ParVector         *Xtemp            = NULL;
   hypre_ParVector         *Ytemp            = NULL;
   hypre_Vector            *Ftemp_upper      = NULL;
   hypre_Vector            *Utemp_lower      = NULL;
#endif 
   
   hypre_ParCSRMatrix   *matA                = hypre_ParILUDataMatA(ilu_data);
   hypre_ParCSRMatrix   *matL                = hypre_ParILUDataMatL(ilu_data);
   HYPRE_Real           *matD                = hypre_ParILUDataMatD(ilu_data);   
   hypre_ParCSRMatrix   *matU                = hypre_ParILUDataMatU(ilu_data);
   hypre_ParCSRMatrix   *matS                = hypre_ParILUDataMatS(ilu_data);
//   hypre_ParCSRMatrix   *matM                = NULL;
   HYPRE_Int            nnzBEF;
   HYPRE_Int            nnzG;/* g stands for global */
   HYPRE_Real           nnzS;/* total nnz in S */
   HYPRE_Int            nnzS_offd;
   HYPRE_Int            size_C/* total size of coarse grid */;
   
   HYPRE_Int            n                    = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));
   /* reordering option */
   HYPRE_Int            perm_opt;
   //HYPRE_Int            m;/* m = n-LU */
   HYPRE_Int            num_procs,  my_id;

   hypre_ParVector      *Utemp               = NULL;
   hypre_ParVector      *Ftemp               = NULL;
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
   
   /* ----- begin -----*/

   //num_threads = hypre_NumThreads();

   hypre_MPI_Comm_size(comm,&num_procs);
   hypre_MPI_Comm_rank(comm,&my_id);
   
#ifdef HYPRE_USING_CUDA
   /* create cuda and cusparse information when needed */
   /* Use most of them from global information */
   /* set matrix L descripter, L is a lower triangular matrix with unit diagonal entries */
   if(!matL_des)
   {
      HYPRE_CUSPARSE_CALL(cusparseCreateMatDescr(&(hypre_ParILUDataMatLMatrixDescription(ilu_data))));
      matL_des = hypre_ParILUDataMatLMatrixDescription(ilu_data);
      HYPRE_CUSPARSE_CALL(cusparseSetMatIndexBase(matL_des, CUSPARSE_INDEX_BASE_ZERO));
      HYPRE_CUSPARSE_CALL(cusparseSetMatType(matL_des, CUSPARSE_MATRIX_TYPE_GENERAL));
      HYPRE_CUSPARSE_CALL(cusparseSetMatFillMode(matL_des, CUSPARSE_FILL_MODE_LOWER));
      HYPRE_CUSPARSE_CALL(cusparseSetMatDiagType(matL_des, CUSPARSE_DIAG_TYPE_UNIT));
   }
   /* set matrix U descripter, U is a upper triangular matrix with non-unit diagonal entries */
   if(!matU_des)
   {
      HYPRE_CUSPARSE_CALL(cusparseCreateMatDescr(&(hypre_ParILUDataMatUMatrixDescription(ilu_data))));
      matU_des = hypre_ParILUDataMatUMatrixDescription(ilu_data);
      HYPRE_CUSPARSE_CALL(cusparseSetMatIndexBase(matU_des, CUSPARSE_INDEX_BASE_ZERO));
      HYPRE_CUSPARSE_CALL(cusparseSetMatType(matU_des, CUSPARSE_MATRIX_TYPE_GENERAL));
      HYPRE_CUSPARSE_CALL(cusparseSetMatFillMode(matU_des, CUSPARSE_FILL_MODE_UPPER));
      HYPRE_CUSPARSE_CALL(cusparseSetMatDiagType(matU_des, CUSPARSE_DIAG_TYPE_NON_UNIT));
   }
   if(!matBL_info)
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info((ilu_data) -> matBL_info)) );
      matBL_info = NULL;
   }
   if(!matBU_info)
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info((ilu_data) -> matBU_info)) );
      matBU_info = NULL;
   }
   if(!matSL_info)
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info((ilu_data) -> matSL_info)) );
      matSL_info = NULL;
   }
   if(!matSU_info)
   {
      HYPRE_CUSPARSE_CALL( (cusparseDestroyCsrsv2Info((ilu_data) -> matSU_info)) );
      matSU_info = NULL;
   }
   if(ilu_solve_buffer)
   {
      hypre_TFree(ilu_solve_buffer, HYPRE_MEMORY_DEVICE);
      ilu_solve_buffer = NULL;
   }
   if(matBLU_d)
   {
      hypre_CSRMatrixDestroy( matBLU_d );
      matBLU_d = NULL;
   }
   if(matE_d)
   {
      hypre_CSRMatrixDestroy( matE_d );
      matE_d = NULL;
   }
   if(matF_d)
   {
      hypre_CSRMatrixDestroy( matF_d );
      matF_d = NULL;
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
      hypre_TFree(hypre_ParILUDataMatAFakeDiagonal(ilu_data), HYPRE_MEMORY_SHARED);
      hypre_ParILUDataMatAFakeDiagonal(ilu_data) = NULL;
   }
#endif

   /* Free Previously allocated data, if any not destroyed */
   if(matL)
   {
       hypre_ParCSRMatrixDestroy(matL);
       matL = NULL;
   }
   if(matU)
   {
      hypre_ParCSRMatrixDestroy(matU);
      matU = NULL;
   }
   if(matS)
   {
      hypre_ParCSRMatrixDestroy(matS);
      matS = NULL;
   }
   if(matD)
   {
      hypre_TFree(matD, HYPRE_MEMORY_HOST);
      matD = NULL;
   }
   if(CF_marker_array)
   {
      hypre_TFree(CF_marker_array, HYPRE_MEMORY_HOST);
      CF_marker_array = NULL;
   }     


   /* clear old l1_norm data, if created */
   if(hypre_ParILUDataL1Norms(ilu_data))
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
      case 10: case 11: 
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
   if(hypre_ParILUDataSchurPrecond(ilu_data))
   {
      switch(ilu_type){
      case 10: case 11: 
         HYPRE_ILUDestroy(hypre_ParILUDataSchurPrecond(ilu_data)); //ILU as precond for Schur
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
   hypre_ParVectorSetPartitioningOwner(Utemp,0);
   hypre_ParILUDataUTemp(ilu_data) = Utemp;

   Ftemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                          hypre_ParCSRMatrixGlobalNumRows(A),
                          hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Ftemp);
   hypre_ParVectorSetPartitioningOwner(Ftemp,0);
   hypre_ParILUDataFTemp(ilu_data) = Ftemp;
   
   /* set matrix, solution and rhs pointers */
   matA = A;
   F_array = f;
   U_array = u;
   
   // create perm arary if necessary
   if(perm == NULL)
   {
      /* default to use local RCM */
      perm_opt = 1;
      switch(ilu_type)
      {
         case 10: case 11: case 20: case 21: case 30: case 31: /* symmetric */
            hypre_ILUGetPerm(matA, &perm, &nLU, perm_opt);
            break;
         case 40: case 41:/* ddPQ */
            hypre_ILUGetPermddPQ(matA, &perm, &qperm, tol_ddPQ, &nLU, &nI, perm_opt);
            break;
         case 0: case 1:
            hypre_ILUGetLocalPerm(matA, &perm, &nLU, perm_opt);
            break;
         default:
            hypre_ILUGetLocalPerm(matA, &perm, &nLU, perm_opt);
            break;
      }
   }
   //m = n - nLU;
   /* factorization */
   switch(ilu_type)
   {
      case 0:  
#ifdef HYPRE_USING_CUDA
               if(fill_level == 0)
               {
                  hypre_ILUSetupCusparseILU0(matA, perm, perm, n, n, matL_des, matU_des, ilu_solve_policy, &ilu_solve_buffer, 
                                                         &matBL_info, &matBU_info, &matSL_info, &matSU_info, &matBLU_d, &matS, 
                                                         &matE_d, &matF_d, &A_diag_fake);//BJ + cusparse_ilu0()
               }
               else
               {
#endif
                  hypre_ILUSetupILUK(matA, fill_level, perm, perm, n, n, &matL, &matD, &matU, &matS, &u_end); //BJ + hypre_iluk()
#ifdef HYPRE_USING_CUDA
               }
#endif
         break;
      case 1:  hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, perm, n, n, &matL, &matD, &matU, &matS, &u_end); //BJ + hypre_ilut()
         break;
      case 10: 
#ifdef HYPRE_USING_CUDA
               if(fill_level == 0)
               {
                  /* Only support ILU0 */
                  hypre_ILUSetupCusparseILU0(matA, perm, perm, n, nLU, matL_des, matU_des, ilu_solve_policy, &ilu_solve_buffer, 
                                                         &matBL_info, &matBU_info, &matSL_info, &matSU_info, &matBLU_d, &matS, 
                                                         &matE_d, &matF_d, &A_diag_fake);//BJ + cusparse_ilu0()
               }
               else
               {
#endif
                  hypre_ILUSetupILUK(matA, fill_level, perm, perm, nLU, nLU, &matL, &matD, &matU, &matS, &u_end); //GMRES + hypre_iluk()
#ifdef HYPRE_USING_CUDA
               }
#endif
         break;
      case 11: hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, perm, nLU, nLU, &matL, &matD, &matU, &matS, &u_end); //GMRES + hypre_ilut()
         break;
      case 20: hypre_ILUSetupILUK(matA, fill_level, perm, perm, nLU, nLU, &matL, &matD, &matU, &matS, &u_end); //Newton–Schulz–Hotelling + hypre_iluk()
         break;
      case 21: hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, perm, nLU, nLU, &matL, &matD, &matU, &matS, &u_end); //Newton–Schulz–Hotelling + hypre_ilut()
         break;
      case 30: hypre_ILUSetupILUKRAS(matA, fill_level, perm, nLU, &matL, &matD, &matU); //RAS + hypre_iluk()
         break;
      case 31: hypre_ILUSetupILUTRAS(matA, max_row_elmts, droptol, perm, nLU, &matL, &matD, &matU); //RAS + hypre_ilut()
         break;
      case 40: hypre_ILUSetupILUK(matA, fill_level, perm, qperm, nLU, nI, &matL, &matD, &matU, &matS, &u_end); //ddPQ + GMRES + hypre_iluk()
         break;
      case 41: hypre_ILUSetupILUT(matA, max_row_elmts, droptol, perm, qperm, nLU, nI, &matL, &matD, &matU, &matS, &u_end); //ddPQ + GMRES + hypre_ilut()
         break;
      default: 
#ifdef HYPRE_USING_CUDA
               hypre_ILUSetupCusparseILU0(matA, perm, perm, n, n, matL_des, matU_des, ilu_solve_policy, &ilu_solve_buffer, 
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
      case 10: case 11: case 40: case 41:
         if(matS)
         {
#ifdef HYPRE_USING_CUDA
            if(fill_level == 0 && ilu_type == 10)
            {
               /* create working vectors */
               
               Xtemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matS),
                                      hypre_ParCSRMatrixGlobalNumRows(matS),
                                      hypre_ParCSRMatrixRowStarts(matS));
               hypre_ParVectorInitialize(Xtemp);
               hypre_ParVectorSetPartitioningOwner(Xtemp,0);

               Ytemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matS),
                                      hypre_ParCSRMatrixGlobalNumRows(matS),
                                      hypre_ParCSRMatrixRowStarts(matS));
               hypre_ParVectorInitialize(Ytemp);
               hypre_ParVectorSetPartitioningOwner(Ytemp,0);
               
               Ftemp_upper = hypre_SeqVectorCreate(nLU);
               hypre_VectorOwnsData(Ftemp_upper)   = 0;
               hypre_VectorData(Ftemp_upper)       = hypre_VectorData(hypre_ParVectorLocalVector(Ftemp));
               hypre_SeqVectorInitialize(Ftemp_upper);
               
               Utemp_lower = hypre_SeqVectorCreate(n - nLU);
               hypre_VectorOwnsData(Utemp_lower)   = 0;
               hypre_VectorData(Utemp_lower)       = hypre_VectorData(hypre_ParVectorLocalVector(Utemp)) + nLU;
               hypre_SeqVectorInitialize(Utemp_lower);
               
               /* create GMRES */
               HYPRE_ParCSRGMRESCreate(comm, &schur_solver);
               
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
               HYPRE_GMRESSetTol             (schur_solver, (ilu_data -> ss_tol));
               HYPRE_GMRESSetAbsoluteTol     (schur_solver, (ilu_data -> ss_absolute_tol));
               HYPRE_GMRESSetLogging         (schur_solver, (ilu_data -> ss_logging));
               HYPRE_GMRESSetPrintLevel      (schur_solver, (ilu_data -> ss_print_level));/* set to zero now, don't print */
               HYPRE_GMRESSetRelChange       (schur_solver, (ilu_data -> ss_rel_change));
               
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
               hypre_ParVectorSetPartitioningOwner(rhs,0);        
               x = hypre_ParVectorCreate(comm,
                                       hypre_ParCSRMatrixGlobalNumRows(matS),
                                       hypre_ParCSRMatrixRowStarts(matS));
               hypre_ParVectorInitialize(x);
               hypre_ParVectorSetPartitioningOwner(x,0);
               
               /* setup solver */
               HYPRE_GMRESSetup(schur_solver,(HYPRE_Matrix)ilu_vdata,(HYPRE_Vector)rhs,(HYPRE_Vector)x);
               
               /* update ilu_data */
               hypre_ParILUDataSchurSolver   (ilu_data) = schur_solver;
               hypre_ParILUDataSchurPrecond  (ilu_data) = schur_precond;
               hypre_ParILUDataRhs           (ilu_data) = rhs;
               hypre_ParILUDataX             (ilu_data) = x;
            }
            else
            {
#endif
               /* setup GMRES parameters */
               HYPRE_ParCSRGMRESCreate(comm, &schur_solver);
               
               HYPRE_GMRESSetKDim            (schur_solver, hypre_ParILUDataSchurGMRESKDim(ilu_data));
               HYPRE_GMRESSetMaxIter         (schur_solver, hypre_ParILUDataSchurGMRESMaxIter(ilu_data));/* we don't need that many solves */
               HYPRE_GMRESSetTol             (schur_solver, (ilu_data -> ss_tol));
               HYPRE_GMRESSetAbsoluteTol     (schur_solver, (ilu_data -> ss_absolute_tol));
               HYPRE_GMRESSetLogging         (schur_solver, (ilu_data -> ss_logging));
               HYPRE_GMRESSetPrintLevel      (schur_solver, (ilu_data -> ss_print_level));/* set to zero now, don't print */
               HYPRE_GMRESSetRelChange       (schur_solver, (ilu_data -> ss_rel_change));
               
               /* setup preconditioner parameters */
               /* create precond, the default is ILU0 */
               HYPRE_ILUCreate               (&schur_precond);    
               HYPRE_ILUSetType              (schur_precond, (ilu_data -> sp_ilu_type));
               HYPRE_ILUSetLevelOfFill       (schur_precond, (ilu_data -> sp_ilu_lfil));
               HYPRE_ILUSetMaxNnzPerRow      (schur_precond, (ilu_data -> sp_ilu_max_row_nnz));
               HYPRE_ILUSetDropThresholdArray(schur_precond, (ilu_data -> sp_ilu_droptol));
               hypre_ILUSetOwnDropThreshold  (schur_precond, 0);/* using exist droptol */
               HYPRE_ILUSetPrintLevel        (schur_precond, (ilu_data -> sp_print_level));
               HYPRE_ILUSetMaxIter           (schur_precond, (ilu_data -> sp_max_iter));
               HYPRE_ILUSetTol               (schur_precond, (ilu_data -> sp_tol));
                           
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
               hypre_ParVectorSetPartitioningOwner(rhs,0);        
               x = hypre_ParVectorCreate(comm,
                                       hypre_ParCSRMatrixGlobalNumRows(matS),
                                       hypre_ParCSRMatrixRowStarts(matS));
               hypre_ParVectorInitialize(x);
               hypre_ParVectorSetPartitioningOwner(x,0);
               
               /* setup solver */
               HYPRE_GMRESSetup(schur_solver,(HYPRE_Matrix)matS,(HYPRE_Vector)rhs,(HYPRE_Vector)x);
               
               /* update ilu_data */
               hypre_ParILUDataSchurSolver   (ilu_data) = schur_solver;
               hypre_ParILUDataSchurPrecond  (ilu_data) = schur_precond;
               hypre_ParILUDataRhs           (ilu_data) = rhs;
               hypre_ParILUDataX             (ilu_data) = x;
#ifdef HYPRE_USING_CUDA
            }
#endif
         }
         break;
      case 20: case 21:
         if(matS)
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
            hypre_ParVectorSetPartitioningOwner(rhs,0);        
            x = hypre_ParVectorCreate(comm,
                                    hypre_ParCSRMatrixGlobalNumRows(matS),
                                    hypre_ParCSRMatrixRowStarts(matS));
            hypre_ParVectorInitialize(x);
            hypre_ParVectorSetPartitioningOwner(x,0);

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
         if(!comm_pkg)
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
      default:
         break;
   }
   /* set pointers to ilu data */
#ifdef HYPRE_USING_CUDA
   /* set cusparse pointers */
   //hypre_ParILUDataILUSolveBuffer(ilu_data)  = ilu_solve_buffer;
   hypre_ParILUDataMatBILUDevice(ilu_data)      = matBLU_d;
   hypre_ParILUDataMatEDevice(ilu_data)         = matE_d;
   hypre_ParILUDataMatFDevice(ilu_data)         = matF_d;
   hypre_ParILUDataILUSolveBuffer(ilu_data)     = ilu_solve_buffer;
   hypre_ParILUDataMatBLILUSolveInfo(ilu_data)  = matBL_info;
   hypre_ParILUDataMatBUILUSolveInfo(ilu_data)  = matBU_info;
   hypre_ParILUDataMatSLILUSolveInfo(ilu_data)  = matSL_info;
   hypre_ParILUDataMatSUILUSolveInfo(ilu_data)  = matSU_info;
   hypre_ParILUDataXTemp(ilu_data)              = Xtemp;
   hypre_ParILUDataYTemp(ilu_data)              = Ytemp;
   hypre_ParILUDataFTempUpper(ilu_data)         = Ftemp_upper;
   hypre_ParILUDataUTempLower(ilu_data)         = Utemp_lower;
   hypre_ParILUDataMatAFakeDiagonal(ilu_data)   = A_diag_fake;
#endif
   hypre_ParILUDataMatA(ilu_data)               = matA;
   hypre_ParILUDataF(ilu_data)                  = F_array;
   hypre_ParILUDataU(ilu_data)                  = U_array;
   hypre_ParILUDataMatL(ilu_data)               = matL;
   hypre_ParILUDataMatD(ilu_data)               = matD;
   hypre_ParILUDataMatU(ilu_data)               = matU;
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
   nnzBEF = 0;
   /* size_C is the size of global coarse grid, upper left part */
   size_C = hypre_ParCSRMatrixGlobalNumRows(matA);
   /* switch to compute complexity */
   
#ifdef HYPRE_USING_CUDA
   if(ilu_type == 0 && fill_level == 0)
   {
      /* The nnz is for sure 1.0 in this case */
      (ilu_data -> operator_complexity) =  1.0;
   }
   else if(ilu_type == 10 && fill_level == 0)
   {
      /* The nnz is the sum of different parts */
      if(matBLU_d)
      {
         nnzBEF  += hypre_CSRMatrixNumNonzeros(matBLU_d);
      }
      if(matE_d)
      {
         nnzBEF  += hypre_CSRMatrixNumNonzeros(matE_d);
      }
      if(matF_d)
      {
         nnzBEF  += hypre_CSRMatrixNumNonzeros(matF_d);
      }
      hypre_MPI_Allreduce(&nnzBEF, &nnzG, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
      if(matS)
      {
         hypre_ParCSRMatrixSetDNumNonzeros(matS);
         nnzS = hypre_ParCSRMatrixDNumNonzeros(matS);
         /* if we have Schur system need to reduce it from size_C */
      }
      (ilu_data -> operator_complexity) =  ((HYPRE_Real)nnzG + nnzS) / 
                                           hypre_ParCSRMatrixDNumNonzeros(matA);
   }
   else
   {
#endif   
      if(matS)
      {
         hypre_ParCSRMatrixSetDNumNonzeros(matS);
         nnzS = hypre_ParCSRMatrixDNumNonzeros(matS);
         /* if we have Schur system need to reduce it from size_C */
         size_C -= hypre_ParCSRMatrixGlobalNumRows(matS);
         switch(ilu_type)
         {
            case 10: case 11: case 40: case 41:
               /* now we need to compute the preoconditioner */
               schur_precond_ilu = (hypre_ParILUData*) (ilu_data -> schur_precond);
               /* borrow i for local nnz of S */
               i = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(matS));
               hypre_MPI_Allreduce(&i, &nnzS_offd, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
               nnzS = nnzS * (schur_precond_ilu -> operator_complexity) +nnzS_offd;
               break;
            case 20: case 21:
               schur_solver_nsh = (hypre_ParNSHData*) hypre_ParILUDataSchurSolver(ilu_data);
               nnzS = nnzS * (hypre_ParNSHDataOperatorComplexity(schur_solver_nsh));
               break; 
            default:
               break;
         }
      }
      
      (ilu_data -> operator_complexity) =  ((HYPRE_Real)size_C + nnzS +
                                          hypre_ParCSRMatrixDNumNonzeros(matL) +
                                          hypre_ParCSRMatrixDNumNonzeros(matU))/ 
                                          hypre_ParCSRMatrixDNumNonzeros(matA);
#ifdef HYPRE_USING_CUDA
   }
#endif    
   if ((my_id == 0) && (print_level > 0))
   {
      hypre_printf("ILU SETUP: operator complexity = %f  \n", ilu_data -> operator_complexity);
   }

   if ( logging > 1 ) {
      residual =
      hypre_ParVectorCreate(hypre_ParCSRMatrixComm(matA),
                              hypre_ParCSRMatrixGlobalNumRows(matA),
                              hypre_ParCSRMatrixRowStarts(matA) );
      hypre_ParVectorInitialize(residual);
      hypre_ParVectorSetPartitioningOwner(residual,0);
      (ilu_data -> residual) = residual;
   }
   else{
      (ilu_data -> residual) = NULL;
   }
   rel_res_norms = hypre_CTAlloc(HYPRE_Real, (ilu_data -> max_iter), HYPRE_MEMORY_HOST);
   (ilu_data -> rel_res_norms) = rel_res_norms;      
   
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
hypre_ParILUCusparseExtractDiagonalCSR(hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int *rqperm, hypre_CSRMatrix **A_diagp)
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
      for(j = A_diag_i[perm[i]] ; j < A_diag_i[perm[i]+1] ; j ++)
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
 *          Be careful if you use this function in any other routines.
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
   
   if(nLU == n)
   {
      /* No schur complement makes everything easy :) */
      hypre_CSRMatrix  *B              = NULL;
      B                                = hypre_CSRMatrixCreate(n, n, nnz_A_diag);
      hypre_CSRMatrixInitialize(B);
      hypre_CSRMatrixCopy(A_diag, B, 1);
      *Bp = B;
   }
   else if(nLU ==0)
   {
      /* All schur complement also makes everything easy :) */
      hypre_CSRMatrix  *C              = NULL;
      C                                = hypre_CSRMatrixCreate(n, n, nnz_A_diag);
      hypre_CSRMatrixInitialize(C);
      hypre_CSRMatrixCopy(A_diag, C, 1);
      *Cp = C;
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
      B_j                              = hypre_CTAlloc(HYPRE_Int, capacity_B, HYPRE_MEMORY_SHARED);
      B_data                           = hypre_CTAlloc(HYPRE_Real, capacity_B, HYPRE_MEMORY_SHARED);
      C_i                              = hypre_CSRMatrixI(C);
      C_j                              = hypre_CTAlloc(HYPRE_Int, capacity_C, HYPRE_MEMORY_SHARED);
      C_data                           = hypre_CTAlloc(HYPRE_Real, capacity_C, HYPRE_MEMORY_SHARED);
      E_i                              = hypre_CSRMatrixI(E);
      E_j                              = hypre_CTAlloc(HYPRE_Int, capacity_E, HYPRE_MEMORY_SHARED);
      E_data                           = hypre_CTAlloc(HYPRE_Real, capacity_E, HYPRE_MEMORY_SHARED);
      F_i                              = hypre_CSRMatrixI(F);
      F_j                              = hypre_CTAlloc(HYPRE_Int, capacity_F, HYPRE_MEMORY_SHARED);
      F_data                           = hypre_CTAlloc(HYPRE_Real, capacity_F, HYPRE_MEMORY_SHARED);
      ctrB                             = 0;
      ctrC                             = 0;
      ctrE                             = 0;
      ctrF                             = 0;
      
      /* Loop to copy data */
      /* B and F first */
      for ( i = 0; i < nLU; i++ )
      {
         B_i[i]   = ctrB;
         F_i[i]   = ctrF;
         for(j = A_diag_i[i] ; j < A_diag_i[i+1] ; j ++)
         {
            col = A_diag_j[j];
            if(col >= nLU)
            {
               break;
            }
            B_j[ctrB] = col;
            B_data[ctrB++] = A_diag_data[j];
            /* check capacity */
            if(ctrB >= capacity_B)
            {
               capacity_B = capacity_B * EXPAND_FACT + 1;
               B_j = hypre_TReAlloc(B_j, HYPRE_Int, capacity_B, HYPRE_MEMORY_SHARED);
               B_data = hypre_TReAlloc(B_data, HYPRE_Real, capacity_B, HYPRE_MEMORY_SHARED);
            }
         }
         for(; j < A_diag_i[i+1] ; j ++)
         {
            col = A_diag_j[j];
            col = col - nLU;
            F_j[ctrF] = col;
            F_data[ctrF++] = A_diag_data[j];
            if(ctrF >= capacity_F)
            {
               capacity_F = capacity_F * EXPAND_FACT + 1;
               F_j = hypre_TReAlloc(F_j, HYPRE_Int, capacity_F, HYPRE_MEMORY_SHARED);
               F_data = hypre_TReAlloc(F_data, HYPRE_Real, capacity_F, HYPRE_MEMORY_SHARED);
            }
         }
      }
      B_i[nLU] = ctrB;
      F_i[nLU] = ctrF;
      
      /* E and C afterward */
      for ( i = nLU; i < n; i++ )
      {
         row = i - nLU;
         E_i[row] = ctrE;
         C_i[row] = ctrC;
         for(j = A_diag_i[i] ; j < A_diag_i[i+1] ; j ++)
         {
            col = A_diag_j[j];
            if(col >= nLU)
            {
               break;
            }
            E_j[ctrE] = col;
            E_data[ctrE++] = A_diag_data[j];
            /* check capacity */
            if(ctrE >= capacity_E)
            {
               capacity_E = capacity_E * EXPAND_FACT + 1;
               E_j = hypre_TReAlloc(E_j, HYPRE_Int, capacity_E, HYPRE_MEMORY_SHARED);
               E_data = hypre_TReAlloc(E_data, HYPRE_Real, capacity_E, HYPRE_MEMORY_SHARED);
            }
         }
         for(; j < A_diag_i[i+1] ; j ++)
         {
            col = A_diag_j[j];
            col = col - nLU;
            C_j[ctrC] = col;
            C_data[ctrC++] = A_diag_data[j];
            if(ctrC >= capacity_C)
            {
               capacity_C = capacity_C * EXPAND_FACT + 1;
               C_j = hypre_TReAlloc(C_j, HYPRE_Int, capacity_C, HYPRE_MEMORY_SHARED);
               C_data = hypre_TReAlloc(C_data, HYPRE_Real, capacity_C, HYPRE_MEMORY_SHARED);
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
   HYPRE_Int               nnz_A                = A_i[n];
   
   /* pointers to cusparse data */
   csrilu02Info_t          matA_info            = NULL;
   
   /* variables and working arrays used during the ilu */
   HYPRE_Int               zero_pivot;
   size_t                  colperm_buffersize;
   HYPRE_Int               matA_buffersize;
   HYPRE_Int               *colperm             = NULL;
   void                    *colperm_buffer      = NULL;
   void                    *matA_buffer         = NULL;
   HYPRE_Real              *A_data_temp         = NULL;
   
   HYPRE_Int               isDoublePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int               isSinglePrecision    = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;
   
   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle);
   cusparseMatDescr_t descr = hypre_HandleCusparseMatDescr(hypre_handle);
   
   hypre_assert(isDoublePrecision || isSinglePrecision);
   
   /* 1. Sort columns inside each row first, we can't assume that's sorted */
   /* 1-1. Get buffer size */
   HYPRE_CUSPARSE_CALL(cusparseXcsrsort_bufferSizeExt(handle, n, n, nnz_A, A_i, A_j, &colperm_buffersize));
   colperm_buffer                               = hypre_MAlloc(colperm_buffersize, HYPRE_MEMORY_DEVICE);
   
   /* 1-2. Create perm array */
   colperm                                      = hypre_CTAlloc(HYPRE_Int, nnz_A, HYPRE_MEMORY_DEVICE);
   HYPRE_CUSPARSE_CALL(cusparseCreateIdentityPermutation(handle, nnz_A, colperm));
   
   /* 1-3. Sort column */
   cusparseXcsrsort(handle, n, n, nnz_A, descr, A_i, A_j, colperm, colperm_buffer);
   hypre_TFree(colperm_buffer, HYPRE_MEMORY_DEVICE);
   A_data_temp                                  = hypre_CTAlloc(HYPRE_Real, nnz_A, HYPRE_MEMORY_SHARED);
   
   /* 1-4. Extend to column value */
   if(isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDgthr(handle, nnz_A, (double *) A_data, (double *) A_data_temp, colperm, CUSPARSE_INDEX_BASE_ZERO));
   }
   else if(isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseSgthr(handle, nnz_A, (float *) A_data, (float *) A_data_temp, colperm, CUSPARSE_INDEX_BASE_ZERO));
   }
   hypre_TFree(A_data, HYPRE_MEMORY_SHARED);
   hypre_CSRMatrixData(A)                       = A_data_temp;
   A_data                                       = hypre_CSRMatrixData(A);
   hypre_TFree(colperm, HYPRE_MEMORY_DEVICE);
   
   /* 2. Create info for ilu setup and solve */
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrilu02Info(&matA_info));
   
   /* 3. Get working array size */
   if(isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrilu02_bufferSize(handle, n, nnz_A, descr,
                                                         (double *) A_data, A_i, A_j, 
                                                         matA_info, &matA_buffersize));
   }
   else if(isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrilu02_bufferSize(handle, n, nnz_A, descr,
                                                         (float *) A_data, A_i, A_j, 
                                                         matA_info, &matA_buffersize));
   }
   /* 4. Create working array, since they won't be visited by host, allocate on device */
   matA_buffer                                  = hypre_MAlloc(matA_buffersize, HYPRE_MEMORY_DEVICE);
      
   /* 5. Now perform the analysis */
   /* 5-1. Analysis */
   if(isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrilu02_analysis(handle, n, nnz_A, descr,
                                                      (double *) A_data, A_i, A_j, 
                                                      matA_info, ilu_solve_policy, matA_buffer));
   }
   else if(isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseScsrilu02_analysis(handle, n, nnz_A, descr,
                                                      (float *) A_data, A_i, A_j, 
                                                      matA_info, ilu_solve_policy, matA_buffer));
   }                                                
   /* 5-2. Check for zero pivot */
   HYPRE_CUSPARSE_CALL(cusparseXcsrilu02_zeroPivot(handle, matA_info, &zero_pivot));
   
   /* 6. Apply the factorization */
   if(isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL(cusparseDcsrilu02(handle, n, nnz_A, descr,
                                             (double *) A_data, A_i, A_j, 
                                             matA_info, ilu_solve_policy, matA_buffer));
   }
   else if(isSinglePrecision)
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
   
   /* data objects for A */
   HYPRE_Int               n                    = hypre_CSRMatrixNumRows(A);
   HYPRE_Int               m                    = hypre_CSRMatrixNumCols(A);
   
   hypre_assert(n == m);
   
   HYPRE_Real              *A_data              = hypre_CSRMatrixData(A);
   HYPRE_Int               *A_i                 = hypre_CSRMatrixI(A);
   HYPRE_Int               *A_j                 = hypre_CSRMatrixJ(A);
   HYPRE_Int               nnz_A                = A_i[n];
   
   /* pointers to cusparse data */
   csrsv2Info_t            matL_info            = *matL_infop;
   csrsv2Info_t            matU_info            = *matU_infop;
   
   /* clear data if already exists */
   if(matL_info)
   {
      HYPRE_CUSPARSE_CALL( cusparseDestroyCsrsv2Info(matL_info) );
      matL_info = NULL;
   }
   if(matU_info)
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
   
   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle);
   
   /* 1. Create info for ilu setup and solve */
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrsv2Info(&(matL_info)));
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrsv2Info(&(matU_info)));
   
   /* 2. Get working array size */
   if(isDoublePrecision)
   {

      HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz_A,
                                                      matL_des, (double *) A_data, A_i, A_j, 
                                                      matL_info, &matL_buffersize));

      HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, nnz_A,
                                                      matU_des, (double *) A_data, A_i, A_j, 
                                                      matU_info, &matU_buffersize));
   }
   else if(isSinglePrecision)
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
   if(solve_buffersize > solve_oldbuffersize)
   {
      if(solve_buffer)
      {
         solve_buffer                           = hypre_ReAlloc(solve_buffer, solve_buffersize, HYPRE_MEMORY_DEVICE);
      }
      else
      {
         solve_buffer                           = hypre_MAlloc(solve_buffersize, HYPRE_MEMORY_DEVICE);
      }
   }
      
   /* 4. Now perform the analysis */
   if(isDoublePrecision)
   {

      HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                      n, nnz_A, matL_des,
                                                      (double *) A_data, A_i, A_j, 
                                                      matL_info, ilu_solve_policy, solve_buffer));

      HYPRE_CUSPARSE_CALL(cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                                      n, nnz_A, matU_des,
                                                      (double *) A_data, A_i, A_j, 
                                                      matU_info, ilu_solve_policy, solve_buffer));
   }
   else if(isSinglePrecision)
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
hypre_ILUSetupCusparseILU0(hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int *qperm, HYPRE_Int n, HYPRE_Int nLU, 
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
   hypre_ParCSRCommPkg     *comm_pkg;
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int               num_sends, begin, end;
   HYPRE_Int               *send_buf            = NULL;
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
   HYPRE_Int               total_rows;
   HYPRE_Int               *col_starts          = NULL;
   HYPRE_Int               *S_diag_i            = NULL;
   HYPRE_Int               S_diag_nnz;
   hypre_CSRMatrix         *S_offd              = NULL;
   HYPRE_Int               *S_offd_i            = NULL;
   HYPRE_Int               *S_offd_j            = NULL;
   HYPRE_Real              *S_offd_data         = NULL;
   HYPRE_Int               *S_offd_colmap       = NULL;
   HYPRE_Int               S_offd_nnz;
   HYPRE_Int               S_offd_ncols;
   S_offd = hypre_ParCSRMatrixOffd(matS);
   
   /* set data slots */
   A_offd                                       = hypre_ParCSRMatrixOffd(A);
   A_offd_i                                     = hypre_CSRMatrixI(A_offd);
   A_offd_j                                     = hypre_CSRMatrixJ(A_offd);
   A_offd_data                                  = hypre_CSRMatrixData(A_offd);
   
   /* unfortunately we need to build the reverse permutation array */
   rperm                                        = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   rqperm                                       = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   for(i = 0 ; i < n ; i ++)
   {
      rperm[perm[i]] = i;
      rqperm[qperm[i]] = i;
   }
   
   /* Only call ILU when we really have a matrix on this processor */
   if(n > 0)
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
   if( nLU > 0 )
   { 
      /* Analysis of BILU */
      HYPRE_ILUSetupCusparseCSRILU0SetupSolve(*BLUptr, matL_des, matU_des,
                                 ilu_solve_policy, &matBL_info, &matBU_info,
                                 &buffer_size, &buffer);
   }
   
   /* now create S */
   /* col_starts is the new local start and end for matS */
   col_starts = hypre_CTAlloc(HYPRE_Int,2,HYPRE_MEMORY_HOST);
   hypre_MPI_Scan(&m, &total_rows, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   col_starts[1] = total_rows;
   col_starts[0] = total_rows - m;
   /* now need to get the total length */
   hypre_MPI_Allreduce(&m, &total_rows, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   
   /* only form S when at least one block has nLU < n */
   if( total_rows > 0 )
   { 
      A_fake_diag_i = hypre_CTAlloc(HYPRE_Int, m + 1, HYPRE_MEMORY_SHARED);
      if(SLU)
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
         
      /* S owns different start/end */
      hypre_ParCSRMatrixSetColStartsOwner(matS,1);
      hypre_ParCSRMatrixSetRowStartsOwner(matS,0);/* square matrix, use same row and col start */
      
      /* first put diagonal data in */
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(matS));
      hypre_ParCSRMatrixDiag(matS) = SLU;
      
      /* now start to construct offdiag of S */
      S_offd = hypre_ParCSRMatrixOffd(matS);
      S_offd_i = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_SHARED);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_SHARED);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_SHARED);
      S_offd_colmap = hypre_CTAlloc(HYPRE_Int, S_offd_ncols, HYPRE_MEMORY_HOST);
      
      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for(i = 1 ; i <= e ; i ++)
      {
         S_offd_i[i] = k3;
      }
      for(i = 0 ; i < m_e ; i ++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col+1];
         for(j = k1 ; j < k2 ; j ++)
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
      if(!comm_pkg)
      {
         hypre_MatvecCommPkgCreate(A);
         comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      }
      /* get total num of send */
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg,0);
      end = hypre_ParCSRCommPkgSendMapStart(comm_pkg,num_sends);
      send_buf = hypre_TAlloc(HYPRE_Int, end - begin, HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for(i = begin ; i < end ; i ++)
      {
         send_buf[i-begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)] - nLU + col_starts[0]; 
      }
         
      /* main communication */
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_buf, S_offd_colmap);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;
      
      hypre_ILUSortOffdColmap(matS);
      
      /* free */
      hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
      
   }/* end of forming S */
   
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
#endif

/* ILU(0) 
 * A = input matrix 
 * perm = permutation array indicating ordering of rows. Perm could come from a 
 *    CF_marker array or a reordering routine.
 * qperm = permutation array indicating ordering of columns
 * nI = number of interial unknowns
 * nLU = size of incomplete factorization, nLU should obey nLU <= nI.
 *    Schur complement is formed if nLU < n
 * Lptr, Dptr, Uptr, Sptr = L, D, U, S factors.
 * 
 * will form global Schur Matrix if nLU < n 
 */
HYPRE_Int
hypre_ILUSetupILU0(hypre_ParCSRMatrix *A, HYPRE_Int *perm, HYPRE_Int *qperm, HYPRE_Int nLU, HYPRE_Int nI, 
      hypre_ParCSRMatrix **Lptr, HYPRE_Real** Dptr, hypre_ParCSRMatrix **Uptr, hypre_ParCSRMatrix **Sptr, HYPRE_Int **u_end)
{
   HYPRE_Int                i, ii, j, k, k1, k2, k3, ctrU, ctrL, ctrS, lenl, lenu, jpiv, col, jpos;
   HYPRE_Int                *iw, *iL, *iU;
   HYPRE_Real               dd, t, dpiv, lxu, *wU, *wL;
   
   /* communication stuffs for S */
   MPI_Comm                 comm             = hypre_ParCSRMatrixComm(A);
   HYPRE_Int                S_offd_nnz, S_offd_ncols;
   hypre_ParCSRCommPkg      *comm_pkg;
   hypre_ParCSRCommHandle   *comm_handle;
   HYPRE_Int                num_sends, begin, end;
   HYPRE_Int                *send_buf        = NULL;
   
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
   HYPRE_Int                u_end_location;
   
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
   HYPRE_Int                *S_offd_colmap   = NULL;
   HYPRE_Real               *S_offd_data;
   HYPRE_Int                *col_starts;
   HYPRE_Int                total_rows;
   
   /* memory management */
   HYPRE_Int                initial_alloc    = 0;
   HYPRE_Int                capacity_L;
   HYPRE_Int                capacity_U;
   HYPRE_Int                capacity_S;
   HYPRE_Int                nnz_A            = A_diag_i[n];        
   
   /* reverse permutation array */
   HYPRE_Int                *rperm;
   
   /* start setup
    * get communication stuffs first
    */
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   /* setup if not yet built */
   if(!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }
   
   /* check for correctness */
   if(nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU out of range.\n");
   }
   if(e < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU should not exceed nI.\n");
   }
   
   /* Allocate memory for u_end array */
   u_end_array    = hypre_TAlloc(HYPRE_Int, nLU, HYPRE_MEMORY_HOST);
   
   /* Allocate memory for L,D,U,S factors */
   if(n > 0)
   {
      initial_alloc  = nLU + ceil((nnz_A / 2.0)*nLU/n);
      capacity_S     = m + ceil((nnz_A / 2.0)*m/n);
   }
   capacity_L     = initial_alloc;   
   capacity_U     = initial_alloc;      
   
   D_data         = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_SHARED);
   L_diag_i       = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_SHARED);
   L_diag_j       = hypre_TAlloc(HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
   L_diag_data    = hypre_TAlloc(HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
   U_diag_i       = hypre_TAlloc(HYPRE_Int, n+1, HYPRE_MEMORY_SHARED);
   U_diag_j       = hypre_TAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
   U_diag_data    = hypre_TAlloc(HYPRE_Real, capacity_U, HYPRE_MEMORY_SHARED);
   S_diag_i       = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_SHARED);
   S_diag_j       = hypre_TAlloc(HYPRE_Int, capacity_S, HYPRE_MEMORY_SHARED);
   S_diag_data    = hypre_TAlloc(HYPRE_Real, capacity_S, HYPRE_MEMORY_SHARED);   
                 
   /* allocate working arrays */
   iw             = hypre_TAlloc(HYPRE_Int, 3*n, HYPRE_MEMORY_HOST);
   iL             = iw+n;
   rperm          = iw + 2*n;
   wL             = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
   
   ctrU        = ctrL        = ctrS        = 0;
   L_diag_i[0] = U_diag_i[0] = S_diag_i[0] = 0;
   /* set marker array iw to -1 */
   for( i = 0; i < n; i++ ) 
   {
     iw[i] = -1;
   }   

   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
    * rperm only used for column
   */
   for(i=0; i<n; i++)
   {
     rperm[qperm[i]] = i;   
   }   
   /*---------  Begin Factorization. Work in permuted space  ----*/
   for( ii = 0; ii < nLU; ii++ ) 
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
      for(j=k1; j < k2; j++) 
      {
         col = rperm[A_diag_j[j]];
         t = A_diag_data[j];
         if( col < ii ) 
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
      hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      for(j=0; j<lenl; j++)
      {   
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;
                                         
         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for(k=U_diag_i[jpiv]; k<U_diag_i[jpiv+1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if(jpos < 0) 
            {
               continue;
            }
            
            lxu = - U_diag_data[k] * dpiv;
            if(col < ii)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if(col > ii)
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
      for( j = 0; j < lenu; j++ ) 
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      if(lenl > 0)
      {
         while((ctrL+lenl) > capacity_L)
         {
            capacity_L = capacity_L * EXPAND_FACT + 1;         
            L_diag_j = hypre_TReAlloc(L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
            L_diag_data = hypre_TReAlloc(L_diag_data, HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
         }
         hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      }
      L_diag_i[ii+1] = (ctrL+=lenl); 

      /* diagonal part (we store the inverse) */
      if(fabs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;     
      }
      D_data[ii] = 1./dd;

      /* U part */
      /* Check that memory is sufficient */
      if(lenu > 0)
      {
         while((ctrU+lenu) > capacity_U)
         {
            capacity_U = capacity_U * EXPAND_FACT + 1;
            U_diag_j = hypre_TReAlloc(U_diag_j, HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
            U_diag_data = hypre_TReAlloc(U_diag_data, HYPRE_Real, capacity_U, HYPRE_MEMORY_SHARED);
         } 
         hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      }
      U_diag_i[ii+1] = (ctrU+=lenu); 
      
      /* check and build u_end array */
      if(m > 0)
      {
         hypre_qsort1(U_diag_j,U_diag_data,U_diag_i[ii],U_diag_i[ii+1]-1);
         u_end_location = hypre_BinarySearch2(U_diag_j,nLU,U_diag_i[ii],U_diag_i[ii+1]-1,u_end_array + ii);
         if(u_end_location >= 0)
         {
            u_end_array[ii] = u_end_location + 1;
         }
      }
      else
      {
         /* Everything is in U */
         u_end_array[ii] = ctrU;
      }
      
   }
   /*---------  Begin Factorization in Schur Complement part  ----*/
   for( ii = nLU; ii < n; ii++ ) 
   {
      // get row i
      i = perm[ii];
      // get extents of row i    
      k1=A_diag_i[i];   
      k2=A_diag_i[i+1]; 

/*-------------------- unpack L & U-parts of row of A in arrays w */
      iU = iL+nLU + 1;
      wU = wL+nLU + 1;
/*--------------------  diagonal entry */
      dd = 0.0;
      lenl  = lenu = 0;
      iw[ii] = nLU;
/*-------------------- scan & unwrap column */
      for(j=k1; j < k2; j++) 
      {
         col = rperm[A_diag_j[j]];
         t = A_diag_data[j];
         if( col < nLU ) 
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
      hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      for(j=0; j<lenl; j++)
      {   
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;
                                         
         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for(k=U_diag_i[jpiv]; k<U_diag_i[jpiv+1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if(jpos < 0) 
            {  
               continue;
            }
            
            lxu = - U_diag_data[k] * dpiv;
            if(col < nLU)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if(col != ii)
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
      for( j = 0; j < lenu; j++ ) 
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      if(lenl > 0)
      {
         while((ctrL+lenl) > capacity_L)
         {
            capacity_L = capacity_L * EXPAND_FACT + 1;         
            L_diag_j = hypre_TReAlloc(L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
            L_diag_data = hypre_TReAlloc(L_diag_data, HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
         }
         hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
         hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      }
      L_diag_i[ii+1] = (ctrL+=lenl); 

      /* S part */
      /* Check that memory is sufficient */
      while((ctrS+lenu+1) > capacity_S)
      {
         capacity_S = capacity_S * EXPAND_FACT + 1;
         S_diag_j = hypre_TReAlloc(S_diag_j, HYPRE_Int, capacity_S, HYPRE_MEMORY_SHARED);
         S_diag_data = hypre_TReAlloc(S_diag_data, HYPRE_Real, capacity_S, HYPRE_MEMORY_SHARED);
      } 
      /* remember S in under a new index system! */
      S_diag_j[ctrS] = ii - nLU;
      S_diag_data[ctrS] = dd;
      for(j = 0 ; j< lenu ; j ++)
      {
         S_diag_j[ctrS+1+j] = iU[j] - nLU;
      }
      //hypre_TMemcpy(S_diag_j+ctrS+1, iU, HYPRE_Int, lenu, HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(S_diag_data+ctrS+1, wU, HYPRE_Real, lenu, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      S_diag_i[ii-nLU+1] = ctrS+=(lenu+1); 
   }
   /* Assemble LDUS matrices */
   /* zero out unfactored rows for U and D */
   for(k=nLU; k<n; k++)
   {
      U_diag_i[k+1] = ctrU;
      D_data[k] = 1.;
   }
   
   /* now create S */
   /* col_starts is the new local start and end for matS */
   col_starts = hypre_CTAlloc(HYPRE_Int,2,HYPRE_MEMORY_HOST);
   hypre_MPI_Scan(&m, &total_rows, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   col_starts[1] = total_rows;
   col_starts[0] = total_rows - m;
   /* now need to get the total length */
   hypre_MPI_Allreduce(&m, &total_rows, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   
   /* only form S when at least one block has nLU < n */
   if( total_rows > 0 )
   { 
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
         
      /* S owns different start/end */
      hypre_ParCSRMatrixSetColStartsOwner(matS,1);
      hypre_ParCSRMatrixSetRowStartsOwner(matS,0);/* square matrix, use same row and col start */
      
      /* first put diagonal data in */
      S_diag = hypre_ParCSRMatrixDiag(matS);
      
      hypre_CSRMatrixI(S_diag) = S_diag_i;
      hypre_CSRMatrixData(S_diag) = S_diag_data; 
      hypre_CSRMatrixJ(S_diag) = S_diag_j;
      
      /* now start to construct offdiag of S */
      S_offd = hypre_ParCSRMatrixOffd(matS);
      S_offd_i = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_SHARED);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_SHARED);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_SHARED);
      S_offd_colmap = hypre_CTAlloc(HYPRE_Int, S_offd_ncols, HYPRE_MEMORY_HOST);
      
      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for(i = 1 ; i <= e ; i ++)
      {
         S_offd_i[i] = k3;
      }
      for(i = 0 ; i < m_e ; i ++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col+1];
         for(j = k1 ; j < k2 ; j ++)
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
      send_buf = hypre_TAlloc(HYPRE_Int, end - begin, HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for(i = begin ; i < end ; i ++)
      {
         send_buf[i-begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)] - nLU + col_starts[0]; 
      }
         
      /* main communication */
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_buf, S_offd_colmap);
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;
      
      hypre_ILUSortOffdColmap(matS);
      
      /* free */
      hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
   }/* end of forming S */
   
   /* create S finished */

   matL = hypre_ParCSRMatrixCreate( comm,
                       hypre_ParCSRMatrixGlobalNumRows(A),
                       hypre_ParCSRMatrixGlobalNumRows(A),
                       hypre_ParCSRMatrixRowStarts(A),
                       hypre_ParCSRMatrixColStarts(A),
                       0,
                       ctrL,
                       0 );

   /* Have A own row/col partitioning instead of L */
   hypre_ParCSRMatrixSetColStartsOwner(matL,0);
   hypre_ParCSRMatrixSetRowStartsOwner(matL,0);
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
      hypre_TFree(L_diag_j,HYPRE_MEMORY_SHARED);
      hypre_TFree(L_diag_data,HYPRE_MEMORY_SHARED);
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

   /* Have A own row/col partitioning instead of U */
   hypre_ParCSRMatrixSetColStartsOwner(matU,0);
   hypre_ParCSRMatrixSetRowStartsOwner(matU,0);
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
      hypre_TFree(U_diag_j,HYPRE_MEMORY_SHARED);
      hypre_TFree(U_diag_data,HYPRE_MEMORY_SHARED);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) ctrU;
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;
   /* free memory */
   hypre_TFree(wL,HYPRE_MEMORY_HOST);
   hypre_TFree(iw,HYPRE_MEMORY_HOST);  
   if(!matS)
   {
      /* we allocate some memory for S, need to free if unused */
      hypre_TFree(S_diag_i,HYPRE_MEMORY_SHARED);
      hypre_TFree(col_starts,HYPRE_MEMORY_HOST);
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
   HYPRE_Int         u_end_location;
   
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
   if(n > 0)
   {
      initial_alloc  = nLU + ceil((nnz_A / 2.0) * nLU / n);
   }
   capacity_L        = initial_alloc;
   capacity_U        = initial_alloc;
   
   /* allocate other memory for L and U struct */
   temp_L_diag_j     = hypre_CTAlloc(HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
   temp_U_diag_j     = hypre_CTAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
   
   if(m > 0)
   {
      capacity_S     = m + ceil(nnz_A / 2.0 * m / n);
      temp_S_diag_j  = hypre_CTAlloc(HYPRE_Int, capacity_S, HYPRE_MEMORY_SHARED);
   }
   
   u_end_array       = hypre_TAlloc(HYPRE_Int, nLU, HYPRE_MEMORY_HOST);
   u_levels          = hypre_CTAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
   ctrL = ctrU = ctrS = 0;
   
   /* set initial value for working array */
   for(ii = 0 ; ii < n ; ii ++)
   {
      iw[ii] = -1;
   }
   
   /*
    * 2: Start of main loop
    * those in iL are NEW col index (after permutation)
    */
   for(ii = 0 ; ii < nLU ; ii ++)
   {
      i = perm[ii];
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = ii;
      lena = A_diag_i[i+1];
      /* put those already inside original pattern, and set their level to 0 */
      for(j = A_diag_i[i] ; j < lena ; j ++)
      {
         /* get the neworder of that col */
         col = rperm[A_diag_j[j]];
         if(col < ii)
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
         else if(col > ii)
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
      while(lenh > 0)
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
         for(j = U_diag_i[k] ; j < ku ; j ++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if(lev > lfil)
            {
               continue;
            }
            if(icol < 0)
            {
               /* not yet in */
               if(col < ii)
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
               else if(col > ii)
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
      if(lenl > 0)
      {
         /* check if memory is enough */
         while(ctrL + lenl > capacity_L)
         {
            capacity_L = capacity_L * EXPAND_FACT + 1;
            temp_L_diag_j = hypre_TReAlloc(temp_L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
         }
         /* now copy L data, reverse order */
         for(j = 0 ; j < lenl ; j ++)
         {
            temp_L_diag_j[ctrL+j] = iL[ii-j-1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii+1] = U_diag_i[ii] + k;
      if(k > 0)
      {
         /* check if memory is enough */
         while(ctrU + k > capacity_U)
         {
            capacity_U = capacity_U * EXPAND_FACT + 1;
            temp_U_diag_j = hypre_TReAlloc(temp_U_diag_j, HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
            u_levels = hypre_TReAlloc(u_levels, HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
         }
         hypre_TMemcpy(temp_U_diag_j+ctrU,iL+ii,HYPRE_Int,k,HYPRE_MEMORY_SHARED,HYPRE_MEMORY_HOST);
         hypre_TMemcpy(u_levels+ctrU,iLev+ii,HYPRE_Int,k,HYPRE_MEMORY_HOST,HYPRE_MEMORY_HOST);
         ctrU += k;
      }
      if(m > 0)
      {
         hypre_qsort2i(temp_U_diag_j,u_levels,U_diag_i[ii],U_diag_i[ii+1]-1);
         u_end_location = hypre_BinarySearch2(temp_U_diag_j,nLU,U_diag_i[ii],U_diag_i[ii+1]-1,u_end_array + ii);
         if(u_end_location >= 0)
         {
            u_end_array[ii] = u_end_location + 1;
         }
      }
      else
      {
         /* Everything is in U */
         u_end_array[ii] = ctrU;
      }
      
      /* reset iw */
      for(j = ii ; j < lenu ; j ++)
      {
         iw[iL[j]] = -1;
      }
      
   }/* end of main loop ii from 0 to nLU-1 */
   
   /* another loop to set EU^-1 and Schur complement */
   for(ii = nLU ; ii < n ; ii ++)
   {
      i = perm[ii];
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = nLU;/* now this stores S, start from nLU */
      lena = A_diag_i[i+1];
      /* put those already inside original pattern, and set their level to 0 */
      for(j = A_diag_i[i] ; j < lena ; j ++)
      {
         /* get the neworder of that col */
         col = rperm[A_diag_j[j]];
         if(col < nLU)
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
         else if(col != ii) /* we for sure to add ii, avoid duplicate */
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
      while(lenh > 0)
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
         for(j = U_diag_i[k] ; j < ku ; j ++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if(lev > lfil)
            {
               continue;
            }
            if(icol < 0)
            {
               /* not yet in */
               if(col < nLU)
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
               else if(col != ii)
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
      if(lenl > 0)
      {
         /* check if memory is enough */
         while(ctrL + lenl > capacity_L)
         {
            capacity_L = capacity_L * EXPAND_FACT + 1;
            temp_L_diag_j = hypre_TReAlloc(temp_L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
         }
         /* now copy L data, reverse order */
         for(j = 0 ; j < lenl ; j ++)
         {
            temp_L_diag_j[ctrL+j] = iL[nLU-j-1];
         }
         ctrL += lenl;
      }
      k = lenu - nLU + 1;
      /* check if memory is enough */
      while(ctrS + k > capacity_S)
      {
         capacity_S = capacity_S * EXPAND_FACT + 1;
         temp_S_diag_j = hypre_TReAlloc(temp_S_diag_j, HYPRE_Int, capacity_S, HYPRE_MEMORY_SHARED);
      }
      temp_S_diag_j[ctrS] = ii;/* must have diagonal */
      hypre_TMemcpy(temp_S_diag_j+ctrS+1,iL+nLU,HYPRE_Int,k-1,HYPRE_MEMORY_SHARED,HYPRE_MEMORY_HOST);
      ctrS += k;
      S_diag_i[ii-nLU+1] = ctrS;
      
      /* reset iw */
      for(j = nLU ; j < lenu ; j ++)
      {
         iw[iL[j]] = -1;
      }
      
   }/* end of main loop ii from nLU to n-1 */
   
   /*
    * 3: Update the struct for L, U and S
    */
   for(k = nLU ; k < n ; k ++)
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
 * perm: permutation array indicating ordering of factorization. Perm could come from a 
 *    CF_marker: array or a reordering routine.
 * qperm: column permutation array.
 * nLU: size of computed LDU factorization.
 * nI: number of interial unknowns, nI should obey nI >= nLU
 * Lptr, Dptr, Uptr: L, D, U factors.
 * Sprt: Schur Complement, if no Schur Complement is needed it will be set to NULL
 */
HYPRE_Int
hypre_ILUSetupILUK(hypre_ParCSRMatrix *A, HYPRE_Int lfil, HYPRE_Int *perm, HYPRE_Int *qperm, HYPRE_Int nLU, HYPRE_Int nI,
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
   if(lfil == 0)
   {
      return hypre_ILUSetupILU0(A,perm,qperm,nLU,nI,Lptr,Dptr,Uptr,Sptr,u_end);
   }
   HYPRE_Real              local_nnz, total_nnz;
   HYPRE_Int               i, ii, j, k, k1, k2, k3, kl, ku, jpiv, col, icol;
   HYPRE_Int               *iw;
   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);

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
   HYPRE_Int               *S_offd_colmap = NULL;
   HYPRE_Real              *S_offd_data;
   HYPRE_Int               S_offd_nnz, S_offd_ncols;
   HYPRE_Int               *col_starts;
   HYPRE_Int               total_rows;
   /* communication */
   hypre_ParCSRCommPkg     *comm_pkg;
   hypre_ParCSRCommHandle  *comm_handle;
   HYPRE_Int               *send_buf      = NULL;
   
   /* problem size */
   HYPRE_Int               n;
   HYPRE_Int               m;
   HYPRE_Int               e;
   HYPRE_Int               m_e;
   /* reverse permutation array */
   HYPRE_Int               *rperm;
   
   /* start setup */
   /* check input and get problem size */
   n =  hypre_CSRMatrixNumRows(A_diag);
   if(nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU out of range.\n");
   }
   m = n - nLU;
   e = nI - nLU;
   m_e = n - nI;
   if(e < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU should not exceed nI.\n");
   }

   /* Init I array anyway. S's might be freed later */
   D_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_SHARED);
   L_diag_i = hypre_CTAlloc(HYPRE_Int, (n+1), HYPRE_MEMORY_SHARED);
   U_diag_i = hypre_CTAlloc(HYPRE_Int, (n+1), HYPRE_MEMORY_SHARED);
   S_diag_i = hypre_CTAlloc(HYPRE_Int, (m+1), HYPRE_MEMORY_SHARED);
   
   /* set Comm_Pkg if not yet built */
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   if(!comm_pkg)
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
   for(i=0; i<n; i++)
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
   if(L_diag_i[n])
   { 
      L_diag_data = hypre_CTAlloc(HYPRE_Real, L_diag_i[n], HYPRE_MEMORY_SHARED);
   }
   if(U_diag_i[n])
   {
      U_diag_data = hypre_CTAlloc(HYPRE_Real, U_diag_i[n], HYPRE_MEMORY_SHARED);
   }
   if(S_diag_i[m])
   {
      S_diag_data = hypre_CTAlloc(HYPRE_Real, S_diag_i[m], HYPRE_MEMORY_SHARED);
   }
   
   /*
    * 3: Begin real factorization
    * we already have L and U structure ready, so no extra working array needed 
    */  
   /* first loop for upper part */
   for( ii = 0; ii < nLU; ii++ ) 
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii+1];
      ku = U_diag_i[ii+1];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      /* set up working arrays */
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      D_data[ii] = 0.0;
      iw[ii] = ii;
      for(j = U_diag_i[ii] ; j < ku ; j ++)
      {
         col = U_diag_j[j];
         iw[col] = j;
      }
      /* copy data from A into L, D and U */
      for(j = k1 ; j < k2 ; j ++)
      {
         /* compute everything in new index */
         col = rperm[A_diag_j[j]];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if(col < ii)
         {
            L_diag_data[icol] = A_diag_data[j];
         }
         else if(col == ii)
         {
            D_data[ii] = A_diag_data[j];
         }
         else
         {
            U_diag_data[icol] = A_diag_data[j];
         }
      }
      /* elimination */
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv+1];
         
         for(k = U_diag_i[jpiv] ; k < ku ; k ++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if(icol < 0)
            {
               /* not in partern */
               continue;
            }
            if(col < ii)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
            else if(col == ii)
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
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      iw[ii] = -1;
      for(j = U_diag_i[ii] ; j < ku ; j ++)
      {
         col = U_diag_j[j];
         iw[col] = -1;
      }

      /* diagonal part (we store the inverse) */
      if(fabs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1e-06;
      }
      D_data[ii] = 1./ D_data[ii];
      
   }
   
   /* Now lower part for Schur complement */
   for( ii = nLU; ii < n; ii++ ) 
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii+1];
      ku = S_diag_i[ii - nLU +1];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      /* set up working arrays */
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      for(j = S_diag_i[ii - nLU] ; j < ku ; j ++)
      {
         col = S_diag_j[j];
         iw[col] = j;
      }
      /* copy data from A into L, and S */
      for(j = k1 ; j < k2 ; j ++)
      {
         /* compute everything in new index */
         col = rperm[A_diag_j[j]];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if(col < nLU)
         {
            L_diag_data[icol] = A_diag_data[j];
         }
         else
         {
            S_diag_data[icol] = A_diag_data[j];
         }
      }
      /* elimination */
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv+1];
         for(k = U_diag_i[jpiv] ; k < ku ; k ++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if(icol < 0)
            {
               /* not in partern */
               continue;
            }
            if(col < nLU)
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
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      ku = S_diag_i[ii-nLU+1];
      for(j = S_diag_i[ii-nLU] ; j < ku ; j ++)
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
   col_starts = hypre_CTAlloc(HYPRE_Int,2,HYPRE_MEMORY_HOST);
   /* use scan to get local start and end */
   hypre_MPI_Scan(&m, &total_rows, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   col_starts[1] = total_rows;
   col_starts[0] = total_rows - m;
   /* now need to get the total length */
   hypre_MPI_Allreduce(&m, &total_rows, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   
   /* only form when total_rows > 0 */
   if( total_rows > 0 )
   {
      
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
         
      /* S owns different start/end */
      hypre_ParCSRMatrixSetColStartsOwner(matS,1);
      hypre_ParCSRMatrixSetRowStartsOwner(matS,0);/* square matrix, use same row and col start */
      
      /* first put diagonal data in */
      S_diag = hypre_ParCSRMatrixDiag(matS);
      
      hypre_CSRMatrixI(S_diag) = S_diag_i;
      hypre_CSRMatrixData(S_diag) = S_diag_data; 
      hypre_CSRMatrixJ(S_diag) = S_diag_j;
      
      /* now start to construct offdiag of S */
      S_offd = hypre_ParCSRMatrixOffd(matS);
      S_offd_i = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_SHARED);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_SHARED);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_SHARED);
      S_offd_colmap = hypre_CTAlloc(HYPRE_Int, S_offd_ncols, HYPRE_MEMORY_HOST);
      
      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for(i = 1 ; i <= e ; i ++)
      {
         S_offd_i[i+1] = k3;
      }
      for(i = 0 ; i < m_e ; i ++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col+1];
         for(j = k1 ; j < k2 ; j ++)
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
      send_buf = hypre_TAlloc(HYPRE_Int, end - begin, HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for(i = begin ; i < end ; i ++)
      {
         send_buf[i-begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)] - nLU + col_starts[0]; 
      }
         
      /* main communication */
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_buf, S_offd_colmap);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      
      /* setup index */
      hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;
      
      hypre_ILUSortOffdColmap(matS);
      
      /* free */
      hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
   }/* end of forming S */
   
   /* Assemble LDU matrices */
   /* zero out unfactored rows */
   for(k=nLU; k<n; k++)
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

   /* Have A own coarse_partitioning instead of L */
   hypre_ParCSRMatrixSetColStartsOwner(matL,0);
   hypre_ParCSRMatrixSetRowStartsOwner(matL,0);
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
      hypre_TFree(L_diag_j, HYPRE_MEMORY_SHARED);
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

   /* Have A own coarse_partitioning instead of U */
   hypre_ParCSRMatrixSetColStartsOwner(matU,0);
   hypre_ParCSRMatrixSetRowStartsOwner(matU,0);

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
      hypre_TFree(U_diag_j, HYPRE_MEMORY_SHARED);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (U_diag_i[n]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;  
   
   /* free */
   hypre_TFree(iw,HYPRE_MEMORY_HOST);
   if(!matS)
   {
      /* we allocate some memory for S, need to free if unused */
      hypre_TFree(S_diag_i,HYPRE_MEMORY_SHARED);
      hypre_TFree(col_starts,HYPRE_MEMORY_HOST);
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
      HYPRE_Int *perm, HYPRE_Int *qperm, HYPRE_Int nLU, HYPRE_Int nI, hypre_ParCSRMatrix **Lptr, 
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
   HYPRE_Int                *col_starts;
   HYPRE_Int                total_rows;
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
   HYPRE_Int                *S_offd_colmap   = NULL;
   HYPRE_Real               *S_offd_data;
   HYPRE_Int                *send_buf        = NULL;
   HYPRE_Int                *u_end_array;
   HYPRE_Int                u_end_location;
   
   /* reverse permutation */
   HYPRE_Int                *rperm;
   
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
   if(nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU out of range.\n");
   }
   m = n - nLU;
   e = nI - nLU;
   m_e = n - nI;
   if(e < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU should not exceed nI.\n");
   }
   
   u_end_array = hypre_TAlloc(HYPRE_Int, nLU, HYPRE_MEMORY_HOST);
   
   /* start set up
    * setup communication stuffs first
    */
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   /* create if not yet built */
   if(!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }
   
   /* setup initial memory, in ILUT, just guess with max nnz per row */
   nnz_A = A_diag_i[nLU];
   if(n > 0)
   {
      initial_alloc = hypre_min(nLU + ceil((nnz_A / 2.0) * nLU / n), nLU * lfil);
   }
   capacity_L = initial_alloc;
   capacity_U = initial_alloc;
   
   D_data = hypre_CTAlloc(HYPRE_Real, n, HYPRE_MEMORY_SHARED);
   L_diag_i = hypre_CTAlloc(HYPRE_Int, (n+1), HYPRE_MEMORY_SHARED);
   U_diag_i = hypre_CTAlloc(HYPRE_Int, (n+1), HYPRE_MEMORY_SHARED);
   
   L_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
   U_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
   L_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
   U_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_U, HYPRE_MEMORY_SHARED);
   
   ctrL = ctrU = 0;
   
   ctrS = 0;
   S_diag_i = hypre_CTAlloc(HYPRE_Int, (m + 1), HYPRE_MEMORY_SHARED);
   S_diag_i[0] = 0;
   /* only setup S part when n > nLU */
   if(m > 0)
   {
      capacity_S = hypre_min(m + ceil((nnz_A / 2.0) * m / n), m * lfil);
      S_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_S, HYPRE_MEMORY_SHARED);
      S_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_S, HYPRE_MEMORY_SHARED);
   }
   
   /* setting up working array */
   iw = hypre_CTAlloc(HYPRE_Int,3*n,HYPRE_MEMORY_HOST);
   iL = iw + n;
   w = hypre_CTAlloc(HYPRE_Real,n,HYPRE_MEMORY_HOST);
   for(i = 0 ; i < n ; i ++)
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
   for(i = 0 ; i < n ; i ++)
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
   for(ii = 0 ; ii < nLU ; ii ++)
   {
      /* get real row with perm */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      kl = ii-1;
      /* reset row norm of ith row */
      inorm = .0;
      for(j = k1 ; j < k2 ; j ++)
      {
         inorm += fabs(A_diag_data[j]);
      }
      if(inorm == .0)
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
      for(j = k1 ; j < k2 ; j ++)
      {
         /* get now col number */
         col = rperm[A_diag_j[j]];
         if(col < ii)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = A_diag_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
         }
         else if(col == ii)
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
      while(lenhll > 0)
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
         for(j = U_diag_i[jrow] ; j < ku ; j ++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv*U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if( icol == -1 && ( (col < nLU && fabs(lxu) < itolb) || (col >= nLU && fabs(lxu) < itolef) ) )
            {
               continue;
            }
            if(icol == -1)
            {
               if(col < ii)
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
               else if(col == ii)
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
      
      if(fabs(w[ii]) < MAT_TOL)
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
      if(lenl > 0)
      {
         /* test if memory is enough */
         while(ctrL + lenl > capacity_L)
         {
            capacity_L = capacity_L * EXPAND_FACT + 1;
            L_diag_j = hypre_TReAlloc(L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
            L_diag_data = hypre_TReAlloc(L_diag_data, HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
         }
         ctrL += lenl;
         /* copy large data in */
         for(j = L_diag_i[ii] ; j < ctrL ; j ++)
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
      for(j = ii + 1 ; j <= ku ; j ++)
      {
         iw[iL[j]] = -1;
      }
      
      if(lenu < lfil)
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
      if(lenhu > 0)
      {
        /* test if memory is enough */
         while(ctrU + lenhu > capacity_U)
         {
            capacity_U = capacity_U * EXPAND_FACT + 1;
            U_diag_j = hypre_TReAlloc(U_diag_j, HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
            U_diag_data = hypre_TReAlloc(U_diag_data, HYPRE_Real, capacity_U, HYPRE_MEMORY_SHARED);
         }
         ctrU += lenhu;
         /* copy large data in */
         for(j = U_diag_i[ii] ; j < ctrU ; j ++)
         {
            jpos = ii+1+j-U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
      /* check and build u_end array */
      if(m > 0)
      {
         hypre_qsort1(U_diag_j,U_diag_data,U_diag_i[ii],U_diag_i[ii+1]-1);
         u_end_location = hypre_BinarySearch2(U_diag_j,nLU,U_diag_i[ii],U_diag_i[ii+1]-1,u_end_array + ii);
         if(u_end_location >= 0)
         {
            u_end_array[ii] = u_end_location + 1;
         }
      }
      else
      {
         /* Everything is in U */
         u_end_array[ii] = ctrU;
      }
   }/* end of ii loop from 0 to nLU-1 */
   
   
   /* now main loop for Schur comlement part */
   for(ii = nLU ; ii < n ; ii ++)
   {
      /* get real row with perm */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      kl = nLU-1;
      /* reset row norm of ith row */
      inorm = .0;
      for(j = k1 ; j < k2 ; j ++)
      {
         inorm += fabs(A_diag_data[j]);
      }
      if(inorm == .0)
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
      for(j = k1 ; j < k2 ; j ++)
      {
         /* get now col number */
         col = rperm[A_diag_j[j]];
         if(col < nLU)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = A_diag_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
         }
         else if(col == ii)
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
      while(lenhll > 0)
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
         for(j = U_diag_i[jrow] ; j < ku ; j ++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv*U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if(icol == -1 && ( (col < nLU && fabs(lxu) < itolef) || ( col >= nLU && fabs(lxu) < itols ) ) )
            {
               continue;
            }
            if(icol == -1)
            {
               if(col < nLU)
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
               else if(col == ii)
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
      if(lenl > 0)
      {
         /* test if memory is enough */
         while(ctrL + lenl > capacity_L)
         {
            capacity_L = capacity_L * EXPAND_FACT + 1;
            L_diag_j = hypre_TReAlloc(L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
            L_diag_data = hypre_TReAlloc(L_diag_data, HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
         }
         ctrL += lenl;
         /* copy large data in */
         for(j = L_diag_i[ii] ; j < ctrL ; j ++)
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
      for(j = nLU ; j <= ku ; j ++)
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
      while(ctrS + lenhu + 1 > capacity_S)
      {
         capacity_S = capacity_S * EXPAND_FACT + 1;
         S_diag_j = hypre_TReAlloc(S_diag_j, HYPRE_Int, capacity_S, HYPRE_MEMORY_SHARED);
         S_diag_data = hypre_TReAlloc(S_diag_data, HYPRE_Real, capacity_S, HYPRE_MEMORY_SHARED);
      }
      
      ctrS += (lenhu+1);
      S_diag_i[ii-nLU+1] = ctrS;
      
      /* copy large data in, diagonal first */
      S_diag_j[S_diag_i[ii-nLU]] = iL[nLU]-nLU;
      S_diag_data[S_diag_i[ii-nLU]] = w[nLU];
      for(j = S_diag_i[ii-nLU] + 1 ; j < ctrS ; j ++)
      {
         jpos = nLU+j-S_diag_i[ii-nLU];
         S_diag_j[j] = iL[jpos]-nLU;
         S_diag_data[j] = w[jpos];
      }
   }/* end of ii loop from nLU to n-1 */
   
   /*
    * 3: Finishing up and free
    */
   
   /* First create Schur Complement if needed
    * Check if we need to create Schur complement
    * Some might not holding Schur Complement, need a new comm
    */
   col_starts = hypre_CTAlloc(HYPRE_Int,2,HYPRE_MEMORY_HOST);
   /* use scan to get local start and end */
   hypre_MPI_Scan(&m, &total_rows, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   col_starts[1] = total_rows;
   col_starts[0] = total_rows - m;
   /* now need to get the total length */
   hypre_MPI_Allreduce(&m, &total_rows, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   
   /* only form when total_rows > 0 */
   if( total_rows > 0 )
   {
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
         
      /* S owns different start/end */
      hypre_ParCSRMatrixSetColStartsOwner(matS,1);
      hypre_ParCSRMatrixSetRowStartsOwner(matS,0);/* square matrix, use same row and col start */
      
      /* first put diagonal data in */
      S_diag = hypre_ParCSRMatrixDiag(matS);
      
      hypre_CSRMatrixI(S_diag) = S_diag_i;
      hypre_CSRMatrixData(S_diag) = S_diag_data; 
      hypre_CSRMatrixJ(S_diag) = S_diag_j;
      
      /* now start to construct offdiag of S */
      S_offd = hypre_ParCSRMatrixOffd(matS);
      S_offd_i = hypre_TAlloc(HYPRE_Int, m+1, HYPRE_MEMORY_SHARED);
      S_offd_j = hypre_TAlloc(HYPRE_Int, S_offd_nnz, HYPRE_MEMORY_SHARED);
      S_offd_data = hypre_TAlloc(HYPRE_Real, S_offd_nnz, HYPRE_MEMORY_SHARED);
      S_offd_colmap = hypre_CTAlloc(HYPRE_Int, S_offd_ncols, HYPRE_MEMORY_HOST);
      
      /* simply use a loop to copy data from A_offd */
      S_offd_i[0] = 0;
      k3 = 0;
      for(i = 1 ; i <= e ; i ++)
      {
         S_offd_i[i] = k3;
      }
      for(i = 0 ; i < m_e ; i ++)
      {
         col = perm[i + nI];
         k1 = A_offd_i[col];
         k2 = A_offd_i[col+1];
         for(j = k1 ; j < k2 ; j ++)
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
      send_buf = hypre_TAlloc(HYPRE_Int, end - begin, HYPRE_MEMORY_HOST);
      /* copy new index into send_buf */
      for(i = begin ; i < end ; i ++)
      {
         send_buf[i-begin] = rperm[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)] - nLU + col_starts[0]; 
      }
         
      /* main communication */
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, send_buf, S_offd_colmap);
      /* need this to synchronize, Isend & Irecv used in above functions */
      hypre_ParCSRCommHandleDestroy(comm_handle);

      /* setup index */
      hypre_ParCSRMatrixColMapOffd(matS) = S_offd_colmap;
      
      hypre_ILUSortOffdColmap(matS);
      
      /* free */
      hypre_TFree(send_buf, HYPRE_MEMORY_HOST);
   }/* end of forming S */
   
   /* now start to construct L and U */
   for(k=nLU; k<n; k++)
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

   /* Have A own coarse_partitioning instead of L */
   hypre_ParCSRMatrixSetColStartsOwner(matL,0);
   hypre_ParCSRMatrixSetRowStartsOwner(matL,0);
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
      hypre_TFree(L_diag_j,HYPRE_MEMORY_SHARED);
      hypre_TFree(L_diag_data,HYPRE_MEMORY_SHARED);
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

   /* Have A own coarse_partitioning instead of U */
   hypre_ParCSRMatrixSetColStartsOwner(matU,0);
   hypre_ParCSRMatrixSetRowStartsOwner(matU,0);

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
      hypre_TFree(U_diag_j,HYPRE_MEMORY_SHARED);
      hypre_TFree(U_diag_data,HYPRE_MEMORY_SHARED);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (U_diag_i[n]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;
   
   /* free working array */
   hypre_TFree(iw,HYPRE_MEMORY_HOST);
   hypre_TFree(w,HYPRE_MEMORY_HOST);
   
   if(!matS)
   {
      hypre_TFree(S_diag_i,HYPRE_MEMORY_SHARED);
      hypre_TFree(col_starts,HYPRE_MEMORY_HOST);
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
   if(matM)
   {
      hypre_TFree(matM, HYPRE_MEMORY_HOST);
      matM = NULL;
   }    

   /* clear old l1_norm data, if created */
   if(hypre_ParNSHDataL1Norms(nsh_data))
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
   hypre_ParVectorSetPartitioningOwner(Utemp,0);
   hypre_ParNSHDataUTemp(nsh_data) = Utemp;

   Ftemp = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                          hypre_ParCSRMatrixGlobalNumRows(A),
                          hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(Ftemp);
   hypre_ParVectorSetPartitioningOwner(Ftemp,0);
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
      hypre_ParVectorSetPartitioningOwner(residual,0);
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
   HYPRE_Real               global_start, global_num_rows;
   HYPRE_Int                *col_starts;
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
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   /* setup if not yet built */
   if(!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }
   
   /* check for correctness */
   if(nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU out of range.\n");
   }
   
   /* Allocate memory for L,D,U,S factors */
   if(n > 0)
   {
      initial_alloc = total_rows + ceil((nnz_A / 2.0)*total_rows/n);
   }
   capacity_L = initial_alloc;   
   capacity_U = initial_alloc;
   
   D_data      = hypre_TAlloc(HYPRE_Real, total_rows, HYPRE_MEMORY_SHARED);
   L_diag_i    = hypre_TAlloc(HYPRE_Int, total_rows+1, HYPRE_MEMORY_SHARED);
   L_diag_j    = hypre_TAlloc(HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
   L_diag_data = hypre_TAlloc(HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
   U_diag_i    = hypre_TAlloc(HYPRE_Int, total_rows+1, HYPRE_MEMORY_SHARED);
   U_diag_j    = hypre_TAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
   U_diag_data = hypre_TAlloc(HYPRE_Real, capacity_U, HYPRE_MEMORY_SHARED);
                 
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
   for( i = 0 ; i < total_rows ; i++ ) 
   {
     iw[i] = -1;
   }   
   
   /* expand perm to suit extra data, remember to free */
   for( i = 0 ; i < n ; i ++)
   {
      perm[i] = perm_old[i];
   }
   for( i = n ; i < total_rows ; i ++)
   {
      perm[i] = i;
   }
   
   /* get reverse permutation (rperm).
    * rperm holds the reordered indexes.
   */
   for(i=0 ; i < total_rows ; i++)
   {
     rperm[perm[i]] = i;   
   } 
   
   /* get external rows */
   hypre_ILUBuildRASExternalMatrix(A, rperm, &E_i, &E_j, &E_data);
   
   /*---------  Begin Factorization. Work in permuted space  ----
    * this is the first part, without offd 
    */
   for( ii = 0; ii < nLU; ii++ ) 
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
      for(j=k1; j < k2; j++) 
      {
         col = rperm[A_diag_j[j]];
         t = A_diag_data[j];
         if( col < ii ) 
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
      hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      for(j=0; j<lenl; j++)
      {   
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;
                                         
         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for(k=U_diag_i[jpiv]; k<U_diag_i[jpiv+1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if(jpos < 0) 
            {
               continue;
            }
            
            lxu = - U_diag_data[k] * dpiv;
            if(col < ii)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if(col > ii)
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
      for( j = 0; j < lenu; j++ ) 
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      while((ctrL+lenl) > capacity_L)
      {
         capacity_L = capacity_L * EXPAND_FACT + 1;         
         L_diag_j = hypre_TReAlloc(L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
         L_diag_data = hypre_TReAlloc(L_diag_data, HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
      }
      hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      L_diag_i[ii+1] = (ctrL+=lenl); 

      /* diagonal part (we store the inverse) */
      if(fabs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;     
      }
      D_data[ii] = 1./dd;

      /* U part */
      /* Check that memory is sufficient */
      while((ctrU+lenu) > capacity_U)
      {
         capacity_U = capacity_U * EXPAND_FACT + 1;
         U_diag_j = hypre_TReAlloc(U_diag_j, HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
         U_diag_data = hypre_TReAlloc(U_diag_data, HYPRE_Real, capacity_U, HYPRE_MEMORY_SHARED);
      } 
      hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      U_diag_i[ii+1] = (ctrU+=lenu); 
      
   }
   /*---------  Begin Factorization in lower part  ----
    * here we need to get off diagonals in 
    */
   for( ii = nLU ; ii < n ; ii++ ) 
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
      for(j=k1; j < k2; j++) 
      {
         col = rperm[A_diag_j[j]];
         t = A_diag_data[j];
         if( col < ii ) 
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
      for(j = k1 ; j < k2 ; j ++)
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
      hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      for(j=0; j<lenl; j++)
      {   
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;
                                         
         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for(k=U_diag_i[jpiv]; k<U_diag_i[jpiv+1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if(jpos < 0) 
            {
               continue;
            }
            
            lxu = - U_diag_data[k] * dpiv;
            if(col < ii)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if(col > ii)
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
      for( j = 0; j < lenu; j++ ) 
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      while((ctrL+lenl) > capacity_L)
      {
         capacity_L = capacity_L * EXPAND_FACT + 1;         
         L_diag_j = hypre_TReAlloc(L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
         L_diag_data = hypre_TReAlloc(L_diag_data, HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
      }
      hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      L_diag_i[ii+1] = (ctrL+=lenl); 

      /* diagonal part (we store the inverse) */
      if(fabs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;     
      }
      D_data[ii] = 1./dd;

      /* U part */
      /* Check that memory is sufficient */
      while((ctrU+lenu) > capacity_U)
      {
         capacity_U = capacity_U * EXPAND_FACT + 1;
         U_diag_j = hypre_TReAlloc(U_diag_j, HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
         U_diag_data = hypre_TReAlloc(U_diag_data, HYPRE_Real, capacity_U, HYPRE_MEMORY_SHARED);
      } 
      hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      U_diag_i[ii+1] = (ctrU+=lenu); 
      
   }
   
   /*---------  Begin Factorization in external part  ----
    * here we need to get off diagonals in 
    */
   for( ii = n ; ii < total_rows ; ii++ ) 
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
      for(j=k1; j < k2; j++) 
      {
         col = rperm[E_j[j]];
         t = E_data[j];
         if( col < ii ) 
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
      hypre_quickSortIR(iL, wL, iw, 0, (lenl-1));
      for(j=0; j<lenl; j++)
      {   
         jpiv = iL[j];
         /* get factor/ pivot element */
         dpiv = wL[j] * D_data[jpiv];
         /* store entry in L */
         wL[j] = dpiv;
                                         
         /* zero out element - reset pivot */
         iw[jpiv] = -1;
         /* combine current row and pivot row */
         for(k=U_diag_i[jpiv]; k<U_diag_i[jpiv+1]; k++)
         {
            col = U_diag_j[k];
            jpos = iw[col];

            /* Only fill-in nonzero pattern (jpos != 0) */
            if(jpos < 0) 
            {
               continue;
            }
            
            lxu = - U_diag_data[k] * dpiv;
            if(col < ii)
            {
               /* dealing with L part */
               wL[jpos] += lxu;
            }
            else if(col > ii)
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
      for( j = 0; j < lenu; j++ ) 
      {
         iw[iU[j]] = -1;
      }

      /* Update LDU factors */
      /* L part */
      /* Check that memory is sufficient */
      while((ctrL+lenl) > capacity_L)
      {
         capacity_L = capacity_L * EXPAND_FACT + 1;         
         L_diag_j = hypre_TReAlloc(L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
         L_diag_data = hypre_TReAlloc(L_diag_data, HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
      }
      hypre_TMemcpy(&(L_diag_j)[ctrL], iL, HYPRE_Int, lenl, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(L_diag_data)[ctrL], wL, HYPRE_Real, lenl, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      L_diag_i[ii+1] = (ctrL+=lenl); 

      /* diagonal part (we store the inverse) */
      if(fabs(dd) < MAT_TOL)
      {
         dd = 1.0e-6;     
      }
      D_data[ii] = 1./dd;

      /* U part */
      /* Check that memory is sufficient */
      while((ctrU+lenu) > capacity_U)
      {
         capacity_U = capacity_U * EXPAND_FACT + 1;
         U_diag_j = hypre_TReAlloc(U_diag_j, HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
         U_diag_data = hypre_TReAlloc(U_diag_data, HYPRE_Real, capacity_U, HYPRE_MEMORY_SHARED);
      } 
      hypre_TMemcpy(&(U_diag_j)[ctrU], iU, HYPRE_Int, lenu, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      hypre_TMemcpy(&(U_diag_data)[ctrU], wU, HYPRE_Real, lenu, HYPRE_MEMORY_SHARED, HYPRE_MEMORY_HOST);
      U_diag_i[ii+1] = (ctrU+=lenu); 
      
   }
   
   hypre_MPI_Allreduce( &total_rows, &global_num_rows, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   /* need to get new column start */
   col_starts = hypre_CTAlloc(HYPRE_Int,2,HYPRE_MEMORY_HOST);
   hypre_MPI_Scan( &total_rows, &global_start, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   col_starts[1] = global_start;
   col_starts[0] = global_start - total_rows;

   matL = hypre_ParCSRMatrixCreate( comm,
                       global_num_rows,
                       global_num_rows,
                       col_starts,
                       col_starts,
                       0,
                       ctrL,
                       0 );

   /* Have A own row/col partitioning instead of L */
   hypre_ParCSRMatrixSetColStartsOwner(matL,1);
   hypre_ParCSRMatrixSetRowStartsOwner(matL,0);
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
      hypre_TFree(L_diag_j,HYPRE_MEMORY_SHARED);
      hypre_TFree(L_diag_data,HYPRE_MEMORY_SHARED);
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

   /* Have A own row/col partitioning instead of U */
   hypre_ParCSRMatrixSetColStartsOwner(matU,0);
   hypre_ParCSRMatrixSetRowStartsOwner(matU,0);
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
      hypre_TFree(U_diag_j,HYPRE_MEMORY_SHARED);
      hypre_TFree(U_diag_data,HYPRE_MEMORY_SHARED);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) ctrU;
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;
   /* free memory */
   hypre_TFree(wL,HYPRE_MEMORY_HOST);
   hypre_TFree(iw,HYPRE_MEMORY_HOST);  
   
   /* free external data */
   if(E_i)
   {
      hypre_TFree(E_i, HYPRE_MEMORY_HOST);
   }
   if(E_j)
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
   if( n > 0 )
   {
      initial_alloc  = total_rows + ceil((nnz_A / 2.0) * total_rows / n);
   }
   capacity_L     = initial_alloc;
   capacity_U     = initial_alloc;
   
   /* allocate other memory for L and U struct */
   temp_L_diag_j  = hypre_CTAlloc(HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
   temp_U_diag_j  = hypre_CTAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
   
   u_levels       = hypre_CTAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
   ctrL = ctrU = 0;
   
   /* set initial value for working array */
   for(ii = 0 ; ii < total_rows ; ii ++)
   {
      iw[ii] = -1;
   }
   
   /*
    * 2: Start of main loop
    * those in iL are NEW col index (after permutation)
    */
   for(ii = 0 ; ii < nLU ; ii ++)
   {
      i = perm[ii];
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = ii;
      lena = A_diag_i[i+1];
      /* put those already inside original pattern, and set their level to 0 */
      for(j = A_diag_i[i] ; j < lena ; j ++)
      {
         /* get the neworder of that col */
         col = rperm[A_diag_j[j]];
         if(col < ii)
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
         else if(col > ii)
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
      while(lenh > 0)
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
         for(j = U_diag_i[k] ; j < ku ; j ++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if(lev > lfil)
            {
               continue;
            }
            if(icol < 0)
            {
               /* not yet in */
               if(col < ii)
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
               else if(col > ii)
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
      if(lenl > 0)
      {
         /* check if memory is enough */
         while(ctrL + lenl > capacity_L)
         {
            capacity_L = capacity_L * EXPAND_FACT + 1;
            temp_L_diag_j = hypre_TReAlloc(temp_L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
         }
         /* now copy L data, reverse order */
         for(j = 0 ; j < lenl ; j ++)
         {
            temp_L_diag_j[ctrL+j] = iL[ii-j-1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii+1] = U_diag_i[ii] + k;
      if(k > 0)
      {
         /* check if memory is enough */
         while(ctrU + k > capacity_U)
         {
            capacity_U = capacity_U * EXPAND_FACT + 1;
            temp_U_diag_j = hypre_TReAlloc(temp_U_diag_j, HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
            u_levels = hypre_TReAlloc(u_levels, HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
         }
         hypre_TMemcpy(temp_U_diag_j+ctrU,iL+ii,HYPRE_Int,k,HYPRE_MEMORY_SHARED,HYPRE_MEMORY_HOST);
         hypre_TMemcpy(u_levels+ctrU,iLev+ii,HYPRE_Int,k,HYPRE_MEMORY_HOST,HYPRE_MEMORY_HOST);
         ctrU += k;
      }
      
      /* reset iw */
      for(j = ii ; j < lenu ; j ++)
      {
         iw[iL[j]] = -1;
      }
      
   }/* end of main loop ii from 0 to nLU-1 */
   
   /*
    * Offd part
    */
   for(ii = nLU ; ii < n ; ii ++)
   {
      i = perm[ii];
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = ii;
      lena = A_diag_i[i+1];
      /* put those already inside original pattern, and set their level to 0 */
      for(j = A_diag_i[i] ; j < lena ; j ++)
      {
         /* get the neworder of that col */
         col = rperm[A_diag_j[j]];
         if(col < ii)
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
         else if(col > ii)
         {
            /* this is an entry in U */
            iL[lenu] = col;
            iLev[lenu] = 0;
            iw[col] = lenu++;
         }
      }/* end of j loop for adding pattern in original matrix */
      
      /* put those already inside offd pattern in, and set their level to 0 */
      lena = A_offd_i[i+1];
      for( j = A_offd_i[i] ; j < lena ; j ++ )
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
      while(lenh > 0)
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
         for(j = U_diag_i[k] ; j < ku ; j ++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if(lev > lfil)
            {
               continue;
            }
            if(icol < 0)
            {
               /* not yet in */
               if(col < ii)
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
               else if(col > ii)
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
      if(lenl > 0)
      {
         /* check if memory is enough */
         while(ctrL + lenl > capacity_L)
         {
            capacity_L = capacity_L * EXPAND_FACT + 1;
            temp_L_diag_j = hypre_TReAlloc(temp_L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
         }
         /* now copy L data, reverse order */
         for(j = 0 ; j < lenl ; j ++)
         {
            temp_L_diag_j[ctrL+j] = iL[ii-j-1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii+1] = U_diag_i[ii] + k;
      if(k > 0)
      {
         /* check if memory is enough */
         while(ctrU + k > capacity_U)
         {
            capacity_U = capacity_U * EXPAND_FACT + 1;
            temp_U_diag_j = hypre_TReAlloc(temp_U_diag_j, HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
            u_levels = hypre_TReAlloc(u_levels, HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
         }
         hypre_TMemcpy(temp_U_diag_j+ctrU,iL+ii,HYPRE_Int,k,HYPRE_MEMORY_SHARED,HYPRE_MEMORY_HOST);
         hypre_TMemcpy(u_levels+ctrU,iLev+ii,HYPRE_Int,k,HYPRE_MEMORY_HOST,HYPRE_MEMORY_HOST);
         ctrU += k;
      }
      
      /* reset iw */
      for(j = ii ; j < lenu ; j ++)
      {
         iw[iL[j]] = -1;
      }
      
   }/* end of main loop ii from nLU to n */
   
   /* external part matrix */
   for(ii = n ; ii < total_rows ; ii ++)
   {
      i = ii - n;
      lenl = 0;
      lenh = 0;/* this is the current length of heap */
      lenu = ii;
      lena = E_i[i+1];
      /* put those already inside original pattern, and set their level to 0 */
      for(j = E_i[i] ; j < lena ; j ++)
      {
         /* get the neworder of that col */
         col = E_j[j];
         if(col < ii)
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
         else if(col > ii)
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
      while(lenh > 0)
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
         for(j = U_diag_i[k] ; j < ku ; j ++)
         {
            col = temp_U_diag_j[j];
            lev = u_levels[j] + ilev + 1;
            /* ignore large level */
            icol = iw[col];
            /* skill large level */
            if(lev > lfil)
            {
               continue;
            }
            if(icol < 0)
            {
               /* not yet in */
               if(col < ii)
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
               else if(col > ii)
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
      if(lenl > 0)
      {
         /* check if memory is enough */
         while(ctrL + lenl > capacity_L)
         {
            capacity_L = capacity_L * EXPAND_FACT + 1;
            temp_L_diag_j = hypre_TReAlloc(temp_L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
         }
         /* now copy L data, reverse order */
         for(j = 0 ; j < lenl ; j ++)
         {
            temp_L_diag_j[ctrL+j] = iL[ii-j-1];
         }
         ctrL += lenl;
      }
      k = lenu - ii;
      U_diag_i[ii+1] = U_diag_i[ii] + k;
      if(k > 0)
      {
         /* check if memory is enough */
         while(ctrU + k > capacity_U)
         {
            capacity_U = capacity_U * EXPAND_FACT + 1;
            temp_U_diag_j = hypre_TReAlloc(temp_U_diag_j, HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
            u_levels = hypre_TReAlloc(u_levels, HYPRE_Int, capacity_U, HYPRE_MEMORY_HOST);
         }
         hypre_TMemcpy(temp_U_diag_j+ctrU,iL+ii,HYPRE_Int,k,HYPRE_MEMORY_SHARED,HYPRE_MEMORY_HOST);
         hypre_TMemcpy(u_levels+ctrU,iLev+ii,HYPRE_Int,k,HYPRE_MEMORY_HOST,HYPRE_MEMORY_HOST);
         ctrU += k;
      }
      
      /* reset iw */
      for(j = ii ; j < lenu ; j ++)
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
   if(lfil == 0)
   {
      return hypre_ILUSetupILU0RAS(A,perm,nLU,Lptr,Dptr,Uptr);
   }
   HYPRE_Int               i, ii, j, k, k1, k2, kl, ku, jpiv, col, icol;
   HYPRE_Int               *iw;
   MPI_Comm                comm           = hypre_ParCSRMatrixComm(A);

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
   HYPRE_Real              global_start, global_num_rows;
   HYPRE_Int               *col_starts;
   HYPRE_Real              local_nnz, total_nnz;
   
   /* data objects for E, external matrix */
   HYPRE_Int               *E_i;
   HYPRE_Int               *E_j;
   HYPRE_Real              *E_data;
   
   /* communication */
   hypre_ParCSRCommPkg     *comm_pkg;
//   hypre_ParCSRCommHandle  *comm_handle;
//   HYPRE_Int               *send_buf      = NULL;
   
   /* reverse permutation array */
   HYPRE_Int               *rperm;
   /* temp array for old permutation */
   HYPRE_Int               *perm_old;
   
   /* start setup */
   /* check input and get problem size */
   n =  hypre_CSRMatrixNumRows(A_diag);
   if(nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU out of range.\n");
   }

   /* Init I array anyway. S's might be freed later */
   D_data   = hypre_CTAlloc(HYPRE_Real, total_rows, HYPRE_MEMORY_SHARED);
   L_diag_i = hypre_CTAlloc(HYPRE_Int, (total_rows+1), HYPRE_MEMORY_SHARED);
   U_diag_i = hypre_CTAlloc(HYPRE_Int, (total_rows+1), HYPRE_MEMORY_SHARED);
   
   /* set Comm_Pkg if not yet built */
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   if(!comm_pkg)
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
   for(i=0; i<n; i++)
   {
     perm[i] = perm_old[i];   
   }
   for(i=n; i<total_rows; i++)
   {
     perm[i] = i;   
   }
   for(i=0; i<total_rows; i++)
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
   if(L_diag_i[total_rows])
   { 
      L_diag_data = hypre_CTAlloc(HYPRE_Real, L_diag_i[total_rows], HYPRE_MEMORY_SHARED);
   }
   if(U_diag_i[total_rows])
   {
      U_diag_data = hypre_CTAlloc(HYPRE_Real, U_diag_i[total_rows], HYPRE_MEMORY_SHARED);
   }
   
   /*
    * 3: Begin real factorization
    * we already have L and U structure ready, so no extra working array needed 
    */  
   /* first loop for upper part */
   for( ii = 0; ii < nLU; ii++ ) 
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii+1];
      ku = U_diag_i[ii+1];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      /* set up working arrays */
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      D_data[ii] = 0.0;
      iw[ii] = ii;
      for(j = U_diag_i[ii] ; j < ku ; j ++)
      {
         col = U_diag_j[j];
         iw[col] = j;
      }
      /* copy data from A into L, D and U */
      for(j = k1 ; j < k2 ; j ++)
      {
         /* compute everything in new index */
         col = rperm[A_diag_j[j]];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if(col < ii)
         {
            L_diag_data[icol] = A_diag_data[j];
         }
         else if(col == ii)
         {
            D_data[ii] = A_diag_data[j];
         }
         else
         {
            U_diag_data[icol] = A_diag_data[j];
         }
      }
      /* elimination */
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv+1];
         
         for(k = U_diag_i[jpiv] ; k < ku ; k ++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if(icol < 0)
            {
               /* not in partern */
               continue;
            }
            if(col < ii)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
            else if(col == ii)
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
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      iw[ii] = -1;
      for(j = U_diag_i[ii] ; j < ku ; j ++)
      {
         col = U_diag_j[j];
         iw[col] = -1;
      }

      /* diagonal part (we store the inverse) */
      if(fabs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1e-06;
      }
      D_data[ii] = 1./ D_data[ii];
      
   }/* end of loop for upper part */
   
   /* first loop for upper part */
   for( ii = nLU; ii < n; ii++ ) 
   {
      // get row i
      i = perm[ii];
      kl = L_diag_i[ii+1];
      ku = U_diag_i[ii+1];
      /* set up working arrays */
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      D_data[ii] = 0.0;
      iw[ii] = ii;
      for(j = U_diag_i[ii] ; j < ku ; j ++)
      {
         col = U_diag_j[j];
         iw[col] = j;
      }
      /* copy data from A into L, D and U */
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      for(j = k1 ; j < k2 ; j ++)
      {
         /* compute everything in new index */
         col = rperm[A_diag_j[j]];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if(col < ii)
         {
            L_diag_data[icol] = A_diag_data[j];
         }
         else if(col == ii)
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
      for(j = k1 ; j < k2 ; j ++)
      {
         /* compute everything in new index */
         col = A_offd_j[j] + n;
         icol = iw[col];
         U_diag_data[icol] = A_offd_data[j];
      }
      /* elimination */
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv+1];
         
         for(k = U_diag_i[jpiv] ; k < ku ; k ++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if(icol < 0)
            {
               /* not in partern */
               continue;
            }
            if(col < ii)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
            else if(col == ii)
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
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      iw[ii] = -1;
      for(j = U_diag_i[ii] ; j < ku ; j ++)
      {
         col = U_diag_j[j];
         iw[col] = -1;
      }

      /* diagonal part (we store the inverse) */
      if(fabs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1e-06;
      }
      D_data[ii] = 1./ D_data[ii];
      
   }/* end of loop for lower part */
   
   /* last loop through external */
   for( ii = n; ii < total_rows; ii++ ) 
   {
      // get row i
      i = ii - n;
      kl = L_diag_i[ii+1];
      ku = U_diag_i[ii+1];
      k1 = E_i[i];
      k2 = E_i[i+1];
      /* set up working arrays */
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         col = L_diag_j[j];
         iw[col] = j;
      }
      D_data[ii] = 0.0;
      iw[ii] = ii;
      for(j = U_diag_i[ii] ; j < ku ; j ++)
      {
         col = U_diag_j[j];
         iw[col] = j;
      }
      /* copy data from E into L, D and U */
      for(j = k1 ; j < k2 ; j ++)
      {
         /* compute everything in new index */
         col = E_j[j];
         icol = iw[col];
         /* A for sure to be inside the pattern */
         if(col < ii)
         {
            L_diag_data[icol] = E_data[j];
         }
         else if(col == ii)
         {
            D_data[ii] = E_data[j];
         }
         else
         {
            U_diag_data[icol] = E_data[j];
         }
      }
      /* elimination */
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         jpiv = L_diag_j[j];
         L_diag_data[j] *= D_data[jpiv];
         ku = U_diag_i[jpiv+1];
         
         for(k = U_diag_i[jpiv] ; k < ku ; k ++)
         {
            col = U_diag_j[k];
            icol = iw[col];
            if(icol < 0)
            {
               /* not in partern */
               continue;
            }
            if(col < ii)
            {
               /* L part */
               L_diag_data[icol] -= L_diag_data[j]*U_diag_data[k];
            }
            else if(col == ii)
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
      for(j = L_diag_i[ii] ; j < kl ; j ++)
      {
         col = L_diag_j[j];
         iw[col] = -1;
      }
      iw[ii] = -1;
      for(j = U_diag_i[ii] ; j < ku ; j ++)
      {
         col = U_diag_j[j];
         iw[col] = -1;
      }

      /* diagonal part (we store the inverse) */
      if(fabs(D_data[ii]) < MAT_TOL)
      {
         D_data[ii] = 1e-06;
      }
      D_data[ii] = 1./ D_data[ii];
      
   }/* end of loop for external loop */
   
   /*
    * 4: Finishing up and free
    */
   
   hypre_MPI_Allreduce( &total_rows, &global_num_rows, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   /* need to get new column start */
   col_starts = hypre_CTAlloc(HYPRE_Int,2,HYPRE_MEMORY_HOST);
   hypre_MPI_Scan( &total_rows, &global_start, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   col_starts[1] = global_start;
   col_starts[0] = global_start - total_rows;
   
   /* Assemble LDU matrices */    
   matL = hypre_ParCSRMatrixCreate( comm,
                       global_num_rows,
                       global_num_rows,
                       col_starts,
                       col_starts,
                       0 /* num_cols_offd */,
                       L_diag_i[total_rows],
                       0 /* num_nonzeros_offd */);

   /* Have A own coarse_partitioning instead of L */
   hypre_ParCSRMatrixSetColStartsOwner(matL,1);
   hypre_ParCSRMatrixSetRowStartsOwner(matL,0);
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
      hypre_TFree(L_diag_j, HYPRE_MEMORY_SHARED);
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

   /* Have A own coarse_partitioning instead of U */
   hypre_ParCSRMatrixSetColStartsOwner(matU,0);
   hypre_ParCSRMatrixSetRowStartsOwner(matU,0);

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
      hypre_TFree(U_diag_j, HYPRE_MEMORY_SHARED);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (U_diag_i[total_rows]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;  
   
   /* free */
   hypre_TFree(iw,HYPRE_MEMORY_HOST);
   
   /* free external data */
   if(E_i)
   {
      hypre_TFree(E_i, HYPRE_MEMORY_HOST);
   }
   if(E_j)
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
   hypre_ParCSRCommPkg      *comm_pkg;
//   hypre_ParCSRCommHandle   *comm_handle;
   HYPRE_Int                *col_starts;
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
   HYPRE_Real               global_start, global_num_rows;
   
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
   if(nLU < 0 || nLU > n)
   {
      hypre_error_w_msg(HYPRE_ERROR_ARG,"WARNING: nLU out of range.\n");
   }
   
   /* start set up
    * setup communication stuffs first
    */
   comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   /* create if not yet built */
   if(!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }
   
   /* setup initial memory */
   nnz_A = A_diag_i[nLU];
   
   if(n > 0)
   {
      initial_alloc = hypre_min(total_rows + ceil((nnz_A / 2.0) * total_rows / n), total_rows * lfil);
   }
   capacity_L = initial_alloc;
   capacity_U = initial_alloc;
   
   D_data = hypre_CTAlloc(HYPRE_Real, total_rows, HYPRE_MEMORY_SHARED);
   L_diag_i = hypre_CTAlloc(HYPRE_Int, (total_rows+1), HYPRE_MEMORY_SHARED);
   U_diag_i = hypre_CTAlloc(HYPRE_Int, (total_rows+1), HYPRE_MEMORY_SHARED);
   
   L_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
   U_diag_j = hypre_CTAlloc(HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
   L_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
   U_diag_data = hypre_CTAlloc(HYPRE_Real, capacity_U, HYPRE_MEMORY_SHARED);
   
   ctrL = ctrU = 0;
   
   /* setting up working array */
   iw = hypre_CTAlloc(HYPRE_Int,4*total_rows,HYPRE_MEMORY_HOST);
   iL = iw + total_rows;
   w = hypre_CTAlloc(HYPRE_Real,total_rows,HYPRE_MEMORY_HOST);
   for(i = 0 ; i < total_rows ; i ++)
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
   for(i = 0 ; i < n ; i ++)
   {
      perm[i] = perm_old[i];
   }
   for(i = n ; i < total_rows ; i ++)
   {
      perm[i] = i;
   }
   for(i = 0 ; i < total_rows ; i ++)
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
   for(ii = 0 ; ii < nLU ; ii ++)
   {
      /* get real row with perm */
      i = perm[ii];
      k1 = A_diag_i[i];
      k2 = A_diag_i[i+1];
      kl = ii-1;
      /* reset row norm of ith row */
      inorm = .0;
      for(j = k1 ; j < k2 ; j ++)
      {
         inorm += fabs(A_diag_data[j]);
      }
      if(inorm == .0)
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
      for(j = k1 ; j < k2 ; j ++)
      {
         /* get now col number */
         col = rperm[A_diag_j[j]];
         if(col < ii)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = A_diag_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
         }
         else if(col == ii)
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
      while(lenhll > 0)
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
         for(j = U_diag_i[jrow] ; j < ku ; j ++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv*U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if( icol == -1 && ( (col < nLU && fabs(lxu) < itolb) || (col >= nLU && fabs(lxu) < itolef) ) )
            {
               continue;
            }
            if(icol == -1)
            {
               if(col < ii)
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
               else if(col == ii)
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
      
      if(fabs(w[ii]) < MAT_TOL)
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
      if(lenl > 0)
      {
         /* test if memory is enough */
         while(ctrL + lenl > capacity_L)
         {
            capacity_L = capacity_L * EXPAND_FACT + 1;
            L_diag_j = hypre_TReAlloc(L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
            L_diag_data = hypre_TReAlloc(L_diag_data, HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
         }
         ctrL += lenl;
         /* copy large data in */
         for(j = L_diag_i[ii] ; j < ctrL ; j ++)
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
      for(j = ii + 1 ; j <= ku ; j ++)
      {
         iw[iL[j]] = -1;
      }
      
      if(lenu < lfil)
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
      if(lenhu > 0)
      {
        /* test if memory is enough */
         while(ctrU + lenhu > capacity_U)
         {
            capacity_U = capacity_U * EXPAND_FACT + 1;
            U_diag_j = hypre_TReAlloc(U_diag_j, HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
            U_diag_data = hypre_TReAlloc(U_diag_data, HYPRE_Real, capacity_U, HYPRE_MEMORY_SHARED);
         }
         ctrU += lenhu;
         /* copy large data in */
         for(j = U_diag_i[ii] ; j < ctrU ; j ++)
         {
            jpos = ii+1+j-U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
   }/* end of ii loop from 0 to nLU-1 */
   
   
   /* second outer loop for lower part */
   for(ii = nLU ; ii < n ; ii ++)
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
      for(j = k1 ; j < k2 ; j ++)
      {
         inorm += fabs(A_diag_data[j]);
      }
      for(j = k12 ; j < k22 ; j ++)
      {
         inorm += fabs(A_offd_data[j]);
      }
      if(inorm == .0)
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
      for(j = k1 ; j < k2 ; j ++)
      {
         /* get now col number */
         col = rperm[A_diag_j[j]];
         if(col < ii)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = A_diag_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
         }
         else if(col == ii)
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
      for(j = k12 ; j < k22 ; j ++)
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
      while(lenhll > 0)
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
         for(j = U_diag_i[jrow] ; j < ku ; j ++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv*U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if( icol == -1 && ( (col < nLU && fabs(lxu) < itolb) || (col >= nLU && fabs(lxu) < itolef) ) )
            {
               continue;
            }
            if(icol == -1)
            {
               if(col < ii)
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
               else if(col == ii)
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
      
      if(fabs(w[ii]) < MAT_TOL)
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
      if(lenl > 0)
      {
         /* test if memory is enough */
         while(ctrL + lenl > capacity_L)
         {
            capacity_L = capacity_L * EXPAND_FACT + 1;
            L_diag_j = hypre_TReAlloc(L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
            L_diag_data = hypre_TReAlloc(L_diag_data, HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
         }
         ctrL += lenl;
         /* copy large data in */
         for(j = L_diag_i[ii] ; j < ctrL ; j ++)
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
      for(j = ii + 1 ; j <= ku ; j ++)
      {
         iw[iL[j]] = -1;
      }
      
      if(lenu < lfil)
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
      if(lenhu > 0)
      {
        /* test if memory is enough */
         while(ctrU + lenhu > capacity_U)
         {
            capacity_U = capacity_U * EXPAND_FACT + 1;
            U_diag_j = hypre_TReAlloc(U_diag_j, HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
            U_diag_data = hypre_TReAlloc(U_diag_data, HYPRE_Real, capacity_U, HYPRE_MEMORY_SHARED);
         }
         ctrU += lenhu;
         /* copy large data in */
         for(j = U_diag_i[ii] ; j < ctrU ; j ++)
         {
            jpos = ii+1+j-U_diag_i[ii];
            U_diag_j[j] = iL[jpos];
            U_diag_data[j] = w[jpos];
         }
      }
   }/* end of ii loop from nLU to n */
   
   
   /* main outer loop for upper part */
   for(ii = n ; ii < total_rows ; ii ++)
   {
      /* get real row with perm */
      i = ii-n;
      k1 = E_i[i];
      k2 = E_i[i+1];
      kl = ii-1;
      /* reset row norm of ith row */
      inorm = .0;
      for(j = k1 ; j < k2 ; j ++)
      {
         inorm += fabs(E_data[j]);
      }
      if(inorm == .0)
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
      for(j = k1 ; j < k2 ; j ++)
      {
         /* get now col number */
         col = rperm[E_j[j]];
         if(col < ii)
         {
            /* L part of it */
            iL[lenhll] = col;
            w[lenhll] = E_data[j];
            iw[col] = lenhll++;
            /* add to heap, by col number */
            hypre_ILUMinHeapAddIRIi(iL,w,iw,lenhll);
         }
         else if(col == ii)
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
      while(lenhll > 0)
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
         for(j = U_diag_i[jrow] ; j < ku ; j ++)
         {
            col = U_diag_j[j];
            icol = iw[col];
            lxu = - dpiv*U_diag_data[j];
            /* we don't want to fill small number to empty place */
            if( icol == -1 && ( (col < nLU && fabs(lxu) < itolb) || (col >= nLU && fabs(lxu) < itolef) ) )
            {
               continue;
            }
            if(icol == -1)
            {
               if(col < ii)
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
               else if(col == ii)
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
      
      if(fabs(w[ii]) < MAT_TOL)
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
      if(lenl > 0)
      {
         /* test if memory is enough */
         while(ctrL + lenl > capacity_L)
         {
            capacity_L = capacity_L * EXPAND_FACT + 1;
            L_diag_j = hypre_TReAlloc(L_diag_j, HYPRE_Int, capacity_L, HYPRE_MEMORY_SHARED);
            L_diag_data = hypre_TReAlloc(L_diag_data, HYPRE_Real, capacity_L, HYPRE_MEMORY_SHARED);
         }
         ctrL += lenl;
         /* copy large data in */
         for(j = L_diag_i[ii] ; j < ctrL ; j ++)
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
      for(j = ii + 1 ; j <= ku ; j ++)
      {
         iw[iL[j]] = -1;
      }
      
      if(lenu < lfil)
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
      if(lenhu > 0)
      {
        /* test if memory is enough */
         while(ctrU + lenhu > capacity_U)
         {
            capacity_U = capacity_U * EXPAND_FACT + 1;
            U_diag_j = hypre_TReAlloc(U_diag_j, HYPRE_Int, capacity_U, HYPRE_MEMORY_SHARED);
            U_diag_data = hypre_TReAlloc(U_diag_data, HYPRE_Real, capacity_U, HYPRE_MEMORY_SHARED);
         }
         ctrU += lenhu;
         /* copy large data in */
         for(j = U_diag_i[ii] ; j < ctrU ; j ++)
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
   
   hypre_MPI_Allreduce( &total_rows, &global_num_rows, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   /* need to get new column start */
   col_starts = hypre_CTAlloc(HYPRE_Int,2,HYPRE_MEMORY_HOST);
   hypre_MPI_Scan( &total_rows, &global_start, 1, HYPRE_MPI_INT, hypre_MPI_SUM, comm);
   col_starts[1] = global_start;
   col_starts[0] = global_start - total_rows;
   
   /* create parcsr matrix */
   matL = hypre_ParCSRMatrixCreate( comm,
                       global_num_rows,
                       global_num_rows,
                       col_starts,
                       col_starts,
                       0,
                       L_diag_i[total_rows],
                       0 );

   /* Have A own coarse_partitioning instead of L */
   hypre_ParCSRMatrixSetColStartsOwner(matL,1);
   hypre_ParCSRMatrixSetRowStartsOwner(matL,0);
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
      hypre_TFree(L_diag_j,HYPRE_MEMORY_SHARED);
      hypre_TFree(L_diag_data,HYPRE_MEMORY_SHARED);
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

   /* Have A own coarse_partitioning instead of U */
   hypre_ParCSRMatrixSetColStartsOwner(matU,0);
   hypre_ParCSRMatrixSetRowStartsOwner(matU,0);

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
      hypre_TFree(U_diag_j,HYPRE_MEMORY_SHARED);
      hypre_TFree(U_diag_data,HYPRE_MEMORY_SHARED);
   }
   /* store (global) total number of nonzeros */
   local_nnz = (HYPRE_Real) (U_diag_i[total_rows]);
   hypre_MPI_Allreduce(&local_nnz, &total_nnz, 1, HYPRE_MPI_REAL, hypre_MPI_SUM, comm);
   hypre_ParCSRMatrixDNumNonzeros(matU) = total_nnz;
   
   /* free working array */
   hypre_TFree(iw,HYPRE_MEMORY_HOST);
   hypre_TFree(w,HYPRE_MEMORY_HOST);
   
   /* free external data */
   if(E_i)
   {
      hypre_TFree(E_i, HYPRE_MEMORY_HOST);
   }
   if(E_j)
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

