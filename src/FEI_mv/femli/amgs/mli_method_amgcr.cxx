/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.14 $
 ***********************************************************************EHEADER*/





#ifdef WIN32
#define strcmp _stricmp
#endif

#include <string.h>
#include <assert.h>
#include "HYPRE.h"
#include "_hypre_utilities.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "util/mli_utils.h"
#include "matrix/mli_matrix.h"
#include "matrix/mli_matrix_misc.h"
#include "vector/mli_vector.h"
#include "solver/mli_solver.h"
#include "amgs/mli_method_amgcr.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "distributed_ls/ParaSails/Matrix.h"
#include "distributed_ls/ParaSails/ParaSails.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"
#ifdef __cplusplus
}
#endif

#define habs(x) ((x > 0) ? x : (-x))

/* ********************************************************************* *
 * constructor
 * --------------------------------------------------------------------- */

MLI_Method_AMGCR::MLI_Method_AMGCR( MPI_Comm comm ) : MLI_Method( comm )
{
   char name[100];

   strcpy(name, "AMGCR");
   setName( name );
   setID( MLI_METHOD_AMGCR_ID );
   maxLevels_     = 40;
   numLevels_     = 2;
   currLevel_     = 0;
   outputLevel_   = 0;
   findMIS_       = 0;
   targetMu_      = 0.25;
   numTrials_     = 1;
   numVectors_    = 1;
   minCoarseSize_ = 100;
   cutThreshold_  = 0.01;
   strcpy(smoother_, "Jacobi");
   smootherNum_  = 1;
   smootherWgts_ = new double[2];
   smootherWgts_[0] = smootherWgts_[1] = 1.0;
   strcpy(coarseSolver_, "SuperLU");
   coarseSolverNum_ = 1;
   coarseSolverWgts_ = new double[20];
   for (int j = 0; j < 20; j++) coarseSolverWgts_ [j] = 1.0;
   RAPTime_            = 0.0;
   totalTime_          = 0.0;
   strcpy(paramFile_, "empty");
   PDegree_ = 2;
}

/* ********************************************************************* *
 * destructor
 * --------------------------------------------------------------------- */

MLI_Method_AMGCR::~MLI_Method_AMGCR()
{
   if (smootherWgts_     != NULL) delete [] smootherWgts_;
   if (coarseSolverWgts_ != NULL) delete [] coarseSolverWgts_;
}

/* ********************************************************************* *
 * set parameters
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::setParams(char *inName, int argc, char *argv[])
{
   int        i, mypid, level, nSweeps=1;
   double     *weights=NULL;
   char       param1[256], param2[256], *param3;
   MPI_Comm   comm;

   comm = getComm();
   MPI_Comm_rank( comm, &mypid );
   sscanf(inName, "%s", param1);
   if ( outputLevel_ >= 1 && mypid == 0 ) 
      printf("\tMLI_Method_AMGCR::setParam = %s\n", inName);
   if ( !strcmp(param1, "setOutputLevel" ))
   {
      sscanf(inName,"%s %d", param1, &level);
      return (setOutputLevel(level));
   }
   else if ( !strcmp(param1, "setNumLevels" ))
   {
      sscanf(inName,"%s %d", param1, &level);
      return (setNumLevels(level));
   }
   else if ( !strcmp(param1, "useMIS" ))
   {
      findMIS_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "setTargetMu" ))
   {
      sscanf(inName,"%s %lg", param1, &targetMu_);
      if (targetMu_ < 0.0) targetMu_ = 0.5;
      if (targetMu_ > 1.0) targetMu_ = 0.5;
      return 0;
   }
   else if ( !strcmp(param1, "setNumTrials" ))
   {
      sscanf(inName,"%s %d", param1, &numTrials_);
      if (numTrials_ < 1) numTrials_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "setNumVectors" ))
   {
      sscanf(inName,"%s %d", param1, &numVectors_);
      if (numVectors_ < 1) numVectors_ = 1;
      return 0;
   }
   else if ( !strcmp(param1, "setPDegree" ))
   {
      sscanf(inName,"%s %d", param1, &PDegree_);
      if (PDegree_ < 0) PDegree_ = 0;
      if (PDegree_ > 3) PDegree_ = 3;
      return 0;
   }
   else if ( !strcmp(param1, "setSmoother" ))
   {
      sscanf(inName,"%s %s", param1, param2);
      if ( argc != 2 )
      {
         printf("MLI_Method_AMGCR::setParams ERROR - setSmoother needs");
         printf(" 2 arguments.\n");
         printf("     argument[0] : number of relaxation sweeps \n");
         printf("     argument[1] : relaxation weights\n");
         return 1;
      } 
      nSweeps = *(int *)   argv[0];
      weights = (double *) argv[1];
      smootherNum_ = nSweeps;
      if (smootherWgts_ != NULL) delete [] smootherWgts_;
      smootherWgts_ = new double[nSweeps];
      for (i = 0; i < nSweeps; i++) smootherWgts_[i] = weights[i];
      strcpy(smoother_, param2);
      return 0;
   }
   else if (!strcmp(param1, "setCoarseSolver"))
   {
      sscanf(inName,"%s %s", param1, param2);
      if ( strcmp(param2, "SuperLU") && argc != 2 )
      {
         printf("MLI_Method_AMGCR::setParams ERROR - setCoarseSolver needs");
         printf(" 2 arguments.\n");
         printf("     argument[0] : number of relaxation sweeps \n");
         printf("     argument[1] : relaxation weights\n");
         return 1;
      } 
      else if ( strcmp(param2, "SuperLU") )
      {
         strcpy(coarseSolver_, param2);
         coarseSolverNum_ = *(int *) argv[0];
         if (coarseSolverWgts_ != NULL) delete [] coarseSolverWgts_;
         coarseSolverWgts_ = new double[coarseSolverNum_];
         weights = (double *) argv[1];
         for (i = 0; i < coarseSolverNum_; i++) smootherWgts_[i] = weights[i];
      }
      else if ( !strcmp(param2, "SuperLU") )
      {
         if (coarseSolverWgts_ != NULL) delete [] coarseSolverWgts_;
         coarseSolverWgts_ = NULL;
         weights = NULL;
         coarseSolverNum_ = 1;
      }
      return 0;
   }
   else if ( !strcmp(param1, "setParamFile" ))
   {
      param3 = (char *) argv[0];
      strcpy( paramFile_, param3 ); 
      return 0;
   }
   else if ( !strcmp(param1, "print" ))
   {
      print();
      return 0;
   }
   return 1;
}

/***********************************************************************
 * generate multilevel structure
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::setup( MLI *mli ) 
{
   int         level, mypid, *ISMarker, localNRows;
   int         irow, nrows, gNRows, numFpts, *fList;;
   int         *ADiagI, *ADiagJ, jcol;
   double      startTime, elapsedTime;
   char        paramString[100], *targv[10];
   MLI_Matrix  *mli_Pmat, *mli_Rmat, *mli_Amat, *mli_cAmat, *mli_Affmat;
   MLI_Matrix  *mli_Afcmat;
   MLI_Solver  *smootherPtr, *csolvePtr;
   MPI_Comm    comm;
   hypre_ParCSRMatrix *hypreA, *hypreP, *hypreR, *hypreAP, *hypreAC;
   hypre_CSRMatrix *ADiag;
   MLI_Function    *funcPtr;

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGCR::setup begins...\n");
#endif

   /* --------------------------------------------------------------- */
   /* traverse all levels                                             */
   /* --------------------------------------------------------------- */

   RAPTime_ = 0.0;
   level    = 0;
   comm     = getComm();
   MPI_Comm_rank( comm, &mypid );
   totalTime_ = MLI_Utils_WTime();

   for (level = 0; level < numLevels_; level++ )
   {
      currLevel_ = level;
      if (level == numLevels_-1) break;

      /* -------------------------------------------------- */
      /* fetch fine grid matrix information                 */
      /* -------------------------------------------------- */

      mli_Amat = mli->getSystemMatrix(level);
      assert (mli_Amat != NULL);
      hypreA = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
      gNRows = hypre_ParCSRMatrixGlobalNumRows(hypreA);
      ADiag = hypre_ParCSRMatrixDiag(hypreA);
      localNRows = hypre_CSRMatrixNumRows(ADiag);
      if (localNRows < minCoarseSize_) break;

      if (mypid == 0 && outputLevel_ > 0)
      {
         printf("\t*****************************************************\n");
         printf("\t*** AMGCR : level = %d, nrows = %d\n", level, gNRows);
         printf("\t-----------------------------------------------------\n");
      }

      /* -------------------------------------------------- */
      /* perform coarsening and P                           */
      /* -------------------------------------------------- */

      if (findMIS_ > 0)
      {
#if 0
         hypre_BoomerAMGCoarsen(hypreA, hypreA, 0, 0, &ISMarker);
#else
         ISMarker = new int[localNRows];
         for (irow = 0; irow < localNRows; irow++) ISMarker[irow] = 0;
         ADiag  = hypre_ParCSRMatrixDiag(hypreA);
         ADiagI = hypre_CSRMatrixI(ADiag);
         ADiagJ = hypre_CSRMatrixJ(ADiag);
         for (irow = 0; irow < localNRows; irow++) 
         {
            if (ISMarker[irow] == 0) 
            {
               ISMarker[irow] = 1;
               for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++) 
                  if (ISMarker[ADiagJ[jcol]] == 0)
                     ISMarker[ADiagJ[jcol]] = -1;
            }
         }
         for (irow = 0; irow < localNRows; irow++) 
            if (ISMarker[irow] < 0) ISMarker[irow] = 0;
#endif
      }
      else 
      {
         ISMarker = new int[localNRows];
         for (irow = 0; irow < localNRows; irow++) ISMarker[irow] = 0;
      }
      for (irow = 0; irow < localNRows; irow++) 
         if (ISMarker[irow] < 0) ISMarker[irow] = 0;
      mli_Affmat = performCR(mli_Amat, ISMarker, &mli_Afcmat);

      nrows = 0;
      for (irow = 0; irow < localNRows; irow++) 
         if (ISMarker[irow] == 1) nrows++;
      if (nrows < minCoarseSize_) break;
      mli_Pmat = createPmat(ISMarker, mli_Amat, mli_Affmat, mli_Afcmat);
      delete mli_Afcmat;
      if (mli_Pmat == NULL) break;
      mli->setProlongation(level+1, mli_Pmat);
      mli_Rmat = createRmat(ISMarker, mli_Amat, mli_Affmat);
      mli->setRestriction(level, mli_Rmat);

      /* -------------------------------------------------- */
      /* construct and set the coarse grid matrix           */
      /* -------------------------------------------------- */

      startTime = MLI_Utils_WTime();
      if (mypid == 0 && outputLevel_ > 0) printf("\tComputing RAP\n");
      hypreP = (hypre_ParCSRMatrix *) mli_Pmat->getMatrix();
      hypreR = (hypre_ParCSRMatrix *) mli_Rmat->getMatrix();
      hypreAP = hypre_ParMatmul(hypreA, hypreP);
      hypreAC = hypre_ParMatmul(hypreR, hypreAP);
      sprintf(paramString, "HYPRE_ParCSR");
      funcPtr = new MLI_Function();
      MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
      mli_cAmat = new MLI_Matrix((void*) hypreAC, paramString, funcPtr);
      delete funcPtr;
      hypre_ParCSRMatrixDestroy(hypreAP);

      mli->setSystemMatrix(level+1, mli_cAmat);
      elapsedTime = (MLI_Utils_WTime() - startTime);
      RAPTime_ += elapsedTime;
      if (mypid == 0 && outputLevel_ > 0) 
         printf("\tRAP computed, time = %e seconds.\n", elapsedTime);

      /* -------------------------------------------------- */
      /* set the smoothers                                  */
      /* (if domain decomposition and ARPACKA SuperLU       */
      /* smoothers is requested, perform special treatment, */
      /* and if domain decomposition and SuperLU smoother   */
      /* is requested with multiple local subdomains, again */
      /* perform special treatment.)                        */
      /* -------------------------------------------------- */

      smootherPtr = MLI_Solver_CreateFromName(smoother_);
      targv[0] = (char *) &smootherNum_;
      targv[1] = (char *) smootherWgts_;
      sprintf(paramString, "relaxWeight");
      smootherPtr->setParams(paramString, 2, targv);
      numFpts = 0;
      for (irow = 0; irow < localNRows; irow++) 
         if (ISMarker[irow] == 0) numFpts++;
#if 1
      if (numFpts > 0) 
      {
         fList = new int[numFpts];
         numFpts = 0;
         for (irow = 0; irow < localNRows; irow++) 
            if (ISMarker[irow] == 0) fList[numFpts++] = irow;
         targv[0] = (char *) &numFpts;
         targv[1] = (char *) fList;
         sprintf(paramString, "setFptList");
         smootherPtr->setParams(paramString, 2, targv);
      } 
      sprintf(paramString, "setModifiedDiag");
      smootherPtr->setParams(paramString, 0, NULL);
      smootherPtr->setup(mli_Affmat);
      mli->setSmoother(level, MLI_SMOOTHER_PRE, smootherPtr);
      sprintf(paramString, "ownAmat");
      smootherPtr->setParams(paramString, 0, NULL);
#else
      printf("whole grid smoothing\n");
      smootherPtr->setup(mli_Amat);
      mli->setSmoother(level, MLI_SMOOTHER_PRE, smootherPtr);
      mli->setSmoother(level, MLI_SMOOTHER_POST, smootherPtr);
#endif
   }

   /* --------------------------------------------------------------- */
   /* set the coarse grid solver                                      */
   /* --------------------------------------------------------------- */

   if (mypid == 0 && outputLevel_ > 0) printf("\tCoarse level = %d\n",level);
   csolvePtr = MLI_Solver_CreateFromName( coarseSolver_ );
   if (strcmp(coarseSolver_, "SuperLU"))
   {
      targv[0] = (char *) &coarseSolverNum_;
      targv[1] = (char *) coarseSolverWgts_ ;
      sprintf(paramString, "relaxWeight");
      csolvePtr->setParams(paramString, 2, targv);
   }
   mli_Amat = mli->getSystemMatrix(level);
   csolvePtr->setup(mli_Amat);
   mli->setCoarseSolve(csolvePtr);
   totalTime_ = MLI_Utils_WTime() - totalTime_;

   if (outputLevel_ >= 2) printStatistics(mli);

#ifdef MLI_DEBUG_DETAILED
   printf("MLI_Method_AMGCR::setup ends.");
#endif
   return (level+1);
}

/* ********************************************************************* *
 * set diagnostics output level
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::setOutputLevel( int level )
{
   outputLevel_ = level;
   return 0;
}

/* ********************************************************************* *
 * set number of levels 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::setNumLevels( int nlevels )
{
   if ( nlevels < maxLevels_ && nlevels > 0 ) numLevels_ = nlevels;
   return 0;
}

/* ********************************************************************* *
 * select independent set 
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::selectIndepSet(MLI_Matrix *mli_Amat, int **indepSet)
{
   int    irow, localNRows, numColsOffd, graphArraySize;
   int    *graphArray, *graphArrayOffd, *ISMarker, *ISMarkerOffd;
   int    nprocs, *ADiagI, *ADiagJ;
   double *measureArray;
   hypre_ParCSRMatrix *hypreA, *hypreS;
   hypre_CSRMatrix    *ADiag, *AOffd, *SExt=NULL;
   MPI_Comm comm;

   hypreA = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   ADiag = hypre_ParCSRMatrixDiag(hypreA);
   ADiagI = hypre_CSRMatrixI(ADiag);
   ADiagJ = hypre_CSRMatrixJ(ADiag);
   AOffd = hypre_ParCSRMatrixOffd(hypreA);
   localNRows = hypre_CSRMatrixNumRows(ADiag);
   numColsOffd = hypre_CSRMatrixNumCols(AOffd);
   comm = getComm();
   MPI_Comm_size(comm, &nprocs);

   measureArray = new double[localNRows+numColsOffd];
   for (irow = 0; irow < localNRows+numColsOffd; irow++) 
      measureArray[irow] = 0;
   for (irow = 0; irow < ADiagI[localNRows]; irow++) 
      measureArray[ADiagJ[irow]] += 1;

   hypre_BoomerAMGCreateS(hypreA, 0.0e0, 0.0e0, 1, NULL, &hypreS);
   hypre_BoomerAMGIndepSetInit(hypreS, measureArray, 0);

   graphArraySize = localNRows;
   graphArray = new int[localNRows];
   for (irow = 0; irow < localNRows; irow++) graphArray[irow] = irow; 

   if (numColsOffd) graphArrayOffd = new int[numColsOffd];
   else             graphArrayOffd = NULL;
   for (irow = 0; irow < numColsOffd; irow++) graphArrayOffd[irow] = irow;

   ISMarker = new int[localNRows];
   for (irow = 0; irow < localNRows; irow++) ISMarker[irow] = 0;
   if (numColsOffd) 
   {
      ISMarkerOffd = new int[numColsOffd];
      for (irow = 0; irow < numColsOffd; irow++) ISMarkerOffd[irow] = 0;
   }
   if (nprocs > 1) SExt = hypre_ParCSRMatrixExtractBExt(hypreA,hypreA,0);

   hypre_BoomerAMGIndepSet(hypreS, measureArray, graphArray,
                           graphArraySize, graphArrayOffd, numColsOffd,
                           ISMarker, ISMarkerOffd);
   
   delete [] measureArray;
   delete [] graphArray;
   if (numColsOffd > 0) delete [] graphArrayOffd;
   if (nprocs > 1) hypre_CSRMatrixDestroy(SExt);
   hypre_ParCSRMatrixDestroy(hypreS);
   if (numColsOffd > 0) delete [] ISMarkerOffd;
   (*indepSet) = ISMarker;
   return 0;
}

/* ********************************************************************* *
 * perform compatible relaxation
 * --------------------------------------------------------------------- */

MLI_Matrix *MLI_Method_AMGCR::performCR(MLI_Matrix *mli_Amat, int *indepSet,
                                        MLI_Matrix **AfcMat)
{
   int    nprocs, mypid, localNRows, iT, numFpts, irow, *reduceArray1;
   int    *reduceArray2, iP, FStartRow, FNRows, ierr, *rowLengs;
   int    startRow, rowIndex, colIndex, rowCount;
   int    one=1, *ADiagI, *ADiagJ, *sortIndices, *fList;
   int    idata, fPt, iV, ranSeed, jcol, rowCount2, CStartRow, CNRows;
   int    *rowStarts, *newRowStarts, *colStarts, *newColStarts;
   int    newCount, it;
#if 0
   double relaxWts[5];
#endif
   double colValue, rnorm0, rnorm1, dOne=1.0;
   double *XaccData, ddata, threshold, *XData, arnorm0, arnorm1;
   double aratio, ratio1, ratio2, *ADiagA;
   char   paramString[200];
   HYPRE_IJMatrix     IJPFF, IJPFC;
   hypre_ParCSRMatrix *hypreA, *hypreAff, *hyprePFC, *hypreAffT;
   hypre_ParCSRMatrix *hypreAfc, *hyprePFF, *hyprePFFT, *hypreAPFC;
   hypre_CSRMatrix    *ADiag;
   HYPRE_IJVector     IJB, IJX, IJXacc;
   hypre_ParVector    *hypreB, *hypreX, *hypreXacc;
   MLI_Matrix *mli_PFFMat, *mli_AffMat, *mli_AfcMat, *mli_AffTMat;
   MLI_Vector *mli_Xvec, *mli_Bvec;
#if 0
   MLI_Solver *smootherPtr;
#endif
   MPI_Comm   comm;
   HYPRE_Solver hypreSolver;

   /* ------------------------------------------------------ */
   /* get matrix and machine information                     */
   /* ------------------------------------------------------ */

   comm = getComm();
   MPI_Comm_size(comm, &nprocs);
   MPI_Comm_rank(comm, &mypid);
   hypreA = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   ADiag = hypre_ParCSRMatrixDiag(hypreA);
   ADiagI = hypre_CSRMatrixI(ADiag);
   ADiagJ = hypre_CSRMatrixJ(ADiag);
   ADiagA = hypre_CSRMatrixData(ADiag);
   localNRows = hypre_CSRMatrixNumRows(ADiag);
   startRow = hypre_ParCSRMatrixFirstRowIndex(hypreA);
   fList = new int[localNRows];

   /* ------------------------------------------------------ */
   /* loop over number of trials                             */
   /* ------------------------------------------------------ */

   printf("\tPerform compatible relaxation\n");
   arnorm1 = arnorm0 = 1;
   for (iT = 0; iT < numTrials_; iT++)
   {
      /* --------------------------------------------------- */
      /* get Aff and Afc matrices (get dimension)            */
      /* --------------------------------------------------- */

      numFpts = 0;
      for (irow = 0; irow < localNRows; irow++)
         if (indepSet[irow] != 1) fList[numFpts++] = irow;
      printf("\tTrial %3d (%3d) : number of F-points = %d\n", iT,
             numTrials_, numFpts);
      reduceArray1 = new int[nprocs+1];
      reduceArray2 = new int[nprocs+1];
      for (iP = 0; iP < nprocs; iP++) reduceArray1[iP] = 0;
      reduceArray1[mypid] = numFpts;
      MPI_Allreduce(reduceArray1,reduceArray2,nprocs,MPI_INT,MPI_SUM,comm);
      for (iP = nprocs-1; iP >= 0; iP--) 
         reduceArray2[iP+1] = reduceArray2[iP];
      reduceArray2[0] = 0;
      for (iP = 2; iP <= nprocs; iP++) reduceArray2[iP] += reduceArray2[iP-1];
      FStartRow = reduceArray2[mypid];
      FNRows = reduceArray2[mypid+1] - FStartRow;
      delete [] reduceArray1;
      delete [] reduceArray2;
      CStartRow = startRow - FStartRow;
      CNRows = localNRows - FNRows;

      /* --------------------------------------------------- */
      /* get Aff and Afc matrices (create permute matrices)  */
      /* --------------------------------------------------- */

      ierr = HYPRE_IJMatrixCreate(comm,startRow,startRow+localNRows-1,
                           FStartRow,FStartRow+FNRows-1,&IJPFF);
      ierr = HYPRE_IJMatrixSetObjectType(IJPFF, HYPRE_PARCSR);
      assert(!ierr);
      rowLengs = new int[localNRows];
      for (irow = 0; irow < localNRows; irow++) rowLengs[irow] = 1;
      ierr = HYPRE_IJMatrixSetRowSizes(IJPFF, rowLengs);
      ierr = HYPRE_IJMatrixInitialize(IJPFF);
      assert(!ierr);

      ierr = HYPRE_IJMatrixCreate(comm,startRow,startRow+localNRows-1,
                   CStartRow,CStartRow+CNRows-1, &IJPFC);
      ierr = HYPRE_IJMatrixSetObjectType(IJPFC, HYPRE_PARCSR);
      assert(!ierr);
      ierr = HYPRE_IJMatrixSetRowSizes(IJPFC, rowLengs);
      ierr = HYPRE_IJMatrixInitialize(IJPFC);
      assert(!ierr);
      delete [] rowLengs;

      /* --------------------------------------------------- */
      /* get Aff and Afc matrices (load permute matrices)    */
      /* --------------------------------------------------- */

      colValue = 1.0;
      rowCount = rowCount2 = 0;
      for (irow = 0; irow < localNRows; irow++)
      {
         rowIndex = startRow + irow;
         if (indepSet[irow] == 0) 
         {
            colIndex = FStartRow + rowCount;
            HYPRE_IJMatrixSetValues(IJPFF,1,&one,(const int *) &rowIndex,
                    (const int *) &colIndex, (const double *) &colValue);
            rowCount++;
         }
         else
         {
            colIndex = CStartRow + rowCount2;
            HYPRE_IJMatrixSetValues(IJPFC,1,&one,(const int *) &rowIndex,
                    (const int *) &colIndex, (const double *) &colValue);
            rowCount2++;
         }
      }
      ierr = HYPRE_IJMatrixAssemble(IJPFF);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJPFF, (void **) &hyprePFF);
      //hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hyprePFF);
      sprintf(paramString, "HYPRE_ParCSR" );
      mli_PFFMat = new MLI_Matrix((void *)hyprePFF,paramString,NULL);

      ierr = HYPRE_IJMatrixAssemble(IJPFC);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJPFC, (void **) &hyprePFC);
      //hypre_MatvecCommPkgCreate((hypre_ParCSRMatrix *) hyprePFC);
      hypreAPFC = hypre_ParMatmul(hypreA, hyprePFC);
      hypre_ParCSRMatrixTranspose(hyprePFF, &hyprePFFT, 1);
      hypreAfc = hypre_ParMatmul(hyprePFFT, hypreAPFC);
      rowStarts = hypre_ParCSRMatrixRowStarts(hyprePFFT);
      newRowStarts = (int *) malloc((nprocs+1) * sizeof(int));
      for (irow = 0; irow <= nprocs; irow++) 
         newRowStarts[irow] = rowStarts[irow];
      hypre_ParCSRMatrixRowStarts(hypreAfc) = newRowStarts;
      colStarts = hypre_ParCSRMatrixColStarts(hypreAPFC);
      newColStarts = (int *) malloc((nprocs+1) * sizeof(int));
      for (irow = 0; irow <= nprocs; irow++) 
         newColStarts[irow] = colStarts[irow];
      hypre_ParCSRMatrixColStarts(hypreAfc) = newColStarts;
      hypre_ParCSRMatrixOwnsRowStarts(hypreAfc) = 1;
      hypre_ParCSRMatrixOwnsColStarts(hypreAfc) = 1;
      hypre_ParCSRMatrixOwnsColStarts(hyprePFF) = 0;

      sprintf(paramString, "HYPRE_ParCSR" );
      mli_AfcMat = new MLI_Matrix((void *)hypreAfc,paramString,NULL);

      MLI_Matrix_ComputePtAP(mli_PFFMat, mli_Amat, &mli_AffMat);
      hypreAff  = (hypre_ParCSRMatrix *) mli_AffMat->getMatrix();
      colStarts = hypre_ParCSRMatrixColStarts(hyprePFF);
      newColStarts = (int *) malloc((nprocs+1) * sizeof(int));
      for (irow = 0; irow <= nprocs; irow++) 
         newColStarts[irow] = colStarts[irow];
      hypre_ParCSRMatrixColStarts(hypreAff) = newColStarts;
      newColStarts = (int *) malloc((nprocs+1) * sizeof(int));
      for (irow = 0; irow <= nprocs; irow++) 
         newColStarts[irow] = colStarts[irow];
      hypre_ParCSRMatrixRowStarts(hypreAff) = newColStarts;

      if (arnorm1/arnorm0 < targetMu_) break;

#define HAVE_TRANS
#ifdef HAVE_TRANS
      MLI_Matrix_Transpose(mli_AffMat, &mli_AffTMat);
      hypreAffT = (hypre_ParCSRMatrix *) mli_AffTMat->getMatrix();
#endif
      HYPRE_IJVectorCreate(comm,FStartRow, FStartRow+FNRows-1,&IJX);
      HYPRE_IJVectorSetObjectType(IJX, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(IJX);
      HYPRE_IJVectorAssemble(IJX);
      HYPRE_IJVectorGetObject(IJX, (void **) &hypreX);
      sprintf(paramString, "HYPRE_ParVector" );
      mli_Xvec = new MLI_Vector((void *)hypreX,paramString,NULL);

      HYPRE_IJVectorCreate(comm,FStartRow, FStartRow+FNRows-1,&IJXacc);
      HYPRE_IJVectorSetObjectType(IJXacc, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(IJXacc);
      HYPRE_IJVectorAssemble(IJXacc);
      HYPRE_IJVectorGetObject(IJXacc, (void **) &hypreXacc);
      hypre_ParVectorSetConstantValues(hypreXacc, 0.0);

      HYPRE_IJVectorCreate(comm,FStartRow, FStartRow+FNRows-1,&IJB);
      HYPRE_IJVectorSetObjectType(IJB, HYPRE_PARCSR);
      HYPRE_IJVectorInitialize(IJB);
      HYPRE_IJVectorAssemble(IJB);
      HYPRE_IJVectorGetObject(IJB, (void **) &hypreB);
      hypre_ParVectorSetConstantValues(hypreB, 0.0);
      sprintf(paramString, "HYPRE_ParVector" );
      mli_Bvec = new MLI_Vector((void *)hypreB,paramString,NULL);

#if 0
      /* --------------------------------------------------- */
      /* set up Jacobi smoother with 4 sweeps and weight=1   */
      /* --------------------------------------------------- */

      strcpy(paramString, "Jacobi");
      smootherPtr = MLI_Solver_CreateFromName(paramString);
      targc = 2;
      numSweeps = 1;
      targv[0] = (char *) &numSweeps;
      for (i = 0; i < 5; i++) relaxWts[i] = 1.0;
      targv[1] = (char *) relaxWts;
      strcpy(paramString, "relaxWeight");
      smootherPtr->setParams(paramString, targc, targv);
      maxEigen = 1.0;
      targc = 1;
      targv[0] = (char *) &maxEigen;
      strcpy(paramString, "setMaxEigen");
      smootherPtr->setParams(paramString, targc, targv);
      smootherPtr->setup(mli_AffMat);

      /* --------------------------------------------------- */
      /* relaxation                                          */
      /* --------------------------------------------------- */

      targc = 2;
      targv[0] = (char *) &numSweeps;
      targv[1] = (char *) relaxWts;
      strcpy(paramString, "relaxWeight");
      arnorm0 = 1.0;
      arnorm1 = 0.0;
      aratio = 0.0;
      XData = (double *) hypre_VectorData(hypre_ParVectorLocalVector(hypreX));
      for (iV = 0; iV < numVectors_; iV++)
      {
         ranSeed = 9001 * 7901 * iV *iV * iV * iV + iV * iV * iV + 101;
         HYPRE_ParVectorSetRandomValues((HYPRE_ParVector) hypreX,ranSeed);
         for (irow = 0; irow < FNRows; irow++)
            XData[irow] = 0.5 * XData[irow] + 0.5;
         hypre_ParVectorSetConstantValues(hypreB, 0.0);
         hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm0 = sqrt(hypre_ParVectorInnerProd(hypreB, hypreB));
         hypre_ParVectorSetConstantValues(hypreB, 0.0);

         numSweeps = 5;
         smootherPtr->setParams(paramString, targc, targv);
         smootherPtr->solve(mli_Bvec, mli_Xvec);
         hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(hypre_ParVectorInnerProd(hypreB, hypreB));

         rnorm0 = rnorm1;
         hypre_ParVectorSetConstantValues(hypreB, 0.0);
         numSweeps = 1;
         smootherPtr->setParams(paramString, targc, targv);
         smootherPtr->solve(mli_Bvec, mli_Xvec);
         hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(hypre_ParVectorInnerProd(hypreB, hypreB));

         hypre_ParVectorAxpy(dOne, hypreX, hypreXacc);
         printf("\tTrial %3d : Jacobi norms = %16.8e %16.8e\n",iT,
                rnorm0,rnorm1);
         if (iV == 0) arnorm0 = rnorm0;
         else         arnorm0 += rnorm0;
         arnorm1 += rnorm1;
         if (rnorm0 < 1.0e-10) rnorm0 = 1.0;
         ratio1 = ratio2 = rnorm1 / rnorm0;
         aratio += ratio1;
         if (ratio1 < targetMu_) break;
      }
      delete smootherPtr;
#else
      MLI_Utils_mJacobiCreate(comm, &hypreSolver);
      MLI_Utils_mJacobiSetParams(hypreSolver, PDegree_); 
      XData = (double *) hypre_VectorData(hypre_ParVectorLocalVector(hypreX));
      aratio = 0.0;
      for (iV = 0; iV < numVectors_; iV++)
      {
         ranSeed = 9001 * 7901 * iV *iV * iV * iV + iV * iV * iV + 101;

         /* ------------------------------------------------------- */
         /* CR with A                                               */
         /* ------------------------------------------------------- */

         HYPRE_ParVectorSetRandomValues((HYPRE_ParVector) hypreX,ranSeed);
         for (irow = 0; irow < FNRows; irow++)
            XData[irow] = 0.5 * XData[irow] + 0.5;
         hypre_ParVectorSetConstantValues(hypreB, 0.0);
         hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm0 = sqrt(hypre_ParVectorInnerProd(hypreB, hypreB));

         hypre_ParVectorSetConstantValues(hypreB, 0.0);
         strcpy(paramString, "pJacobi");
         MLI_Utils_HypreGMRESSolve(hypreSolver, (HYPRE_Matrix) hypreAff,
                     (HYPRE_Vector) hypreB, (HYPRE_Vector) hypreX, paramString);
         hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(hypre_ParVectorInnerProd(hypreB, hypreB));
         if (rnorm1 < rnorm0 * 1.0e-10 || rnorm1 < 1.0e-10) 
         {
            printf("\tperformCR : rnorm0, rnorm1 = %e %e\n",rnorm0,rnorm1);
            break;
         }
         rnorm0 = rnorm1;

         hypre_ParVectorSetConstantValues(hypreB, 0.0);
         strcpy(paramString, "mJacobi");
         MLI_Utils_HypreGMRESSolve(hypreSolver, (HYPRE_Matrix) hypreAff,
                     (HYPRE_Vector) hypreB, (HYPRE_Vector) hypreX, paramString);
         hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(hypre_ParVectorInnerProd(hypreB, hypreB));
         rnorm1 = 0.2 * log10(rnorm1/rnorm0);
         rnorm1 = pow(1.0e1, rnorm1);   
         ratio1 = rnorm1;

         /* ------------------------------------------------------- */
         /* CR with A^T                                             */
         /* ------------------------------------------------------- */

#ifdef HAVE_TRANS
         HYPRE_ParVectorSetRandomValues((HYPRE_ParVector) hypreX,ranSeed);
         for (irow = 0; irow < FNRows; irow++)
            XData[irow] = 0.5 * XData[irow] + 0.5;
         hypre_ParVectorSetConstantValues(hypreB, 0.0);
         hypre_ParCSRMatrixMatvec(-1.0, hypreAff, hypreX, 1.0, hypreB);
         rnorm0 = sqrt(hypre_ParVectorInnerProd(hypreB, hypreB));

         hypre_ParVectorSetConstantValues(hypreB, 0.0);
         strcpy(paramString, "pJacobi");
         MLI_Utils_HypreGMRESSolve(hypreSolver, (HYPRE_Matrix) hypreAffT,
                     (HYPRE_Vector) hypreB, (HYPRE_Vector) hypreX, paramString);
         hypre_ParCSRMatrixMatvec(-1.0, hypreAffT, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(hypre_ParVectorInnerProd(hypreB, hypreB));
         if (rnorm1 < rnorm0 * 1.0e-10 || rnorm1 < 1.0e-10) break;
         rnorm0 = rnorm1;

         hypre_ParVectorSetConstantValues(hypreB, 0.0);
         strcpy(paramString, "mJacobi");
         MLI_Utils_HypreGMRESSolve(hypreSolver, (HYPRE_Matrix) hypreAffT,
                     (HYPRE_Vector) hypreB, (HYPRE_Vector) hypreX, paramString);
         hypre_ParCSRMatrixMatvec(-1.0, hypreAffT, hypreX, 1.0, hypreB);
         rnorm1 = sqrt(hypre_ParVectorInnerProd(hypreB, hypreB));
         rnorm1 = 0.2 * log10(rnorm1/rnorm0);
         ratio2 = pow(1.0e1, rnorm1);   
         if (ratio1 > ratio2) aratio += ratio1;
         else                 aratio += ratio2;
#else
         aratio += ratio1;
         ratio2 = 0;
#endif

         /* ------------------------------------------------------- */
         /* accumulate error vector                                 */
         /* ------------------------------------------------------- */

         hypre_ParVectorAxpy(dOne, hypreX, hypreXacc);
         if (ratio1 < targetMu_ && ratio2 < targetMu_) 
         {
            printf("\tTrial %3d(%3d) : GMRES norms ratios = %16.8e %16.8e ##\n",
                   iT, iV, ratio1, ratio2);
            break;
         }
         else
            printf("\tTrial %3d(%3d) : GMRES norms ratios = %16.8e %16.8e\n",
                   iT, iV, ratio1, ratio2);
      }
      MLI_Utils_mJacobiDestroy(hypreSolver);
#endif

      /* --------------------------------------------------- */
      /* select coarse points                                */
      /* --------------------------------------------------- */
      
      if (iV == numVectors_) aratio /= (double) numVectors_; 
printf("aratio = %e\n", aratio);
      if ((aratio >= targetMu_ || (iT == 0 && localNRows == FNRows)) && 
           iT < (numTrials_-1)) 
      {
         XaccData = (double *) 
                 hypre_VectorData(hypre_ParVectorLocalVector(hypreXacc));
         sortIndices = new int[FNRows];
         for (irow = 0; irow < FNRows; irow++) sortIndices[irow] = irow;
         for (irow = 0; irow < FNRows; irow++) 
            if (XaccData[irow] < 0.0) XaccData[irow] = - XaccData[irow];
         //MLI_Utils_DbleQSort2a(XaccData, sortIndices, 0, FNRows-1);
         if (FNRows > 0) threshold = XaccData[FNRows-1] * cutThreshold_;
#if 0
         newCount = 0;
         for (ic = 0; ic < localNRows; ic++) 
         {
            threshold = XaccData[FNRows-1] * cutThreshold_;
            for (it = 0; it < 6; it++) 
            {
               for (irow = FNRows-1; irow >= 0; irow--) 
               {
                  ddata = XaccData[irow];
                  if (ddata > threshold)
                  {
                     idata = sortIndices[irow];
                     fPt = fList[idata];
                     if (indepSet[fPt] == 0) 
                     {
                        count = 0;
                        for (jcol = ADiagI[fPt]; jcol < ADiagI[fPt+1]; jcol++) 
                           if (indepSet[ADiagJ[jcol]] == 1) count++;
                        if (count <= ic)
                        {
                           newCount++;
                           indepSet[fPt] = 1;
                           for (jcol = ADiagI[fPt];jcol < ADiagI[fPt+1];jcol++) 
                              if (indepSet[ADiagJ[jcol]] == 0)
                                 indepSet[ADiagJ[jcol]] = -1;
                        }
                     }
                  }
               }
               threshold *= 0.1;
               for (irow = 0; irow < localNRows; irow++) 
                  if (indepSet[irow] < 0) indepSet[irow] = 0;
               if ((localNRows+newCount-FNRows) > (localNRows/2) && ic > 2) 
               {
                  if (((double) newCount/ (double) localNRows) > 0.05)
                     break;
               }
            }
            if ((localNRows+newCount-FNRows) > (localNRows/2) && ic > 2) 
            {
               if (((double) newCount/ (double) localNRows) > 0.05)
                  break;
            }
         }
#else
         newCount = 0;
         threshold = XaccData[FNRows-1] * cutThreshold_;
         for (it = 0; it < 1; it++) 
         {
            for (irow = FNRows-1; irow >= 0; irow--) 
            {
               ddata = XaccData[irow];
               if (ddata > threshold)
               {
                  idata = sortIndices[irow];
                  fPt = fList[idata];
                  if (indepSet[fPt] == 0) 
                  {
                     newCount++;
                     indepSet[fPt] = 1;
                     for (jcol = ADiagI[fPt];jcol < ADiagI[fPt+1];jcol++) 
                        if (indepSet[ADiagJ[jcol]] == 0 &&
                            habs(ADiagA[jcol]/ADiagA[ADiagI[fPt]]) > 1.0e-12)
                           indepSet[ADiagJ[jcol]] = -1;
                  }
               }
            }
            threshold *= 0.1;
            for (irow = 0; irow < localNRows; irow++) 
               if (indepSet[irow] < 0) indepSet[irow] = 0;
            if ((localNRows+newCount-FNRows) > (localNRows/2)) 
            {
               if (((double) newCount/ (double) localNRows) > 0.1)
                  break;
            }
         }
#endif
         delete [] sortIndices;
         if (newCount == 0) 
         {
            printf("CR stops because newCount = 0\n");
            break;
         }
      }

      /* --------------------------------------------------- */
      /* clean up                                            */
      /* --------------------------------------------------- */
      
      HYPRE_IJMatrixDestroy(IJPFF);
      hypre_ParCSRMatrixDestroy(hyprePFFT);
      hypre_ParCSRMatrixDestroy(hypreAPFC);
#ifdef HAVE_TRANS
      delete mli_AffTMat;
#endif
      HYPRE_IJMatrixDestroy(IJPFC);
      HYPRE_IJVectorDestroy(IJX);
      HYPRE_IJVectorDestroy(IJB);
      HYPRE_IJVectorDestroy(IJXacc);
      delete mli_Bvec;
      delete mli_Xvec;
      if (aratio < targetMu_ && iT != 0) break;
      if (numTrials_ == 1) break;
      delete mli_AffMat;
      delete mli_AfcMat;
      hypre_ParCSRMatrixDestroy(hypreAfc);
   }

   /* ------------------------------------------------------ */
   /* final clean up                                         */
   /* ------------------------------------------------------ */

   delete [] fList;
   (*AfcMat) = mli_AfcMat;
   return mli_AffMat;
}

/* ********************************************************************* *
 * create the prolongation matrix
 * --------------------------------------------------------------------- */

MLI_Matrix *MLI_Method_AMGCR::createPmat(int *indepSet, MLI_Matrix *mli_Amat,
                               MLI_Matrix *mli_Affmat, MLI_Matrix *mli_Afcmat)
{
   int    *ADiagI, *ADiagJ, localNRows, AffNRows, AffStartRow, irow;
   int    *rowLengs, ierr, startRow, rowCount, rowIndex, colIndex;
   int    *colInd, rowSize, jcol, one=1, maxRowLeng, nnz;
   int    *tPDiagI, *tPDiagJ, cCount, fCount, ncount, *ADDiagI, *ADDiagJ;
   int    *AD2DiagI, *AD2DiagJ, *newColInd, newRowSize, *rowStarts;
   int    *newRowStarts, nprocs, AccStartRow, AccNRows;
   double *ADiagA, *colVal, colValue, *newColVal, *DDiagA;
   double *tPDiagA, *ADDiagA, *AD2DiagA, omega=1, dtemp;
   char   paramString[100];
   HYPRE_IJMatrix     IJInvD, IJP;
   hypre_ParCSRMatrix *hypreA, *hypreAff, *hypreInvD, *hypreP, *hypreAD;
   hypre_ParCSRMatrix *hypreAD2, *hypreAfc, *hypreTmp;
   hypre_CSRMatrix    *ADiag, *DDiag, *tPDiag, *ADDiag, *AD2Diag;
   MLI_Function       *funcPtr;
   MLI_Matrix         *mli_Pmat;
   MPI_Comm           comm;
   HYPRE_Solver       ps;

   /* ------------------------------------------------------ */
   /* get matrix information                                 */
   /* ------------------------------------------------------ */

   comm = getComm();
   MPI_Comm_size(comm, &nprocs);
   hypreA = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   startRow = hypre_ParCSRMatrixFirstRowIndex(hypreA);
   localNRows = hypre_ParCSRMatrixNumRows(hypreA);

   hypreAff = (hypre_ParCSRMatrix *) mli_Affmat->getMatrix();
   AffStartRow = hypre_ParCSRMatrixFirstRowIndex(hypreAff);
   AffNRows = hypre_ParCSRMatrixNumRows(hypreAff);

   /* ------------------------------------------------------ */
   /* create the diagonal matrix of A                        */
   /* ------------------------------------------------------ */

   ierr = HYPRE_IJMatrixCreate(comm,AffStartRow,AffStartRow+AffNRows-1,
                           AffStartRow,AffStartRow+AffNRows-1,&IJInvD);
   ierr = HYPRE_IJMatrixSetObjectType(IJInvD, HYPRE_PARCSR);
   assert(!ierr);
   rowLengs = new int[AffNRows];
   for (irow = 0; irow < AffNRows; irow++) rowLengs[irow] = 1;
   ierr = HYPRE_IJMatrixSetRowSizes(IJInvD, rowLengs);
   ierr = HYPRE_IJMatrixInitialize(IJInvD);
   assert(!ierr);
   delete [] rowLengs;

   /* ------------------------------------------------------ */
   /* load the diagonal matrix of A                          */
   /* ------------------------------------------------------ */

   rowCount = 0;
   for (irow = 0; irow < localNRows; irow++)
   {
      rowIndex = startRow + irow;
      if (indepSet[irow] == 0) 
      {
         HYPRE_ParCSRMatrixGetRow((HYPRE_ParCSRMatrix) hypreA, rowIndex, 
                                  &rowSize, &colInd, &colVal);
         colValue = 1.0;
         for (jcol = 0; jcol < rowSize; jcol++)
         {
            if (colInd[jcol] == rowIndex) 
            {
               colValue = colVal[jcol];
               break;
            }
         }
         if (colValue >= 0.0)
         {
            for (jcol = 0; jcol < rowSize; jcol++)
               if (colInd[jcol] != rowIndex && 
                   (indepSet[colInd[jcol]-startRow] == 0) && 
                   colVal[jcol] > 0.0) 
                  colValue += colVal[jcol];
         }
         else
         {
            for (jcol = 0; jcol < rowSize; jcol++)
               if (colInd[jcol] != rowIndex && 
                   (indepSet[colInd[jcol]-startRow] == 0) && 
                   colVal[jcol] < 0.0) 
                  colValue += colVal[jcol];
         }
         colValue = 1.0 / colValue;
         colIndex = AffStartRow + rowCount;
         HYPRE_IJMatrixSetValues(IJInvD,1,&one,(const int *) &colIndex,
                    (const int *) &colIndex, (const double *) &colValue);
         rowCount++;
         HYPRE_ParCSRMatrixRestoreRow((HYPRE_ParCSRMatrix) hypreA, rowIndex, 
                                      &rowSize, &colInd, &colVal);
      }
   }

   /* ------------------------------------------------------ */
   /* finally assemble the diagonal matrix of A              */
   /* ------------------------------------------------------ */

   ierr = HYPRE_IJMatrixAssemble(IJInvD);
   assert( !ierr );
   HYPRE_IJMatrixGetObject(IJInvD, (void **) &hypreInvD);
   ierr += HYPRE_IJMatrixSetObjectType(IJInvD, -1);
   ierr += HYPRE_IJMatrixDestroy(IJInvD);
   assert( !ierr );

   /* ------------------------------------------------------ */
   /* generate polynomial of Aff and invD                    */
   /* ------------------------------------------------------ */

   if (PDegree_ == 0)
   {
      hypreP = hypreInvD;
      hypreInvD = NULL;
      ADiag  = hypre_ParCSRMatrixDiag(hypreP);
      ADiagI = hypre_CSRMatrixI(ADiag);
      ADiagJ = hypre_CSRMatrixJ(ADiag);
      ADiagA = hypre_CSRMatrixData(ADiag);
      for (irow = 0; irow < AffNRows; irow++) 
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++) 
            ADiagA[jcol] = - ADiagA[jcol];
   }
   else if (PDegree_ == 1)
   {
#if 1
      hypreP = hypre_ParMatmul(hypreAff, hypreInvD);
      DDiag  = hypre_ParCSRMatrixDiag(hypreInvD);
      DDiagA = hypre_CSRMatrixData(DDiag);
      ADiag  = hypre_ParCSRMatrixDiag(hypreP);
      ADiagI = hypre_CSRMatrixI(ADiag);
      ADiagJ = hypre_CSRMatrixJ(ADiag);
      ADiagA = hypre_CSRMatrixData(ADiag);
      for (irow = 0; irow < AffNRows; irow++) 
      {
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++) 
         {
            if (ADiagJ[jcol] == irow) 
                 ADiagA[jcol] = - omega*DDiagA[irow]*(2.0-omega*ADiagA[jcol]);
            else ADiagA[jcol] = omega * omega * DDiagA[irow] * ADiagA[jcol];
         }
      }
      hypre_ParCSRMatrixOwnsColStarts(hypreInvD) = 0;
      rowStarts = hypre_ParCSRMatrixRowStarts(hypreA);
      newRowStarts = (int *) malloc((nprocs+1) * sizeof(int));
      for (irow = 0; irow <= nprocs; irow++) 
         newRowStarts[irow] = rowStarts[irow];
      hypre_ParCSRMatrixRowStarts(hypreP) = newRowStarts;
#else
      ierr = HYPRE_IJMatrixCreate(comm,AffStartRow,AffStartRow+AffNRows-1,
                           AffStartRow,AffStartRow+AffNRows-1,&IJP);
      ierr = HYPRE_IJMatrixSetObjectType(IJP, HYPRE_PARCSR);
      assert(!ierr);
      rowLengs = new int[AffNRows];
      maxRowLeng = 0;
      ADiag   = hypre_ParCSRMatrixDiag(hypreAff);
      ADiagI  = hypre_CSRMatrixI(ADiag);
      ADiagJ  = hypre_CSRMatrixJ(ADiag);
      ADiagA  = hypre_CSRMatrixData(ADiag);
      for (irow = 0; irow < AffNRows; irow++)
      {
         newRowSize = 1;
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
            if (ADiagJ[jcol] == irow) {index = jcol; break;}
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
            if (ADiagJ[jcol] != irow && ADiagA[jcol]*ADiagA[index] < 0.0)
               newRowSize++;
         rowLengs[irow] = newRowSize;
         if (newRowSize > maxRowLeng) maxRowLeng = newRowSize; 
      }
      ierr = HYPRE_IJMatrixSetRowSizes(IJP, rowLengs);
      ierr = HYPRE_IJMatrixInitialize(IJP);
      assert(!ierr);
      delete [] rowLengs;
      newColInd = new int[maxRowLeng];
      newColVal = new double[maxRowLeng];
      for (irow = 0; irow < AffNRows; irow++)
      {
         newRowSize = 0;
         index = -1;
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
            if (ADiagJ[jcol] == irow) {index = jcol; break;}
         if (index == -1) printf("WARNING : zero diagonal.\n");
         newColInd[0] = AffStartRow + irow;
         newColVal[0] = ADiagA[index];
         newRowSize++;
         for (jcol = ADiagI[irow]; jcol < ADiagI[irow+1]; jcol++)
         {
            if (ADiagJ[jcol] != irow && ADiagA[jcol]*ADiagA[index] < 0.0)
            {
               newColInd[newRowSize] = AffStartRow + ADiagJ[jcol]; 
               newColVal[newRowSize++] = ADiagA[jcol];
            }
            else
            {
               newColVal[0] += ADiagA[jcol];
            }
         }
         for (jcol = 1; jcol < newRowSize; jcol++)
            newColVal[jcol] /= (-newColVal[0]);
         newColVal[0] = 1.0;
         rowIndex = AffStartRow + irow;
         ierr = HYPRE_IJMatrixSetValues(IJP, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
         assert(!ierr);
      }
      delete [] newColInd;
      delete [] newColVal;
      ierr = HYPRE_IJMatrixAssemble(IJP);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJP, (void **) &hypreAD);
      hypreP = hypre_ParMatmul(hypreAD, hypreInvD);
      hypre_ParCSRMatrixOwnsRowStarts(hypreP) = 1;
      hypre_ParCSRMatrixOwnsRowStarts(hypreAD) = 0;
      ierr += HYPRE_IJMatrixDestroy(IJP);
#endif
   }
   else if (PDegree_ == 2)
   {
      hypreAD  = hypre_ParMatmul(hypreAff, hypreInvD);
      hypreAD2 = hypre_ParMatmul(hypreAD, hypreAD);
      ADDiag   = hypre_ParCSRMatrixDiag(hypreAD);
      AD2Diag  = hypre_ParCSRMatrixDiag(hypreAD2);
      ADDiagI  = hypre_CSRMatrixI(ADDiag);
      ADDiagJ  = hypre_CSRMatrixJ(ADDiag);
      ADDiagA  = hypre_CSRMatrixData(ADDiag);
      AD2DiagI = hypre_CSRMatrixI(AD2Diag);
      AD2DiagJ = hypre_CSRMatrixJ(AD2Diag);
      AD2DiagA = hypre_CSRMatrixData(AD2Diag);
      DDiag    = hypre_ParCSRMatrixDiag(hypreInvD);
      DDiagA   = hypre_CSRMatrixData(DDiag);
      newColInd = new int[2*AffNRows];
      newColVal = new double[2*AffNRows];
      ierr = HYPRE_IJMatrixCreate(comm,AffStartRow,AffStartRow+AffNRows-1,
                           AffStartRow,AffStartRow+AffNRows-1,&IJP);
      ierr = HYPRE_IJMatrixSetObjectType(IJP, HYPRE_PARCSR);
      assert(!ierr);
      rowLengs = new int[AffNRows];
      maxRowLeng = 0;
      for (irow = 0; irow < AffNRows; irow++)
      {
         newRowSize = 0;
         for (jcol = ADDiagI[irow]; jcol < ADDiagI[irow+1]; jcol++)
            newColInd[newRowSize] = ADDiagJ[jcol]; 
         for (jcol = AD2DiagI[irow]; jcol < AD2DiagI[irow+1]; jcol++)
            newColInd[newRowSize] = AD2DiagJ[jcol]; 
         if (newRowSize > maxRowLeng) maxRowLeng = newRowSize; 
         qsort0(newColInd, 0, newRowSize-1);
         ncount = 0;
         for ( jcol = 0; jcol < newRowSize; jcol++ )
         {
            if ( newColInd[jcol] != newColInd[ncount] )
            {
               ncount++;
               newColInd[ncount] = newColInd[jcol];
            } 
         }
         newRowSize = ncount + 1;
         rowLengs[irow] = newRowSize;
      }
      ierr = HYPRE_IJMatrixSetRowSizes(IJP, rowLengs);
      ierr = HYPRE_IJMatrixInitialize(IJP);
      assert(!ierr);
      delete [] rowLengs;
      nnz = 0;
      for (irow = 0; irow < AffNRows; irow++)
      {
         rowIndex = AffStartRow + irow;
         newRowSize = 0;
         for (jcol = ADDiagI[irow]; jcol < ADDiagI[irow+1]; jcol++)
         {
            newColInd[newRowSize] = ADDiagJ[jcol]; 
            if (ADDiagJ[jcol] == irow) 
               newColVal[newRowSize++] = 3.0 * (1.0 - ADDiagA[jcol]); 
            else
               newColVal[newRowSize++] = - 3.0 * ADDiagA[jcol]; 
         }
         for (jcol = AD2DiagI[irow]; jcol < AD2DiagI[irow+1]; jcol++)
         {
            newColInd[newRowSize] = AD2DiagJ[jcol]; 
            newColVal[newRowSize++] = AD2DiagA[jcol]; 
         }
         qsort1(newColInd, newColVal, 0, newRowSize-1);
         ncount = 0;
         for ( jcol = 0; jcol < newRowSize; jcol++ )
         {
            if ( jcol != ncount && newColInd[jcol] == newColInd[ncount] )
               newColVal[ncount] += newColVal[jcol];
            else if ( newColInd[jcol] != newColInd[ncount] )
            {
               ncount++;
               newColVal[ncount] = newColVal[jcol];
               newColInd[ncount] = newColInd[jcol];
            } 
         }
         newRowSize = ncount + 1;
         for ( jcol = 0; jcol < newRowSize; jcol++ )
            newColVal[jcol] = - (DDiagA[irow] * newColVal[jcol]);
     
         ierr = HYPRE_IJMatrixSetValues(IJP, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
         nnz += newRowSize;
         assert(!ierr);
      }
      delete [] newColInd;
      delete [] newColVal;
      ierr = HYPRE_IJMatrixAssemble(IJP);
      assert( !ierr );
      HYPRE_IJMatrixGetObject(IJP, (void **) &hypreP);
      ierr += HYPRE_IJMatrixSetObjectType(IJP, -1);
      ierr += HYPRE_IJMatrixDestroy(IJP);
      assert(!ierr);
      hypre_ParCSRMatrixDestroy(hypreAD);
      hypre_ParCSRMatrixDestroy(hypreAD2);
   }
   else if (PDegree_ == 3)
   {
printf("start parasails\n");
      HYPRE_ParaSailsCreate(comm, &ps);
      HYPRE_ParaSailsSetParams(ps, 1.0e-2, 2);
      HYPRE_ParaSailsSetFilter(ps, 1.0e-2);
      HYPRE_ParaSailsSetSym(ps, 0);
      HYPRE_ParaSailsSetLogging(ps, 1);
      HYPRE_ParaSailsSetup(ps, (HYPRE_ParCSRMatrix) hypreAff, NULL, NULL);
      HYPRE_ParaSailsBuildIJMatrix(ps, &IJP);
      HYPRE_IJMatrixGetObject(IJP, (void **) &hypreP);
printf("finish parasails\n");
   }
   if (hypreInvD != NULL) hypre_ParCSRMatrixDestroy(hypreInvD);

   /* ------------------------------------------------------ */
   /* create the final P matrix (from hypreP)                */
   /* ------------------------------------------------------ */

   hypreAfc = (hypre_ParCSRMatrix *) mli_Afcmat->getMatrix();
   hypreTmp = hypre_ParMatmul(hypreP, hypreAfc);
   hypre_ParCSRMatrixOwnsRowStarts(hypreP) = 0;
   hypre_ParCSRMatrixOwnsColStarts(hypreAfc) = 0;
   hypre_ParCSRMatrixOwnsRowStarts(hypreTmp) = 1;
   hypre_ParCSRMatrixOwnsColStarts(hypreTmp) = 1;
   hypre_ParCSRMatrixDestroy(hypreP);
   hypreP = hypreTmp;
   tPDiag   = hypre_ParCSRMatrixDiag(hypreP);
   tPDiagI  = hypre_CSRMatrixI(tPDiag);
   tPDiagJ  = hypre_CSRMatrixJ(tPDiag);
   tPDiagA  = hypre_CSRMatrixData(tPDiag);
   AccStartRow = startRow - AffStartRow;
   AccNRows = localNRows - AffNRows;
   ierr = HYPRE_IJMatrixCreate(comm,startRow,startRow+localNRows-1,
                        AccStartRow,AccStartRow+AccNRows-1,&IJP);
   ierr = HYPRE_IJMatrixSetObjectType(IJP, HYPRE_PARCSR);
   assert(!ierr);
   rowLengs = new int[localNRows];
   maxRowLeng = 0;
   ncount = 0;
   for (irow = 0; irow < localNRows; irow++)
   {
      if (indepSet[irow] == 1) rowLengs[irow] = 1;
      else                     
      {
         rowLengs[irow] = tPDiagI[ncount+1] - tPDiagI[ncount];
         ncount++;
      }
      if (rowLengs[irow] > maxRowLeng) maxRowLeng = rowLengs[irow]; 
   }
   ierr = HYPRE_IJMatrixSetRowSizes(IJP, rowLengs);
   ierr = HYPRE_IJMatrixInitialize(IJP);
   assert(!ierr);
   delete [] rowLengs;
   fCount = 0;
   cCount = 0;
   newColInd = new int[maxRowLeng];
   newColVal = new double[maxRowLeng];
   for (irow = 0; irow < localNRows; irow++)
   {
      rowIndex = startRow + irow;
      if (indepSet[irow] == 1)
      {
         newRowSize = 1;
         newColInd[0] = AccStartRow + cCount;
         newColVal[0] = 1.0;
         cCount++;
      } 
      else
      {
         newRowSize = 0;
         for (jcol = tPDiagI[fCount]; jcol < tPDiagI[fCount+1]; jcol++)
         {
            newColInd[newRowSize] = tPDiagJ[jcol] + AccStartRow; 
            newColVal[newRowSize++] = tPDiagA[jcol]; 
         }
         fCount++;
      }
// pruning
#if 1
if (irow == 0) printf("pruning and scaling\n");
dtemp = 0.0;
for (jcol = 0; jcol < newRowSize; jcol++)
if (habs(newColVal[jcol]) > dtemp) dtemp = habs(newColVal[jcol]);
dtemp *= 0.25;
ncount = 0;
for (jcol = 0; jcol < newRowSize; jcol++)
if (habs(newColVal[jcol]) > dtemp) 
{
newColInd[ncount] = newColInd[jcol];
newColVal[ncount++] = newColVal[jcol];
}
newRowSize = ncount;
#endif
// scaling
#if 0
dtemp = 0.0;
for (jcol = 0; jcol < newRowSize; jcol++)
dtemp += habs(newColVal[jcol]);
dtemp = 1.0 / dtemp;
for (jcol = 0; jcol < newRowSize; jcol++)
newColVal[jcol] *= dtemp;
#endif
      if (PDegree_ == 3)
      {
         for (jcol = 0; jcol < newRowSize; jcol++)
            newColVal[jcol] = - newColVal[jcol];
      }
      ierr = HYPRE_IJMatrixSetValues(IJP, 1, &newRowSize,
                   (const int *) &rowIndex, (const int *) newColInd,
                   (const double *) newColVal);
      assert(!ierr);
   }
   delete [] newColInd;
   delete [] newColVal;
   ierr = HYPRE_IJMatrixAssemble(IJP);
   assert( !ierr );
   hypre_ParCSRMatrixDestroy(hypreP);
   HYPRE_IJMatrixGetObject(IJP, (void **) &hypreP);
   ierr += HYPRE_IJMatrixSetObjectType(IJP, -1);
   ierr += HYPRE_IJMatrixDestroy(IJP);
   assert(!ierr);

   /* ------------------------------------------------------ */
   /* package the P matrix                                   */
   /* ------------------------------------------------------ */

   sprintf(paramString, "HYPRE_ParCSR");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   mli_Pmat = new MLI_Matrix((void*) hypreP, paramString, funcPtr);
   delete funcPtr;
   return mli_Pmat;
}

/* ********************************************************************* *
 * create the restriction matrix
 * --------------------------------------------------------------------- */

MLI_Matrix *MLI_Method_AMGCR::createRmat(int *indepSet, MLI_Matrix *mli_Amat, 
                                         MLI_Matrix *mli_Affmat)
{
   int      startRow, localNRows, AffStartRow, AffNRows, RStartRow;
   int      RNRows, ierr, *rowLengs, rowCount, rowIndex, colIndex;
   int      one=1, irow;
   double   colValue;
   char     paramString[100];
   MPI_Comm comm;
   HYPRE_IJMatrix     IJR;
   hypre_ParCSRMatrix *hypreA, *hypreAff, *hypreR;
   MLI_Function *funcPtr;
   MLI_Matrix   *mli_Rmat;

   /* ------------------------------------------------------ */
   /* get matrix information                                 */
   /* ------------------------------------------------------ */

   comm = getComm();
   hypreA = (hypre_ParCSRMatrix *) mli_Amat->getMatrix();
   startRow = hypre_ParCSRMatrixFirstRowIndex(hypreA);
   localNRows = hypre_ParCSRMatrixNumRows(hypreA);

   hypreAff = (hypre_ParCSRMatrix *) mli_Affmat->getMatrix();
   AffStartRow = hypre_ParCSRMatrixFirstRowIndex(hypreAff);
   AffNRows = hypre_ParCSRMatrixNumRows(hypreAff);

   /* ------------------------------------------------------ */
   /* create a matrix context                                */
   /* ------------------------------------------------------ */

   RStartRow = startRow - AffStartRow;
   RNRows = localNRows - AffNRows;
   ierr = HYPRE_IJMatrixCreate(comm,RStartRow,RStartRow+RNRows-1,
                           startRow,startRow+localNRows-1,&IJR);
   ierr = HYPRE_IJMatrixSetObjectType(IJR, HYPRE_PARCSR);
   assert(!ierr);
   rowLengs = new int[RNRows];
   for (irow = 0; irow < RNRows; irow++) rowLengs[irow] = 1;
   ierr = HYPRE_IJMatrixSetRowSizes(IJR, rowLengs);
   ierr = HYPRE_IJMatrixInitialize(IJR);
   assert(!ierr);
   delete [] rowLengs;

   /* ------------------------------------------------------ */
   /* load the R matrix                                      */
   /* ------------------------------------------------------ */

   rowCount = 0;
   colValue = 1.0;
   for (irow = 0; irow < localNRows; irow++)
   {
      if (indepSet[irow] == 1) 
      {
         rowIndex = RStartRow + rowCount;
         colIndex = startRow + irow;
         HYPRE_IJMatrixSetValues(IJR,1,&one,(const int *) &rowIndex,
                    (const int *) &colIndex, (const double *) &colValue);
         rowCount++;
      }
   }

   /* ------------------------------------------------------ */
   /* assemble the R matrix                                  */
   /* ------------------------------------------------------ */

   ierr = HYPRE_IJMatrixAssemble(IJR);
   assert(!ierr);
   HYPRE_IJMatrixGetObject(IJR, (void **) &hypreR);
   ierr += HYPRE_IJMatrixSetObjectType(IJR, -1);
   ierr += HYPRE_IJMatrixDestroy(IJR);
   assert( !ierr );
   sprintf(paramString, "HYPRE_ParCSR");
   funcPtr = new MLI_Function();
   MLI_Utils_HypreParCSRMatrixGetDestroyFunc(funcPtr);
   mli_Rmat = new MLI_Matrix((void*) hypreR, paramString, funcPtr);
   delete funcPtr;
   return mli_Rmat;
}

/* ********************************************************************* *
 * print AMG information
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::print()
{
   int      mypid;
   MPI_Comm comm = getComm();

   MPI_Comm_rank( comm, &mypid);
   if ( mypid == 0 )
   {
      printf("\t********************************************************\n");
      printf("\t*** method name             = %s\n", getName());
      printf("\t*** number of levels        = %d\n", numLevels_);
      printf("\t*** use MIS                 = %d\n", findMIS_);
      printf("\t*** target relaxation rate  = %e\n", targetMu_);
      printf("\t*** truncation threshold    = %e\n", cutThreshold_);
      printf("\t*** number of trials        = %d\n", numTrials_);
      printf("\t*** number of trial vectors = %d\n", numVectors_);
      printf("\t*** polynomial degree       = %d\n", PDegree_);
      printf("\t*** minimum coarse size     = %d\n", minCoarseSize_);
      printf("\t*** smoother type           = %s\n", smoother_); 
      printf("\t*** smoother nsweeps        = %d\n", smootherNum_);
      printf("\t*** smoother weight         = %e\n", smootherWgts_[0]);
      printf("\t*** coarse solver type      = %s\n", coarseSolver_); 
      printf("\t*** coarse solver nsweeps   = %d\n", coarseSolverNum_);  
      printf("\t********************************************************\n");
   }
   return 0;
}

/* ********************************************************************* *
 * print AMG statistics information
 * --------------------------------------------------------------------- */

int MLI_Method_AMGCR::printStatistics(MLI *mli)
{
   int          mypid, level, globalNRows, totNRows, fineNRows;
   int          maxNnz, minNnz, fineNnz, totNnz, thisNnz, itemp;
   double       maxVal, minVal, dtemp;
   char         paramString[100];
   MLI_Matrix   *mli_Amat, *mli_Pmat;
   MPI_Comm     comm = getComm();

   /* --------------------------------------------------------------- */
   /* output header                                                   */
   /* --------------------------------------------------------------- */

   MPI_Comm_rank( comm, &mypid);
   if ( mypid == 0 )
      printf("\t****************** AMGCR Statistics ********************\n");

   /* --------------------------------------------------------------- */
   /* output processing time                                          */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      printf("\t*** number of levels = %d\n", currLevel_+1);
      printf("\t*** total RAP   time = %e seconds\n", RAPTime_);
      printf("\t*** total GenMG time = %e seconds\n", totalTime_);
      printf("\t******************** Amatrix ***************************\n");
      printf("\t*level   Nrows MaxNnz MinNnz TotalNnz  maxValue  minValue*\n");
   }

   /* --------------------------------------------------------------- */
   /* fine and coarse matrix complexity information                   */
   /* --------------------------------------------------------------- */

   totNnz = totNRows = 0;
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
      if ( mypid == 0 )
      {
         printf("\t*%3d %9d %5d  %5d %10d %8.3e %8.3e *\n",level,
                globalNRows, maxNnz, minNnz, thisNnz, maxVal, minVal);
      }
      if ( level == 0 ) fineNnz = thisNnz;
      totNnz += thisNnz;
      if ( level == 0 ) fineNRows = globalNRows;
      totNRows += globalNRows;
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
         printf("\t*%3d %9d %5d  %5d %10d %8.3e %8.3e *\n",level,
                globalNRows, maxNnz, minNnz, thisNnz, maxVal, minVal);
      }
   }

   /* --------------------------------------------------------------- */
   /* other complexity information                                    */
   /* --------------------------------------------------------------- */

   if ( mypid == 0 )
   {
      printf("\t********************************************************\n");
      dtemp = (double) totNnz / (double) fineNnz;
      printf("\t*** Amat complexity  = %e\n", dtemp);
      dtemp = (double) totNRows / (double) fineNRows;
      printf("\t*** grid complexity  = %e\n", dtemp);
      printf("\t********************************************************\n");
      fflush(stdout);
   }
   return 0;
}

