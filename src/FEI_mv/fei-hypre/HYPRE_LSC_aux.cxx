/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

//***************************************************************************
// This file holds the other functions for HYPRE_LinSysCore
//***************************************************************************

//***************************************************************************
// include files
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// system include files
//---------------------------------------------------------------------------

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#if 0 /* RDF: Not sure this is really needed */
#ifdef WIN32
#define strcmp _stricmp
#endif
#endif

//#define HAVE_SYSPDE

//---------------------------------------------------------------------------
// HYPRE include files
//---------------------------------------------------------------------------

#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_bicgstabl.h"
#include "HYPRE_parcsr_lsicg.h"
#include "HYPRE_parcsr_TFQmr.h"
#include "HYPRE_parcsr_bicgs.h"
#include "HYPRE_parcsr_symqmr.h"
#include "HYPRE_parcsr_fgmres.h"
#include "HYPRE_LinSysCore.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "HYPRE_LSI_schwarz.h"
#include "HYPRE_LSI_ddilut.h"
#include "HYPRE_LSI_ddict.h"
#include "HYPRE_LSI_poly.h"
#include "HYPRE_LSI_block.h"
#include "HYPRE_LSI_Uzawa_c.h"
#include "HYPRE_LSI_Dsuperlu.h"
#include "HYPRE_MLMaxwell.h"
#include "HYPRE_FEI.h"
//---------------------------------------------------------------------------
// SUPERLU include files
//---------------------------------------------------------------------------

#ifdef HAVE_SUPERLU
#include "slu_ddefs.h"
#include "slu_util.h"
#endif

//---------------------------------------------------------------------------
// MLI include files
//---------------------------------------------------------------------------

#ifdef HAVE_MLI
#include "HYPRE_LSI_mli.h"
#endif

//***************************************************************************
// external functions needed here
//---------------------------------------------------------------------------

extern "C" {

/*-------------------------------------------------------------------------*
 * ML functions
 *-------------------------------------------------------------------------*/

#ifdef HAVE_ML
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
   int HYPRE_LSI_MLSetDampingFactor( HYPRE_Solver, double );
   int HYPRE_LSI_MLSetMethod( HYPRE_Solver, int );
   int HYPRE_LSI_MLSetCoarsenScheme( HYPRE_Solver , int );
   int HYPRE_LSI_MLSetCoarseSolver( HYPRE_Solver, int );
   int HYPRE_LSI_MLSetNumPDEs( HYPRE_Solver, int );
#endif

/*-------------------------------------------------------------------------*
 * MLMaxwell functions
 *-------------------------------------------------------------------------*/

#ifdef HAVE_MLMAXWELL
   int HYPRE_LSI_MLMaxwellCreate(MPI_Comm, HYPRE_Solver *);
   int HYPRE_LSI_MLMaxwellDestroy(HYPRE_Solver);
   int HYPRE_LSI_MLMaxwellSetup(HYPRE_Solver, HYPRE_ParCSRMatrix,
                                HYPRE_ParVector, HYPRE_ParVector);
   int HYPRE_LSI_MLMaxwellSolve(HYPRE_Solver, HYPRE_ParCSRMatrix,
                                HYPRE_ParVector, HYPRE_ParVector);
   int HYPRE_LSI_MLMaxwellSetGMatrix(HYPRE_Solver, HYPRE_ParCSRMatrix);
   int HYPRE_LSI_MLMaxwellSetANNMatrix(HYPRE_Solver, HYPRE_ParCSRMatrix);
   int HYPRE_LSI_MLMaxwellSetStrongThreshold(HYPRE_Solver, double);
#endif

/*-------------------------------------------------------------------------*
 * other functions
 *-------------------------------------------------------------------------*/

   void  hypre_qsort1(int *, double *, int, int);
   int   HYPRE_DummyFunction(HYPRE_Solver, HYPRE_ParCSRMatrix,
                            HYPRE_ParVector, HYPRE_ParVector) {return 0;}
   int   HYPRE_LSI_Search(int*, int, int);
   int   HYPRE_LSI_SolveIdentity(HYPRE_Solver, HYPRE_ParCSRMatrix,
                                 HYPRE_ParVector , HYPRE_ParVector);
   int   HYPRE_LSI_GetParCSRMatrix(HYPRE_IJMatrix,int nrows,int nnz,int*,
                                   int*,double*);

/*-------------------------------------------------------------------------*
 * Y12 functions (obsolete)
 *-------------------------------------------------------------------------*/

#ifdef Y12M
   void y12maf_(int*,int*,double*,int*,int*,int*,int*,double*,
                int*,int*, double*,int*,double*,int*);
#endif

}

//***************************************************************************
// this function takes parameters for setting internal things like solver
// and preconditioner choice, etc.
//---------------------------------------------------------------------------

int HYPRE_LinSysCore::parameters(int numParams, char **params)
{
   int    i, k, nsweeps, rtype, olevel, reuse=0, precon_override=0;
   int    solver_override=0, solver_index=-1, precon_index=-1;
   double weight, dtemp;
   char   param[256], param1[256], param2[80], param3[80];
   int    recognized;

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
   {
      printf("%4d : HYPRE_LSC::entering parameters function.\n",mypid_);
      if ( mypid_ == 0 )
      {
         printf("%4d : HYPRE_LSC::parameters - numParams = %d\n", mypid_,
                numParams);
         for ( i = 0; i < numParams; i++ )
            printf("           param %d = %s \n", i, params[i]);
      }
   }
   if ( numParams <= 0 ) return (0);

   //-------------------------------------------------------------------
   // process the solver and preconditioner selection first
   //-------------------------------------------------------------------

   for ( i = 0; i < numParams; i++ )
   {
      sscanf(params[i],"%s", param1);
      strcpy(param3, "invalid");
      if ( !strcmp(param1, "solver") && (!solver_override) )
      {
         sscanf(params[i],"%s %s %s", param, param2, param3);
         solver_index = i;
         if (!strcmp(param3, "override")) solver_override = 1;
      }
      if ( !strcmp(param1, "preconditioner") && (!precon_override) )
      {
         sscanf(params[i],"%s %s %s", param, param2, param3);
         if ( strcmp(param2, "reuse") )
         {
            precon_index = i;
            if (!strcmp(param3, "override")) precon_override = 1;
         }
      }
   }

   //-------------------------------------------------------------------
   // select solver : cg, gmres, superlu, superlux, y12m
   //-------------------------------------------------------------------

   if ( solver_index >= 0 )
   {
      sscanf(params[solver_index],"%s %s", param, HYSolverName_);
      selectSolver(HYSolverName_);
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 && mypid_ == 0 )
         printf("       HYPRE_LSC::parameters solver = %s\n",HYSolverName_);
   }

   //-------------------------------------------------------------------
   // select preconditioner : diagonal, pilut, boomeramg, parasails
   //-------------------------------------------------------------------

   if ( precon_index >= 0 )
   {
      sscanf(params[precon_index],"%s %s", param, param1);
      selectPreconditioner(param1);
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 && mypid_ == 0 )
         printf("       HYPRE_LSC::parameters preconditioner = %s\n",
                param1);
   }

   //-------------------------------------------------------------------
   // parse all parameters
   //-------------------------------------------------------------------

   for ( i = 0; i < numParams; i++ )
   {
      sscanf(params[i],"%s", param1);

      //----------------------------------------------------------------
      // help menu
      //----------------------------------------------------------------

      recognized = 1;
      if ( !strcmp(param1, "help") )
      {
         printf("%4d : HYPRE_LinSysCore::parameters - available ones : \n",
                mypid_);
         printf("    - outputLevel <d> \n");
         printf("    - optimizeMemory \n");
         printf("    - setDebug <slideReduction1,amgDebug,printFEInfo>\n");
         printf("    - haveFEData <0,1>\n");
         printf("    - schurReduction\n");
         printf("    - slideReduction, slideReduction2, slideReduction3\n");
         printf("    - slideReductionMinNorm <f>\n");
         printf("    - matrixPartition\n");
         printf("    - AConjugateProjection <dsize>\n");
         printf("    - minResProjection <dsize>\n");
         printf("    - solver <cg,gmres,bicgstab,boomeramg,superlux,..>\n");
         printf("    - maxIterations <d>\n");
         printf("    - tolerance <f>\n");
         printf("    - gmresDim <d>\n");
         printf("    - stopCrit <absolute,relative>\n");
         printf("    - pcgRecomputeResiudal\n");
         printf("    - preconditioner <identity,diagonal,pilut,parasails,\n");
         printf("    -    boomeramg,ddilut,schwarz,ddict,poly,euclid,...\n");
         printf("    -    blockP,ml,mli,reuse,parasails_reuse> <override>\n");
         printf("    - pilutFillin or pilutRowSize <d>\n");
         printf("    - pilutDropTol <f>\n");
         printf("    - ddilutFillin <f>\n");
         printf("    - ddilutDropTol <f> (f*sparsity(A))\n");
         printf("    - ddilutOverlap\n");
         printf("    - ddilutReorder\n");
         printf("    - ddictFillin <f>\n");
         printf("    - ddictDropTol <f> (f*sparsity(A))\n");
         printf("    - schwarzNBlocks <d>\n");
         printf("    - schwarzBlockSize <d>\n");
         printf("    - polyorder <d>\n");
         printf("    - superluOrdering <natural,mmd>\n");
         printf("    - superluScale <y,n>\n");
         printf("    - amgMaxLevels <d>\n");
         printf("    - amgCoarsenType <cljp,falgout,ruge,ruge3c,pmis,hmis>\n");
         printf("    - amgMeasureType <global,local>\n");
         printf("    - amgRelaxType <jacobi,gsFast,hybrid,hybridsym,l1gs>\n");
         printf("    - amgNumSweeps <d>\n");
         printf("    - amgRelaxWeight <f>\n");
         printf("    - amgRelaxOmega <f>\n");
         printf("    - amgStrongThreshold <f>\n");
         printf("    - amgSystemSize <d>\n");
         printf("    - amgMaxIterations <d>\n");
         printf("    - amgSmoothType <d>\n");
         printf("    - amgSmoothNumLevels <d>\n");
         printf("    - amgSmoothNumSweeps <d>\n");
         printf("    - amgSchwarzRelaxWt <d>\n");
         printf("    - amgSchwarzVariant <d>\n");
         printf("    - amgSchwarzOverlap <d>\n");
         printf("    - amgSchwarzDomainType <d>\n");
         printf("    - amgUseGSMG\n");
         printf("    - amgGSMGNumSamples\n");
         printf("    - amgAggLevels <d>\n");
         printf("    - amgInterpType <d>\n");
         printf("    - amgPmax <d>\n");
         printf("    - parasailsThreshold <f>\n");
         printf("    - parasailsNlevels <d>\n");
         printf("    - parasailsFilter <f>\n");
         printf("    - parasailsLoadbal <f>\n");
         printf("    - parasailsSymmetric\n");
         printf("    - parasailsUnSymmetric\n");
         printf("    - parasailsReuse <0,1>\n");
         printf("    - euclidNlevels <d>\n");
         printf("    - euclidThreshold <f>\n");
         printf("    - blockP help (to get blockP options) \n");
         printf("    - amsNumPDEs <d>\n");
         printf("    - MLI help (to get MLI options) \n");
         printf("    - syspdeNVars <d> \n");
#ifdef HAVE_ML
         printf("    - mlNumSweeps <d>\n");
         printf("    - mlRelaxType <jacobi,sgs,vbsgs>\n");
         printf("    - mlRelaxWeight <f>\n");
         printf("    - mlStrongThreshold <f>\n");
         printf("    - mlMethod <amg>\n");
         printf("    - mlCoarseSolver <superlu,aggregation,GS>\n");
         printf("    - mlCoarsenScheme <uncoupled,coupled,mis>\n");
         printf("    - mlNumPDEs <d>\n");
#endif
      }

      //----------------------------------------------------------------
      // output level
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "outputLevel") )
      {
         sscanf(params[i],"%s %d", param, &olevel);
         if ( olevel < 0 ) olevel = 0;
         if ( olevel > 7 ) olevel = 7;
         HYOutputLevel_ = ( HYOutputLevel_ & HYFEI_HIGHMASK ) + olevel;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters outputLevel = %d\n",
                   HYOutputLevel_);
      }

      //----------------------------------------------------------------
      // turn on memory optimizer
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "optimizeMemory") )
      {
         memOptimizerFlag_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters optimizeMemory on\n");
      }

      //----------------------------------------------------------------
      // put no boundary condition on the matrix (for diagnostics only)
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "imposeNoBC") )
      {
         HYOutputLevel_ |= HYFEI_IMPOSENOBC;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters imposeNoBC on\n");
      }

      //----------------------------------------------------------------
      // turn on multiple right hand side
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "mRHS") )
      {
         mRHSFlag_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters multiple rhs on\n");
      }

      //----------------------------------------------------------------
      // set matrix trunction threshold
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "setTruncationThreshold") )
      {
         sscanf(params[i],"%s %lg", param, &truncThresh_);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters truncThresh = %e\n",
                   truncThresh_);
      }

      //----------------------------------------------------------------
      // turn on fetching diagonal
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "set_mixed_diag") )
      {
         FEI_mixedDiagFlag_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters set mixed diagonal\n");
      }

      //----------------------------------------------------------------
      // scale the matrix
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "slideReductionScaleMatrix") )
      {
         slideReductionScaleMatrix_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters slideReduction scaleMat\n");
      }

      //----------------------------------------------------------------
      // special output level
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "setDebug") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      (!strcmp(param2, "slideReduction1"))
            HYOutputLevel_ |= HYFEI_SLIDEREDUCE1;
         else if (!strcmp(param2, "slideReduction2"))
            HYOutputLevel_ |= HYFEI_SLIDEREDUCE2;
         else if (!strcmp(param2, "slideReduction3"))
            HYOutputLevel_ |= HYFEI_SLIDEREDUCE3;
         else if (!strcmp(param2, "schurReduction1"))
            HYOutputLevel_ |= HYFEI_SCHURREDUCE1;
         else if (!strcmp(param2, "schurReduction2"))
            HYOutputLevel_ |= HYFEI_SCHURREDUCE2;
         else if (!strcmp(param2, "schurReduction3"))
            HYOutputLevel_ |= HYFEI_SCHURREDUCE3;
         else if (!strcmp(param2, "amgDebug"))
            HYOutputLevel_ |= HYFEI_AMGDEBUG;
         else if (!strcmp(param2, "printMat"))
            HYOutputLevel_ |= HYFEI_PRINTMAT;
         else if (!strcmp(param2, "printSol"))
            HYOutputLevel_ |= HYFEI_PRINTSOL;
         else if (!strcmp(param2, "printReducedMat"))
            HYOutputLevel_ |= HYFEI_PRINTREDMAT;
         else if (!strcmp(param2, "printParCSRMat"))
            HYOutputLevel_ |= HYFEI_PRINTPARCSRMAT;
         else if (!strcmp(param2, "printFEInfo"))
            HYOutputLevel_ |= HYFEI_PRINTFEINFO;
         else if (!strcmp(param2, "ddilut"))
           HYOutputLevel_ |= HYFEI_DDILUT;
         else if (!strcmp(param2, "stopAfterPrint"))
           HYOutputLevel_ |= HYFEI_STOPAFTERPRINT;
         else if (!strcmp(param2, "off"))
            HYOutputLevel_ = 0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters setDebug %s.\n", param2);
      }

      //----------------------------------------------------------------
      // turn on MLI's FEData module
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "haveFEData") )
      {
         sscanf(params[i],"%s %d", param, &haveFEData_);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters haveFEData = %d\n",
                   haveFEData_);
      }
      else if ( !strcmp(param1, "haveSFEI") )
      {
         haveFEData_ = 2;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters haveSFEI\n");
      }

      //----------------------------------------------------------------
      // turn on normal equation option
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "normalEquation") )
      {
         normalEqnFlag_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters - normal equation on.\n");
      }

      //----------------------------------------------------------------
      // perform Schur complement reduction
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "schurReduction") )
      {
         schurReduction_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters - schur reduction.\n");
      }

      //----------------------------------------------------------------
      // perform slide reduction
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "slideReduction") )
      {
         slideReduction_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters - slideReduction.\n");
      }
      else if ( !strcmp(param1, "slideReduction2") )
      {
         slideReduction_ = 2;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters - slideReduction2.\n");
      }
      else if ( !strcmp(param1, "slideReduction3") )
      {
         slideReduction_ = 3;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters - slideReduction3.\n");
      }
      else if ( !strcmp(param1, "slideReduction4") )
      {
         slideReduction_ = 4;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters - slideReduction4.\n");
      }
      else if ( !strcmp(param1, "slideReductionMinNorm") )
      {
         sscanf(params[i],"%s %lg", param, &slideReductionMinNorm_);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters - slideReductionMinNorm.\n");
      }
      else if ( !strcmp(param1, "matrixPartition") )
      {
         matrixPartition_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters - matrixPartition.\n");
      }

      //----------------------------------------------------------------
      // perform A-conjugate projection
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "AConjugateProjection") )
      {
         if ( HYpbs_ != NULL )
         {
            for ( k = 0; k <= projectSize_; k++ )
               if ( HYpbs_[k] != NULL ) HYPRE_IJVectorDestroy(HYpbs_[k]);
            delete [] HYpbs_;
            HYpbs_ = NULL;
         }
         if ( HYpxs_ != NULL )
         {
            for ( k = 0; k <= projectSize_; k++ )
               if ( HYpxs_[k] != NULL ) HYPRE_IJVectorDestroy(HYpxs_[k]);
            delete [] HYpxs_;
            HYpxs_ = NULL;
         }
         sscanf(params[i],"%s %d", param, &k);
         if ( k > 0 && k < 100 ) projectSize_ = k; else projectSize_ = 10;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters AConjugateProjection = %d\n",
                   projectSize_);
         projectionScheme_ = 1;
      }

      //----------------------------------------------------------------
      // perform minimal residual projection
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "minResProjection") )
      {
         if ( HYpbs_ != NULL )
         {
            for ( k = 0; k <= projectSize_; k++ )
               if ( HYpbs_[k] != NULL ) HYPRE_IJVectorDestroy(HYpbs_[k]);
            delete [] HYpbs_;
            HYpbs_ = NULL;
         }
         if ( HYpxs_ != NULL )
         {
            for ( k = 0; k <= projectSize_; k++ )
               if ( HYpxs_[k] != NULL ) HYPRE_IJVectorDestroy(HYpxs_[k]);
            delete [] HYpxs_;
            HYpxs_ = NULL;
         }
         sscanf(params[i],"%s %d", param, &k);
         if ( k > 0 && k < 100 ) projectSize_ = k; else projectSize_ = 10;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters minResProjection = %d\n",
                   projectSize_);
         projectionScheme_ = 2;
      }

      //----------------------------------------------------------------
      // which solver to pick : cg, gmres, superlu, superlux, y12m
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "solver") )
      {
         solver_override = 0;
      }

      //----------------------------------------------------------------
      // for GMRES, the restart size
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "gmresDim") )
      {
         sscanf(params[i],"%s %d", param, &gmresDim_);
         if ( gmresDim_ < 1 ) gmresDim_ = 100;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters gmresDim = %d\n",
                   gmresDim_);
      }

      //----------------------------------------------------------------
      // for FGMRES, tell it to update the convergence criterion of its
      // preconditioner, if it is Block preconditioning
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "fgmresUpdateTol") )
      {
         fgmresUpdateTol_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters fgmresUpdateTol on\n");
      }

      //----------------------------------------------------------------
      // for GMRES, the convergence criterion
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "gmresStopCrit") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      ( !strcmp(param2, "absolute" ) ) normAbsRel_ = 1;
         else if ( !strcmp(param2, "relative" ) ) normAbsRel_ = 0;
         else                                     normAbsRel_ = 0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
             printf("       HYPRE_LSC::parameters gmresStopCrit = %s\n",
                   param2);
      }

      else if ( !strcmp(param1, "stopCrit") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      ( !strcmp(param2, "absolute") ) normAbsRel_ = 1;
         else if ( !strcmp(param2, "relative") ) normAbsRel_ = 0;
         else                                    normAbsRel_ = 0;

         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters stopCrit = %s\n",
                   param2);
      }

      //----------------------------------------------------------------
      // for PCG only
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "pcgRecomputeResidual") )
      {
         pcgRecomputeRes_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters pcgRecomputeResidual\n");
      }

      //----------------------------------------------------------------
      // preconditioner reuse
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "precond_reuse") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      ( !strcmp(param2, "on") )  HYPreconReuse_ = reuse = 1;
         else                               HYPreconReuse_ = 0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters precond_reuse = %s\n",
                   param2);
      }

      //----------------------------------------------------------------
      // which preconditioner : diagonal, pilut, boomeramg, parasails
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "preconditioner") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      ( !strcmp(param2, "reuse") ) HYPreconReuse_ = reuse = 1;
         else if ( !strcmp(param2, "parasails_reuse") ) parasailsReuse_ = 1;
      }

      //----------------------------------------------------------------
      // maximum number of iterations for pcg or gmres
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "maxIterations") )
      {
         sscanf(params[i],"%s %d", param, &maxIterations_);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters maxIterations = %d\n",
                   maxIterations_);
      }

      //----------------------------------------------------------------
      // tolerance as termination criterion
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "tolerance") )
      {
         sscanf(params[i],"%s %lg", param, &tolerance_);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters tolerance = %e\n",
                   tolerance_);
      }

      //----------------------------------------------------------------
      // pilut preconditioner : max number of nonzeros to keep per row
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "pilutFillin") )
      {
         sscanf(params[i],"%s %d", param, &pilutFillin_);
         if ( pilutFillin_ < 1 ) pilutFillin_ = 50;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters pilutFillin_ = %d\n",
                   pilutFillin_);
      }
      else if ( !strcmp(param1, "pilutRowSize") )
      {
         sscanf(params[i],"%s %d", param, &pilutFillin_);
         if ( pilutFillin_ < 1 ) pilutFillin_ = 50;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters pilutFillin_ = %d\n",
                   pilutFillin_);
      }

      //----------------------------------------------------------------
      // pilut preconditioner : threshold to drop small nonzeros
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "pilutDropTol") )
      {
         sscanf(params[i],"%s %lg", param, &pilutDropTol_);
         if (pilutDropTol_<0.0 || pilutDropTol_ >=1.0) pilutDropTol_ = 0.0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters pilutDropTol = %e\n",
                   pilutDropTol_);
      }

      //----------------------------------------------------------------
      // DDILUT preconditioner : amount of fillin (0 == same as A)
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "ddilutFillin") )
      {
         sscanf(params[i],"%s %lg", param, &ddilutFillin_);
         if ( ddilutFillin_ < 0.0 ) ddilutFillin_ = 0.0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters ddilutFillin = %e\n",
                   ddilutFillin_);
      }

      //----------------------------------------------------------------
      // DDILUT preconditioner : threshold to drop small nonzeros
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "ddilutDropTol") )
      {
         sscanf(params[i],"%s %lg", param, &ddilutDropTol_);
         if (ddilutDropTol_<0.0 || ddilutDropTol_ >=1.0) ddilutDropTol_ = 0.0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters ddilutDropTol = %e\n",
                   ddilutDropTol_);
      }

      //----------------------------------------------------------------
      // DDILUT preconditioner : turn on processor overlap
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "ddilutOverlap") )
      {
         ddilutOverlap_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters ddilutOverlap = on\n");
      }

      //----------------------------------------------------------------
      // DDILUT preconditioner : reorder based on Cuthill McKee
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "ddilutReorder") )
      {
         ddilutReorder_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters ddilutReorder = on\n");
      }

      //----------------------------------------------------------------
      // DDICT preconditioner : amount of fillin (0 == same as A)
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "ddictFillin") )
      {
         sscanf(params[i],"%s %lg", param, &ddictFillin_);
         if ( ddictFillin_ < 0.0 ) ddictFillin_ = 0.0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters ddictFillin = %e\n",
                   ddictFillin_);
      }

      //----------------------------------------------------------------
      // DDICT preconditioner : threshold to drop small nonzeros
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "ddictDropTol") )
      {
         sscanf(params[i],"%s %lg", param, &ddictDropTol_);
         if (ddictDropTol_<0.0 || ddictDropTol_ >=1.0) ddictDropTol_ = 0.0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters ddictDropTol = %e\n",
                   ddictDropTol_);
      }

      //----------------------------------------------------------------
      // Schwarz preconditioner : Fillin
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "schwarzFillin") )
      {
         sscanf(params[i],"%s %lg", param, &schwarzFillin_);
         if ( schwarzFillin_ < 0.0 ) schwarzFillin_ = 0.0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters schwarzFillin = %e\n",
                   schwarzFillin_);
      }

      //----------------------------------------------------------------
      // Schwarz preconditioner : block size
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "schwarzNBlocks") )
      {
         sscanf(params[i],"%s %d", param, &schwarzNblocks_);
         if ( schwarzNblocks_ <= 0 ) schwarzNblocks_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters schwarzNblocks = %d\n",
                   schwarzNblocks_);
      }

      //----------------------------------------------------------------
      // Schwarz preconditioner : block size
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "schwarzBlockSize") )
      {
         sscanf(params[i],"%s %d", param, &schwarzBlksize_);
         if ( schwarzBlksize_ <= 0 ) schwarzBlksize_ = 1000;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters schwarzBlockSize = %d\n",
                   schwarzBlksize_);
      }

      //----------------------------------------------------------------
      // Polynomial preconditioner : order
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "polyOrder") )
      {
         sscanf(params[i],"%s %d", param, &polyOrder_);
         if ( polyOrder_ < 0 ) polyOrder_ = 0;
         if ( polyOrder_ > 8 ) polyOrder_ = 8;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters polyOrder = %d\n",
                   polyOrder_);
      }

      //----------------------------------------------------------------
      // superlu : ordering to use (natural, mmd)
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "superluOrdering") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      ( !strcmp(param2, "natural" ) ) superluOrdering_ = 0;
         else if ( !strcmp(param2, "mmd") )      superluOrdering_ = 2;
         else                                    superluOrdering_ = 0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters superluOrdering = %s\n",
                   param2);
      }

      //----------------------------------------------------------------
      // superlu : scaling none ('N') or both col/row ('B')
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "superluScale") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if   ( !strcmp(param2, "y" ) ) superluScale_[0] = 'B';
         else                           superluScale_[0] = 'N';
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters superluScale = %s\n",
                   param2);
      }

      else recognized = 0;

      //----------------------------------------------------------------
      // amg preconditoner : coarsening type
      //----------------------------------------------------------------

      if (!recognized)
      {
      recognized = 1;
      if ( !strcmp(param1, "amgMaxLevels") )
      {
         sscanf(params[i],"%s %d", param, &amgMaxLevels_);
         if ( amgMaxLevels_ <= 0 ) amgMaxLevels_ = 30;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgMaxLevels = %d\n",
                   amgMaxLevels_);
      }

      //----------------------------------------------------------------
      // amg preconditoner : coarsening type
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amgCoarsenType") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      ( !strcmp(param2, "cljp" ) )    amgCoarsenType_ = 0;
         else if ( !strcmp(param2, "ruge" ) )    amgCoarsenType_ = 1;
         else if ( !strcmp(param2, "ruge3c" ) )  amgCoarsenType_ = 4;
         else if ( !strcmp(param2, "falgout" ) ) amgCoarsenType_ = 6;
         else if ( !strcmp(param2, "pmis" ) )    amgCoarsenType_ = 8;
         else if ( !strcmp(param2, "hmis" ) )    amgCoarsenType_ = 10;
         else                                    amgCoarsenType_ = 0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgCoarsenType = %s\n",
                   param2);
      }

      //----------------------------------------------------------------
      // amg preconditoner : measure
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amgMeasureType") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      ( !strcmp(param2, "local" ) )   amgMeasureType_ = 0;
         else if ( !strcmp(param2, "global" ) )  amgMeasureType_ = 1;
         else                                    amgMeasureType_ = 0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgCoarsenType = %s\n",
                   param2);
      }

      //----------------------------------------------------------------
      // amg preconditoner : no of relaxation sweeps per level
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amgNumSweeps") )
      {
         sscanf(params[i],"%s %d", param, &nsweeps);
         if ( nsweeps < 1 ) nsweeps = 1;
         for ( k = 0; k < 4; k++ ) amgNumSweeps_[k] = nsweeps;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgNumSweeps = %d\n",
                   nsweeps);
      }

      //---------------------------------------------------------------
      // amg preconditoner : which smoother to use
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amgRelaxType") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      ( !strcmp(param2, "jacobi" ) ) rtype = 0;
         else if ( !strcmp(param2, "CFjacobi" ) )
		{rtype = 0; amgGridRlxType_ = 1;}
         else if ( !strcmp(param2, "gsSlow") )  rtype = 1;
         else if ( !strcmp(param2, "gsFast") )  rtype = 4;
         else if ( !strcmp(param2, "hybrid" ) ) rtype = 3;
         else if ( !strcmp(param2, "CFhybrid" ) )
		{rtype = 3; amgGridRlxType_ = 1;}
         else if ( !strcmp(param2, "hybridsym" ) ) rtype = 6;
         else if ( !strcmp(param2, "l1gs" ) ) rtype = 8;
         else if ( !strcmp(param2, "CFhybridsym" ) )
		{rtype = 6; amgGridRlxType_ = 1;}
         else                                   rtype = 4;
         for ( k = 0; k < 3; k++ ) amgRelaxType_[k] = rtype;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgRelaxType = %s\n",
                   param2);
      }

      //---------------------------------------------------------------
      // amg preconditoner : damping factor for Jacobi smoother
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgRelaxWeight") )
      {
         sscanf(params[i],"%s %lg", param, &weight);
         for ( k = 0; k < 25; k++ ) amgRelaxWeight_[k] = weight;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgRelaxWeight = %e\n",
                   weight);
      }

      //---------------------------------------------------------------
      // amg preconditoner : relaxation factor for hybrid smoother
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgRelaxOmega") )
      {
         sscanf(params[i],"%s %lg", param, &weight);
         for ( k = 0; k < 25; k++ ) amgRelaxOmega_[k] = weight;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgRelaxOmega = %e\n",
                   weight);
      }

      //---------------------------------------------------------------
      // amg preconditoner : threshold to determine strong coupling
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgStrongThreshold") )
      {
         sscanf(params[i],"%s %lg", param, &amgStrongThreshold_);
         if ( amgStrongThreshold_ < 0.0 || amgStrongThreshold_ > 1.0 )
            amgStrongThreshold_ = 0.25;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgStrongThreshold = %e\n",
                    amgStrongThreshold_);
      }
      else if ( !strcmp(param1, "amgStrengthThreshold") )
      {
         sscanf(params[i],"%s %lg", param, &amgStrongThreshold_);
         if ( amgStrongThreshold_ < 0.0 || amgStrongThreshold_ > 1.0 )
            amgStrongThreshold_ = 0.25;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgStrengthThreshold = %e\n",
                    amgStrongThreshold_);
      }

      //---------------------------------------------------------------
      // amg preconditoner : choose system size
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgSystemSize") )
      {
         sscanf(params[i],"%s %d", param, &amgSystemSize_);
         if ( amgSystemSize_ <= 0 ) amgSystemSize_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgSystemSize = %d\n",
                   amgSystemSize_);
      }

      //---------------------------------------------------------------
      // amg preconditoner : choose max iterations
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgMaxIterations") )
      {
         sscanf(params[i],"%s %d", param, &amgMaxIter_);
         if ( amgMaxIter_ <= 0 ) amgMaxIter_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgMaxIter = %d\n",
                   amgMaxIter_);
      }

      //---------------------------------------------------------------
      // amg preconditoner : choose more complex smoother
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgSmoothType") )
      {
         sscanf(params[i],"%s %d", param, &amgSmoothType_);
         if ( amgSmoothType_ < 0 ) amgSmoothType_ = 0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgSmoothType = %d\n",
                   amgSmoothType_);
      }

      //---------------------------------------------------------------
      // amg preconditoner : choose no. of levels for complex smoother
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgSmoothNumLevels") )
      {
         sscanf(params[i],"%s %d", param, &amgSmoothNumLevels_);
         if ( amgSmoothNumLevels_ < 0 ) amgSmoothNumLevels_ = 0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgSmoothNumLevels = %d\n",
                   amgSmoothNumLevels_);
      }

      //---------------------------------------------------------------
      // amg preconditoner : choose no. of sweeps for complex smoother
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgSmoothNumSweeps") )
      {
         sscanf(params[i],"%s %d", param, &amgSmoothNumSweeps_);
         if ( amgSmoothNumSweeps_ < 0 ) amgSmoothNumSweeps_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgSmoothNumSweeps = %d\n",
                   amgSmoothNumSweeps_);
      }

      //---------------------------------------------------------------
      // amg preconditoner : choose relaxation weight for Schwarz smoother
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgSchwarzRelaxWt") )
      {
         sscanf(params[i],"%s %lg", param, &amgSchwarzRelaxWt_);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgSchwarzRelaxWt = %e\n",
                   amgSchwarzRelaxWt_);
      }

      //---------------------------------------------------------------
      // amg preconditoner : choose Schwarz smoother variant
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgSchwarzVariant") )
      {
         sscanf(params[i],"%s %d", param, &amgSchwarzVariant_);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgSchwarzVariant = %d\n",
                   amgSchwarzVariant_);
      }

      //---------------------------------------------------------------
      // amg preconditoner : choose Schwarz smoother overlap
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgSchwarzOverlap") )
      {
         sscanf(params[i],"%s %d", param, &amgSchwarzOverlap_);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgSchwarzOverlap = %d\n",
                   amgSchwarzOverlap_);
      }

      //----------------------------------------------------------------
      //---------------------------------------------------------------
      // amg preconditoner : choose Schwarz smoother domain type
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgSchwarzDomainType") )
      {
         sscanf(params[i],"%s %d", param, &amgSchwarzDomainType_);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgSchwarzDomainType = %d\n",
                   amgSchwarzDomainType_);
      }

      //----------------------------------------------------------------
      //----------------------------------------------------------------
      // amg preconditoner : use gsmg
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amgUseGSMG") )
      {
         amgUseGSMG_ = 1;
         if ( amgGSMGNSamples_ == 0 ) amgGSMGNSamples_ = 5;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgUseGSMG.\n");
      }

      //----------------------------------------------------------------
      // amg preconditoner : levels of aggresive coarsening
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amgAggLevels") )
      {
 	 sscanf(params[i],"%s %d", param, &amgAggLevels_);
	 if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
	   printf("       HYPRE_LSC::parameters %s = %d\n",
		  param1, amgAggLevels_);
      }

      //----------------------------------------------------------------
      // amg preconditoner : interpolation type
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amgInterpType") )
      {
 	 sscanf(params[i],"%s %d", param, &amgInterpType_);
	 if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
	   printf("       HYPRE_LSC::parameters %s = %d\n",
		  param1, amgInterpType_);
      }

      //----------------------------------------------------------------
      // amg preconditoner : interpolation truncation
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amgPmax") )
      {
 	 sscanf(params[i],"%s %d", param, &amgPmax_);
	 if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
	   printf("       HYPRE_LSC::parameters %s = %d\n",
		  param1, amgPmax_);
      }

      //---------------------------------------------------------------
      // parasails preconditoner : gsmg number of samples
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgGSMGNumSamples") )
      {
         sscanf(params[i],"%s %d", param, &amgGSMGNSamples_);
         if ( amgGSMGNSamples_ < 0 ) amgGSMGNSamples_ = 5;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgGSMGNumSamples = %d\n",
                   amgGSMGNSamples_);
      }

      else recognized = 0;
      }

      //---------------------------------------------------------------
      // parasails preconditoner : threshold ( >= 0.0 )
      //---------------------------------------------------------------

      if (!recognized)
      {
      recognized = 1;
      if ( !strcmp(param1, "parasailsThreshold") )
      {
         sscanf(params[i],"%s %lg", param, &parasailsThreshold_);
         if ( parasailsThreshold_ < 0.0 ) parasailsThreshold_ = 0.1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters parasailsThreshold = %e\n",
                   parasailsThreshold_);
      }

      //---------------------------------------------------------------
      // parasails preconditoner : nlevels ( >= 0)
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "parasailsNlevels") )
      {
         sscanf(params[i],"%s %d", param, &parasailsNlevels_);
         if ( parasailsNlevels_ < 0 ) parasailsNlevels_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters parasailsNlevels = %d\n",
                   parasailsNlevels_);
      }

      //---------------------------------------------------------------
      // parasails preconditoner : filter
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "parasailsFilter") )
      {
         sscanf(params[i],"%s %lg", param, &parasailsFilter_);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters parasailsFilter = %e\n",
                   parasailsFilter_);
      }

      //---------------------------------------------------------------
      // parasails preconditoner : loadbal
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "parasailsLoadbal") )
      {
         sscanf(params[i],"%s %lg", param, &parasailsLoadbal_);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters parasailsLoadbal = %e\n",
                   parasailsLoadbal_);
      }

      //---------------------------------------------------------------
      // parasails preconditoner : symmetry flag (1 - symm, 0 - nonsym)
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "parasailsSymmetric") )
      {
         parasailsSym_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters parasailsSym = sym\n");
      }
      else if ( !strcmp(param1, "parasailsUnSymmetric") )
      {
         parasailsSym_ = 0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters parasailsSym = nonsym\n");
      }

      //---------------------------------------------------------------
      // parasails preconditoner : reuse flag
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "parasailsReuse") )
      {
         sscanf(params[i],"%s %d", param, &parasailsReuse_);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters parasailsReuse = %d\n",
                   parasailsReuse_);
      }

      else recognized = 0;
      }

      //---------------------------------------------------------------
      // Euclid preconditoner : fill-in
      //---------------------------------------------------------------

      if (!recognized)
      {
      recognized = 1;
      if ( !strcmp(param1, "euclidNlevels") )
      {
         sscanf(params[i],"%s %d", param, &olevel);
         if ( olevel < 0 ) olevel = 0;
         sprintf( euclidargv_[1], "%d", olevel);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters euclidNlevels = %d\n",
                   olevel);
      }

      //---------------------------------------------------------------
      // Euclid preconditoner : threshold
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "euclidThreshold") )
      {
         sscanf(params[i],"%s %lg", param, &dtemp);
         if ( dtemp < 0.0 ) dtemp = 0.0;
         sprintf( euclidargv_[3], "%e", dtemp);
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters euclidThreshold = %e\n",
                   dtemp);
      }

      //---------------------------------------------------------------
      // block preconditoner (hold this until this end)
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "blockP") )
      {
         if ( HYPreconID_ == HYBLOCK )
            HYPRE_LSI_BlockPrecondSetParams(HYPrecon_, params[i]);
      }

      //---------------------------------------------------------------
      // MLI preconditoners  (hold this until the end)
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "MLI_Hybrid") )
      {
         MLI_Hybrid_GSA_ = 1;
      }
      else if ( !strcmp(param1, "MLI_Hybrid_NSIncr") )
      {
         sscanf(params[i],"%s %d", param, &MLI_Hybrid_NSIncr_);
         if ( MLI_Hybrid_NSIncr_ <= 0 ) MLI_Hybrid_NSIncr_ = 1;
         if ( MLI_Hybrid_NSIncr_ > 10 ) MLI_Hybrid_NSIncr_ = 10;
      }
      else if ( !strcmp(param1, "MLI_Hybrid_MaxIter") )
      {
         sscanf(params[i],"%s %d", param, &MLI_Hybrid_MaxIter_);
         if ( MLI_Hybrid_MaxIter_ <=  0 ) MLI_Hybrid_MaxIter_ = 10;
      }
      else if ( !strcmp(param1, "MLI_Hybrid_NTrials") )
      {
         sscanf(params[i],"%s %d", param, &MLI_Hybrid_NTrials_);
         if ( MLI_Hybrid_NTrials_ <=  0 ) MLI_Hybrid_NTrials_ = 1;
      }
      else if ( !strcmp(param1, "MLI_Hybrid_ConvRate") )
      {
         sscanf(params[i],"%s %lg", param, &MLI_Hybrid_ConvRate_);
         if ( MLI_Hybrid_ConvRate_ >=  1.0 ) MLI_Hybrid_ConvRate_ = 1.0;
         if ( MLI_Hybrid_ConvRate_ <=  0.0 ) MLI_Hybrid_ConvRate_ = 0.0;
      }
      else if ( !strcmp(param1, "MLI") )
      {
#ifdef HAVE_MLI
         if ( HYPreconID_ == HYMLI )
            HYPRE_LSI_MLISetParams(HYPrecon_, params[i]);
#else
//         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 && mypid_ == 0 )
//            printf("       HYPRE_LSC::MLI SetParams - MLI unavailable.\n");
#endif
      }

      //----------------------------------------------------------------
      // for Uzawa, the various parameters
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "Uzawa") )
      {
         if ( HYPreconID_ == HYUZAWA )
            HYPRE_LSI_UzawaSetParams(HYPrecon_, params[i]);
      }

      else recognized = 0;
      }

      //---------------------------------------------------------------
      // mlpack preconditoner : no of relaxation sweeps per level
      //---------------------------------------------------------------

      if (!recognized)
      {
      recognized = 1;
      if ( !strcmp(param1, "mlNumPresweeps") )
      {
         sscanf(params[i],"%s %d", param, &nsweeps);
         if ( nsweeps < 1 ) nsweeps = 1;
         mlNumPreSweeps_ = nsweeps;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters mlNumPresweeps = %d\n",
                   nsweeps);
      }
      else if ( !strcmp(param1, "mlNumPostsweeps") )
      {
         sscanf(params[i],"%s %d", param, &nsweeps);
         if ( nsweeps < 1 ) nsweeps = 1;
         mlNumPostSweeps_ = nsweeps;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters mlNumPostsweeps = %d\n",
                   nsweeps);
      }
      else if ( !strcmp(param1, "mlNumSweeps") )
      {
         sscanf(params[i],"%s %d", param, &nsweeps);
         if ( nsweeps < 1 ) nsweeps = 1;
         mlNumPreSweeps_  = nsweeps;
         mlNumPostSweeps_ = nsweeps;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters mlNumSweeps = %d\n",
                   nsweeps);
      }

      //---------------------------------------------------------------
      // mlpack preconditoner : which smoother to use
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "mlPresmootherType") )
      {
         sscanf(params[i],"%s %s", param, param2);
         rtype = 1;
         if      ( !strcmp(param2, "jacobi" ) )  rtype = 0;
         else if ( !strcmp(param2, "sgs") )      rtype = 1;
         else if ( !strcmp(param2, "sgsseq") )   rtype = 2;
         else if ( !strcmp(param2, "vbjacobi"))  rtype = 3;
         else if ( !strcmp(param2, "vbsgs") )    rtype = 4;
         else if ( !strcmp(param2, "vbsgsseq"))  rtype = 5;
         else if ( !strcmp(param2, "ilut") )     rtype = 6;
         else if ( !strcmp(param2, "aSchwarz") ) rtype = 7;
         else if ( !strcmp(param2, "mSchwarz") ) rtype = 8;
         mlPresmootherType_  = rtype;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters mlPresmootherType = %s\n",
                   param2);
      }
      else if ( !strcmp(param1, "mlPostsmootherType") )
      {
         sscanf(params[i],"%s %s", param, param2);
         rtype = 1;
         if      ( !strcmp(param2, "jacobi" ) ) rtype = 0;
         else if ( !strcmp(param2, "sgs") )     rtype = 1;
         else if ( !strcmp(param2, "sgsseq") )  rtype = 2;
         else if ( !strcmp(param2, "vbjacobi")) rtype = 3;
         else if ( !strcmp(param2, "vbsgs") )   rtype = 4;
         else if ( !strcmp(param2, "vbsgsseq")) rtype = 5;
         mlPostsmootherType_  = rtype;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters mlPostsmootherType = %s\n",
                    param2);
      }
      else if ( !strcmp(param1, "mlRelaxType") )
      {
         sscanf(params[i],"%s %s", param, param2);
         rtype = 1;
         if      ( !strcmp(param2, "jacobi" ) ) rtype = 0;
         else if ( !strcmp(param2, "sgs") )     rtype = 1;
         else if ( !strcmp(param2, "sgsseq") )  rtype = 2;
         else if ( !strcmp(param2, "vbjacobi")) rtype = 3;
         else if ( !strcmp(param2, "vbsgs") )   rtype = 4;
         else if ( !strcmp(param2, "vbsgsseq")) rtype = 5;
         mlPresmootherType_ = rtype;
         mlPostsmootherType_ = rtype;
         if ( rtype == 6 ) mlPostsmootherType_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters mlRelaxType = %s\n",
                   param2);
      }

      //---------------------------------------------------------------
      // mlpack preconditoner : damping factor for Jacobi smoother
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "mlRelaxWeight") )
      {
         sscanf(params[i],"%s %lg", param, &weight);
         if ( weight < 0.0 || weight > 1.0 ) weight = 0.5;
         mlRelaxWeight_ = weight;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters mlRelaxWeight = %e\n",
                    weight);
      }

      //---------------------------------------------------------------
      // mlpack preconditoner : threshold to determine strong coupling
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "mlStrongThreshold") )
      {
         sscanf(params[i],"%s %lg", param, &mlStrongThreshold_);
         if ( mlStrongThreshold_ < 0.0 || mlStrongThreshold_ > 1.0 )
            mlStrongThreshold_ = 0.08;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters mlStrongThreshold = %e\n",
                   mlStrongThreshold_);
      }

      //---------------------------------------------------------------
      // mlpack preconditoner : method to use
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "mlMethod") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      ( !strcmp(param2, "amg" ) ) mlMethod_ = 0;
         else                                    mlMethod_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters mlMethod = %d\n",mlMethod_);
      }

      //---------------------------------------------------------------
      // mlpack preconditoner : coarse solver to use
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "mlCoarseSolver") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      ( !strcmp(param2, "superlu" ) )     mlCoarseSolver_ = 0;
         else if ( !strcmp(param2, "aggregation" ) ) mlCoarseSolver_ = 1;
         else if ( !strcmp(param2, "GS" ) )          mlCoarseSolver_ = 2;
         else                                            mlCoarseSolver_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters mlCoarseSolver = %d\n",
                   mlCoarseSolver_);
      }

      //---------------------------------------------------------------
      // mlpack preconditoner : coarsening scheme to use
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "mlCoarsenScheme") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      ( !strcmp(param2, "uncoupled" ) ) mlCoarsenScheme_ = 1;
         else if ( !strcmp(param2, "coupled" ) )   mlCoarsenScheme_ = 2;
         else if ( !strcmp(param2, "mis" ) )       mlCoarsenScheme_ = 3;
         else if ( !strcmp(param2, "hybridum" ) )  mlCoarsenScheme_ = 5;
         else if ( !strcmp(param2, "hybriduc" ) )  mlCoarsenScheme_ = 6;
         else                                          mlCoarsenScheme_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters mlCoarsenScheme = %d\n",
                   mlCoarsenScheme_);
      }

      //---------------------------------------------------------------
      // mlpack preconditoner : no of PDEs (block size)
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "mlNumPDEs") )
      {
         sscanf(params[i],"%s %d", param, &mlNumPDEs_);
         if ( mlNumPDEs_ < 1 ) mlNumPDEs_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters mlNumPDEs = %d\n",
                   mlNumPDEs_);
      }

      else recognized = 0;
      }

      //---------------------------------------------------------------
      // ams preconditoner : no of PDEs (block size)
      //---------------------------------------------------------------

      if (!recognized)
      {
      recognized = 1;
      if ( !strcmp(param1, "amsNumPDEs") )
      {
         sscanf(params[i],"%s %d", param, &amsNumPDEs_);
         if ( amsNumPDEs_ < 1 ) amsNumPDEs_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amsNumPDEs = %d\n",
                   amsNumPDEs_);
      }

      //---------------------------------------------------------------
      // ams preconditoner : which smoother to use
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amsRelaxType") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      (!strcmp(param2, "jacobi"))    amsRelaxType_ = 0;
         else if (!strcmp(param2, "scjacobi"))  amsRelaxType_ = 1;
         else if (!strcmp(param2, "scgs"))      amsRelaxType_ = 2;
         else if (!strcmp(param2, "kaczmarz"))  amsRelaxType_ = 3;
         else if (!strcmp(param2, "l1gs*"))     amsRelaxType_ = 4;
         else if (!strcmp(param2, "hybridsym")) amsRelaxType_ = 6;
         else if (!strcmp(param2, "cheby"))     amsRelaxType_ = 16;
         else                                   amsRelaxType_ = 2;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amsRelaxType = %s\n",
                   param2);
      }

      //----------------------------------------------------------------
      // ams preconditoner : no of relaxation sweeps per level
      //----------------------------------------------------------------

      else if (!strcmp(param1, "amsRelaxTimes"))
      {
         sscanf(params[i],"%s %d", param, &amsRelaxTimes_);
         if (amsRelaxTimes_ < 1) amsRelaxTimes_ = 1;
         if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgRelaxTimes = %d\n",
                   amsRelaxTimes_);
      }

      //---------------------------------------------------------------
      // ams preconditoner : damping factor for Jacobi smoother
      //---------------------------------------------------------------

      else if (!strcmp(param1, "amsRelaxWeight"))
      {
         sscanf(params[i],"%s %lg", param, &amsRelaxWt_);
         if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0)
            printf("       HYPRE_LSC::parameters amgRelaxWeight = %e\n",
                   amsRelaxWt_);
      }

      //---------------------------------------------------------------
      // ams preconditoner : omega
      //---------------------------------------------------------------

      else if (!strcmp(param1, "amsRelaxOmega"))
      {
         sscanf(params[i],"%s %lg", param, &amsRelaxOmega_);
         if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0)
            printf("       HYPRE_LSC::parameters amgRelaxWeight = %e\n",
                   amsRelaxOmega_);
      }

      //---------------------------------------------------------------
      // ams preconditoner : cycle type
      //---------------------------------------------------------------

      else if (!strcmp(param1, "amsCycleType"))
      {
         sscanf(params[i],"%s %d", param, &amsCycleType_);
         if (amsCycleType_ < 1) amsCycleType_ = 1;
         if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0)
            printf("       HYPRE_LSC::parameters amgCycleType = %s\n",
                   param2);
      }

      //---------------------------------------------------------------
      // ams preconditoner : max iterations
      //---------------------------------------------------------------

      else if (!strcmp(param1, "amsMaxIterations"))
      {
         sscanf(params[i],"%s %d", param, &amsMaxIter_);
         if (amsMaxIter_ < 1) amsMaxIter_ = 1;
         if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0)
            printf("       HYPRE_LSC::parameters amsMaxIterations = %d\n",
                   amsMaxIter_);
      }

      //---------------------------------------------------------------
      // ams preconditoner : tolerance
      //---------------------------------------------------------------

      else if (!strcmp(param1, "amsTolerance"))
      {
         sscanf(params[i],"%s %lg", param, &amsTol_);
         if (amsTol_ < 0.0) amsTol_ = 0.0;
         if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0)
            printf("       HYPRE_LSC::parameters amsTolerance = %e\n",
                   amsTol_);
      }

      //---------------------------------------------------------------
      // ams preconditoner : print level
      //---------------------------------------------------------------

      else if (!strcmp(param1, "amsPrintLevel"))
      {
         sscanf(params[i],"%s %d", param, &amsPrintLevel_);
         if (amsPrintLevel_ < 0) amsPrintLevel_ = 0;
         if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0)
            printf("       HYPRE_LSC::parameters amsPrintLevel = %d\n",
                   amsPrintLevel_);
      }

      //----------------------------------------------------------------
      // amg preconditoner : alpha coarsening type
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amsAlphaCoarsenType") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      (!strcmp(param2, "cljp"))    amsAlphaCoarsenType_ = 0;
         else if (!strcmp(param2, "ruge"))    amsAlphaCoarsenType_ = 1;
         else if (!strcmp(param2, "ruge3c"))  amsAlphaCoarsenType_ = 4;
         else if (!strcmp(param2, "falgout")) amsAlphaCoarsenType_ = 6;
         else if (!strcmp(param2, "pmis"))    amsAlphaCoarsenType_ = 8;
         else if (!strcmp(param2, "hmis"))    amsAlphaCoarsenType_ = 10;
         else                                 amsAlphaCoarsenType_ = 0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amsAlphaCoarsenType = %s\n",
                   param2);
      }

      //----------------------------------------------------------------
      // amg preconditoner : coarsening type
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amsBetaCoarsenType") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      (!strcmp(param2, "cljp"))    amsBetaCoarsenType_ = 0;
         else if (!strcmp(param2, "ruge"))    amsBetaCoarsenType_ = 1;
         else if (!strcmp(param2, "ruge3c"))  amsBetaCoarsenType_ = 4;
         else if (!strcmp(param2, "falgout")) amsBetaCoarsenType_ = 6;
         else if (!strcmp(param2, "pmis"))    amsBetaCoarsenType_ = 8;
         else if (!strcmp(param2, "hmis"))    amsBetaCoarsenType_ = 10;
         else                                 amsBetaCoarsenType_ = 0;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amsBetaCoarsenType = %s\n",
                   param2);
      }

      //---------------------------------------------------------------
      // ams preconditoner : levels of aggresive coarseinig
      //---------------------------------------------------------------

      else if (!strcmp(param1, "amsAlphaAggLevels"))
      {
         sscanf(params[i],"%s %d", param, &amsAlphaAggLevels_);
         if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0)
            printf("       HYPRE_LSC::parameters amsAlphaAggLevels = %d\n",
                   amsAlphaAggLevels_);
      }

      //----------------------------------------------------------------
      // ams preconditoner : interpolation type
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amsAlphaInterpType") )
      {
 	 sscanf(params[i],"%s %d", param, &amsAlphaInterpType_);
	 if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0)
	   printf("       HYPRE_LSC::parameters amsAlphaInterpType = %d\n",
		  amsAlphaInterpType_);
      }

      //----------------------------------------------------------------
      // ams preconditoner : interpolation truncation
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amsAlphaPmax") )
      {
 	 sscanf(params[i],"%s %d", param, &amsAlphaPmax_);
	 if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0)
	   printf("       HYPRE_LSC::parameters amsAlphaPmax = %d\n",
		  amsAlphaPmax_);
      }

      //---------------------------------------------------------------
      // ams preconditoner : levels of aggresive coarseinig
      //---------------------------------------------------------------

      else if (!strcmp(param1, "amsBetaAggLevels"))
      {
         sscanf(params[i],"%s %d", param, &amsBetaAggLevels_);
         if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0)
            printf("       HYPRE_LSC::parameters amsBetaAggLevels = %d\n",
                   amsAlphaAggLevels_);
      }

      //----------------------------------------------------------------
      // ams preconditoner : interpolation type
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amsBetaInterpType") )
      {
 	 sscanf(params[i],"%s %d", param, &amsBetaInterpType_);
	 if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0)
	   printf("       HYPRE_LSC::parameters amsBetaInterpType = %d\n",
		  amsBetaInterpType_);
      }

      //----------------------------------------------------------------
      // ams preconditoner : interpolation truncation
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amsBetaPmax") )
      {
 	 sscanf(params[i],"%s %d", param, &amsBetaPmax_);
	 if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0)
	   printf("       HYPRE_LSC::parameters amsBetaPmax = %d\n",
		  amsBetaPmax_);
      }

      //---------------------------------------------------------------
      // ams preconditoner : which smoother to use
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amsAlphaRelaxType") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      (!strcmp(param2, "jacobi"))    amsAlphaRelaxType_ = 0;
         else if (!strcmp(param2, "gsSlow"))    amsAlphaRelaxType_ = 1;
         else if (!strcmp(param2, "gsFast"))    amsAlphaRelaxType_ = 4;
         else if (!strcmp(param2, "hybrid"))    amsAlphaRelaxType_ = 3;
         else if (!strcmp(param2, "hybridsym")) amsAlphaRelaxType_ = 6;
         else if (!strcmp(param2, "l1gs" ) )    amsAlphaRelaxType_ = 8;
         else                                   amsAlphaRelaxType_ = 4;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amsAlphaRelaxType = %s\n",
                   param2);
      }

      //---------------------------------------------------------------
      // ams preconditoner : which smoother to use
      //----------------------------------------------------------------

      else if ( !strcmp(param1, "amsBetaRelaxType") )
      {
         sscanf(params[i],"%s %s", param, param2);
         if      (!strcmp(param2, "jacobi"))    amsBetaRelaxType_ = 0;
         else if (!strcmp(param2, "gsSlow"))    amsBetaRelaxType_ = 1;
         else if (!strcmp(param2, "gsFast"))    amsBetaRelaxType_ = 4;
         else if (!strcmp(param2, "hybrid"))    amsBetaRelaxType_ = 3;
         else if (!strcmp(param2, "hybridsym")) amsBetaRelaxType_ = 6;
         else if (!strcmp(param2, "l1gs" ) )    amsBetaRelaxType_ = 8;
         else                                   amsBetaRelaxType_ = 4;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amsBetaRelaxType = %s\n",
                   param2);
      }

      //---------------------------------------------------------------
      // amg preconditoner : threshold to determine strong coupling
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amsAlphaStrengthThreshold") )
      {
         sscanf(params[i],"%s %lg", param, &amsAlphaStrengthThresh_);
         if (amsAlphaStrengthThresh_<0.0 || amsAlphaStrengthThresh_>1.0)
            amsAlphaStrengthThresh_ = 0.25;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amsAlphaStrengthThresh = %e\n",
                    amsAlphaStrengthThresh_);
      }

      //---------------------------------------------------------------
      // amg preconditoner : threshold to determine strong coupling
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amsBetaStrengthThreshold") )
      {
         sscanf(params[i],"%s %lg", param, &amsBetaStrengthThresh_);
         if (amsBetaStrengthThresh_<0.0 || amsBetaStrengthThresh_>1.0)
            amsBetaStrengthThresh_ = 0.25;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amsBetaStrengthThresh = %e\n",
                    amsBetaStrengthThresh_);
      }

      //---------------------------------------------------------------
      // syspde preconditoner : nvars
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "syspdeNVars") )
      {
         sscanf(params[i],"%s %d", param, &sysPDENVars_);
         if (sysPDENVars_<0 || sysPDENVars_>10)
            sysPDENVars_ = 1;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters syspdeNVars = %d\n",
                   sysPDENVars_);
      }

      else recognized = 0;
      }

      //---------------------------------------------------------------
      // error
      //---------------------------------------------------------------

      if (!recognized)
      {
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 && mypid_ == 0 )
            printf("HYPRE_LSC::parameters WARNING : %s not recognized\n",
                    params[i]);
      }
   }

   //-------------------------------------------------------------------
   // if reuse is requested, set preconditioner reuse flag
   //-------------------------------------------------------------------

   if ( reuse == 1 ) HYPreconReuse_ = 1;
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  parameters function.\n",mypid_);
   return(0);
}

//***************************************************************************
// set up preconditioners for PCG
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPCGPrecon()
{
   //-------------------------------------------------------------------
   // if matrix has been reloaded, reset preconditioner
   //-------------------------------------------------------------------

   if ( HYPreconReuse_ == 0 && HYPreconSetup_ == 1 )
      selectPreconditioner( HYPreconName_ );

   //-------------------------------------------------------------------
   // set up preconditioners
   //-------------------------------------------------------------------

   switch ( HYPreconID_ )
   {
      case HYIDENTITY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("No preconditioning \n");
           HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_SolveIdentity,
                                     HYPRE_DummyFunction, HYPrecon_);
           break;

      case HYDIAGONAL :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Diagonal preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                        HYPRE_ParCSRDiagScaleSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPILUT :
           if ( mypid_ == 0 )
              printf("HYPRE_LSI : PCG does not work with pilut.\n");
           exit(1);
           break;

      case HYDDILUT :
           if ( mypid_ == 0 )
              printf("HYPRE_LSI : PCG does not work with ddilut.\n");
           exit(1);
           break;

      case HYDDICT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDICT();
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                        HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSchwarz();
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                        HYPRE_LSI_SchwarzSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPoly();
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                        HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_ParCSRParaSailsSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconParaSails();
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_ParCSRParaSailsSolve,
                                        HYPRE_ParCSRParaSailsSetup, HYPrecon_);
               HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconBoomerAMG();
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                        HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconEuclid();
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                       HYPRE_EuclidSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBLOCK :
           printf("PCG : block preconditioning not available.\n");
           exit(1);
           break;

      case HYML :
#ifdef HAVE_ML
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconML();
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                        HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("PCG : ML preconditioning not available.\n");
#endif
           break;

      case HYMLMAXWELL :
#ifdef HAVE_MLMAXWELL
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_MLMaxwellSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconMLMaxwell();
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_MLMaxwellSolve,
                                        HYPRE_LSI_MLMaxwellSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("PCG : ML preconditioning not available.\n");
#endif
           break;

      case HYMLI :
#ifdef HAVE_MLI
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("MLI preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_MLISolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRPCGSetPrecond(HYSolver_,HYPRE_LSI_MLISolve,
                                        HYPRE_LSI_MLISetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("PCG : MLI preconditioning not available.\n");
#endif
           break;

      case HYAMS :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("AMS preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_AMSSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconAMS();
              HYPRE_ParCSRPCGSetPrecond(HYSolver_,HYPRE_AMSSolve,
                                        HYPRE_AMSSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYUZAWA :
           printf("PCG : Uzawa preconditioning not available.\n");
           exit(1);
           break;

      case HYSYSPDE :
#ifdef HAVE_SYSPDE
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("SysPDe preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSysPDE();
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
                                        HYPRE_ParCSRSysPDESetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("PCG : SysPDE preconditioning not available.\n");
#endif
           break;

      case HYDSLU :
#ifdef HYPRE_USING_DSUPERLU
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("DSuperLU preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DSuperLUSetOutputLevel(HYPrecon_, HYOutputLevel_);
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                        HYPRE_LSI_DSuperLUSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("PCG : DSuperLU preconditioning not available.\n");
#endif
   }
   return;
}

//***************************************************************************
// set up preconditioners for LSICG
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupLSICGPrecon()
{
   //-------------------------------------------------------------------
   // if matrix has been reloaded, reset preconditioner
   //-------------------------------------------------------------------

   if ( HYPreconReuse_ == 0 && HYPreconSetup_ == 1 )
      selectPreconditioner( HYPreconName_ );

   //-------------------------------------------------------------------
   // set up preconditioners
   //-------------------------------------------------------------------

   switch ( HYPreconID_ )
   {
      case HYIDENTITY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("No preconditioning \n");
           HYPRE_ParCSRLSICGSetPrecond(HYSolver_, HYPRE_LSI_SolveIdentity,
                                       HYPRE_DummyFunction, HYPrecon_);
           break;

      case HYDIAGONAL :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Diagonal preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRLSICGSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRLSICGSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                  HYPRE_ParCSRDiagScaleSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPILUT :
           if ( mypid_ == 0 )
              printf("HYPRE_LSI : LSICG does not work with pilut.\n");
           exit(1);
           break;

      case HYDDILUT :
           if ( mypid_ == 0 )
              printf("HYPRE_LSI : LSICG does not work with ddilut.\n");
           exit(1);
           break;

      case HYDDICT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRLSICGSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDICT();
              HYPRE_ParCSRLSICGSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( mypid_ == 0 )
              printf("HYPRE_LSI : LSICG does not work with Schwarz.\n");
           exit(1);
           break;

      case HYPOLY :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRLSICGSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPoly();
              HYPRE_ParCSRLSICGSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRLSICGSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconParaSails();
              HYPRE_ParCSRLSICGSetPrecond(HYSolver_, HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_ParCSRParaSailsSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRLSICGSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconBoomerAMG();
              HYPRE_ParCSRLSICGSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                          HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ( mypid_ == 0 )
              printf("HYPRE_LSI : LSICG does not work with Euclid.\n");
           exit(1);
           break;

      case HYBLOCK :
           if ( mypid_ == 0 )
              printf("HYPRE_LSI : LSICG does not work with blkprec.\n");
           exit(1);
           break;

      case HYML :
           printf("HYPRE_LSI : LSICG - MLI preconditioning not available.\n");
           break;

      case HYMLMAXWELL :
           printf("HYPRE_LSI : LSICG - MLMAXWELL not available.\n");
           break;

      case HYMLI :
#ifdef HAVE_MLI
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("MLI preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRLSICGSetPrecond(HYSolver_, HYPRE_LSI_MLISolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRLSICGSetPrecond(HYSolver_,HYPRE_LSI_MLISolve,
                                        HYPRE_LSI_MLISetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("HYPRE_LSI LSICG : MLI preconditioning not available.\n");
#endif
           break;

      case HYUZAWA :
           if ( mypid_ == 0 )
              printf("HYPRE_LSI : LSICG does not work with Uzawa.\n");
           exit(1);
           break;

      default :
           printf("CG : preconditioner unknown.\n");
           exit(1);
           break;
   }
   return;
}

//***************************************************************************
// set up preconditioners for GMRES
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupGMRESPrecon()
{
   //-------------------------------------------------------------------
   // if matrix has been reloaded, reset preconditioner
   //-------------------------------------------------------------------

   if ( HYPreconReuse_ == 0 && HYPreconSetup_ == 1 )
      selectPreconditioner( HYPreconName_ );

   //-------------------------------------------------------------------
   // set up preconditioners
   //-------------------------------------------------------------------

   switch ( HYPreconID_ )
   {
      case HYIDENTITY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("No preconditioning \n");
           HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_SolveIdentity,
                                       HYPRE_DummyFunction, HYPrecon_);
           break;

      case HYDIAGONAL :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Diagonal preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                          HYPRE_ParCSRDiagScaleSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPILUT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPILUT();
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_ParCSRPilutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDILUT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDILUT();
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                          HYPRE_LSI_DDIlutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDICT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDICT();
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSchwarz();
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_LSI_SchwarzSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPoly();
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconParaSails();
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_ParCSRParaSailsSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconBoomerAMG();
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                          HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconEuclid();
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                          HYPRE_EuclidSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBLOCK :
           printf("GMRES : block preconditioning not available.\n");
           exit(1);
           break;

      case HYML :
#ifdef HAVE_ML
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconML();
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                          HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("GMRES : ML preconditioning not available.\n");
#endif
           break;

      case HYMLMAXWELL :
#ifdef HAVE_MLMAXWELL
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_MLMaxwellSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconMLMaxwell();
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_LSI_MLMaxwellSolve,
                                          HYPRE_LSI_MLMaxwellSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("GMRES : ML preconditioning not available.\n");
#endif
           break;

      case HYMLI :
#ifdef HAVE_MLI
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("MLI preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_MLISolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_LSI_MLISolve,
                                          HYPRE_LSI_MLISetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("GMRES : MLI preconditioning not available.\n");
#endif
           break;

      case HYUZAWA :
           printf("GMRES : Uzawa preconditioning not available.\n");
           exit(1);
           break;

      case HYAMS :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("AMS preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_AMSSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconAMS();
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_AMSSolve,
                                          HYPRE_AMSSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSYSPDE :
#ifdef HY_SYSPDE
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("SysPDe preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSysPDE();
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
                                          HYPRE_ParCSRSysPDESetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("GMRES : SysPDe preconditioning not available.\n");
#endif
           break;

      case HYDSLU :
#ifdef HYPRE_USING_DSUPERLU
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("DSuperLU preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DSuperLUSetOutputLevel(HYPrecon_, HYOutputLevel_);
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                          HYPRE_LSI_DSuperLUSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("GMRES : DSuperLU preconditioning not available.\n");
#endif
           break;
   }
   return;
}

//***************************************************************************
// set up preconditioners for FGMRES
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupFGMRESPrecon()
{
   //-------------------------------------------------------------------
   // if matrix has been reloaded, reset preconditioner
   //-------------------------------------------------------------------

   if ( HYPreconReuse_ == 0 && HYPreconSetup_ == 1 )
      selectPreconditioner( HYPreconName_ );

   //-------------------------------------------------------------------
   // set up preconditioners
   //-------------------------------------------------------------------

   switch ( HYPreconID_ )
   {
      case HYIDENTITY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("No preconditioning \n");
           HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_SolveIdentity,
                                        HYPRE_DummyFunction, HYPrecon_);
           break;

      case HYDIAGONAL :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Diagonal preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                         HYPRE_ParCSRDiagScaleSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPILUT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPILUT();
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                           HYPRE_ParCSRPilutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDILUT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDILUT();
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                           HYPRE_LSI_DDIlutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDICT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDICT();
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                           HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSchwarz();
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                           HYPRE_LSI_SchwarzSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPoly();
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                           HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconParaSails();
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                         HYPRE_ParCSRParaSailsSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconBoomerAMG();
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                           HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconEuclid();
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                           HYPRE_EuclidSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBLOCK :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_,
                    HYPRE_LSI_BlockPrecondSolve, HYPRE_DummyFunction,
                    HYPrecon_);
           else
           {
              setupPreconBlock();
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_,
                                      HYPRE_LSI_BlockPrecondSolve,
                                      HYPRE_LSI_BlockPrecondSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYML :
#ifdef HAVE_ML
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconML();
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                           HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("FGMRES : ML preconditioning not available.\n");
#endif
           break;

      case HYMLMAXWELL :
           printf("FGMRES : MLMaxwell preconditioning not available.\n");
           break;

      case HYMLI :
#ifdef HAVE_MLI
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("MLI preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_MLISolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_,HYPRE_LSI_MLISolve,
                                           HYPRE_LSI_MLISetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("FGMRES : ML preconditioning not available.\n");
#endif
           break;

      case HYUZAWA :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("Uzawa preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_UzawaSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_,HYPRE_LSI_UzawaSolve,
                                           HYPRE_LSI_UzawaSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYAMS :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("AMS preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_,HYPRE_AMSSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconAMS();
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_AMSSolve,
                                           HYPRE_AMSSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSYSPDE :
#ifdef HY_SYSPDE
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("SysPDe preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
					   HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSysPDE();
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
					   HYPRE_ParCSRSysPDESetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("FGMRES : SysPDe preconditioning not available.\n");
#endif
           break;

      case HYDSLU :
#ifdef HYPRE_USING_DSUPERLU
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("DSuperLU preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DSuperLUSetOutputLevel(HYPrecon_, HYOutputLevel_);
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                           HYPRE_LSI_DSuperLUSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("FGMRES : DSuperLU preconditioning not available.\n");
#endif
           break;
   }
   return;
}

//***************************************************************************
// set up preconditioners for BiCGSTAB
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupBiCGSTABPrecon()
{
   //-------------------------------------------------------------------
   // if matrix has been reloaded, reset preconditioner
   //-------------------------------------------------------------------

   if ( HYPreconReuse_ == 0 && HYPreconSetup_ == 1 )
      selectPreconditioner( HYPreconName_ );

   //-------------------------------------------------------------------
   // set up preconditioners
   //-------------------------------------------------------------------

   switch ( HYPreconID_ )
   {
      case HYIDENTITY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("No preconditioning \n");
           HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_SolveIdentity,
                                          HYPRE_DummyFunction, HYPrecon_);
           break;

      case HYDIAGONAL :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Diagonal preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                        HYPRE_ParCSRDiagScaleSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPILUT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPILUT();
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                             HYPRE_ParCSRPilutSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDILUT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDILUT();
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                             HYPRE_LSI_DDIlutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDICT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDICT();
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                             HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSchwarz();
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                             HYPRE_LSI_SchwarzSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPoly();
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                             HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                                             HYPRE_ParCSRParaSailsSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconParaSails();
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                                             HYPRE_ParCSRParaSailsSolve,
                                             HYPRE_ParCSRParaSailsSetup,
                                             HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconBoomerAMG();
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                             HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconEuclid();
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                             HYPRE_EuclidSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBLOCK :
           printf("BiCGSTAB : block preconditioning not available.\n");
           exit(1);
           break;

      case HYML :
#ifdef HAVE_ML
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconML();
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                             HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("BiCGSTAB : ML preconditioning not available.\n");
#endif
           break;

      case HYMLMAXWELL :
           printf("BiCGSTAB : MLMaxwell preconditioning not available.\n");
           break;

      case HYMLI :
#ifdef HAVE_MLI
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("MLI preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_MLISolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,HYPRE_LSI_MLISolve,
                                             HYPRE_LSI_MLISetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("BiCGSTAB : MLI preconditioning not available.\n");
#endif
           break;

      case HYUZAWA :
           printf("BiCGSTAB : Uzawa preconditioning not available.\n");
           exit(1);
           break;

      case HYAMS :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("AMS preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,HYPRE_AMSSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconAMS();
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_AMSSolve,
                                             HYPRE_AMSSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSYSPDE :
#ifdef HY_SYSPDE
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("SysPDe preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
					     HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSysPDE();
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
					     HYPRE_ParCSRSysPDESetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("BiCGSTAB : SysPDe preconditioning not available.\n");
#endif
           break;

      case HYDSLU :
#ifdef HYPRE_USING_DSUPERLU
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("DSuperLU preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DSuperLUSetOutputLevel(HYPrecon_, HYOutputLevel_);
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                             HYPRE_LSI_DSuperLUSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("BiCGSTAB : DSuperLU preconditioning not available.\n");
#endif
           break;
   }
   return;
}

//***************************************************************************
// set up preconditioners for BiCGSTABL
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupBiCGSTABLPrecon()
{
   //-------------------------------------------------------------------
   // if matrix has been reloaded, reset preconditioner
   //-------------------------------------------------------------------

   if ( HYPreconReuse_ == 0 && HYPreconSetup_ == 1 )
      selectPreconditioner( HYPreconName_ );

   //-------------------------------------------------------------------
   // set up preconditioners
   //-------------------------------------------------------------------

   switch ( HYPreconID_ )
   {
      case HYIDENTITY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("No preconditioning \n");
           HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_SolveIdentity,
                                           HYPRE_DummyFunction, HYPrecon_);
           break;

      case HYDIAGONAL :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Diagonal preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                         HYPRE_ParCSRDiagScaleSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPILUT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_ParCSRPilutSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPILUT();
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_ParCSRPilutSolve,
                                            HYPRE_ParCSRPilutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDILUT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDILUT();
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                              HYPRE_LSI_DDIlutSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDICT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDICT();
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                              HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_LSI_SchwarzSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSchwarz();
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_LSI_SchwarzSolve,
                                             HYPRE_LSI_SchwarzSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPoly();
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                              HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,
                                              HYPRE_ParCSRParaSailsSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconParaSails();
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,
                                              HYPRE_ParCSRParaSailsSolve,
                                              HYPRE_ParCSRParaSailsSetup,
                                              HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconBoomerAMG();
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                              HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_EuclidSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconEuclid();
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_EuclidSolve,
                                              HYPRE_EuclidSetup,
                                              HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBLOCK :
           printf("BiCGSTABL : block preconditioning not available.\n");
           exit(1);
           break;

      case HYML :
#ifdef HAVE_ML
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconML();
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                              HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("BiCGSTABL : ML preconditioning not available.\n");
#endif
           break;

      case HYMLMAXWELL :
           printf("BiCGSTABL : MLMaxwell preconditioning not available.\n");
           break;

      case HYMLI :
#ifdef HAVE_MLI
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("MLI preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_MLISolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_LSI_MLISolve,
                                              HYPRE_LSI_MLISetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("BiCGSTABL : ML preconditioning not available.\n");
#endif
           break;

      case HYUZAWA :
           printf("BiCGSTABL : Uzawa preconditioning not available.\n");
           exit(1);
           break;

      case HYAMS :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("AMS preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_AMSSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconAMS();
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_AMSSolve,
                                              HYPRE_AMSSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSYSPDE :
#ifdef HY_SYSPDE
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("SysPDe preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
					      HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSysPDE();
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
					      HYPRE_ParCSRSysPDESetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("BiCGSTABL : SysPDe preconditioning not available.\n");
#endif
           break;

      case HYDSLU :
#ifdef HYPRE_USING_DSUPERLU
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("DSuperLU preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DSuperLUSetOutputLevel(HYPrecon_, HYOutputLevel_);
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                              HYPRE_LSI_DSuperLUSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("BiCGSTABL : DSuperLU preconditioning not available.\n");
#endif
           break;
   }
   return;
}

//***************************************************************************
// set up preconditioners for TFQMR
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupTFQmrPrecon()
{
   //-------------------------------------------------------------------
   // if matrix has been reloaded, reset preconditioner
   //-------------------------------------------------------------------

   if ( HYPreconReuse_ == 0 && HYPreconSetup_ == 1 )
      selectPreconditioner( HYPreconName_ );

   //-------------------------------------------------------------------
   // set up preconditioners
   //-------------------------------------------------------------------

   switch ( HYPreconID_ )
   {
      case HYIDENTITY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("No preconditioning \n");
           HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_SolveIdentity,
                                        HYPRE_DummyFunction, HYPrecon_);
           break;

      case HYDIAGONAL :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Diagonal preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                          HYPRE_ParCSRDiagScaleSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPILUT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPILUT();
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_ParCSRPilutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDILUT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDILUT();
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                          HYPRE_LSI_DDIlutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDICT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDICT();
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSchwarz();
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_LSI_SchwarzSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPoly();
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconParaSails();
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_ParCSRParaSailsSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconBoomerAMG();
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                          HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconEuclid();
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                          HYPRE_EuclidSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBLOCK :
           printf("TFQMR : block preconditioning not available.\n");
           exit(1);
           break;

      case HYML :
#ifdef HAVE_ML
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconML();
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                          HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("TFQMR : ML preconditioning not available.\n");
#endif
           break;

      case HYMLMAXWELL :
           printf("TFQMR : MLMaxwell preconditioning not available.\n");
           break;

      case HYMLI :
#ifdef HAVE_MLI
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("MLI preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_MLISolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_LSI_MLISolve,
                                          HYPRE_LSI_MLISetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("TFQMR : MLI preconditioning not available.\n");
#endif
           break;

      case HYUZAWA :
           printf("TFQMR : Uzawa preconditioning not available.\n");
           exit(1);
           break;

      case HYAMS :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("AMS preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_AMSSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconAMS();
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_AMSSolve,
                                          HYPRE_AMSSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSYSPDE :
#ifdef HY_SYSPDE
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("SysPDe preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSysPDE();
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
                                          HYPRE_ParCSRSysPDESetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("TFQMR : SysPDe preconditioning not available.\n");
#endif
           break;

      case HYDSLU :
#ifdef HYPRE_USING_DSUPERLU
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("DSuperLU preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DSuperLUSetOutputLevel(HYPrecon_, HYOutputLevel_);
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                          HYPRE_LSI_DSuperLUSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("TFQMR : DSuperLU preconditioning not available.\n");
#endif
           break;
   }
   return;
}

//***************************************************************************
// set up preconditioners for BiCGS
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupBiCGSPrecon()
{
   //-------------------------------------------------------------------
   // if matrix has been reloaded, reset preconditioner
   //-------------------------------------------------------------------

   if ( HYPreconReuse_ == 0 && HYPreconSetup_ == 1 )
      selectPreconditioner( HYPreconName_ );

   //-------------------------------------------------------------------
   // set up preconditioners
   //-------------------------------------------------------------------

   switch ( HYPreconID_ )
   {
      case HYIDENTITY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("No preconditioning \n");
           HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_SolveIdentity,
                                       HYPRE_DummyFunction, HYPrecon_);
           break;

      case HYDIAGONAL :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Diagonal preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                          HYPRE_ParCSRDiagScaleSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPILUT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPILUT();
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_ParCSRPilutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDILUT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDILUT();
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                          HYPRE_LSI_DDIlutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDICT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDICT();
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSchwarz();
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_LSI_SchwarzSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPoly();
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconParaSails();
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_ParCSRParaSailsSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconBoomerAMG();
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                          HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconEuclid();
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                          HYPRE_EuclidSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBLOCK :
           printf("BiCGS : block preconditioning not available.\n");
           exit(1);
           break;

      case HYML :
#ifdef HAVE_ML
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconML();
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                          HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("BiCGS : ML preconditioning not available.\n");
#endif
           break;

      case HYMLMAXWELL :
           printf("BiCGS : MLMaxwell preconditioning not available.\n");
           break;

      case HYMLI :
#ifdef HAVE_MLI
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("MLI preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_MLISolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_LSI_MLISolve,
                                          HYPRE_LSI_MLISetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("BiCGS : MLI preconditioning not available.\n");
#endif
           break;

      case HYUZAWA :
           printf("BiCGS : Uzawa preconditioning not available.\n");
           exit(1);
           break;

      case HYAMS :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("AMS preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_AMSSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconAMS();
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_AMSSolve,
                                          HYPRE_AMSSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSYSPDE :
#ifdef HY_SYSPDE
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("SysPDe preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
					  HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSysPDE();
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
					  HYPRE_ParCSRSysPDESetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("BiCGS : SysPDe preconditioning not available.\n");
#endif
           break;

      case HYDSLU :
#ifdef HYPRE_USING_DSUPERLU
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("DSuperLU preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DSuperLUSetOutputLevel(HYPrecon_, HYOutputLevel_);
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_DSuperLUSolve,
                                          HYPRE_LSI_DSuperLUSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("BiCGS : DSuperLU preconditioning not available.\n");
#endif
           break;
   }
   return;
}

//***************************************************************************
// set up preconditioners for Symmetric QMR
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupSymQMRPrecon()
{
   //-------------------------------------------------------------------
   // if matrix has been reloaded, reset preconditioner
   //-------------------------------------------------------------------

   if ( HYPreconReuse_ == 0 && HYPreconSetup_ == 1 )
      selectPreconditioner( HYPreconName_ );

   //-------------------------------------------------------------------
   // set up preconditioners
   //-------------------------------------------------------------------

   switch ( HYPreconID_ )
   {
      case HYIDENTITY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("No preconditioning \n");
           HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_LSI_SolveIdentity,
                                        HYPRE_DummyFunction, HYPrecon_);
           break;

      case HYDIAGONAL :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Diagonal preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                           HYPRE_ParCSRDiagScaleSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPILUT :
           printf("ERROR : PILUT does not match SymQMR in general.\n");
           exit(1);
           break;

      case HYDDILUT :
           printf("ERROR : DDILUT does not match SymQMR in general.\n");
           exit(1);
           break;

      case HYDDICT :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconDDICT();
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                           HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           printf("ERROR : Schwarz does not match SymQMR in general.\n");
           exit(1);
           break;

      case HYPOLY :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconPoly();
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                           HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
              HYPRE_ParCSRParaSailsSetLogging(HYPrecon_, 1);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconParaSails();
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_ParCSRParaSailsSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconBoomerAMG();
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                           HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           printf("ERROR : Euclid does not match SymQMR in general.\n");
           exit(1);
           break;

      case HYBLOCK :
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_,
                    HYPRE_LSI_BlockPrecondSolve, HYPRE_DummyFunction,
                    HYPrecon_);
           else
           {
              setupPreconBlock();
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_,
                                      HYPRE_LSI_BlockPrecondSolve,
                                      HYPRE_LSI_BlockPrecondSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYML :
#ifdef HAVE_ML
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconML();
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                           HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("SymQMR : ML preconditioning not available.\n");
#endif
           break;

      case HYMLMAXWELL :
           printf("SymQMR : MLMaxwell preconditioning not available.\n");
           break;

      case HYMLI :
#ifdef HAVE_MLI
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("MLI preconditioning \n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_LSI_MLISolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_,HYPRE_LSI_MLISolve,
                                           HYPRE_LSI_MLISetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("SymQMR : MLI preconditioning not available.\n");
#endif
           break;

      case HYUZAWA :
           printf("SymQMR : Uzawa preconditioning not available.\n");
           exit(1);
           break;

      case HYAMS :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("AMS preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_,HYPRE_AMSSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconAMS();
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_AMSSolve,
                                           HYPRE_AMSSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSYSPDE :
#ifdef HY_SYSPDE
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("SysPDe preconditioning\n");
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
					   HYPRE_DummyFunction, HYPrecon_);
           else
           {
              setupPreconSysPDE();
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_ParCSRSysPDESolve,
					   HYPRE_ParCSRSysPDESetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
#else
           printf("SymQMR : SysPDe preconditioning not available.\n");
#endif
           break;

      case HYDSLU :
           printf("BiCGS : DSuperLU preconditioning not an option.\n");
           break;
   }
   return;
}

//***************************************************************************
// this function sets up BOOMERAMG preconditioner
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPreconBoomerAMG()
{
   int          i, j, *num_sweeps, *relax_type, **relax_points;
   double       *relax_wt, *relax_omega;

   if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
   {
      printf("AMG max levels   = %d\n", amgMaxLevels_);
      printf("AMG coarsen type = %d\n", amgCoarsenType_);
      printf("AMG measure type = %d\n", amgMeasureType_);
      printf("AMG threshold    = %e\n", amgStrongThreshold_);
      printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
      printf("AMG relax type   = %d\n", amgRelaxType_[0]);
      if (amgGridRlxType_) printf("AMG CF smoothing \n");
      printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
      printf("AMG relax omega  = %e\n", amgRelaxOmega_[0]);
      printf("AMG system size  = %d\n", amgSystemSize_);
      printf("AMG smooth type  = %d\n", amgSmoothType_);
      printf("AMG smooth numlevels  = %d\n", amgSmoothNumLevels_);
      printf("AMG smooth numsweeps  = %d\n", amgSmoothNumSweeps_);
      printf("AMG Schwarz variant = %d\n", amgSchwarzVariant_);
      printf("AMG Schwarz overlap = %d\n", amgSchwarzOverlap_);
      printf("AMG Schwarz domain type = %d\n", amgSchwarzDomainType_);
      printf("AMG Schwarz relax weight = %e\n", amgSchwarzRelaxWt_);
   }
   if ( HYOutputLevel_ & HYFEI_AMGDEBUG )
   {
      HYPRE_BoomerAMGSetDebugFlag(HYPrecon_, 0);
      HYPRE_BoomerAMGSetPrintLevel(HYPrecon_, 1);
   }
   if ( amgSystemSize_ > 1 )
      HYPRE_BoomerAMGSetNumFunctions(HYPrecon_, amgSystemSize_);
   HYPRE_BoomerAMGSetMaxLevels(HYPrecon_, amgMaxLevels_);
   HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
   HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
   HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_,amgStrongThreshold_);
   HYPRE_BoomerAMGSetTol(HYPrecon_, 0.0e0);
   HYPRE_BoomerAMGSetMaxIter(HYPrecon_, 1);
   num_sweeps = hypre_CTAlloc(int,4,HYPRE_MEMORY_HOST);
   for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

   HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
   relax_type = hypre_CTAlloc(int,4,HYPRE_MEMORY_HOST);
   for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

   HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
   relax_wt = hypre_CTAlloc(double,amgMaxLevels_,HYPRE_MEMORY_HOST);
   for ( i = 0; i < amgMaxLevels_; i++ ) relax_wt[i] = amgRelaxWeight_[i];
   HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);

   relax_omega = hypre_CTAlloc(double,amgMaxLevels_,HYPRE_MEMORY_HOST);
   for ( i = 0; i < amgMaxLevels_; i++ ) relax_omega[i] = amgRelaxOmega_[i];
   HYPRE_BoomerAMGSetOmega(HYPrecon_, relax_omega);

   if (amgGridRlxType_)
   {
      relax_points = hypre_CTAlloc(int*,4,HYPRE_MEMORY_HOST);
      relax_points[0] = hypre_CTAlloc(int,num_sweeps[0],HYPRE_MEMORY_HOST);
      for ( j = 0; j < num_sweeps[0]; j++ ) relax_points[0][j] = 0;
      relax_points[1] = hypre_CTAlloc(int,2*num_sweeps[1],HYPRE_MEMORY_HOST);
      for ( j = 0; j < num_sweeps[1]; j+=2 )
	 {relax_points[1][j] = -1;relax_points[1][j+1] =  1;}
      relax_points[2] = hypre_CTAlloc(int,2*num_sweeps[2],HYPRE_MEMORY_HOST);
      for ( j = 0; j < num_sweeps[2]; j+=2 )
	 {relax_points[2][j] = -1;relax_points[2][j+1] =  1;}
      relax_points[3] = hypre_CTAlloc(int,num_sweeps[3],HYPRE_MEMORY_HOST);
      for ( j = 0; j < num_sweeps[3]; j++ ) relax_points[3][j] = 0;
   }
   else
   {
      relax_points = hypre_CTAlloc(int*,4,HYPRE_MEMORY_HOST);
      for ( i = 0; i < 4; i++ )
      {
         relax_points[i] = hypre_CTAlloc(int,num_sweeps[i],HYPRE_MEMORY_HOST);
         for ( j = 0; j < num_sweeps[i]; j++ ) relax_points[i][j] = 0;
      }
   }
   HYPRE_BoomerAMGSetGridRelaxPoints(HYPrecon_, relax_points);

   if ( amgSmoothNumLevels_ > 0 )
   {
      HYPRE_BoomerAMGSetSmoothType(HYPrecon_, amgSmoothType_);
      HYPRE_BoomerAMGSetSmoothNumLevels(HYPrecon_, amgSmoothNumLevels_);
      HYPRE_BoomerAMGSetSmoothNumSweeps(HYPrecon_, amgSmoothNumSweeps_);
      HYPRE_BoomerAMGSetSchwarzRlxWeight(HYPrecon_, amgSchwarzRelaxWt_);
      HYPRE_BoomerAMGSetVariant(HYPrecon_, amgSchwarzVariant_);
      HYPRE_BoomerAMGSetOverlap(HYPrecon_, amgSchwarzOverlap_);
      HYPRE_BoomerAMGSetDomainType(HYPrecon_, amgSchwarzDomainType_);
   }

   if ( amgUseGSMG_ == 1 )
   {
      HYPRE_BoomerAMGSetGSMG(HYPrecon_, 4);
      HYPRE_BoomerAMGSetNumSamples(HYPrecon_,amgGSMGNSamples_);
   }

   HYPRE_BoomerAMGSetAggNumLevels(HYPrecon_, amgAggLevels_);
   HYPRE_BoomerAMGSetInterpType(HYPrecon_, amgInterpType_);
   HYPRE_BoomerAMGSetPMaxElmts(HYPrecon_, amgPmax_);
}

//***************************************************************************
// this function sets up ML preconditioner
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPreconML()
{
#ifdef HAVE_ML
   if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
   {
      printf("ML strong threshold = %e\n", mlStrongThreshold_);
      printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
      printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
      printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
      printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
      printf("ML relax weight     = %e\n", mlRelaxWeight_);
   }
   HYPRE_LSI_MLSetMethod(HYPrecon_,mlMethod_);
   HYPRE_LSI_MLSetCoarseSolver(HYPrecon_,mlCoarseSolver_);
   HYPRE_LSI_MLSetCoarsenScheme(HYPrecon_,mlCoarsenScheme_);
   HYPRE_LSI_MLSetStrongThreshold(HYPrecon_,mlStrongThreshold_);
   HYPRE_LSI_MLSetNumPreSmoothings(HYPrecon_,mlNumPreSweeps_);
   HYPRE_LSI_MLSetNumPostSmoothings(HYPrecon_,mlNumPostSweeps_);
   HYPRE_LSI_MLSetPreSmoother(HYPrecon_,mlPresmootherType_);
   HYPRE_LSI_MLSetPostSmoother(HYPrecon_,mlPostsmootherType_);
   HYPRE_LSI_MLSetDampingFactor(HYPrecon_,mlRelaxWeight_);
   HYPRE_LSI_MLSetNumPDEs(HYPrecon_,mlNumPDEs_);
#else
   return;
#endif
}

//***************************************************************************
// this function sets up MLMaxwell preconditioner
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPreconMLMaxwell()
{
#ifdef HAVE_MLMAXWELL
   HYPRE_ParCSRMatrix A_csr;

   if (maxwellGEN_ != NULL)
      HYPRE_LSI_MLMaxwellSetGMatrix(HYPrecon_,maxwellGEN_);
   else
   {
      printf("HYPRE_LSC::setupPreconMLMaxwell ERROR - no G matrix.\n");
      exit(1);
   }
   if (maxwellANN_ == NULL)
   {
      HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
      hypre_BoomerAMGBuildCoarseOperator((hypre_ParCSRMatrix *) maxwellGEN_,
                                      (hypre_ParCSRMatrix *) A_csr,
                                      (hypre_ParCSRMatrix *) maxwellGEN_,
                                      (hypre_ParCSRMatrix **) &maxwellANN_);
   }
   HYPRE_LSI_MLMaxwellSetANNMatrix(HYPrecon_,maxwellANN_);
#else
   return;
#endif
}

//***************************************************************************
// this function sets up AMS preconditioner
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPreconAMS()
{
#if 0
   HYPRE_ParVector    parVecX, parVecY, parVecZ;
#endif

   /* Set AMS parameters */
   HYPRE_AMSSetDimension(HYPrecon_, amsNumPDEs_);
   HYPRE_AMSSetMaxIter(HYPrecon_, amsMaxIter_);
   HYPRE_AMSSetTol(HYPrecon_, amsTol_);
   HYPRE_AMSSetCycleType(HYPrecon_, amsCycleType_);
   HYPRE_AMSSetPrintLevel(HYPrecon_, amsPrintLevel_);
   HYPRE_AMSSetSmoothingOptions(HYPrecon_, amsRelaxType_, amsRelaxTimes_,
                                amsRelaxWt_, amsRelaxOmega_);

   if (amsBetaPoisson_ != NULL)
      HYPRE_AMSSetBetaPoissonMatrix(HYPrecon_, amsBetaPoisson_);

   HYPRE_AMSSetAlphaAMGOptions(HYPrecon_, amsAlphaCoarsenType_,
            amsAlphaAggLevels_, amsAlphaRelaxType_, amsAlphaStrengthThresh_,
            amsAlphaInterpType_, amsAlphaPmax_);
   HYPRE_AMSSetBetaAMGOptions(HYPrecon_, amsBetaCoarsenType_,
            amsBetaAggLevels_, amsBetaRelaxType_, amsBetaStrengthThresh_,
            amsBetaInterpType_, amsBetaPmax_);

#if 0
   if (maxwellGEN_ != NULL)
      HYPRE_AMSSetDiscreteGradient(HYPrecon_, maxwellGEN_);
   else
   {
      printf("HYPRE_LSC::setupPreconAMS ERROR - no G matrix.\n");
      exit(1);
   }
   if (amsX_ == NULL && amsY_ != NULL)
   {
      HYPRE_IJVectorGetObject(amsX_, (void **) &parVecX);
      HYPRE_IJVectorGetObject(amsY_, (void **) &parVecY);
      HYPRE_IJVectorGetObject(amsZ_, (void **) &parVecZ);
      HYPRE_AMSSetCoordinateVectors(HYPrecon_,parVecX,parVecY,parVecZ);
   }
#endif

   // Call AMS to construct the discrete gradient matrix G
   // and the nodal coordinate vectors
   {
      HYPRE_ParCSRMatrix A_csr;
      HYPRE_ParVector    b_csr;
      HYPRE_ParVector    x_csr;

      HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
      HYPRE_IJVectorGetObject(currB_, (void **) &b_csr);
      HYPRE_IJVectorGetObject(currX_, (void **) &x_csr);

      if( amsG_ == NULL )  {

        //Old way of doing things
        //only works for 1 domain per processor (in ALE3D)
        //not compatible with contact
        HYPRE_AMSFEISetup(HYPrecon_,
                          A_csr,
                          b_csr,
                          x_csr,
                          AMSData_.EdgeNodeList_,
                          AMSData_.NodeNumbers_,
                          AMSData_.numEdges_,
                          AMSData_.numLocalNodes_,
                          AMSData_.numNodes_,
                          AMSData_.NodalCoord_);
      } else {
        //New Code//
        HYPRE_ParCSRMatrix G_csr;
        HYPRE_ParVector X_csr;
        HYPRE_ParVector Y_csr;
        HYPRE_ParVector Z_csr;
        HYPRE_IJMatrixGetObject(amsG_, (void **) &G_csr);
        HYPRE_IJVectorGetObject(amsX_, (void **) &X_csr);
        HYPRE_IJVectorGetObject(amsY_, (void **) &Y_csr);
        HYPRE_IJVectorGetObject(amsZ_, (void **) &Z_csr);
        HYPRE_AMSSetCoordinateVectors(HYPrecon_,X_csr,Y_csr,Z_csr);
        bool debugprint = false;
        if( debugprint ) {
          HYPRE_ParCSRMatrixPrint( G_csr, "G.parcsr" );
          HYPRE_ParCSRMatrixPrint( A_csr, "A.parcsr" );
          HYPRE_ParVectorPrint(    b_csr, "B.parvector" );
          HYPRE_ParVectorPrint(    X_csr, "X.parvector" );
          HYPRE_ParVectorPrint(    Y_csr, "Y.parvector" );
          HYPRE_ParVectorPrint(    Z_csr, "Z.parvector" );
        }
        HYPRE_AMSSetDiscreteGradient(HYPrecon_,G_csr);
      }
      if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
        printf("AMSprecon: finished building auxiliary info, calling AMSSetup\n");
      //int ierr = HYPRE_AMSSetup(HYPrecon_,A_csr,b_csr,x_csr);
      if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
        printf("AMSprecon: finished with AMSSetup\n");
   }

}

//***************************************************************************
// this function sets up DDICT preconditioner
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPreconDDICT()
{
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
   {
      printf("DDICT - fillin   = %e\n", ddictFillin_);
      printf("DDICT - drop tol = %e\n", ddictDropTol_);
   }
   if ( HYOutputLevel_ & HYFEI_DDILUT )
      HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);
   HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
   HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
}

//***************************************************************************
// this function sets up DDILUT preconditioner
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPreconDDILUT()
{
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
   {
      printf("DDILUT - fillin   = %e\n", ddilutFillin_);
      printf("DDILUT - drop tol = %e\n", ddilutDropTol_);
   }
   if ( HYOutputLevel_ & HYFEI_DDILUT )
      HYPRE_LSI_DDIlutSetOutputLevel(HYPrecon_,2);
   if ( ddilutReorder_ ) HYPRE_LSI_DDIlutSetReorder(HYPrecon_);
   HYPRE_LSI_DDIlutSetFillin(HYPrecon_,ddilutFillin_);
   HYPRE_LSI_DDIlutSetDropTolerance(HYPrecon_,ddilutDropTol_);
   if ( ddilutOverlap_ == 1 ) HYPRE_LSI_DDIlutSetOverlap(HYPrecon_);
   if ( ddilutReorder_ == 1 ) HYPRE_LSI_DDIlutSetReorder(HYPrecon_);
}

//***************************************************************************
// this function sets up Schwarz preconditioner
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPreconSchwarz()
{
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
   {
      printf("Schwarz - ILU fillin = %e\n", schwarzFillin_);
      printf("Schwarz - nBlocks    = %d\n", schwarzNblocks_);
      printf("Schwarz - blockSize  = %d\n", schwarzBlksize_);
   }
   if ( HYOutputLevel_ & HYFEI_DDILUT )
      HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);
   HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
   HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
   HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
}

//***************************************************************************
// this function sets up Polynomial preconditioner
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPreconPoly()
{
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
      printf("Polynomial preconditioning - order = %d\n",polyOrder_ );
   HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
}

//***************************************************************************
// this function sets up ParaSails preconditioner
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPreconParaSails()
{
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
   {
      printf("ParaSails - nlevels   = %d\n",parasailsNlevels_);
      printf("ParaSails - threshold = %e\n",parasailsThreshold_);
      printf("ParaSails - filter    = %e\n",parasailsFilter_);
      printf("ParaSails - sym       = %d\n",parasailsSym_);
      printf("ParaSails - loadbal   = %e\n",parasailsLoadbal_);
   }
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
      HYPRE_ParCSRParaSailsSetLogging(HYPrecon_, 1);
   HYPRE_ParCSRParaSailsSetSym(HYPrecon_,parasailsSym_);
   HYPRE_ParCSRParaSailsSetParams(HYPrecon_, parasailsThreshold_,
                                  parasailsNlevels_);
   HYPRE_ParCSRParaSailsSetFilter(HYPrecon_, parasailsFilter_);
   HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_, parasailsLoadbal_);
   HYPRE_ParCSRParaSailsSetReuse(HYPrecon_, parasailsReuse_);
}

//***************************************************************************
// this function sets up Euclid preconditioner
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPreconEuclid()
{
   if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
   {
      for ( int i = 0; i < euclidargc_; i++ )
         printf("Euclid parameter : %s %s\n", euclidargv_[2*i],
                                              euclidargv_[2*i+1]);
   }
   HYPRE_EuclidSetParams(HYPrecon_,euclidargc_*2,euclidargv_);
}

//***************************************************************************
// this function sets up Pilut preconditioner
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPreconPILUT()
{
   if (pilutFillin_ == 0) pilutFillin_ = pilutMaxNnzPerRow_;
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
   {
      printf("PILUT - row size = %d\n", pilutFillin_);
      printf("PILUT - drop tol = %e\n", pilutDropTol_);
   }
   HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutFillin_);
   HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
}

//***************************************************************************
// this function sets up block preconditioner
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPreconBlock()
{
   HYPRE_Lookup *newLookup;
   newLookup = hypre_TAlloc(HYPRE_Lookup, 1, HYPRE_MEMORY_HOST);
   newLookup->object = (void *) lookup_;
   HYPRE_LSI_BlockPrecondSetLookup( HYPrecon_, newLookup );
   free( newLookup );
}

//***************************************************************************
// this function sets up system pde preconditioner
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPreconSysPDE()
{
#ifdef HAVE_SYSPDE
   if (sysPDEMethod_ >= 0)
      HYPRE_ParCSRSysPDESetMethod(HYPrecon_, sysPDEMethod_);
   if (sysPDEFormat_ >= 0)
      HYPRE_ParCSRFormat(HYPrecon_, sysPDEFormat_);
   if (sysPDETol_ > 0.0)
      HYPRE_ParCSRSysPDESetTol(HYPrecon_, sysPDETol_);
   if (sysPDEMaxIter_ > 0)
      HYPRE_ParCSRSysPDESetMaxIter(HYPrecon_, sysPDEMaxIter_);
   if (sysPDENumPre_ > 0)
      HYPRE_ParCSRSysPDESetNPre(HYPrecon_, sysPDENumPre_);
   if (sysPDENumPost_ > 0)
      HYPRE_ParCSRSysPDESetNPost(HYPrecon_, sysPDENumPost_);
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
      HYPRE_ParCSRSysPDESetLogging(HYPrecon_, 1);
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
      HYPRE_ParCSRSysPDESetPrintLevel(HYPrecon_,  1);
#endif
}

//***************************************************************************
// this function solve the incoming linear system using Boomeramg
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::solveUsingBoomeramg(int& status)
{
   int                i, j, *relax_type, *num_sweeps, **relax_points;
   double             *relax_wt, *relax_omega;
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_ParVector    b_csr;
   HYPRE_ParVector    x_csr;

   //-------------------------------------------------------------------
   // get matrix and vectors in ParCSR format
   //-------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
   HYPRE_IJVectorGetObject(currB_, (void **) &b_csr);
   HYPRE_IJVectorGetObject(currX_, (void **) &x_csr);

   //-------------------------------------------------------------------
   // set BoomerAMG parameters
   //-------------------------------------------------------------------

   HYPRE_BoomerAMGSetCoarsenType(HYSolver_, amgCoarsenType_);
   HYPRE_BoomerAMGSetMeasureType(HYSolver_, amgMeasureType_);
   HYPRE_BoomerAMGSetStrongThreshold(HYSolver_, amgStrongThreshold_);

   num_sweeps = hypre_CTAlloc(int,4,HYPRE_MEMORY_HOST);
   for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];
   HYPRE_BoomerAMGSetNumGridSweeps(HYSolver_, num_sweeps);

   relax_type = hypre_CTAlloc(int,4,HYPRE_MEMORY_HOST);
   for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];
   HYPRE_BoomerAMGSetGridRelaxType(HYSolver_, relax_type);

   HYPRE_BoomerAMGSetMaxLevels(HYPrecon_, amgMaxLevels_);
   relax_wt = hypre_CTAlloc(double, amgMaxLevels_,HYPRE_MEMORY_HOST);
   for ( i = 0; i <  amgMaxLevels_; i++ ) relax_wt[i] = amgRelaxWeight_[i];
   HYPRE_BoomerAMGSetRelaxWeight(HYSolver_, relax_wt);

   relax_omega = hypre_CTAlloc(double, amgMaxLevels_,HYPRE_MEMORY_HOST);
   for ( i = 0; i <  amgMaxLevels_; i++ ) relax_omega[i] = amgRelaxOmega_[i];
   HYPRE_BoomerAMGSetOmega(HYPrecon_, relax_omega);

   relax_points = hypre_CTAlloc(int*,4,HYPRE_MEMORY_HOST);
   for ( i = 0; i < 4; i++ )
   {
      relax_points[i] = hypre_CTAlloc(int,num_sweeps[i],HYPRE_MEMORY_HOST);
      for ( j = 0; j < num_sweeps[i]; j++ ) relax_points[i][j] = 0;
   }
   HYPRE_BoomerAMGSetGridRelaxPoints(HYPrecon_, relax_points);
   if ( amgSmoothNumLevels_ > 0 )
   {
      HYPRE_BoomerAMGSetSmoothType(HYPrecon_, amgSmoothType_);
      HYPRE_BoomerAMGSetSmoothNumLevels(HYPrecon_, amgSmoothNumLevels_);
      HYPRE_BoomerAMGSetSmoothNumSweeps(HYPrecon_, amgSmoothNumSweeps_);
      HYPRE_BoomerAMGSetSchwarzRlxWeight(HYPrecon_, amgSchwarzRelaxWt_);
      HYPRE_BoomerAMGSetVariant(HYPrecon_, amgSchwarzVariant_);
      HYPRE_BoomerAMGSetOverlap(HYPrecon_, amgSchwarzOverlap_);
      HYPRE_BoomerAMGSetDomainType(HYPrecon_, amgSchwarzDomainType_);
   }

   if ( amgUseGSMG_ == 1 )
   {
      HYPRE_BoomerAMGSetGSMG(HYPrecon_, 4);
      HYPRE_BoomerAMGSetNumSamples(HYPrecon_,amgGSMGNSamples_);
   }
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
   {
      printf("***************************************************\n");
      printf("* Boomeramg (AMG) solver \n");
      printf("* coarsen type          = %d\n", amgCoarsenType_);
      printf("* measure type          = %d\n", amgMeasureType_);
      printf("* threshold             = %e\n", amgStrongThreshold_);
      printf("* numsweeps             = %d\n", amgNumSweeps_[0]);
      printf("* relax type            = %d\n", amgRelaxType_[0]);
      printf("* relax weight          = %e\n", amgRelaxWeight_[0]);
      printf("* maximum iterations    = %d\n", maxIterations_);
      printf("* smooth type  = %d\n", amgSmoothType_);
      printf("* smooth numlevels  = %d\n", amgSmoothNumLevels_);
      printf("* smooth numsweeps  = %d\n", amgSmoothNumSweeps_);
      printf("* Schwarz variant = %d\n", amgSchwarzVariant_);
      printf("* Schwarz overlap = %d\n", amgSchwarzOverlap_);
      printf("* Schwarz domain type = %d\n", amgSchwarzDomainType_);
      printf("* Schwarz relax weight = %e\n", amgSchwarzRelaxWt_);
      printf("* convergence tolerance = %e\n", tolerance_);
      printf("*--------------------------------------------------\n");
   }
   if ( HYOutputLevel_ & HYFEI_AMGDEBUG )
   {
      HYPRE_BoomerAMGSetDebugFlag(HYSolver_, 0);
      HYPRE_BoomerAMGSetPrintLevel(HYSolver_, 1);
   }
   HYPRE_BoomerAMGSetMaxIter(HYSolver_, maxIterations_);
   HYPRE_BoomerAMGSetMeasureType(HYSolver_, 0);
   HYPRE_BoomerAMGSetup( HYSolver_, A_csr, b_csr, x_csr );

   //-------------------------------------------------------------------
   // BoomerAMG solve
   //-------------------------------------------------------------------

   HYPRE_BoomerAMGSolve( HYSolver_, A_csr, b_csr, x_csr );

   status = 0;
}

//***************************************************************************
// this function solve the incoming linear system using SuperLU
//---------------------------------------------------------------------------

double HYPRE_LinSysCore::solveUsingSuperLU(int& status)
{
  double             rnorm=-1.0;
#ifdef HAVE_SUPERLU
   int                i, nnz, nrows, ierr;
   int                rowSize, *colInd, *new_ia, *new_ja, *ind_array;
   int                nz_ptr, *partition, start_row, end_row;
   double             *colVal, *new_a;
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_ParVector    r_csr;
   HYPRE_ParVector    b_csr;
   HYPRE_ParVector    x_csr;

   int                info=0, permc_spec;
   int                *perm_r, *perm_c;
   double             *rhs, *soln;
   superlu_options_t  slu_options;
   SuperLUStat_t      slu_stat;
   SuperMatrix        A2, B, L, U;
   NRformat           *Ustore;
   SCformat           *Lstore;

   //-------------------------------------------------------------------
   // available for sequential processing only for now
   //-------------------------------------------------------------------

   if ( numProcs_ > 1 )
   {
      printf("solveUsingSuperLU ERROR - too many processors.\n");
      status = -1;
      return rnorm;
   }

   //-------------------------------------------------------------------
   // need to construct a CSR matrix, and the column indices should
   // have been stored in colIndices and rowLengths
   //-------------------------------------------------------------------

   if ( localStartRow_ != 1 )
   {
      printf("solveUsingSuperLU ERROR - row does not start at 1\n");
      status = -1;
      return rnorm;
   }

   //-------------------------------------------------------------------
   // get information about the current matrix
   //-------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &partition );
   start_row = partition[0];
   end_row   = partition[1] - 1;
   nrows     = end_row - start_row + 1;
   free( partition );

   nnz = 0;
   for ( i = start_row; i <= end_row; i++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
      nnz += rowSize;
      HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
   }

   new_ia = new int[nrows+1];
   new_ja = new int[nnz];
   new_a  = new double[nnz];
   nz_ptr = HYPRE_LSI_GetParCSRMatrix(currA_,nrows,nnz,new_ia,new_ja,new_a);
   nnz    = nz_ptr;

   //-------------------------------------------------------------------
   // set up SuperLU CSR matrix and the corresponding rhs
   //-------------------------------------------------------------------

   dCreate_CompRow_Matrix(&A2,nrows,nrows,nnz,new_a,new_ja,new_ia,
                          SLU_NR,SLU_D,SLU_GE);
   ind_array = new int[nrows];
   for ( i = 0; i < nrows; i++ ) ind_array[i] = i;
   rhs = new double[nrows];

   ierr = HYPRE_IJVectorGetValues(currB_, nrows, ind_array, rhs);

   hypre_assert(!ierr);
   dCreate_Dense_Matrix(&B, nrows, 1, rhs, nrows, SLU_DN, SLU_D, SLU_GE);

   //-------------------------------------------------------------------
   // set up the rest and solve (permc_spec=0 : natural ordering)
   //-------------------------------------------------------------------

   perm_r = new int[nrows];
   perm_c = new int[nrows];
   permc_spec = superluOrdering_;
   get_perm_c(permc_spec, &A2, perm_c);
   for ( i = 0; i < nrows; i++ ) perm_r[i] = 0;

   set_default_options(&slu_options);
   slu_options.ColPerm = MY_PERMC;
   slu_options.Fact = DOFACT;
   StatInit(&slu_stat);
   dgssv(&slu_options, &A2, perm_c, perm_r, &L, &U, &B, &slu_stat, &info);

   //-------------------------------------------------------------------
   // postprocessing of the return status information
   //-------------------------------------------------------------------

   if ( info == 0 )
   {
      status = 1;
      Lstore = (SCformat *) L.Store;
      Ustore = (NRformat *) U.Store;
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      {
         printf("No of nonzeros in factor L = %d\n", Lstore->nnz);
         printf("No of nonzeros in factor U = %d\n", Ustore->nnz);
         printf("SuperLU : NNZ in L+U = %d\n",Lstore->nnz+Ustore->nnz-nrows);
      }
   }
   else
   {
      status = 0;
      printf("HYPRE_LinSysCore::solveUsingSuperLU - dgssv error = %d\n",info);
   }

   //-------------------------------------------------------------------
   // fetch the solution and find residual norm
   //-------------------------------------------------------------------

   if ( info == 0 )
   {
      soln = (double *) ((DNformat *) B.Store)->nzval;
      ierr = HYPRE_IJVectorSetValues(currX_, nrows, (const int *) ind_array,
                   	       (const double *) soln);
      hypre_assert(!ierr);

      HYPRE_IJVectorGetObject(currX_, (void **) &x_csr);
      HYPRE_IJVectorGetObject(currB_, (void **) &b_csr);
      HYPRE_IJVectorGetObject(currR_, (void **) &r_csr);

      ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
      hypre_assert(!ierr);
      HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
      ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
      hypre_assert(!ierr);
      rnorm = sqrt( rnorm );
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
         printf("HYPRE_LSC::solveUsingSuperLU - FINAL NORM = %e.\n",rnorm);
   }

   //-------------------------------------------------------------------
   // clean up
   //-------------------------------------------------------------------

   delete [] ind_array;
   delete [] rhs;
   delete [] perm_c;
   delete [] perm_r;
   delete [] new_ia;
   delete [] new_ja;
   delete [] new_a;
   Destroy_SuperMatrix_Store(&B);
   Destroy_SuperNode_Matrix(&L);
   SUPERLU_FREE( A2.Store );
   SUPERLU_FREE( ((NRformat *) U.Store)->colind);
   SUPERLU_FREE( ((NRformat *) U.Store)->rowptr);
   SUPERLU_FREE( ((NRformat *) U.Store)->nzval);
   SUPERLU_FREE( U.Store );
   StatFree(&slu_stat);
#else
   status = -1;
   printf("HYPRE_LSC::solveUsingSuperLU : not available.\n");
#endif
   return rnorm;
}

//***************************************************************************
// this function solve the incoming linear system using SuperLU
// using expert mode
//---------------------------------------------------------------------------

double HYPRE_LinSysCore::solveUsingSuperLUX(int& status)
{
   double             rnorm=-1.0;
#ifdef HAVE_SUPERLU
   int                i, nnz, nrows, ierr;
   int                rowSize, *colInd, *new_ia, *new_ja, *ind_array;
   int                nz_ptr;
   int                *partition, start_row, end_row;
   double             *colVal, *new_a;
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_ParVector    r_csr;
   HYPRE_ParVector    b_csr;
   HYPRE_ParVector    x_csr;

   int                info, permc_spec;
   int                *perm_r, *perm_c, *etree, lwork;
   double             *rhs, *soln, *sol2;
   double             *R, *C;
   double             *ferr, *berr;
   double             rpg, rcond;
   void               *work=NULL;
   char               equed[1];
   GlobalLU_t         Glu;
   mem_usage_t        mem_usage;
   superlu_options_t  slu_options;
   SuperLUStat_t      slu_stat;
   SuperMatrix        A2, B, X, L, U;
   NRformat           *Ustore;
   SCformat           *Lstore;

   //-------------------------------------------------------------------
   // available for sequential processing only for now
   //-------------------------------------------------------------------

   if ( numProcs_ > 1 )
   {
      printf("solveUsingSuperLUX ERROR - too many processors.\n");
      status = -1;
      return rnorm;
   }

   //-------------------------------------------------------------------
   // need to construct a CSR matrix, and the column indices should
   // have been stored in colIndices and rowLengths
   //-------------------------------------------------------------------

   if ( localStartRow_ != 1 )
   {
      printf("solveUsingSuperLUX ERROR - row not start at 1\n");
      status = -1;
      return rnorm;
   }

   //-------------------------------------------------------------------
   // get information about the current matrix
   //-------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(currA_, (void**) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &partition );
   start_row = partition[0];
   end_row   = partition[1] - 1;
   nrows     = end_row - start_row + 1;
   free( partition );

   nnz = 0;
   for ( i = 0; i < nrows; i++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
      nnz += rowSize;
      HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
   }

   new_ia = new int[nrows+1];
   new_ja = new int[nnz];
   new_a  = new double[nnz];
   nz_ptr = HYPRE_LSI_GetParCSRMatrix(currA_,nrows,nnz,new_ia,new_ja,new_a);
   nnz    = nz_ptr;

   //-------------------------------------------------------------------
   // set up SuperLU CSR matrix and the corresponding rhs
   //-------------------------------------------------------------------

   dCreate_CompRow_Matrix(&A2,nrows,nrows,nnz,new_a,new_ja,new_ia,SLU_NR,
                          SLU_D,SLU_GE);
   ind_array = new int[nrows];
   for ( i = 0; i < nrows; i++ ) ind_array[i] = i;

   rhs = new double[nrows];
   ierr = HYPRE_IJVectorGetValues(currB_, nrows, ind_array, rhs);
   hypre_assert(!ierr);
   dCreate_Dense_Matrix(&B, nrows, 1, rhs, nrows, SLU_DN, SLU_D, SLU_GE);

   soln = new double[nrows];
   for ( i = 0; i < nrows; i++ ) soln[i] = 0.0;
   dCreate_Dense_Matrix(&X, nrows, 1, soln, nrows, SLU_DN, SLU_D, SLU_GE);

   //-------------------------------------------------------------------
   // set up the other parameters (permc_spec=0 : natural ordering)
   //-------------------------------------------------------------------

   perm_r = new int[nrows];
   for ( i = 0; i < nrows; i++ ) perm_r[i] = 0;
   perm_c = new int[nrows];
   etree  = new int[nrows];
   permc_spec = superluOrdering_;
   get_perm_c(permc_spec, &A2, perm_c);
   lwork                    = 0;
   set_default_options(&slu_options);
   slu_options.ColPerm      = MY_PERMC;
   slu_options.Equil        = YES;
   slu_options.Trans        = NOTRANS;
   slu_options.Fact         = DOFACT;
   slu_options.IterRefine   = SLU_DOUBLE;
   slu_options.DiagPivotThresh = 1.0;
   slu_options.PivotGrowth = YES;
   slu_options.ConditionNumber = YES;

   StatInit(&slu_stat);
   *equed = 'N';
   R    = (double *) SUPERLU_MALLOC(A2.nrow * sizeof(double));
   C    = (double *) SUPERLU_MALLOC(A2.ncol * sizeof(double));
   ferr = (double *) SUPERLU_MALLOC(sizeof(double));
   berr = (double *) SUPERLU_MALLOC(sizeof(double));

   //-------------------------------------------------------------------
   // solve
   //-------------------------------------------------------------------

//   dgssvx(&slu_options, &A2, perm_c, perm_r, etree,
//          equed, R, C, &L, &U, work, lwork, &B, &X,
//          &rpg, &rcond, ferr, berr, &mem_usage, &slu_stat, &info);
   dgssvx(&slu_options, &A2, perm_c, perm_r, etree,
          equed, R, C, &L, &U, work, lwork, &B, &X,
          &rpg, &rcond, ferr, berr, &Glu, &mem_usage, &slu_stat, &info);

   //-------------------------------------------------------------------
   // print SuperLU internal information at the first step
   //-------------------------------------------------------------------

   if ( info == 0 || info == nrows+1 )
   {
      status = 1;
      Lstore = (SCformat *) L.Store;
      Ustore = (NRformat *) U.Store;
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      {
         printf("Recip. pivot growth = %e\n", rpg);
         printf("%8s%16s%16s\n", "rhs", "FERR", "BERR");
         printf("%8d%16e%16e\n", 1, ferr[0], berr[0]);
         if ( rcond != 0.0 )
            printf("   SuperLU : condition number = %e\n", 1.0/rcond);
         else
            printf("   SuperLU : Recip. condition number = %e\n", rcond);
         printf("No of nonzeros in factor L = %d\n", Lstore->nnz);
         printf("No of nonzeros in factor U = %d\n", Ustore->nnz);
         printf("SuperLUX : NNZ in L+U = %d\n", Lstore->nnz+Ustore->nnz-nrows);
      }
   }
   else
   {
      printf("solveUsingSuperLUX - dgssvx error code = %d\n",info);
      status = 0;
   }

   //-------------------------------------------------------------------
   // fetch the solution and find residual norm
   //-------------------------------------------------------------------

   if ( status == 1 )
   {
      sol2 = (double *) ((DNformat *) X.Store)->nzval;

      ierr = HYPRE_IJVectorSetValues(currX_, nrows, (const int *) ind_array,
                   	       (const double *) sol2);
      hypre_assert(!ierr);

      HYPRE_IJVectorGetObject(currX_, (void **) &x_csr);
      HYPRE_IJVectorGetObject(currR_, (void **) &r_csr);
      HYPRE_IJVectorGetObject(currB_, (void **) &b_csr);
      ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
      hypre_assert(!ierr);
      ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
      hypre_assert(!ierr);
      ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
      hypre_assert(!ierr);
      rnorm = sqrt( rnorm );
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
         printf("HYPRE_LSC::solveUsingSuperLUX - FINAL NORM = %e.\n",rnorm);
   }

   //-------------------------------------------------------------------
   // clean up
   //-------------------------------------------------------------------

   delete [] ind_array;
   delete [] perm_c;
   delete [] perm_r;
   delete [] etree;
   delete [] rhs;
   delete [] soln;
   delete [] new_ia;
   delete [] new_ja;
   delete [] new_a;
   Destroy_SuperMatrix_Store(&B);
   Destroy_SuperMatrix_Store(&X);
   Destroy_SuperNode_Matrix(&L);
   SUPERLU_FREE( A2.Store );
   SUPERLU_FREE( ((NRformat *) U.Store)->colind);
   SUPERLU_FREE( ((NRformat *) U.Store)->rowptr);
   SUPERLU_FREE( ((NRformat *) U.Store)->nzval);
   SUPERLU_FREE( U.Store );
   SUPERLU_FREE (R);
   SUPERLU_FREE (C);
   SUPERLU_FREE (ferr);
   SUPERLU_FREE (berr);
   StatFree(&slu_stat);
#else
   status = -1;
   printf("HYPRE_LSC::solveUsingSuperLUX : not available.\n");
#endif
   return rnorm;
}

//***************************************************************************
// this function solve the incoming linear system using distributed SuperLU
//---------------------------------------------------------------------------

double HYPRE_LinSysCore::solveUsingDSuperLU(int& status)
{
   double rnorm=1.0;
#ifdef HYPRE_USING_DSUPERLU
   int                ierr;
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_ParVector    x_csr, b_csr, r_csr;

   HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
   HYPRE_IJVectorGetObject(currX_, (void **) &x_csr);
   HYPRE_IJVectorGetObject(currB_, (void **) &b_csr);
   HYPRE_IJVectorGetObject(currR_, (void **) &r_csr);

   HYPRE_LSI_DSuperLUCreate(comm_, &HYSolver_);
   HYPRE_LSI_DSuperLUSetOutputLevel(HYSolver_, HYOutputLevel_);
   HYPRE_LSI_DSuperLUSetup(HYSolver_, A_csr, b_csr, x_csr);
   HYPRE_LSI_DSuperLUSolve(HYSolver_, A_csr, b_csr, x_csr);
   HYPRE_LSI_DSuperLUDestroy(HYSolver_);
   ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
   hypre_assert(!ierr);
   ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
   hypre_assert(!ierr);
   ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
   hypre_assert(!ierr);
   rnorm = sqrt( rnorm );
   //if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
   //   printf("HYPRE_LSC::solveUsingDSuperLU - FINAL NORM = %e.\n",rnorm);
#endif
   return rnorm;
}

//***************************************************************************
// this function solve the incoming linear system using Y12M
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::solveUsingY12M(int& status)
{
#ifdef Y12M
   int                i, k, nnz, nrows, ierr;
   int                rowSize, *colInd, *ind_array;
   int                j, nz_ptr, *colLengths, count, maxRowSize;
   double             *colVal, rnorm;
   double             upperSum, lowerSum, *accuSoln, *origRhs;
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_ParVector    r_csr;
   HYPRE_ParVector    b_csr;
   HYPRE_ParVector    x_csr;

   int                n, nn, nn1, *rnr, *snr, *ha, iha, iflag[10], ifail;
   double             *pivot, *val, *rhs, aflag[8];

   //-------------------------------------------------------------------
   // available for sequential processing only for now
   //-------------------------------------------------------------------

   if ( numProcs_ > 1 )
   {
      printf("solveUsingY12M ERROR - too many processors.\n");
      status = 0;
      return;
   }

   //-------------------------------------------------------------------
   // need to construct a CSR matrix, and the column indices should
   // have been stored in colIndices and rowLengths
   //-------------------------------------------------------------------

   if ( localStartRow_ != 1 )
   {
      printf("solveUsingY12M ERROR - row does not start at 1.\n");
      status = -1;
      return;
   }
   if (slideReduction_  == 1)
        nrows = localEndRow_ - 2 * nConstraints_;
   else if (slideReduction_  == 2 || slideReduction_ == 3)
        nrows = localEndRow_ - nConstraints_;
   else if (schurReduction_ == 1)
        nrows = localEndRow_ - localStartRow_ + 1 - A21NRows_;
   else nrows = localEndRow_;

   colLengths = new int[nrows];
   for ( i = 0; i < nrows; i++ ) colLengths[i] = 0;

   maxRowSize = 0;
   HYPRE_IJMatrixGetObject(currA_, (void**) &A_csr);

   for ( i = 0; i < nrows; i++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
      maxRowSize = ( rowSize > maxRowSize ) ? rowSize : maxRowSize;
      for ( j = 0; j < rowSize; j++ )
         if ( colVal[j] != 0.0 ) colLengths[colInd[j]]++;
      HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
   }
   nnz   = 0;
   for ( i = 0; i < nrows; i++ ) nnz += colLengths[i];

   nn     = 2 * nnz;
   nn1    = 2 * nnz;
   snr    = new int[nn];
   rnr    = new int[nn1];
   val    = new double[nn];
   pivot  = new double[nrows];
   iha    = nrows;
   ha     = new int[iha*11];

   nz_ptr = 0;
   for ( i = 0; i < nrows; i++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
      for ( j = 0; j < rowSize; j++ )
      {
         if ( colVal[j] != 0.0 )
         {
            rnr[nz_ptr] = i + 1;
            snr[nz_ptr] = colInd[j] + 1;
            val[nz_ptr] = colVal[j];
            nz_ptr++;
         }
      }
      HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
   }

   nnz = nz_ptr;

   //-------------------------------------------------------------------
   // set up other parameters and the right hand side
   //-------------------------------------------------------------------

   aflag[0] = 16.0;
   aflag[1] = 0.0;
   aflag[2] = 1.0e8;
   aflag[3] = 1.0e-12;
   iflag[0] = 1;
   iflag[1] = 3;
   iflag[2] = 1;
   iflag[3] = 0;
   iflag[4] = 2;
   ind_array = new int[nrows];
   for ( i = 0; i < nrows; i++ ) ind_array[i] = i;
   rhs = new double[nrows];

   ierr = HYPRE_IJVectorGetValues(currB_, nrows, ind_array, rhs);
   hypre_assert(!ierr);

   //-------------------------------------------------------------------
   // call Y12M to solve the linear system
   //-------------------------------------------------------------------

   y12maf_(&nrows,&nnz,val,snr,&nn,rnr,&nn1,pivot,ha,&iha,aflag,iflag,
           rhs,&ifail);
   if ( ifail != 0 && (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
   {
      printf("solveUsingY12M WARNING - ifail = %d\n", ifail);
   }

   //-------------------------------------------------------------------
   // postprocessing
   //-------------------------------------------------------------------

   if ( ifail == 0 )
   {
      ierr = HYPRE_IJVectorSetValues(currX_, nrows, (const int *) &ind_array,
                   	       (const double *) rhs);
      hypre_assert(!ierr);

      HYPRE_IJVectorGetObject(currX_, (void**) &x_csr);
      HYPRE_IJVectorGetObject(currR_, (void**) &r_csr);
      HYPRE_IJVectorGetObject(currB_, (void**) &b_csr);
      ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
      hypre_assert(!ierr);
      ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
      hypre_assert(!ierr);
      ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
      hypre_assert(!ierr);
      rnorm = sqrt( rnorm );
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
         printf("HYPRE_LSC::solveUsingY12M - final norm = %e.\n", rnorm);
   }

   //-------------------------------------------------------------------
   // clean up
   //-------------------------------------------------------------------

   delete [] ind_array;
   delete [] rhs;
   delete [] val;
   delete [] snr;
   delete [] rnr;
   delete [] ha;
   delete [] pivot;
#else
   status = -1;
   printf("HYPRE_LSC::solveUsingY12M - not available.\n");
#endif
}

//***************************************************************************
// this function solve the incoming linear system using Y12M
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::solveUsingAMGe(int &iterations)
{
#ifdef HAVE_AMGE
   int                i, nrows, ierr, *ind_array, status;
   double             rnorm, *rhs, *sol;
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_ParVector    r_csr;
   HYPRE_ParVector    b_csr;
   HYPRE_ParVector    x_csr;

   //-------------------------------------------------------------------
   // available for sequential processing only for now
   //-------------------------------------------------------------------

   if ( numProcs_ > 1 )
   {
      printf("solveUsingAMGE ERROR - too many processors.\n");
      iterations = 0;
      return;
   }

   //-------------------------------------------------------------------
   // need to construct a CSR matrix, and the column indices should
   // have been stored in colIndices and rowLengths
   //-------------------------------------------------------------------

   if ( localStartRow_ != 1 )
   {
      printf("solveUsingAMGe ERROR - row does not start at 1.\n");
      status = -1;
      return;
   }
   if (slideReduction_  == 1)
        nrows = localEndRow_ - 2 * nConstraints_;
   else if (slideReduction_  == 2 || slideReduction_ == 3)
        nrows = localEndRow_ - nConstraints_;
   else if (schurReduction_ == 1)
        nrows = localEndRow_ - localStartRow_ + 1 - A21NRows_;
   else nrows = localEndRow_;

   //-------------------------------------------------------------------
   // set up the right hand side
   //-------------------------------------------------------------------

   ind_array = new int[nrows];
   for ( i = 0; i < nrows; i++ ) ind_array[i] = i;
   rhs = new double[nrows];

   ierr = HYPRE_IJVectorGetValues(currB_, nrows, ind_array, rhs);
   hypre_assert(!ierr);

   //-------------------------------------------------------------------
   // call Y12M to solve the linear system
   //-------------------------------------------------------------------

   sol = new double[nrows];
   status = HYPRE_LSI_AMGeSolve( rhs, sol );

   //-------------------------------------------------------------------
   // postprocessing
   //-------------------------------------------------------------------

   ierr = HYPRE_IJVectorSetValues(currX_, nrows, (const int *) &ind_array,
                                  (const double *) sol);
   hypre_assert(!ierr);

   HYPRE_IJVectorGetObject(currX_, (void**) &x_csr);
   HYPRE_IJVectorGetObject(currR_, (void**) &r_csr);
   HYPRE_IJVectorGetObject(currB_, (void**) &b_csr);

   ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
   hypre_assert(!ierr);
   ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
   hypre_assert(!ierr);
   ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
   hypre_assert(!ierr);
   rnorm = sqrt( rnorm );
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
      printf("HYPRE_LSC::solveUsingAMGe - final norm = %e.\n", rnorm);

   //-------------------------------------------------------------------
   // clean up
   //-------------------------------------------------------------------

   delete [] ind_array;
   delete [] rhs;
   delete [] sol;
#else
   iterations = 0;
   printf("HYPRE_LSC::solveUsingAMGe - not available.\n");
#endif
}

//***************************************************************************
// this function loads in the constraint numbers for reduction
// (to activate automatic slave search, constrList should be NULL)
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::loadConstraintNumbers(int nConstr, int *constrList)
{
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::loadConstraintNumbers - size = %d\n",
                    mypid_, nConstr);
   nConstraints_ = nConstr;
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  loadConstraintNumbers\n", mypid_);
}

//***************************************************************************
// this function extracts the the version number from HYPRE
//---------------------------------------------------------------------------

char *HYPRE_LinSysCore::getVersion()
{
   static char extVersion[100];
   char        hypre[200], hypreVersion[50], ctmp[50];
   sprintf(hypre, "%s", HYPRE_VERSION);
   sscanf(hypre, "%s %s", ctmp, hypreVersion);
   sprintf(extVersion, "%s-%s", HYPRE_FEI_Version(), hypreVersion);
   return extVersion;
}

//***************************************************************************
// create a node to equation map from the solution vector
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::beginCreateMapFromSoln()
{
   mapFromSolnFlag_    = 1;
   mapFromSolnLengMax_ = 10;
   mapFromSolnLeng_    = 0;
   mapFromSolnList_    = new int[mapFromSolnLengMax_];
   mapFromSolnList2_   = new int[mapFromSolnLengMax_];
   return;
}

//***************************************************************************
// create a node to equation map from the solution vector
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::endCreateMapFromSoln()
{
   int    i, *iarray;
   double *darray;

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::entering endCreateMapFromSoln.\n",mypid_);

   mapFromSolnFlag_ = 0;
   if ( mapFromSolnLeng_ > 0 )
      darray = new double[mapFromSolnLeng_];
   for ( i = 0; i < mapFromSolnLeng_; i++ )
      darray[i] = (double) mapFromSolnList_[i];

   hypre_qsort1(mapFromSolnList2_, darray, 0, mapFromSolnLeng_-1);
   iarray = mapFromSolnList2_;
   mapFromSolnList2_ = mapFromSolnList_;
   mapFromSolnList_ = iarray;
   for ( i = 0; i < mapFromSolnLeng_; i++ )
      mapFromSolnList2_[i] = (int) darray[i];
   delete [] darray;

   for ( i = 0; i < mapFromSolnLeng_; i++ )
      printf("HYPRE_LSC::mapFromSoln %d = %d\n",mapFromSolnList_[i],
             mapFromSolnList2_[i]);

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving  endCreateMapFromSoln.\n",mypid_);
}

//***************************************************************************
// add extra nonzero entries into the matrix data structure
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::putIntoMappedMatrix(int row, int numValues,
                  const double* values, const int* scatterIndices)
{
   int    i, index, colIndex, localRow, mappedRow, mappedCol, newLeng;
   int    *tempInd, ind2;
   double *tempVal;

   //-------------------------------------------------------------------
   // error checking
   //-------------------------------------------------------------------

   if ( systemAssembled_ == 1 )
   {
      printf("putIntoMappedMatrix ERROR : matrix already assembled\n");
      exit(1);
   }
   if ( (row+1) < localStartRow_ || (row+1) > localEndRow_ )
   {
      printf("putIntoMappedMatrix ERROR : invalid row number %d.\n",row);
      exit(1);
   }
   index = HYPRE_LSI_Search(mapFromSolnList_, row, mapFromSolnLeng_);

   if ( index >= 0 ) mappedRow = mapFromSolnList2_[index];
   else              mappedRow = row;
   localRow = mappedRow - localStartRow_ + 1;

   //-------------------------------------------------------------------
   // load the local matrix
   //-------------------------------------------------------------------

   newLeng = rowLengths_[localRow] + numValues;
   tempInd = new int[newLeng];
   tempVal = new double[newLeng];
   for ( i = 0; i < rowLengths_[localRow]; i++ )
   {
      tempVal[i] = colValues_[localRow][i];
      tempInd[i] = colIndices_[localRow][i];
   }
   delete [] colValues_[localRow];
   delete [] colIndices_[localRow];
   colValues_[localRow] = tempVal;
   colIndices_[localRow] = tempInd;

   index = rowLengths_[localRow];

   for ( i = 0; i < numValues; i++ )
   {
      colIndex = scatterIndices[i];

      ind2 = HYPRE_LSI_Search(mapFromSolnList_,colIndex,mapFromSolnLeng_);
      if ( mapFromSolnList_ != NULL ) mappedCol = mapFromSolnList2_[ind2];
      else                            mappedCol = colIndex;

      ind2 = HYPRE_LSI_Search(colIndices_[localRow],mappedCol+1,index);
      if ( ind2 >= 0 )
      {
         newLeng--;
         colValues_[localRow][ind2] = values[i];
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
            printf("%4d : putIntoMappedMatrix (add) : row, col = %8d %8d %e \n",
                   mypid_, localRow, colIndices_[localRow][ind2]-1,
                   colValues_[localRow][ind2]);
      }
      else
      {
         ind2 = index;
         colIndices_[localRow][index] = mappedCol + 1;
         colValues_[localRow][index++] = values[i];
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
            printf("%4d : putIntoMappedMatrix : row, col = %8d %8d %e \n",
                   mypid_, localRow, colIndices_[localRow][ind2]-1,
                   colValues_[localRow][ind2]);
         hypre_qsort1(colIndices_[localRow],colValues_[localRow],0,index-1);
      }
   }
   rowLengths_[localRow] = newLeng;
}

//***************************************************************************
// project the initial guess into the previous solutions (x + X inv(R) Q^T b)
// Given r and B (a collection of right hand vectors such that A X = B)
//
//          min   || r - B v ||
//           v
//
// = min (trans(r) r - trans(r) B v-trans(v) trans(B) r+trans(v) trans(B) B v)
//
// ==> trans(B) r = trans(B) B v ==> v = inv(trans(B) B) trans(B) r
//
// Once v is computed, x = x + X v
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::computeMinResProjection(HYPRE_ParCSRMatrix A_csr,
                              HYPRE_ParVector x_csr, HYPRE_ParVector b_csr)
{
   int             i;
   double          alpha;
   HYPRE_ParVector r_csr, v_csr, w_csr;

   //-----------------------------------------------------------------------
   // diagnostic message
   //-----------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
       printf("%4d : HYPRE_LSC::entering computeMinResProjection %d\n",mypid_,
             projectCurrSize_);
   if ( projectCurrSize_ == 0 && HYpxs_ == NULL ) return;

   //-----------------------------------------------------------------------
   // compute r = b - A x, save Ax (to w)
   //-----------------------------------------------------------------------

   HYPRE_IJVectorGetObject(currR_, (void **) &r_csr);
   HYPRE_IJVectorGetObject(HYpbs_[projectSize_], (void **) &w_csr);
   HYPRE_ParCSRMatrixMatvec( 1.0, A_csr, x_csr, 0.0, w_csr );
   HYPRE_ParVectorCopy( b_csr, r_csr );
   alpha = -1.0;
   hypre_ParVectorAxpy(alpha,(hypre_ParVector*)w_csr,(hypre_ParVector*)r_csr);

   //-----------------------------------------------------------------------
   // compute x + X v, accumulate offset to b (in w)
   //-----------------------------------------------------------------------

   for ( i = 0; i < projectCurrSize_; i++ )
   {
      HYPRE_IJVectorGetObject(HYpbs_[i], (void **) &v_csr);
      HYPRE_ParVectorInnerProd( r_csr, v_csr, &alpha);
      hypre_ParVectorAxpy(alpha,(hypre_ParVector*)v_csr,
                                (hypre_ParVector*)w_csr);
      HYPRE_IJVectorGetObject(HYpxs_[i], (void **) &v_csr);
      hypre_ParVectorAxpy(alpha,(hypre_ParVector*)v_csr,
                                (hypre_ParVector*)x_csr);
   }

   //-----------------------------------------------------------------------
   // save x and b away (and adjust b)
   //-----------------------------------------------------------------------

   alpha = - 1.0;
   hypre_ParVectorAxpy(alpha,(hypre_ParVector*)w_csr,(hypre_ParVector*)b_csr);

   HYPRE_IJVectorGetObject(HYpxs_[projectSize_], (void **) &v_csr);
   HYPRE_ParVectorCopy( x_csr, v_csr );
   hypre_ParVectorScale(0.0,(hypre_ParVector*)x_csr);

   //-----------------------------------------------------------------------
   // diagnostic message
   //-----------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
      printf("%4d : HYPRE_LSC:: leaving computeMinResProjection n", mypid_);
   return;
}

//***************************************************************************
// add a new pair of (x,b) vectors to the projection space
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::addToMinResProjectionSpace(HYPRE_IJVector xvec,
                                                  HYPRE_IJVector bvec)
{
   int                i, ierr, *partition, start_row, end_row;
   double             alpha;
   HYPRE_ParVector    v_csr, x_csr, xn_csr, b_csr, r_csr, bn_csr;
   HYPRE_ParCSRMatrix A_csr;

   //-----------------------------------------------------------------------
   // diagnostic message
   //-----------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
      printf("%4d : HYPRE_LSC::addToProjectionSpace %d\n",mypid_,
             projectCurrSize_);

   //-----------------------------------------------------------------------
   // fetch the matrix and vectors
   //-----------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
   HYPRE_IJVectorGetObject(xvec, (void **) &x_csr);
   HYPRE_IJVectorGetObject(bvec, (void **) &b_csr);
   HYPRE_IJVectorGetObject(currR_, (void **) &r_csr);

   //-----------------------------------------------------------------------
   // initially, allocate space for B's and X's and R
   //-----------------------------------------------------------------------

   if ( projectCurrSize_ == 0 && HYpbs_ == NULL )
   {
      HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &partition );
      start_row = partition[mypid_];
      end_row   = partition[mypid_+1] - 1;
      free( partition );
      HYpxs_    = new HYPRE_IJVector[projectSize_+1];
      HYpbs_    = new HYPRE_IJVector[projectSize_+1];

      for ( i = 0; i <= projectSize_; i++ )
      {
         ierr = HYPRE_IJVectorCreate(comm_, start_row, end_row, &(HYpbs_[i]));
         ierr = HYPRE_IJVectorSetObjectType(HYpbs_[i], HYPRE_PARCSR);
         ierr = HYPRE_IJVectorInitialize(HYpbs_[i]);
         ierr = HYPRE_IJVectorAssemble(HYpbs_[i]);
         hypre_assert( !ierr );
      }
      for ( i = 0; i <= projectSize_; i++ )
      {
         ierr = HYPRE_IJVectorCreate(comm_, start_row, end_row, &(HYpxs_[i]));
         ierr = HYPRE_IJVectorSetObjectType(HYpxs_[i], HYPRE_PARCSR);
         ierr = HYPRE_IJVectorInitialize(HYpxs_[i]);
         ierr = HYPRE_IJVectorAssemble(HYpxs_[i]);
         hypre_assert(!ierr);
      }
   }

   //-----------------------------------------------------------------------
   // if buffer has been filled, move things up (but for now, restart)
   //-----------------------------------------------------------------------

   if ( projectCurrSize_ >= projectSize_ )
   {
      //projectCurrSize_--;
      //tmpxvec = HYpxs_[0];
      //tmpbvec = HYpbs_[0];
      //for ( i = 0; i < projectCurrSize_; i++ )
      //{
      //   HYpbs_[i] = HYpbs_[i+1];
      //   HYpxs_[i] = HYpxs_[i+1];
      //}
      //HYpxs_[projectCurrSize_] = tmpxvec;
      //HYpbs_[projectCurrSize_] = tmpbvec;
      projectCurrSize_ = 0;
   }

   //-----------------------------------------------------------------------
   // fetch projection vectors
   //-----------------------------------------------------------------------

   HYPRE_IJVectorGetObject(HYpxs_[projectCurrSize_], (void **) &xn_csr);
   HYPRE_IJVectorGetObject(HYpbs_[projectCurrSize_], (void **) &bn_csr);

   //-----------------------------------------------------------------------
   // copy incoming initial guess to buffer
   //-----------------------------------------------------------------------

   HYPRE_ParVectorCopy( x_csr, xn_csr );

   //-----------------------------------------------------------------------
   // compute bn = A * x
   //-----------------------------------------------------------------------

   HYPRE_ParCSRMatrixMatvec( 1.0, A_csr, x_csr, 0.0, bn_csr );
   HYPRE_ParVectorCopy( bn_csr, r_csr );

   //-----------------------------------------------------------------------
   // compute new vectors
   //-----------------------------------------------------------------------

   for ( i = 0; i < projectCurrSize_; i++ )
   {
      HYPRE_IJVectorGetObject(HYpbs_[i], (void **) &v_csr);
      HYPRE_ParVectorInnerProd(r_csr, v_csr, &alpha);
      alpha = - alpha;
      if ( alpha != 0.0 )
      {
         hypre_ParVectorAxpy(alpha,(hypre_ParVector*)v_csr,
                                   (hypre_ParVector*)bn_csr);
         HYPRE_IJVectorGetObject(HYpxs_[i], (void **) &v_csr);
         hypre_ParVectorAxpy(alpha,(hypre_ParVector*)v_csr,
                                   (hypre_ParVector*)xn_csr);
      }
   }
   HYPRE_ParVectorInnerProd( bn_csr, bn_csr, &alpha);
   alpha = sqrt( alpha );
   if ( alpha != 0.0 )
   {
      alpha = 1.0 / alpha;
      hypre_ParVectorScale(alpha,(hypre_ParVector*)bn_csr);
      hypre_ParVectorScale(alpha,(hypre_ParVector*)xn_csr);
      projectCurrSize_++;
   }

   //-----------------------------------------------------------------------
   // update final solution
   //-----------------------------------------------------------------------

   if ( alpha != 0.0 )
   {
      HYPRE_IJVectorGetObject(HYpxs_[projectSize_], (void **) &v_csr);
      hypre_ParVectorAxpy(1.0,(hypre_ParVector*)v_csr,(hypre_ParVector*)x_csr);

      HYPRE_IJVectorGetObject(HYpbs_[projectSize_], (void **) &v_csr);
      hypre_ParVectorAxpy(1.0,(hypre_ParVector*)v_csr,(hypre_ParVector*)b_csr);
   }

   //-----------------------------------------------------------------------
   // diagnostic message
   //-----------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
      printf("%4d : HYPRE_LSC::leaving addToProjectionSpace %d\n",mypid_,
              projectCurrSize_);
}

//***************************************************************************
// project the initial guess into the previous solution space
//
//          min   || trans(x - xbar) A (x - xbar) ||
//
// solutions (phi_i) at previous steps
//
// (1) compute r = b - A * x_0
// (2) compute alpha_i = (r, phi_i) for all previous stored vectors
// (3) x_stored = x_0 + sum (alpha_i * phi_i)
// (4) b_stored = A * x_0 + sum (alpha_i * psi_i)
// (5) b = b - b_stored, x = 0
//
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::computeAConjProjection(HYPRE_ParCSRMatrix A_csr,
                              HYPRE_ParVector x_csr, HYPRE_ParVector b_csr)
{
   int                i;
   double             alpha;
   HYPRE_ParVector    r_csr, v_csr, w_csr;

   //-----------------------------------------------------------------------
   // diagnostic message
   //-----------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
      printf("%4d : HYPRE_LSC::entering computeAConjProjection %d\n",mypid_,
             projectCurrSize_);
   if ( projectCurrSize_ == 0 && HYpxs_ == NULL ) return;

   //-----------------------------------------------------------------------
   // fetch vectors
   //-----------------------------------------------------------------------

   HYPRE_IJVectorGetObject(currR_, (void **) &r_csr);
   HYPRE_IJVectorGetObject(HYpbs_[projectSize_], (void **) &w_csr);

   //-----------------------------------------------------------------------
   // compute r = b - A x_0, save A * x_0 (to w)
   //-----------------------------------------------------------------------

   HYPRE_ParCSRMatrixMatvec( 1.0, A_csr, x_csr, 0.0, w_csr );
   HYPRE_ParVectorCopy( b_csr, r_csr );
   alpha = -1.0;
   hypre_ParVectorAxpy(alpha,(hypre_ParVector*)w_csr,(hypre_ParVector*)r_csr);

   //-----------------------------------------------------------------------
   // compute alpha_i = (phi_i, r)
   // then x = x + alpha_i * phi_i for all i
   // then w = w + alpha_i * psi_i for all i
   //-----------------------------------------------------------------------

   for ( i = 0; i < projectCurrSize_; i++ )
   {
      HYPRE_IJVectorGetObject(HYpxs_[i], (void **) &v_csr);
      HYPRE_ParVectorInnerProd(r_csr,  v_csr, &alpha);
      hypre_ParVectorAxpy(alpha,(hypre_ParVector*)v_csr,
                                (hypre_ParVector*)x_csr);

      HYPRE_IJVectorGetObject(HYpbs_[i], (void **) &v_csr);
      hypre_ParVectorAxpy(alpha,(hypre_ParVector*)v_csr,
                                (hypre_ParVector*)w_csr);
   }

   //-----------------------------------------------------------------------
   // store x away
   //-----------------------------------------------------------------------

   HYPRE_IJVectorGetObject(HYpxs_[projectSize_], (void **) &v_csr);
   HYPRE_ParVectorCopy( x_csr, v_csr );
   hypre_ParVectorScale(0.0,(hypre_ParVector*)x_csr);

   //-----------------------------------------------------------------------
   // compute new residual b = b - w
   //-----------------------------------------------------------------------

   alpha = -1.0;
   hypre_ParVectorAxpy(alpha,(hypre_ParVector*)w_csr,(hypre_ParVector*)b_csr);

   //-----------------------------------------------------------------------
   // diagnostic message
   //-----------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
      printf("%4d : HYPRE_LSC:: leaving computeAConjProjection n", mypid_);
   return;
}

//***************************************************************************
// add x to the projection space
//
// (1) compute alpha_i = (x, psi_i) for all previous stored vectors
// (2) phi_n = x - sum(alpha_i * phi_i)
// (3) phi_n = phi_n / norm(phi_n)_A
// (4) psi_n = A * phi_n
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::addToAConjProjectionSpace(HYPRE_IJVector xvec,
                                                 HYPRE_IJVector bvec)
{
   int                i, ierr, *partition, start_row, end_row;
   double             alpha;
   HYPRE_ParVector    v_csr, x_csr, b_csr, bn_csr, xn_csr;
   //HYPRE_IJVector     tmpxvec;
   HYPRE_ParCSRMatrix A_csr;

   //-----------------------------------------------------------------------
   // diagnostic message
   //-----------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
      printf("%4d : HYPRE_LSC::addToAConjProjectionSpace %d\n",mypid_,
             projectCurrSize_);

   //-----------------------------------------------------------------------
   // fetch the matrix and vectors
   //-----------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
   HYPRE_IJVectorGetObject(xvec, (void **) &x_csr);
   HYPRE_IJVectorGetObject(bvec, (void **) &b_csr);

   //-----------------------------------------------------------------------
   // initially, allocate space for the phi's and psi's
   //-----------------------------------------------------------------------

   if ( projectCurrSize_ == 0 && HYpxs_ == NULL )
   {
      HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &partition );
      start_row = partition[mypid_];
      end_row   = partition[mypid_+1] - 1;
      free( partition );
      HYpxs_    = new HYPRE_IJVector[projectSize_+1];
      HYpbs_    = new HYPRE_IJVector[projectSize_+1];

      for ( i = 0; i <= projectSize_; i++ )
      {
         ierr = HYPRE_IJVectorCreate(comm_, start_row, end_row, &(HYpbs_[i]));
         ierr = HYPRE_IJVectorSetObjectType(HYpbs_[i], HYPRE_PARCSR);
         ierr = HYPRE_IJVectorInitialize(HYpbs_[i]);
         ierr = HYPRE_IJVectorAssemble(HYpbs_[i]);
         hypre_assert( !ierr );
      }
      for ( i = 0; i <= projectSize_; i++ )
      {
         ierr = HYPRE_IJVectorCreate(comm_, start_row, end_row, &(HYpxs_[i]));
         ierr = HYPRE_IJVectorSetObjectType(HYpxs_[i], HYPRE_PARCSR);
         ierr = HYPRE_IJVectorInitialize(HYpxs_[i]);
         ierr = HYPRE_IJVectorAssemble(HYpxs_[i]);
         hypre_assert(!ierr);
      }
   }

   //-----------------------------------------------------------------------
   // if buffer has been filled, move things up (but for now, restart)
   //-----------------------------------------------------------------------

   if ( projectCurrSize_ >= projectSize_ )
   {
      //projectCurrSize_--;
      //tmpxvec = HYpxs_[0];
      //for ( i = 0; i < projectCurrSize_; i++ ) HYpxs_[i] = HYpxs_[i+1];
      //HYpxs_[projectCurrSize_] = tmpxvec;
      projectCurrSize_ = 0;
   }

   //-----------------------------------------------------------------------
   // fetch the projection vectors
   //-----------------------------------------------------------------------

   HYPRE_IJVectorGetObject(HYpxs_[projectCurrSize_], (void **) &xn_csr);
   HYPRE_IJVectorGetObject(HYpbs_[projectCurrSize_], (void **) &bn_csr);

   //-----------------------------------------------------------------------
   // compute the new A-conjugate vector and its A-norm
   //-----------------------------------------------------------------------

   HYPRE_ParVectorCopy( x_csr, xn_csr );
   for ( i = 0; i < projectCurrSize_; i++ )
   {
      HYPRE_IJVectorGetObject(HYpbs_[i], (void **) &v_csr);
      HYPRE_ParVectorInnerProd( x_csr, v_csr, &alpha);
      if ( alpha != 0.0 )
      {
         alpha = - alpha;
         HYPRE_IJVectorGetObject(HYpxs_[i], (void **) &v_csr);
         hypre_ParVectorAxpy(alpha,(hypre_ParVector*)v_csr,
                                   (hypre_ParVector*)xn_csr);
      }
   }
   HYPRE_ParCSRMatrixMatvec( 1.0, A_csr, xn_csr, 0.0, bn_csr );
   HYPRE_ParVectorInnerProd( xn_csr, bn_csr, &alpha);
   if ( alpha != 0.0 )
   {
      alpha = 1.0 / sqrt( alpha );
      hypre_ParVectorScale(alpha,(hypre_ParVector*)xn_csr);
      hypre_ParVectorScale(alpha,(hypre_ParVector*)bn_csr);
      projectCurrSize_++;
   }

   //-----------------------------------------------------------------------
   // update final solution
   //-----------------------------------------------------------------------

   if ( alpha != 0.0 )
   {
      HYPRE_IJVectorGetObject(HYpxs_[projectSize_], (void **) &v_csr);
      hypre_ParVectorAxpy(1.0,(hypre_ParVector*)v_csr,(hypre_ParVector*)x_csr);

      HYPRE_IJVectorGetObject(HYpbs_[projectSize_], (void **) &v_csr);
      hypre_ParVectorAxpy(1.0,(hypre_ParVector*)v_csr,(hypre_ParVector*)b_csr);
   }

   //-----------------------------------------------------------------------
   // diagnostic message
   //-----------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 4 )
      printf("%4d : HYPRE_LSC::leaving addToAConjProjectionSpace %d\n",mypid_,
              projectCurrSize_);
}

//***************************************************************************
//***************************************************************************
// MLI specific functions
//***************************************************************************
//***************************************************************************

//***************************************************************************
// initialize field information
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::FE_initFields(int nFields, int *fieldSizes,
                                     int *fieldIDs)
{
#ifdef HAVE_MLI
   if ( haveFEData_ == 1 && feData_ != NULL )
      HYPRE_LSI_MLIFEDataInitFields(feData_,nFields,fieldSizes,fieldIDs);
#else
   (void) nFields;
   (void) fieldSizes;
   (void) fieldIDs;
#endif
   return;
}

//***************************************************************************
// initialize element block
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::FE_initElemBlock(int nElems, int nNodesPerElem,
                                        int numNodeFields, int *nodeFieldIDs)
{
#ifdef HAVE_MLI
   int status;
   if ( haveFEData_ == 1 && feData_ != NULL )
   {
      status = HYPRE_LSI_MLIFEDataInitElemBlock(feData_, nElems,
                           nNodesPerElem, numNodeFields, nodeFieldIDs);
      if ( status )
      {
         if      (haveFEData_ == 1) HYPRE_LSI_MLIFEDataDestroy(feData_);
         else if (haveFEData_ == 2) HYPRE_LSI_MLISFEIDestroy(feData_);
         feData_ = NULL;
         haveFEData_ = 0;
      }
   }
#else
   (void) nElems;
   (void) nNodesPerElem;
   (void) numNodeFields;
   (void) nodeFieldIDs;
#endif
   return;
}

//***************************************************************************
// initialize element node list
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::FE_initElemNodeList(int elemID, int nNodesPerElem,
                                           int *nodeIDs)
{
#ifdef HAVE_MLI
   if ( haveFEData_ == 1 && feData_ != NULL )
      HYPRE_LSI_MLIFEDataInitElemNodeList(feData_, elemID, nNodesPerElem,
                                          nodeIDs);
#else
   (void) elemID;
   (void) nNodesPerElem;
   (void) nodeIDs;
#endif
   return;
}

//***************************************************************************
// initialize shared nodes
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::FE_initSharedNodes(int nShared, int *sharedIDs,
                                        int *sharedPLengs, int **sharedProcs)
{
#ifdef HAVE_MLI
   if ( haveFEData_ == 1 && feData_ != NULL )
      HYPRE_LSI_MLIFEDataInitSharedNodes(feData_, nShared, sharedIDs,
                                         sharedPLengs, sharedProcs);
#else
   (void) nShared;
   (void) sharedIDs;
   (void) sharedPLengs;
   (void) sharedProcs;
#endif
   return;
}

//***************************************************************************
// initialize complete
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::FE_initComplete()
{
#ifdef HAVE_MLI
   if ( haveFEData_ == 1 && feData_ != NULL )
      HYPRE_LSI_MLIFEDataInitComplete(feData_);
#endif
   return;
}

//***************************************************************************
// load element matrix
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::FE_loadElemMatrix(int elemID, int nNodes,
                         int *elemNodeList, int matDim, double **elemMat)
{
#ifdef HAVE_MLI
   if ( haveFEData_ == 1 && feData_ != NULL )
      HYPRE_LSI_MLIFEDataLoadElemMatrix(feData_, elemID, nNodes, elemNodeList,
                                        matDim, elemMat);
#else
   (void) elemID;
   (void) nNodes;
   (void) elemNodeList;
   (void) matDim;
   (void) elemMat;
#endif
   return;
}

//***************************************************************************
// build nodal coordinates
// (to be used by AMS preconditioner, but it has been replaced with better
//  scheme. So when time is ripe, this function should be deleted.)
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::HYPRE_LSI_BuildNodalCoordinates()
{
   int    localNRows, *procNRows, *iTempArray, iP, iN, iS, *nodeProcMap;
   int    *procList, nSends, *sendProcs, *sendLengs, **iSendBufs;
   int    nRecvs, *recvLengs, *recvProcs, **iRecvBufs, iR, numNodes, eqnInd;
   int    *flags, arrayLeng, coordLength, ierr, procIndex, iD;
   double **dSendBufs, **dRecvBufs, *nCoords, *vecData;
   MPI_Request *mpiRequests;
   MPI_Status  mpiStatus;
   HYPRE_ParVector parVec;

   /* -------------------------------------------------------- */
   /* construct procNRows array                                */
   /* -------------------------------------------------------- */

   localNRows = localEndRow_ - localStartRow_ + 1;
   procNRows = new int[numProcs_+1];
   iTempArray = new int[numProcs_];
   for (iP = 0; iP <= numProcs_; iP++) procNRows[iP] = 0;
   procNRows[mypid_] = localNRows;
   MPI_Allreduce(procNRows,iTempArray,numProcs_,MPI_INT,MPI_SUM,comm_);
   procNRows[0] = 0;
   for (iP = 1; iP <= numProcs_; iP++)
      procNRows[iP] = procNRows[iP-1] + iTempArray[iP-1];

   /* -------------------------------------------------------- */
   /* construct node to processor map                          */
   /* -------------------------------------------------------- */

   nodeProcMap = new int[MLI_NumNodes_];
   for (iN = 0; iN < MLI_NumNodes_; iN++)
   {
      nodeProcMap[iN] = -1;
      if (MLI_EqnNumbers_[iN] < procNRows[mypid_] ||
          MLI_EqnNumbers_[iN] >= procNRows[mypid_+1])
      {
         for (iP = 0; iP < numProcs_; iP++)
            if (MLI_EqnNumbers_[iN] < procNRows[iP]) break;
         nodeProcMap[iN] = iP - 1;
      }
   }

   /* -------------------------------------------------------- */
   /* construct send information                               */
   /* -------------------------------------------------------- */

   procList = new int[numProcs_];
   for (iP = 0; iP < numProcs_; iP++) procList[iP] = 0;
   for (iN = 0; iN < numProcs_; iN++)
      if (nodeProcMap[iN] >= 0) procList[nodeProcMap[iN]]++;
   nSends = 0;
   for (iP = 0; iP < numProcs_; iP++) if (procList[iP] > 0) nSends++;
   if (nSends > 0)
   {
      sendProcs = new int[nSends];
      sendLengs = new int[nSends];
      iSendBufs = new int*[nSends];
      dSendBufs = new double*[nSends];
   }
   nSends = 0;
   for (iP = 0; iP < numProcs_; iP++)
   {
      if (procList[iP] > 0)
      {
         sendLengs[nSends] = procList[iP];
         sendProcs[nSends++] = iP;
      }
   }

   /* -------------------------------------------------------- */
   /* construct recv information                               */
   /* -------------------------------------------------------- */

   for (iP = 0; iP < numProcs_; iP++) procList[iP] = 0;
   for (iP = 0; iP < nSends; iP++) procList[sendProcs[iP]]++;
   MPI_Allreduce(procList,iTempArray,numProcs_,MPI_INT,MPI_SUM,comm_);
   nRecvs = iTempArray[mypid_];
   if (nRecvs > 0)
   {
      recvLengs = new int[nRecvs];
      recvProcs = new int[nRecvs];
      iRecvBufs = new int*[nRecvs];
      dRecvBufs = new double*[nRecvs];
      mpiRequests = new MPI_Request[nRecvs];
   }
   for (iP = 0; iP < nRecvs; iP++)
      MPI_Irecv(&(recvLengs[iP]), 1, MPI_INT, MPI_ANY_SOURCE, 29421,
                comm_, &(mpiRequests[iP]));
   for (iP = 0; iP < nSends; iP++)
      MPI_Send(&(sendLengs[iP]), 1, MPI_INT, sendProcs[iP], 29421, comm_);
   for (iP = 0; iP < nRecvs; iP++)
   {
      MPI_Wait(&(mpiRequests[iP]), &mpiStatus);
      recvProcs[iP] = mpiStatus.MPI_SOURCE;
   }

   /* -------------------------------------------------------- */
   /* communicate equation numbers information                */
   /* -------------------------------------------------------- */

   for (iP = 0; iP < nRecvs; iP++)
   {
      iRecvBufs[iP] = new int[recvLengs[iP]];
      MPI_Irecv(iRecvBufs[iP], recvLengs[iP], MPI_INT, recvProcs[iP],
                29422, comm_, &(mpiRequests[iP]));
   }
   for (iP = 0; iP < nSends; iP++)
   {
      iSendBufs[iP] = new int[sendLengs[iP]];
      sendLengs[iP] = 0;
   }
   for (iN = 0; iN < MLI_NumNodes_; iN++)
   {
      if (nodeProcMap[iN] >= 0)
      {
         procIndex = nodeProcMap[iN];
         for (iP = 0; iP < nSends; iP++)
            if (procIndex == sendProcs[iP]) break;
         iSendBufs[iP][sendLengs[iP]++] = MLI_EqnNumbers_[iN];
      }
   }
   for (iP = 0; iP < nSends; iP++)
   {
      MPI_Send(iSendBufs[iP], sendLengs[iP], MPI_INT, sendProcs[iP],
               29422, comm_);
   }
   for (iP = 0; iP < nRecvs; iP++) MPI_Wait(&(mpiRequests[iP]),&mpiStatus);

   /* -------------------------------------------------------- */
   /* communicate coordinate information                       */
   /* -------------------------------------------------------- */

   for (iP = 0; iP < nRecvs; iP++)
   {
      dRecvBufs[iP] = new double[recvLengs[iP]*MLI_FieldSize_];
      MPI_Irecv(dRecvBufs[iP], recvLengs[iP]*MLI_FieldSize_, MPI_DOUBLE,
                recvProcs[iP], 29425, comm_, &(mpiRequests[iP]));
   }
   for (iP = 0; iP < nSends; iP++)
   {
      dSendBufs[iP] = new double[sendLengs[iP]*MLI_FieldSize_];
      sendLengs[iP] = 0;
   }
   for (iN = 0; iN < MLI_NumNodes_; iN++)
   {
      if (nodeProcMap[iN] >= 0)
      {
         procIndex = nodeProcMap[iN];
         for (iP = 0; iP < nSends; iP++)
            if (procIndex == sendProcs[iP]) break;
         for (iD = 0; iD < MLI_FieldSize_; iD++)
            dSendBufs[iP][sendLengs[iP]++] =
                      MLI_NodalCoord_[iN*MLI_FieldSize_+iD];
      }
   }
   for (iP = 0; iP < nSends; iP++)
   {
      sendLengs[iP] /= MLI_FieldSize_;
      MPI_Send(dSendBufs[iP], sendLengs[iP]*MLI_FieldSize_, MPI_DOUBLE,
               sendProcs[iP], 29425, comm_);
   }
   for (iP = 0; iP < nRecvs; iP++) MPI_Wait(&(mpiRequests[iP]),&mpiStatus);

   /* -------------------------------------------------------- */
   /* check any duplicate coordinate information               */
   /* -------------------------------------------------------- */

   arrayLeng = MLI_NumNodes_;
   for (iP = 0; iP < nRecvs; iP++) arrayLeng += recvLengs[iP];
   flags = new int[arrayLeng];
   for (iN = 0; iN < arrayLeng; iN++) flags[iN] = 0;
   for (iN = 0; iN < MLI_NumNodes_; iN++)
   {
      if (nodeProcMap[iN] < 0)
      {
         eqnInd = (MLI_EqnNumbers_[iN] - procNRows[mypid_]) / MLI_FieldSize_;
         if (eqnInd >= arrayLeng)
         {
            printf("%d : LoadNodalCoordinates - ERROR(1).\n", mypid_);
            exit(1);
         }
         flags[eqnInd] = 1;
      }
   }
   for (iP = 0; iP < nRecvs; iP++)
   {
      for (iR = 0; iR < recvLengs[iP]; iR++)
      {
         eqnInd = (iRecvBufs[iP][iR] - procNRows[mypid_]) / MLI_FieldSize_;
         if (eqnInd >= arrayLeng)
         {
            printf("%d : LoadNodalCoordinates - ERROR(2).\n", mypid_);
            exit(1);
         }
         flags[eqnInd] = 1;
      }
   }
   numNodes = 0;
   for (iN = 0; iN < arrayLeng; iN++)
   {
      if ( flags[iN] == 0 ) break;
      else                  numNodes++;
   }
   delete [] flags;

   /* -------------------------------------------------------- */
   /* set up nodal coordinate information in correct order     */
   /* -------------------------------------------------------- */

   coordLength = MLI_NumNodes_ * MLI_FieldSize_;
   nCoords = new double[coordLength];

   arrayLeng = MLI_NumNodes_ * MLI_FieldSize_;
   for (iN = 0; iN < MLI_NumNodes_; iN++)
   {
      if (nodeProcMap[iN] < 0)
      {
         eqnInd = (MLI_EqnNumbers_[iN] - procNRows[mypid_]) / MLI_FieldSize_;
         if (eqnInd >= 0 && eqnInd < arrayLeng)
            for (iD = 0; iD < MLI_FieldSize_; iD++)
               nCoords[eqnInd*MLI_FieldSize_+iD] =
                      MLI_NodalCoord_[iN*MLI_FieldSize_+iD];
      }
   }
   for (iP = 0; iP < nRecvs; iP++)
   {
      for (iR = 0; iR < recvLengs[iP]; iR++)
      {
         eqnInd = (iRecvBufs[iP][iR] - procNRows[mypid_]) / MLI_FieldSize_;
         if (eqnInd >= 0 && eqnInd < arrayLeng)
            for (iD = 0; iD < MLI_FieldSize_; iD++)
               nCoords[eqnInd*MLI_FieldSize_+iD] =
                     dRecvBufs[iP][iR*MLI_FieldSize_+iD];
      }
   }

   /* -------------------------------------------------------- */
   /* create AMS vectors                                       */
   /* -------------------------------------------------------- */

   localNRows = localEndRow_ - localStartRow_ + 1;
   ierr  = HYPRE_IJVectorCreate(comm_,(localStartRow_-1)/MLI_FieldSize_,
                 localEndRow_/MLI_FieldSize_-1, &amsX_);
   ierr += HYPRE_IJVectorSetObjectType(amsX_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(amsX_);
   ierr += HYPRE_IJVectorAssemble(amsX_);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(amsX_, (void **) &parVec);
   vecData = (double *) hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) parVec));
   for (iN = 0; iN < localNRows/MLI_FieldSize_; iN++)
      vecData[iN] = nCoords[iN*MLI_FieldSize_];
   ierr  = HYPRE_IJVectorCreate(comm_,(localStartRow_-1)/MLI_FieldSize_,
                 localEndRow_/MLI_FieldSize_-1, &amsY_);
   ierr += HYPRE_IJVectorSetObjectType(amsY_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(amsY_);
   ierr += HYPRE_IJVectorAssemble(amsY_);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(amsY_, (void **) &parVec);
   vecData = (double *) hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) parVec));
   for (iN = 0; iN < localNRows/MLI_FieldSize_; iN++)
      vecData[iN] = nCoords[iN*MLI_FieldSize_+1];
   ierr  = HYPRE_IJVectorCreate(comm_,(localStartRow_-1)/MLI_FieldSize_,
                 localEndRow_/MLI_FieldSize_-1, &amsZ_);
   ierr += HYPRE_IJVectorSetObjectType(amsZ_, HYPRE_PARCSR);
   ierr += HYPRE_IJVectorInitialize(amsZ_);
   ierr += HYPRE_IJVectorAssemble(amsZ_);
   hypre_assert(!ierr);
   HYPRE_IJVectorGetObject(amsZ_, (void **) &parVec);
   vecData = (double *) hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *) parVec));
   for (iN = 0; iN < localNRows/MLI_FieldSize_; iN++)
      vecData[iN] = nCoords[iN*MLI_FieldSize_+2];

   /* -------------------------------------------------------- */
   /* clean up                                                 */
   /* -------------------------------------------------------- */

   delete [] procList;
   delete [] iTempArray;
   delete [] nodeProcMap;
   delete [] procNRows;
   delete [] nCoords;
   if (nSends > 0)
   {
      delete [] sendProcs;
      delete [] sendLengs;
      for (iS = 0; iS < nSends; iS++) delete [] iSendBufs[iS];
      for (iS = 0; iS < nSends; iS++) delete [] dSendBufs[iS];
      delete [] dSendBufs;
      delete [] iSendBufs;
   }
   if (nRecvs > 0)
   {
      delete [] recvProcs;
      delete [] recvLengs;
      for (iR = 0; iR < nRecvs; iR++) delete [] iRecvBufs[iR];
      for (iR = 0; iR < nRecvs; iR++) delete [] dRecvBufs[iR];
      delete [] iRecvBufs;
      delete [] dRecvBufs;
      delete [] mpiRequests;
   }
   return;
}

