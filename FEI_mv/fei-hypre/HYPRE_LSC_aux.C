/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

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
#include <iostream.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

//---------------------------------------------------------------------------
// HYPRE include files
//---------------------------------------------------------------------------

#include "HYPRE.h"
#include "utilities/utilities.h"
#include "../../IJ_mv/HYPRE_IJ_mv.h"
#include "../../parcsr_mv/HYPRE_parcsr_mv.h"
#include "../../parcsr_ls/HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_bicgstabl.h"
#include "HYPRE_parcsr_TFQmr.h"
#include "HYPRE_parcsr_bicgs.h"
#include "HYPRE_parcsr_symqmr.h"
#include "HYPRE_parcsr_fgmres.h"
#include "HYPRE_LinSysCore.h"
#include "parcsr_mv/parcsr_mv.h"
#include "HYPRE_LSI_schwarz.h"
#include "HYPRE_LSI_ddilut.h"
#include "HYPRE_LSI_ddict.h"
#include "HYPRE_LSI_poly.h"
#include "HYPRE_LSI_block.h"

//---------------------------------------------------------------------------
// FEI include files
//---------------------------------------------------------------------------

#include "HYPRE_FEI_includes.h"

//---------------------------------------------------------------------------
// SUPERLU include files
//---------------------------------------------------------------------------

#ifdef HAVE_SUPERLU
#include "dsp_defs.h"
#include "util.h"
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
 * other functions   
 *-------------------------------------------------------------------------*/

   void  qsort1(int *, double *, int, int);
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

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
         printf("       HYPRE_LSC::parameters solver = %s\n",HYSolverName_);
   }

   //-------------------------------------------------------------------
   // select preconditioner : diagonal, pilut, boomeramg, parasails
   //-------------------------------------------------------------------

   if ( precon_index >= 0 )
   {
      sscanf(params[precon_index],"%s %s", param, HYPreconName_);
      selectPreconditioner(HYPreconName_);
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
         printf("       HYPRE_LSC::parameters preconditioner = %s\n",
                HYPreconName_);
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

      if ( !strcmp(param1, "help") )
      {
         printf("%4d : HYPRE_LinSysCore::parameters - available ones : \n",
                mypid_);
         printf("    - outputLevel <d> \n");
         printf("    - setDebug <slideReduction1,amgDebug,printFEInfo>\n");
         printf("    - haveFEData <0,1>\n");
         printf("    - schurReduction\n");
         printf("    - slideReduction, slideReduction2, slideReduction3\n");
         printf("    - AConjugateProjection <dsize>\n");
         printf("    - minResProjection <dsize>\n");
         printf("    - solver <cg,gmres,bicgstab,boomeramg,superlux,..>\n");
         printf("    - maxIterations <d>\n");
         printf("    - tolerance <f>\n");
         printf("    - gmresDim <d>\n");
         printf("    - stopCrit <absolute,relative>\n");
         printf("    - preconditioner <identity,diagonal,pilut,parasails,\n");
         printf("    -    boomeramg,ddilut,schwarz,ddict,poly,euclid,...\n");
         printf("    -    blockP,ml,mli,reuse,parasails_reuse> <override>\n");
         printf("    - pilutFillin or pilutRowSize <d>\n");
         printf("    - pilutDropTol <f>\n");
         printf("    - ddilutFillin <f>\n");
         printf("    - ddilutDropTol <f> (f*sparsity(A))\n");
         printf("    - ddilutReorder\n");
         printf("    - ddictFillin <f>\n");
         printf("    - ddictDropTol <f> (f*sparsity(A))\n");
         printf("    - schwarzNBlocks <d>\n");
         printf("    - schwarzBlockSize <d>\n");
         printf("    - polyorder <d>\n");
         printf("    - superluOrdering <natural,mmd>\n");
         printf("    - superluScale <y,n>\n");
         printf("    - amgCoarsenType <cljp,falgout,ruge,ruge3c>\n");
         printf("    - amgMeasureType <global,local>\n");
         printf("    - amgRelaxType <jacobi,gsFast,hybrid,hybridsym>\n");
         printf("    - amgNumSweeps <d>\n");
         printf("    - amgRelaxWeight <f>\n");
         printf("    - amgStrongThreshold <f>\n");
         printf("    - amgSystemSize <d>\n");
         printf("    - amgMaxIterations <d>\n");
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
         printf("    - MLI help (to get MLI options) \n");
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
         if ( olevel > 4 ) olevel = 4;
         HYOutputLevel_ = ( HYOutputLevel_ & HYFEI_HIGHMASK ) + olevel;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters outputLevel = %d\n",
                   HYOutputLevel_);
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
            printf("       HYPRE_LSC::parameters - slide reduction.\n");
      }
      else if ( !strcmp(param1, "slideReduction2") )
      {
         slideReduction_ = 2;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters - slide reduction.\n");
      }
      else if ( !strcmp(param1, "slideReduction3") )
      {
         slideReduction_ = 3;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters - slide reduction.\n");
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
            printf("       HYPRE_LSC::parameters ddictFillin = %d\n",
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
         for ( k = 0; k < 3; k++ ) amgNumSweeps_[k] = nsweeps;
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
         else if ( !strcmp(param2, "gsSlow") )  rtype = 1;
         else if ( !strcmp(param2, "gsFast") )  rtype = 4;
         else if ( !strcmp(param2, "hybrid" ) ) rtype = 3;
         else if ( !strcmp(param2, "hybridsym" ) ) rtype = 6;
         else                                   rtype = 4;
         for ( k = 0; k < 3; k++ ) amgRelaxType_[k] = rtype;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgRelaxType = %s\n",
                   params);
      }

      //---------------------------------------------------------------
      // amg preconditoner : damping factor for Jacobi smoother
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "amgRelaxWeight") )
      {
         sscanf(params[i],"%s %lg", param, &weight);
         if ( weight < 0.0 || weight > 1.0 ) weight = 0.5;
         for ( k = 0; k < 25; k++ ) amgRelaxWeight_[k] = weight;
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 && mypid_ == 0 )
            printf("       HYPRE_LSC::parameters amgRelaxWeight = %e\n",
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
      // parasails preconditoner : threshold ( >= 0.0 )
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "parasailsThreshold") )
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

      //---------------------------------------------------------------
      // Euclid preconditoner : fill-in 
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "euclidNlevels") )
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

      else if ( !strcmp(param1, "MLI") )
      {
#ifdef HAVE_MLI
         if ( HYPreconID_ == HYMLI )
            HYPRE_LSI_MLISetParams(HYPrecon_, params[i]); 
#else
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 && mypid_ == 0 )
            printf("       HYPRE_LSC::MLI SetParams - MLI unavailable.\n");
#endif
      }

      //---------------------------------------------------------------
      // mlpack preconditoner : no of relaxation sweeps per level
      //---------------------------------------------------------------

      else if ( !strcmp(param1, "mlNumPresweeps") )
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
         mlPresmootherType_  = rtype;
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
         else                                mlMethod_ = 1;
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
         else                                        mlCoarseSolver_ = 1;
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
         else                                      mlCoarsenScheme_ = 1;
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

      //---------------------------------------------------------------
      // error 
      //---------------------------------------------------------------

      else
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
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
      printf("%4d : HYPRE_LSC::leaving  parameters function.\n",mypid_);
   return(0);
}

//***************************************************************************
// set up preconditioners for PCG
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupPCGPrecon()
{
   int    i, *num_sweeps, *relax_type;
   double *relax_wt;

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
              printf("HYPRE_LSI : CG does not work with pilut.\n");
           exit(1);
           break;

      case HYDDILUT :
           if ( mypid_ == 0 )
              printf("HYPRE_LSI : CG does not work with ddilut.\n");
           exit(1);
           break;

      case HYDDICT :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDICT - fillin   = %e\n", ddictFillin_);
              printf("DDICT - drop tol = %e\n", ddictDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
              HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                        HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
              HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
              HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                        HYPRE_LSI_SchwarzSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              printf("Polynomial preconditioning - order = %d\n",polyOrder_ );
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                        HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
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
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_ParCSRParaSailsSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRParaSailsSetSym(HYPrecon_,parasailsSym_);
              HYPRE_ParCSRParaSailsSetParams(HYPrecon_, parasailsThreshold_, 
                                             parasailsNlevels_);
              HYPRE_ParCSRParaSailsSetFilter(HYPrecon_, parasailsFilter_);
              HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_, parasailsLoadbal_);
              HYPRE_ParCSRParaSailsSetReuse(HYPrecon_, parasailsReuse_);
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_ParCSRParaSailsSolve,
                                        HYPRE_ParCSRParaSailsSetup, HYPrecon_);
               HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("AMG coarsen type = %d\n", amgCoarsenType_);
              printf("AMG measure type = %d\n", amgMeasureType_);
              printf("AMG threshold    = %e\n", amgStrongThreshold_);
              printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
              printf("AMG relax type   = %d\n", amgRelaxType_[0]);
              printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
              printf("AMG system size  = %d\n", amgSystemSize_);
           }
           if ( HYOutputLevel_ & HYFEI_AMGDEBUG )
           {
              HYPRE_BoomerAMGSetDebugFlag(HYPrecon_, 0);
              HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 3);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( amgSystemSize_ > 1 )
                 HYPRE_BoomerAMGSetNumFunctions(HYPrecon_, amgSystemSize_);
              HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
              HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
              HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_,amgStrongThreshold_);
              num_sweeps = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

              HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
              relax_type = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

              HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
              relax_wt = hypre_CTAlloc(double,25);
              for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
              HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                        HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              for ( i = 0; i < euclidargc_; i++ )
                 printf("Euclid parameter : %s %s\n", euclidargv_[2*i], 
                                                      euclidargv_[2*i+1]);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_EuclidSetParams(HYPrecon_,euclidargc_*2,euclidargv_);
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                       HYPRE_EuclidSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBLOCK :
           printf("CG : block preconditioning not available.\n");
           exit(1);
           break;

#ifdef HAVE_ML
      case HYML :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("ML strong threshold = %e\n", mlStrongThreshold_);
              printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
              printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
              printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
              printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
              printf("ML relax weight     = %e\n", mlRelaxWeight_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                        HYPRE_DummyFunction, HYPrecon_);
           else
           {
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
              HYPRE_ParCSRPCGSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                        HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;
#endif

#ifdef HAVE_MLI
      case HYMLI :
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
           break;
#endif
   }
   return;
}

//***************************************************************************
// set up preconditioners for GMRES
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupGMRESPrecon()
{
   int    i, *num_sweeps, *relax_type;
   double *relax_wt;

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
           {
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                          HYPRE_DummyFunction, HYPrecon_);
           }
           else
           {
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_ParCSRDiagScale,
                                          HYPRE_ParCSRDiagScaleSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPILUT :
           if (pilutFillin_ == 0) pilutFillin_ = pilutMaxNnzPerRow_;
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("PILUT - row size = %d\n", pilutFillin_);
              printf("PILUT - drop tol = %e\n", pilutDropTol_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutFillin_);
              HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_ParCSRPilutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDILUT :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDILUT - fillin   = %e\n", ddilutFillin_);
              printf("DDILUT - drop tol = %e\n", ddilutDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDIlutSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( ddilutReorder_ ) HYPRE_LSI_DDIlutSetReorder(HYPrecon_);
              HYPRE_LSI_DDIlutSetFillin(HYPrecon_,ddilutFillin_);
              HYPRE_LSI_DDIlutSetDropTolerance(HYPrecon_,ddilutDropTol_);
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                          HYPRE_LSI_DDIlutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDICT :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDICT - fillin   = %e\n", ddictFillin_);
              printf("DDICT - drop tol = %e\n", ddictDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
              HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
              HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
              HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_LSI_SchwarzSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Polynomial preconditioning - order = %d\n",polyOrder_);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
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
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRParaSailsSetSym(HYPrecon_, parasailsSym_);
              HYPRE_ParCSRParaSailsSetParams(HYPrecon_, parasailsThreshold_,
                                             parasailsNlevels_);
              HYPRE_ParCSRParaSailsSetFilter(HYPrecon_, parasailsFilter_);
              HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_, parasailsLoadbal_);
              HYPRE_ParCSRParaSailsSetReuse(HYPrecon_, parasailsReuse_);
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_ParCSRParaSailsSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("AMG coarsen type = %d\n", amgCoarsenType_);
              printf("AMG measure type = %d\n", amgMeasureType_);
              printf("AMG threshold    = %e\n", amgStrongThreshold_);
              printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
              printf("AMG relax type   = %d\n", amgRelaxType_[0]);
              printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
              printf("AMG system size  = %d\n", amgSystemSize_);
           }
           if ( HYOutputLevel_ & HYFEI_AMGDEBUG )
           {
              HYPRE_BoomerAMGSetDebugFlag(HYPrecon_, 0);
              HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 3);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( amgSystemSize_ > 1 )
                 HYPRE_BoomerAMGSetNumFunctions(HYPrecon_, amgSystemSize_);
              HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
              HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
              HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_,amgStrongThreshold_);
              num_sweeps = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

              HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
              relax_type = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

              HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
              relax_wt = hypre_CTAlloc(double,25);
              for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
              HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                          HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              for ( i = 0; i < euclidargc_; i++ )
                 printf("Euclid parameter : %s %s\n", euclidargv_[2*i], 
                                                      euclidargv_[2*i+1]);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_EuclidSetParams(HYPrecon_,euclidargc_*2,euclidargv_);
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                          HYPRE_EuclidSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBLOCK :
           printf("GMRES : block preconditioning not available.\n");
           exit(1);
           break;

#ifdef HAVE_ML
      case HYML :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("ML strong threshold = %e\n", mlStrongThreshold_);
              printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
              printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
              printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
              printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
              printf("ML relax weight     = %e\n", mlRelaxWeight_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
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
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                          HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;
#endif

#ifdef HAVE_MLI
      case HYMLI :
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
           break;
#endif
   }
   return;
}

//***************************************************************************
// set up preconditioners for FGMRES
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupFGMRESPrecon()
{
   int    i, *num_sweeps, *relax_type;
   double *relax_wt;
   HYPRE_Lookup *newLookup;

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
           if (pilutFillin_ == 0) pilutFillin_ = pilutMaxNnzPerRow_;
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("PILUT - row size = %d\n", pilutFillin_);
              printf("PILUT - drop tol = %e\n", pilutDropTol_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutFillin_);
              HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                           HYPRE_ParCSRPilutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDILUT :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDILUT - fillin   = %e\n", ddilutFillin_);
              printf("DDILUT - drop tol = %e\n", ddilutDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDIlutSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( ddilutReorder_ ) HYPRE_LSI_DDIlutSetReorder(HYPrecon_);
              HYPRE_LSI_DDIlutSetFillin(HYPrecon_,ddilutFillin_);
              HYPRE_LSI_DDIlutSetDropTolerance(HYPrecon_,ddilutDropTol_);
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                           HYPRE_LSI_DDIlutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDICT :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDICT - fillin   = %e\n", ddictFillin_);
              printf("DDICT - drop tol = %e\n", ddictDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
              HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                           HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRGMRESSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
              HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
              HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                           HYPRE_LSI_SchwarzSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Polynomial preconditioning - order = %d\n",polyOrder_);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                           HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
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
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRParaSailsSetSym(HYPrecon_, parasailsSym_);
              HYPRE_ParCSRParaSailsSetParams(HYPrecon_, parasailsThreshold_,
                                             parasailsNlevels_);
              HYPRE_ParCSRParaSailsSetFilter(HYPrecon_, parasailsFilter_);
              HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_, parasailsLoadbal_);
              HYPRE_ParCSRParaSailsSetReuse(HYPrecon_, parasailsReuse_);
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                         HYPRE_ParCSRParaSailsSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("AMG coarsen type = %d\n", amgCoarsenType_);
              printf("AMG measure type = %d\n", amgMeasureType_);
              printf("AMG threshold    = %e\n", amgStrongThreshold_);
              printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
              printf("AMG relax type   = %d\n", amgRelaxType_[0]);
              printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
              printf("AMG system size  = %d\n", amgSystemSize_);
           }
           if ( HYOutputLevel_ & HYFEI_AMGDEBUG )
           {
              HYPRE_BoomerAMGSetDebugFlag(HYPrecon_, 0);
              HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 3);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( amgSystemSize_ > 1 )
                 HYPRE_BoomerAMGSetNumFunctions(HYPrecon_, amgSystemSize_);
              HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
              HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
              HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_, amgStrongThreshold_);
              num_sweeps = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

              HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
              relax_type = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

              HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
              relax_wt = hypre_CTAlloc(double,25);
              for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
              HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                           HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              for ( i = 0; i < euclidargc_; i++ )
                 printf("Euclid parameter : %s %s\n", euclidargv_[2*i], 
                                                      euclidargv_[2*i+1]);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_EuclidSetParams(HYPrecon_,euclidargc_*2,euclidargv_);
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                           HYPRE_EuclidSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBLOCK :
           newLookup = (HYPRE_Lookup *) malloc(sizeof(HYPRE_Lookup));
           newLookup->object = (void *) lookup_;
           HYPRE_LSI_BlockPrecondSetLookup( HYPrecon_, newLookup );
           free( newLookup );
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, 
                    HYPRE_LSI_BlockPrecondSolve, HYPRE_DummyFunction, 
                    HYPrecon_);
           else
           {
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, 
                                      HYPRE_LSI_BlockPrecondSolve, 
                                      HYPRE_LSI_BlockPrecondSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

#ifdef HAVE_ML
      case HYML :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("ML strong threshold = %e\n", mlStrongThreshold_);
              printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
              printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
              printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
              printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
              printf("ML relax weight     = %e\n", mlRelaxWeight_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
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
              HYPRE_ParCSRFGMRESSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                           HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;
#endif

#ifdef HAVE_MLI
      case HYMLI :
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
           break;
#endif
   }
   return;
}

//***************************************************************************
// set up preconditioners for BiCGSTAB
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupBiCGSTABPrecon()
{
   int    i, *num_sweeps, *relax_type;
   double *relax_wt;

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
           if (pilutFillin_ == 0) pilutFillin_ = pilutMaxNnzPerRow_;
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("PILUT - row size = %d\n", pilutFillin_);
              printf("PILUT - drop tol = %e\n", pilutDropTol_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutFillin_);
              HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                             HYPRE_ParCSRPilutSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDILUT :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDILUT - fillin   = %e\n", ddilutFillin_);
              printf("DDILUT - drop tol = %e\n", ddilutDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDIlutSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( ddilutReorder_ ) HYPRE_LSI_DDIlutSetReorder(HYPrecon_);
              HYPRE_LSI_DDIlutSetFillin(HYPrecon_,ddilutFillin_);
              HYPRE_LSI_DDIlutSetDropTolerance(HYPrecon_,ddilutDropTol_);
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                             HYPRE_LSI_DDIlutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDICT :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDICT - fillin   = %e\n", ddictFillin_);
              printf("DDICT - drop tol = %e\n", ddictDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
              HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                             HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
              HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
              HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                             HYPRE_LSI_SchwarzSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Polynomial preconditioning - order = %d\n",polyOrder_);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                             HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
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
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, 
                                             HYPRE_ParCSRParaSailsSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRParaSailsSetSym(HYPrecon_, parasailsSym_);
              HYPRE_ParCSRParaSailsSetParams(HYPrecon_, parasailsThreshold_,
                                             parasailsNlevels_);
              HYPRE_ParCSRParaSailsSetFilter(HYPrecon_, parasailsFilter_);
              HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_, parasailsLoadbal_);
              HYPRE_ParCSRParaSailsSetReuse(HYPrecon_, parasailsReuse_);
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,
                                             HYPRE_ParCSRParaSailsSolve,
                                             HYPRE_ParCSRParaSailsSetup, 
                                             HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("AMG coarsen type = %d\n", amgCoarsenType_);
              printf("AMG measure type = %d\n", amgMeasureType_);
              printf("AMG threshold    = %e\n", amgStrongThreshold_);
              printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
              printf("AMG relax type   = %d\n", amgRelaxType_[0]);
              printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
              printf("AMG system size  = %d\n", amgSystemSize_);
           }
           if ( HYOutputLevel_ & HYFEI_AMGDEBUG )
           {
              HYPRE_BoomerAMGSetDebugFlag(HYPrecon_, 0);
              HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 3);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( amgSystemSize_ > 1 )
                 HYPRE_BoomerAMGSetNumFunctions(HYPrecon_, amgSystemSize_);
              HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
              HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
              HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_,amgStrongThreshold_);
              num_sweeps = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

              HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
              relax_type = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

              HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
              relax_wt = hypre_CTAlloc(double,25);
              for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
              HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                             HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              for ( i = 0; i < euclidargc_; i++ )
                 printf("Euclid parameter : %s %s\n", euclidargv_[2*i], 
                                                      euclidargv_[2*i+1]);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_EuclidSetParams(HYPrecon_,euclidargc_*2,euclidargv_);
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                             HYPRE_EuclidSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBLOCK :
           printf("BiCGSTAB : block preconditioning not available.\n");
           exit(1);
           break;

#ifdef HAVE_ML
      case HYML :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("ML strong threshold = %e\n", mlStrongThreshold_);
              printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
              printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
              printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
              printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
              printf("ML relax weight     = %e\n", mlRelaxWeight_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                             HYPRE_DummyFunction, HYPrecon_);
           else
           {
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
              HYPRE_ParCSRBiCGSTABSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                             HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;
#endif

#ifdef HAVE_MLI
      case HYMLI :
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
           break;
#endif
   }
   return;
}

//***************************************************************************
// set up preconditioners for BiCGSTABL
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupBiCGSTABLPrecon()
{
   int    i, *num_sweeps, *relax_type;
   double *relax_wt;

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
           if (pilutFillin_ == 0) pilutFillin_ = pilutMaxNnzPerRow_;
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("PILUT - row size = %d\n", pilutFillin_);
              printf("PILUT - drop tol = %e\n", pilutDropTol_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_ParCSRPilutSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutFillin_);
              HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_ParCSRPilutSolve,
                                            HYPRE_ParCSRPilutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDILUT :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDILUT - fillin   = %e\n", ddilutFillin_);
              printf("DDILUT - drop tol = %e\n", ddilutDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDIlutSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( ddilutReorder_ ) HYPRE_LSI_DDIlutSetReorder(HYPrecon_);
              HYPRE_LSI_DDIlutSetFillin(HYPrecon_,ddilutFillin_);
              HYPRE_LSI_DDIlutSetDropTolerance(HYPrecon_,ddilutDropTol_);
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                              HYPRE_LSI_DDIlutSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDICT :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDICT - fillin   = %e\n", ddictFillin_);
              printf("DDICT - drop tol = %e\n", ddictDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
              HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                              HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_LSI_SchwarzSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
              HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
              HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_LSI_SchwarzSolve,
                                             HYPRE_LSI_SchwarzSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Polynomial preconditioning - order = %d\n",polyOrder_);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                              HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
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
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, 
                                              HYPRE_ParCSRParaSailsSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRParaSailsSetSym(HYPrecon_, parasailsSym_);
              HYPRE_ParCSRParaSailsSetParams(HYPrecon_, parasailsThreshold_,
                                             parasailsNlevels_);
              HYPRE_ParCSRParaSailsSetFilter(HYPrecon_, parasailsFilter_);
              HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_, parasailsLoadbal_);
              HYPRE_ParCSRParaSailsSetReuse(HYPrecon_, parasailsReuse_);
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, 
                                              HYPRE_ParCSRParaSailsSolve,
                                              HYPRE_ParCSRParaSailsSetup, 
                                              HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("AMG coarsen type = %d\n", amgCoarsenType_);
              printf("AMG measure type = %d\n", amgMeasureType_);
              printf("AMG threshold    = %e\n", amgStrongThreshold_);
              printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
              printf("AMG relax type   = %d\n", amgRelaxType_[0]);
              printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
              printf("AMG system size  = %d\n", amgSystemSize_);
           }
           if ( HYOutputLevel_ & HYFEI_AMGDEBUG )
           {
              HYPRE_BoomerAMGSetDebugFlag(HYPrecon_, 0);
              HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 3);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( amgSystemSize_ > 1 )
                 HYPRE_BoomerAMGSetNumFunctions(HYPrecon_, amgSystemSize_);
              HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
              HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
              HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_,amgStrongThreshold_);
              num_sweeps = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

              HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
              relax_type = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

              HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
              relax_wt = hypre_CTAlloc(double,25);
              for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
              HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                              HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              for ( i = 0; i < euclidargc_; i++ )
                 printf("Euclid parameter : %s %s\n", euclidargv_[2*i], 
                                                      euclidargv_[2*i+1]);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_EuclidSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_EuclidSetParams(HYPrecon_,euclidargc_*2,euclidargv_);
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

#ifdef HAVE_ML
      case HYML :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("ML strong threshold = %e\n", mlStrongThreshold_);
              printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
              printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
              printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
              printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
              printf("ML relax weight     = %e\n", mlRelaxWeight_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                              HYPRE_DummyFunction, HYPrecon_);
           else
           {
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
              HYPRE_ParCSRBiCGSTABLSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                              HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;
#endif

#ifdef HAVE_MLI
      case HYMLI :
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
           break;
#endif
   }
   return;
}

//***************************************************************************
// set up preconditioners for TFQMR
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupTFQmrPrecon()
{
   int    i, *num_sweeps, *relax_type;
   double *relax_wt;

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
           if (pilutFillin_ == 0) pilutFillin_ = pilutMaxNnzPerRow_;
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("PILUT - row size = %d\n", pilutFillin_);
              printf("PILUT - drop tol = %e\n", pilutDropTol_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutFillin_);
              HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_ParCSRPilutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDILUT :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDILUT - fillin   = %e\n", ddilutFillin_);
              printf("DDILUT - drop tol = %e\n", ddilutDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDIlutSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( ddilutReorder_ ) HYPRE_LSI_DDIlutSetReorder(HYPrecon_);
              HYPRE_LSI_DDIlutSetFillin(HYPrecon_,ddilutFillin_);
              HYPRE_LSI_DDIlutSetDropTolerance(HYPrecon_,ddilutDropTol_);
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                          HYPRE_LSI_DDIlutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDICT :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDICT - fillin   = %e\n", ddictFillin_);
              printf("DDICT - drop tol = %e\n", ddictDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
              HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
              HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
              HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_LSI_SchwarzSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Polynomial preconditioning - order = %d\n",polyOrder_);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
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
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRParaSailsSetSym(HYPrecon_, parasailsSym_);
              HYPRE_ParCSRParaSailsSetParams(HYPrecon_, parasailsThreshold_,
                                             parasailsNlevels_);
              HYPRE_ParCSRParaSailsSetFilter(HYPrecon_, parasailsFilter_);
              HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_, parasailsLoadbal_);
              HYPRE_ParCSRParaSailsSetReuse(HYPrecon_, parasailsReuse_);
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_ParCSRParaSailsSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("AMG coarsen type = %d\n", amgCoarsenType_);
              printf("AMG measure type = %d\n", amgMeasureType_);
              printf("AMG threshold    = %e\n", amgStrongThreshold_);
              printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
              printf("AMG relax type   = %d\n", amgRelaxType_[0]);
              printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
              printf("AMG system size  = %d\n", amgSystemSize_);
           }
           if ( HYOutputLevel_ & HYFEI_AMGDEBUG )
           {
              HYPRE_BoomerAMGSetDebugFlag(HYPrecon_, 0);
              HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 3);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( amgSystemSize_ > 1 )
                 HYPRE_BoomerAMGSetNumFunctions(HYPrecon_, amgSystemSize_);
              HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
              HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
              HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_, amgStrongThreshold_);
              num_sweeps = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

              HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
              relax_type = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

              HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
              relax_wt = hypre_CTAlloc(double,25);
              for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
              HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                          HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              for ( i = 0; i < euclidargc_; i++ )
                 printf("Euclid parameter : %s %s\n", euclidargv_[2*i], 
                                                      euclidargv_[2*i+1]);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_EuclidSetParams(HYPrecon_,euclidargc_*2,euclidargv_);
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                          HYPRE_EuclidSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBLOCK :
           printf("TFQMR : block preconditioning not available.\n");
           exit(1);
           break;

#ifdef HAVE_ML
      case HYML :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("ML strong threshold = %e\n", mlStrongThreshold_);
              printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
              printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
              printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
              printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
              printf("ML relax weight     = %e\n", mlRelaxWeight_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
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
              HYPRE_ParCSRTFQmrSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                          HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;
#endif

#ifdef HAVE_MLI
      case HYMLI :
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
           break;
#endif
   }
   return;
}

//***************************************************************************
// set up preconditioners for BiCGS
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupBiCGSPrecon()
{
   int    i, *num_sweeps, *relax_type;
   double *relax_wt;

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
           if (pilutFillin_ == 0) pilutFillin_ = pilutMaxNnzPerRow_;
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("PILUT - row size = %d\n", pilutFillin_);
              printf("PILUT - drop tol = %e\n", pilutDropTol_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRPilutSetFactorRowSize(HYPrecon_,pilutFillin_);
              HYPRE_ParCSRPilutSetDropTolerance(HYPrecon_,pilutDropTol_);
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_ParCSRPilutSolve,
                                          HYPRE_ParCSRPilutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDILUT :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDILUT - fillin   = %e\n", ddilutFillin_);
              printf("DDILUT - drop tol = %e\n", ddilutDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDIlutSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( ddilutReorder_ ) HYPRE_LSI_DDIlutSetReorder(HYPrecon_);
              HYPRE_LSI_DDIlutSetFillin(HYPrecon_,ddilutFillin_);
              HYPRE_LSI_DDIlutSetDropTolerance(HYPrecon_,ddilutDropTol_);
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_DDIlutSolve,
                                          HYPRE_LSI_DDIlutSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYDDICT :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDICT - fillin   = %e\n", ddictFillin_);
              printf("DDICT - drop tol = %e\n", ddictDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
              HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                          HYPRE_LSI_DDICTSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYSCHWARZ :
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_SchwarzSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_SchwarzSetILUTFillin(HYPrecon_,schwarzFillin_);
              HYPRE_LSI_SchwarzSetNBlocks(HYPrecon_, schwarzNblocks_);
              HYPRE_LSI_SchwarzSetBlockSize(HYPrecon_, schwarzBlksize_);
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_SchwarzSolve,
                                          HYPRE_LSI_SchwarzSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPOLY :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Polynomial preconditioning - order = %d\n",polyOrder_);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                          HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
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
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRParaSailsSetSym(HYPrecon_, parasailsSym_);
              HYPRE_ParCSRParaSailsSetParams(HYPrecon_, parasailsThreshold_,
                                             parasailsNlevels_);
              HYPRE_ParCSRParaSailsSetFilter(HYPrecon_, parasailsFilter_);
              HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_, parasailsLoadbal_);
              HYPRE_ParCSRParaSailsSetReuse(HYPrecon_, parasailsReuse_);
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_ParCSRParaSailsSolve,
                                          HYPRE_ParCSRParaSailsSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("AMG coarsen type = %d\n", amgCoarsenType_);
              printf("AMG measure type = %d\n", amgMeasureType_);
              printf("AMG threshold    = %e\n", amgStrongThreshold_);
              printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
              printf("AMG relax type   = %d\n", amgRelaxType_[0]);
              printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
              printf("AMG system size  = %d\n", amgSystemSize_);
           }
           if ( HYOutputLevel_ & HYFEI_AMGDEBUG )
           {
              HYPRE_BoomerAMGSetDebugFlag(HYPrecon_, 0);
              HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 3);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( amgSystemSize_ > 1 )
                 HYPRE_BoomerAMGSetNumFunctions(HYPrecon_, amgSystemSize_);
              HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
              HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
              HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_,amgStrongThreshold_);
              num_sweeps = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

              HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
              relax_type = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

              HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
              relax_wt = hypre_CTAlloc(double,25);
              for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
              HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_BoomerAMGSolve,
                                          HYPRE_BoomerAMGSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYEUCLID :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
              for ( i = 0; i < euclidargc_; i++ )
                 printf("Euclid parameter : %s %s\n", euclidargv_[2*i], 
                                                      euclidargv_[2*i+1]);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_EuclidSetParams(HYPrecon_,euclidargc_*2,euclidargv_);
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_EuclidSolve,
                                          HYPRE_EuclidSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBLOCK :
           printf("BiCGS : block preconditioning not available.\n");
           exit(1);
           break;

#ifdef HAVE_ML
      case HYML :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("ML strong threshold = %e\n", mlStrongThreshold_);
              printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
              printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
              printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
              printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
              printf("ML relax weight     = %e\n", mlRelaxWeight_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                          HYPRE_DummyFunction, HYPrecon_);
           else
           {
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
              HYPRE_ParCSRBiCGSSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                          HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;
#endif

#ifdef HAVE_MLI
      case HYMLI :
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
           break;
#endif
   }
   return;
}

//***************************************************************************
// set up preconditioners for Symmetric QMR
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::setupSymQMRPrecon()
{
   int          i, *num_sweeps, *relax_type;
   double       *relax_wt;
   HYPRE_Lookup *newLookup;

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
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("DDICT - fillin   = %e\n", ddictFillin_);
              printf("DDICT - drop tol = %e\n", ddictDropTol_);
           }
           if ( HYOutputLevel_ & HYFEI_DDILUT )
              HYPRE_LSI_DDICTSetOutputLevel(HYPrecon_,2);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_LSI_DDICTSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_DDICTSetFillin(HYPrecon_,ddictFillin_);
              HYPRE_LSI_DDICTSetDropTolerance(HYPrecon_,ddictDropTol_);
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
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
              printf("Polynomial preconditioning - order = %d\n",polyOrder_);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_LSI_PolySetOrder(HYPrecon_, polyOrder_);
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_LSI_PolySolve,
                                           HYPRE_LSI_PolySetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYPARASAILS :
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0 )
           {
              printf("ParaSails - nlevels   = %d\n",parasailsNlevels_);
              printf("ParaSails - threshold = %e\n",parasailsThreshold_);
              printf("ParaSails - filter    = %e\n",parasailsFilter_);
              printf("ParaSails - sym       = 1\n");
              printf("ParaSails - loadbal   = %e\n",parasailsLoadbal_);
           }
           if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
              HYPRE_ParCSRParaSailsSetLogging(HYPrecon_, 1);
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_ParCSRParaSailsSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              HYPRE_ParCSRParaSailsSetSym(HYPrecon_, 1);
              HYPRE_ParCSRParaSailsSetParams(HYPrecon_, parasailsThreshold_,
                                             parasailsNlevels_);
              HYPRE_ParCSRParaSailsSetFilter(HYPrecon_, parasailsFilter_);
              HYPRE_ParCSRParaSailsSetLoadbal(HYPrecon_, parasailsLoadbal_);
              HYPRE_ParCSRParaSailsSetReuse(HYPrecon_, parasailsReuse_);
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_ParCSRParaSailsSolve,
                                           HYPRE_ParCSRParaSailsSetup,HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

      case HYBOOMERAMG :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("AMG coarsen type = %d\n", amgCoarsenType_);
              printf("AMG measure type = %d\n", amgMeasureType_);
              printf("AMG threshold    = %e\n", amgStrongThreshold_);
              printf("AMG numsweeps    = %d\n", amgNumSweeps_[0]);
              printf("AMG relax type   = %d\n", amgRelaxType_[0]);
              printf("AMG relax weight = %e\n", amgRelaxWeight_[0]);
              printf("AMG system size  = %d\n", amgSystemSize_);
           }
           if ( HYOutputLevel_ & HYFEI_AMGDEBUG )
           {
              HYPRE_BoomerAMGSetDebugFlag(HYPrecon_, 0);
              HYPRE_BoomerAMGSetIOutDat(HYPrecon_, 3);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_,HYPRE_BoomerAMGSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
              if ( amgSystemSize_ > 1 )
                 HYPRE_BoomerAMGSetNumFunctions(HYPrecon_, amgSystemSize_);
              HYPRE_BoomerAMGSetCoarsenType(HYPrecon_, amgCoarsenType_);
              HYPRE_BoomerAMGSetMeasureType(HYPrecon_, amgMeasureType_);
              HYPRE_BoomerAMGSetStrongThreshold(HYPrecon_, amgStrongThreshold_);
              num_sweeps = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];

              HYPRE_BoomerAMGSetNumGridSweeps(HYPrecon_, num_sweeps);
              relax_type = hypre_CTAlloc(int,4);
              for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];

              HYPRE_BoomerAMGSetGridRelaxType(HYPrecon_, relax_type);
              relax_wt = hypre_CTAlloc(double,25);
              for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
              HYPRE_BoomerAMGSetRelaxWeight(HYPrecon_, relax_wt);
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
           newLookup = (HYPRE_Lookup *) malloc(sizeof(HYPRE_Lookup));
           newLookup->object = (void *) lookup_;
           HYPRE_LSI_BlockPrecondSetLookup( HYPrecon_, newLookup );
           free( newLookup );
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, 
                    HYPRE_LSI_BlockPrecondSolve, HYPRE_DummyFunction, 
                    HYPrecon_);
           else
           {
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, 
                                      HYPRE_LSI_BlockPrecondSolve, 
                                      HYPRE_LSI_BlockPrecondSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;

#ifdef HAVE_ML
      case HYML :
           if ((HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 && mypid_ == 0)
           {
              printf("ML strong threshold = %e\n", mlStrongThreshold_);
              printf("ML numsweeps(pre)   = %d\n", mlNumPreSweeps_);
              printf("ML numsweeps(post)  = %d\n", mlNumPostSweeps_);
              printf("ML smoother (pre)   = %d\n", mlPresmootherType_);
              printf("ML smoother (post)  = %d\n", mlPostsmootherType_);
              printf("ML relax weight     = %e\n", mlRelaxWeight_);
           }
           if ( HYPreconReuse_ == 1 && HYPreconSetup_ == 1 )
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_, HYPRE_LSI_MLSolve,
                                           HYPRE_DummyFunction, HYPrecon_);
           else
           {
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
              HYPRE_ParCSRSymQMRSetPrecond(HYSolver_,HYPRE_LSI_MLSolve,
                                           HYPRE_LSI_MLSetup, HYPrecon_);
              HYPreconSetup_ = 1;
           }
           break;
#endif

#ifdef HAVE_MLI
      case HYMLI :
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
           break;
#endif
   }
   return;
}

//***************************************************************************
// this function solve the incoming linear system using Boomeramg
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::solveUsingBoomeramg(int& status)
{
   int                i, *relax_type, *num_sweeps;
   double             *relax_wt;
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

   num_sweeps = hypre_CTAlloc(int,4);
   for ( i = 0; i < 4; i++ ) num_sweeps[i] = amgNumSweeps_[i];
   HYPRE_BoomerAMGSetNumGridSweeps(HYSolver_, num_sweeps);

   relax_type = hypre_CTAlloc(int,4);
   for ( i = 0; i < 4; i++ ) relax_type[i] = amgRelaxType_[i];
   HYPRE_BoomerAMGSetGridRelaxType(HYSolver_, relax_type);

   relax_wt = hypre_CTAlloc(double,25);
   for ( i = 0; i < 25; i++ ) relax_wt[i] = amgRelaxWeight_[i];
   HYPRE_BoomerAMGSetRelaxWeight(HYSolver_, relax_wt);

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
      printf("* convergence tolerance = %e\n", tolerance_);
      printf("*--------------------------------------------------\n");
   }
   if ( HYOutputLevel_ & HYFEI_AMGDEBUG )
   {
      HYPRE_BoomerAMGSetDebugFlag(HYSolver_, 0);
      HYPRE_BoomerAMGSetIOutDat(HYSolver_, 3);
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

void HYPRE_LinSysCore::solveUsingSuperLU(int& status)
{
#ifdef HAVE_SUPERLU
   int                i, nnz, nrows, ierr;
   int                rowSize, *colInd, *new_ia, *new_ja, *ind_array;
   int                j, nz_ptr, *partition, start_row, end_row;
   double             *colVal, *new_a, rnorm;
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_ParVector    r_csr;
   HYPRE_ParVector    b_csr;
   HYPRE_ParVector    x_csr;

   int                info, panel_size, permc_spec;
   int                *perm_r, *perm_c;
   double             *rhs, *soln;
   mem_usage_t        mem_usage;
   SuperMatrix        A2, B, L, U;
   NRformat           *Astore, *Ustore;
   SCformat           *Lstore;
   DNformat           *Bstore;

   //-------------------------------------------------------------------
   // available for sequential processing only for now
   //-------------------------------------------------------------------

   if ( numProcs_ > 1 )
   {
      printf("solveUsingSuperLU ERROR - too many processors.\n");
      status = -1;
      return;
   }

   //-------------------------------------------------------------------
   // need to construct a CSR matrix, and the column indices should
   // have been stored in colIndices and rowLengths
   //-------------------------------------------------------------------
      
   if ( localStartRow_ != 1 )
   {
      printf("solveUsingSuperLU ERROR - row does not start at 1\n");
      status = -1;
      return;
   }

   //-------------------------------------------------------------------
   // get information about the current matrix
   //-------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(currA_, (void **) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &partition );
   start_row = partition[0];
   end_row   = partition[1] - 1;
   nrows     = end_row - start_row + 1;

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

   dCreate_CompRow_Matrix(&A2,nrows,nrows,nnz,new_a,new_ja,new_ia,NR,D_D,GE);
   ind_array = new int[nrows];
   for ( i = 0; i < nrows; i++ ) ind_array[i] = i;
   rhs = new double[nrows];

   ierr = HYPRE_IJVectorGetValues(currB_, nrows, ind_array, rhs);

   assert(!ierr);
   dCreate_Dense_Matrix(&B, nrows, 1, rhs, nrows, DN, D_D, GE);

   //-------------------------------------------------------------------
   // set up the rest and solve (permc_spec=0 : natural ordering)
   //-------------------------------------------------------------------
 
   perm_r = new int[nrows];
   perm_c = new int[nrows];
   permc_spec = superluOrdering_;
   get_perm_c(permc_spec, &A2, perm_c);
   panel_size = sp_ienv(1);

   dgssv(&A2, perm_c, perm_r, &L, &U, &B, &info);

   //-------------------------------------------------------------------
   // postprocessing of the return status information
   //-------------------------------------------------------------------

   if ( info == 0 ) 
   {
       status = 1;
       Lstore = (SCformat *) L.Store;
       Ustore = (NRformat *) U.Store;
       //printf("No of nonzeros in factor L = %d\n", Lstore->nnz);
       //printf("No of nonzeros in factor U = %d\n", Ustore->nnz);
       //printf("SuperLU : NNZ in L+U = %d\n",Lstore->nnz+Ustore->nnz-nrows);

       //dQuerySpace(&L, &U, panel_size, &mem_usage);
       //printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
       //       mem_usage.for_lu/1e6, mem_usage.total_needed/1e6,
       //       mem_usage.expansions);

   } 
   else 
   {
       status = 0;
       printf("HYPRE_LinSysCore::solveUsingSuperLU - dgssv error = %d\n",info);
       //if ( info <= nrows ) { /* factorization completes */
       //    dQuerySpace(&L, &U, panel_size, &mem_usage);
       //    printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
       //           mem_usage.for_lu/1e6, mem_usage.total_needed/1e6,
       //           mem_usage.expansions);
       //}
   }

   //-------------------------------------------------------------------
   // fetch the solution and find residual norm
   //-------------------------------------------------------------------

   if ( info == 0 )
   {
      soln = (double *) ((DNformat *) B.Store)->nzval;
      ierr = HYPRE_IJVectorSetValues(currX_, nrows, (const int *) ind_array,
                   	       (const double *) soln);
      assert(!ierr);

      HYPRE_IJVectorGetObject(currX_, (void **) &x_csr);
      HYPRE_IJVectorGetObject(currB_, (void **) &b_csr);
      HYPRE_IJVectorGetObject(currR_, (void **) &r_csr);

      ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
      assert(!ierr);
      HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
      ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
      assert(!ierr);
      rnorm = sqrt( rnorm );
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
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
#else
   status = -1;
   printf("HYPRE_LSC::solveUsingSuperLU : not available.\n");
#endif
}

//***************************************************************************
// this function solve the incoming linear system using SuperLU
// using expert mode
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::solveUsingSuperLUX(int& status)
{
#ifdef HAVE_SUPERLU
   int                i, k, nnz, nrows, ierr;
   int                rowSize, *colInd, *new_ia, *new_ja, *ind_array;
   int                j, nz_ptr, *colLengths, count, maxRowSize, rowSize2;
   int                *partition, start_row, end_row;
   double             *colVal, *new_a, rnorm;
   HYPRE_ParCSRMatrix A_csr;
   HYPRE_ParVector    r_csr;
   HYPRE_ParVector    b_csr;
   HYPRE_ParVector    x_csr;

   int                info, panel_size, permc_spec;
   int                *perm_r, *perm_c, *etree, lwork, relax;
   double             *rhs, *soln;
   double             *R, *C;
   double             *ferr, *berr;
   double             rpg, rcond;
   char               fact[1], equed[1], trans[1], refact[1];
   void               *work=NULL;
   mem_usage_t        mem_usage;
   SuperMatrix        A2, B, X, L, U;
   NRformat           *Astore, *Ustore;
   SCformat           *Lstore;
   DNformat           *Bstore;
   factor_param_t     iparam;

   //-------------------------------------------------------------------
   // available for sequential processing only for now
   //-------------------------------------------------------------------

   if ( numProcs_ > 1 )
   {
      printf("solveUsingSuperLUX ERROR - too many processors.\n");
      status = -1;
      return;
   }

   //-------------------------------------------------------------------
   // need to construct a CSR matrix, and the column indices should
   // have been stored in colIndices and rowLengths
   //-------------------------------------------------------------------
      
   if ( localStartRow_ != 1 )
   {
      printf("solveUsingSuperLUX ERROR - row not start at 1\n");
      status = -1;
      return;
   }

   //-------------------------------------------------------------------
   // get information about the current matrix
   //-------------------------------------------------------------------

   HYPRE_IJMatrixGetObject(currA_, (void**) &A_csr);
   HYPRE_ParCSRMatrixGetRowPartitioning( A_csr, &partition );
   start_row = partition[0];
   end_row   = partition[1] - 1;
   nrows     = end_row - start_row + 1;

   colLengths = new int[nrows];
   for ( i = 0; i < nrows; i++ ) colLengths[i] = 0;
   
   maxRowSize = 0;
   for ( i = 0; i < nrows; i++ )
   {
      HYPRE_ParCSRMatrixGetRow(A_csr,i,&rowSize,&colInd,&colVal);
      maxRowSize = ( rowSize > maxRowSize ) ? rowSize : maxRowSize;
      for ( j = 0; j < rowSize; j++ ) 
         if ( colVal[j] != 0.0 ) colLengths[colInd[j]]++;
      HYPRE_ParCSRMatrixRestoreRow(A_csr,i,&rowSize,&colInd,&colVal);
   }   
   nnz = 0;
   for ( i = 0; i < nrows; i++ ) nnz += colLengths[i];

   new_ia = new int[nrows+1];
   new_ja = new int[nnz];
   new_a  = new double[nnz];
   nz_ptr = HYPRE_LSI_GetParCSRMatrix(currA_,nrows,nnz,new_ia,new_ja,new_a);
   nnz = nz_ptr;

   //-------------------------------------------------------------------
   // set up SuperLU CSR matrix and the corresponding rhs
   //-------------------------------------------------------------------

   dCreate_CompRow_Matrix(&A2,nrows,nrows,nnz,new_a,new_ja,new_ia,NR,D_D,GE);
   ind_array = new int[nrows];
   for ( i = 0; i < nrows; i++ ) ind_array[i] = i;
   rhs = new double[nrows];

   ierr = HYPRE_IJVectorGetValues(currB_, nrows, ind_array, rhs);
   assert(!ierr);
   dCreate_Dense_Matrix(&B, nrows, 1, rhs, nrows, DN, D_D, GE);
   soln = new double[nrows];
   for ( i = 0; i < nrows; i++ ) soln[i] = 0.0;
   dCreate_Dense_Matrix(&X, nrows, 1, soln, nrows, DN, D_D, GE);

   //-------------------------------------------------------------------
   // set up the other parameters (permc_spec=0 : natural ordering)
   //-------------------------------------------------------------------
 
   perm_r = new int[nrows];
   perm_c = new int[nrows];
   etree  = new int[nrows];
   permc_spec = superluOrdering_;
   get_perm_c(permc_spec, &A2, perm_c);
   panel_size               = sp_ienv(1);
   iparam.panel_size        = panel_size;
   iparam.relax             = sp_ienv(2);
   iparam.diag_pivot_thresh = 1.0;
   iparam.drop_tol          = -1;
   lwork                    = 0;
   *fact                    = 'N';
   *equed                   = 'N';
   *trans                   = 'N';
   *refact                  = 'N';
   R    = (double *) SUPERLU_MALLOC(A2.nrow * sizeof(double));
   C    = (double *) SUPERLU_MALLOC(A2.ncol * sizeof(double));
   ferr = (double *) SUPERLU_MALLOC(sizeof(double));
   berr = (double *) SUPERLU_MALLOC(sizeof(double));

   //-------------------------------------------------------------------
   // solve
   //-------------------------------------------------------------------

   dgssvx(fact, trans, refact, &A2, &iparam, perm_c, perm_r, etree,
          equed, R, C, &L, &U, work, lwork, &B, &X, &rpg, &rcond,
          ferr, berr, &mem_usage, &info);

   //-------------------------------------------------------------------
   // print SuperLU internal information at the first step
   //-------------------------------------------------------------------
       
   if ( info == 0 || info == nrows+1 ) 
   {
      status = 1;
      Lstore = (SCformat *) L.Store;
      Ustore = (NRformat *) U.Store;
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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
         dQuerySpace(&L, &U, panel_size, &mem_usage);
         printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
                mem_usage.for_lu/1e6, mem_usage.total_needed/1e6,
                mem_usage.expansions);
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
      ierr = HYPRE_IJVectorSetValues(currX_, nrows, (const int *) &ind_array,
                   	       (const double *) soln);
      assert(!ierr);

      HYPRE_IJVectorGetObject(currX_, (void **) &x_csr);
      HYPRE_IJVectorGetObject(currR_, (void **) &r_csr);
      HYPRE_IJVectorGetObject(currB_, (void **) &b_csr);
      ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
      assert(!ierr);
      ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
      assert(!ierr);
      ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
      assert(!ierr);
      rnorm = sqrt( rnorm );
      if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 1 )
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
   delete [] new_ia;
   delete [] new_ja;
   delete [] new_a;
   delete [] soln;
   delete [] colLengths;
   Destroy_SuperMatrix_Store(&B);
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
#else
   status = -1;
   printf("HYPRE_LSC::solveUsingSuperLUX : not available.\n");
#endif
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
   assert(!ierr);

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
      assert(!ierr);

      HYPRE_IJVectorGetObject(currX_, (void**) &x_csr);
      HYPRE_IJVectorGetObject(currR_, (void**) &r_csr);
      HYPRE_IJVectorGetObject(currB_, (void**) &b_csr);
      ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
      assert(!ierr);
      ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
      assert(!ierr);
      ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
      assert(!ierr);
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
   assert(!ierr);

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
   assert(!ierr);

   HYPRE_IJVectorGetObject(currX_, (void**) &x_csr);
   HYPRE_IJVectorGetObject(currR_, (void**) &r_csr);
   HYPRE_IJVectorGetObject(currB_, (void**) &b_csr);

   ierr = HYPRE_ParVectorCopy( b_csr, r_csr );
   assert(!ierr);
   ierr = HYPRE_ParCSRMatrixMatvec( -1.0, A_csr, x_csr, 1.0, r_csr );
   assert(!ierr);
   ierr = HYPRE_ParVectorInnerProd( r_csr, r_csr, &rnorm);
   assert(!ierr);
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
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
      printf("%4d : HYPRE_LSC::loadConstraintNumbers - size = %d\n", 
                    mypid_, nConstr);
   nConstraints_ = nConstr;
   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
      printf("%4d : HYPRE_LSC::leaving  loadConstraintNumbers\n", mypid_);
}

//***************************************************************************
// this function extracts the the version number from HYPRE
//---------------------------------------------------------------------------

char *HYPRE_LinSysCore::getVersion()
{
   static char version[100];
   char        hypre[200], hypre_version[50], ctmp[50];
   sprintf(hypre, "%s", HYPRE_Version());
   sscanf(hypre, "%s %s", ctmp, hypre_version);
   sprintf(version, "%s-%s", HYPRE_FEI_Version(), hypre_version);
   return version;
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
   int    i, ierr, *equations, local_nrows, *iarray;
   double *darray, *answers;

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
      printf("%4d : HYPRE_LSC::entering endCreateMapFromSoln.\n",mypid_);

   mapFromSolnFlag_ = 0;
   if ( mapFromSolnLeng_ > 0 )
      darray = new double[mapFromSolnLeng_];
   for ( i = 0; i < mapFromSolnLeng_; i++ )
      darray[i] = (double) mapFromSolnList_[i];

   qsort1(mapFromSolnList2_, darray, 0, mapFromSolnLeng_-1);
   iarray = mapFromSolnList2_;
   mapFromSolnList2_ = mapFromSolnList_;
   mapFromSolnList_ = iarray;
   for ( i = 0; i < mapFromSolnLeng_; i++ )
      mapFromSolnList2_[i] = (int) darray[i];
   delete [] darray;

   for ( i = 0; i < mapFromSolnLeng_; i++ )
      printf("HYPRE_LSC::mapFromSoln %d = %d\n",mapFromSolnList_[i],
             mapFromSolnList2_[i]);

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 2 )
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
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 0 )
            printf("%4d : putIntoMappedMatrix (add) : row, col = %8d %8d %e \n",
                   mypid_, localRow, colIndices_[localRow][ind2]-1,
                   colValues_[localRow][ind2]);
      }
      else
      {
         ind2 = index;
         colIndices_[localRow][index] = mappedCol + 1;
         colValues_[localRow][index++] = values[i];
         if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 0 )
            printf("%4d : putIntoMappedMatrix : row, col = %8d %8d %e \n",
                   mypid_, localRow, colIndices_[localRow][ind2]-1,
                   colValues_[localRow][ind2]);
         qsort1(colIndices_[localRow],colValues_[localRow],0,index-1);
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

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
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

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC:: leaving computeMinResProjection n", mypid_);
   return;
}

//***************************************************************************
// add a new pair of (x,b) vectors to the projection space
//---------------------------------------------------------------------------

void HYPRE_LinSysCore::addToMinResProjectionSpace(HYPRE_IJVector xvec,
                                                  HYPRE_IJVector bvec)
{
   int                i, ierr, nrows, *partition, start_row, end_row;
   double             alpha;
   HYPRE_ParVector    v_csr, x_csr, xn_csr, b_csr, r_csr, bn_csr;
   HYPRE_IJVector     tmpxvec, tmpbvec;
   HYPRE_ParCSRMatrix A_csr;

   //-----------------------------------------------------------------------
   // diagnostic message
   //-----------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
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
      nrows     = end_row - start_row + 1;
      HYpxs_    = new HYPRE_IJVector[projectSize_+1];
      HYpbs_    = new HYPRE_IJVector[projectSize_+1];

      for ( i = 0; i <= projectSize_; i++ )
      {
         ierr = HYPRE_IJVectorCreate(comm_, start_row, end_row, &(HYpbs_[i]));
         ierr = HYPRE_IJVectorSetObjectType(HYpbs_[i], HYPRE_PARCSR);
         ierr = HYPRE_IJVectorInitialize(HYpbs_[i]);
         ierr = HYPRE_IJVectorAssemble(HYpbs_[i]);
         assert( !ierr );
      }
      for ( i = 0; i <= projectSize_; i++ )
      {
         ierr = HYPRE_IJVectorCreate(comm_, start_row, end_row, &(HYpxs_[i]));
         ierr = HYPRE_IJVectorSetObjectType(HYpxs_[i], HYPRE_PARCSR);
         ierr = HYPRE_IJVectorInitialize(HYpxs_[i]);
         ierr = HYPRE_IJVectorAssemble(HYpxs_[i]);
         assert(!ierr);
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

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
      printf("%4d : HYPRE_LSC::leaving addToProjectionSpace %d\n",mypid_,
              projectCurrSize_);
}

//***************************************************************************
// project the initial guess into the previous solution space
//
//          min   || trans(x - xbar) A (x - xbar) ||
//
// where xbar is a linear combination of the A-conjugate vectors built from
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

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
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

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
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
   int                i, k, ierr, nrows, *partition, start_row, end_row;
   double             alpha, acc_norm;
   HYPRE_ParVector    v_csr, x_csr, b_csr, bn_csr, xn_csr;
   HYPRE_IJVector     tmpxvec;
   HYPRE_ParCSRMatrix A_csr;

   //-----------------------------------------------------------------------
   // diagnostic message
   //-----------------------------------------------------------------------

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
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
      nrows     = end_row - start_row + 1;
      HYpxs_    = new HYPRE_IJVector[projectSize_+1];
      HYpbs_    = new HYPRE_IJVector[projectSize_+1];

      for ( i = 0; i <= projectSize_; i++ )
      {
         ierr = HYPRE_IJVectorCreate(comm_, start_row, end_row, &(HYpbs_[i]));
         ierr = HYPRE_IJVectorSetObjectType(HYpbs_[i], HYPRE_PARCSR);
         ierr = HYPRE_IJVectorInitialize(HYpbs_[i]);
         ierr = HYPRE_IJVectorAssemble(HYpbs_[i]);
         assert( !ierr );
      }
      for ( i = 0; i <= projectSize_; i++ )
      {
         ierr = HYPRE_IJVectorCreate(comm_, start_row, end_row, &(HYpxs_[i]));
         ierr = HYPRE_IJVectorSetObjectType(HYpxs_[i], HYPRE_PARCSR);
         ierr = HYPRE_IJVectorInitialize(HYpxs_[i]);
         ierr = HYPRE_IJVectorAssemble(HYpxs_[i]);
         assert(!ierr);
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

   if ( (HYOutputLevel_ & HYFEI_SPECIALMASK) >= 3 )
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
   if ( haveFEData_ && feData_ != NULL )
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
   if ( haveFEData_ && feData_ != NULL )
   {
      status = HYPRE_LSI_MLIFEDataInitElemBlock(feData_, nElems, 
                           nNodesPerElem, numNodeFields, nodeFieldIDs);
      if ( status )
      {
         delete feData_;
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
   if ( haveFEData_ && feData_ != NULL )
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
   if ( haveFEData_ && feData_ != NULL )
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
   if ( haveFEData_ && feData_ != NULL )
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
   if ( haveFEData_ && feData_ != NULL )
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

