// *************************************************************************
// This is the HYPRE implementation of block preconditioners
// *************************************************************************

#ifndef _HYPRE_INCFLOW_BLOCKPRECOND_
#define _HYPRE_INCFLOW_BLOCKPRECOND_

// *************************************************************************
// system libraries used
// -------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_ls/parcsr_ls.h"
#include "parcsr_mv/parcsr_mv.h"

// *************************************************************************
// local defines 
// -------------------------------------------------------------------------

#define HYPRE_INCFLOW_BDIAG  1
#define HYPRE_INCFLOW_BTRI   2
#define HYPRE_INCFLOW_BAI    3

// *************************************************************************
// class definition
// -------------------------------------------------------------------------

class HYPRE_IncFlow_BlockPrecond
{
   HYPRE_IJMatrix Amat_;
   HYPRE_IJMatrix A11mat_;
   HYPRE_IJMatrix A12mat_;
   HYPRE_IJMatrix A22mat_;
   int            P22Size_, P22GSize_;
   int            *P22LocalInds_, *P22GlobalInds_;
   int            *APartition_;
   int            assembled_;
   int            *P22Offsets_;
   int            outputLevel_;
   double         diffusionCoef_;
   double         timeStep_;
   int            M22Length_;
   double         *M22Diag_;
   int            scheme_;
   HYPRE_Solver   A11Solver_;
   HYPRE_Solver   A11Precond_;
   HYPRE_Solver   A22Solver_;
   HYPRE_Solver   A22Precond_;

 public:

   HYPRE_IncFlow_BlockPrecond(HYPRE_IJMatrix Amat);
   virtual ~HYPRE_IncFlow_BlockPrecond();
   int     setSchemeBDiag()   {scheme_ = HYPRE_INCFLOW_BDIAG; return 0;}
   int     setSchemeBTRI()    {scheme_ = HYPRE_INCFLOW_BTRI;  return 0;}
   int     setSchemeBAI()     {scheme_ = HYPRE_INCFLOW_BAI;   return 0;}
   int     setScalarParams( double timeStep, double diffusion );
   int     setVectorParams( int length, double *Mdiag );
   int     computeBlockInfo();
   int     buildBlocks();
   int     setup();
   int     solve( HYPRE_IJVector xvec, HYPRE_IJVector fvec );
   int     solveBSolve( HYPRE_IJVector x1, HYPRE_IJVector x2,
                        HYPRE_IJVector f1, HYPRE_IJVector f2 );
   int     solveBAI   ( HYPRE_IJVector x1, HYPRE_IJVector x2,
                        HYPRE_IJVector f1, HYPRE_IJVector f2 );
};

#endif

