/*BHEADER**********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <stdio.h>
#include <math.h>

#include "solver/mli_solver_chebyshev.h"
#include "base/mli_defs.h"
#include "parcsr_mv/parcsr_mv.h"

#define min(x,y) (((x) < (y)) ? (x) : (y))

/******************************************************************************
 * constructor
 *---------------------------------------------------------------------------*/

MLI_Solver_Chebyshev::MLI_Solver_Chebyshev() : 
                          MLI_Solver(MLI_SOLVER_CHEBYSHEV_ID)
{
   Amat      = NULL;
   mli_Vtemp = NULL;
   mli_Wtemp = NULL;
   mli_Ytemp = NULL;
   max_eigen = -1.0;
   degree    = 2;
}

/******************************************************************************
 * destructor
 *---------------------------------------------------------------------------*/

MLI_Solver_Chebyshev::~MLI_Solver_Chebyshev()
{
   Amat = NULL;
   if ( mli_Vtemp != NULL ) delete mli_Vtemp;
   if ( mli_Wtemp != NULL ) delete mli_Wtemp;
   if ( mli_Ytemp != NULL ) delete mli_Ytemp;
}

/******************************************************************************
 * set up the smoother
 *---------------------------------------------------------------------------*/

int MLI_Solver_Chebyshev::setup(MLI_Matrix *mat)
{
   int    i, j, globalNRows;
   double cosData0, cosData1, coord;
   double sample, gridStep, rho, rho2, ddeg;
   double pi=4.e0 * atan(1.e0); /* 3.141592653589793115998e0; */

   /*-----------------------------------------------------------------
    * check that proper spectral radius is passed in
    *-----------------------------------------------------------------*/

   Amat = mat;
   if ( max_eigen == 0.0 )
   {
      cout << "MLI_Solver_Chebyshev ERROR : max_eigen <= 0.0.\n";
      exit(1);
   }
   A = (hypre_ParCSRMatrix *) Amat->getMatrix();
   

    /*-----------------------------------------------------------------
    * compute the coefficients
    *-----------------------------------------------------------------*/

   for ( i = 0; i < MAX_DEG; i++ ) mlsOm[i] = 0.e0;
   ddeg = (double) mlsDeg;
   rho  = mlsOver * max_eigen;
   cosData1 = 1.e0 / (2.e0 * ddeg + 1.e0);
   for ( i = 0; i < mlsDeg; i++ ) 
   {
      cosData0 = 2.e0 * (double)(i+1) * pi;
      mlsOm[i] = 2.e0 / (rho * (1.e0 - cos(cosData0 * cosData1)));
   }
   mlsCf[0] = + mlsOm[0] + mlsOm[1] + mlsOm[2] + mlsOm[3] + mlsOm[4];
   mlsCf[1] = -(mlsOm[0]*mlsOm[1] + mlsOm[0]*mlsOm[2]
              + mlsOm[0]*mlsOm[3] + mlsOm[0]*mlsOm[4]
              + mlsOm[1]*mlsOm[2] + mlsOm[1]*mlsOm[3]
              + mlsOm[1]*mlsOm[4] + mlsOm[2]*mlsOm[3]
              + mlsOm[2]*mlsOm[4] + mlsOm[3]*mlsOm[4]);
   mlsCf[2] = +(mlsOm[0]*mlsOm[1]*mlsOm[2] + mlsOm[0]*mlsOm[1]*mlsOm[3]
              + mlsOm[0]*mlsOm[1]*mlsOm[4] + mlsOm[0]*mlsOm[2]*mlsOm[3]
              + mlsOm[0]*mlsOm[2]*mlsOm[4] + mlsOm[0]*mlsOm[3]*mlsOm[4]
              + mlsOm[1]*mlsOm[2]*mlsOm[3] + mlsOm[1]*mlsOm[2]*mlsOm[4]
              + mlsOm[1]*mlsOm[3]*mlsOm[4] + mlsOm[2]*mlsOm[3]*mlsOm[4]);
   mlsCf[3] = -(mlsOm[0]*mlsOm[1]*mlsOm[2]*mlsOm[3]
              + mlsOm[0]*mlsOm[1]*mlsOm[2]*mlsOm[4]
              + mlsOm[0]*mlsOm[1]*mlsOm[3]*mlsOm[4]
              + mlsOm[0]*mlsOm[2]*mlsOm[3]*mlsOm[4]
              + mlsOm[1]*mlsOm[2]*mlsOm[3]*mlsOm[4]);
   mlsCf[4] = mlsOm[0] * mlsOm[1] * mlsOm[2] * mlsOm[3] * mlsOm[4];

   if ( mlsDeg == 1 )
   {
      gridStep = rho / (double) nSamples;
      nGrid    = (int) min(rint(rho/gridStep)+1, nSamples);

      rho2 = 0.e0;
      for ( i = 0; i < nGrid; i++ ) 
      {
         coord  = (double)(i+1) * gridStep;
         sample = 1.e0 - mlsOm[0] * coord;
         for ( j = 1; j < mlsDeg; j++) sample *= sample * coord;
         if (sample > rho2) rho2 = sample;
      }
      /* this original code seems not as good
      rho2 = 4.0e0/(27.e0 * mlsOm[0]);
      */
   }
   else if ( mlsDeg > 1 )
   {
      gridStep = rho / (double) nSamples;
      nGrid    = (int) min(rint(rho/gridStep)+1, nSamples);

      rho2 = 0.e0;
      for ( i = 0; i < nGrid; i++ ) 
      {
         coord  = (double)(i+1) * gridStep;
         sample = 1.e0 - mlsOm[0] * coord;
         for ( j = 1; j < mlsDeg; j++) sample *= sample * coord;
         sample *= sample * coord;
         if (sample > rho2) rho2 = sample;
      }
   }

   if ( mlsDeg < 2) mlsBoost = 1.029e0;
   else             mlsBoost = 1.025e0;
   rho2 *= mlsBoost;
   mlsOm2 = 2.e0 / rho2;

   /*-----------------------------------------------------------------
    * allocate temporary vectors
    *-----------------------------------------------------------------*/

   if ( mli_Vtemp != NULL ) delete mli_Vtemp;
   if ( mli_Wtemp != NULL ) delete mli_Wtemp;
   if ( mli_Ytemp != NULL ) delete mli_Ytemp;
   mli_Vtemp = mat->createVector();
   mli_Wtemp = mat->createVector();
   mli_Ytemp = mat->createVector();
   return 0;
}

/******************************************************************************
 * apply function
 *---------------------------------------------------------------------------*/

int MLI_Solver_Chebyshev::solve(MLI_Vector *f_in, MLI_Vector *u_in)
{
   hypre_ParCSRMatrix  *A;
   hypre_CSRMatrix     *A_diag;
   hypre_Vector        *u_local, *Vtemp_local, *Wtemp_local, *Ytemp_local;
   hypre_ParVector     *Vtemp, *Wtemp, *Ytemp, *f, *u;
   int                 i, nrows, globalNRows, degree;
   double              omega, coef, *u_data, *Vtemp_data;
   double              *Wtemp_data, *Ytemp_data;
   void                *void_vector;

   /*-----------------------------------------------------------------
    * check that proper spectral radius is passed in
    *-----------------------------------------------------------------*/

   if ( max_eigen <= 0.0 )
   {
      cout << "Solver_Chebyshev::solver ERROR : max_eig <= 0.\n"; 
      exit(1);
   }

   /*-----------------------------------------------------------------
    * fetch machine and smoother parameters
    *-----------------------------------------------------------------*/

   A               = (hypre_ParCSRMatrix *) Amat->getMatrix();
   A_diag          = hypre_ParCSRMatrixDiag(A);
   nrows           = hypre_CSRMatrixNumRows(A_diag);
   f               = (hypre_ParVector *) f_in->getVector();
   u               = (hypre_ParVector *) u_in->getVector();
   u_local         = hypre_ParVectorLocalVector(u);
   u_data          = hypre_VectorData(u_local);
   globalNRows     = hypre_ParCSRMatrixGlobalNumRows(A);
   
   /*-----------------------------------------------------------------
    * fetch temporary vector
    *-----------------------------------------------------------------*/

   Vtemp       = (hypre_ParVector *) mli_Vtemp->getVector();
   Vtemp_local = hypre_ParVectorLocalVector(Vtemp);
   Vtemp_data  = hypre_VectorData(Vtemp_local);

   Wtemp       = (hypre_ParVector *) mli_Wtemp->getVector();
   Wtemp_local = hypre_ParVectorLocalVector(Wtemp);
   Wtemp_data  = hypre_VectorData(Wtemp_local);

   Ytemp       = (hypre_ParVector *) mli_Ytemp->getVector();
   Ytemp_local = hypre_ParVectorLocalVector(Ytemp);
   Ytemp_data  = hypre_VectorData(Ytemp_local);

   /*-----------------------------------------------------------------
    * Perform Chebyshev iterations
    *-----------------------------------------------------------------*/
 
   for ( i = 0 ; i < degree; i++ )
   {
      /* compute Ztemp = f - A u */

      hypre_ParVectorCopy(f,Ztemp); 
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, Ztemp);

   if ( mlsDeg == 1 )
   {
      coef = mlsCf[0] * mlsOver;

      /* u = u + coef * Vtemp */

#define HYPRE_SMP_PRIVATE i
#include "utilities/hypre_smp_forloop.h"
      for (i = 0; i < nrows; i++) u_data[i] += ( coef * Vtemp_data[i] );

      /* compute residual Vtemp = A u - f */

      hypre_ParVectorCopy(f,Vtemp); 
      hypre_ParCSRMatrixMatvec(1.0, A, u, -1.0, Vtemp);

      /* compute residual Wtemp = (I - omega * A) Vtemp */

      hypre_ParVectorCopy(Vtemp,Wtemp); 
      for ( deg = 0; deg < mlsDeg; deg++ ) 
      {
         omega = mlsOm[deg];
         hypre_ParCSRMatrixMatvec(-omega, A, Vtemp, 1.0, Wtemp);
      }

      /* compute residual Vtemp = (I - omega * A) Wtemp */

      hypre_ParVectorCopy(Wtemp,Vtemp); 
      for ( deg = mlsDeg-1; deg > -1; deg-- ) 
      {
         omega = mlsOm[deg];
         hypre_ParCSRMatrixMatvec(-omega, A, Wtemp, 1.0, Vtemp);
      }

      /* compute u = u - coef * Vtemp */

      coef = mlsOver * mlsOm2;

#define HYPRE_SMP_PRIVATE i
#include "utilities/hypre_smp_forloop.h"
      for (i = 0; i < nrows; i++) u_data[i] -= ( coef * Vtemp_data[i] );

   }
   else
   {
      /* Ytemp = coef * Vtemp */

      coef = mlsCf[0];

#define HYPRE_SMP_PRIVATE i
#include "utilities/hypre_smp_forloop.h"
      for (i = 0; i < nrows; i++) Ytemp_data[i] = ( coef * Vtemp_data[i] );

      /* Wtemp = coef * Vtemp */

      for ( deg = 1; deg < deg; deg++ ) 
      {
         hypre_ParCSRMatrixMatvec(1.0, A, Vtemp, -1.0, Wtemp);
         coef = mlsCf[deg-1];

#define HYPRE_SMP_PRIVATE i
#include "utilities/hypre_smp_forloop.h"
         for (i = 0; i < nrows; i++) 
         {
            Vtemp_data[i] = Wtemp_data[i];
            Ytemp_data[i] += ( coef * Wtemp_data[i] );
         }

      }

#define HYPRE_SMP_PRIVATE i
#include "utilities/hypre_smp_forloop.h"
      for (i = 0; i < nrows; i++) u_data[i] += ( mlsOver * Ytemp_data[i] );

      /* compute residual Vtemp = A u - f */

      hypre_ParVectorCopy(f,Vtemp); 
      hypre_ParCSRMatrixMatvec(1.0, A, u, -1.0, Vtemp);

      /* compute residual Wtemp = (I - omega * A) Vtemp */

      hypre_ParVectorCopy(Vtemp,Wtemp); 
      for ( deg = 0; deg < mlsDeg; deg++ ) 
      {
         omega = mlsOm[deg];
         hypre_ParCSRMatrixMatvec(-omega, A, Vtemp, 1.0, Wtemp);
      }

      /* compute residual Vtemp = (I - omega * A) Wtemp */

      hypre_ParVectorCopy(Wtemp,Vtemp); 
      for ( deg = mlsDeg-1; deg > -1; deg-- ) 
      {
         omega = mlsOm[deg];
         hypre_ParCSRMatrixMatvec(-omega, A, Wtemp, 1.0, Vtemp);
      }

      /* compute u = u - coef * Vtemp */

      coef = mlsOver * mlsOm2;

#define HYPRE_SMP_PRIVATE i
#include "utilities/hypre_smp_forloop.h"
      for (i = 0; i < nrows; i++) u_data[i] -= ( coef * Vtemp_data[i] );

   }

   return(0); 
}

/******************************************************************************
 * set Chebyshev parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_Chebyshev::setParams( char *param_string, int argc, char **argv )
{
   int    nsweeps;
   double *weights;

   if ( !strcmp(param_string, "relaxWeight") )
   {
      if ( argc != 2 && argc != 1 ) 
      {
         cout << "Solver_Chebyshev::setParams ERROR : needs 1 or 2 args.\n";
         return 1;
      }
      if ( argc >= 1 ) nsweeps = *(int*)   argv[0];
      if ( argc == 2 ) weights = (double*) argv[1];
      max_eigen = weights[0];
      if ( max_eigen < 0.0 ) 
      {
         cout << "Solver_Chebyshev::setParams ERROR : max_eig <= 0 (" 
              << max_eigen << ")\n";
         return 1;
      }
   }
   return 0;
}

/******************************************************************************
 * set Chebyshev parameters
 *---------------------------------------------------------------------------*/

int MLI_Solver_Chebyshev::setParams( double eigen_in )
{
   if ( max_eigen <= 0.0 )
   {
      cerr << "Solver_Chebyshev::setParams WARNING : max_eigen <= 0." << endl;
      return 1; 
   }
   max_eigen = eigen_in;
   return 0;
}

