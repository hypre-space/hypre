
/******************************************************
 *
 *  File:  Hypre_ParAMG.c
 *
 *********************************************************/

#include "Hypre_ParAMG_Skel.h" 
#include "Hypre_ParAMG_Data.h" 

#include "Hypre_ParCSRVector_Skel.h"
#include "Hypre_ParCSRVector_Data.h"
#include "Hypre_ParCSRMatrix_Skel.h"
#include "Hypre_ParCSRMatrix_Data.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_IJ_mv.h"
#include "IJ_matrix_vector.h"


/* *************************************************
 * Constructor
 *    Allocate Memory for private data
 *    and initialize here
 ***************************************************/
void Hypre_ParAMG_constructor(Hypre_ParAMG this) {
   this->d_table = (struct Hypre_ParAMG_private_type *)
      malloc( sizeof( struct Hypre_ParAMG_private_type ) );

   this->d_table->Hsolver = (HYPRE_Solver *)
     malloc( sizeof( HYPRE_Solver ) );

} /* end constructor */

/* *************************************************
 *  Destructor
 *      deallocate memory for private data here.
 ***************************************************/
void Hypre_ParAMG_destructor(Hypre_ParAMG this) {
   struct Hypre_ParAMG_private_type *HSp = this->d_table;
   HYPRE_Solver *S = HSp->Hsolver;

   HYPRE_ParAMGDestroy( *S );
   free(this->d_table);
} /* end destructor */

/* ********************************************************
 * impl_Hypre_ParAMGGetParameterDouble
 **********************************************************/
int impl_Hypre_ParAMG_GetParameterDouble
( Hypre_ParAMG this, char* name, double *value ) {
   printf( "Hypre_ParAMG_GetParameterDouble does not recognize name %s\n", name );
   *value = -123.456;
   return 1;
} /* end impl_Hypre_ParAMGGetParameterDouble */

/* ********************************************************
 * impl_Hypre_ParAMGGetParameterInt
 **********************************************************/
int impl_Hypre_ParAMG_GetParameterInt
( Hypre_ParAMG this, char* name, int *value ) {
   printf( "Hypre_ParAMG_GetParameterInt does not recognize name %s\n", name );
   *value = -123456;
   return 1;
} /* end impl_Hypre_ParAMGGetParameterInt */

/* ********************************************************
 * impl_Hypre_ParAMGSetParameterDouble
 **********************************************************/
int  impl_Hypre_ParAMG_SetParameterDouble
( Hypre_ParAMG this, char* name, double value ) {
/* This function just dispatches to the parameter's set function. */
   struct Hypre_ParAMG_private_type *HSp = this->d_table;
   HYPRE_Solver *S = HSp->Hsolver;

   if ( !strcmp(name,"tol") ) {
      return HYPRE_ParAMGSetTol( *S, value );
   }
   else if ( !strcmp(name,"strong threshold") ) {
      return HYPRE_ParAMGSetStrongThreshold( *S, value );
   }
   else if ( !strcmp(name,"trunc factor") ) {
      return HYPRE_ParAMGSetTruncFactor( *S, value );
   }
   else if ( !strcmp(name,"max row sum") ) {
      return HYPRE_ParAMGSetMaxRowSum( *S, value );
   }
   else
      printf( "Hypre_ParAMG_SetParameterDouble does not recognize name %s\n", name );

   return 1;
} /* end impl_Hypre_ParAMGSetParameterDouble */

/* ********************************************************
 * impl_Hypre_ParAMGSetParameterDoubleArray
 **********************************************************/
int impl_Hypre_ParAMG_SetParameterDoubleArray
( Hypre_ParAMG this, char* name, array1double value ) {
   struct Hypre_ParAMG_private_type *HSp = this->d_table;
   HYPRE_Solver *S = HSp->Hsolver;

   if ( !strcmp(name,"relax weight") ) {
      return HYPRE_ParAMGSetRelaxWeight( *S, value.data );
   }
   else
      printf(
         "Hypre_ParAMG_SetParameterDoubleArray does not recognize name %s\n", name );

   return 1;
} /* end impl_Hypre_ParAMGSetParameterDoubleArray */

/* ********************************************************
 * impl_Hypre_ParAMGSetParameterDoubleArray2
 **********************************************************/
int  impl_Hypre_ParAMG_SetParameterDoubleArray2
(Hypre_ParAMG this, char* name, array2double value) {
   printf(
      "Hypre_ParAMG_SetParameterDoubleArray2 does not recognize name %s\n", name );
   return 1;
} /* end impl_Hypre_ParAMGSetParameterDoubleArray2 */

/* ********************************************************
 * impl_Hypre_ParAMGSetParameterInt
 **********************************************************/
int  impl_Hypre_ParAMG_SetParameterInt(Hypre_ParAMG this, char* name, int value) {
   struct Hypre_ParAMG_private_type *HSp = this->d_table;
   HYPRE_Solver *S = HSp->Hsolver;

   if ( !strcmp(name,"restriction") ) {
      return HYPRE_ParAMGSetRestriction( *S, value );
   }
   else if ( !strcmp(name,"max levels") ) {
      return HYPRE_ParAMGSetMaxLevels( *S, value );
   }
   else if ( !strcmp(name,"interp type") ) {
      return HYPRE_ParAMGSetInterpType( *S, value );
   }
   else if ( !strcmp(name,"min iter") ) {
      return HYPRE_ParAMGSetMinIter( *S, value );
   }
   else if ( !strcmp(name,"max iter") ) {
      return HYPRE_ParAMGSetMaxIter( *S, value );
   }
   else if ( !strcmp(name,"coarsen type") ) {
      return HYPRE_ParAMGSetCoarsenType( *S, value );
   }
   else if ( !strcmp(name,"measure type") ) {
      return HYPRE_ParAMGSetMeasureType( *S, value );
   }
   else if ( !strcmp(name,"cycle type") ) {
      return HYPRE_ParAMGSetCycleType( *S, value );
   }
   else if ( !strcmp(name,"ioutdat") ) {
      return HYPRE_ParAMGSetIOutDat( *S, value );
   }
   else if ( !strcmp(name,"debug") ) {
      return HYPRE_ParAMGSetDebugFlag( *S, value );
   }
   else if ( !strcmp(name,"logging") ) {
      return HYPRE_ParAMGSetLogging( *S, value, "Hypre_log_file" );
   }
   else
      printf( "Hypre_ParAMG_SetParameterInt does not recognize name %s\n", name );

   return 1;

} /* end impl_Hypre_ParAMGSetParameterInt */

/* ********************************************************
 * impl_Hypre_ParAMGSetParameterIntArray
 **********************************************************/
int  impl_Hypre_ParAMG_SetParameterIntArray
( Hypre_ParAMG this, char* name, array1int value ) {
   struct Hypre_ParAMG_private_type *HSp = this->d_table;
   HYPRE_Solver *S = HSp->Hsolver;

   if ( !strcmp(name,"num grid sweeps") ) {
      return HYPRE_ParAMGSetNumGridSweeps( *S, value.data );
   }
   else if ( !strcmp(name,"grid relax type") ) {
      return HYPRE_ParAMGSetGridRelaxType( *S, value.data );
   }
   else
      printf(
         "Hypre_ParAMG_SetParameterIntArray does not recognize name %s\n", name );

   return 1;
} /* end impl_Hypre_ParAMGSetParameterIntArray */

/* ********************************************************
 * impl_Hypre_ParAMGSetParameterIntArray2
 **********************************************************/
int  impl_Hypre_ParAMG_SetParameterIntArray2
( Hypre_ParAMG this, char* name, array2int value ) {
   int dim0, dim1, i, j;
   int ** valuepp;
   struct Hypre_ParAMG_private_type *HSp = this->d_table;
   HYPRE_Solver *S = HSp->Hsolver;

   if ( !strcmp(name,"grid relax points") ) {
      /* You need intimate knowledge of the algorithm
         for this to do anything for you) */
      /* This is awkward because we get data in the form of a pseudo-Fortran
         array (continuous data, 2-d indexing) but have to pass it on in the
         form of a int**.
         And - this is a real memory management problem because nobody else
         can (or should) know that a data structure got copied here.  How
         can other codes do their duty to release memory at the right time?
      */
      dim0 = value.upper[0]-value.lower[0];
      dim1 = value.upper[1]-value.lower[1];
      valuepp = hypre_CTAlloc(int *, dim0);
      for ( i=0; i<dim0; ++i ) {
         valuepp[i] = hypre_CTAlloc(int,dim1);
         for ( j=0; j<dim1; ++j )
            valuepp[i][j] = value.data[i+j*dim0];
         /* ... I'm guessing that the Babel standard index ordering is Fortran-
            like, as the array has to be passable to Fortran code */
      };
      return HYPRE_ParAMGSetGridRelaxPoints( *S, valuepp );
   }
   else
      printf(
         "Hypre_ParAMG_SetParameterIntArray2 does not recognize name %s\n", name );
   return 1;
} /* end impl_Hypre_ParAMGSetParameterIntArray2 */

/* ********************************************************
 * impl_Hypre_ParAMGSetParameterString
 **********************************************************/
int  impl_Hypre_ParAMG_SetParameterString
( Hypre_ParAMG this, char* name, char* value ) {
   struct Hypre_ParAMG_private_type *HSp = this->d_table;
   HYPRE_Solver *S = HSp->Hsolver;

   if ( !strcmp(name,"log file name") ) {
      return HYPRE_ParAMGSetLogFileName( *S, value );
   }
   else
      printf( "Hypre_ParAMG_SetParameterString does not recognize name %s\n", name );

   return 1;
} /* end impl_Hypre_ParAMGSetParameterString */

/* ********************************************************
 * impl_Hypre_ParAMGNew
 **********************************************************/
int  impl_Hypre_ParAMG_New(Hypre_ParAMG this, Hypre_MPI_Com comm) {
   struct Hypre_ParAMG_private_type *HSp = this->d_table;
   HYPRE_Solver *S = HSp->Hsolver;

   return HYPRE_ParAMGCreate( S );
} /* end impl_Hypre_ParAMGNew */

/* ********************************************************
 * impl_Hypre_ParAMGConstructor
 **********************************************************/
Hypre_ParAMG  impl_Hypre_ParAMG_Constructor(Hypre_MPI_Com comm) {
   /* declared static; just combines the new and New functions */
   Hypre_ParAMG PS = Hypre_ParAMG_new();
   Hypre_ParAMG_New( PS, comm );
   return PS;

} /* end impl_Hypre_ParAMGConstructor */

/* ********************************************************
 * impl_Hypre_ParAMGSetup
 **********************************************************/
int  impl_Hypre_ParAMG_Setup
(Hypre_ParAMG this, Hypre_LinearOperator A, Hypre_Vector b, Hypre_Vector x) {
   struct Hypre_ParAMG_private_type *HSp = this->d_table;
   HYPRE_Solver *S = HSp->Hsolver;

   Hypre_ParCSRMatrix M;
   Hypre_ParCSRVector VPb, VPx;
   struct Hypre_ParCSRMatrix_private_type * Mp;
   HYPRE_IJMatrix * MIJ;
   hypre_IJMatrix * Mij;
   hypre_ParCSRMatrix *parM;
   struct Hypre_ParCSRVector_private_type * Vbp;
   HYPRE_IJVector * Vb;
   hypre_IJVector *vb;
   hypre_ParVector *b_par;
   struct Hypre_ParCSRVector_private_type * Vxp;
   HYPRE_IJVector * Vx;
   hypre_IJVector * vx;
   hypre_ParVector * x_par;

   M = (Hypre_ParCSRMatrix) Hypre_LinearOperator_castTo( A, "Hypre_ParCSRMatrix" );
   if ( M==NULL ) return 1;
   VPb = (Hypre_ParCSRVector) Hypre_Vector_castTo( b, "Hypre_ParCSRVector" );
   if ( VPb==NULL ) return 1;
   VPx = (Hypre_ParCSRVector) Hypre_Vector_castTo( x, "Hypre_ParCSRVector" );
   if ( VPx==NULL ) return 1;

   Mp = M->d_table;
   MIJ = Mp->Hmat;
   Mij = (hypre_IJMatrix *) (*MIJ);
   parM = hypre_IJMatrixLocalStorage(Mij);

   Vbp = VPb->d_table;
   Vb = Vbp->Hvec;
   vb = (hypre_IJVector *) *Vb;
   b_par = hypre_IJVectorLocalStorage(vb);

   Vxp = VPx->d_table;
   Vx = Vxp->Hvec;
   vx = (hypre_IJVector *) *Vx;
   x_par = hypre_IJVectorLocalStorage(vx);

   this->d_table->Hmatrix = M;

/* it's easier to create the hypre_ParCSR* objects than the HYPRE_ParCSR* objects,
   so we go directly to the hypre-level solver ... 
   return HYPRE_ParAMGSetup( *S, *MA, *Vb, *Vx ); */
   return hypre_ParAMGSetup( *S, parM, b_par, x_par );

} /* end impl_Hypre_ParAMGSetup */

/* ********************************************************
 * impl_Hypre_ParAMGGetConstructedObject
 **********************************************************/
int impl_Hypre_ParAMG_GetConstructedObject
( Hypre_ParAMG this, Hypre_Solver *obj ) {
   *obj = (Hypre_Solver) this;
   return 0;
} /* end impl_Hypre_ParAMGGetConstructedObject */

/* ********************************************************
 * impl_Hypre_ParAMGGetDims
 **********************************************************/
int  impl_Hypre_ParAMG_GetDims(Hypre_ParAMG this, int* m, int* n) {
   Hypre_ParCSRMatrix Hmatrix = this->d_table->Hmatrix;
   return Hypre_ParCSRMatrix_GetDims( Hmatrix, m, n );
} /* end impl_Hypre_ParAMGGetDims */

/* ********************************************************
 * impl_Hypre_ParAMGApply
 *       solves Mx=b for x, where this=M
 **********************************************************/
int  impl_Hypre_ParAMG_Apply(Hypre_ParAMG this, Hypre_Vector b, Hypre_Vector* x) {
   struct Hypre_ParAMG_private_type *HSp = this->d_table;
   HYPRE_Solver *S = HSp->Hsolver;

   Hypre_ParCSRMatrix M;
   Hypre_ParCSRVector VPb, VPx;
   struct Hypre_ParCSRMatrix_private_type * Mp;
   HYPRE_IJMatrix * MIJ;
   hypre_IJMatrix * Mij;
   hypre_ParCSRMatrix *parM;
   struct Hypre_ParCSRVector_private_type * Vbp;
   HYPRE_IJVector * Vb;
   hypre_IJVector *vb;
   hypre_ParVector *b_par;
   struct Hypre_ParCSRVector_private_type * Vxp;
   HYPRE_IJVector * Vx;
   hypre_IJVector * vx;
   hypre_ParVector * x_par;

   VPb = (Hypre_ParCSRVector) Hypre_Vector_castTo( b, "Hypre_ParCSRVector" );
   if ( VPb==NULL ) return 1;
   VPx = (Hypre_ParCSRVector) Hypre_Vector_castTo( *x, "Hypre_ParCSRVector" );
   if ( VPx==NULL ) return 1;

   M = this->d_table->Hmatrix;
   Mp = M->d_table;
   MIJ = Mp->Hmat;
   Mij = (hypre_IJMatrix *) (*MIJ);
   parM = hypre_IJMatrixLocalStorage(Mij);

   Vbp = VPb->d_table;
   Vb = Vbp->Hvec;
   vb = (hypre_IJVector *) *Vb;
   b_par = hypre_IJVectorLocalStorage(vb);

   Vxp = VPx->d_table;
   Vx = Vxp->Hvec;
   vx = (hypre_IJVector *) *Vx;
   x_par = hypre_IJVectorLocalStorage(vx);


   return hypre_ParAMGSolve( *S, parM, b_par, x_par );

} /* end impl_Hypre_ParAMGApply */

/* ********************************************************
 * impl_Hypre_ParAMGGetSystemOperator
 **********************************************************/
int impl_Hypre_ParAMG_GetSystemOperator
( Hypre_ParAMG this, Hypre_LinearOperator *op ) {
   Hypre_ParCSRMatrix mat =  this->d_table->Hmatrix;
   
   *op = (Hypre_LinearOperator)
      Hypre_ParCSRMatrix_castTo( mat, "Hypre_LinearOperator" );
   return 0;
} /* end impl_Hypre_ParAMGGetSystemOperator */

/* ********************************************************
 * impl_Hypre_ParAMGGetResidual
 *  Not implemented.  The residual is not conveniently available, so
 * the way to implement this function would be to recompute it.
 **********************************************************/
int impl_Hypre_ParAMG_GetResidual( Hypre_ParAMG this, Hypre_Vector *resid ) {
   *resid = Hypre_Vector_new(); /* legal type but certain to be incorrect */
   printf( "called Hypre_parAMG_GetResidual, which doesn't work!\n");
   return 1;
} /* end impl_Hypre_ParAMGGetResidual */

/* ********************************************************
 * impl_Hypre_ParAMGGetConvergenceInfo
 *       insert the library code below
 **********************************************************/
int  impl_Hypre_ParAMG_GetConvergenceInfo(Hypre_ParAMG this, char* name, double* value) {
   int ivalue, ierr;
   struct Hypre_ParAMG_private_type *HSp = this->d_table;
   HYPRE_Solver *S = HSp->Hsolver;

   if ( !strcmp(name,"num iterations") ) {
      ierr = HYPRE_ParAMGGetNumIterations( *S, &ivalue );
      *value = ivalue;
      return ierr;
   }
   else if ( !strcmp(name,"final relative residual norm") ) {
      ierr = HYPRE_ParAMGGetFinalRelativeResidualNorm( *S, value );
      return ierr;
   }
   else
      printf("Hypre_ParAMG_GetConvergenceInfo does not recognize name %s\n", name );

   return 1;
} /* end impl_Hypre_ParAMGGetConvergenceInfo */

