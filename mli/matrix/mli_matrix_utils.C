/*BHEADER**********************************************************************
 * (c) 2001   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

#include <string.h>
#include <iostream.h>
#include <assert.h>
#include "HYPRE.h"
#include "utilities/utilities.h"
#include "parcsr_mv/parcsr_mv.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "matrix/mli_matrix.h"
#include "util/mli_utils.h"

/***************************************************************************
 * compute triple matrix product function 
 *--------------------------------------------------------------------------*/

int MLI_Matrix_ComputePtAP(MLI_Matrix *Pmat, MLI_Matrix *Amat, 
                           MLI_Matrix **RAPmat_out)
{
   int          ierr;
   char         param_string[200];
   void         *Pmat2, *Amat2, *RAPmat2;
   MLI_Matrix   *RAPmat;
   MLI_Function *func_ptr;

   if ( strcmp(Pmat->getName(),"HYPRE_ParCSR") || 
        strcmp(Amat->getName(),"HYPRE_ParCSR") )
   {
      cout << "MLI_Matrix_computePtAP ERROR - matrix has invalid type.\n";
      exit(1);
   }
   Pmat2 = (void *) Pmat->getMatrix();
   Amat2 = (void *) Amat->getMatrix();
   ierr = MLI_Utils_HypreMatrixComputeRAP(Pmat2,Amat2,&RAPmat2);
   if ( ierr ) cout << "ERROR in MLI_Matrix_ComputePtAP\n";
   sprintf(param_string, "HYPRE_ParCSR");
   func_ptr = new MLI_Function();
   MLI_Utils_HypreMatrixGetDestroyFunc(func_ptr);
   RAPmat = new MLI_Matrix(RAPmat2,param_string,func_ptr);
   delete func_ptr;
   (*RAPmat_out) = RAPmat;
   return 0;
}

/***************************************************************************
 * compute triple matrix product function 
 *--------------------------------------------------------------------------*/

int MLI_Matrix_FormJacobi(MLI_Matrix *Amat, double alpha, MLI_Matrix **Jmat)
{
   int          ierr;
   char         param_string[200];
   void         *A, *J;
   MLI_Function *func_ptr;
   
   if ( strcmp(Amat->getName(),"HYPRE_ParCSR") ) 
   {
      cout << "MLI_Matrix_FormJacobi ERROR - matrix has invalid type.\n";
      exit(1);
   }
   A = (void *) Amat->getMatrix();;
   ierr = MLI_Utils_HypreMatrixFormJacobi(A, alpha, &J);
   if ( ierr ) cout << "ERROR in MLI_Matrix_FormJacobi\n";
   sprintf(param_string, "HYPRE_ParCSR");
   func_ptr = new MLI_Function();
   MLI_Utils_HypreMatrixGetDestroyFunc(func_ptr);
   (*Jmat) = new MLI_Matrix(J,param_string,func_ptr);
   delete func_ptr;
   return ierr;
}

/***************************************************************************
 * compress matrix by block size > 1
 *--------------------------------------------------------------------------*/

int MLI_Matrix_Compress(MLI_Matrix *Amat, int blksize, MLI_Matrix **Amat2)
{
   int          ierr;
   char         param_string[200];
   void         *A, *A2;
   MLI_Function *func_ptr;
   
   if ( strcmp(Amat->getName(),"HYPRE_ParCSR") ) 
   {
      cout << "MLI_Matrix_Compress ERROR - matrix has invalid type.\n";
      exit(1);
   }
   if ( blksize <= 1 )
   {
      cout << "MLI_Matrix_Compress WARNING - blksize <= 1.\n";
      (*Amat2) = NULL;
      return 1;
   }
   A = (void *) Amat->getMatrix();;
   ierr = MLI_Utils_HypreMatrixCompress(A, blksize, &A2);
   if ( ierr ) cout << "ERROR in MLI_Matrix_Compress\n";
   sprintf(param_string, "HYPRE_ParCSR");
   func_ptr = new MLI_Function();
   MLI_Utils_HypreMatrixGetDestroyFunc(func_ptr);
   (*Amat2) = new MLI_Matrix(A2,param_string,func_ptr);
   delete func_ptr;
   return ierr;
}

