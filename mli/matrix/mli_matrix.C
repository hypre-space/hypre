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
#include "utilities/utilities.h"
#include "HYPRE.h"
#include "parcsr_mv/parcsr_mv.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "matrix/mli_matrix.h"
#include "util/mli_utils.h"

/***************************************************************************
 * constructor function for the MLI_Matrix 
 *--------------------------------------------------------------------------*/

MLI_Matrix::MLI_Matrix(void *in_matrix,char *in_name, MLI_Function *func)
{
#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_Matrix::MLI_Matrix : " << in_name << endl;
   cout.flush();
#endif
   matrix = in_matrix;
   if ( func != NULL ) destroy_func = (int (*)(void *)) func->func_;
   else                    destroy_func = NULL;
   strncpy(name, in_name, 100);
   g_nrows_ = -1;
   max_nnz_ = -1;
   min_nnz_ = -1;
   tot_nnz_ = -1;
   max_val_ = 0.0;
   min_val_ = 0.0;
}

/***************************************************************************
 * destructor function for the MLI_Matrix 
 *--------------------------------------------------------------------------*/

MLI_Matrix::~MLI_Matrix()
{
#ifdef MLI_DEBUG
   cout << "MLI_Matrix::~MLI_Matrix : " << name << endl;
   cout.flush();
#endif
   if ( matrix != NULL && destroy_func != NULL ) destroy_func(matrix);
   matrix       = NULL;
   destroy_func = NULL;
}

/***************************************************************************
 * apply function ( vec3 = alpha * Matrix * vec1 + beta * vec2)
 *--------------------------------------------------------------------------*/

int MLI_Matrix::apply(double alpha, MLI_Vector *vec1, double beta, 
                      MLI_Vector *vec2, MLI_Vector *vec3)
{
   int                status;
   char               *vname;
   hypre_ParVector    *hypreV1, *hypreV2, *hypreV3;
   hypre_ParCSRMatrix *hypreA = (hypre_ParCSRMatrix *) matrix;

#ifdef MLI_DEBUG_DETAILED
   cout << "MLI_Matrix::MLI_Matrix apply : " << name << endl;
   cout.flush();
#endif

   /* -----------------------------------------------------------------------
    * error checking
    * ----------------------------------------------------------------------*/

   if ( !strcmp(name, "HYPRE_ParCSR") && !strcmp(name, "HYPRE_ParCSRT") )
   {
      cout << "MLI_Matrix::apply ERROR : matrix not HYPRE_ParCSR." << endl;
      exit(1);
   }
   vname = vec1->getName();
   if ( strcmp(vname, "HYPRE_ParVector") )
   {
      cout << "MLI_Matrix::apply ERROR : vec1 not HYPRE_ParVector." << endl;
      cout << "MLI_Matrix::vec1 of type = " << vname << endl;
      exit(1);
   }
   if ( vec2 != NULL )
   {
      vname = vec2->getName();
      if ( strcmp(vname, "HYPRE_ParVector") )
      {
         cout << "MLI_Matrix::apply ERROR : vec2 not HYPRE_ParVector." << endl;
         exit(1);
      }
   }
   vname = vec3->getName();
   if ( strcmp(vname, "HYPRE_ParVector") )
   {
      cout << "MLI_Matrix::apply ERROR : vec3 not HYPRE_ParVector." << endl;
      exit(1);
   }

   /* -----------------------------------------------------------------------
    * fetch matrix and vectors; and then operate
    * ----------------------------------------------------------------------*/

   hypreA  = (hypre_ParCSRMatrix *) matrix;
   hypreV1 = (hypre_ParVector *) vec1->getVector();
   hypreV3 = (hypre_ParVector *) vec3->getVector();
   if ( vec2 != NULL )
   {
      hypreV2 = (hypre_ParVector *) vec2->getVector();
      status  = hypre_ParVectorCopy( hypreV2, hypreV3 );
      status += hypre_ParVectorScale( beta, hypreV3 );
   }
   else status = hypre_ParVectorSetConstantValues( hypreV3, 0.0e0 );

   if ( !strcmp(name, "HYPRE_ParCSR" ) )
   {
      status += hypre_ParCSRMatrixMatvec(alpha,hypreA,hypreV1,beta,hypreV3);
   }
   else
   {
      status += hypre_ParCSRMatrixMatvecT(alpha,hypreA,hypreV1,beta,hypreV3);
   }
   return status;
}

/******************************************************************************
 * create a vector from information of this matrix 
 *---------------------------------------------------------------------------*/

MLI_Vector *MLI_Matrix::createVector()
{
   int                i, mypid, nprocs, start_row, end_row, global_nrows;
   int                ierr, *partitioning;
   char               param_string[100];
   MPI_Comm           comm;
   HYPRE_ParVector    new_vec;
   hypre_ParCSRMatrix *hypreA;
   HYPRE_IJVector     IJvec;
   MLI_Vector         *mli_vec;
   MLI_Function       *func_ptr;

   if ( strcmp( name, "HYPRE_ParCSR" ) )
   {
      cout << "MLI_Matrix::createVector ERROR - matrix has invalid type.\n";
      exit(1);
   }
   hypreA = (hypre_ParCSRMatrix *) matrix;
   comm = hypre_ParCSRMatrixComm(hypreA);
   MPI_Comm_rank(comm, &mypid);
   MPI_Comm_size(comm, &nprocs);
   HYPRE_ParCSRMatrixGetRowPartitioning((HYPRE_ParCSRMatrix)hypreA,&partitioning);
   global_nrows = partitioning[nprocs];
   start_row    = partitioning[mypid];
   end_row      = partitioning[mypid+1];
   free( partitioning );
   ierr = HYPRE_IJVectorCreate(comm, start_row, end_row-1, &IJvec);
   ierr = HYPRE_IJVectorSetObjectType(IJvec, HYPRE_PARCSR);
   ierr = HYPRE_IJVectorInitialize(IJvec);
   ierr = HYPRE_IJVectorAssemble(IJvec);
   ierr = HYPRE_IJVectorGetObject(IJvec, (void **) &new_vec);
   ierr = HYPRE_IJVectorSetObjectType(IJvec, -1);
   ierr = HYPRE_IJVectorDestroy(IJvec);
   HYPRE_ParVectorSetConstantValues( new_vec, 0.0 );
   sprintf( param_string, "HYPRE_ParVector" );
   func_ptr = new MLI_Function();
   MLI_Utils_HypreVectorGetDestroyFunc(func_ptr); 
   mli_vec = new MLI_Vector((void*) new_vec, param_string, func_ptr);
   delete func_ptr;
   return mli_vec;
}

/******************************************************************************
 * create a vector from information of this matrix 
 *---------------------------------------------------------------------------*/

int MLI_Matrix::getMatrixInfo(char *param_string, int &int_param, 
                              double &dble_param)
{
   int      mat_info[4];
   double   val_info[2];

   if ( !strcmp(name, "HYPRE_ParCSR") && !strcmp(name, "HYPRE_ParCSRT") )
   {
      cout << "MLI_Matrix::getInfo ERROR : matrix not HYPRE_ParCSR." << endl;
      int_param  = -1;
      dble_param = 0.0;
      return 1;
   }
   if ( g_nrows_ < 0 )
   {
      MLI_Utils_HypreMatrixGetInfo(matrix, mat_info, val_info);
      g_nrows_ = mat_info[0];
      max_nnz_ = mat_info[1];
      min_nnz_ = mat_info[2];
      tot_nnz_ = mat_info[3];
      max_val_ = val_info[0];
      min_val_ = val_info[1];
   }
   int_param  = 0;
   dble_param = 0.0;
   if      ( !strcmp( param_string, "nrows" )) int_param  = g_nrows_;
   else if ( !strcmp( param_string, "maxnnz")) int_param  = max_nnz_;
   else if ( !strcmp( param_string, "minnnz")) int_param  = min_nnz_;
   else if ( !strcmp( param_string, "totnnz")) int_param  = tot_nnz_;
   else if ( !strcmp( param_string, "maxval")) dble_param = max_val_;
   else if ( !strcmp( param_string, "minval")) dble_param = min_val_;
   return 0;
}

/******************************************************************************
 * print a matrix
 *---------------------------------------------------------------------------*/

int MLI_Matrix::print(char *filename)
{
   int      mypid, nprocs, icol, isum[4], ibuf[4], *partition, this_nnz;
   int      local_nrows, irow, rownum, rowsize, *colind, startrow;
   double   *colval, dsum[2], dbuf[2];

   if ( !strcmp(name, "HYPRE_ParCSR") && !strcmp(name, "HYPRE_ParCSRT") )
   {
      cout << "MLI_Matrix::print ERROR : matrix not HYPRE_ParCSR." << endl;
      return 1;
   }
   MLI_Utils_HypreMatrixPrint((void *) matrix, filename);
   return 0;
}

