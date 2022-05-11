/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * C wrapper functions
 *
 *****************************************************************************/

#include <string.h>
#include <stdlib.h>

#include "cmli.h"
#include "mli.h"
#include "mli_vector.h"
#include "mli_matrix.h"
#include "mli_solver.h"
#include "mli_method.h"
#include "mli_fedata.h"
#include "mli_fedata_utils.h"
#include "mli_mapper.h"

/*****************************************************************************
 * CMLI : constructor 
 *---------------------------------------------------------------------------*/

extern "C" CMLI *MLI_Create( MPI_Comm comm )
{
   MLI  *mli  = new MLI( comm );
   CMLI *cmli = hypre_CTAlloc(CMLI,  1, HYPRE_MEMORY_HOST); 
   cmli->mli_ = mli;
   return cmli;
}

/*****************************************************************************
 * CMLI : destructor 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_Destroy( CMLI *cmli )
{
   int err=0;
   MLI *mli;

   if ( cmli == NULL ) err = 1;
   else
   {
      mli = (MLI *) cmli->mli_;
      if ( mli == NULL ) err = 1;
      else               delete mli;
      free( cmli );
   }
   return err;
}

/*****************************************************************************
 * CMLI : set convergence tolerance 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SetTolerance( CMLI *cmli, double tolerance )
{
   int err=0;
   MLI *mli;

   if ( cmli == NULL ) err = 0;
   else 
   {
      mli = (MLI *) cmli->mli_;
      if ( mli == NULL ) err = 1;
      else               mli->setTolerance( tolerance );
   }
   if ( err ) printf("MLI_SetTolerance ERROR !!\n");
   return err;
}

/*****************************************************************************
 * CMLI : set maximum number of iterations 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SetMaxIterations( CMLI *cmli, int iterations )
{
   int err=0;
   MLI *mli;

   if ( cmli == NULL ) err = 1;
   else
   {
      mli = (MLI *) cmli->mli_;
      if ( mli == NULL ) err = 1;
      else               mli->setMaxIterations( iterations );
   }
   if ( err ) printf("MLI_SetMaxIterations ERROR !!\n");
   return err;
}

/*****************************************************************************
 * CMLI : set number of levels 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SetNumLevels( CMLI *cmli, int num_levels )
{
   int err=0;
   MLI *mli;

   if ( cmli == NULL ) err = 1;
   else
   {
      mli = (MLI *) cmli->mli_;
      if ( mli == NULL ) err = 1;
      else               mli->setNumLevels( num_levels );
   }
   if ( err ) printf("MLI_SetNumLevels ERROR !!\n");
   return err;
}

/*****************************************************************************
 * CMLI : set number of cycles for a given level
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SetCyclesAtLevel( CMLI *cmli, int level, int cycles )
{
   int err=0;
   MLI *mli;

   if ( cmli == NULL ) err = 1;
   else
   {
      mli = (MLI *) cmli->mli_;
      if ( mli == NULL ) err = 1;
      else               mli->setCyclesAtLevel( level, cycles );
   }
   if ( err ) printf("MLI_SetCyclesAtLevel ERROR !!\n");
   return err;
}

/*****************************************************************************
 * CMLI : set system matrix
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SetSystemMatrix( CMLI *cmli, int level, CMLI_Matrix *CAmat )
{
   int        err=0;
   MLI        *mli;
   MLI_Matrix *matrix;

   if ( cmli == NULL || CAmat == NULL ) err = 1;
   else
   {
      mli    = (MLI *) cmli->mli_;
      matrix = (MLI_Matrix *) CAmat->matrix_;
      if ( mli == NULL || CAmat == NULL ) err = 1;
      else                                mli->setSystemMatrix(level, matrix);
      CAmat->owner_ = 0;
   }
   if ( err ) printf("MLI_SetSystemMatrix ERROR !!!\n");
   return err;
}

/*****************************************************************************
 * CMLI : set restriction matrix
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SetRestriction( CMLI *cmli, int level, CMLI_Matrix *CRmat )
{
   int        err=0;
   MLI        *mli;
   MLI_Matrix *matrix;

   if ( cmli == NULL || CRmat == NULL ) err = 1;
   else
   {
      mli    = (MLI *) cmli->mli_;
      matrix = (MLI_Matrix *) CRmat->matrix_;
      if ( mli == NULL || CRmat == NULL ) err = 1;
      else                                mli->setRestriction(level, matrix);
      CRmat->owner_ = 0;
   }
   if ( err ) printf("MLI_SetRestriction ERROR !!\n");
   return err;
}

/*****************************************************************************
 * CMLI : set prolongation 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SetProlongation( CMLI *cmli, int level, CMLI_Matrix *CPmat )
{
   int        err=0;
   MLI        *mli;
   MLI_Matrix *matrix;

   if ( cmli == NULL || CPmat == NULL ) err = 1;
   else
   {
      mli    = (MLI *) cmli->mli_;
      matrix = (MLI_Matrix *) CPmat->matrix_;
      if ( mli == NULL || CPmat == NULL ) err = 1;
      else                                mli->setProlongation(level, matrix);
      CPmat->owner_ = 0;
   }
   if ( err ) printf("MLI_SetProlongation ERROR !!\n");
   return err;
}

/*****************************************************************************
 * CMLI : set finite element information object 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SetFEData( CMLI *cmli, int level, CMLI_FEData *cfedata,
                              CMLI_Mapper *cmapper )
{
   int        err=0;
   MLI        *mli;
   MLI_FEData *fedata;
   MLI_Mapper *mapper;

   if ( cmli == NULL || cfedata == NULL ) err = 1;
   else
   {
      mli    = (MLI *) cmli->mli_;
      fedata = (MLI_FEData *) cfedata->fedata_;
      mapper = (MLI_Mapper *) cmapper->mapper_;
      if (mli == NULL || fedata == NULL) err = 1;
      else                               mli->setFEData(level,fedata,mapper);
      cfedata->owner_ = 0;
   }
   if ( err ) printf("MLI_SetFEData ERROR !!\n");
   return err;
}

/*****************************************************************************
 * CMLI : set smoother 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SetSmoother( CMLI *cmli, int level, int pre_post, 
                                CMLI_Solver *csolver )
{
   int        err=0;
   MLI        *mli;
   MLI_Solver *solver;

   if ( cmli == NULL || csolver == NULL ) err = 1;
   else
   {
      mli    = (MLI *) cmli->mli_;
      solver = (MLI_Solver *) csolver->solver_;
      if (mli == NULL || solver == NULL) err = 1;
      else
         mli->setSmoother(level,pre_post,solver);
      csolver->owner_ = 0;
   }
   if ( err ) printf("MLI_SetSmoother ERROR !!\n");
   return err;
}

/*****************************************************************************
 * CMLI : set coarse solver 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SetCoarseSolve( CMLI *cmli, CMLI_Solver *csolver )
{
   int        err=0;
   MLI        *mli;
   MLI_Solver *solver;

   if ( cmli == NULL || csolver == NULL ) err = 1;
   else
   {
      mli    = (MLI *) cmli->mli_;
      solver = (MLI_Solver *) csolver->solver_;
      if ( mli == NULL || solver == NULL ) err = 1;
      else                                 mli->setCoarseSolve( solver );
      csolver->owner_ = 0;
   }
   if ( err ) printf("MLI_SetCoarseSolve ERROR !!\n");
   return err;
}

/*****************************************************************************
 * set multilevel method 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SetMethod( CMLI *cmli, CMLI_Method *method_data )
{
   int        err=0;
   MLI        *mli;
   MLI_Method *data;

   if ( cmli == NULL || method_data == NULL ) err = 1;
   else
   {
      mli = (MLI *) cmli->mli_;
      data = (MLI_Method *) method_data->method_;
      if ( mli == NULL || data == NULL ) err = 1;
      else                               mli->setMethod( data );
      method_data->owner_ = 0;
   }
   if ( err ) printf("MLI_SetMethod ERROR !!\n");
   return err;
}


/*****************************************************************************
 * set up the multilevel method
 *---------------------------------------------------------------------------*/

extern "C" int MLI_Setup( CMLI *cmli )
{
   int err=0;
   MLI *mli;

   if ( cmli == NULL ) err = 1;
   else
   {
      mli = (MLI *) cmli->mli_;
      if ( mli == NULL ) err = 1;
      else               mli->setup();
   }
   if ( err ) printf("MLI_Setup ERROR !!\n");
   return err;
}

/*****************************************************************************
 * perform one multilevel cycle
 *---------------------------------------------------------------------------*/

extern "C" int MLI_Cycle( CMLI *cmli, CMLI_Vector *csol, CMLI_Vector *crhs )
{
   int        err=0;
   MLI        *mli;
   MLI_Vector *sol, *rhs;

   if ( cmli == NULL || csol == NULL || crhs == NULL ) err = 1;
   else
   {
      mli = (MLI *) cmli->mli_;
      sol = (MLI_Vector *) csol->vector_;
      rhs = (MLI_Vector *) crhs->vector_;
      if (mli == NULL || sol == NULL || rhs == NULL) err = 1;
      else                                           mli->cycle(sol, rhs);
      csol->owner_ = 0;
      crhs->owner_ = 0;
   }
   if ( err ) printf("MLI_Cycle ERROR !!\n");
   return err;
}

/*****************************************************************************
 * perform multilevel cycles until convergence
 *---------------------------------------------------------------------------*/

extern "C" int MLI_Solve( CMLI *cmli, CMLI_Vector *csol, CMLI_Vector *crhs )
{
   int        err=0;
   MLI        *mli;
   MLI_Vector *sol, *rhs;

   if ( cmli == NULL || csol == NULL || crhs == NULL ) err = 1;
   else
   {
      mli = (MLI *) cmli->mli_;
      sol = (MLI_Vector *) csol->vector_;
      rhs = (MLI_Vector *) crhs->vector_;
      if (mli == NULL || sol == NULL || rhs == NULL) err = 1;
      else                                           mli->solve(sol, rhs);
      csol->owner_ = 0;
      crhs->owner_ = 0;
   }
   if ( err ) printf("MLI_Solve ERROR !!\n");
   return err;
}

/*****************************************************************************
 * set output levels 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SetOutputLevel( CMLI *cmli, int level )
{
   int err=0;
   MLI *mli;

   if ( cmli == NULL ) err = 1;
   else
   {
      mli = (MLI *) cmli->mli_;
      if ( mli == NULL ) err = 1; 
      else               mli->setOutputLevel( level );
   }
   if ( err ) printf("MLI_SetOutputLevel ERROR !!\n");
   return err;
}

/*****************************************************************************
 * print MLI internal information
 *---------------------------------------------------------------------------*/

extern "C" int MLI_Print( CMLI *cmli )
{
   int err=0;
   MLI *mli;

   if ( cmli == NULL ) err = 1;
   else
   {
      mli = (MLI *) cmli->mli_;
      if ( mli == NULL ) err = 1;
      else               mli->print();
   }
   if ( err ) printf("MLI_Print ERROR !!\n");
   return err;
}

/*****************************************************************************
 * print MLI timing information
 *---------------------------------------------------------------------------*/

extern "C" int MLI_PrintTiming( CMLI *cmli )
{
   int err=0;
   MLI *mli;

   if ( cmli == NULL ) err = 1;
   else
   {
      mli = (MLI *) cmli->mli_;
      if ( mli == NULL ) err = 1;
      else               mli->printTiming();
   }
   if ( err ) printf("MLI_PrintTiming ERROR !!\n");
   return err;
}

/*****************************************************************************
 * create a "C" finite element data object
 *---------------------------------------------------------------------------*/

extern "C" CMLI_FEData *MLI_FEDataCreate(MPI_Comm comm, void *fedata, char *name)
{
   int         mypid;
   MLI_FEData  *mli_fedata;
   CMLI_FEData *cmli_fedata;

   MPI_Comm_rank(comm, &mypid);
   mli_fedata  = new MLI_FEData( comm );
   cmli_fedata = hypre_CTAlloc(CMLI_FEData,  1, HYPRE_MEMORY_HOST); 
   cmli_fedata->fedata_ = (void *) mli_fedata;
   cmli_fedata->owner_  = 1;
   return cmli_fedata;
}

/*****************************************************************************
 * destroy a "C" FEData object
 *---------------------------------------------------------------------------*/

extern "C" int MLI_FEDataDestroy( CMLI_FEData *cfedata )
{
   int        err=0;
   MLI_FEData  *fedata;

   if ( cfedata == NULL ) err = 1;
   else
   {
      fedata = (MLI_FEData *) cfedata->fedata_;
      if ( fedata == NULL ) err = 1;
      else if ( cfedata->owner_ ) delete fedata;
      free( cfedata );
   }
   return err;
}

/*****************************************************************************
 * create element node matrix from FEData 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_FEDataGetElemNodeMatrix( CMLI_FEData *cfedata,
                                            MPI_Comm comm, void ** mat )
{
   int        err=0;
   MLI_FEData *fedata;
   MLI_Matrix *elemNodeMat;

   if ( cfedata == NULL ) err = 1;
   else
   {
      fedata = (MLI_FEData *) cfedata->fedata_;
      if ( fedata == NULL ) err = 1;
      else 
      {
         MLI_FEDataConstructElemNodeMatrix(comm, fedata, &elemNodeMat);
         (*mat) = elemNodeMat;
      }
   }
   return err;
}

/*****************************************************************************
 * create element face matrix from FEData 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_FEDataGetElemFaceMatrix( CMLI_FEData *cfedata,
                                            MPI_Comm comm, void ** mat )
{
   int        err=0;
   MLI_FEData *fedata;
   MLI_Matrix *elemFaceMat;

   if ( cfedata == NULL ) err = 1;
   else
   {
      fedata = (MLI_FEData *) cfedata->fedata_;
      if ( fedata == NULL ) err = 1;
      else 
      {
         MLI_FEDataConstructElemFaceMatrix(comm, fedata, &elemFaceMat);
         (*mat) = elemFaceMat;
      }
   }
   return err;
}

/*****************************************************************************
 * create face node matrix from FEData 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_FEDataGetFaceNodeMatrix( CMLI_FEData *cfedata,
                                            MPI_Comm comm, void ** mat )
{
   int        err=0;
   MLI_FEData *fedata;
   MLI_Matrix *faceNodeMat;

   if ( cfedata == NULL ) err = 1;
   else
   {
      fedata = (MLI_FEData *) cfedata->fedata_;
      if ( fedata == NULL ) err = 1;
      else 
      {
         MLI_FEDataConstructFaceNodeMatrix(comm, fedata, &faceNodeMat);
         (*mat) = faceNodeMat;
      }
   }
   return err;
}

/*****************************************************************************
 * create node element matrix from FEData 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_FEDataGetNodeElemMatrix( CMLI_FEData *cfedata,
                                            MPI_Comm comm, void ** mat )
{
   int        err=0;
   MLI_FEData *fedata;
   MLI_Matrix *nodeElemMat;

   if ( cfedata == NULL ) err = 1;
   else
   {
      fedata = (MLI_FEData *) cfedata->fedata_;
      if ( fedata == NULL ) err = 1;
      else 
      {
         MLI_FEDataConstructNodeElemMatrix(comm, fedata, &nodeElemMat);
         (*mat) = nodeElemMat;
      }
   }
   return err;
}

/*****************************************************************************
 * create face element matrix from FEData 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_FEDataGetFaceElemMatrix( CMLI_FEData *cfedata,
                                            MPI_Comm comm, void ** mat )
{
   int        err=0;
   MLI_FEData *fedata;
   MLI_Matrix *faceElemMat;

   if ( cfedata == NULL ) err = 1;
   else
   {
      fedata = (MLI_FEData *) cfedata->fedata_;
      if ( fedata == NULL ) err = 1;
      else 
      {
         MLI_FEDataConstructFaceElemMatrix(comm, fedata, &faceElemMat);
         (*mat) = faceElemMat;
      }
   }
   return err;
}

/*****************************************************************************
 * create node face matrix from FEData 
 *---------------------------------------------------------------------------*/

extern "C" int MLI_FEDataGetNodeFaceMatrix( CMLI_FEData *cfedata,
                                            MPI_Comm comm, void ** mat )
{
   int        err=0;
   MLI_FEData *fedata;
   MLI_Matrix *nodeFaceMat;

   if ( cfedata == NULL ) err = 1;
   else
   {
      fedata = (MLI_FEData *) cfedata->fedata_;
      if ( fedata == NULL ) err = 1;
      else 
      {
         MLI_FEDataConstructNodeFaceMatrix(comm, fedata, &nodeFaceMat);
         (*mat) = nodeFaceMat;
      }
   }
   return err;
}

/*****************************************************************************
 * create a "C" matrix object
 *---------------------------------------------------------------------------*/

extern "C" CMLI_Matrix *MLI_MatrixCreate(void *matrix, char *name, 
                                         MLI_Function *func)
{
   MLI_Matrix  *mli_matrix;
   CMLI_Matrix *cmli_matrix;

   mli_matrix  = new MLI_Matrix( matrix, name, func );
   cmli_matrix = hypre_CTAlloc(CMLI_Matrix,  1, HYPRE_MEMORY_HOST);
   cmli_matrix->matrix_ = (void *) mli_matrix;
   cmli_matrix->owner_ = 1;
   return cmli_matrix;
}

/*****************************************************************************
 * destroy a "C" matrix object
 *---------------------------------------------------------------------------*/

extern "C" int MLI_MatrixDestroy( CMLI_Matrix *cmatrix )
{
   int        err=0;
   MLI_Matrix *matrix;

   if ( cmatrix == NULL ) err = 1;
   else
   {
      matrix = (MLI_Matrix *) cmatrix->matrix_;
      if ( matrix == NULL ) err = 1;
      else if ( cmatrix->owner_ ) delete matrix;
      free( cmatrix );
   }
   return err;
}

/*****************************************************************************
 * create a "C" vector object
 *---------------------------------------------------------------------------*/

CMLI_Vector *MLI_VectorCreate(void *vector,const char *name, 
                              MLI_Function *func)
{
   MLI_Vector  *mli_vector  = new MLI_Vector( vector, name, func );
   CMLI_Vector *cmli_vector = hypre_CTAlloc(CMLI_Vector,  1, HYPRE_MEMORY_HOST);
   cmli_vector->vector_ = (void *) mli_vector;
   cmli_vector->owner_ = 1;
   return cmli_vector;
}

/*****************************************************************************
 * destroy a "C" vector object
 *---------------------------------------------------------------------------*/

extern "C" int MLI_VectorDestroy( CMLI_Vector *cvector )
{
   int        err=0;
   MLI_Vector *mli_vector;

   if ( cvector == NULL ) err = 1;
   else
   {
      mli_vector = (MLI_Vector *) cvector->vector_;
      if ( mli_vector == NULL ) err = 1;
      else if ( cvector->owner_ ) delete mli_vector;
      free( cvector );
   }
   return err;
}

/*****************************************************************************
 * create a "C" solver object
 *---------------------------------------------------------------------------*/

extern "C" CMLI_Solver *MLI_SolverCreate(char *name)
{
   MLI_Solver  *mli_solver;
   CMLI_Solver *cmli_solver = hypre_CTAlloc(CMLI_Solver, 1, HYPRE_MEMORY_HOST);

   mli_solver = MLI_Solver_CreateFromName( name );
   cmli_solver->solver_ = (void *) mli_solver;
   cmli_solver->owner_  = 1;
   return cmli_solver;
}

/*****************************************************************************
 * destroy a "C" solver object
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SolverDestroy( CMLI_Solver *csolver )
{
   int        err=0;
   MLI_Solver *mli_solver;

   if ( csolver == NULL ) err = 1;
   else
   {
      mli_solver = (MLI_Solver *) csolver->solver_;
      if ( mli_solver == NULL ) err = 1;
      else if ( csolver->owner_ ) delete mli_solver;
      free( csolver );
   }
   return err;
}

/*****************************************************************************
 * set solver parameters
 *---------------------------------------------------------------------------*/

extern "C" int MLI_SolverSetParams(CMLI_Solver *solver, char *param_string,
                                   int argc, char **argv)
{
   int        err=0;
   MLI_Solver *mli_solver;

   if ( solver == NULL ) err = 1;
   else
   {
      mli_solver = (MLI_Solver *) solver->solver_;
      if ( mli_solver == NULL ) err = 1;
      else                      mli_solver->setParams(param_string,argc,argv);
   }
   if ( err ) printf("MLI_SolverSetParams ERROR !!\n");
   return err;
}

/*****************************************************************************
 * create a "C" mapper object
 *---------------------------------------------------------------------------*/

extern "C" CMLI_Mapper *MLI_MapperCreate()
{
   MLI_Mapper  *mli_mapper;
   CMLI_Mapper *cmli_mapper = hypre_CTAlloc(CMLI_Mapper, 1, HYPRE_MEMORY_HOST);

   mli_mapper = new MLI_Mapper();
   cmli_mapper->mapper_ = (void *) mli_mapper;
   cmli_mapper->owner_  = 1;
   return cmli_mapper;
}

/*****************************************************************************
 * destroy a "C" mapper object
 *---------------------------------------------------------------------------*/

extern "C" int MLI_MapperDestroy( CMLI_Mapper *cmapper )
{
   int        err=0;
   MLI_Mapper *mli_mapper;

   if ( cmapper == NULL ) err = 1;
   else
   {
      mli_mapper = (MLI_Mapper *) cmapper->mapper_;
      if ( mli_mapper == NULL ) err = 1;
      else if ( cmapper->owner_ ) delete mli_mapper;
      free( cmapper );
   }
   return err;
}

/*****************************************************************************
 * set mapper parameters
 *---------------------------------------------------------------------------*/

extern "C" int MLI_MapperSetParams(CMLI_Mapper *mapper, char *param_string,
                                   int argc, char **argv)
{
   int        err=0;
   MLI_Mapper *mli_mapper;

   if ( mapper == NULL ) err = 1;
   else
   {
      mli_mapper = (MLI_Mapper *) mapper->mapper_;
      if ( mli_mapper == NULL ) err = 1;
      else                      mli_mapper->setParams(param_string,argc,argv);
   }
   if ( err ) printf("MLI_MapperSetParams ERROR !!\n");
   return err;
}

/*****************************************************************************
 * create a "C" method object
 *---------------------------------------------------------------------------*/

extern "C" CMLI_Method *MLI_MethodCreate(char *name, MPI_Comm comm)
{
   int         err=0;
   MLI_Method  *mli_method;
   CMLI_Method *cmli_method;

   mli_method = MLI_Method_CreateFromName(name, comm);
   cmli_method = hypre_CTAlloc(CMLI_Method,  1, HYPRE_MEMORY_HOST);
   if ( mli_method == NULL || cmli_method == NULL ) err = 1;
   else
   {
      cmli_method->method_ = (void *) mli_method;
      cmli_method->owner_  = 1;
   }
   if ( err ) printf("MLI_MethodCreate ERROR !!\n");
   return cmli_method;
}

/*****************************************************************************
 * destroy a "C" method object
 *---------------------------------------------------------------------------*/

extern "C" int MLI_MethodDestroy( CMLI_Method *cmli_method )
{
   int        err=0;
   MLI_Method *mli_method;

   if ( cmli_method == NULL ) err = 1;
   else
   {
      if ( cmli_method->owner_ != 0 )
      {
         mli_method = (MLI_Method *) cmli_method->method_;
         if ( mli_method == NULL ) err = 1;
         else                      delete mli_method;
      }
      free( cmli_method );
   }
   if ( err ) printf("MLI_MethodDestroy ERROR !!\n");
   return err;
}

/*****************************************************************************
 * set method parameters
 *---------------------------------------------------------------------------*/

extern "C" int MLI_MethodSetParams(CMLI_Method *cmethod, char *param_string, 
                                   int argc, char **argv)
{
   int        err=0;
   MLI_Method *mli_method;

   if ( cmethod == NULL ) err = 1;
   else
   {
      mli_method = (MLI_Method *) cmethod->method_;
      if (mli_method == NULL) err = 1;
      else                    mli_method->setParams(param_string, argc, argv);
   }
   if ( err ) printf("MLI_MethodSetParams ERROR !!\n");
   return err;
}

