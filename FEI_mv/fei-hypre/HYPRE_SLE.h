/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef __HYPRE_SLE_H
#define __HYPRE_SLE_H

#include <iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fstream.h>
#include <math.h>

#include "utilities/utilities.h"

// include matrix-vector files from ISIS++

#include "other/basicTypes.h"
#include "RealArray.h"
#include "IntArray.h"
#include "GlobalIDArray.h"
#include "CommInfo.h"
#include "Map.h"
#include "Vector.h"
#include "Matrix.h"

// include the Hypre package header here

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_parcsr_ls.h"

// includes needed in order to be able to include BASE_SLE.h

#include "other/basicTypes.h"
#include "fei.h"
#include "src/CommBufferDouble.h"
#include "src/CommBufferInt.h"
#include "src/NodePackets.h"
#include "src/BCRecord.h"
#include "src/FieldRecord.h"
#include "src/BlockRecord.h"
#include "src/MultConstRecord.h"
#include "src/PenConstRecord.h"
#include "src/SimpleList.h"
#include "src/NodeRecord.h"
#include "src/SharedNodeRecord.h"
#include "src/SharedNodeBuffer.h"
#include "src/ExternNodeRecord.h"
#include "src/SLE_utils.h"
#include "src/BASE_SLE.h"

// local enumerations

enum HYsolverID { HYPCG, HYGMRES, HYSUPERLU, HYSUPERLUX, HYY12M };
enum HYpreconID { HYDIAGONAL, HYPILUT, HYPARASAILS, HYBOOMERAMG, HYNONE };

// class definition

class HYPRE_SLE : public BASE_SLE {

  public:
    //Constructor
    HYPRE_SLE(MPI_Comm PASSED_COMM_WORLD, int masterRank=0);

    //Destructor
    virtual ~HYPRE_SLE();

    // set parameters associated with solver choice, etc.
    void parameters(int numParams, char **paramStrings);

    void getSolnVectorPtr(Vector** vec) {};
    void getMatrixPtr(Matrix** mat) {};
    void getRHSVectorPtr(Vector** vec) {};
    void setRHSIndex(int index) {};

//==============================================================================
  protected:

//The following functions are associated with the specific tasks of
//using the underlying solver sub-library -- filling data structures,
//selecting and launching solver-methods, etc.
//
    void initLinearAlgebraCore();
    void deleteLinearAlgebraCore();
    void createLinearAlgebraCore(int globalNumEqns, int localStartRow,
                                 int localEndRow, int localStartCol,
                                 int localEndCol);
    void resetMatrixAndVector(double s);
    void matrixConfigure(IntArray* sysRowLengths);
    void sumIntoRHSVector(int num, const int* indices, const double* values);
    void putIntoSolnVector(int num, const int* indices, const double* values);
    double accessSolnVector(int equation);
    void sumIntoSystemMatrix(int row, int numValues, const double* values,
                             const int* scatterIndices);

    void enforceEssentialBC(int* globalEqn, double* alpha,
                                    double* gamma, int len);

    void enforceOtherBC(int* globalEqn, double* alpha, double* beta,
                        double* gamma, int len);

    void matrixLoadComplete();
    void selectSolver(char *name);
    void selectPreconditioner(char *name);
    void launchSolver(int* solveStatus);
    void solveUsingSuperLU(int&);
    void solveUsingSuperLUX(int&);
    void solveUsingY12M(int&);
    void buildReducedSystem();
    void loadSlaveList(int, int*);

private:

    int  getMatrixCSR(int nrows, int nnz, int *ia_ptr, int *ja_ptr, double *a_ptr);

private:

    MPI_Comm        comm;
    int             my_pid;

    int             globalNumEqns_;
    int             StartRow_;
    int             EndRow_;
    int             nConstr, nSlaves, *slaveList;

    HYPRE_IJMatrix  HY_A;
    HYPRE_IJVector  HY_b;
    HYPRE_IJVector  HY_x;
    HYPRE_IJVector  HY_r;

    HYPRE_Solver    pcg_solver;
    HYsolverID      solverID_;
    HYPRE_Solver    pcg_precond;
    HYpreconID      preconID_;

    int             max_iterations;
    double          final_res_norm;
    double          tolerance;

    char            *HYSolverName_;
    char            *HYPrecondName_;
    int             **colIndices;
    int             *rowLengths;

    int             assemble_flag;

    int             amg_coarsen_type;
    int             amg_num_sweeps[4];
    int             amg_relax_type[4];
    double          amg_relax_weight[25];
    double          amg_strong_threshold;
    int             pilut_row_size;
    double          pilut_drop_tol;
    int             pilut_max_nz_per_row;
    int             parasails_nlevels;
    double          parasails_threshold;
    int             superlu_ordering;
    char            superlu_scale[1];
    int             krylov_dim;

friend void fei_hypre_test(int argc, char *argv[]);
friend void fei_hypre_test2(int argc, char *argv[]);

};

#endif

