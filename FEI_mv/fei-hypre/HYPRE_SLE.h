/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include <HYPRE_config.h>

#include "mpi.h"
#include "HYPRE_IJ_mv.h"

#ifndef __HYPRE_SLE_H
#define __HYPRE_SLE_H


class HYPRE_SLE : public BASE_SLE {

  public:
    //Constructor
    HYPRE_SLE(MPI_Comm PASSED_COMM_WORLD, int masterRank=0);

    //Destructor
    virtual ~HYPRE_SLE();

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


private:

    MPI_Comm        comm;

    int first_row;
    int last_row;

    HYPRE_IJMatrix  A;
    HYPRE_IJVector  b;
    HYPRE_IJVector  x;

    HYPRE_Solver pcg_solver;
    HYPRE_Solver pcg_precond;

    int num_iterations;
    double final_res_norm;

friend void fei_hypre_test(int argc, char *argv[]);

};

#endif

