//
// This is a simple program to exercise the FEI, for the purposes of
// unit testing, code tuning and scaling studies.
//
// The input file for this program should provide five integers:
//   L -- the global length of a 2D square
//   W -- the width of the square
//   DOF -- the number of scalars (the 'field' cardinality) per node
//
//   outputLevel -- 0: no output
//               -- 1: some output
//               -- 2: lots of output
//
//   debugOutput -- 0: don't create debug files
//               -- 1: create debug files
//
// Alan Williams 6-20-99
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <fstream.h>

#include "fei-isis.h"

#include "Poisson_Elem.h"
#include "PoissonData.h"

//==============================================================================
//
//prototypes for the support functions that will do misc. tasks associated
//with calling PoissonData and exercising an FEI implementation.
//

void initialize_mpi(int argc, char** argv, int& localProc, int& numProcs);

void print_args(int argc, char** argv);

void read_and_broadcast_input(char* fileName, int& L, int& W, int& DOF,
                              int numProcs, int& outputLevel, int& debugOutput);

void receive_input(int& L, int& W, int& DOF,
                   int& outputLevel, int& debugOutput);

void setup_IDs(int& numMatrices, int*& matrixIDs,
               int*& numRHSs, int**& rhsIDs, int outputLevel);

void setup_scalars(int numMatrices, int* matrixIDs, int* numRHSs, int** rhsIDs,
                   int& numMatScalars, double*& matScalars,
                   int& numRHSScalars, int*& rhsScaleIDs, double*& rhsScalars);

void delete_scalars(int& numMatScalars, double*& matScalars,
                    int& numRHSScalars, int*& rhsScaleIDs, double*& rhsScalars);

void load_FEI_parameters(FEI& linearSystem, int outputLevel);

void put_FEI_block_solution(PoissonData& poissonData, FEI& linearSystem);

void save_FEI_block_solution(PoissonData& poissonData, FEI& linearSystem,
                             int solve, int numProcs, int localProc);

void delete_IDs(int& numMatrices, int*& matrixIDs,
                int*& numRHSs, int**& rhsIDs);

void print_solve_status(FEI& linearSystem,
                        double solve_time, double total_time);
//==============================================================================

//==============================================================================
int main(int argc, char** argv){

    int numProcs = 1, localProc = 0, masterProc = 0;
    int L = 0, W = 0, DOF = 0, outputLevel = 0, debugOutput = 0;

    initialize_mpi(argc, argv, localProc, numProcs);

    double total_time0 = MPI_Wtime();

    if ((masterProc == localProc)&&(outputLevel>0)) print_args(argc, argv);

    if (masterProc == localProc) 
        read_and_broadcast_input(argv[1], L, W, DOF, numProcs,
                                 outputLevel, debugOutput);
    else
        receive_input(L, W, DOF, outputLevel, debugOutput);

    PoissonData poissonData(L, W, DOF, numProcs, localProc, outputLevel);

//    if (outputLevel>0) poissonData.print_storage_summary();

    FEI* fei_isis = ISIS_Builder::FEIBuilder(MPI_COMM_WORLD, 0);

    if (debugOutput>0) {
        char *param = new char[32];
        sprintf(param, "debugOutput .");
        fei_isis->parameters(1, &param);
        delete [] param;
    }

    int numMatrices = 1;
    int* matrixIDs = NULL;
    int* numRHSs = NULL;
    int** rhsIDs = NULL;

    setup_IDs(numMatrices, matrixIDs, numRHSs, rhsIDs, outputLevel);

    int numMatScalars = 0;
    double* matScalars = NULL;
    int numRHSScalars = 0;
    int* rhsScaleIDs = NULL;
    double* rhsScalars = NULL;

    setup_scalars(numMatrices, matrixIDs, numRHSs, rhsIDs,
                  numMatScalars, matScalars,
                  numRHSScalars, rhsScaleIDs, rhsScalars);

    int solveType = 2;
    int numElemBlocks = 1; // 1 per processor

    if (outputLevel>0) cout << "initSolveStep" << endl;
    fei_isis->initSolveStep(numElemBlocks, solveType,
                            numMatrices, matrixIDs, numRHSs, rhsIDs);

    poissonData.set_FEI_instance(*fei_isis);

    int i;

    poissonData.do_init_phase();
 
    for(i=0; i<numMatrices; i++) {
        if (outputLevel>0) cout << "setMatrixID " << matrixIDs[i] << endl;
        fei_isis->setMatrixID(matrixIDs[i]);

        poissonData.do_begin_load_block();

            poissonData.do_load_stiffness();

            for(int j=0; j<numRHSs[i]; j++) {
                if (outputLevel>0) cout << "setRHSID " << rhsIDs[i][j] << endl;
                fei_isis->setRHSID(rhsIDs[i][j]);

                poissonData.do_load_rhs();
            }

        poissonData.do_end_load_block();
    }

    load_FEI_parameters(*fei_isis, outputLevel);

    //put_FEI_block_solution(poissonData, *fei_isis);

    for(i=0; i<numMatrices; i++) {
        if (outputLevel>0) cout << "setMatrixID " << matrixIDs[i] << endl;
        fei_isis->setMatrixID(matrixIDs[i]);

        poissonData.do_load_BC_phase();

        poissonData.do_loadComplete();
    }

    if (outputLevel>0) cout << "setMatScalars" << endl;
    fei_isis->setMatScalars(matrixIDs, matScalars, numMatScalars);

    if (outputLevel>0) cout << "setRHSScalars" << endl;
    fei_isis->setRHSScalars(rhsScaleIDs, rhsScalars, numRHSScalars);

    double solve_time0 = MPI_Wtime();

    int status;
    if (outputLevel>0) cout << "iterateToSolve..." << endl;
    int err = fei_isis->iterateToSolve(status);

    double solve_time1 = MPI_Wtime();
    double total_time1 = MPI_Wtime();

    save_FEI_block_solution(poissonData, *fei_isis,
                            1, numProcs, localProc);

    if ((outputLevel>0)&&(status == 1)&&(localProc==masterProc)) {
        double stime = solve_time1-solve_time0;
        double ttime = total_time1-total_time0;
        print_solve_status(*fei_isis, stime, ttime);
    }

    delete_IDs(numMatrices, matrixIDs, numRHSs, rhsIDs);
    delete_scalars(numMatScalars, matScalars,
                   numRHSScalars, rhsScaleIDs, rhsScalars);

    delete fei_isis;

    MPI_Finalize();

    return(0);
}

//==============================================================================
void initialize_mpi(int argc, char** argv, int& localProc, int& numProcs) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &localProc);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
}

//==============================================================================
void print_args(int argc, char** argv){
    cout << "argc: " << argc << endl;

    for(int i=0; i<argc; i++){
        cout << "argv["<<i<<"]: " << argv[i] << endl;
    }
    cout << endl;
}

//==============================================================================
void read_and_broadcast_input(char* fileName, int& L, int& W, int& DOF,
                              int numProcs, int& outputLevel, int& debugOutput){
//
//Yeah, I'm using C file IO here... I should be using an iostream.
//
    FILE* infile = fopen(fileName, "r");
    if (!infile) {
        cerr << "ERROR, failed to open file " << fileName << ", aborting."
            << endl;
        abort();
    }

    char line[128];
    (void)numProcs;

    fgets(line, 128, infile);
    sscanf(line,"%d", &L);

    fgets(line, 128, infile);
    sscanf(line,"%d", &W);

    fgets(line, 128, infile);
    sscanf(line, "%d", &DOF);

    fgets(line, 128, infile);
    sscanf(line, "%d", &outputLevel);

    fgets(line, 128, infile);
    sscanf(line, "%d", &debugOutput);

    MPI_Bcast(&L, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DOF, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&outputLevel, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&debugOutput, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int nodes = (L+1)*(W+1);
    int eqns = (L+1)*(W+1)*DOF;

    if (outputLevel>0) {
        cout << endl;
        cout << "========================================================" 
             << endl;
        cout << "Bar length     L: " << L << " elements." << endl;
        cout << "Bar width      W: " << W << " elements." << endl;
        cout << "Eqns per node DOF: " << DOF << "." << endl << endl;
        cout << "Global number of elements: " << L*W << endl;
        cout << "Global number of nodes: " << nodes << endl;
        cout << "Global number of equations: " << eqns << endl<<endl;
    }
}

//==============================================================================
void receive_input(int& L, int& W, int& DOF,
                   int& outputLevel, int& debugOutput){

    MPI_Bcast(&L, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&DOF, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&outputLevel, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&debugOutput, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

//==============================================================================
void setup_IDs(int& numMatrices, int*& matrixIDs,
               int*& numRHSs, int**& rhsIDs, int outputLevel) {
//
//Set up the IDs for managing multiple matrix and RHS contexts.
//
//There will be a list of matrix IDs, of length numMatrices. For each
//matrix, there can be 0 or more RHSs. Thus, there is a list called
//numRHSs, of length numMatrices, which indicates how many RHSs there
//are for each matrix. The rhsIDs must then be a table -- i.e., a list
//of rhsIDs for each matrix. 
//
    if (outputLevel>0) cout << "numMatrices: " << numMatrices << endl;
    matrixIDs = new int[numMatrices];

    numRHSs = new int[numMatrices];
    rhsIDs = new int*[numMatrices];

    int rhsID = 0;

    for(int i=0; i<numMatrices; i++) {

        matrixIDs[i] = i;
        if (i==0)numRHSs[i] = 1;
        if (i==1)numRHSs[i] = 1;
        if (i>=2)numRHSs[i] = 0;

        if (outputLevel>0)cout << "matrixIDs["<<i<<"]: " << matrixIDs[i];
        if (outputLevel>0)cout << ", numRHSs["<<i<<"]: " << numRHSs[i];
        if (outputLevel>0)cout << ", rhsIDs["<<i<<"]: ";

        if (numRHSs[i] > 0) {
            rhsIDs[i] = new int[numRHSs[i]];
            for(int j=0; j<numRHSs[i]; j++) {
                rhsIDs[i][j] = rhsID++;
                if (outputLevel>0)cout << rhsIDs[i][j] << " ";
            }
        }
        else {
            rhsIDs[i] = NULL;
        }
        if (outputLevel>0)cout << endl;
    }
}

//==============================================================================
void setup_scalars(int numMatrices, int* matrixIDs, int* numRHSs, int** rhsIDs,
                  int& numMatScalars, double*& matScalars,
                  int& numRHSScalars, int*& rhsScaleIDs, double*& rhsScalars) {

    (void)matrixIDs;

    if (numMatrices <= 0) return;

    numMatScalars = numMatrices;
    matScalars = new double[numMatScalars];

    int i;

    //how many RHSs are there?
    numRHSScalars = numRHSs[0];
    for(i=1; i<numMatrices; i++) numRHSScalars += numRHSs[i];

    rhsScaleIDs = new int[numRHSScalars];
    rhsScalars = new double[numRHSScalars];

    int rhsOffset = 0;
    for(i=0; i<numMatrices; i++) {
        matScalars[i] = 1.0;

        for(int j=0; j<numRHSs[i]; j++) {
            rhsScaleIDs[rhsOffset] = rhsIDs[i][j];
            rhsScalars[rhsOffset] = 1.0;
            rhsOffset++;
        }
    } 
}

//==============================================================================
void delete_scalars(int& numMatScalars, double*& matScalars,
                    int& numRHSScalars, int*& rhsScaleIDs, double*& rhsScalars){

    numMatScalars = 0;
    numRHSScalars = 0;
    delete [] matScalars;
    delete [] rhsScaleIDs;
    delete [] rhsScalars;
}

//==============================================================================
void load_FEI_parameters(FEI& linearSystem, int outputLevel){

    int i;
    int numParams = 5;
    char **params = new char*[numParams];

    for(i=0; i<numParams; i++){
        params[i] = new char[64];
    }
    sprintf(params[0],"outputLevel %d", outputLevel);
    sprintf(params[1],"solver gmres");
    sprintf(params[2],"preconditioner identity");
    sprintf(params[3],"tolerance 1.e-10");
    sprintf(params[4],"maxIterations 500");

    linearSystem.parameters(numParams, params);

    for(i=0; i<numParams; i++) delete [] params[i];
    delete [] params;
}

//==============================================================================
void put_FEI_block_solution(PoissonData& poissonData, FEI& linearSystem) {

    GlobalID localBlockID = poissonData.getLocalBlockID();
    int numActNodes = linearSystem.getNumBlockActNodes(localBlockID);
    int numActEqns = linearSystem.getNumBlockActEqns(localBlockID);

    GlobalID* nodeList = new GlobalID[numActNodes];
    int checkNumNodes = 0;

    linearSystem.getBlockNodeIDList(localBlockID, nodeList, checkNumNodes);
    if (checkNumNodes!=numActNodes) {
        cout << "put_FEI_block_solution: ERROR: checkNumNodes != numActNodes"
             << endl << "Aborting." << endl;
        abort();
    }

    double* solnValues = new double[numActEqns];
    int* offsets = new int[numActNodes+1];

    int offset = 0;

    for(int i=0; i<numActNodes; i++){
        int dof = linearSystem.getNumSolnParams(nodeList[i]);

        for(int j=0; j<dof; j++){
            solnValues[offset+j] = 0.1; //some arbitrary value
        }
        offsets[i] = offset;
        offset += dof;
    }

    linearSystem.putBlockNodeSolution(localBlockID, nodeList, numActNodes,
                                      offsets, solnValues);

    delete [] nodeList;
    delete [] solnValues;
    delete [] offsets;
}

//==============================================================================
void save_FEI_block_solution(PoissonData& poissonData, FEI& linearSystem,
                             int solve, int numProcs, int localProc) {

    GlobalID localBlockID = poissonData.getLocalBlockID();
    int numActNodes = linearSystem.getNumBlockActNodes(localBlockID);
    int numActEqns = linearSystem.getNumBlockActEqns(localBlockID);

    GlobalID* nodeList = new GlobalID[numActNodes];

    double* solnValues = new double[numActEqns];
    int* offsets = new int[numActNodes+1];

    int checkNumNodes = 0;

    linearSystem.getBlockNodeSolution(localBlockID, nodeList, checkNumNodes,
                                      offsets, solnValues);
    if (checkNumNodes!=numActNodes) {
        cout << "save_FEI_block_solution: ERROR: checkNumNodes != numActNodes"
             << endl << "Aborting." << endl;
        abort();
    }

    char* fileName = new char[32];
    sprintf(fileName,"soln.%d.%d.%d", solve, numProcs, localProc);
    ofstream outfile(fileName);
    delete [] fileName;

    for(int i=0; i<numActNodes; i++) {
        outfile << (int)nodeList[i] << ": ";

        int dof = linearSystem.getNumSolnParams(nodeList[i]);
        for(int j=0; j<dof; j++) {
            outfile << solnValues[offsets[i]+j] << " ";
        }
        outfile << endl;
    }

    outfile.close();

    delete [] nodeList;
    delete [] solnValues;
    delete [] offsets;
}

//==============================================================================
void delete_IDs(int& numMatrices, int*& matrixIDs,
               int*& numRHSs, int**& rhsIDs) {

    for(int i=0; i<numMatrices; i++) {
        if (rhsIDs[i] != NULL) delete [] rhsIDs[i];
    }
    delete [] rhsIDs;
    delete [] numRHSs;
    delete [] matrixIDs;
}

//==============================================================================
void print_solve_status(FEI& linearSystem,
                        double solve_time, double total_time) {
    cout << "print_solve_status:" << endl;
    cout << "=======================================================" << endl;
    cout << "    successful solve, iterations: " << linearSystem.iterations();
    cout << ", MPI_Wtime: " << solve_time << endl << endl;
    cout << "    Total program time: " << total_time << endl;
    cout << "=======================================================" <<endl;
}
