#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream.h>
#include <strstream.h>
#include <fstream.h>
#include <math.h>
#include <mpi.h>
#include <vector.h>

#include "Method.h"
#include "Subdomain.h"

//============================================================================

#if  HYPRE_SOLVE == ON
#include "server.cpp"
#include <cfei-hypre.h>

//============================================================================

void Subdomain::Solve_HYPRE(double *solution, char *hypre_input){

  int    i, j, k, err, status, iterations;
  int    numProcs, localProc, masterProc = 0;
  int    numParams, DOF, outputLevel, debugOutput, set = 0;
  double total_time0, solve_time1, total_time1, stime, ttime;
  char   **params;
  LinSysCore* lsc ;
  CFEI* cfei ;

  //===================================================================
  MPI_Comm_rank(MPI_COMM_WORLD, &localProc);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  //===================================================================

  // The HYPRE parameters are initialized from file "input"
  char buf[128], buf2[128];
  if (localProc == 0){
    ifstream in(hypre_input);
    if (!in){
      cout << "\nFile "<<hypre_input<<" is not available. The format is :\n"
	   << "========================================================\n"
	   << "7                      # number of parameters passed    \n"
	   << "gmres                  # solver                         \n"
	   << "diagonal               # preconditioner                 \n"
	   << "1.e-9                  # tolerance                      \n"
	   << "0                      # outputLevel                    \n"
	   << "0                      # debugOutput                    \n"
	   << "1                      # DOF                            \n"
	   << "500                    # maxIterations                  \n"
	   << "========================================================\n";
      cout << "\nTerminate the program or on your local machine start  \n"
	   << "HYPRE_Init and after setting the desired initialization \n"
	   << "hit Run.\n\n";
      cout.flush();
      char Buf[1024];
      if (read_socket(2000, Buf)){
	istrstream ins(Buf);

	ins >> numParams; numParams -= 2;
	params = new char*[numParams+1];
	for( i = 0; i <= numParams; i++ ) params[i] = new char[64];
	ins >> buf; sprintf(params[1],"solver %s", buf); 
	ins >> buf; sprintf(params[2],"preconditioner %s", buf);
	ins >> buf; sprintf(params[3],"tolerance %s", buf);
	ins >> outputLevel; sprintf(params[0],"outputLevel %d",outputLevel);
	ins >> debugOutput;                               
	ins >> DOF;                                       
	ins >> buf; sprintf(params[4], "maxIterations %s", buf);
      }
      else{
	cout << " Fail to establish connection. Exit.\n";
	cout.flush();
      }
    }
    else{
      in >> numParams; in.getline(buf, 127); numParams -= 2;
      params = new char*[numParams+1];
      for( i = 0; i <= numParams; i++ ) params[i] = new char[64];

      in>> debugOutput; in.getline(buf,127); 
      in>> DOF;         in.getline(buf,127);
      in>> outputLevel; in.getline(buf,127);                         
      
      sprintf(params[0],"outputLevel %d",outputLevel); 
      for(i=1; i<numParams; i++){
	in >> buf >> buf2; 
	sprintf(params[i],"%s %s", buf, buf2);
	in.getline(buf, 127);
      }
      /* TTT     
      in>>buf;sprintf(params[1],"solver %s", buf);       in.getline(buf,127); 
      in>>buf;sprintf(params[2],"preconditioner %s",buf);in.getline(buf,127);
      in>>buf;sprintf(params[3],"tolerance %s",buf);     in.getline(buf,127);
      in>> outputLevel; 
      sprintf(params[0],"outputLevel %d",outputLevel);   in.getline(buf,127);
      in>> debugOutput;                                  in.getline(buf,127); 
      in>> DOF;                                          in.getline(buf,127); 
      in>>buf; sprintf(params[4],"maxIterations %s",buf);in.getline(buf,127); 
      */
      
      in.close();
    }
  }
  
  MPI_Bcast(&numParams, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (localProc != 0){
    params = new char*[numParams+1];
    for( i = 0; i <= numParams; i++ ) params[i] = new char[64];
  }
  for(i=0; i<numParams; i++)
    MPI_Bcast(params[i], 64, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&DOF, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&debugOutput, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&outputLevel, 1, MPI_INT, 0, MPI_COMM_WORLD); 

  // -----------------------------------------------------------------
  // program initialization
  // -----------------------------------------------------------------
  
  total_time0 = MPI_Wtime();
  
  // -----------------------------------------------------------------
  // instantiate a FEI object
  // -----------------------------------------------------------------

  HYPRE_LinSysCore_create(&lsc, MPI_COMM_WORLD) ;

  void *grid;
  HYPRE_LSC_GetFEGridObject(lsc, &grid);
  Initialize_FEGrid(grid);

  FEI_create(&cfei, lsc, MPI_COMM_WORLD, 0) ;

  // -----------------------------------------------------------------
  // set FEI and LSC parameters
  // -----------------------------------------------------------------

  if ( debugOutput > 0 ) {
    sprintf(params[numParams], "debugOutput .");
    FEI_parameters(cfei, numParams+1, params);
  } 
  else 
    FEI_parameters(cfei, numParams, params);
  
  for ( i = 0; i <= numParams; i++ ) 
    delete [] params[i];
  delete [] params;

  // -----------------------------------------------------------------
  // set up matrix and rhs IDs
  // -----------------------------------------------------------------

  int numMatrices = 1;
  int *matrixIDs  = new int[numMatrices];
  int *numRHSs    = new int[numMatrices];
  int **rhsIDs    = new int*[numMatrices];
  matrixIDs[0] = 0;
  numRHSs[0]   = 1;
  rhsIDs[0]    = new int[numMatrices];
  rhsIDs[0][0] = 0;

  // -----------------------------------------------------------------
  // set up scalars ???
  // -----------------------------------------------------------------

  int     numMatScalars = numMatrices;
  double* matScalars    = new double[numMatScalars];
  int     numRHSScalars = 1;
  int*    rhsScaleIDs   = new int[numRHSScalars];
  double* rhsScalars    = new double[numRHSScalars];
  matScalars[0]  = 1.0;
  rhsScaleIDs[0] = rhsIDs[0][0];
  rhsScalars[0]  = 1.0;

  // -----------------------------------------------------------------
  // initialize the FEI solve
  // -----------------------------------------------------------------

  int solveType     = 0; // standard Ax=b solution
  int numElemBlocks = 1; // 1 per processor

  if (outputLevel>0) cout << "initSolveStep" << endl;
  //FEI_initSolveStep(cfei, numElemBlocks, solveType, numMatrices, matrixIDs, 
  //		     numRHSs, rhsIDs);

  FEI_initSolveStep(cfei, numElemBlocks, solveType );

  // -----------------------------------------------------------------
  // User sets up problem here
  // -----------------------------------------------------------------
  
  int    elemBlockID      = localProc;
  int    numLocalElements = NTR;
  int    fieldsPerNode    = 1;
  int    fieldSize        = DOF;
  int    numFields        = fieldsPerNode;
  int    nodesPerElement  = 4;
  int    *fieldIDs        = new int[nodesPerElement];
  int    *elemIDs         = new GlobalID[numLocalElements];
  int   **elemConn        = new int*[numLocalElements];
  int    elemSetID        = 0;
  int    elemFormat       = 0;
  double ***elemStiff     = new double**[numLocalElements];
  double **elemLoad       = new double*[numLocalElements];

  for ( i = 0; i < fieldsPerNode; i++) 
    fieldIDs[i] = fieldsPerNode;

  // Create global element IDs
  CreateGlobalElemIndex(elemIDs);
  
  // Create global node IDs; "slaves" gives the number of slave nodes.
  int slaves, *nodeIDs = new int[NN[level]];
  CreateGlobalNodeIndex(nodeIDs, slaves);

  for ( i = 0; i < numLocalElements; i++ ) {
    elemConn[i]  = new int[nodesPerElement];
    elemStiff[i] = new double*[nodesPerElement];
    elemLoad[i]  = new double[nodesPerElement];
    for ( j = 0; j < nodesPerElement; j++ ){ 
      elemConn[i][j]  = nodeIDs[TR[i].node[j]];  // global index used
      elemStiff[i][j] = new double[nodesPerElement];
    }
    // 1. calculate elemStiff : fill in elemStiff[i][[0:nPE-1][0:nPE-1]
    // 2. calculate elemLoad  : fill in elemLoad[i][0:nodesPerElement-1]
    // The first argument is the 4x4 local stiffness matrix, the second is
    // array of 4 elements giving element i's contribution to the RHS 
    ComputeLocalMatrix(i, elemStiff[i], elemLoad[i]);
  } 

  int *numFieldArray = new int[nodesPerElement];
  for ( i = 0; i < nodesPerElement; i++ ) numFieldArray[i] = numFields;
  int **fieldIDArray = new int*[nodesPerElement];
  for ( i = 0; i < nodesPerElement; i++ ){
    fieldIDArray[i] = new int[1];
    fieldIDArray[i][0] = fieldIDs[0];
  } 
  
  err = FEI_initFields(cfei,numFields, &fieldSize, &fieldIDs[0]);
  err = FEI_beginInitElemBlock(cfei,elemBlockID, nodesPerElement,
		 numFieldArray, fieldIDArray, 0, 0, 0, 1, numLocalElements);
  err = FEI_initElemSet(cfei,numLocalElements, elemIDs, elemConn);
  err = FEI_endInitElemBlock(cfei);
  
  // According to the example this may be 1 (why not NPackets?)
  int numSharedNodeSets = 1;
  
  int *sharedNodeIDs, **sharedNodeProcs, numSharedNodes = 0;
  int *numSharedNodeProcs;

  // no external node sets
  err = FEI_beginInitNodeSets(cfei,numSharedNodeSets, 0);

  for(i = 0; i < NPackets; i++)
    numSharedNodes += Pa[i].NPoints;                  // points in set i
  
  sharedNodeIDs  = new int[numSharedNodes];
  int vari = 0;
  for(i=0; i < NPackets; i++)
    for(j=0; j<Pa[i].NPoints; j++)
      sharedNodeIDs[vari++] = nodeIDs[Pa[i].Ve[j]];     
  
  sharedNodeProcs = new int*[numSharedNodes];
  numSharedNodeProcs  = new int[numSharedNodes];

  vari = 0;
  for(i=0; i < NPackets; i++)   
    for(j=0; j<Pa[i].NPoints; j++){
      numSharedNodeProcs[vari]  = Pa[i].NSubdom + 1;
      sharedNodeProcs[vari] = new int[numSharedNodeProcs[vari]];
      for(k=0; k<Pa[i].NSubdom; k++)
	sharedNodeProcs[vari][k] = Pa[i].Subdns[k];
      sharedNodeProcs[vari][Pa[i].NSubdom] = SN;
      vari++;
    }
    
  err=FEI_initSharedNodeSet(cfei,sharedNodeIDs, numSharedNodes, 
			    sharedNodeProcs, numSharedNodeProcs);
  err = FEI_endInitNodeSets(cfei);
  err = FEI_initComplete(cfei);

  // -----------------------------------------------------------------
  // load element information, eg.  stiffness matrices
  // -----------------------------------------------------------------
  
  //err = FEI_setMatrixID(cfei,matrixIDs[0]);
  
  err = FEI_beginLoadElemBlock(cfei, elemBlockID, 1, numLocalElements);
  
  err = FEI_loadElemSet(cfei, elemSetID, numLocalElements, elemIDs, 
			elemConn, elemStiff, elemLoad, elemFormat);
  //err = FEI_setRHSID(cfei, rhsIDs[0][0]);
  //err = FEI_loadElemSetRHS(cfei, elemSetID, numLocalElements, elemIDs, 
  //			    elemConn, elemLoad);
  err = FEI_endLoadElemBlock(cfei );
  
  // -----------------------------------------------------------------
  // load boundary conditions
  // -----------------------------------------------------------------
  
  double      **alpha;
  double      **beta;
  double      **gamma;

  int    numBCNodes = dim_Dir[level];
  int    *BCNodeIDs = new int[numBCNodes];

  alpha = new double*[numBCNodes];
  beta  = new double*[numBCNodes];
  gamma = new double*[numBCNodes];

  // fillin BCNodeIDs : global node IDs
  for(i=0; i<numBCNodes; i++){
    alpha[i] = new double[fieldIDs[0]];
    beta[ i] = new double[fieldIDs[0]];
    gamma[i] = new double[fieldIDs[0]];

    // for the test fieldIDs[0] = 1, so we init only ...[i][0]
    // The essential boundaru values for u_i are given by the equations 
    // alpha[i]*u_i + beta[i]*g_i = gamma[i], where g_i is the dual of u_i 
    alpha[i][0] = 1.0;
    beta[i][0]  = 0.0;
    gamma[i][0] = func_u0( Z[Dir[i]].GetCoord()); //Gives the Dirichlet BValue

    BCNodeIDs[i]= nodeIDs[Dir[i]];
  }

  err = FEI_beginLoadNodeSets(cfei,1);
  err = FEI_loadBCSet(cfei,BCNodeIDs,numBCNodes,fieldIDs[0],alpha,beta,gamma);
  err = FEI_endLoadNodeSets(cfei);
  
  // -----------------------------------------------------------------
  // done loading
  // -----------------------------------------------------------------
  
  err = FEI_loadComplete(cfei);
  
  //if (outputLevel>0) cout << "setMatScalars" << endl;
  //FEI_setMatScalars(cfei,matrixIDs, matScalars, numMatScalars);
  
  //if (outputLevel>0) cout << "setRHSScalars" << endl;
  //FEI_setRHSScalars(cfei,rhsScaleIDs, rhsScalars, numRHSScalars);
  
  // -----------------------------------------------------------------
  // solve
  // -----------------------------------------------------------------
  
  double solve_time0 = MPI_Wtime();
  
  if (outputLevel>0) cout << "iterateToSolve..." << endl;
  
  err = FEI_iterateToSolve(cfei,&status);
  
  if (err) cout << "iterateToSolve returned err: " << err << endl;
  
  solve_time1 = MPI_Wtime();
  total_time1 = MPI_Wtime();
  
  // -----------------------------------------------------------------
  // output status and timing
  // -----------------------------------------------------------------
  
  if ((outputLevel>0)&&(status == 1)&&(localProc==masterProc)) {
    stime = solve_time1-solve_time0;
    ttime = total_time1-total_time0;
    cout << "print_solve_status:" << endl;
    cout << "==================================================" << endl;
    FEI_iterations(cfei, &iterations);
    cout << "    successful solve, iterations: " << iterations;
    cout << ", MPI_Wtime: " << stime << endl << endl;
    cout << "    Total program time: " << ttime << endl;
    cout << "==================================================" <<endl;
  }
  
  int lenNodeIDList, *nodeIDs_inv = new int[NN[level]];
  int *nodeIDList = new int[NN[level]], *offset = new int[NN[level]+1];
  double *answers = new double[NN[level]];
  
  FEI_getBlockNodeSolution(cfei, elemBlockID, nodeIDList,
                           &lenNodeIDList, offset, answers);

  // -----------------------------------------------------------------  
  // Initialize a Hash Table
  // -----------------------------------------------------------------
  int ind, nn = NN[level], found;
  for(i=0; i<lenNodeIDList; i++) nodeIDs_inv[i] = -1;
  for(i=0; i<lenNodeIDList; i++){
    ind = nodeIDs[i] % nn;
    found = 0;
    for(j=ind; j<nn; j++)
      if (nodeIDs_inv[j] == -1){
	nodeIDs_inv[j] = i;
	found = 1;
	break;
      }
    if (found == 0)
      for(j=0; j<ind ; j++)
	if (nodeIDs_inv[j] == -1){
	  nodeIDs_inv[j] = i;
	  break;
	}
  }
  
  for(i=0; i<lenNodeIDList; i++){ // compute the inverse of nodeIDs 
    ind = nodeIDList[i]%nn;
    found = 0;
    for(j=ind; j<nn; j++)
      if ( nodeIDs[nodeIDs_inv[j]] == nodeIDList[i]){
	found = 1;
	solution[nodeIDs_inv[j]] = answers[offset[i]];
	break;
      }
    if (found == 0)
      for(j=0; j<ind ; j++)
	if ( nodeIDs[nodeIDs_inv[j]] == nodeIDList[i]){
	  solution[nodeIDs_inv[j]] = answers[offset[i]];
	  break;
	}
  }
    
  // -----------------------------------------------------------------
  // clean up
  // -----------------------------------------------------------------
  
  delete [] nodeIDs_inv;
  delete [] nodeIDList;
  delete [] offset;
  delete [] answers;

  delete [] matrixIDs;
  delete [] numRHSs;
  delete [] rhsIDs[0];
  delete [] rhsIDs;
  delete [] matScalars;
  delete [] rhsScaleIDs;
  delete [] rhsScalars;

  FEI_destroy(&cfei) ;
  LinSysCore_destroy(&lsc) ;

  delete [] numFieldArray;
  for(i=0; i < nodesPerElement; i++ )
    delete [] fieldIDArray[i];
  delete [] fieldIDArray;

  delete [] elemIDs;
  for (i = 0; i < numLocalElements; i++ )
  {
    delete [] elemConn[i];
    delete [] elemLoad[i];
    for ( j = 0; j < nodesPerElement; j++ ) delete [] elemStiff[i][j];
    delete [] elemStiff[i];
  }
  delete [] elemConn;
  delete [] elemStiff;
  delete [] elemLoad;
  delete [] fieldIDs;
  delete [] BCNodeIDs;
  
  for(i=0; i<numBCNodes; i++){
    delete [] alpha[i];
    delete []  beta[i];
    delete [] gamma[i];
  }
  delete [] alpha;
  delete [] beta;
  delete [] gamma;

  delete [] nodeIDs;
  delete [] sharedNodeIDs;
  for(i=0; i<numSharedNodes; i++) delete [] sharedNodeProcs[i];
  delete [] sharedNodeProcs;
  delete [] numSharedNodeProcs;
}


//============================================================================

void Subdomain::Initialize_FEGrid(void *grid){

  int i, j, num;

  // --------------------------------------------------------------------
  // User sets the following variables
  // --------------------------------------------------------------------

  int    DOF             = 1;
  int    Dimension       = 3;
  int    nNodes          = NN[level];
  int    nElems          = NTR;
  int    *elemIDs        = new GlobalID[nElems];
  int    *nodeIDs        = new int[nNodes];
  int    **elemConn      = new int*[nElems];
  double ***elemStiff    = new double**[nElems];
  double **elemLoad      = new double*[nElems];
  int    nodesPerElement = 4;
  
  // --------------------------------------------------------------------
  // Set the space dimension
  // --------------------------------------------------------------------
  //HYPRE_FEGrid_setSpaceDimension(Dimension);

  // --------------------------------------------------------------------
  // Create the global node and element index.
  // --------------------------------------------------------------------

  int slaves;                             // number of slave nodes
  CreateGlobalNodeIndex(nodeIDs, slaves);
  CreateGlobalElemIndex(elemIDs);

  // --------------------------------------------------------------------
  // Initialize the elements 
  // --------------------------------------------------------------------
  
  HYPRE_FEGrid_beginInitElemSet(grid, nElems, elemIDs);

  for ( i = 0; i < nElems; i++ ) {
    elemConn[i]  = new int[nodesPerElement];
    elemStiff[i] = new double*[nodesPerElement];
    elemLoad[i]  = new double[nodesPerElement];
    for ( j = 0; j < nodesPerElement; j++ ){ 
      elemConn[i][j]  = nodeIDs[TR[i].node[j]];  // global index used
      elemStiff[i][j] = new double[nodesPerElement];
    }
    // 1. calculate elemStiff : fill in elemStiff[i][[0:nPE-1][0:nPE-1]
    // 2. calculate elemLoad : fill in elemLoad[i][0:nodesPerElement-1]
    // The first argument is the 4x4 local stiffness matrix, the second is
    // array of 4 elements giving element i's contribution to the RHS 
    ComputeLocalMatrix(i, elemStiff[i], elemLoad[i]);

    HYPRE_FEGrid_loadElemSet(grid, elemIDs[i], nodesPerElement, elemConn[i], 
			     nodesPerElement, elemStiff[i]);
  } 

  HYPRE_FEGrid_endInitElemSet(grid);

  // --------------------------------------------------------------------
  // Initialize the nodes
  // --------------------------------------------------------------------

  HYPRE_FEGrid_beginInitNodeSet(grid);

  for ( i = 0; i < nNodes; i++ ){
    HYPRE_FEGrid_loadNodeDOF(grid, nodeIDs[i], DOF);
    //HYPRE_FEGrid_loadNodeCoordinate(grid, nodeIDs[i], Z[i].GetCoord());
  }

  // Initialize the essential boundary condition
  int    numBCNodes = dim_Dir[level];
  int    *BCNodeIDs = new int[numBCNodes];
  int    *dofList   = new int[numBCNodes];
  double *val       = new double[numBCNodes];

  for(i=0; i<numBCNodes; i++){
    BCNodeIDs[i] = nodeIDs[Dir[i]];
    dofList[i]   = 0;
    val[i]       = func_u0( Z[Dir[i]].GetCoord()); //Gives the Dir BValue
  }

  HYPRE_FEGrid_loadNodeEssBCs(grid, numBCNodes, BCNodeIDs, dofList, val);

  // --------------------------------------------------------------------
  // Initialize the shared nodes 
  // --------------------------------------------------------------------
  int numSharedNodes = 0, *numSharedNodeProcs;
  int *sharedNodeIDs, **sharedNodeProcs;
  
  for(i = 0; i < NPackets; i++)
    numSharedNodes += Pa[i].NPoints;                  // points in set i

  sharedNodeIDs  = new int[numSharedNodes];
  int vari = 0, k;
  for(i=0; i < NPackets; i++)
    for(j=0; j<Pa[i].NPoints; j++)
      sharedNodeIDs[vari++] = nodeIDs[Pa[i].Ve[j]];

  sharedNodeProcs = new int*[numSharedNodes];
  numSharedNodeProcs  = new int[numSharedNodes];

  vari = 0;
  for(i=0; i < NPackets; i++)   
    for(j=0; j<Pa[i].NPoints; j++){
      numSharedNodeProcs[vari]  = Pa[i].NSubdom + 1;
      sharedNodeProcs[vari] = new int[numSharedNodeProcs[vari]];
      for(k=0; k<Pa[i].NSubdom; k++)
	sharedNodeProcs[vari][k] = Pa[i].Subdns[k];
      sharedNodeProcs[vari][Pa[i].NSubdom] = SN;
      vari++;
    }
  
  HYPRE_FEGrid_loadSharedNodes(grid, numSharedNodes, sharedNodeIDs, 
			       numSharedNodeProcs, sharedNodeProcs);

  HYPRE_FEGrid_endInitNodeSet(grid);

  // --------------------------------------------------------------------
  // Initialize element properties (Volume and material property)
  // --------------------------------------------------------------------
  /*
  for(i=0; i<nElems; i++){
    HYPRE_FEGrid_loadElemVolume(grid, elemIDs[i], area(i));
    HYPRE_FEGrid_loadElemMaterial(grid, elemIDs[i], TR[i].atribut);
  }
  */

  // --------------------------------------------------------------------
  // clean up
  // --------------------------------------------------------------------

  delete [] elemIDs;
  delete [] nodeIDs;
  for (i=0; i<nElems;i++){
    delete [] elemConn[i];
    delete [] elemLoad[i];
    for ( j = 0; j < nodesPerElement; j++ ) delete [] elemStiff[i][j];
    delete [] elemStiff[i];
  }
  delete [] elemConn;
  delete [] elemStiff;
  delete [] elemLoad;
 
  delete [] BCNodeIDs;
  delete [] dofList;
  delete [] val;

  delete [] sharedNodeIDs;
  for(i=0; i<numSharedNodes; i++) delete [] sharedNodeProcs[i];
  delete [] sharedNodeProcs;
  delete [] numSharedNodeProcs;
}

//============================================================================

#endif

