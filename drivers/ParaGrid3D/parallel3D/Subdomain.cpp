#include <stdio.h>
#include <iostream.h>
#include <iomanip.h>
#include <math.h>
#include "extension_MPI.h"
#include <sys/time.h>       // these twow are for measuring the time
#include <sys/resource.h>
#include <set.h>

#include "Subdomain.h"

//============================================================================
// This function updates the values of x - add contributions from no owners,
// making the values at the no-owned nodes 0.
//============================================================================
void Subdomain::Update(reals x){
  int i, j, k, PNum, n = 0, start;
  double Buf[MAX_PACKET];

  MPI_Request request;
  MPI_Status Status;

  MPI_Barrier(MPI_COMM_WORLD); 
  // Put in Bdr all the information that has to be send. We do this because
  // the send will be non-blocking. After one entire package is in Bdr we
  // send (non-blocking) the information to the owner.
  for(i=0; i<NPackets; i++)
    if (Pa[i].Owner != SN){
      start = n;
      for(j=0; j<Pa[i].NPoints; j++){	
	Bdr[n++] = x[Pa[i].Ve[j]];
	x[Pa[i].Ve[j]] = 0.;
      }
      
      MPI_Isend(&Bdr[start], Pa[i].NPoints, MPI_DOUBLE, Pa[i].Owner, 
		Pa[i].Pack_Num_Owner, MPI_COMM_WORLD, &request);
    }

  // The owners receive packets from neighboring subdomains and do the
  // necessary corrections.
  for(i=0; i<NPackets; i++)
    if (Pa[i].Owner == SN){   // every owner should read Pa[i].NSubdom packets
      for(j=0; j<Pa[i].NSubdom; j++){
	MPI_Recv( Buf, MAX_PACKET, MPI_DOUBLE, MPI_ANY_SOURCE,
		  MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
	// MPI_Get_count( &Status, MPI_DOUBLE, &n); // how many were read   
	PNum = Status.MPI_TAG;

	for(k=0; k< Pa[PNum].NPoints; k++)
	  x[Pa[PNum].Ve[k]] += Buf[k];
      }
    }
  MPI_Barrier(MPI_COMM_WORLD);
}


//============================================================================
// The owners send their values to the slave nodes and update them.
//============================================================================
void Subdomain::Update_Slave_Values(reals x){
  int i, j, k, PNum, n = 0, start;
  double Buf[MAX_PACKET];
  
  MPI_Request request;
  MPI_Status Status;
  
  // Put in Bdr all the information that has to be send. We do this because
  // the send will be non-blocking. After one entire package is in Bdr we
  // send (non-blocking) the information to the no owners.
  for(i=0; i<NPackets; i++)
    if (Pa[i].Owner == SN){   // every owner should send Pa[i].NSubdom packets
      start = n;
      for(j=0; j<Pa[i].NPoints; j++)	 // prepare the packet
	Bdr[n++] = x[Pa[i].Ve[j]];
      
      for(j=0; j<Pa[i].NSubdom; j++){    // send to the neighbors
	MPI_Isend(&Bdr[start], Pa[i].NPoints, MPI_DOUBLE, Pa[i].Subdns[j], 
		  Pa[i].Pack_Num[j], MPI_COMM_WORLD, &request);
      }
    }

  // The owners receive packets from neighboring subdomains and do the
  // necessary corrections.
  for(i=0; i<NPackets; i++)
    if (Pa[i].Owner != SN){   // every no-owner should read 1 packet
      MPI_Recv( Buf, MAX_PACKET, MPI_DOUBLE, MPI_ANY_SOURCE,
		MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
      // MPI_Get_count( &Status, MPI_DOUBLE, &n); // how many were read   
      PNum = Status.MPI_TAG;

      for(k=0; k< Pa[PNum].NPoints; k++)
	x[Pa[PNum].Ve[k]] = Buf[k];
    }

  MPI_Barrier(MPI_COMM_WORLD);
}


//============================================================================
// The same as the above function but for integer type of argument.
//============================================================================
void Subdomain::Update_Slave_Values(int *x){
  int i, j, k, PNum, n = 0, start;
  int Buf[MAX_PACKET], Bdr[6*MAX_PACKET];
  
  MPI_Request request;
  MPI_Status  Status;

  // Put in Bdr all the information that has to be send. We do this because
  // the send will be non-blocking. After one entire package is in Bdr we
  // send (non-blocking) the information to the no owners.
  for(i=0; i<NPackets; i++)
    if (Pa[i].Owner == SN){   // every owner should send Pa[i].NSubdom packets
      start = n;
      for(j=0; j<Pa[i].NPoints; j++)	 // prepare the packet
	Bdr[n++] = x[Pa[i].Ve[j]];
      
      for(j=0; j<Pa[i].NSubdom; j++)     // send to the neighbors
	MPI_Isend(&Bdr[start], Pa[i].NPoints, MPI_INT, Pa[i].Subdns[j], 
		  Pa[i].Pack_Num[j], MPI_COMM_WORLD, &request);
    }

  // The owners receive packets from neighboring subdomains and do the
  // necessary corrections.
  for(i=0; i<NPackets; i++)
    if (Pa[i].Owner != SN){   // every no-owner should read 1 packet
      MPI_Recv( Buf, MAX_PACKET, MPI_INT, MPI_ANY_SOURCE,
		MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
      // MPI_Get_count( &Status, MPI_DOUBLE, &n); // how many were read   
      PNum = Status.MPI_TAG;  // get who sent the packet

      for(k=0; k< Pa[PNum].NPoints; k++)
	x[Pa[PNum].Ve[k]] = Buf[k];
    }
  // We need this barrier becouse the memory for Bdr is reserved inside the 
  // the procedure (one processor may finish earlier and destroy it).
  MPI_Barrier(MPI_COMM_WORLD);
}


//============================================================================
// The same as the above function but for the faces.
//============================================================================
void Subdomain::Update_Slave_Face_Values(int *x){
  int i, j, PNum, n = 0, start;
  int Buf[MAX_PACKET], Bdr[6*MAX_PACKET];
  
  MPI_Request request;
  MPI_Status  Status;

  for(i=0; i<NPackets; i++)
    if (Pa[i].Owner == SN && Pa[i].Faces.size()){
      start = n;
      for(j=0; j<Pa[i].Faces.size(); j++)// prepare the packet
	Bdr[n++] = x[Pa[i].Faces[j]];
      
      MPI_Isend(&Bdr[start], Pa[i].Faces.size(), MPI_INT, Pa[i].Subdns[0], 
		Pa[i].Pack_Num[0], MPI_COMM_WORLD, &request);
    }

  for(i=0; i<NPackets; i++)
    if (Pa[i].Owner != SN && Pa[i].Faces.size()){
      MPI_Recv( Buf, MAX_PACKET, MPI_INT, MPI_ANY_SOURCE,
		MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
      PNum = Status.MPI_TAG;             // get who sent the packet

      for(j=0; j< Pa[PNum].Faces.size(); j++)
	x[Pa[PNum].Faces[j]] = Buf[j];
    }

  MPI_Barrier(MPI_COMM_WORLD);
}


//============================================================================
// This function makes the values at the slave nodes zero.
//============================================================================
void Subdomain::Zero_Slave_Values(reals x){
  int i, j;
  for(i=0; i<NPackets; i++)
    if (Pa[i].Owner != SN)   // no-owner should make it's node values zero
      for(j=0; j<Pa[i].NPoints; j++)
	x[Pa[i].Ve[j]] = 0.;
}

//============================================================================

void Subdomain::Print(){
  FILE *pfile;
  char FName[40];
  int i, j;
  
  if (SN == 0)
    cout << "Packet information printed in file ./Packets\n";
  sprintf(FName, "Packets%d", SN);
  pfile = fopen(FName,"w+");
  
  // Print some mesh information:
  fprintf(pfile,"Subdomain %d\n", SN);
  fprintf(pfile,"+==========================================+\n");
  fprintf(pfile,"| Structure initialization on Level 0 done |\n");
  fprintf(pfile,"+==========================================+\n");
  fprintf(pfile,"|  Number of tetrahedrons    = %7d     |\n", NTR);
  fprintf(pfile,"|  Number of faces           = %7d     |\n", NF);
  fprintf(pfile,"|  Number of nodes           = %7d     |\n", NN[level]);
  fprintf(pfile,"|  Number of Dirichlet nodes = %7d     |\n", dim_Dir[0]);
  fprintf(pfile,"+==========================================+\n\n");

  for(i=0; i<NPackets; i++){
    fprintf(pfile,"Packet %d, (Owner %d, %d), connected to:\n", 
	    i, Pa[i].Owner, Pa[i].Pack_Num_Owner);
    fprintf(pfile,"============================================\n");

    for(j=0; j<Pa[i].NSubdom; j++) 
      fprintf(pfile,"   (Subdomain %2d, local %d) ",
	      Pa[i].Subdns[j],Pa[i].Pack_Num[j]);

    fprintf(pfile,"\n  Points %4d:\n", Pa[i].NPoints);
    for(j=0; j<Pa[i].NPoints; j++)
      fprintf(pfile,"    %3d:(%5.2f %5.2f %5.2f)\n",j,Z[Pa[i].Ve[j]].coord[0],
	      Z[Pa[i].Ve[j]].coord[1], Z[Pa[i].Ve[j]].coord[2]);

    fprintf(pfile,"\n  Edges %4d:\n", Pa[i].Edges.size());
    for(j=0; j<Pa[i].Edges.size(); j++)
      fprintf(pfile,"    %3d:(%5.2f %5.2f %5.2f) - (%5.2f %5.2f %5.2f)\n", j,
	      Z[Pa[i].Edges[j].node[0]].coord[0], 
	      Z[Pa[i].Edges[j].node[0]].coord[1],
	      Z[Pa[i].Edges[j].node[0]].coord[2],
	      Z[Pa[i].Edges[j].node[1]].coord[0],
	      Z[Pa[i].Edges[j].node[1]].coord[1],
	      Z[Pa[i].Edges[j].node[1]].coord[2] );

    fprintf(pfile,"\n  Faces %4d:\n", Pa[i].Faces.size());
    for(j=0; j<Pa[i].Faces.size(); j++)
      fprintf(pfile,
	      "    %3d:(%5.2f %5.2f %5.2f) - (%5.2f %5.2f %5.2f) - (%5.2f %5.2f %5.2f)\n", j,
	      Z[F[Pa[i].Faces[j]].node[0]].coord[0],
	      Z[F[Pa[i].Faces[j]].node[0]].coord[1],
	      Z[F[Pa[i].Faces[j]].node[0]].coord[2],
	      Z[F[Pa[i].Faces[j]].node[1]].coord[0],
	      Z[F[Pa[i].Faces[j]].node[1]].coord[1],
	      Z[F[Pa[i].Faces[j]].node[1]].coord[2],
	      Z[F[Pa[i].Faces[j]].node[2]].coord[0],
	      Z[F[Pa[i].Faces[j]].node[2]].coord[1],
	      Z[F[Pa[i].Faces[j]].node[2]].coord[2] );
    fprintf(pfile,"\n\n");
  }
  
  fclose(pfile);
}

//============================================================================

void Subdomain::Solve(){
  double *solution=new double [NN[level]];
  char FName1[128], FName2[128];
  int i, Refinement, refine, *array, sn, numprocs;

  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  #if EXACT == ON
    double *zero;
  #endif

  struct rusage t;   // used to measure the time
  long int time;
  double t1;

  for(i=0; i<NN[level]; i++) solution[i]=0.;

  do {
    for(i=0; i < NN[level]; i++) solution[i] = 1.;
    Init_Dir(level, solution);
    Update( b);                                 //=== new 1 ===
    Null_Dir(level, b);

    getrusage( RUSAGE_SELF, &t);
    time = 1000000 * (t).ru_utime.tv_sec + ( t).ru_utime.tv_usec;
  
    #if HYPRE_SOLVE == ON
      Solve_HYPRE(solution, "hypre");
    #else
      CG( level, 3000, solution, b, &GLOBAL[level]);
    #endif

    getrusage( RUSAGE_SELF, &t);
    time = 1000000 * (t).ru_utime.tv_sec + ( t).ru_utime.tv_usec - time;
    t1 = time/1000000.;

    //critical();
    //cout.precision(5);
    //cout << "Proc " << SN << ". Elapsed time in seconds : "
    //     << setw(10) << t1 << endl; 
    //end_critical();
    
    Update_Slave_Values(solution);
    MPI_Barrier(MPI_COMM_WORLD);
    #if EXACT == ON
      double e1, e2;
      zero = new double[NN[level]];
      for(i=0; i<NN[level]; i++) zero[i] = 0.;
    
      e1 = error_En( solution);
      critical();
      cout << "SN = " << setw(2) << SN 
	   << ". The discrete En. norm error is : " << e1 << endl;
      end_critical();

      e1 = error_L2( solution); e2 = 100*e1/error_L2(zero);
      critical();
      cout << "SN = " << setw(2) << SN
	   << ". The discrete L^2 norm error is : " << e1 << "  "
	   << e2 << "%%\n";
      end_critical();

      e1 = error_max(solution); e2 = 100*e1/error_max(zero);
      critical();
      cout << "SN = " << setw(2) << SN
	   << ". The discrete max norm error is : " << e1 
	   << "  " << e2 << "%%\n"; 
      end_critical();

      delete [] zero;
    #endif

    switch (OUTPUT_TYPE){
    case 1:
      sprintf(FName1,"~/Visual/Visualization%d.off", SN);
      PrintMCGL(FName1, solution);
      do {
	if (SN == 0){
	  cout << "Plot subdomain (subdomain number or '-1' to continue): ";
	  cin >> sn;
	}
	MPI_Bcast(&sn, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (sn>=0 && sn < numprocs)
	  PrintMCGL(FName1, solution, sn);
      }
      while (sn!=-1);
     
      MPI_Barrier(MPI_COMM_WORLD);
      break;
    case 0:
      break;
    }; 

    if (SN == 0){
      cout<<"Input desired parallel refinement (1:ZZ, 2:F, 3:Uni, 4:Exit): ";
      cout.flush();
      cin >> Refinement;
    } 
    MPI_Bcast(&Refinement, 1, MPI_INT, 0, MPI_COMM_WORLD);

    refine = 1;
    array = new int[NTR];
    switch (Refinement){
    case 1:
      MPI_Barrier(MPI_COMM_WORLD);
      double percent;
      if (SN == 0){
	cout << "\nInput desired percent error reduction (0 for exit) : ";
	cin  >> percent;
      }
      MPI_Bcast(&percent, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      Refine_ZZ(percent, solution, array);
      break;
    case 2:
      Refine_F(array, level, SN, NTR, Z, TR);
      break;
    case 3:
      for(i=0; i<NTR; i++) array[i] = 1;
      break;
    case 4:
      refine = 0;
      break;
    }

    if (refine){
      LocalRefine( array);

      Create_level_matrices();

      GLOBAL[level].InitSubdomain( this);
      if (level!=0){
	#if MAINTAIN_HB == ON
	  HB[level].InitSubdomain( this);
        #endif
        #if REORDER == OFF  
          Interp[level].InitSubdomain( this);
	  //Interp[level].Update_Interpolation_Bdr();
        #endif
      }	 
      
      double *new_sol = new double[NN[level]];
      //Interp[level].Action(solution, new_sol);
      
      delete [] solution;
      solution = new_sol;
    }
    delete []array;
  }
  while (refine);
  delete [] solution;
}

//============================================================================
// Add face with index face_number to the corresponding packet - has to be 
// the one connected only to subdomain subd_number. If there is no such the 
// function creates a new package with no points in with and only the 
// considered face.
//============================================================================
void Subdomain::add_face_to_packet(int subd_number, int face_number){
  int i, found = 0;
  
  for(i=0; i<NPackets; i++)           // for all packets (find the right one)
    if (Pa[i].add_face(subd_number, face_number, SN)==1){
      found = 1;
      break;
    }
  
  // add_face adds face to a packet or creates a new packet if NFaces = 0
  if (found==0){
    if (NPackets == PACKETS){
      cout << "Increase PACKETS in definitions.h. Exit.\n";
      exit(1);
    }
    Pa[NPackets].add_face(subd_number, face_number, SN);
    NPackets++;
  }
}

//============================================================================
// Add new edge to corresponding packet. The edge is given by end points
// node1 and node2 in local coordinates with node1 < node2. The edge
// belongs to subdomain SN and the others are given in set subdomains.
//============================================================================
void Subdomain::add_edge_to_packet(int node1,int node2,set<int, less<int> > &subdomains){
  int i, found = 0;
  
  for(i=0; i<NPackets; i++)           // for all packets (find the right one)
    if (Pa[i].add_edge(node1, node2, subdomains, SN)==1){
      found = 1;
      break;
    }
  
  // add_edge adds edge to a packet or creates a new packet if necessary
  if (found==0){
    if (NPackets == PACKETS){
      cout << "Increase PACKETS in definitions.h. Exit.\n";
      exit(1);
    }
    Pa[NPackets++].add_edge(node1, node2, subdomains, SN);
  }
}

//============================================================================
// This function is used to create global index. The output is array parallel
// to the nodes in the sub-domain and give their new number. The number of 
// slave nodes is given in nslaves. The memory allocation has to be outside 
// the function.
//============================================================================
void Subdomain::CreateGlobalNodeIndex(int *g_index, int &nslaves){
  int i, j, nn = NN[level], ind = 0;
  int numprocs, flag = 1, shift1, shift2;
  MPI_Status Status;
  
  MPI_Barrier(MPI_COMM_WORLD);

  for(i=0; i<nn; i++) g_index[i] = -1;
  
  // We first enumerate the nodes on the boundary. Only the owned nodes
  // are counted.
  for(i=0; i<NPackets; i++)
    if (Pa[i].Owner == SN)      // this subdomain is owner of the nodes
      for(j=0; j<Pa[i].NPoints; j++)
	g_index[Pa[i].Ve[j]] = ind++;
    else                        // these are slave nodes (mark them with -2)
      for(j=0; j<Pa[i].NPoints; j++)
	g_index[Pa[i].Ve[j]] = -2;

  for(i=0; i<nn; i++)
    if (g_index[i] == -1)
      g_index[i] = ind++;
  
  nslaves = nn - ind;

  shift1 = 0;                   // This is the shift for subdomain 0
  if (SN != 0)
    MPI_Recv( &shift1, 1, MPI_INT, MPI_ANY_SOURCE,
	      MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  if (SN + 1 < numprocs){
    shift2 = ind + shift1;
    MPI_Send(&shift2, 1, MPI_INT, SN+1, flag, MPI_COMM_WORLD);
  }
  for(i=0; i<nn; i++)
    if (g_index[i] != -2)
      g_index[i] += shift1;

  MPI_Barrier(MPI_COMM_WORLD);
  Update_Slave_Values(g_index);
}

//============================================================================
// This function is used to create global element index. The output is array
// "parallel" to the elements in the sub-domain giving the new numbers. 
//============================================================================
void Subdomain::CreateGlobalElemIndex(int *g_index){
  int i, ind = 0;
  int numprocs, flag = 1, shift1, shift2;
  MPI_Status Status;

  MPI_Barrier(MPI_COMM_WORLD);
  shift1 = 0;                   // This is the shift for subdomain 0
  if (SN != 0)
    MPI_Recv( &shift1, 1, MPI_INT, MPI_ANY_SOURCE,
	      MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  if (SN + 1 < numprocs){
    shift2 = NTR + shift1;
    MPI_Send(&shift2, 1, MPI_INT, SN+1, flag, MPI_COMM_WORLD);
  }
  for(i=0; i<NTR; i++)
    g_index[i] = i + shift1;

  MPI_Barrier(MPI_COMM_WORLD);
}

//============================================================================
// This function is used to create global face index. The output is array 
// "parallel" to the face in the sub-domain and give their new number. The 
// number of slave faces is given in nslaves. The memory allocation has to 
// be outside the function. The global index may be used in Mixed FE.
//============================================================================
void Subdomain::CreateGlobalFaceIndex(int *g_index, int &nslaves){
  int i, j, ind = 0;
  int numprocs, flag = 1, shift1, shift2;
  MPI_Status Status;
  
  MPI_Barrier(MPI_COMM_WORLD);

  for(i=0; i<NF; i++) g_index[i] = -1;
  
  // We first enumerate the faces on the boundary. Only the owned faces
  // are counted.
  for(i=0; i<NPackets; i++)
    if (Pa[i].Owner == SN)      // this subdomain is owner of the edges
      for(j=0; j<Pa[i].Faces.size(); j++)
	g_index[Pa[i].Faces[j]] = ind++;
    else                        // these are slave edges (mark them with -2)
      for(j=0; j<Pa[i].Faces.size(); j++)
	g_index[Pa[i].Faces[j]] = -2;

  for(i=0; i<NF; i++)
    if (g_index[i] == -1)
      g_index[i] = ind++;
  
  nslaves = NF - ind;

  shift1 = 0;                   // This is the shift for subdomain 0
  if (SN != 0)
    MPI_Recv( &shift1, 1, MPI_INT, MPI_ANY_SOURCE,
	      MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  if (SN + 1 < numprocs){
    shift2 = ind + shift1;
    MPI_Send(&shift2, 1, MPI_INT, SN+1, flag, MPI_COMM_WORLD);
  }
  for(i=0; i<NF; i++)
    if (g_index[i] != -2)
      g_index[i] += shift1;

  MPI_Barrier(MPI_COMM_WORLD);
  Update_Slave_Face_Values(g_index);
}

//============================================================================
// This function returns the number of the face that has "node" as one of
// its nodes. The search is among faces given by the array "faces". The 
// number of faces "number_faces" is reduced by one and the face is removed
// from the array.
//============================================================================
int Subdomain::find(int node, int *faces, int &number_faces){
  int i, j, k, result;

  for(i=0; i<number_faces; i++)
    for(j=0; j<3; j++)
      if (F[faces[i]].node[j] == node){
	result = faces[i];
	for(k=i+1; k<number_faces; k++)
	  faces[k-1] = faces[k];
	number_faces--;
	return result;
      }

  return -1;
}

//============================================================================
// We update the packet information : Ve, Edges and Faces. The new edges and 
// faces have to be ordered in the same way for the corresponding interfaces.
// In general the faces determine the process of refinement to conformity
// between the subdomains, the edges determine where to add the new points.
//============================================================================
void Subdomain::Update_packets(int *new_nodes, vector<queue> &new_faces,
			       vector<face> &OldF, int **face_splittings){
  int i, j, ind, ind2, number_of_edges, number_of_faces, f_ind;

  // First we update the vertices in this packet. We go through the old
  // edges and if they were refined add the middle point to the node
  // list "Ve". At the same time we update the edges that have been split.
  for(i=0; i<NPackets; i++){
    number_of_edges = Pa[i].Edges.size();
    
    for(j=0; j<number_of_edges; j++){
      ind = new_nodes[ ind_ij(level, Pa[i].Edges[j].node[0],
			      Pa[i].Edges[j].node[1]) ];
      if (ind != -1){
	Pa[i].Ve.push_back(ind);              // add the middle to the packet
	Pa[i].NPoints++;

	edge e(Pa[i].Edges[j].node[1], ind);  // update the edges
	Pa[i].Edges[j].node[1] = ind;
	Pa[i].Edges.push_back(e);
      }
    }
  }

  // Add the edges that appeared after the face splitting. We go trough the
  // faces and add the new edges =============================================
  for(i=0; i<NPackets; i++){
    number_of_faces = Pa[i].Faces.size();

    for(j=0; j<number_of_faces; j++){
      f_ind = Pa[i].Faces[j];
      ind=new_nodes[ ind_ij(level, OldF[f_ind].node[0], OldF[f_ind].node[1])];
      if (ind != -1){
	edge e( OldF[f_ind].node[2], ind);
	Pa[i].Edges.push_back( e);
      }
      
      ind2=new_nodes[ind_ij(level, OldF[f_ind].node[0], OldF[f_ind].node[2])];
      if (ind2 != -1){
	edge e(ind2, ind);
	Pa[i].Edges.push_back( e);
      }
      
      ind2=new_nodes[ind_ij(level, OldF[f_ind].node[1], OldF[f_ind].node[2])];
      if (ind2 != -1){
	edge e(ind2, ind);
	Pa[i].Edges.push_back( e);
      } 
    }
  } //=== edges updated ===================================================== 

  //  Go through the faces and if they are split update the packages
  int sub_faces[4], points[4], number_sub_faces;
  for(i=0; i<NPackets; i++){
    number_of_faces = Pa[i].Faces.size();

    for(j=0; j<number_of_faces; j++){
      f_ind = Pa[i].Faces[j];        // For every face put in sub_faces the
      number_sub_faces = 0;          // indexes of its sub-faces
      sub_faces[number_sub_faces++] = f_ind;
      if (!new_faces[f_ind].empty()){
	sub_faces[number_sub_faces++] = new_faces[f_ind].first();
	if (new_faces[f_ind].size() == 2)
	  sub_faces[number_sub_faces++] = new_faces[f_ind].get(1);
	if (!new_faces[ new_faces[f_ind].first() ].empty() )
	  sub_faces[number_sub_faces++] = 
	    new_faces[new_faces[f_ind].first()].first();
      }
      
      if (number_sub_faces>1){
	Pa[i].Faces[j]=find(OldF[f_ind].node[0], sub_faces, number_sub_faces);
	Pa[i].Faces.push_back( find( OldF[f_ind].node[1], sub_faces, 
				     number_sub_faces));
	
	if (number_sub_faces>0){
	  ind2 = new_nodes[ ind_ij(level, OldF[f_ind].node[0], 
				   OldF[f_ind].node[2]) ];
	  if (ind2!=-1){
	    ind = find( ind2, sub_faces, number_sub_faces);
	    if (ind != -1) Pa[i].Faces.push_back( ind);
	  }
	}
	
	if (number_sub_faces>0)
	  Pa[i].Faces.push_back(sub_faces[0]);
      }
    }
  }
}


//============================================================================
// Mark the faces. This function calls Mesh::MarkFaces. The refinement edge
// for a face is determined by the owner. The owners send information and
// the slaves update their values.
//============================================================================
void Subdomain::MarkFaces(double *dBuf){
  int i, j, n = 0, k, l, faces_in_packet, start, PNum;
  MPI_Request request;
  MPI_Status  Status;

  Mesh::MarkFaces();

  //=== start of interface synchronization ===================================
  MPI_Barrier(MPI_COMM_WORLD); 
  // The owners send their initialization
  for(i=0; i<NPackets; i++){
    faces_in_packet = Pa[i].Faces.size();
    if (faces_in_packet!=0 && Pa[i].Owner == SN){
      start = n;
      for(j=0; j<faces_in_packet; j++)
	for(k=0; k<2; k++)        // the first two node indexes in the face
	  for(l=0; l<3; l++)      // for the 3 coordinates
	    Bdr[n++] = Z[F[Pa[i].Faces[j]].node[k]].coord[l];
      MPI_Isend(&Bdr[start], 6*faces_in_packet, MPI_DOUBLE, Pa[i].Subdns[0],
		Pa[i].Pack_Num[0], MPI_COMM_WORLD, &request);
    }
  }

  // Non-owners read the initialization from the Owners
  for(i=0; i<NPackets; i++){
    faces_in_packet = Pa[i].Faces.size();
    if (faces_in_packet!=0 && Pa[i].Owner != SN){
      MPI_Recv( dBuf, MAX_PACKET, MPI_DOUBLE, MPI_ANY_SOURCE,
		MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
      PNum = Status.MPI_TAG;
      ReorderFaceNodes(PNum, dBuf);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

//============================================================================

//============================================================================
// Here Refinement is 1 for uniform refinement and 2 for refinement based on
// user provided function.
//============================================================================
void Subdomain::Refine(int num_levels, int Refinement){
  int i, j, *array;

  for(i=0; i<num_levels; i++){
    array = new int[NTR];
    switch (Refinement){
    case 1:
      for(j=0; j<NTR; j++) array[j] = 1;
      break;
    case 2:
      Refine_F(array, level, SN, NTR, Z, TR);
      break;
    }
    Subdomain::LocalRefine( array);
    delete [] array;
  }
}

//============================================================================

void Subdomain::Solve_HYPRE(int interactive, int visualize, int exact,
			    char *hypre_init){
  double *solution=new double [NN[level]];
  char FName1[128], FName2[128];
  int i, Refinement, refine, *array, sn, numprocs;

  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  do{
    if (SN == 0){
      cout << "Start Hypre solve\n";
      cout.flush();
    }

    cout << " " << endl;
    cout.flush();
    Solve_HYPRE(solution, hypre_init);
    cout << " " << endl;
    cout.flush();
  
    Update_Slave_Values(solution);

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (exact){
      double e1, total;

      e1 = error_L2( solution);
      e1 = e1*e1;
      MPI_Reduce(&e1, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if (SN == 0)
	cout << "|| e ||_L^2 = " << sqrt(total) << endl;
    }

    if (visualize){
      //sprintf(FName1,"/g/g99/tomov/Visual/Visualization%d.off", SN);
      sprintf(FName1,"Visual/Visualization%d.off", SN);
      PrintMCGL(FName1, solution);
      do {
	if (SN == 0){
	  cout << "Plot subdomain (subdomain number or '-1' to continue): ";
	  cout.flush();
	  cin >> sn;
	}
	MPI_Bcast(&sn, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (sn>=0 && sn < numprocs)
	  PrintMCGL(FName1, solution, sn);
      }
      while (sn!=-1);
     
      MPI_Barrier(MPI_COMM_WORLD);
    }; 

    if (interactive){
      if (SN == 0){
	cout<<"Input desired parallel refinement (1:F, 2:Uni, 3:Exit) : ";
	cout.flush();
	cin >> Refinement;
      } 
      MPI_Bcast(&Refinement, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
      refine = 1;
      array = new int[NTR];
      switch (Refinement){
      case 1:
	Refine_F(array, level, SN, NTR, Z, TR);
	break;
      case 2:
	for(i=0; i<NTR; i++) array[i] = 1;
	break;
      case 3:
	refine = 0;
	break;
      }
    
      if (refine){
	LocalRefine( array);

	double *new_sol = new double[NN[level]];
	
	delete [] solution;
	solution = new_sol;
      }
      delete []array;
    }
    else
      refine = 0;
  }
  while (refine);
  delete [] solution;
}

//============================================================================
