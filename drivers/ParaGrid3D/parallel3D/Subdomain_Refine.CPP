#include <stdio.h>
#include <math.h>
#include <iostream.h>
#include <iomanip.h>
#include "definitions.h"
#include "Subdomain.h"
#include "extension_MPI.h"


//============================================================================
// If points p1 and p2 are the same ,up to a certain tolerance, return 1
// otherwise return 0.
//============================================================================
int equal(double *p1, double *p2){
  int i;
  for(i=0; i<3; i++)
    if (fabs(p1[i] - p2[i]) > 0.000001)
      return 0;
  return 1;
}

//============================================================================
// The following function is equivalent of Mesh::LocalRefine. The only new 
// part is that after a conformity is reached we send to the other subdomains
// information about the refinement on the interfaces. Based on this info
// we do more refinement if needed.
//============================================================================
void Subdomain::LocalRefine(int *r_array){
  int i, j, ind, start, end, edges_in_packet, faces_in_packet, ONTR=NTR;

  int   *iBuf  = new int[20000];
  double *dBuf = new double[20000];
  //int    iBuf[MAX_PACKET];
  //double dBuf[MAX_PACKET];

  MarkFaces(dBuf);
  InitializeRefinementEdges();

  MPI_Request request;
  MPI_Status  Status;
  int PNum, PSource;

  vector<int> r_vector;
  vector<int> refined(NTR);

  // The following two arrays are important since from them we can construct
  // our interpolation matrices (has to be done into a sparce form).
  int *new_nodes = new int[DimPN[level]];
  vector<queue> new_faces(NF);
  vector<face> OldF( NF);
  vector<v_int> new_conn(NN[level]);

  // face_splittings gives in how many times a face on the boundary has been
  // splitted, i.e the values are 0, 1, 2 or 3. face_splittings_coord gives
  // the coordinate of the new node where the splitting occurred in case
  // face_splittings == 2.
  int **face_splittings = new int*[NPackets];
  double3 **face_splittings_coord = new double3*[NPackets];
  for(i=0; i<NPackets; i++){
    faces_in_packet = Pa[i].Faces.size();
    face_splittings[i]       = new      int[faces_in_packet];
    face_splittings_coord[i] = new  double3[faces_in_packet];
  }

  for(i=0; i<DimPN[level]; i++) new_nodes[i] =   -1;
  for(i=0; i<NF;           i++)      OldF[i] = F[i];

  for(i=0; i<NTR; i++){
    if (r_array[i])
      r_vector.push_back(i);
    refined[i] = r_array[i];
  }

  int i1;
  end = r_vector.size();
  for(i=0; i<end; i++){
    // Bisect the tetrahedron marked for refinement
    BisectTet(r_vector[i],new_nodes,OldF,new_faces,r_vector,refined,new_conn);
    
    i1 = TR.size()-1;
    BisectTet(i1, new_nodes, OldF, new_faces, r_vector, refined, new_conn);
    BisectTet(TR.size()-1,new_nodes,OldF,new_faces,r_vector,refined,new_conn);
    BisectTet(i1, new_nodes, OldF, new_faces, r_vector, refined, new_conn);
    
    BisectTet(r_vector[i],new_nodes,OldF,new_faces,r_vector,refined,new_conn);
    BisectTet(TR.size()-1,new_nodes,OldF,new_faces,r_vector,refined,new_conn);
    BisectTet(r_vector[i],new_nodes,OldF,new_faces,r_vector,refined,new_conn);
  }

  for(i=end; i<r_vector.size(); i++)
    if (need_refinement(r_vector[i], new_faces))
      BisectTet(r_vector[i], new_nodes, OldF, new_faces, r_vector, refined,
		new_conn);
  int flag2 = 1, face_ind;
  double middle[3];
  while (flag2){
    flag2=0;
    for(i=0; i<TR.size(); i++)
      if (need_refinement(i, new_faces)) {
	BisectTet(i, new_nodes, OldF, new_faces, r_vector, refined, new_conn);
	flag2=1;
      }

    //========================================================================
    // For parallel computations at this point we have synchronization and
    // check if the mesh is conforming (this is the new part).
    //========================================================================
    if (flag2==0){                 // only when the mesh is locally conforming
      // This is a set of tetrahedrons that have to be refined in order to
      // achieve global conformity
      set<int, less<int> > refine_tetr;
      set<int, less<int> >::iterator setItr;

      //======================================================================
      // The boundary faces will determine the global refinement to conformity
      // process. First we go through the faces on the boundary and initia-
      // lize structure face_splittings. Then we exchange that info with the
      // corresponding subdomains.
      //==== (1) send number of boundary face splittings =============
      MPI_Barrier(MPI_COMM_WORLD);
      for(i=0; i<NPackets; i++){
	faces_in_packet = Pa[i].Faces.size();
	for(j=0; j<faces_in_packet; j++)
	  get_face_splittings(Pa[i].Faces[j], face_splittings[i][j],
			      face_splittings_coord[i][j], new_faces, OldF);

	if (faces_in_packet!=0){ // send the number of splittings
	  MPI_Isend( face_splittings[i], faces_in_packet, MPI_INT,
		     Pa[i].Subdns[0], Pa[i].Pack_Num[0], 
		     MPI_COMM_WORLD, &request);
	  MPI_Isend( face_splittings_coord[i], 3*faces_in_packet, MPI_DOUBLE,
		     Pa[i].Subdns[0], Pa[i].Pack_Num[0], 
		     MPI_COMM_WORLD, &request);
	}
      }
      //==== (2) receive the number of boundary face splittings ======
      for(i=0; i<NPackets; i++){
	faces_in_packet = Pa[i].Faces.size();
	if (faces_in_packet!=0){
	  MPI_Recv( iBuf, MAX_PACKET, MPI_INT, MPI_ANY_SOURCE,
		    MPI_ANY_TAG, MPI_COMM_WORLD, &Status);
	  PNum = Status.MPI_TAG;
	  PSource = Status.MPI_SOURCE;
	  MPI_Recv( Bdr, 6*MAX_PACKET, MPI_DOUBLE, PSource, MPI_ANY_TAG,
		    MPI_COMM_WORLD, &Status);
	  
	  faces_in_packet = Pa[PNum].Faces.size();
	  for(j=0; j<faces_in_packet; j++){
	    face_ind = Pa[PNum].Faces[j];
	    if (iBuf[j] == face_splittings[PNum][j] && iBuf[j] == 2){
	      // There may be need for refinement in this case
	      if ( !equal(Bdr+3*j, face_splittings_coord[PNum][j])){ 
		if (new_faces[face_ind].size() == 2)
		  ind = new_faces[face_ind].first();
		else 
		  ind = face_ind;
		refine_tetr.insert(F[ind].tetr[0]);
	      }
	    }
	    else if ( face_splittings[PNum][j] < iBuf[j] )
	      if (face_splittings[PNum][j] == 0)
		refine_tetr.insert(F[face_ind].tetr[0]);
	      else if (face_splittings[PNum][j]==1 && iBuf[j] ==3){
		refine_tetr.insert(F[face_ind].tetr[0]);
		refine_tetr.insert(F[new_faces[face_ind].first()].tetr[0]);}
	      else if (face_splittings[PNum][j]==2){
		if (new_faces[face_ind].size() == 2)
		  refine_tetr.insert(F[new_faces[face_ind].first()].tetr[0]);
		else
		  refine_tetr.insert(F[face_ind].tetr[0]);
	      }
	      else if (face_splittings[PNum][j]==1 && iBuf[j] == 2){
		GetMiddleEdge(F[face_ind].node[0],F[face_ind].node[1],middle);
		if (equal(middle, Bdr+3*j))
		  refine_tetr.insert(F[face_ind].tetr[0]);
		else
		  refine_tetr.insert(F[new_faces[face_ind].first()].tetr[0]);
	      }
	  }
	}
      }

      //==== (3) refine the marked tetrahedrons ==============================
      if (refine_tetr.size() != 0){
	flag2 = 1;
	for(setItr=refine_tetr.begin(); setItr!=refine_tetr.end(); ++setItr)
	  BisectTet(*setItr, new_nodes, OldF, new_faces, r_vector, 
	  	    refined, new_conn);
      }
      // Now broadcast the flag. If only one flag == 1 the refinement 
      // should continue. The result of logical or from the flags of all
      // the processors will be written in flag. Because of Allreduce there
      // is no need for additional synchronization. 
      start = flag2;
      MPI_Allreduce( &start, &flag2, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD );
    }
    //========================================================================
    //==== end of procedure to get conformity between the subdomains =========
    //========================================================================
  }
  // Now the mesh is conforming. Update the packet information.

  Update_packets(new_nodes, new_faces, OldF, face_splittings);
  
  for(i=0; i<NPackets; i++){
    delete [] face_splittings[i];
    delete [] face_splittings_coord[i];
  }
  delete [] face_splittings;
  delete [] face_splittings_coord;
  /*
  cout << "Checking conformity ... SN = " << SN << endl;
  for(i=0; i<TR.size(); i++){
    if (need_refinement(i, new_faces)) 
      cout << "Not Conforming\n";
    CheckConformity(i);
  }
  cout << "done.\n";
  */

  int ntr = TR.size(), nf = F.size(), nz = Z.size(), total;

  /*  
  critical();
  cout << endl 
       <<"+============================================+\n"
       <<"|  Local Refinement done, Level " << setw(2) << level+1
       <<". SN = " << setw(2) << SN << "  |\n"
       <<"+============================================+\n"
       <<"| Number of tetrahedrons =" << setw(7) << ntr
       <<"," << setw(7) << ntr - NTR  << " new|\n"
       <<"| Number of faces        =" << setw(7) << nf 
       <<"," << setw(7) << nf - NF << " new|\n"
       <<"| Number of nodes        =" << setw(7) << nz
       <<", " << setw(6) << nz-NN[level] << " new|\n"
       <<"+============================================+\n\n";
  end_critical();
  */
  
  critical();
  if (SN == 0)
    cout << "#faces =" <<setw(7) << nf;
  else
    cout << " +" << setw(7)<<nf;
  cout.flush();
  end_critical();
  MPI_Reduce(&nf, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (SN == 0) cout << " = " << total << endl;

  critical();
  if (SN == 0)
    cout << "#tetr. =" <<setw(7) << ntr;
  else
    cout << " +" << setw(7)<<ntr;
  cout.flush();
  end_critical();
  MPI_Reduce(&ntr, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (SN == 0) cout << " = " << total << endl;

  critical();
  if (SN == 0)
    cout << "#nodes ="<< setw(7) << nz;
  else
    cout << " +" << setw(7) << nz;
  cout.flush();
  end_critical();
  MPI_Reduce(&nz, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if (SN == 0) cout << " = " << total << endl;
  if (SN == 0) { cout << endl; cout.flush();}
  

  
  level++;
 
  NN[level] = nz;
  NTR       = ntr;
  NF        = nf;

  // Here we initialize the structures used in the solvers. Later, according
  // to the solver that we plan to use, we may ask which structures to
  // initialize.
  InitializeNewEdgeStructure(new_nodes, new_conn);
  #if MAINTAIN_HB == ON
    InitializeNewHBEdgeStructure( new_conn);
  #endif

  InitializeNewIEdgeStructure(new_nodes);
  if (MAINTAIN_MG == OFF && MAINTAIN_HB == OFF){
    delete []  V[level-1];
    delete [] PN[level-1];
  }

  /* Do Atribut initialization */
  for(i=0; i<NF; i++)
    if (F[i].tetr[1] == DIRICHLET)
      for(j=0;j<3; j++)
	Z[F[i].node[j]].Atribut = DIRICHLET;
  
  // Update the Dirichlet nodes
  delete [] Dir;
  delete [] Atribut;
  dim_Dir[level] = dim_Dir[level-1];
  end = NN[level];
  for(i=NN[level-1]; i<end; i++)
    if (Z[i].Atribut == DIRICHLET)
      dim_Dir[level]++;
  
  Dir     = new int[dim_Dir[level]];
  Atribut = new int[NN[level]];
 
  j=0;
  for(i=0; i<end; i++){
    Atribut[i] = Z[i].Atribut;
    if (Z[i].Atribut == DIRICHLET)
      Dir[j++] = i;
  }

  // After densen-ing we destroy the arrays
  delete [] new_nodes;

  // TTT
  delete [] iBuf;
  delete [] dBuf;
}

//============================================================================
// This function gets the number of splittings for face with index face_ind
// in number_of_splittings. The coordinates of the new points (resulted from
// the splitting are stored in coord. The other two arguments are input.
//============================================================================
void Subdomain::get_face_splittings(int face_ind, int &number_of_splittings, 
				    double3 &coord, vector<queue> &new_faces,
				    vector<face> &OldF){
  int i, j, ind1, ind2, index_middle;

  // determine the number of splittings
  number_of_splittings = 0;
  if (!new_faces[face_ind].empty()){
    number_of_splittings = 1;             // split at least once
    if (new_faces[face_ind].size() == 2)  // face_ind splitted again
      number_of_splittings++;
    if ( !new_faces[ new_faces[face_ind].first() ].empty() )
      number_of_splittings++;
  }
  
  // if number_of_splittings == 2 determine the coordinates ot the splitting
  // node (the second one since the first one is ! defined)
  if (number_of_splittings == 2){         // ind1 gives the face in which
    if (new_faces[face_ind].size() == 2)  // we will find the node
      ind1 = face_ind;
    else
      ind1 = new_faces[ face_ind].first();

    ind2 = new_faces[face_ind].first();   // it should be != first splitting
    for(i=0; i<3; i++)                    // node (with index index_middle)
      if ((index_middle=OldF[ind2].node[i]) >= NN[level])
	break;
  
    for(i=0; i<3; i++)                    // get the coordinates
      if (F[ind1].node[i] >= NN[level] && F[ind1].node[i] != index_middle){
	for(j=0; j<3; j++)
	  coord[j] = Z[ F[ind1].node[i] ].coord[j];
	break;
      }
  }
}

//============================================================================
// The face nodes in packet packet_number are reordered according to the
// coordinates given in coord.
//============================================================================
void Subdomain::ReorderFaceNodes(int packet_number, double *coord){
  int i, j, k, l, n, face_ind, v[3], error_flag; 
  int number_of_faces = Pa[packet_number].Faces.size();

  n = 0;
  for(i=0; i<number_of_faces; i++){
    face_ind = Pa[packet_number].Faces[i];
    
    // reorder face with index face_ind : the node coordinates start
    // from coord[n]
    for(j=0; j<3; j++) v[j] = F[face_ind].node[j];
    for(j=0; j<2; j++){        // Fix the first two points, the 3rd one
      error_flag = 1;
      for(k=0; k<3-j; k++)     // will be automatically determined
	if (equal(Z[v[k]].GetCoord(), coord + n + 3*j)){
	  F[face_ind].node[j] = v[k];
	  for(l=k+1; l<3-j; l++)
	    v[l-1] = v[l];
	  error_flag = 0;
	  break;
	}
      if (error_flag)
	cout << "ERROR!!!\n";
    }
    F[face_ind].node[2] = v[0];
    n += 6;
  }
}

//============================================================================
