#include "Mesh.h"
#include <stdio.h>
#include <iomanip.h>
#include <iostream.h>

//============================================================================
// This function bisects tetrahedron with index "t" in the way given by 
// D.Arnold. The output is two marked tetrahedrons - the first one leaves 
// with the number of the old one, the second one we put at the end. We
// do the same when new faces or edges appear. The new nodes are put at the
// end. 
// input  : t; new_nodes (array of dimentsion PN, we use it to find out
//          wether to introduce a new node - if not to take the index from 
//          there); new_faces (the same but applied to faces).
// output : refine_tetr[2] - positive, containing tetrahedron index if it
//          needs refinement. Otherwise negative (no refinement enforced)
//============================================================================
void Mesh::BisectTet( int t, int *new_nodes, vector<face> &OldF,
		      vector<queue> &new_face, vector<int> &r_vector,
		      vector<int> &refined, vector<v_int> &new_conn){
  queue nill_queue;
  int i, j, start, end;
  int v[4];       // We will initialize these to be the vertices of the
                  // tetrahedron (with the first two the refinement ones
  face *faces[2]; // These will be the corresponding faces (global)
  face faces2;
  face faces3R;   // Refinement face
  int face_ind[4];// indexes of the tetrahedron's faces "faces" - the ones 
                  // opposite nodes v[4] 
  int  *VV = V[level];
  int *PNN = PN[level];

  tetrahedron *T = &TR[t];  // the tetrahedron that we refine
  int current_NTR = TR.size();

  refined[t] = 0;           // will be refined now

  face_ind[3] = T->face[T->refine];

  int flag3, flag2;

  if ( (!new_face[face_ind[3]].empty())&&
       (F[face_ind[3]]. tetr[0]!=t )){
    faces3R = OldF[ T->face[T->refine]];
    flag3   = 0;            // face faces3R was split before
  }
  else{
    faces3R = F[ T->face[T->refine]];
    flag3   = 1;   // face faces3R wasn't split before
  }

  // The refinement edge is given in terms of local face index (and the face
  // has the refinement edge given by its first two nodes. In a tetrahedron
  // node i is opposite face i. 
  if ( faces3R.node[0] > faces3R.node[1]){
    v[1] = faces3R.node[0];
    v[0] = faces3R.node[1];
  }
  else{
    v[0] = faces3R.node[0];
    v[1] = faces3R.node[1];
  }
  v[2] = faces3R.node[2];         // the last point of the face
  v[3] = T->node[ T->refine];     // the node opposite the ref face

  for(i=0; i<2; i++)
    for(j=0; j<4; j++)
      if (v[i]==T->node[j]){
	face_ind[i] = T->face[j];
	faces[i] = &F[T->face[j]];
	if (!new_face[face_ind[i]].empty())
	  if (faces[i]->tetr[0] != t)
	    faces[i] = &OldF[T->face[j]];
	break;
      }

  for(j=0; j<4; j++)
    if (v[2]==T->node[j]){
      face_ind[2] = T->face[j];
      if ( (!new_face[face_ind[2]].empty()) && 
	   (F[face_ind[2]].tetr[0] != t) ){
	faces2 = OldF[T->face[j]];
	flag2  = 0;  // face faces2 was split before  
      }
      else {
	faces2 = F[T->face[j]];
	flag2  = 1;  // face faces2 wasn't split before
      }
      break;
    }
  //========== initialization of v & faces done ==============================

  // Now find the index of the middle point (ind_new_node)
  int ind_new_node, current_NN = Z.size();  

  i = ind_ij(level, v[0], v[1]);
  if ( new_nodes[i] == -1 || new_nodes[i] == -2 ) { // it wasn't there
    ind_new_node = new_nodes[ i] = current_NN;
	
    vertex new_vertex;
    for(j=0; j<3; j++)
      new_vertex.coord[j]=(Z[v[0]].coord[j] + Z[v[1]].coord[j])/2.;

    new_vertex.Atribut = INNER;        // inner node by default
    
    Z.push_back(new_vertex);           // add it to Z
    v_int empty;
    new_conn.push_back(empty);
    new_conn[ind_new_node].push_back(v[0]);
    new_conn[ind_new_node].push_back(v[1]);

    // put it in the other direction also
    new_nodes[ind_ij(level, v[1], v[0])] = ind_new_node;
  }
  else
    ind_new_node = new_nodes[i];
  //=========== index of the new point found - in ind_new_point =============

  // Now find the indexes of faces that we split : (v[0], v[1], v[2]) - this
  // is face "faces[3]" & (v[0], v[1], v[3]), which is faces[2]
  int face2[2], face3[2];
  face f;

  int current_NF = F.size();
  // first check (v[0], v[1], v[2])
  if (flag3){                          // the face wasn't split before
    face3[0] = face_ind[3];            // this one remains the same
    face3[1] = current_NF;

    if (faces3R.tetr[0]==t) {
      if ((faces3R.tetr[1]>=0) && (refined[faces3R.tetr[1]]==0)){
	refined[faces3R.tetr[1]]=1;
	r_vector.push_back(faces3R.tetr[1]);
      }
      f.tetr[1] = faces3R.tetr[1];
    }
    else {
	if (refined[faces3R.tetr[0]]==0){
	refined[faces3R.tetr[0]] = 1;
	r_vector.push_back(faces3R.tetr[0] );
      }
      faces3R.tetr[1] = f.tetr[1] = faces3R.tetr[0];
    }

    new_face[ face_ind[3]].push( current_NF);

    faces3R.node[0] = v[0];  // the one that remains with the same index
    faces3R.node[1] = v[2];
    faces3R.node[2] = ind_new_node;
    faces3R.tetr[0] = t;     // tetr[1] will be changed by the tetr on the
                             // other side (if needed-may be on the bdr)
    f.node[0] = v[1];
    f.node[1] = v[2];
    f.node[2] = ind_new_node;
    f.tetr[0] = current_NTR;

    F.push_back( f); 
    F[face_ind[3]] = faces3R;

    OldF[face_ind[3]].tetr[0] = t;
    OldF[face_ind[3]].tetr[1] = faces3R.tetr[1];
    OldF.push_back( f);

    new_face.push_back(nill_queue);
    current_NF++;   // increase the number of the current N of Faces

    new_conn[v[2]].push_back( ind_new_node);
    new_conn[ind_new_node].push_back( v[2]);
  }
  else { // The face was split before (so we can change the one that inhereted
         // the old index). We process the face and pop from the queue the 
         // corresponding splitting face.
    face3[0] = face_ind[3];            
    face3[1] = new_face[face_ind[3]].pop(); // Old face with ind face_ind[3]
                                            // needs no more refinement
    OldF[face_ind[3]].node[0] = v[0];
    OldF[face_ind[3]].node[1] = v[2];
    OldF[face_ind[3]].node[2] = ind_new_node;

    F[face3[0]].tetr[1] = t;
    F[face3[1]].tetr[1] = current_NTR;

    OldF[face3[0]].tetr[1] = t;
    OldF[face3[1]].tetr[1] = current_NTR;
  }

  // Now check (v[0], v[1], v[3]) - faces[2]
  if (flag2){
    face2[0] = face_ind[2];            // this one remains the same
    face2[1] = current_NF;

    if (faces2.tetr[0]==t) {
      if ((faces2.tetr[1]>= 0)&&(refined[faces2.tetr[1]]==0)){
	refined[faces2.tetr[1]] = 1;
	r_vector.push_back( faces2.tetr[1]);
      }
      f.tetr[1] = faces2.tetr[1];
    }
    else {
      if (refined[faces2.tetr[0]]==0){
	refined[faces2.tetr[0]] = 1;
	r_vector.push_back( faces2.tetr[0]);
      }
      faces2.tetr[1] = f.tetr[1] = faces2.tetr[0];
    }

    new_face[face_ind[2]].push(current_NF);

    faces2.node[0] = v[0];  // the one that remains with the same index
    faces2.node[1] = v[3];
    faces2.node[2] = ind_new_node;
    faces2.tetr[0] = t;     // tetr[1] will be changed by the tetr on the
                            // other side (if needed-may be on the bdr)
    f.node[0] = v[1];
    f.node[1] = v[3];
    f.node[2] = ind_new_node;
    f.tetr[0] = current_NTR;

    F.push_back( f);
    F[face_ind[2]] = faces2;
    
    OldF[face_ind[2]].tetr[0] = t;
    OldF[face_ind[2]].tetr[1] = faces2.tetr[1];
    OldF.push_back( f);

    new_face.push_back(nill_queue);
    current_NF++;   // increase the number of the current N of Faces

    new_conn[v[3]].push_back( ind_new_node);
    new_conn[ind_new_node].push_back( v[3]);
  }
  else {  // the face was split before (so we can change the one that
          // inhereted the old index). We process the face and pop from the 
          // queue the corresponding splitting face.
    face2[0] = face_ind[2];            
    face2[1] = new_face[face_ind[2]].pop();

    OldF[face_ind[2]].node[0] = v[0];
    OldF[face_ind[2]].node[1] = v[3];
    OldF[face_ind[2]].node[2] = ind_new_node;

    F[face2[0]].tetr[1] = t;
    F[face2[1]].tetr[1] = current_NTR;

    OldF[face2[0]].tetr[1] = t;
    OldF[face2[1]].tetr[1] = current_NTR;
  }
  //======== initialization of the faces that were split done ================
  
  // Find the index for the new face and the type for the child tetrahedrons
  int child_type;
  int ind_new_face = current_NF;

  if (T->type==TYPE_PF){
    f.node[1] = ind_new_node;
    if ((faces[1]->node[0]==v[3])||(faces[1]->node[1]==v[3])){
      f.node[0] = v[3];  // in this case v[0]-v[3] is the refinement edge 
      f.node[2] = v[2];  // for face faces[1]
    }
    else {
      f.node[0] = v[2];  // in this case v[0]-v[2] is the refinement edge 
      f.node[2] = v[3];  // for face faces[1]
    }
    child_type = TYPE_A;
  }
  else {
    f.node[0] = v[2];
    f.node[1] = v[3];
    f.node[2] = ind_new_node;
    
    if (T->type==TYPE_PU)
      child_type = TYPE_PF;
    else
      child_type = TYPE_PU;
  }

  f.tetr[0] = t;
  f.tetr[1] = current_NTR;

  F.push_back(f);
  OldF.push_back(f);
  new_face.push_back(nill_queue);
  current_NF++;
  //============== new face index & tetrah types found =======================

  tetrahedron t2;   // we will put the second tetrahedron first here
  
  if (faces[0]->tetr[0] == t){
    F[face_ind[0]].tetr[0] = current_NTR;
    OldF[face_ind[0]].tetr[0] = current_NTR;
  }
  else {
    F[face_ind[0]].tetr[1] = current_NTR;
    OldF[face_ind[0]].tetr[1] = current_NTR;
  }

  // Correct the old one
  T->node[0] = v[0];           T->face[0] = ind_new_face; 
  T->node[1] = ind_new_node;   T->face[1] = face_ind[1];
  T->node[2] = v[2];           T->face[2] = face2[0];
  T->node[3] = v[3];           T->face[3] = face3[0];

  T->refine = 1;  // the local index of the face opposite the new vertex
  T->type   = child_type;

  // Correct the new one
  t2.node[0] = ind_new_node;   t2.face[0] = face_ind[0]; 
  t2.node[1] = v[1];           t2.face[1] = ind_new_face;
  t2.node[2] = v[2];           t2.face[2] = face2[1];
  t2.node[3] = v[3];           t2.face[3] = face3[1];

  t2.refine = 0;  // the local index of the face opposite the new vertex
  t2.type   = child_type;
  t2.atribut = T->atribut;

  TR.push_back( t2);
  refined.push_back(0);

  if (need_refinement(t, new_face)){
    refined[t] = 1;
    r_vector.push_back( t);
  }

  if (need_refinement(current_NTR, new_face)){
    refined[current_NTR] = 1;
    r_vector.push_back( current_NTR);
  }
}
//================= end BisectTet ============================================


//============================================================================
// r_array is array of size NTR with 0/1 showing which triangles to be refined
//============================================================================
void Mesh::LocalRefine(int *r_array){
  int i, j, end;
  vector<int> r_vector;
  vector<int> refined(NTR);

  // The following two arrays are important since from them we can construct
  // our interpolation matrices (has to be done into a sparce form).
  int *new_nodes = new int[DimPN[level]];
  vector<queue> new_faces(NF);
  vector<face> OldF( NF);
  vector<v_int> new_conn(NN[level]);
   
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
    /*
    i1 = TR.size()-1;
    BisectTet(i1,new_nodes,OldF,new_faces,r_vector,refined,new_conn);
    BisectTet(TR.size()-1,new_nodes,OldF,new_faces,r_vector,refined,new_conn);
    BisectTet(i1,new_nodes,OldF,new_faces,r_vector,refined,new_conn);
    
    BisectTet(r_vector[i],new_nodes,OldF,new_faces,r_vector,refined,new_conn);
    BisectTet(TR.size()-1,new_nodes,OldF,new_faces,r_vector,refined,new_conn);
    BisectTet(r_vector[i],new_nodes,OldF,new_faces,r_vector,refined,new_conn);
    */
  }

  // refine to get conformity
  for(i=end; i<r_vector.size(); i++)
    if (need_refinement(r_vector[i], new_faces))
      BisectTet(r_vector[i], new_nodes, OldF, new_faces, r_vector, refined,
		new_conn); 

  int flag2=1;
  while (flag2){
    flag2=0;
    for(i=0; i<TR.size(); i++)
      if (need_refinement(i, new_faces)) {
	BisectTet(i, new_nodes, OldF, new_faces, r_vector, refined, new_conn);
	flag2=1;
      }
  }

  //for(i=0; i<TR.size(); i++){
  //  if (need_refinement(i, new_faces)) 
  //    cout << "Not Conforming\n";
  //  CheckConformity(i);
  //}

  int ntr = TR.size(), nf = F.size(), nz = Z.size();
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  /*
  if (myrank == 0){
    cout << endl
	 <<"+============================================+\n"
	 <<"|     Local Refinement done. Level " << setw(2) << level+1
	 << "        |\n"
	 <<"+============================================+\n"
	 <<"| Number of tetrahedrons =" << setw(7) << ntr << ","
	 << setw(7) << ntr - NTR << " new|\n"
	 <<"| Number of faces        =" << setw(7) << NF  << ","
	 << setw(7) << nf - NF << " new|\n"
	 <<"| Number of nodes        =" << setw(7) << nz  << ", "
	 << setw(6) << nz - NN[level] << " new|\n"
	 <<"+============================================+\n\n";
  }
  */
  if (myrank == 0){
    cout << "Level " << level+1 << " : #faces = " << setw(7) << nf  << endl
	 << "          #tetr. = " << setw(7) << ntr << endl
	 << "          #nodes = " << setw(7) << nz  << endl << endl;
    cout.flush();
  }
    
  level++;
  
  NN[level] = nz;
  NTR       = ntr;
  NF        = nf;
  
  //  PrintMesh("Output");

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

  MarkFaces();
  InitializeRefinementEdges();

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
}

//============================================================================
// give the type of the tetrahedron
//============================================================================
int Mesh::tetr_type(int tetr){
  tetrahedron *T = &TR[tetr];
  int i, j, v;
  

  for( j=0;j<4; j++){
    v = T->node[j];
    if ( (v!=F[T->face[0]].node[0]) && (v!=F[T->face[0]].node[1]) &&
	 (v!=F[T->face[1]].node[0]) && (v!=F[T->face[1]].node[1]) &&
	 (v!=F[T->face[2]].node[0]) && (v!=F[T->face[2]].node[1]) &&
	 (v!=F[T->face[3]].node[0]) && (v!=F[T->face[3]].node[1]) ){
      return TYPE_PU;
    }
  }

  int v1 = F[T->face[T->refine]].node[0];
  int v2 = F[T->face[T->refine]].node[1];

  int intersect = 0, ff;

  for(i=1; i<4; i++){
    ff = T->face[(T->refine + i)%4];
    if ((v1==F[ff].node[0]) || (v1==F[ff].node[1]) ||
	(v2==F[ff].node[0]) || (v2==F[ff].node[1]))
      if (!((v1==F[ff].node[0]) && (v2==F[ff].node[1])))
	intersect++;
  }
  
  if (intersect == 0)
    return TYPE_O;
  else if (intersect==1)
    return TYPE_M;
  else return TYPE_A;
}


//============================================================================
// This function checks whether element t needs refinement. If yes it pushes
// it's number at the back of the refinement tetrahedron list. This function
// is used from BisectTet to determine whether one of the two childs needs
// refinement.
//============================================================================
int Mesh::need_refinement(int t, vector<queue> &new_face){
  int i, *faces = TR[t].face;

  for(i=0; i<4; i++)
    if (!new_face[faces[i]].empty()) // there may be need for refinement
      if ( F[faces[i]].tetr[0] != t)
	return 1;

  return 0;
}


//============================================================================
// After the refinement of the current level this function initializes V & PN
// for the new level.
//============================================================================
void Mesh::InitializeNewEdgeStructure(int *new_nodes, 
				      vector<v_int> &new_conn){
  int s1, s2, e1, e2, e3, i, j, k, n = 0, ll = 0;

  e1 = NN[level];
  for(i=0; i<e1; i++)
    ll += new_conn[i].size();
    
  ll += DimPN[level-1] + NN[level] - NN[level-1];

  PN[level] = new int[ll];
  V [level] = new int[NN[level]+1];
  DimPN[level] = ll;

  int  *VV =  V[level-1], *VNew =  V[level];
  int *PNO = PN[level-1], *PNN  = PN[level];

  VNew[0] = 0;
  e1 = NN[level-1];
  for(i=0; i<e1; i++){          // for nodes on the old level
    s2 = VV[i]; 
    e2 = VV[i+1];
    for(j = s2; j<e2; j++){     // for nodes connected to i on the old level
      if (new_nodes[j] != -1)
	PNN[n++] = new_nodes[j];
      else
	PNN[n++] = PNO[j];
    }
      
    e3 = new_conn[i].size();
    for(k=0; k<e3; k++)        // for new connections
      PNN[n++] = new_conn[i][k];
	
    VNew[i+1] = n;
  }

  s1 = NN[level-1];
  e1 = NN[level];
  for(i=s1; i<e1; i++){        // for all the new nodes
    PNN[n++] = i;
    e2 = new_conn[i].size();
    for(k=0; k<e2; k++)        // for new connections
      PNN[n++] = new_conn[i][k];
    VNew[i+1] = VNew[i] + e2+1;
  }
}


//============================================================================
// After the refinement of the current level this function initializes V & PN
// for the new level.
//============================================================================
void Mesh::InitializeNewHBEdgeStructure(vector<v_int> &new_conn){
  int s1, e1, e2, e3, i, j, n = 0, ll = 0;

  s1 = NN[level-1];
  e1 = NN[level];
  e2 = e1 - s1;
  for(i=s1; i<e1; i++){
    e3 = new_conn[i].size();
    for(j=0; j<e3; j++)
      if (new_conn[i][j]>=s1)
	ll ++;
  }
  ll += e2;

  PNHB[level] = new int[ll];
  VHB[ level] = new int[ e2 + 1];
  DimPNHB[level] = ll;

  int  *VV =  VHB[level], *PNN  = PNHB[level];

  VV[0] = 0;
  for(i=s1; i<e1; i++){        // for all the new nodes
    PNN[n++] = i-s1;
    e2 = new_conn[i].size();
    for(j=0; j<e2; j++){      // for new connections
      if (new_conn[i][j] >= s1)
	PNN[n++] = new_conn[i][j]-s1;
    }
    VV[i+1-s1] = n;
  }
}


//============================================================================
// After the refinement of the current level this function initializes V & PN
// for the interpolation matrix for the new level. This is Interpolaiton from
// level "level -1" to "level".
//============================================================================
void Mesh::InitializeNewIEdgeStructure(int *new_nodes){
  int i,j, e1, e2;

  VI[level] = new int[ NN[level]+1];
  int *VV = VI[level];

  DimPNI[level] = NN[level-1] + 2*(NN[level]-NN[level-1]);
  PNI[level]    = new int[DimPNI[level]];
  int *PNPN = PNI[level];

  VV[0] = 0;
  e1 = NN[level-1];
  for(i=0; i<e1; i++){
    VV[i+1] = VV[i] + 1;
    PNPN[i]=i;
  }

  e1 = NN[level];
  for(i=NN[level-1]; i<e1; i++)
    VV[i+1] = VV[i] + 2;

  e1 = NN[level-1];
  for(i=0; i<e1; i++){
    e2 = V[level-1][i+1];
    for(j = V[level-1][i]+1; j< e2; j++)
      if (new_nodes[j]!=-1){
	PNPN[VV[new_nodes[j]]]   = i;
	PNPN[VV[new_nodes[j]]+1] = PN[level-1][j];
      }
  }
}



//============================================================================
int order(int a1, int a2, int b1, int b2);

// Order the nodes in the faces as in the initialization step.
//============================================================================
void Mesh::MarkFaces(){
  int i, j, end = F.size(), f[3], ind;
  double d[3];

  for(i=0; i< end; i++){
    for(j=0; j<3; j++) {
      f[j] = F[i].node[j];
      d[j] = length(F[i].node[j], F[i].node[(j+1)%3]);
    }
    ind = 0;
    for(j=1; j<3; j++)
      if (d[ind] < d[j]) ind = j;
      else
	//if (d[ind]<d[j]+0.00000000000001)
	if (d[ind]==d[j])
	  if (order(F[i].node[ind], F[i].node[(ind+1)%3],
		    F[i].node[j], F[i].node[(j+1)%3]))
	    ind = j;
    
    F[i].node[0] = f[ind];
    F[i].node[1] = f[(ind+1)%3];
    F[i].node[2] = f[(ind+2)%3];
  }
}

//============================================================================

void Mesh::UniformRefine(){
  int *array, i;
  
  array = new int[NTR];
  for(i=0; i<NTR; i++) array[i] = 1;
  LocalRefine(array);
  delete [] array;
}

//============================================================================
