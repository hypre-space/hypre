#include <stdio.h>
#include <iostream.h>
#include <vector.h>
#include <set.h>

#include "Subdomain.h"


//============================================================================
// If arrays a and b of size n are equal return 1, otherwise 0.
//============================================================================
int equal(int *a, int *b, int n){
  int i;
  for(i=0; i<n; i++)
    if (a[i]!=b[i]) return 0;
  return 1;
}


//============================================================================
// Subdomain constructor. sn is the subdomain index, m is pointer to the
// coarse mesh that has been partitioned, tr is pointer to integers giving
// to which subdomain a tetrahedron belongs.
//============================================================================
Subdomain::Subdomain(int sn, Method *m, int *tr){
  int i, j, num_nodes = m->GetNN();

  Bdr = new double[50000];

  level = 0;
  int *newN = new int[num_nodes];
  int *newF = new int[m->NF];
  int *newT = new int[m->NTR];

  SN = sn;

  NN[0] = 0;
  NTR   = 0;
  NF    = 0;

  for(i=0; i<num_nodes; i++) newN[i] = -1;
  for(i=0; i<m->NF; i++)    newF[i] = -10;
  for(i=0; i<m->NTR; i++) 
    if (tr[i] == SN)
      newT[i] = NTR++;
  NTR = 0;

  vertex new_vertex;
  face   new_face;
  tetrahedron new_tetrahedron;

  for(i=0; i<m->NTR; i++){   // we go through all the tetrahedrons and for the
    if (tr[i] == SN){        // ones belonging to the considered subdomain do:

      TR.push_back(new_tetrahedron);

      //=== Fix nodes & tetrahedrons =========================================
      for(j=0; j<4; j++){                     // for the 4 vertices
	if (newN[m->TR[i].node[j]] == -1){    // new node
	  newN[m->TR[i].node[j]] = NN[0];

	  Z.push_back( new_vertex);

	  Z[NN[0]].coord[0] = m->Z[m->TR[i].node[j]].coord[0];
	  Z[NN[0]].coord[1] = m->Z[m->TR[i].node[j]].coord[1];
	  Z[NN[0]].coord[2] = m->Z[m->TR[i].node[j]].coord[2];

	  Z[NN[0]].Atribut  = m->Z[m->TR[i].node[j]].Atribut;

	  TR[NTR].node[j] = NN[0]++;
	}
	else                                 // node already in the structure
	  TR[NTR].node[j] = newN[m->TR[i].node[j]];
      }

      //=== Fix faces ========================================================
      for(j=0; j<4; j++){                     // for the 4 faces
	if (newF[m->TR[i].face[j]] == -10){   // new face
	  newF[m->TR[i].face[j]] = NF;
	  
	  F.push_back( new_face);

	  F[NF].node[0] = newN[m->F[m->TR[i].face[j]].node[0]];
	  F[NF].node[1] = newN[m->F[m->TR[i].face[j]].node[1]];
	  F[NF].node[2] = newN[m->F[m->TR[i].face[j]].node[2]];

	  TR[NTR].face[j] = NF;
	  if (m->F[m->TR[i].face[j]].tetr[0] == i){
	    F[NF].tetr[0] = NTR;
	    if (m->F[m->TR[i].face[j]].tetr[1] >=0)      // if not boundary
	      if (tr[m->F[m->TR[i].face[j]].tetr[1]]==SN)// points this subd
		F[NF].tetr[1] = newT[m->F[m->TR[i].face[j]].tetr[1]];
	      else
		F[NF].tetr[1] = SHIFT - tr[m->F[m->TR[i].face[j]].tetr[1]];
	    else                                         // boundary face
	      F[NF].tetr[1] = m->F[m->TR[i].face[j]].tetr[1];
	  }
	  else {
	    F[NF].tetr[0] = NTR;
	    if (m->F[m->TR[i].face[j]].tetr[0] >=0)      // if not boundary
	      if (tr[m->F[m->TR[i].face[j]].tetr[0]]==SN)// points this subd
		F[NF].tetr[1] = newT[m->F[m->TR[i].face[j]].tetr[0]];
	      else
		F[NF].tetr[1] = SHIFT - tr[m->F[m->TR[i].face[j]].tetr[0]];
	    else                                         // boundary face
	      F[NF].tetr[1] = m->F[m->TR[i].face[j]].tetr[0];
	    /*
	    F[NF].tetr[1] = NTR;
	    if (m->F[m->TR[i].face[j]].tetr[0] >=0) 
	      if (tr[m->F[m->TR[i].face[j]].tetr[0]] == SN)
		F[NF].tetr[0] = newT[m->F[m->TR[i].face[j]].tetr[0]];
	      else
		F[NF].tetr[0] = SHIFT - tr[m->F[m->TR[i].face[j]].tetr[0]];
	    else
	      F[NF].tetr[0] = m->F[m->TR[i].face[j]].tetr[0];
	    */
	  }
	  NF++;
	}
	else
	  TR[NTR].face[j] = newF[m->TR[i].face[j]];
      }

      TR[NTR].refine  = m->TR[i].refine;
      TR[NTR].type    = m->TR[i].type;
      TR[NTR].atribut = m->TR[i].atribut;

      NTR++;
    }
  }

  //=== Initialize the packets ===============================================
  int_16  *S;                         // in which subdomain every node is
  int   *NS;                         // number of subdomains for every node
  int n, k, l, found;
  S  = new int_16[NN[0]];
  NS = new int[NN[0]];

  for(i=0; i<NN[0]; i++) NS[i] = 0;

  //=== First we initialize structure S and array NS. S gives us each node ===
  //=== in which subdomain belongs.                                        ===
  for(i=0; i<m->NTR; i++){

    if ( tr[i] != SN){
      for(j=0; j<4; j++){             // for every node
	n = m->TR[i].node[j];

	if ((newN[n] != -1) && (m->Z[n].Atribut != DIRICHLET)){
	  // the node is in our subdomain and is not Dirichlet
	  found = 0;
	  for(k=0; k<NS[newN[n]]; k++){
	    if (S[newN[n]][k] == tr[i]){ found = 1; break;}
	    if (S[newN[n]][k] > tr[i]){
	      for(l=NS[newN[n]]; l>k; l--)
		S[newN[n]][l] = S[newN[n]][l-1];
	      S[newN[n]][k] = tr[i];
	      NS[newN[n]]++;
	      found = 1;
	      break;
	    }
	  }
	  if (!found)
	    S[newN[n]][NS[newN[n]]++] = tr[i];
	}
      }
    }  // end if
  }

  //=== Now, using S we initialize the packages. We go through the nodes  ====
  //=== in the whole domain since they are ordered, i.e. the constructed  ====
  //=== packages will have their nodes in the right order.                ====

  NPackets = 0;
  for(i=0; i<num_nodes; i++){

    if ((newN[i]!=-1)&&(NS[newN[i]] != 0)){   // make package with this node

      Pa[NPackets].Init(S[newN[i]], NS[newN[i]], SN, newN[i]);

      for(j=i+1; j<num_nodes; j++){
	if ((newN[j]!=-1)&&(NS[newN[j]] == NS[newN[i]]))
	  if (equal(S[newN[i]], S[newN[j]], NS[newN[i]])){
	    Pa[NPackets].add( newN[j]);
	    NS[newN[j]] = 0;
	  }
      }
      NPackets++;
    }
  } 

  //=== Initialize the boundary faces ========================================
  // We go through the global index so that the order in the different 
  // subdomains to be the same.
  for(i=0; i<m->NF; i++){            // for all the faces in this subdomain
    if ( m->F[i].tetr[0]>=0 && m->F[i].tetr[1]>=0 )
      if ( tr[m->F[i].tetr[0]] == SN && tr[m->F[i].tetr[1]] != SN )
	add_face_to_packet( tr[m->F[i].tetr[1]], newF[i]);
      else 
	if (tr[m->F[i].tetr[1]] == SN && tr[m->F[i].tetr[0]] != SN )
	  add_face_to_packet( tr[m->F[i].tetr[0]], newF[i]);
  }

  //=== Initialize the boundary edges ========================================
  int old_level = m->GetLevel(), end;
  int *VV = m->GetV(old_level), *PNN = m->GetPN(old_level);
  int dimensionPNN = m->GetPNDimension(old_level);
  set<int, less<int> > *edge_subdomain = new set<int, less<int> >[dimensionPNN];
  int *face_nodes, i1, i2, i3;
  
  for(i=0; i<m->NF; i++){             // for all the faces in this subdomain

    if (m->F[i].tetr[1]>=0 && (tr[m->F[i].tetr[0]]!=tr[m->F[i].tetr[1]])){
      face_nodes = m->F[i].GetNode();
      
      if ( m->Z[face_nodes[0]].Atribut!=DIRICHLET ||
	   m->Z[face_nodes[1]].Atribut!=DIRICHLET ){
	if (face_nodes[0]<face_nodes[1]){i1=face_nodes[0];i2=face_nodes[1];}
	else {i1=face_nodes[1];i2=face_nodes[0];}
	i3 = m->ind_ij(old_level, i1, i2);
	edge_subdomain[i3].insert(tr[m->F[i].tetr[0]]);
	edge_subdomain[i3].insert(tr[m->F[i].tetr[1]]);
      }

      if ( m->Z[face_nodes[1]].Atribut!=DIRICHLET ||
	   m->Z[face_nodes[2]].Atribut!=DIRICHLET ){
	if (face_nodes[1]<face_nodes[2]){i1=face_nodes[1];i2=face_nodes[2];}
	else {i1=face_nodes[2];i2=face_nodes[1];}
	i3 = m->ind_ij(old_level, i1, i2);
	edge_subdomain[i3].insert(tr[m->F[i].tetr[0]]);
	edge_subdomain[i3].insert(tr[m->F[i].tetr[1]]);
      }
      
      if ( m->Z[face_nodes[0]].Atribut!=DIRICHLET ||
	   m->Z[face_nodes[2]].Atribut!=DIRICHLET ){
	if (face_nodes[0]<face_nodes[2]){i1=face_nodes[0];i2=face_nodes[2];}
	else {i1=face_nodes[2];i2=face_nodes[0];}
	i3 = m->ind_ij(old_level, i1, i2);
	edge_subdomain[i3].insert(tr[m->F[i].tetr[0]]);
	edge_subdomain[i3].insert(tr[m->F[i].tetr[1]]);
      }
    }
  }
  
  for(i=0; i<num_nodes; i++){
    end   = VV[i+1];
    for(j=VV[i]; j < end; j++){
      if (i<PNN[j] && edge_subdomain[j].find(SN)!=edge_subdomain[j].end())
	// edge (i, PNN[j]) is found in subdomain set edge_subdomain[j]
	add_edge_to_packet(newN[i], newN[PNN[j]],edge_subdomain[j]);
    } 
  }

  delete [] edge_subdomain;

  //=== Fix the local addresses ==============================================
  int Buf[16];
  MPI_Status Status;
  
  for(i=0; i<NPackets; i++) Pa[i].send(i);
  for(i=0; i<NPackets; i++)
    for(j=0; j<Pa[i].NSubdom; j++){
      MPI_Recv( Buf, 16, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
		MPI_COMM_WORLD, &Status);
      MPI_Get_count( &Status, MPI_INT, &n);
      for(k=0; k<NPackets; k++)     // find which packet has been received
	if (Pa[k].init_pack_num(Buf, n, Status.MPI_SOURCE, Status.MPI_TAG))
	  break;
    }

  if (NPackets >= PACKETS){
    cout << "Increase PACKETS in definitions.h. Exit.\n";
    exit(1);
  }
  for(i=0; i<NPackets; i++)
    Pa[i].init_pack_num_owner(i);
  //==== Packet initialization done ==========================================

  delete [] NS;
  delete [] S;

  delete [] newT;
  delete [] newF;
  delete [] newN;

  delete m;                          // delete the starting mesh

  InitializeDirichletBdr();
  InitializeEdgeStructure();
  
  // Later this should be used from Method
  A   = new p_real[LEVEL];
  b   = new double[NN[0]];

  GLOBAL = new Matrix[LEVEL];
  
  //=== new =======================
  #if MAINTAIN_HB == ON
    A_HB   = new p_double[LEVEL];
    HB     = new Matrix[LEVEL];
  #endif
  
  I      = new p_double[LEVEL];
  Interp = new Matrix[LEVEL];

  Create_level_matrices();
  GLOBAL[level].InitSubdomain( this);
  //===============================
}


//============================================================================
// Initialize the Dirichlet boundary for this subdomain.
//============================================================================
void Subdomain::InitializeDirichletBdr(){
  int i;

  Atribut = new int[NN[0]];

  // Initialize Dirichlet bdr nodes
  dim_Dir[0] = 0;
  for(i=0; i<NN[0]; i++) {
    Atribut[i] = Z[i].Atribut;
    if (Z[i].Atribut == DIRICHLET) dim_Dir[0]++;
  }

  Dir = new int[dim_Dir[0]];
  int next = 0;
  for(i=0; i<NN[0]; i++)
    if (Z[i].Atribut==DIRICHLET) Dir[next++] = i;
}

//============================================================================
