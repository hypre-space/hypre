#include "Packet.h"
#include <stdlib.h>
#include <stdio.h>

//============================================================================

Packet::Packet(){
  NSubdom =  0;
  Owner   = -1;
  NPoints = 0;
}

//============================================================================
// Initialize a packet. It puts only one node in it initialy - the one with 
// local index "ind". "subdns" is pointer to subdomains that share this
// packet, "ns" is their number, "sn" is subdomain number of subdomain in
// which this packet is.
//============================================================================
void Packet::Init(int *subdns, int ns, int sn, int ind){
  int i;

  NSubdom = ns;
  Subdns  = new int[NSubdom];
  Pack_Num= new int[NSubdom];
  SN      = sn;

  Ve.push_back(ind);
  NPoints = 1;

  Owner   = SN;
  for(i=0; i<NSubdom; i++){   // Fix the Owner. The subdomain with smallest  
    Subdns[i] = subdns[i];    // index is the owner for Nsubdom = 1, otherwise
    if (NSubdom == 1){        // the one with biggest index. We do this for
      if (Subdns[i] < Owner)  // load balancing.
	Owner = Subdns[i];
    }
    else
      if (Subdns[i] > Owner)
	Owner = Subdns[i];
  }
}

//============================================================================

void Packet::add( int ind){
  Ve.push_back(ind);
  NPoints = Ve.size();
}

//============================================================================
// This function sends packet's local address (la) to subdomains that share
// the packet (as MPI_TAG). We also send the subdomains that share the packet
// in order the receiving subdomain to compare and match the packets.
//============================================================================
void Packet::send(int la){
  int i;
  MPI_Request request;

  for(i=0; i<NSubdom; i++)
    MPI_Isend(Subdns,NSubdom,MPI_INT,Subdns[i],la,MPI_COMM_WORLD,&request);
}

//============================================================================
// This function receives
//============================================================================
int Packet::init_pack_num(int *Buf, int n, int source, int la){
  int i, j=0, k=0, num;
  if (n!=NSubdom) return 0;
  for(i=0; i<=NSubdom; i++)
    if (k<NSubdom && Subdns[k] == source){ num = k; k++;}
    else if (j<NSubdom && Buf[j] == SN)  j++;
    else if (k==NSubdom || j == NSubdom || (Subdns[k] != Buf[j])) return 0;
    else { k++; j++;}
  Pack_Num[num] = la;
  return 1;
}

//============================================================================

void Packet::init_pack_num_owner(int pnum){
  int i;

  if (Owner!=SN){
    for(i=0; i<NSubdom; i++)
      if (Owner == Subdns[i]){
	Pack_Num_Owner = Pack_Num[i];
	break;
      }
  }
  else
    Pack_Num_Owner = pnum;
}

//============================================================================
// Return 1 if face_number belongs to this packet or NFaces = 0. In this case
// it adds the edge to the packet. Otherwise return 0.
// Attention : if NPoints == 0 && NEdges == 0 create a new packet !!! - not
// very good to be here.
//============================================================================
int Packet::add_face(int subd_number, int face_number, int sn){

  if (Ve.size() == 0 && Edges.size() == 0 && Faces.size() == 0){     

    // Initialize a new packet
    
    NSubdom   = 1;
    Subdns    = new int[NSubdom];
    Pack_Num  = new int[NSubdom];
    SN        = sn;
    Subdns[0] = subd_number;
    
    Faces.push_back(face_number);

    if (subd_number < SN)
      Owner = SN;
    else 
      Owner = subd_number;
    return 1;
  }
  else
    if (NSubdom==1 && Subdns[0]==subd_number){
      Faces.push_back(face_number);
      return 1;
    }
    else
      return 0;
}


//============================================================================

int Packet::add_edge(int node1, int node2, set<int, less<int> > &subdomains, int sn){
  set<int, less<int> >::iterator setItr;
  int i;

  if (Ve.size() == 0 && Edges.size() == 0 && Faces.size() == 0){     
    
    // Initialize a new packet
    
    NSubdom   = subdomains.size() - 1;  // SN is included so we subtract one
    Subdns    = new int[NSubdom];
    Pack_Num  = new int[NSubdom];
    SN        = sn;
    
    edge e(node1, node2);

    i = 0;
    for(setItr = subdomains.begin(); setItr != subdomains.end(); ++setItr){
      if (Owner < (*setItr)) 
	Owner = (*setItr);
      if ((*setItr)!=SN)
	Subdns[i++] = *setItr; 
    }
    Edges.push_back(e);

    return 1;
  }
  else // check whether the edge has to be added in this packet
    if (NSubdom == (subdomains.size()-1)){
      setItr = subdomains.end();
      for(i=0; i<NSubdom; i++)
	if (subdomains.find(Subdns[i])==setItr)
	  return 0;

      edge e(node1, node2);
      Edges.push_back(e);
      return 1;
    }
    else
      return 0;
}

//============================================================================
