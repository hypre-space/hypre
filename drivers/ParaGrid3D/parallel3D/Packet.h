//
// Packet.h - class "Packet" declaration
//

#ifndef PACKET
#define PACKET

#include "definitions.h"
#include "Mesh_Classes.h"
#include <set.h>


class Subdomain;


class Packet{

friend class Mesh;
friend class Subdomain;

protected:

  int NSubdom;        // Number of subdomains sharing this packet
  int Owner;          // The subdomain that owns the nodes in the packet
  int Pack_Num_Owner; // Address of the packet in Owner subdomain
  int SN;             // Subdomain number of this packet
  int NPoints;        // Number of points in this packet
  		 
  vector<int> Ve;     // Indices for the nodes in the Packet in this subdomain
  int *Subdns;        // Indices of the subdns with which the packet is conn 
  int *Pack_Num;      // Address of the packet in the corresponding subdomain

  //=== The following are used for parallel local refinement ================
  vector<edge> Edges;  // Edges in this packet
  vector<int>  Faces;  // Faces in this packet
  //=========================================================================

public:

  Packet();           // constructor 
  void Init(int *subdns, int ns, int sn, int ind);

  void add(int ind);
  void send(int la);
  int  init_pack_num(int *, int, int, int);
  void init_pack_num_owner(int pnum);

  int add_face(int subd_number, int face_number, int sn);
  int add_edge(int node1, int node2, set<int, less<int> > &subdomains, int sn);
};

#endif // PACKET


