#ifndef _MESH_MIXED_
#define _MESH_MIXED_

#include "Mesh.h"
#include "definitions.h"

class MeshMixed: public virtual Mesh{
  
  protected :
    int  DimPN_A[LEVEL];   // dimension of the array PN
    int     *V_A[LEVEL];   // V[i][j] is start position in PN from which 
                           // we put the nodes with which node j is connected.
    int    *PN_A[LEVEL];   // Their number is V[i][j+1] - V[i][j]
 
    int  DimPN_B[LEVEL];   // dimension of the array PN
    int     *V_B[LEVEL];   // V[i][j] is start position in PN from which 
                           // we put the nodes with which node j is connected.
    int    *PN_B[LEVEL];   // Their number is V[i][j+1] - V[i][j]

  public:

    MeshMixed() {}
    MeshMixed(char *f_name);   // Construct mesh based on "triangle"

    void InitializeVPN(int level);
    void add_new_A(int, int, int, int, int, int, int *);
    void add_new_B(int, int, int, int, int, int, int *);

    int ind_ij(int *V, int *PN, int i, int j);
};

#endif  // _MESH_MIXED_
