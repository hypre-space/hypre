#ifndef _MESH_
#define _MESH_

#include "definitions.h" 
#include "Mesh_Classes.h"
#include "queue.h"
 
class Mesh{

  // friend void ErrorEstimate(Mesh *, int *);

  protected :
    int NN[LEVEL];         // Number of nodes
    int       NTR;         // Number of tetrahedra
    int        NF;         // Number of faces
    int        NE;         // Number of edges

    int  level;            // index of the last level (starting from 0)

    vector<tetrahedron> TR;// TR[i].node[j] j th node of tetrahedron i 
			   // TR[i].face[j] j th face in tetrahedron i
		           // TR[i].refine is the face that has longest edge
                           // The longest edge in the face is determined by
                           // the first two nodes of the face

    vector<face>    F;     // F[i].node[j] j th node of face i
			   // F[i].tetr[j] el. with which a face is connected.
			   // For bdr faces the second number is NEGATIVE and 
			   // shows the type of bdr (Dir, Neumann or Robin).

    vector<edge>    E;     // E[i].node[j] j th node of edge i	    

    vector<vertex>  Z;     // Z[i].coord[j] j th coordinate of node i
			   // Z[i].Atribut -  node atribute

    int  DimPN[LEVEL];     // dimension of the array PN
    int     *V[LEVEL];     // V[i][j] is start position in PN from which 
                           // we put the nodes with which node j is connected.
    int    *PN[LEVEL];     // Their number is V[i][j+1] - V[i][j]
 
    int  DimPNHB[LEVEL];   // similar to the above three but these are for
    int     *VHB[LEVEL];   // hierarchical basis
    int    *PNHB[LEVEL]; 

    int   DimPNI[LEVEL];   // Similar to the above - used for the 
    int      *VI[LEVEL];   // Interpolation matrix
    int     *PNI[LEVEL];

    int  dim_Dir[LEVEL];   // Number of Dirichle nodes		      
    int        *Atribut;   // Node Atributes	      
    int            *Dir;   // Array of node indexes with Dirichle BC

  public:

    Mesh() {}
    Mesh(char *f_name);              // Construct mesh based on netgen output
    void ReadNetgenOutput(char *);   // Read nodes Z and triangles TR
    void InitializeFaces();          // Initialize the faces using TR & Z
    void InitializeEdgeStructure();  // Initialize V and PN.
    void InitializeDirichletBdr();   // Initialize dir.bdr using F & FCon
    void InitializeEdges();          // Initialize edges E using V and PN
    void InitializeTRFace();         // Initialize tetrahedron's faces
    void InitializeRefinementEdges();// Initialize TRRefine

    int tetr_type( int i);

    //=========== refinement procedures (in Mesh_Refine.cpp) =================
    void BisectTet(int t, int *, vector<face> &, vector<queue> &, 
                   vector<int> &, vector<int> &, vector<v_int> &);

    void LocalRefine(int *);
    void UniformRefine();
    int  need_refinement(int, vector<queue> &);
    void InitializeNewEdgeStructure(int *, vector<v_int> &);
    void InitializeNewHBEdgeStructure( vector<v_int> &);
    void InitializeNewIEdgeStructure(int *);
    void MarkFaces();
    //========================================================================

    real length(int i, int j);      // return the length between nodes i&j
     
    // return the index of Aij in the one dimensional array PN for level l
    int ind_ij(int l, int i, int j);

    void PrintSol(char *, double *);   // print the solution in MCGL format
    void PrintSolBdr(char *, double *);// print the solution in MCGL format
    
    // print the solution in MCGL format
    void PrintMCGL(char *,double *,int = -1);
    void PrintMCGL_sequential(char *, double *);		

    void PrintBdr(char *);             // print the bdr in Maple format
    void PrintTetrahedron(int i);      // print information for tetrahedron i
    void PrintMesh(char *);            // print the mesh
    
    void PrintMetis(double *weight);   // print connection gragh (Metis part)

    void PrintMesh();                  // print the mesh (Vassilevski)
    void PrintMeshMaple(char *f_name); // print in Maple format (Vassilevski)
    void PrintFaceMaple(double *, char *, int, double);
    void PrintCutMTVPlot(double *, char *, double, double, double, double);

    void PrintEdgeStructure();         // print vertex coordinates and edges

    int GetNN(int i) { return NN[i];}  // return number of nodes
    int GetNN()  { return NN[level];}
    int GetNTR(){ return NTR;}         // return number of tetrahedrons
    int GetNF() { return NF; }         // return number of faces
    int  *GetV(int l) { return  V[l];}
    int *GetPN(int l) { return PN[l];}
    int GetPNDimension(int l) {return DimPN[l];} 
    int GetLevel() { return level;}
    int GetNodeAtr(int i) { return Z[i].Atribut;}
            
    real *GetNode(int i);             // Get the coord for node i
    int    *GetTetr(int i);           // Get the nodes of tetr i
    void GetMiddle(int i, real *);    // Get the coord of the middle point
                                      // for tetrahedron i
    void GetMiddleFace(int i, real *);// the middle for face i
    void GetMiddleEdge(int i, int j, real *);

    real volume(int i);               // return the volume of tetrahedron i
    real volume();                    // return the volume of the domain
    real area(int tetr, int f);       // return the area of face "f"    

    void CheckConformity(int t);      // check conformity of element t
};

#endif  // _MESH_

