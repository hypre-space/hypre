#ifndef MEMORY
#define MEMORY

#include "definitions.h"
#include "Mesh_Classes.h"

#define STACK 20

class Memory{

  private:
    int NV, NF, NT;   // number of vertices, faces and tetrahedrons
    vertex      *V;   // vertices
    face        *F;   // faces
    tetrahedron *T;   // tetraherdons
    
    int dsp, isp;     
    double *dspace;
    int    *ispace;

    int dst, ist;
    int DStack[STACK];
    int IStack[STACK];

  public:
    Memory();

    viod push_back(vertex);
    void push_back(face);
    void push_back(tetrahedron);

    void memory_check();

    double *alloc(int n);
    int    *alloc(int n);
    void ddel();
    void idel();
}

#endif   // MEMORY
