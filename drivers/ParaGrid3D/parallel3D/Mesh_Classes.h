#ifndef _MESH_CLASSES_
#define _MESH_CLASSES_

#include <vector.h>
#include "definitions.h"


class vertex{
  public:
    real coord[3];
    int Atribut;

    vertex();

    real *GetCoord();
    int GetAtribut();
};



class tetrahedron{
  public:
    int node[4];
    int face[4];
    int refine;
    int type;
    int atribut;

    tetrahedron();

    int *GetNode();
    int *GetFace();
    void GetMiddle(double coord[3], vector<vertex> &V);
    real determinant(vector<vertex> &V);
};


class face{
  public:
    int node[3];
    int tetr[2]; 

    face();

    int *GetNode();
    int *GetTetr();
};


class edge{
  public:
    int node[2];

    edge();
    edge(int node1, int node2);

    int *GetNode();
};


#endif   // _MESH_CLASSES_

