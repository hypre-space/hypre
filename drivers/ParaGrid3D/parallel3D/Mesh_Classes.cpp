#include "Mesh_Classes.h"

//================ tetrahedron class =========================================

tetrahedron::tetrahedron(){}

int *tetrahedron::GetNode(){
  return node;
}

int *tetrahedron::GetFace(){
  return face;
}

void tetrahedron::GetMiddle(double coord[3], vector<vertex> &V){
  double *c[4];
  int j, k;
  
  for(j=0; j<4; j++)
    c[j] = &(V[node[j]].coord[0]);
  
  for(k=0; k<3; k++){          // for the 3 coordinates
    coord[k] = 0.;
    for(j=0; j<4; j++)
      coord[k] += c[j][k];
    coord[k] /= 4.;
  }
}

// the tetrahedron's volume is given by fabs(determinant)/6
real tetrahedron::determinant(vector<vertex> &V){
  real *p[4], v[3][3];
  int i;

  for(i=0; i<4; i++) 
    p[i] = V[node[i]].GetCoord();
 
  for(i=0; i<3; i++){
    v[i][0] = p[i][0] - p[i+1][0];
    v[i][1] = p[i][1] - p[i+1][1];
    v[i][2] = p[i][2] - p[i+1][2];
  }

  return (v[0][0]*(v[1][1]*v[2][2]-v[1][2]*v[2][1]) +
	  v[0][1]*(v[1][2]*v[2][0]-v[1][0]*v[2][2]) +
	  v[0][2]*(v[1][0]*v[2][1]-v[1][1]*v[2][0]) );
}


//============================================================================
//================= face classe ==============================================

face::face(){}

int *face::GetNode(){
  return node;
}

int *face::GetTetr(){
  return tetr;
}

//============================================================================
//================= edge clsass ==============================================

edge::edge(){}

edge::edge(int node1, int node2){
  node[0] = node1;
  node[1] = node2;
}

int *edge::GetNode(){
  return node;
}

//============================================================================
//================= vertex class =============================================

vertex::vertex(){}

real *vertex::GetCoord(){
  return coord;
}

int vertex::GetAtribut(){
  return Atribut;
}

//============================================================================
