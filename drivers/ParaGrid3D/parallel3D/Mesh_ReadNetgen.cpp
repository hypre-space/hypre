#include <stdio.h>
#include <stdlib.h>          // qsort function used
#include <math.h>
#include "definitions.h"
#include "Mesh.h"

//============================================================================
// Given file name (with netgen output) this routine initializes the nodes Z,
// number of nodes NN[0], tetrahedrons TR and their number NTR. If somebody 
// provide these in some other way the rest will still work.
//============================================================================

void Mesh::ReadNetgenOutput(char *f_name){
  FILE *plot;
  int i, NF;

  plot=fopen( f_name, "r");
  if (plot==(FILE *)NULL)
    {printf("file %s is not accessible \n",f_name);exit(1);}

  char ch[100];
  fscanf(plot,"%s", ch);   // read the first row

  // First in the file we have faces. For now we don't use them so we skip
  // them reading "somewhere".
  fscanf(plot,"%d", &NF); 
  int in, n[3]; 
  for(i=0; i<NF; i++)
    fscanf(plot,"%d%d%d%d",&in, &n[0], &n[1], &n[2]);

  // Second we read the tetrahedrons 
  fscanf(plot,"%d", &NTR);
  TR = vector<tetrahedron>(NTR);
  for(i=0; i< NTR; i++){
    // Argument in gives from which "subdomain is this tetrahedron
    fscanf(plot,"%d%d%d%d%d", &TR[i].atribut, &TR[i].node[0], &TR[i].node[1], 
	   &TR[i].node[2], &TR[i].node[3]);
    TR[i].node[0]--; TR[i].node[1]--; TR[i].node[2]--; TR[i].node[3]--;
  } 

  // Finally we read the nodes
  fscanf(plot,"%d", &(NN[0]));
  Z = vector<vertex>(NN[0]);
  for(i=0; i<NN[0]; i++)
    fscanf(plot,"%lf%lf%lf", &Z[i].coord[0], &Z[i].coord[1], &Z[i].coord[2]);

  fclose(plot);
}


//============================================================================
// Given NTR tetrahedrons in TR, NN nodes in Z this routine initializes the
// faces as 3 node indexes. ATTENITON : The order of the nodes in the faces
// is important. The distance between the first two should be biggest. This
// is used in the refinement procedures. It gives us directly the so called
// marked edge (the one with end nodes between the first two)!!!
//============================================================================
int Type_Boundary(double, double, double);
/*
void  qsort(void *,size_t,size_t,int (*compar)(const void *,const void *));
*/

// Order in increasing order
static  int int_compare(const void *ii, const void *jj){
  int *i = (int *)ii, *j = (int *)jj;
  if (*i > *j) return (1);
  if (*i < *j) return (-1);
  return (0);
}

struct four_int_str{
  int n[4];
};

static int face_compare(const void *ii, const void *jj){
  four_int_str *i = (four_int_str *)ii, *j = (four_int_str *)jj; 
  if (i->n[0] > j->n[0]) return 1;
  if (i->n[0] < j->n[0]) return -1;

  if (i->n[1] > j->n[1]) return 1;
  if (i->n[1] < j->n[1]) return -1;

  if (i->n[2] > j->n[2]) return 1;
  if (i->n[2] < j->n[2]) return -1;

  return 0;
}

int order(int a1, int a2, int b1, int b2){
  int c;
  if (a1>a2) { c = a2; a2 = a1; a1 = c;}
  if (b1>b2) { c = b2; b2 = b1; b1 = c;}
  if (a1>b1) return 1;
  else if ((a1==b1)&&(a2>b2)) return 1;
  else return 0;
}

// If you want to sort an integer array "a" of 10 elements call qsort with :
// qsort(a, 10, sizeof(int), intcompare);
//============================================================================
void Mesh::InitializeFaces(){
  int i, j, n_faces = 0, node[4], ii;
  double x, y, z;

  four_int_str *faces =new four_int_str[4*NTR];// fourth int gives the tetr.
  for(i=0; i<NTR; i++){                        // for all tetrahedrons  
    for(j=0; j<4; j++) node[j] = TR[i].node[j];// make the corresponding faces
    qsort(node, 4, sizeof(int), int_compare);  // sort the node indexes

    faces[n_faces].n[0]=node[0]; faces[n_faces].n[1]=node[1];
    faces[n_faces].n[2]=node[2]; faces[n_faces].n[3]=i; 
    n_faces++;

    faces[n_faces].n[0]=node[0]; faces[n_faces].n[1]=node[1]; 
    faces[n_faces].n[2]=node[3]; faces[n_faces].n[3]=i; 
    n_faces++;
    
    faces[n_faces].n[0]=node[0]; faces[n_faces].n[1]=node[2]; 
    faces[n_faces].n[2]=node[3]; faces[n_faces].n[3]=i; 
    n_faces++;

    faces[n_faces].n[0]=node[1]; faces[n_faces].n[1]=node[2]; 
    faces[n_faces].n[2]=node[3]; faces[n_faces].n[3]=i; 
    n_faces++;
  }
  
  // sort the faces - in order to find the boundary faces and the interior
  // faces (the repeated ones are inside)
  qsort(faces, n_faces, sizeof(four_int_str), face_compare);

  int NFaces = n_faces;
  for(i=0; i<(n_faces-1); i++)
    if (face_compare(&faces[i], &faces[i+1])==0) NFaces--;

  // Allocate memory for F
  F    = vector<face>(NFaces);

  NF = 0;
  double d[3];
  int ind;
  // Go through the faces and find the repeated ones
  for(i=0; i<(n_faces-1); i++){

    // ATTENTION!!! We want to order the nodes in the faces such that the
    // distance between the first two nodes is bigest. This will give the 
    // desired MARKED edge for this face directly at construction time
    for(j=0; j<3; j++) d[j] = length(faces[i].n[j],faces[i].n[(j+1)%3]);
    ind = 0;
    for(j=1; j<3; j++)
      if (d[ind] < d[j]) ind = j;
      else
	if (d[ind]==d[j])
	//if (d[ind]<d[j]+0.00000000000001)
	  if (order(faces[i].n[ind], faces[i].n[(ind+1)%3],
		    faces[i].n[j], faces[i].n[(j+1)%3]))
	    ind = j;
    
    F[NF].node[0] = faces[i].n[ind];
    F[NF].node[1] = faces[i].n[(ind+1)%3];
    F[NF].node[2] = faces[i].n[(ind+2)%3];

    if (F[NF].node[1]<F[NF].node[0]){  // first node has to have smaller ind
      ii = F[NF].node[0];
      F[NF].node[0] = F[NF].node[1];
      F[NF].node[1] = ii;
    }

    if (face_compare(&faces[i], &faces[i+1])==0){ // the same face is shared
      F[NF].tetr[0]= faces[i].n[3]; // triangle with which this fase is conn 
      F[NF].tetr[1]= faces[i+1].n[3];

      i++;                       // we jump on the next
    }
    else{                        // this is boundary face
      F[NF].tetr[0]= faces[i].n[3];
      x = y = z = 0.;
      for(j=0;j<3;j++){
	x += Z[faces[i].n[j]].coord[0];
	y += Z[faces[i].n[j]].coord[1];
	z += Z[faces[i].n[j]].coord[2];
      }
      // take the type of the boundary (evaluate at the middle of the face)
      F[NF].tetr[1]= Type_Boundary(x/3, y/3, z/3);
    }
    NF++;
  }

  i = n_faces-1;

  if (NF!=NFaces){               // we check the last one (if there is one it
                                 // will be boundary face
    // Do the same for the order
    for(j=0; j<3; j++) d[j] = length(faces[i].n[j],faces[i].n[(j+1)%3]);
    ind = 0;
    for(j=1; j<3; j++)
      if (d[ind] < d[j]) ind = j;
      else
	if (d[ind]==d[j])
	//if (d[ind]<d[j]+0.00000000000001)
	  if (order(faces[i].n[ind], faces[i].n[(ind+1)%3],
		    faces[i].n[j], faces[i].n[(j+1)%3]))
	    ind = j;

    F[NF].node[0] = faces[i].n[ind];
    F[NF].node[1] = faces[i].n[(ind+1)%3];
    F[NF].node[2] = faces[i].n[(ind+2)%3];

    if (F[NF].node[1]<F[NF].node[0]){  // first node has to have smaller ind
      ii = F[NF].node[0];
      F[NF].node[0] = F[NF].node[1];
      F[NF].node[1] = ii;
    }

    F[NF].tetr[0]= faces[i].n[3];
    x = y = z = 0.;
    for(j=0;j<3;j++){
      x += Z[faces[i].n[j]].coord[0];
      y += Z[faces[i].n[j]].coord[1];
      z += Z[faces[i].n[j]].coord[2];
    }
    F[NF].tetr[1]= Type_Boundary(x/3, y/3, z/3);
    NF++;
  }

  delete [] faces;
}


//============================================================================
// Using the faces F and their boundary given in tetr this function initiali-
// zes the Dirichlet bdr and node atributes Atribut.
// note: if (F[i].tetr[1] < 0) face i is on the bdr.
//============================================================================

void Mesh::InitializeDirichletBdr(){
  int i, j;

  for(i=0;i<NN[0];i++)   // Init. first everything by default to be inside
    Z[i].Atribut = INNER;
  
  for(i=0; i<NF; i++){   // go through the faces to find bdr faces
    if (F[i].tetr[1]<0){ // this face is on the bdr, so all its 3 nodes have
      for(j=0;j<3; j++)  // the same atribut
	Z[F[i].node[j]].Atribut = F[i].tetr[1];
    }
  }

  for(i=0; i<NF; i++){   // go through the faces to find Dirichlet bdr faces
    if (F[i].tetr[1]==DIRICHLET){
      for(j=0;j<3; j++)
	Z[F[i].node[j]].Atribut = F[i].tetr[1];
    }
  } /* Atribut initialization done */

  Atribut = new int[NN[0]];

  // Initialize Dirichlet bdr nodes
  dim_Dir[0] = 0;
  for(i=0; i<NN[0]; i++) {
    Atribut[i] = Z[i].Atribut;
    if (Z[i].Atribut==DIRICHLET) dim_Dir[0]++;
  }

  Dir = new int[dim_Dir[0]];
  int next = 0;
  for(i=0; i<NN[0]; i++)
    if (Z[i].Atribut==DIRICHLET) Dir[next++] = i;
}


//============================================================================
// Here we initialize the edges. Actually they are not generated. Here we
// generate the structures V & PN[0] which are used for the action routines,
// refinement procedures and so on.
//
// input  : NN, DimPN[0], NTR; V, TR
// output : V, PN[0]
//
// The following function is also used here
int notthere (int a, int b, int c,int *P){   
  for(int i= 0; i < c; i++){ 
    if ( P[b+i] == a)
      return 0;
  }
  return 1;
}
//============================================================================

void Mesh::InitializeEdgeStructure(){
  int i, j, q=0;

  // This min angle conditiom (with at most how many nodes one may be connec-
  // ted) may be increased if the program exit because of it or the mesh
  // should be changed. 
  int min_angle_condition = 100;

  V[0]     = new int[NN[0]+1];
  int *Num = new int[NN[0]+1]; 
  DimPN[0] = min_angle_condition*NN[0];

  int *P = new int[DimPN[0]];  

  for (i=0; i <= NN[0]; i++){
    Num[i]  = 0 ;
    V[0][i] = min_angle_condition*i;
  }

  for (i=0; i < DimPN[0]; i++)
    P[i] = - 1;

  for (i=0; i < NTR; i++){  // for all the tetrahedrons do:
    for (j=0; j < 4; j++){  // for all the tetrah's 4 points do
      if ( notthere( TR[i].node[j], V[0][TR[i].node[j]], 
		     Num[TR[i].node[j]], P)){ 
	P[ V[0][TR[i].node[j]] + Num[TR[i].node[j]] ] = TR[i].node[j];
	Num[TR[i].node[j]]++;
      }

      if ( notthere( TR[i].node[(j+1)%4], V[0][TR[i].node[j]], 
		     Num[TR[i].node[j]], P)){
	P[ V[0][TR[i].node[j]] + Num[TR[i].node[j]] ] = 
	  TR[i].node[(j+1)%4];
	Num[TR[i].node[j]]++;
      }  

      if ( notthere( TR[i].node[(j+2)%4], V[0][TR[i].node[j]], 
		     Num[TR[i].node[j]],P )){ 
	P[ V[0][TR[i].node[j]] + Num[TR[i].node[j]] ] = 
	  TR[i].node[(j+2)%4];
	Num[TR[i].node[j]]++;
      } 
      
      if ( notthere( TR[i].node[(j+3)%4], V[0][TR[i].node[j]], 
		     Num[TR[i].node[j]],P )){ 
	P[ V[0][TR[i].node[j]] + Num[TR[i].node[j]] ] = 
	  TR[i].node[(j+3)%4];
	Num[TR[i].node[j]]++;
      } 
      
    }
  }
  
  int ll = 0;
  for (i=0; i<NN[0]; i++){
    if ( Num[i] > min_angle_condition) {
      printf("\nInitial mesh doesn't satisfy min angle condition. Exit\n");
      exit(1);
    }
    ll += Num[i];
  }
  
  DimPN[0] = ll;
  PN[0] = new int [DimPN[0]];   // allocate the actual memory for PN
  
  for(i=0; i< NN[0] ;i++){   // "shrink" P into PN
    for(j=0; j< Num[i]; j++){
      PN[0][q]=P[V[0][i]+j];
      q++;
    }
  }

  delete [] P;

  for(i=1 ;i < NN[0]; i++)
    V[0][i] = V[0][i-1] + Num[i-1];
  V[0][NN[0]] = ll;

  delete [] Num;
}


//============================================================================
// Initialize the edges E (using V & PN[0])
// input  : DimPN[0], NN; V, PN[0]
// output : NE; E
//============================================================================
void Mesh::InitializeEdges(){
  int i, j, begin, end, next = 0;

  NE= (DimPN[0]-NN[0])/2;
  E = vector<edge>(NE);

  for(i=0; i<NN[0]; i++){
    begin = V[0][i] + 1;      // first is for the node itself
    end   = V[0][i+1];

    for(j=begin; j<end; j++)
      if (i<PN[0][j]){ 
	E[next].node[0] = i;    // first end is with smaller index
	E[next].node[1] = PN[0][j];
	next++;
      } // otherwise we have included it already
  }
}

//============================================================================
// Initialize tetrahedron's faces TRFaces. Face i should be opposite node i.
// input  : NF, NTR; TR, F, FCon
// output : TRFace
//============================================================================
void Mesh::InitializeTRFace(){
  int i, j;
  int *nodes;

  for(i=0; i<NF; i++){  // go through all the faces and put them at the
                        // corresponding places

    nodes = TR[F[i].tetr[0]].GetNode();//put the face into first tetrahedron
    for(j=0; j<4; j++){ // find the node opposite the face
      if ((nodes[j]!=F[i].node[0])&&(nodes[j]!=F[i].node[1])&&
	  (nodes[j]!=F[i].node[2])){
	TR[F[i].tetr[0]].face[j] = i;
	break;
      }
    }
    if (F[i].tetr[1]>=0){      // put it in the second
      nodes = TR[ F[i].tetr[1]].GetNode();; 
      for(j=0; j<4; j++){ // find the node opposite the face
	if ((nodes[j]!=F[i].node[0])&&(nodes[j]!=F[i].node[1])&&
	    (nodes[j]!=F[i].node[2])){
	  TR[F[i].tetr[1]].face[j] = i;
	  break;
	}
      }
    }
  } /* end for all faces */
}


//============================================================================
// Initialize the "Refinement edges for the tetrahedrons. The refinement edge
// is given by index of one of the tetrahedron's faces (the local index - so
// that we have the edge opposite the face) and the face keeps the refinement
// edge = marked edge as the egde between its first two nodes.
// input  : NTR; F, TRFace 
// output : TRRefine
//============================================================================
void Mesh::InitializeRefinementEdges(){
  int i, j, ind, v;
  double d[4];
  tetrahedron *T;

  for(i=0; i<NTR; i++){  // for every tetrah determine the refinement "face"
    T = &(TR[i]);
    T->type = TYPE_A;
    for(j=0; j<4; j++){  // for the 4 faces compute the max edges
      // the maximum edge length for an edge is given by the distance between
      // face's first two points.
      d[j] = length( F[ T->face[j] ].node[0], F[ T->face[j] ].node[1]);
      //      printf("d[%d] = %f\n", j, d[j]);
    }
    ind = 0;
    for(j=1; j<4; j++)
      if (d[ind] < d[j]) ind = j;
      else
	if (d[ind]==d[j]) {
	//if (d[ind]<d[j]+0.00000000000001) {
	  // if (T->face[ind] > T->face[j])
	  if (order(F[T->face[ind]].node[0], F[T->face[ind]].node[1],
		    F[T->face[j]].node[0], F[T->face[j]].node[1]))
	    ind = j;
	}

    T->refine = ind;

    // Check the type
    //v = T->node[j];
    //v = T->node[ind];
    for( j=0;j<4; j++){
      v = T->node[j];
      if ( (v!=F[T->face[0]].node[0]) && (v!=F[T->face[0]].node[1]) &&
	   (v!=F[T->face[1]].node[0]) && (v!=F[T->face[1]].node[1]) &&
	   (v!=F[T->face[2]].node[0]) && (v!=F[T->face[2]].node[1]) &&
	   (v!=F[T->face[3]].node[0]) && (v!=F[T->face[3]].node[1]) ){
	T->type = TYPE_PU;
	break;
      }
    }
  }
} 




