#include <stdio.h>
#include <iostream.h>
#include "Mesh.h"

//============================================================================
// Return the number of the edge that has verteces "n1" and "n2". V and PN
// are the standard mesh structures V and PN. edge is array wiwth dimension
// the dimension of PN and gives the edge indexes - in order to find the
// index the node with smaller index is first.
//============================================================================
int get_edge(int n1, int n2, int *edge, int *V, int *PN){
  int i, start, end;

  if (n1<n2) {start = n1; end = n2;}
  else {start = n2; end = n1;}

  for(i=V[start]+1; i<V[start+1]; i++)
    if (PN[i] == end)
      return edge[i];
  return 1;
}


//============================================================================
// Print the mesh in the following files:
// element_node, element_face, face_node, face_edge, node_coordinates,
// edge_node
//============================================================================
void Mesh::PrintMesh(){
  int i, j;
  FILE *plot;

  // print file "element_node"
  plot = fopen("~/output/element_node", "w+");
  if (plot==(FILE *)NULL){
    printf("file ~/output/element_node is not accessible \n");
    exit(1);
  }
  
  for(i=0; i<NTR; i++)
    for(j=0; j<4; j++)
      fprintf(plot,"%5d  %5d\n", i, TR[i].node[j]);
  printf("Output printed in file ~/output/element_node\n");
  fclose(plot);

  // print file "element_face"
  plot = fopen("~/output/element_face", "w+");
  for(i=0; i<NTR; i++)
    for(j=0; j<4; j++)
      fprintf(plot,"%5d  %5d\n", i, TR[i].face[j]);
  printf("Output printed in file ~/output/element_face\n");
  fclose(plot);

  // print file "face_node"
  plot = fopen("~/output/face_node", "w+");
  for(i=0; i<NF; i++)
    for(j=0; j<3; j++)
      fprintf(plot,"%5d  %5d\n", i, F[i].node[j]);
  printf("Output printed in file ~/output/face_node\n");
  fclose(plot);

  // print file "node_coordinates"
  plot = fopen("~/output/node_coordinates","w+");
  for(i=0; i<NN[level]; i++)
    fprintf(plot,"%10.7f  %10.7f  %10.7f\n", Z[i].coord[0],
	    Z[i].coord[1], Z[i].coord[2]);
  printf("Output printed in fie ~/output/node_coordinates\n");
  fclose(plot);

  int *edge = new int[DimPN[level]];
  int n;
  n = 0;
  for(i=0; i<NN[level]; i++)
    for(j=V[level][i]+1; j<V[level][i+1]; j++)
      if (i<PN[level][j])
	edge[j] = n++;
  printf("Number of edges : %d\n", n);

  // print file "face_edge"
  plot = fopen("~/output/face_edge", "w+");
  for(i=0; i<NF; i++)
    for(j=0; j<3; j++)
      fprintf(plot,"%5d  %5d\n", i, get_edge(F[i].node[j],F[i].node[(j+1)%3], 
					     edge, V[level], PN[level]));
  printf("Output printed in file ~output/face_edge\n");
  fclose(plot);

  // print file "edge_node"
  plot = fopen("~/output/edge_node", "w+");
  for(i=0; i<NN[level]; i++)
    for(j=V[level][i]+1; j<V[level][i+1]; j++)
      if (i<PN[level][j]){
	fprintf(plot,"%5d  %5d\n", edge[j], i);
	fprintf(plot,"%5d  %5d\n", edge[j], PN[level][j]);
      }
  printf("Output printed in file ~/output/edge_node\n");
  fclose(plot);
  delete [] edge;
}


//============================================================================
// This procedure prints the nodes in file "Mesh_Nodes" and the edges in file
// "Mesh_Edges". The format for the edge file is:
//  first line : #nodes # edges
//  other lines: neighbor vertices for vertex 1
//               neighbor vertices for vertex 2
//               ...
//               neighbor vertices for vertex n
//============================================================================
void Mesh::PrintEdgeStructure(){
  int i, j;
  FILE *plot;

  // print file "Mesh_Nodes"
  plot = fopen("~/output/Mesh_Nodes", "w+");
  if (plot==(FILE *)NULL){
    printf("file ~/output/Mesh_Nodes is not accessible \n");
    exit(1);
  }
  fprintf(plot,"%d\n", NN[level]);
  for(i=0; i<NN[level]; i++)
    fprintf(plot,"%12.4f %12.4f %12.4f\n", 
	    Z[i].coord[0], Z[i].coord[1], Z[i].coord[2]);
  printf("Output printed in file ~/output/Mesh_Nodes\n");
  fclose(plot);

  // print file "Mesh_Edges"
  plot = fopen("~/output/Mesh_Edges", "w+");
  if (plot==(FILE *)NULL){
    printf("file ~/output/Mesh_Edges is not accessible \n");
    exit(1);
  }
  fprintf(plot,"%d  %d\n", NN[level], (DimPN[level]-NN[level])/2);
  for(i=0; i<NN[level]; i++){
    for(j=V[level][i]+1; j<V[level][i+1]; j++)
      fprintf(plot,"%4d", PN[level][j] + 1);
    fprintf(plot,"\n");
  }
  printf("Output printed in file ~/output/Mesh_Edges\n");
  fclose(plot);
}
