#include <stdio.h>
#include <iostream.h>

int N;

int num(int i, int j, int k){
  return i+(N+1)*j+(N+1)*(N+1)*k+1;
}


//============================================================================
void PrintCube(FILE *plot, int i, int j, int k){
  int L[] = {num(i,j,k),num(i+1,j,k),num(i+1,j+1,k),num(i,j+1,k)}; 
  int U[] = {num(i,j,k+1),num(i+1,j,k+1),num(i+1,j+1,k+1),num(i,j+1,k+1)}; 

  fprintf(plot,"%1d  %4d  %4d  %4d  %4d\n",1,L[0],U[2],L[1],L[2]);
  fprintf(plot,"%1d  %4d  %4d  %4d  %4d\n",1,L[0],U[2],L[1],U[1]);
  fprintf(plot,"%1d  %4d  %4d  %4d  %4d\n",1,L[0],U[2],L[2],L[3]);
  fprintf(plot,"%1d  %4d  %4d  %4d  %4d\n",1,L[0],U[2],U[0],U[3]);
  fprintf(plot,"%1d  %4d  %4d  %4d  %4d\n",1,L[0],U[2],L[3],U[3]);
  fprintf(plot,"%1d  %4d  %4d  %4d  %4d\n",1,L[0],U[2],U[0],U[1]);
}



void split(){
  FILE *plot;
  int i, j, k;

  plot=fopen("UCube.out","w+");
  if (plot==(FILE *)NULL)
    {printf("file UCube is not accessible \n");exit(1);}

  printf("The unit cube will be split in N x N x N cubes, N = ");
  cin >> N;

  int NTR = N*N*N*6, NN = (N+1)*(N+1)*(N+1);
  int TR[NTR][3];

  fprintf(plot,"%d\n", NTR); 

  for(i=0; i<N; i++)
    for(j=0; j<N; j++)
      for(k=0; k<N; k++)
	PrintCube(plot, i, j, k);

  double h = 1./N;
  fprintf(plot,"%d\n", NN);
  for(i=0; i<=N; i++)
    for(j=0; j<=N; j++)
      for(k=0; k<=N; k++)
	fprintf(plot,"%10.6f %10.6f %10.6f\n",h*i,h*j,h*k);
  fclose(plot);
}


main(){
  split();
}
