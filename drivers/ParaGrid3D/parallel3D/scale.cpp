//
// This program scales a mesh
//
#include <stdlib.h>
#include <stdio.h>

main(){
  double scale = 1/500.;
  FILE *plot, *out;
  int i, NF, NTR, NN;

  plot = fopen("myfile.out", "r");
  out  = fopen("scaled.out", "w+");
  if (plot==(FILE *)NULL)
    {printf("file myfile.out is not accessible \n");exit(1);}
  
  char ch[100];
  fscanf(plot,"%s", ch);   // read the first row
  fprintf(out,"%s\n", ch);

  // First in the file we have faces. For now we don't use them so we scip
  // them reading "somewhere".
  fscanf(plot,"%d", &NF);
  fprintf(out,"%d\n",NF);
  int in, n[4]; 
  for(i=0; i<NF; i++){
    fscanf(plot,"%d%d%d%d",&in, &n[0], &n[1], &n[2]);
    fprintf(out,"%4d %4d %4d %4d\n", in, n[0], n[1], n[2]);
  }

  // Second we read the tetrahedrons 
  fscanf(plot,"%d", &NTR);
  fprintf(out,"%d\n",NTR);
  for(i=0; i< NTR; i++){
    // Argument in gives from which "subdomain is this tetrahedron
    fscanf(plot,"%d%d%d%d%d", &in, &n[0], &n[1], &n[2], &n[3]);
    fprintf(out,"%5d %5d %5d %5d %5d\n", in, n[0], n[1], n[2], n[3]);
  } 

  // Finally we read the nodes
  fscanf(plot,"%d", &NN);
  fprintf(out,"%d\n", NN);
  double c[3];
  for(i=0; i<NN; i++){
    fscanf(plot,"%lf%lf%lf", &c[0], &c[1], &c[2]);
    fprintf(out,"%12.6f %12.6f %12.6f\n", scale*c[0], scale*c[1],scale*c[2]); 
  }
  fclose(plot);
  fclose(out);
}
