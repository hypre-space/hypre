#include <stdio.h>

main(){
  int *tt = new int[0];
  delete [] tt;
  printf("Live if beautiful\n");
}
