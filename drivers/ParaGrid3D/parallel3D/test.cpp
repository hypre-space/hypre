// To compile with cxx we include only vector. For all other compilers we 
// include vector.h
#include <vector>
#include <stdio.h>
#include "queue.cpp"

using namespace std;
struct mine{
  int a[2];
};

void add(vector<mine> &a){
  mine b;
  double dd[][2] = {{1., 2.},{1., 2.}};
  b.a[0] = b.a[1] = 5;
  a.push_back( b);

  printf("new size = %d\n", a.size());
  printf("lase el %d %d\n", a[a.size()-1].a[0], a[a.size()-1].a[1]);
} 

int tr (){
  return 7;
}

main(){
  vector<mine> a(5);
  int dd[] = {2+tr(),3};
  printf("\n");
  vector<int> *s = new vector<int>(4);
  for(int i=0; i<4; i++) printf("%5d  ", (*s)[i]);
  printf("\n");

  add(a);

  printf("new size = %d\n", a.size());
  printf("lase el %d %d\n", a[a.size()-1].a[0], a[a.size()-1].a[1]);

  vector<queue> tt(5);
  printf("size = %d\n", tt[1].first());
  tt[1].push(3);
  for(int i=0; i<2; i++){
    queue null_queue;
    tt.push_back(null_queue);
    tt.push_back(null_queue);
    tt[i+2].push(5);
  } 
  for(int i=0; i<tt.size(); i++)
    printf("empty[%d] = %d\n",i, tt[i].empty());
}


