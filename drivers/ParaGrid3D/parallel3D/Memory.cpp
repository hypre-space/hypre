#include "Memory.h"
#include "definitions.h"


//============================================================================

Memory::Memory(){
  V = new vertex[MAXN];
  F = new face[10*MAXN];
  T = new tetrahedron[5*MAXN];
  
  dspace = new double[10*MAXN];
  ispace = new int[10*MAXN];

  NV = NF = NT = dsp = isp = dst = ist = 0;
}

//============================================================================

viod Memory::push_back(vertex v){
  V[NV++] = v;
}

//============================================================================

void Memory::push_back(face f){
  F[NF++] = f;
}

//============================================================================

void Memory::push_back(tetrahedron t){
  T[NT++] = t;
}

//============================================================================

void Memory::memory_check(){
  if ((NV >= MAXN) || (NF >= 10*MAXN) || (NT >= 5*MAXN)){
    printf("MAXN should be increased. Exit.\n");
    exit(1);
  }
}

//============================================================================

double *Memory::alloc(int n){
  DStack[dst++] = n;
  if (dst==STACK) {
    printf("STACK should be increased. Exit.\n");
    exit(1);
  }
  dsp += n;
  return &dspace[dsp - n];
}

//============================================================================

int *Memory::alloc(int n){
  IStack[ist++] = n;
  if (ist==STACK) {
    printf("STACK should be increased. Exit.\n");
    exit(1);
  }
  isp += n;
  return &ispace[isp - n];
}

//============================================================================

void Memory::ddel(){
  dsp -= DStack[--dst];
}

//============================================================================

void Memory::idel(){
  isp -= IStack[--ist];
}

//============================================================================
