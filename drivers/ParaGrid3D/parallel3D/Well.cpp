#include "Well.h"
#include <stdio.h>

//============================================================================
// func is poiter to function. This function determines whether a point
// belongs to the well. The arguments are atribut and pointer to the point's
// coordinates. The well starts from z = z_start and ends at z = z_end.
//============================================================================
Well::Well(Method *mpointer, double zs, double ze, double alpha_well,
	   double Q_well, int (*func)(int, real *)){
  m       = mpointer;
  if (zs < ze){
    z_start = zs;
    z_end   = ze;
  }
  else {
    z_start = ze;
    z_end   = zs;
  }
  alpha   = alpha_well;
  Q       = Q_well;
  is_well = func;

  pressure = NULL;
}

//============================================================================
// The same as above but this is well for the concentration. Its work is
// determined by the pressure.
//============================================================================
Well::Well(Method *mpointer, double zs, double ze, double alpha_well,
	   double Q_well, int (*func)(int, real *), double *pr){
  //Well(mpointer, zs, ze, alpha_well, Q_well, func);
  m       = mpointer;
  if (zs < ze){
    z_start = zs;
    z_end   = ze;
  }
  else {
    z_start = ze;
    z_end   = zs;
  }
  alpha   = alpha_well;
  Q       = Q_well;
  is_well = func;

  pressure = pr;
}

//============================================================================
// ind1 is the index of the current node in the well, z1 is its z coordinate.
// ind2 and z2 are the output - the corresponding values for the next node. 
// The function returns 0 if there are no more nodes in the well (otherwise 1)
//============================================================================
int Well::get_next(int level, int ind1, int &ind2, double z1, double &z2){
  int i, *V=m->GetV(level), *PN=m->GetPN(level);
  real *coord;

  for(i=V[ind1]; i<V[ind1+1]; i++){
    ind2 = PN[i];
    coord = m->GetNode(ind2);
    if ((*is_well)(m->GetNodeAtr(ind2), coord))
      if (coord[2] > z1){
	z2 = coord[2];
        return 1;
      }	
  }
  return 0;                                  // if not found return 0
}

//============================================================================
// Add contribution to the stiffness matrix A and the RHS b.
//============================================================================
void Well::add_contribution(int level, real *A, double *b){
  int i, ind1, ind2, found = 0;
  int NN=m->GetNN(level);
  real *coord;
  double z1 = 1000., z2, d;

  // find the first node
  for(i=0; i<NN; i++){
    coord = m->GetNode(i);
    if ((*is_well)(m->GetNodeAtr(i), coord))
      if (coord[2] < z1){
	z1   = coord[2];
	ind1 = i;
	found = 1;
      }
  }

  if (found){
    if (pressure == NULL)                     // this is well for the pressure
      while (get_next( level, ind1, ind2, z1, z2)){
	d = Q*(z2-z1)/2;                      // Add contribution to RHS b

	b[ind1] -= d;
	b[ind2] -= d;
	
	d = alpha*(z2-z1)/6;                  // Add contribution to A
	A[m->ind_ij(level, ind1, ind1)] += 2*d;
	A[m->ind_ij(level, ind2, ind2)] += 2*d;
	A[m->ind_ij(level, ind1, ind2)] +=   d;
	A[m->ind_ij(level, ind2, ind1)] +=   d;
      
	ind1 = ind2; z1 = z2;
      }
    else                                      // concentration well
      while (get_next( level, ind1, ind2, z1, z2)){
	d = (alpha*(pressure[ind1]+pressure[ind2])*0.5 + Q)*(z2-z1)/6;
	
	A[m->ind_ij(level, ind1, ind1)] += 2*d;
	A[m->ind_ij(level, ind2, ind2)] += 2*d;
	A[m->ind_ij(level, ind1, ind2)] +=   d;
	A[m->ind_ij(level, ind2, ind1)] +=   d;
	
	ind1 = ind2; z1 = z2;
      }
  }
}

//============================================================================
