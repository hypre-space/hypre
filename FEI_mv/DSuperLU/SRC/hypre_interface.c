#include "fortran.h"
double hypre_NAME_FORT_CALLING_C(hy_dlamch,HY_DLAMCH)(char* in)
{
   return hypre_F90_NAME_LAPACK(dlamch,DLAMCH)(in);
} 

