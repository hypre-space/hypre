#include "fortran.h"
double hypre_F90_NAME(hy_dlamch,HY_DLAMCH)(char* in)
{
   return hypre_F90_NAME_LAPACK(dlamch,DLAMCH)(in);
} 

