/*#*****************************************************
#
#	File:  Hypre_MPI_Com_DataMembers.h
#
#********************************************************/

#ifndef Hypre_MPI_Com_DataMembers__
#define Hypre_MPI_Com_DataMembers__
/* JFP ... */
#include "HYPRE_mv.h"
struct Hypre_MPI_Com_private_type /* gkk: added "_type" */
{
   MPI_Comm *hcom;
}
;
#endif

