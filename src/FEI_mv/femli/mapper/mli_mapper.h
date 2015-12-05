/*BHEADER**********************************************************************
 * (c) 2002   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Header info for the MLI_Mapper data structure
 *
 *****************************************************************************/

#ifndef __MLIMAPPERH__
#define __MLIMAPPERH__

/*--------------------------------------------------------------------------
 * include files 
 *--------------------------------------------------------------------------*/

#include <string.h>
#include "utilities/utilities.h"

/*--------------------------------------------------------------------------
 * MLI_Mapper data structure declaration
 *--------------------------------------------------------------------------*/

class MLI_Mapper
{
   int nEntries;
   int *tokenList;
   int *tokenMap;
public :

   MLI_Mapper();
   ~MLI_Mapper();

   int   setMap(int nItems, int *itemList, int *mapList);
   int   adjustMapOffset(MPI_Comm comm, int *procNRows, int *procOffsets);
   int   getMap(int nItems, int *itemList, int *mapList);
   int   setParams(char *param_string, int argc, char **argv);
 
};

#endif

