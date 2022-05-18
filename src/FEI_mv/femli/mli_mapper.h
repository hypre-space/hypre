/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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
#include "_hypre_utilities.h"

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

