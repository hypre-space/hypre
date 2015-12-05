/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/





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
#include "utilities/_hypre_utilities.h"

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

