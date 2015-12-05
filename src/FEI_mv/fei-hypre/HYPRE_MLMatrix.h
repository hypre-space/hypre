/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/





/****************************************************************************/ 
/* data structures  for local matrix                                        */
/*--------------------------------------------------------------------------*/

#ifndef _HYPREMLMAT_
#define _HYPREMLMAT_

typedef struct
{
    int      Nrows;
    int      *rowptr;
    int      *colnum;
    int      *map;
    double   *values;
    int      sendProcCnt;
    int      *sendProc;
    int      *sendLeng;
    int      **sendList;
    int      recvProcCnt;
    int      *recvProc;
    int      *recvLeng;
}
HYPRE_ML_Matrix;

#endif

