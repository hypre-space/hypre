/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




/*--------------------------------------------------------------------------
 * hypre_SStructSendInfo data structure
 *--------------------------------------------------------------------------*/
#ifndef hypre_SENDINFODATA_HEADER
#define hypre_SENDINFODATA_HEADER


typedef struct 
{
   HYPRE_Int             size;

   hypre_BoxArrayArray  *send_boxes;
   HYPRE_Int           **send_procs;
   HYPRE_Int           **send_remote_boxnums;

} hypre_SStructSendInfoData;

#endif
