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




typedef struct
{
    hypre_IJMatrix    *Face_iedge;
    hypre_IJMatrix    *Element_iedge;
    hypre_IJMatrix    *Edge_iedge;
                                                                                                                            
    hypre_IJMatrix    *Element_Face;
    hypre_IJMatrix    *Element_Edge;
                                                                                                                            
} hypre_PTopology;

