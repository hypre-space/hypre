/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.5 $
 ***********************************************************************EHEADER*/




#ifndef ML_VIZ_XYZ_H
#define ML_VIZ_XYZ_H

#include "ml_include.h"
#include "ml_viz_stats.h"

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" {
#endif
#endif

int ML_Aggregate_VisualizeXYZ( ML_Aggregate_Viz_Stats info,
			      char base_filename[],
			      ML_Comm *comm,
			      double * vector);

int ML_PlotXYZ(int Npoints, double* x, double* y, double* z,
	       char base_filename[],
	       USR_COMM comm, double * vector);

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif /* #ifndef ML_VIZ_XYZ_H */

