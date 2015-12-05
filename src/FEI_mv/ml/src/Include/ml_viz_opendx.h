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




/********************************************************************* */
/* See the file COPYRIGHT for a complete copyright notice, contact      */
/* person and disclaimer.                                               */   
/* ******************************************************************** */

/********************************************************************* */
/*          visualization routines                                     */
/********************************************************************* */

#ifndef __MLVIZOPENDX__
#define __MLVIZOPENDX__

#include "ml_viz_stats.h"

#ifndef ML_CPP
#ifdef __cplusplus
extern "C" {
#endif
#endif

 extern int ML_Aggregate_VisualizeWithOpenDX( ML_Aggregate_Viz_Stats info,
					      char base_filename[],
					      ML_Comm * comm);
 extern int ML_Aggregate_VisualizeXYZ( ML_Aggregate_Viz_Stats info,
				      char base_filename[],
				      ML_Comm * comm, double * vector);
  

#ifndef ML_CPP
#ifdef __cplusplus
}
#endif
#endif

#endif /* #ifndef __MLVIZOPENDX__ */
