/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* Mixed precision function protos */

#ifdef HYPRE_MIXED_PRECISION
/* utilities_mp.c */
HYPRE_Int
hypre_RealArrayCopyHost_mp(HYPRE_Precision precision_x, void *x, 
		       HYPRE_Precision precision_y, void *y, HYPRE_Int n);
HYPRE_Int
hypre_RealArrayCopy_mp(HYPRE_Precision precision_x, void *x, HYPRE_MemoryLocation location_x, 
		       HYPRE_Precision precision_y, void *y, HYPRE_MemoryLocation location_y, HYPRE_Int n);
void *
hypre_RealArrayClone_mp(HYPRE_Precision precision_x, void *x, HYPRE_MemoryLocation location_x, 
                       HYPRE_Precision new_precision, HYPRE_MemoryLocation new_location, HYPRE_Int n);	       
HYPRE_Int
hypre_RealArrayAxpynHost_mp(HYPRE_Precision precision_x, hypre_long_double alpha, void *x, 
		       HYPRE_Precision precision_y, void *y, HYPRE_Int n);
HYPRE_Int
hypre_RealArrayAxpyn_mp(HYPRE_Precision precision_x, void *x, HYPRE_Precision precision_y, void *y,
		        HYPRE_MemoryLocation location, HYPRE_Int n, hypre_long_double alpha);
/* utilities_mp_device.c */
HYPRE_Int
hypre_RealArrayCopyDevice_mp(HYPRE_Precision precision_x, void *x, 
		       HYPRE_Precision precision_y, void *y, HYPRE_Int n);
HYPRE_Int
hypre_RealArrayAxpynDevice_mp(HYPRE_Precision precision_x, hypre_long_double alpha, void *x, 
		       HYPRE_Precision precision_y, void *y, HYPRE_Int n);		       
#endif
