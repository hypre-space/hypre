/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_OMP_DEVICE_H
#define HYPRE_OMP_DEVICE_H

#if defined(HYPRE_USING_DEVICE_OPENMP)

#include "omp.h"

/* OpenMP 4.5 device memory management */
extern HYPRE_Int hypre__global_offload;
extern HYPRE_Int hypre__offload_device_num;
extern HYPRE_Int hypre__offload_host_num;

/* stats */
extern size_t hypre__target_allc_count;
extern size_t hypre__target_free_count;
extern size_t hypre__target_allc_bytes;
extern size_t hypre__target_free_bytes;
extern size_t hypre__target_htod_count;
extern size_t hypre__target_dtoh_count;
extern size_t hypre__target_htod_bytes;
extern size_t hypre__target_dtoh_bytes;

/* CHECK MODE: check if offloading has effect (turned on when configured with --enable-debug)
 * if we ``enter'' an address, it should not exist in device [o.w NO EFFECT]
 * if we ``exit'' or ''update'' an address, it should exist in device [o.w ERROR]
 * hypre__offload_flag: 0 == OK; 1 == WRONG
 */
#ifdef HYPRE_DEVICE_OPENMP_CHECK
#define HYPRE_OFFLOAD_FLAG(devnum, hptr, type) HYPRE_Int hypre__offload_flag = (type[1] == 'n') == omp_target_is_present(hptr, devnum);
#else
#define HYPRE_OFFLOAD_FLAG(...) HYPRE_Int hypre__offload_flag = 0; /* non-debug mode, always OK */
#endif

/* OMP 4.5 offloading macro */
#define hypre_omp_device_offload(devnum, hptr, datatype, offset, count, type1, type2) \
{\
   /* devnum: device number \
    * hptr: host poiter \
    * datatype \
    * type1: ``e(n)ter'', ''e(x)it'', or ``u(p)date'' \
    * type2: ``(a)lloc'', ``(t)o'', ``(d)elete'', ''(f)rom'' \
    */ \
   datatype *hypre__offload_hptr = (datatype *) hptr; \
   /* if hypre__global_offload ==    0, or
    *    hptr (host pointer)   == NULL,
    *    this offload will be IGNORED */ \
   if (hypre__global_offload && hypre__offload_hptr != NULL) { \
      /* offloading offset and size (in datatype) */ \
      size_t hypre__offload_offset = offset, hypre__offload_size = count; \
      /* in the CHECK mode, we test if this offload has effect */ \
      HYPRE_OFFLOAD_FLAG(devnum, hypre__offload_hptr, type1) \
      if (hypre__offload_flag) { \
         printf("[!NO Effect! %s %d] device %d target: %6s %6s, data %p, [%ld:%ld]\n", __FILE__, __LINE__, devnum, type1, type2, (void *)hypre__offload_hptr, hypre__offload_offset, hypre__offload_size); exit(0); \
      } else { \
         size_t offload_bytes = count * sizeof(datatype); \
         /* printf("[            %s %d] device %d target: %6s %6s, data %p, [%d:%d]\n", __FILE__, __LINE__, devnum, type1, type2, (void *)hypre__offload_hptr, hypre__offload_offset, hypre__offload_size); */ \
         if (type1[1] == 'n' && type2[0] == 't') { \
            /* enter to */\
            hypre__target_allc_count ++; \
            hypre__target_allc_bytes += offload_bytes; \
            hypre__target_htod_count ++; \
            hypre__target_htod_bytes += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target enter data map(to:hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'n' && type2[0] == 'a') { \
            /* enter alloc */ \
            hypre__target_allc_count ++; \
            hypre__target_allc_bytes += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target enter data map(alloc:hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'x' && type2[0] == 'd') { \
            /* exit delete */\
            hypre__target_free_count ++; \
            hypre__target_free_bytes += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target exit data map(delete:hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'x' && type2[0] == 'f') {\
            /* exit from */ \
            hypre__target_free_count ++; \
            hypre__target_free_bytes += offload_bytes; \
            hypre__target_dtoh_count ++; \
            hypre__target_dtoh_bytes += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target exit data map(from:hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'p' && type2[0] == 't') { \
            /* update to */ \
            hypre__target_htod_count ++; \
            hypre__target_htod_bytes += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target update to(hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else if (type1[1] == 'p' && type2[0] == 'f') {\
            /* update from */ \
            hypre__target_dtoh_count ++; \
            hypre__target_dtoh_bytes += offload_bytes; \
            _Pragma (HYPRE_XSTR(omp target update from(hypre__offload_hptr[hypre__offload_offset:hypre__offload_size]))) \
         } else {\
            printf("error: unrecognized offloading type combination!\n"); exit(-1); \
         } \
      } \
   } \
}

HYPRE_Int HYPRE_OMPOffload(HYPRE_Int device, void *ptr, size_t num, const char *type1,
                           const char *type2);
HYPRE_Int HYPRE_OMPPtrIsMapped(void *p, HYPRE_Int device_num);
HYPRE_Int HYPRE_OMPOffloadOn(void);
HYPRE_Int HYPRE_OMPOffloadOff(void);
HYPRE_Int HYPRE_OMPOffloadStatPrint(void);

#endif /* HYPRE_USING_DEVICE_OPENMP */
#endif /* HYPRE_OMP_DEVICE_H */

