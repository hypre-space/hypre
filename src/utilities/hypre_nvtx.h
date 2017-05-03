/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#ifdef USE_NVTX
#include "nvToolsExt.h"
#include "nvToolsExtCudaRt.h"

static const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
static const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxDomainRangePushEx(HYPRE_DOMAIN,&eventAttrib);	\
}

#define PUSH_RANGE_PAYLOAD(name,cid,load) {		\
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_INT64; \
    eventAttrib.payload.llValue = load; \
    eventAttrib.category=1; \
    nvtxDomainRangePushEx(HYPRE_DOMAIN,&eventAttrib); \
}

#define PUSH_RANGE_DOMAIN(name,cid,dId) {				\
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxDomainRangePushEx(getdomain(dId),&eventAttrib);	\
}

#define POP_RANGE nvtxDomainRangePop(HYPRE_DOMAIN);
#define POP_RANGE_DOMAIN(dId) {			\
  nvtxDomainRangePop(getdomain(dId));		\
  }
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#define PUSH_RANGE_PAYLOAD(name,cid,load)
#define PUSH_RANGE_DOMAIN(name,cid,domainName)
#endif

