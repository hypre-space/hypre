/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

#ifdef HYPRE_USING_NVTX

#include <string>
#include <algorithm>
#include <vector>
#include "nvToolsExt.h"
#include "nvToolsExtCudaRt.h"

/* 16 named colors by HTML 4.01. Repalce white with Orange */
typedef enum
{
   /* White, */
   Orange,
   Silver,
   Gray,
   Black,
   Red,
   Maroon,
   Yellow,
   Olive,
   Lime,
   Green,
   Aqua,
   Teal,
   Blue,
   Navy,
   Fuchsia,
   Purple
} color_names;

static const uint32_t colors[] =
{
   /* 0xFFFFFF, */
   0xFFA500,
   0xC0C0C0,
   0x808080,
   0x000000,
   0xFF0000,
   0x800000,
   0xFFFF00,
   0x808000,
   0x00FF00,
   0x008000,
   0x00FFFF,
   0x008080,
   0x0000FF,
   0x000080,
   0xFF00FF,
   0x800080
};

static const HYPRE_Int hypre_nvtx_num_colors = sizeof(colors) / sizeof(uint32_t);
static std::vector<std::string> hypre_nvtx_range_names;

#endif

void hypre_NvtxPushRangeColor(const char *name, HYPRE_Int color_id)
{
#ifdef HYPRE_USING_NVTX
   color_id = color_id % hypre_nvtx_num_colors;
   nvtxEventAttributes_t eventAttrib = {0};
   eventAttrib.version = NVTX_VERSION;
   eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
   eventAttrib.colorType = NVTX_COLOR_ARGB;
   eventAttrib.color = colors[color_id];
   eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
   eventAttrib.message.ascii = name;
   nvtxRangePushEx(&eventAttrib);
#endif
}

void hypre_NvtxPushRange(const char *name)
{
#ifdef HYPRE_USING_NVTX
   std::vector<std::string>::iterator p = std::find(hypre_nvtx_range_names.begin(),
                                                    hypre_nvtx_range_names.end(),
                                                    name);

   if (p == hypre_nvtx_range_names.end())
   {
      hypre_nvtx_range_names.push_back(name);
      p = hypre_nvtx_range_names.end() - 1;
   }

   HYPRE_Int color = p - hypre_nvtx_range_names.begin();

   hypre_NvtxPushRangeColor(name, color);
#endif
}

void hypre_NvtxPopRange()
{
#ifdef HYPRE_USING_NVTX
   hypre_NvtxPushRangeColor("StreamSync0", Red);
   cudaStreamSynchronize(0);
   nvtxRangePop();
   nvtxRangePop();
#endif
}



//nvtxRangePushA(name);
