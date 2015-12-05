/*
 * File:        sidl_header.h
 * Copyright:   (c) 2001 The Regents of the University of California
 * Revision:    @(#) $Revision: 1.7 $
 * Date:        $Date: 2006/08/29 22:29:50 $
 * Description: 
 *
 * Copyright (c) 2000-2001, The Regents of the University of Calfornia.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * UCRL-CODE-2002-054
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

#ifndef included_sidl_header_h
#define included_sidl_header_h

#ifndef included_sidlType_h
#include "sidlType.h"
#endif

#ifndef SIDL_INLINE_DECL
#ifdef SIDL_C_HAS_INLINE
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)) || !defined(__GNUC__)
#define SIDL_C_INLINE_DECL inline
#define SIDL_C_INLINE_DEFN extern
#define SIDL_C_INLINE_REPEAT_DEFN 0
#else
#define SIDL_C_INLINE_DECL extern inline
#define SIDL_C_INLINE_DEFN
#define SIDL_C_INLINE_REPEAT_DEFN 1
#endif
#else /* !defined(SIDL_C_HAS_INLINE) */ 
#define SIDL_C_INLINE_DECL
#define SIDL_C_INLINE_DEFN
#define SIDL_C_INLINE_REPEAT_DEFN 1
#endif /* SIDL_C_HAS_INLINE */
#endif /* SIDL_INLINE_DECL */

#ifndef included_sidlArray_h
#include "sidlArray.h"
#endif
#ifndef included_sidlOps_h
#include "sidlOps.h"
#endif
#ifndef included_sidl_double_IOR_h
#include "sidl_double_IOR.h"
#endif
#ifndef included_sidl_float_IOR_h
#include "sidl_float_IOR.h"
#endif
#ifndef included_sidl_dcomplex_IOR_h
#include "sidl_dcomplex_IOR.h"
#endif
#ifndef included_sidl_fcomplex_IOR_h
#include "sidl_fcomplex_IOR.h"
#endif
#ifndef included_sidl_char_IOR_h
#include "sidl_char_IOR.h"
#endif
#ifndef included_sidl_int_IOR_h
#include "sidl_int_IOR.h"
#endif
#ifndef included_sidl_long_IOR_h
#include "sidl_long_IOR.h"
#endif
#ifndef included_sidl_string_IOR_h
#include "sidl_string_IOR.h"
#endif
#ifndef included_sidl_opaque_IOR_h
#include "sidl_opaque_IOR.h"
#endif
#ifndef included_sidl_bool_IOR_h
#include "sidl_bool_IOR.h"
#endif

#endif /* included_sidl_header_h */
