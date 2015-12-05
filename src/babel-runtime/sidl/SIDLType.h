/*
 * File:        SIDLType.h
 * Copyright:   (c) 2001 The Regents of the University of California
 * Release:     $Name: V1-9-0b $
 * Revision:    @(#) $Revision: 1.4 $
 * Date:        $Date: 2003/04/07 21:44:31 $
 * Description: Define the fundamental types for SIDL
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

#ifndef included_SIDLType_h
#define included_SIDLType_h

#ifndef included_babel_config_h
#include "babel_config.h"
#endif

#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#else
#include <sys/types.h>
#endif


/*
 * SIDL boolean type (an integer)
 */
typedef int SIDL_bool;

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

/*
 * SIDL structures for float and double complex
 */
struct SIDL_dcomplex {
  double real;
  double imaginary;
};

struct SIDL_fcomplex {
  float real;
  float imaginary;
};

#endif /* included_SIDLType_h */
