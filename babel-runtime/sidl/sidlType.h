/*
 * File:        sidlType.h
 * Copyright:   (c) 2001 The Regents of the University of California
 * Release:     $Name$
 * Revision:    @(#) $Revision$
 * Date:        $Date$
 * Description: Define the fundamental types for sidl
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

#ifndef included_sidlType_h
#define included_sidlType_h

#ifndef included_babel_config_h
#include "babel_config.h"
#endif

#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#else
#include <sys/types.h>
#endif


/*
 * sidl boolean type (an integer)
 */
typedef int sidl_bool;

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

/*
 * sidl structures for float and double complex
 */
struct sidl_dcomplex {
  double real;
  double imaginary;
};

struct sidl_fcomplex {
  float real;
  float imaginary;
};

#endif /* included_sidlType_h */
