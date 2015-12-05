/*
 * File:          sidl_CastException_fStub.h
 * Symbol:        sidl.CastException-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V1-14-0b $
 * Revision:      @(#) $Id: sidl_CastException_fStub.h,v 1.2 2006/09/14 21:51:52 painter Exp $
 * Description:   Client-side documentation text for sidl.CastException
 * 
 * Copyright (c) 2000-2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
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
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#ifndef included_sidl_CastException_fStub_h
#define included_sidl_CastException_fStub_h

/**
 * Symbol "sidl.CastException" (version 0.9.15)
 * 
 * Exception is thrown when a cast fails and the failure needs to
 * be communicated up the call stack.  (Note: babel _cast does NOT
 * throw this exception)
 */

#ifndef included_sidl_CastException_IOR_h
#include "sidl_CastException_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak sidl_CastException__connectI

#pragma weak sidl_CastException__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct sidl_CastException__object*
sidl_CastException__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct sidl_CastException__object*
sidl_CastException__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
