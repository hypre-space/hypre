/*
 * File:          sidl_rmi_InstanceRegistry_fStub.h
 * Symbol:        sidl.rmi.InstanceRegistry-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Release:       $Name: V2-4-0b $
 * Revision:      @(#) $Id: sidl_rmi_InstanceRegistry_fStub.h,v 1.4 2007/09/27 19:56:36 painter Exp $
 * Description:   Client-side documentation text for sidl.rmi.InstanceRegistry
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

#ifndef included_sidl_rmi_InstanceRegistry_fStub_h
#define included_sidl_rmi_InstanceRegistry_fStub_h

/**
 * Symbol "sidl.rmi.InstanceRegistry" (version 0.9.15)
 * 
 *  
 * This singleton class is implemented by Babel's runtime for RMI
 * libraries to invoke methods on server objects.  It maps
 * objectID strings to sidl_BaseClass objects and vice-versa.
 * 
 * The InstanceRegistry creates and returns a unique string when a
 * new object is added to the registry.  When an object's refcount
 * reaches 0 and it is collected, it is removed from the Instance
 * Registry.
 * 
 * Objects are added to the registry in 3 ways:
 * 1) Added to the server's registry when an object is
 * create[Remote]'d.
 * 2) Implicity added to the local registry when an object is
 * passed as an argument in a remote call.
 * 3) A user may manually add a reference to the local registry
 * for publishing purposes.  The user hsould keep a reference
 * to the object.  Currently, the user cannot provide their own
 * objectID, this capability should probably be added.
 */

#ifndef included_sidl_rmi_InstanceRegistry_IOR_h
#include "sidl_rmi_InstanceRegistry_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak sidl_rmi_InstanceRegistry__connectI

#pragma weak sidl_rmi_InstanceRegistry__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct sidl_rmi_InstanceRegistry__object*
sidl_rmi_InstanceRegistry__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct sidl_rmi_InstanceRegistry__object*
sidl_rmi_InstanceRegistry__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
