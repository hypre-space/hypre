/*
 * File:          sidl_rmi_ServerRegistry_fStub.h
 * Symbol:        sidl.rmi.ServerRegistry-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Release:       $Name: V2-2-0b $
 * Revision:      @(#) $Id: sidl_rmi_ServerRegistry_fStub.h,v 1.3 2006/12/29 21:24:29 painter Exp $
 * Description:   Client-side documentation text for sidl.rmi.ServerRegistry
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

#ifndef included_sidl_rmi_ServerRegistry_fStub_h
#define included_sidl_rmi_ServerRegistry_fStub_h

/**
 * Symbol "sidl.rmi.ServerRegistry" (version 0.9.15)
 * 
 *  
 * This singleton class is simply a place to register a
 * ServerInfo interface for general access.  This ServerInfo
 * should give info about the ORB being used to export RMI objects
 * for the current Babel process.
 * 
 * This Registry provides two important functions, a way to get
 * the URL for local object we wish to expose over RMI, and a way
 * to tell if an object passed to this process via RMI is actually
 * a local object.  This abilities are protocol specific, the
 * ServerInfo interface must by implemented by the protocol
 * writer.
 * 
 * THIS CLASS IS NOT DESIGNED FOR CONCURRENT WRITE ACCESS.  (Only
 * one server is assumed per Babel process)
 */

#ifndef included_sidl_rmi_ServerRegistry_IOR_h
#include "sidl_rmi_ServerRegistry_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak sidl_rmi_ServerRegistry__connectI

#pragma weak sidl_rmi_ServerRegistry__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct sidl_rmi_ServerRegistry__object*
sidl_rmi_ServerRegistry__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct sidl_rmi_ServerRegistry__object*
sidl_rmi_ServerRegistry__connectI(const char * url, sidl_bool ar,
  struct sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
