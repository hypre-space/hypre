/*
 * File:          sidl_rmi_InstanceHandle_fStub.h
 * Symbol:        sidl.rmi.InstanceHandle-v0.9.15
 * Symbol Type:   interface
 * Babel Version: 1.0.4
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Client-side documentation text for sidl.rmi.InstanceHandle
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

#ifndef included_sidl_rmi_InstanceHandle_fStub_h
#define included_sidl_rmi_InstanceHandle_fStub_h

/**
 * Symbol "sidl.rmi.InstanceHandle" (version 0.9.15)
 * 
 *  
 * This interface holds the state information for handles to
 * remote objects.  Client-side messaging libraries are expected
 * to implement <code>sidl.rmi.InstanceHandle</code>,
 * <code>sidl.rmi.Invocation</code> and
 * <code>sidl.rmi.Response</code>.
 * 
 * Every stub with a connection to a remote object holds a pointer
 * to an InstanceHandle that manages the connection. Multiple
 * stubs may point to the same InstanceHandle, however.  Babel
 * takes care of the reference counting, but the developer should
 * keep concurrency issues in mind.
 * 
 * When a new remote object is created:
 * sidl_rmi_InstanceHandle c = 
 * sidl_rmi_ProtocolFactory_createInstance( url, typeName,
 * _ex );
 * 
 * When a new stub is created to connect to an existing remote
 * instance:
 * sidl_rmi_InstanceHandle c = 
 * sidl_rmi_ProtocolFactory_connectInstance( url, _ex );
 * 
 * When a method is invoked:
 * sidl_rmi_Invocation i = 
 * sidl_rmi_InstanceHandle_createInvocation( methodname );
 * sidl_rmi_Invocation_packDouble( i, "input_val" , 2.0 );
 * sidl_rmi_Invocation_packString( i, "input_str", "Hello" );
 * ...
 * sidl_rmi_Response r = sidl_rmi_Invocation_invokeMethod( i );
 * sidl_rmi_Response_unpackBool( i, "_retval", &succeeded );
 * sidl_rmi_Response_unpackFloat( i, "output_val", &f );
 */

#ifndef included_sidl_rmi_InstanceHandle_IOR_h
#include "sidl_rmi_InstanceHandle_IOR.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif


#pragma weak sidl_rmi_InstanceHandle__connectI

#pragma weak sidl_rmi_InstanceHandle__rmicast

/**
 * Cast method for interface and class type conversions.
 */
struct sidl_rmi_InstanceHandle__object*
sidl_rmi_InstanceHandle__rmicast(
  void* obj, struct sidl_BaseInterface__object **_ex);

/**
 * RMI connector function for the class. (no addref)
 */
struct sidl_rmi_InstanceHandle__object*
sidl_rmi_InstanceHandle__connectI(const char * url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex);

#ifdef __cplusplus
}
#endif
#endif
