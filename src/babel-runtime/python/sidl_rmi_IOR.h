/*
 * File:          sidl_rmi_IOR.h
 * Symbol:        sidl.rmi-v0.9.15
 * Symbol Type:   package
 * Babel Version: 1.0.4
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Intermediate Object Representation for sidl.rmi
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

#ifndef included_sidl_rmi_IOR_h
#define included_sidl_rmi_IOR_h

/*
 * Symbol "sidl.rmi" (version 0.9.15)
 * 
 *  
 * This package contains necessary interfaces for RMI protocols to
 * hook into Babel, plus a Protocol Factory class.  The intention is
 * that authors of new protocols will create classes that implement
 * InstanceHandle, Invocation and Response (they could even have one
 * object that implements all three interfaces).
 */

#ifndef included_sidl_rmi_BindException_IOR_h
#include "sidl_rmi_BindException_IOR.h"
#endif
#ifndef included_sidl_rmi_Call_IOR_h
#include "sidl_rmi_Call_IOR.h"
#endif
#ifndef included_sidl_rmi_ConnectException_IOR_h
#include "sidl_rmi_ConnectException_IOR.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_IOR_h
#include "sidl_rmi_ConnectRegistry_IOR.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_IOR_h
#include "sidl_rmi_InstanceHandle_IOR.h"
#endif
#ifndef included_sidl_rmi_InstanceRegistry_IOR_h
#include "sidl_rmi_InstanceRegistry_IOR.h"
#endif
#ifndef included_sidl_rmi_Invocation_IOR_h
#include "sidl_rmi_Invocation_IOR.h"
#endif
#ifndef included_sidl_rmi_MalformedURLException_IOR_h
#include "sidl_rmi_MalformedURLException_IOR.h"
#endif
#ifndef included_sidl_rmi_NetworkException_IOR_h
#include "sidl_rmi_NetworkException_IOR.h"
#endif
#ifndef included_sidl_rmi_NoRouteToHostException_IOR_h
#include "sidl_rmi_NoRouteToHostException_IOR.h"
#endif
#ifndef included_sidl_rmi_NoServerException_IOR_h
#include "sidl_rmi_NoServerException_IOR.h"
#endif
#ifndef included_sidl_rmi_ObjectDoesNotExistException_IOR_h
#include "sidl_rmi_ObjectDoesNotExistException_IOR.h"
#endif
#ifndef included_sidl_rmi_ProtocolException_IOR_h
#include "sidl_rmi_ProtocolException_IOR.h"
#endif
#ifndef included_sidl_rmi_ProtocolFactory_IOR_h
#include "sidl_rmi_ProtocolFactory_IOR.h"
#endif
#ifndef included_sidl_rmi_Response_IOR_h
#include "sidl_rmi_Response_IOR.h"
#endif
#ifndef included_sidl_rmi_Return_IOR_h
#include "sidl_rmi_Return_IOR.h"
#endif
#ifndef included_sidl_rmi_ServerInfo_IOR_h
#include "sidl_rmi_ServerInfo_IOR.h"
#endif
#ifndef included_sidl_rmi_ServerRegistry_IOR_h
#include "sidl_rmi_ServerRegistry_IOR.h"
#endif
#ifndef included_sidl_rmi_Ticket_IOR_h
#include "sidl_rmi_Ticket_IOR.h"
#endif
#ifndef included_sidl_rmi_TicketBook_IOR_h
#include "sidl_rmi_TicketBook_IOR.h"
#endif
#ifndef included_sidl_rmi_TimeOutException_IOR_h
#include "sidl_rmi_TimeOutException_IOR.h"
#endif
#ifndef included_sidl_rmi_UnexpectedCloseException_IOR_h
#include "sidl_rmi_UnexpectedCloseException_IOR.h"
#endif
#ifndef included_sidl_rmi_UnknownHostException_IOR_h
#include "sidl_rmi_UnknownHostException_IOR.h"
#endif

#endif
