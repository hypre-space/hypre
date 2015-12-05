#
# File:          __init__.py
# Symbol:        sidl.rmi-v0.9.15
# Symbol Type:   package
# Babel Version: 1.0.0
# Release:       $Name: V1-14-0b $
# Revision:      @(#) $Id: __init__.py,v 1.4 2006/08/29 22:29:38 painter Exp $
# Description:   package initialization code
# 
# Copyright (c) 2000-2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the Components Team <components@llnl.gov>
# All rights reserved.
# 
# This file is part of Babel. For more information, see
# http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
# for Our Notice and the LICENSE file for the GNU Lesser General Public
# License.
# 
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License (as published by
# the Free Software Foundation) version 2.1 dated February 1999.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
# conditions of the GNU Lesser General Public License for more details.
# 
# You should have recieved a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
# 
# WARNING: Automatically generated; changes will be lost
# 
#


""" 
This package contains necessary interfaces for RMI protocols to
hook into Babel, plus a Protocol Factory class.  The intention is
that authors of new protocols will create classes that implement
InstanceHandle, Invocation and Response (they could even have one
object that implements all three interfaces).
"""

__all__ = [
   "BindException",
   "Call",
   "ConnectException",
   "ConnectRegistry",
   "InstanceHandle",
   "InstanceRegistry",
   "Invocation",
   "MalformedURLException",
   "NetworkException",
   "NoRouteToHostException",
   "NoServerException",
   "ObjectDoesNotExistException",
   "ProtocolException",
   "ProtocolFactory",
   "Response",
   "Return",
   "ServerInfo",
   "ServerRegistry",
   "Ticket",
   "TicketBook",
   "TimeOutException",
   "UnexpectedCloseException",
   "UnknownHostException" ]

try:
  from pkgutil import extend_path
  __path__ = extend_path(__path__, __name__)
except: # ignore all exceptions
  pass
