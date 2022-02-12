/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef __SYCL_DEVICE_HPP__
#define __SYCL_DEVICE_HPP__

#ifdef HYPRE_USING_SYCL

#include <memory>
#include <cstring>
#include <mutex>
#include <map>
#include <vector>
#include <thread>

#include <unistd.h>
#include <sys/syscall.h>

class device_ext : public sycl::device {
public:
  device_ext() : sycl::device(), _ctx(*this) {}
  ~device_ext() {
    std::lock_guard<std::mutex> lock(m_mutex);
  }
  device_ext(const sycl::device &base)
      : sycl::device(base), _ctx(*this) {
  }

private:
  sycl::context _ctx;
  mutable std::mutex m_mutex;
};

static inline int get_tid(){
  return syscall(SYS_gettid);
}

class dev_mgr {
public:
  int current_device() {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it=_thread2dev_map.find(get_tid());
    if(it != _thread2dev_map.end()) {
      check_id(it->second);
      return it->second;
    }
    printf("WARNING: no SYCL device found in the map, returning DEFAULT_DEVICE_ID\n");
    return DEFAULT_DEVICE_ID;
  }
  device_ext* get_sycl_device(int id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    check_id(id);
    return _devs[id].get();
  }
  void select_device(int id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    check_id(id);
    _thread2dev_map[get_tid()]=id;
  }
  int device_count() { return _devs.size(); }

  /// Returns the instance of device manager singleton.
  static dev_mgr &instance() {
    static dev_mgr d_m;
    return d_m;
  }
  dev_mgr(const dev_mgr &) = delete;
  dev_mgr &operator=(const dev_mgr &) = delete;
  dev_mgr(dev_mgr &&) = delete;
  dev_mgr &operator=(dev_mgr &&) = delete;

private:
  mutable std::mutex m_mutex;

  dev_mgr() {
    std::vector<sycl::device> sycl_all_devs = sycl::device::get_devices(sycl::info::device_type::gpu);
    for (auto &dev : sycl_all_devs) {
      if (dev.get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
         auto subDevicesDomainNuma = dev.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>
            (sycl::info::partition_affinity_domain::numa);
         for (auto &tile : subDevicesDomainNuma) {
           _devs.push_back(std::make_shared<device_ext>(tile));
         }
      }
      else {
        _devs.push_back(std::make_shared<device_ext>(dev));
      }
    }
  }

  void check_id(int id) const {
    if (id >= _devs.size()) {
      throw std::runtime_error("invalid device id");
    }
  }

  std::vector<std::shared_ptr<device_ext>> _devs;
  /// DEFAULT_DEVICE_ID is used, if current_device() can not find current
  /// thread id in _thread2dev_map, which means default device should be used
  /// for the current thread.
  const int DEFAULT_DEVICE_ID = 0;
  /// thread-id to device-id map.
  std::map<int, int> _thread2dev_map;
};

/// Util function to get the current device (in int).
static inline void syclGetDevice(int* id) {
  *id = dev_mgr::instance().current_device();
}

/// Util function to get the current sycl::device by id.
static inline device_ext* sycl_get_device(int id) {
  return dev_mgr::instance().get_sycl_device(id);
}

/// Util function to set a device by id. (to _thread2dev_map)
static inline void syclSetDevice(int id) {
  dev_mgr::instance().select_device(id);
}

/// Util function to get number of GPU devices (default: explicit scaling)
static inline void syclGetDeviceCount(int* id) {
  *id = dev_mgr::instance().device_count();
}

#endif // HYPRE_USING_SYCL

#endif // __SYCL_DEVICE_HPP__
