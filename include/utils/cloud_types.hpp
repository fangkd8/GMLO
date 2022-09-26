#ifndef CLOUD_TYPES_HPP
#define CLOUD_TYPES_HPP
#include "libs_included.hpp"
#include "nanoflann/nanoflann.hpp"

namespace gicp_mapping{

template <typename PointT>
using PointCloudType = typename pcl::PointCloud<PointT>;

template <typename PointT>
using PointCloudTypePtr = typename pcl::PointCloud<PointT>::Ptr;

template <typename PointT>
using KDTreeType = typename nanoflann::KdTreeFLANN<PointT>;

template <typename PointT>
using KDTreeTypePtr = typename nanoflann::KdTreeFLANN<PointT>::Ptr;

typedef std::vector<Eigen::Matrix3f>* covVecPtr;
typedef std::vector<Eigen::Matrix3f>  covVec;

}

#endif