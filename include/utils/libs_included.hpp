#ifndef LIBS_INCLUDED_HPP
#define LIBS_INCLUDED_HPP

/* C++ standard libraries */
#include <iostream>
#include "boost/filesystem.hpp"
#include <chrono>
#include <assert.h>
#include <string>
#include <vector>
#include <math.h>

/* PCL Utils */
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/geometry.h>

/* Eigen Supports */
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>

/* OpenMP for Multi-threads acceleration. */
#include <omp.h>

#endif