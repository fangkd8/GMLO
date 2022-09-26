/*
 * Copyright (C) 2019 by AutoSense Organization. All rights reserved.
 * Gary Chan <chenshj35@mail2.sysu.edu.cn>
 */
#ifndef POLAR_REGION_GPF_HPP_
#define POLAR_REGION_GPF_HPP_
#define PCL_NO_PRECOMPILE

#include <Eigen/Core>
#include <string>
#include <vector>
#include <deque>
#include <array>
#include <memory>
#include <thread>

#include <pcl/point_cloud.h> 
#include <pcl/point_types.h>
#include <pcl/filters/approximate_voxel_grid.h>
// #include "ground_plane_fitting_segmenter.hpp"

#include "utils/cloud_types.hpp"

namespace gicp_mapping{

typedef struct {
    Eigen::Vector3f center;
    Eigen::MatrixXf normal;
    double d = 0.;
} model_t;

typedef std::vector<model_t> model_vector;

template <typename PointType>
double pointDist(PointType p){
  return std::sqrt(p.x*p.x + + p.y*p.y + p.z*p.z);
}

template <typename PointType>
bool sortByAxisZAsc(PointType p1, PointType p2) {
    return p1.z < p2.z;
}

struct RegionGPFParams {
  double sensor_max_fov = 2;
  double sensor_min_fov = -24.8;
  // Ground Plane Fitting
  double gpf_sensor_height = 1.73;
  // fitting multiple planes
  int gpf_num_bins = 36;
  // number of iterations
  int gpf_num_iter = 3;
  // number of points used to estimate the lowest point representative(LPR)
  // double of senser model???
  int gpf_num_lpr = 15;
  double gpf_th_lprs = 0.15;
  // threshold for points to be considered initial seeds
  double gpf_th_seeds = 0.2;
  // ground points threshold distance from the plane
  //   <== large to guarantee safe removal
  double gpf_th_gnds = 0.23;

  // local plane tests.
  double deg_thresh_ = 30;
  double height_thresh_ = 0.6;
  double d_thre_ = 10;
};

/**
 * @brief Ground Removal based on Ground Plane Fitting(GPF)
 * @refer
 *   Fast Segmentation of 3D Point Clouds: A Paradigm on LiDAR Data for
 * Autonomous Vehicle Applications (ICRA, 2017)
 */

template <typename PointT>
class RegionGPFSegmenter{
  public:
    RegionGPFSegmenter(){
      ground_min_range_ = params_.gpf_sensor_height / std::sin(params_.sensor_min_fov);
      ground_min_range_ -= 1.5;
    }

    ~RegionGPFSegmenter(){}

    void segment(
      const PointCloudType<PointT> &cloud_in,
      PointCloudTypePtr<PointT> gnd_out,
      PointCloudTypePtr<PointT> obj_out
      // PointCloudTypePtr<PointT> obj_out,
      // std::vector<model_t> &ground_info
    );

    // multi-thread version.
    void segment_threads(
      const PointCloudType<PointT> &cloud_in,
      PointCloudTypePtr<PointT> gnd_out,
      PointCloudTypePtr<PointT> obj_out
      // PointCloudTypePtr<PointT> obj_out,
      // std::vector<model_t> &ground_info
    );

    // for visualization of different segments.
    void segment(
      const PointCloudType<PointT> &cloud_in,
      std::vector<PointCloudType<PointT>> &clouds_out
    );

  private:
    int rangeInd(double dist);

  private:
    void extractInitialSeeds(const PointCloudType<PointT> &cloud_in,
                             PointCloudTypePtr<PointT> cloud_seeds);

    model_t estimatePlane(const PointCloudType<PointT> &cloud_ground);

    void mainLoop(const PointCloudType<PointT> &cloud_in,
                  PointCloudTypePtr<PointT> cloud_gnds,
                  PointCloudTypePtr<PointT> cloud_ngnds);

    void segmentGPF(const PointCloudType<PointT> &cloud_in,
                    PointCloudType<PointT> &gnd_pts,
                    PointCloudType<PointT> &obj_pts);
                    // PointCloudType<PointT> &obj_pts,
                    // std::vector<model_t> &gnd_info);

    void threadSegGPF(
      const PointCloudType<PointT> &cloud_raw,
      std::vector<std::vector<int>> &seg_inds,
      std::vector<PointCloudType<PointT>> &gnd_pts,
      std::vector<PointCloudType<PointT>> &obj_pts,
      // std::vector<std::vector<model_t>>   &gnd_info,
      int thread_num
    );

    void voxelFiltering(PointCloudTypePtr<PointT> cloud_in,
                        PointCloudTypePtr<PointT> cloud_out){
      pcl::ApproximateVoxelGrid<PointT> vg;
      vg.setLeafSize (voxel_, voxel_, voxel_);
      vg.setInputCloud(cloud_in);
      vg.filter(*cloud_out);
    }

  private:
    RegionGPFParams params_;
    static const int segment_nums_ = 36;
    float ground_min_range_ = 0;
    float voxel_ = 0.75;

    std::deque<float> range_list_{
      10, 25, 45, 65
    };

    bool test = false;
};

}
#endif  // SEGMENTERS_INCLUDE_SEGMENTERS_GROUND_PLANE_FITTING_SEGMENTER_HPP_
