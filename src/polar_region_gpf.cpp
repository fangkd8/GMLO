/*
 * Copyright (C) 2019 by AutoSense Organization. All rights reserved.
 * Gary Chan <chenshj35@mail2.sysu.edu.cn>
 */
#include "polar_region_gpf.hpp"

#include <pcl/filters/extract_indices.h>  // pcl::ExtractIndices
#include <pcl/io/io.h>                    // pcl::copyPointCloud

namespace gicp_mapping{


template <typename PointT>
void RegionGPFSegmenter<PointT>::segment(
  const PointCloudType<PointT> &cloud_in,
  PointCloudTypePtr<PointT> gnd_out,
  PointCloudTypePtr<PointT> obj_out
){
  std::vector<std::vector<int>> segment_indices(params_.gpf_num_bins);
  const double res = (2 * M_PI) / params_.gpf_num_bins;

  for (size_t pt = 0u; pt < cloud_in.points.size(); ++pt){
    double AlphaVal = -M_PI;
    for (size_t idx = 0u; idx < params_.gpf_num_bins; ++idx){
      const double alpha = std::atan2(cloud_in.points[pt].y, cloud_in.points[pt].x);
      if (alpha >= AlphaVal && alpha < (AlphaVal + res)){
        segment_indices[idx].push_back(pt);
      }
      AlphaVal += res;
    }
  }

  for (size_t segmentIdx = 0u; segmentIdx < params_.gpf_num_bins; ++segmentIdx) {
    PointCloudType<PointT> cloud_segment;
    pcl::copyPointCloud(cloud_in, segment_indices[segmentIdx], cloud_segment);

    int range_seg_num = range_list_.size() + 1;
    std::vector<PointCloudType<PointT>> pts_range(range_seg_num, PointCloudType<PointT>());
    // Single thread implementation.
    for (int index = 0; index < cloud_segment.size(); index++){
      double dist = pointDist<PointT>(cloud_segment.at(index));
      if (dist < ground_min_range_){
        obj_out->push_back(cloud_segment.at(index));
      }
      else{
        pts_range[rangeInd(dist)].push_back(cloud_segment.at(index));
      }
    }

    for (int i = 0; i < pts_range.size(); i++){
      if (pts_range[i].size() != 0){
        segmentGPF(pts_range[i], *gnd_out, *obj_out);
      }
    }
  }
}

template <typename PointT>
void RegionGPFSegmenter<PointT>::segment_threads(
  const PointCloudType<PointT> &cloud_in,
  PointCloudTypePtr<PointT> gnd_out,
  PointCloudTypePtr<PointT> obj_out
){
  std::vector<std::vector<int>> segment_indices(params_.gpf_num_bins);
  const double res = (2 * M_PI) / params_.gpf_num_bins;

  for (size_t pt = 0u; pt < cloud_in.points.size(); ++pt){
    double AlphaVal = -M_PI;
    for (size_t idx = 0u; idx < params_.gpf_num_bins; ++idx){
      const double alpha = std::atan2(cloud_in.points[pt].y, cloud_in.points[pt].x);
      if (alpha >= AlphaVal && alpha < (AlphaVal + res)){
        segment_indices[idx].push_back(pt);
      }
      AlphaVal += res;
    }
  }

  std::vector<std::thread> threads(params_.gpf_num_bins);
  std::vector<PointCloudType<PointT>> gnd_pts_vec(params_.gpf_num_bins);
  std::vector<PointCloudType<PointT>> obj_pts_vec(params_.gpf_num_bins);
  std::vector<std::vector<model_t>>   gnd_info_vec(params_.gpf_num_bins);
  for (int segIdx = 0; segIdx < params_.gpf_num_bins; segIdx++){
    threads[segIdx] = std::thread(
      &RegionGPFSegmenter::threadSegGPF, this, 
      std::ref(cloud_in), 
      std::ref(segment_indices),
      std::ref(gnd_pts_vec),
      std::ref(obj_pts_vec),
      segIdx
    );
  }
  for (auto it = threads.begin(); it != threads.end(); ++it) {
    it->join();
  }
  for (int i = 0; i < params_.gpf_num_bins; i++){
    *gnd_out += gnd_pts_vec[i];
    *obj_out += obj_pts_vec[i];
  }
}

template <typename PointT>
void RegionGPFSegmenter<PointT>::threadSegGPF(
  const PointCloudType<PointT> &cloud_raw,
  std::vector<std::vector<int>> &seg_inds,
  std::vector<PointCloudType<PointT>> &gnd_pts,
  std::vector<PointCloudType<PointT>> &obj_pts,
  int thread_num
){
  PointCloudType<PointT> cloud_segment;
  pcl::copyPointCloud(cloud_raw, seg_inds[thread_num], cloud_segment);
  
  int range_seg_num = range_list_.size() + 1;
  std::vector<PointCloudType<PointT>> pts_range(range_seg_num, PointCloudType<PointT>());
  for (int i = 0; i < pts_range.size(); i++){
    pts_range[i].resize(cloud_segment.size());
  }
  std::vector<int> inds(range_seg_num, 0);
  PointCloudType<PointT> close_obj;
  for (int index = 0; index < cloud_segment.size(); index++){
    double dist = pointDist<PointT>(cloud_segment.at(index));
    if (dist < ground_min_range_){
      close_obj.push_back(cloud_segment.at(index));
    }
    else{
      pts_range[rangeInd(dist)].at(inds[rangeInd(dist)]) = cloud_segment.at(index);
      inds[rangeInd(dist)] += 1;
    }
  }
  for (int i = 0; i < pts_range.size(); i++){
    pts_range[i].resize(inds[i]);
    if (pts_range[i].size() != 0){
      segmentGPF(pts_range[i], gnd_pts[thread_num], obj_pts[thread_num]);
    }
  }
  obj_pts[thread_num] += close_obj;
}

template <typename PointT>
void RegionGPFSegmenter<PointT>::segmentGPF(
  const PointCloudType<PointT> &cloud_in,
  PointCloudType<PointT> &gnd_pts,
  PointCloudType<PointT> &obj_pts)
{
  PointCloudTypePtr<PointT> ngnds(new PointCloudType<PointT>);
  PointCloudTypePtr<PointT>  gnds(new PointCloudType<PointT>);
  mainLoop(cloud_in, gnds, ngnds);
  model_t model = estimatePlane(*gnds);
  float epsilon_rad = (params_.deg_thresh_ / 180) * 3.14;
  Eigen::Vector3f normal = Eigen::Vector3f(model.normal(0, 0), model.normal(1, 0), model.normal(2, 0));
  Eigen::Vector3f neg_G = Eigen::Vector3f(0, 0, 1);
  float normal_angle = std::acos(normal.dot(neg_G));
  if (
    model.center.z() > params_.height_thresh_ - params_.gpf_sensor_height ||
    std::abs(model.d) > params_.d_thre_ ||
    Eigen::isnan(model.normal.array()).any() ||
    (normal_angle > epsilon_rad && normal_angle < (M_PI - epsilon_rad))
  ){
    obj_pts += *ngnds;
    obj_pts += *gnds;
  }
  else{
    // voxelFiltering(gnds, gnds);
    std::vector<model_t> gnd_models(gnds->size(), model);
    gnd_pts += *gnds;
    obj_pts += *ngnds;
    if (test){
      std::cout << model.center.transpose() << std::endl;
      std::cout << model.normal.transpose() << ", d = " << model.d << std::endl;
      Eigen::Vector3f normal = Eigen::Vector3f(model.normal(0, 0), model.normal(1, 0), model.normal(2, 0));
      Eigen::Vector3f neg_G = Eigen::Vector3f(0, 0, 1);
      std::cout << normal_angle << " " << epsilon_rad << std::endl;
      std::cout << std::endl;
    }
    // gnd_info.insert(gnd_info.end(), gnd_models.begin(), gnd_models.end());
  }
}

template <typename PointT>
int RegionGPFSegmenter<PointT>::rangeInd(double dist){
  for (int i = 0; i < range_list_.size(); i++){
    if (dist <= range_list_[i])
      return i;
  }
  return range_list_.size();
}


template <typename PointT>
void RegionGPFSegmenter<PointT>::extractInitialSeeds(
    const PointCloudType<PointT>& cloud_in, PointCloudTypePtr<PointT> cloud_seeds) {
    std::vector<PointT> points(cloud_in.points.begin(), cloud_in.points.end());
    std::sort(points.begin(), points.end(), sortByAxisZAsc<PointT>);

    int cnt_lpr = 0;
    double height_average = 0.;
    // filter negative obstacles
    bool negative = true;
    for (size_t pt = 0u; pt < points.size() && cnt_lpr < params_.gpf_num_lpr;
         ++pt) {
        const double& h = points[pt].z;
        if (negative) {
            if (fabs(h + params_.gpf_sensor_height) > params_.gpf_th_lprs) {
                continue;
            } else {
                // because points are in "Incremental Order"
                negative = false;
            }
        }
        // collect from non-negative obstacles
        height_average += h;
        cnt_lpr++;
    }

    if (cnt_lpr > 0) {
        height_average /= cnt_lpr;
    } else {
        height_average = (-1.0) * params_.gpf_sensor_height;
    }

    // the points inside the height threshold are used as the initial seeds for
    // the plane model estimation
    (*cloud_seeds).clear();
    (*cloud_seeds).resize(points.size());
    int seed_ind = 0;
    for (size_t pt = 0u; pt < points.size(); ++pt) {
        if (points[pt].z < height_average + params_.gpf_th_seeds) {
            (*cloud_seeds).at(seed_ind) = points[pt];
            seed_ind += 1;
        }
    }
    (*cloud_seeds).resize(seed_ind);
}

template <typename PointT>
model_t RegionGPFSegmenter<PointT>::estimatePlane(
    const PointCloudType<PointT>& cloud_ground) {
    model_t model;

    // Create covariance matrix.
    // 1. calculate (x,y,z) mean
    float mean_x = 0., mean_y = 0., mean_z = 0.;
    for (size_t pt = 0u; pt < cloud_ground.points.size(); ++pt) {
        mean_x += cloud_ground.points[pt].x;
        mean_y += cloud_ground.points[pt].y;
        mean_z += cloud_ground.points[pt].z;
    }
    if (cloud_ground.points.size()) {
        mean_x /= cloud_ground.points.size();
        mean_y /= cloud_ground.points.size();
        mean_z /= cloud_ground.points.size();
    }
    // 2. calculate covariance
    // cov(x,x), cov(y,y), cov(z,z)
    // cov(x,y), cov(x,z), cov(y,z)
    float cov_xx = 0., cov_yy = 0., cov_zz = 0.;
    float cov_xy = 0., cov_xz = 0., cov_yz = 0.;
    for (int i = 0; i < cloud_ground.points.size(); i++) {
        cov_xx += (cloud_ground.points[i].x - mean_x) *
                  (cloud_ground.points[i].x - mean_x);
        cov_xy += (cloud_ground.points[i].x - mean_x) *
                  (cloud_ground.points[i].y - mean_y);
        cov_xz += (cloud_ground.points[i].x - mean_x) *
                  (cloud_ground.points[i].z - mean_z);
        cov_yy += (cloud_ground.points[i].y - mean_y) *
                  (cloud_ground.points[i].y - mean_y);
        cov_yz += (cloud_ground.points[i].y - mean_y) *
                  (cloud_ground.points[i].z - mean_z);
        cov_zz += (cloud_ground.points[i].z - mean_z) *
                  (cloud_ground.points[i].z - mean_z);
    }
    // 3. setup covariance matrix Cov
    Eigen::MatrixXf Cov(3, 3);
    Cov << cov_xx, cov_xy, cov_xz, cov_xy, cov_yy, cov_yz, cov_xz, cov_yz,
        cov_zz;
    Cov /= cloud_ground.points.size();

    // Singular Value Decomposition: SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> SVD(
        Cov, Eigen::DecompositionOptions::ComputeFullU);
    // use the least singular vector as normal
    model.normal = (SVD.matrixU().col(2));
    // if (model.normal(2, 0) < 0){
    //   model.normal *= -1;
    // }
    // d is directly computed substituting x with s^ which is a good
    // representative for the points belonging to the plane
    Eigen::MatrixXf mean_seeds(3, 1);
    mean_seeds << mean_x, mean_y, mean_z;
    // according to normal^T*[x,y,z]^T = -d
    model.d = -(model.normal.transpose() * mean_seeds)(0, 0);
    model.center = Eigen::Vector3f(mean_x, mean_y, mean_z);

    return model;
}

template <typename PointT>
void RegionGPFSegmenter<PointT>::mainLoop(const PointCloudType<PointT>& cloud_in,
                                           PointCloudTypePtr<PointT> cloud_gnds,
                                           PointCloudTypePtr<PointT> cloud_ngnds) {
    cloud_gnds->clear();
    cloud_ngnds->clear();

    PointCloudTypePtr<PointT> cloud_seeds(new PointCloudType<PointT>);
    extractInitialSeeds(cloud_in, cloud_seeds);

    pcl::PointIndices::Ptr gnds_indices(new pcl::PointIndices);
    *cloud_gnds = *cloud_seeds;
    for (size_t iter = 0u; iter < params_.gpf_num_iter; ++iter) {
        model_t model = estimatePlane(*cloud_gnds);
        // clear
        cloud_gnds->clear();
        gnds_indices->indices.clear();
        // pointcloud to matrix
        Eigen::MatrixXf cloud_matrix(cloud_in.points.size(), 3);
        size_t pi = 0u;
        for (auto p : cloud_in.points) {
            cloud_matrix.row(pi++) << p.x, p.y, p.z;
        }
        // distance to extimated ground plane model (N^T X)^T = (X^T N)
        Eigen::VectorXf dists = cloud_matrix * model.normal;
        // threshold filter: N^T xi + d = dist < th_dist ==> N^T xi < th_dist -
        // d
        double th_dist = params_.gpf_th_gnds - model.d;

        gnds_indices->indices.resize(dists.rows());
        int gnd_ind_id = 0;
        for (size_t pt = 0u; pt < dists.rows(); ++pt) {
            if (dists[pt] < th_dist) {
                gnds_indices->indices[gnd_ind_id] = pt;
                gnd_ind_id += 1;
            }
        }
        gnds_indices->indices.resize(gnd_ind_id);
        // extract ground points
        pcl::copyPointCloud(cloud_in, *gnds_indices, *cloud_gnds);
    }

    // extract non-ground points
    pcl::ExtractIndices<PointT> indiceExtractor;
    indiceExtractor.setInputCloud(cloud_in.makeShared());
    indiceExtractor.setIndices(gnds_indices);
    indiceExtractor.setNegative(true);
    indiceExtractor.filter(*cloud_ngnds);
}
}

template class gicp_mapping::RegionGPFSegmenter<pcl::PointXYZ>;
template class gicp_mapping::RegionGPFSegmenter<pcl::PointXYZRGB>;
template class gicp_mapping::RegionGPFSegmenter<pcl::PointXYZI>;
