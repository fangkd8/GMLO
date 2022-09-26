#ifndef UTILS_HPP
#define UTILS_HPP
#include "libs_included.hpp"
#include <bits/stdc++.h>
#include <boost/algorithm/string.hpp>

void PoseToFile(Eigen::Matrix4f mat, std::ostream& file) {
  file << std::scientific << mat(0, 0) << " " << mat(0, 1) << " " << mat(0, 2)
       << " " << mat(0, 3) << " " << mat(1, 0) << " " << mat(1, 1) << " "
       << mat(1, 2) << " " << mat(1, 3) << " " << mat(2, 0) << " " << mat(2, 1)
       << " " << mat(2, 2) << " " << mat(2, 3) << std::endl;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr readKittiPclBinData(const std::string in_file){
  // load point cloud
  std::fstream input(in_file.c_str(), std::ios::in | std::ios::binary);
  if(!input.good()){
    std::cerr << "Could not read file: " << in_file << std::endl;
    exit(EXIT_FAILURE);
  }
  input.seekg(0, std::ios::beg);

  pcl::PointCloud<pcl::PointXYZI>::Ptr points (new pcl::PointCloud<pcl::PointXYZI>);

  for (int i=0; input.good() && !input.eof(); i++) {
      pcl::PointXYZI point;
      input.read((char *) &point.x, 3*sizeof(float));
      input.read((char *) &point.intensity, sizeof(float));
      points->push_back(point);
  }
  input.close();

  pcl::PassThrough<pcl::PointXYZI> pass;
  pass.setInputCloud (points);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (-4, 50);
  pass.filter (*points);

  pcl::CropBox<pcl::PointXYZI> boxFilter;
  boxFilter.setNegative(true);
  boxFilter.setMax(Eigen::Vector4f(2.6, 1.7, 0, 1));
  boxFilter.setMin(Eigen::Vector4f(-2.6, -1.7, -1, 1));
  boxFilter.setInputCloud(points);
  boxFilter.filter(*points);
  return points;
}

template<typename PointT>
void voxelFiltering(typename pcl::PointCloud<PointT>::Ptr raw,
                    typename pcl::PointCloud<PointT>::Ptr cloud,
                    float size){
  pcl::ApproximateVoxelGrid<PointT> vg;
  vg.setLeafSize (size, size, size);
  vg.setInputCloud(raw);
  vg.filter(*cloud);
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr getKITTIData(bool voxel, bool camFrame, std::string filename, float voxel_size){
  typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
  cloud = readKittiPclBinData(filename);
  if (camFrame){
    // align together.
    // KITTI's calibration matrix velodyne --> left camera.
    Eigen::Matrix4f align_mat;
    align_mat << -1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03,
                 -6.481465826011e-03,  8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02,
                  9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01,
                                   0,                   0,                   0,                   1;
    pcl::transformPointCloud(*cloud, *cloud, align_mat);
  }
  if (voxel)
    voxelFiltering<PointT>(cloud, cloud, voxel_size);
  return cloud;
}

std::vector<Eigen::Matrix4f> FileToPose(std::string file){
  std::ifstream file_;
  file_.open(file);
  std::string line;
  std::vector<std::string> poses_str;
  if (file_.is_open()) {
    while (std::getline(file_, line)) {
      //std::cout << line << '\n';
      poses_str.push_back(line);
    }
    file_.close();
  }
  else
    std::cout << "no such file" << std::endl;

  std::vector<Eigen::Matrix4f> result;
  for (auto i = 0; i < poses_str.size(); i++){
  // for (auto i = 3; i < 4; i++){
    std::vector<std::string> pose_line;
    boost::split(pose_line, poses_str[i], boost::is_any_of(" "));
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    for (int j = 0; j < 12; j++){
      // cout << j/4 << " " << j%4 << " " << j << endl;
      pose(j/4, j%4) = stod(pose_line[j]);
    }
    // cout << pose << endl;
    result.push_back(pose);
  }
  return result;
}

#endif