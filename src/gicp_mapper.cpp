#include "gicp_mapper.hpp"

namespace gicp_mapping {

template <typename PointT>
gicp_mapper<PointT>::gicp_mapper(){
  num_threads_ = omp_get_num_procs();
}

template <typename PointT>
void gicp_mapper<PointT>::addKeyFrame(const KeyFrame<PointT> &kf){
  keys.push_back(kf);
  if (keys.size() > mapSize)
    keys.pop_front();
  frame_count_ += 1;
}

template <typename PointT>
void gicp_mapper<PointT>::updateMap(){
  currPln.clear();
  currGnd.clear();
  currCovs.clear();

  Sophus::SE3f T_inv = keys.back().Pose.inverse();
  for (int i = 0; i < keys.size() - 1; i++){
    PointCloudType<PointT> curr_pln_data;
    PointCloudType<PointT> curr_gnd_data;
    covVec                 curr_covs;

    Sophus::SE3f delta = T_inv * keys[i].Pose;
    Eigen::Matrix3f R = delta.matrix().block<3, 3>(0, 0);
    transformCov2Curr(curr_covs, keys[i].plnCovs, R);

    pcl::transformPointCloud(keys[i].plnCloud, curr_pln_data, delta.matrix());
    pcl::transformPointCloud(keys[i].gndCloud, curr_gnd_data, delta.matrix());
    currPln += curr_pln_data;
    currGnd += curr_gnd_data;

    currCovs.insert( currCovs.end(), curr_covs.begin(), curr_covs.end() );
  }
}

template <typename PointT>
Sophus::SE3f gicp_mapper<PointT>::mapAlign(
  Sophus::SE3f init,
  KeyFrame<PointT> &frame
){
  if (!dataFull() || !isKey(frame.Pose))
    return init;
  updateMap();
  
  assert(!currPln.points.empty() && !currCovs.empty());
  assert(!currGnd.points.empty());
  gicp_mp<PointT> gicp_map(10, 5);
  gicp_map.setVoxelSize(voxel_);
  gicp_map.setKCorrespondence(30);
  gicp_map.setInputTarget(currPln.makeShared(), currCovs, currGnd.makeShared());
  gicp_map.setInputSource(frame.plnCloud.makeShared(), frame.plnCovs, frame.gndCloud.makeShared());
  gicp_map.align(init);
  return gicp_map.getFinalTransformationSE3();
}

template <typename PointT>
void gicp_mapper<PointT>::clear_all(){
  plnMap.clear();
  gndMap.clear();
  currPln.clear();
  currGnd.clear();
  currCovs.clear();
  keys.clear();
}


template <typename PointT>
bool gicp_mapper<PointT>::isKey(Sophus::SE3f &pose_in){
  if (frame_count_ < map_active_){
    return false;
  }
  Sophus::SE3f delta = keyPose.inverse() * pose_in;
  Eigen::Matrix3f R = delta.matrix().block<3, 3>(0, 0);
  Eigen::Vector3f t = delta.matrix().block<3, 1>(0, 3);
  
  float theta = std::acos((R.trace() - 1) / 2.0) * (180 / M_PI);
  float trans = t.norm();
  if (theta < deg_thre_ && trans < trans_thre_){
    return false;
  }
  return true;
}

template <typename PointT>
void gicp_mapper<PointT>::transformCov2Curr(
  covVec &covs_out, 
  covVec &local_covs, 
  Eigen::Matrix3f &R
)
{
  covs_out.resize(local_covs.size(), Eigen::Matrix3f::Zero());
  #pragma omp parallel for num_threads(num_threads_)
  for (int i = 0; i < covs_out.size(); i++){
    covs_out[i] = R * local_covs[i] * R.transpose();
  }
}

template <typename PointT>
void gicp_mapper<PointT>::getMapData(
  PointCloudType<PointT> &pln_out,
  PointCloudType<PointT> &gnd_out
)
{
  assert(pln_out.size() == 0);
  assert(gnd_out.size() == 0);
  plnMap.clear();
  gndMap.clear();
  for (int i = 0; i < keys.size(); i++){
    PointCloudType<PointT> kf_pln_world;
    PointCloudType<PointT> kf_gnd_world;
    pcl::transformPointCloud(keys[i].plnCloud, kf_pln_world, keys[i].Pose.matrix());
    pcl::transformPointCloud(keys[i].gndCloud, kf_gnd_world, keys[i].Pose.matrix());
    plnMap += kf_pln_world;
    gndMap += kf_gnd_world;
  }
  pln_out = plnMap;
  gnd_out = gndMap;
  assert(pln_out.size() == plnMap.size());
  assert(gnd_out.size() == gndMap.size());
}

template <typename PointT>
void gicp_mapper<PointT>::getCurrMap(
  PointCloudType<PointT> &map_out
)
{
  map_out = currMap;
}

template <typename PointT>
void gicp_mapper<PointT>::getCurrMap(
  PointCloudType<PointT> &pln_out,
  PointCloudType<PointT> &gnd_out
)
{
  assert(pln_out.size() == 0);
  assert(gnd_out.size() == 0);
  pln_out = currPln;
  gnd_out = currGnd;
  assert(pln_out.size() == currPln.size());
  assert(gnd_out.size() == currGnd.size());
}

template <typename PointT>
void gicp_mapper<PointT>::getPlnCov(covVec &covs_out){
  assert(covs_out.size() == 0);
  covs_out = currCovs;
  assert(covs_out.size() == currCovs.size());
}

template class gicp_mapper<pcl::PointXYZ>;
template class gicp_mapper<pcl::PointXYZRGB>;
template class gicp_mapper<pcl::PointXYZI>;
}