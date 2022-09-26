#ifndef GICP_MAPPER_HPP
#define GICP_MAPPER_HPP
#undef NDEBUG
#include "utils/libs_included.hpp"
#include "utils/cloud_types.hpp"
#include "nanoflann/nanoflann.hpp"
#include "polar_region_gpf.hpp"
#include "joint_matching.hpp"
#include "gicp_mp.hpp"
#include <deque>
#include <numeric>
#include <algorithm>

namespace gicp_mapping{

template <typename PointT>
struct KeyFrame{
  /* data */
  Sophus::SE3f           Pose;
  PointCloudType<PointT> plnCloud;
  PointCloudType<PointT> gndCloud;
  covVec                 plnCovs;

  KeyFrame(){};

  KeyFrame(
    Sophus::SE3f           &pose_in, 
    PointCloudType<PointT> &pln_in, 
    PointCloudType<PointT> &gnd_in,
    covVec                 &covs_in
  ){
    Pose = pose_in;
    plnCloud = pln_in;
    gndCloud = gnd_in;
    plnCovs = covs_in;
  }
};

template <typename PointT>
KeyFrame<PointT> generateKeyFrameTgt(joint_matching<PointT> &gicp, Sophus::SE3f &pose){
  PointCloudType<PointT> pln_pts;
  covVec                 pln_cov;
  PointCloudType<PointT> gnd_pts;
  gicp.getTargetPln(pln_pts, pln_cov);
  gicp.getTargetGnd(gnd_pts);
  return KeyFrame<PointT>(pose, pln_pts, gnd_pts, pln_cov);
};

template <typename PointT>
KeyFrame<PointT> generateKeyFrameSrc(joint_matching<PointT> &gicp, Sophus::SE3f &pose){
  PointCloudType<PointT> pln_pts;
  covVec                 pln_cov;
  PointCloudType<PointT> gnd_pts;
  gicp.getSourcePln(pln_pts, pln_cov);
  gicp.getSourceGnd(gnd_pts);
  return KeyFrame<PointT>(pose, pln_pts, gnd_pts, pln_cov);
};


template <typename PointT>
class gicp_mapper{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    gicp_mapper();

    void addKeyFrame(const KeyFrame<PointT> &kf);

    // Transform to most recent frame for scan-to-map registration.
    void updateMap();

    void getMapData(
      PointCloudType<PointT> &pln_out,
      PointCloudType<PointT> &gnd_out  
    );

    void getCurrMap(
      PointCloudType<PointT> &pln_out,
      PointCloudType<PointT> &gnd_out
    );

    void getCurrMap(PointCloudType<PointT> &map_out);

    void getPlnCov(covVec &covs_out);
    void getGndInfo(model_vector &info_out);

    bool dataFull(){
      return (keys.size() == mapSize);
    }

    void setMapSize(int val){
      mapSize = val;
    }

    void setKeyPose(Sophus::SE3f global_pose){
      keyPose = global_pose;
    }

    Sophus::SE3f mapAlign(Sophus::SE3f init, KeyFrame<PointT> &frame);
    void clear_all();

    int mappingSize(){
      return currPln.size() + currGnd.size();
    }

  private:
    void transformCov2Curr(covVec &covs_out, covVec &local_covs, Eigen::Matrix3f &R);

    bool isKey(Sophus::SE3f &pose_in);

    void voxelMap(PointCloudType<PointT> &cloud){
      pcl::ApproximateVoxelGrid<PointT> vg;
      vg.setLeafSize (voxel_, voxel_, voxel_);
      vg.setInputCloud(cloud.makeShared());
      vg.filter(cloud);
    }

    PointCloudType<PointT> plnMap;
    PointCloudType<PointT> gndMap;

    PointCloudType<PointT> currPln;
    PointCloudType<PointT> currGnd;
    covVec                 currCovs;
    model_vector           currInfo;

    PointCloudType<PointT> currMap;

    std::deque<KeyFrame<PointT>> keys;

    Sophus::SE3f keyPose;

    int mapSize = 5;
    int num_threads_ = 1;
    float deg_thre_ = 10.0;
    float trans_thre_ = 10.0;
    float voxel_ = 0.8;
    int frame_count_ = 0;
    int map_active_ = 150;

    std::string s2m_type = "gicp";
};

}

#endif