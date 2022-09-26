#ifndef JOINT_MATCHING_HPP
#define JOINT_MATCHING_HPP
#undef NDEBUG

#include "nanoflann/nanoflann.hpp"
#include "polar_region_gpf.hpp"
#include "utils/libs_included.hpp"
#include "utils/cloud_types.hpp"

using namespace std::chrono;

namespace gicp_mapping{

template <typename PointT>
class joint_matching {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    // time configuration.
    std::vector<int> voxel_time;
    std::vector<int> ground_time;
    
    joint_matching(int max_iter_in, int max_iter_GN_in);
    
    void setInputSource(PointCloudTypePtr<PointT> input_cloud);
    void setInputTarget(PointCloudTypePtr<PointT> input_cloud);

    // Set target/source information for scan-to-map align.
    void setSourcePln(const PointCloudType<PointT> &cloud_in, const covVec &covs_in);
    void setTargetPln(const PointCloudType<PointT> &cloud_in, const covVec &covs_in);
    void setSourceGnd(const PointCloudType<PointT> &cloud_in);
    void setTargetGnd(const PointCloudType<PointT> &cloud_in);
    // Return computed informations.
    void getSourcePln(PointCloudType<PointT> &cloud_out, covVec &covs_out);
    void getTargetPln(PointCloudType<PointT> &cloud_out, covVec &covs_out);
    void getSourcePln(PointCloudType<PointT> &cloud_out);
    void getTargetPln(PointCloudType<PointT> &cloud_out);
    void getSourceGnd(PointCloudType<PointT> &cloud_out);
    void getTargetGnd(PointCloudType<PointT> &cloud_out);
    
    bool align(Sophus::SE3f initial_guess);

    void setNextTarget();

    Eigen::Matrix4f getFinalTransformation(){
      return T1_.matrix();
    }
    Sophus::SE3f getFinalTransformationSE3(){
      return T1_;
    }

    void setVoxelSize(float voxel){
      voxel_size_x_ = voxel;
      voxel_size_y_ = voxel;
      voxel_size_z_ = voxel;
    }

    void setKCorrespondence(int k){
      nn_k_ = k;
    }

    void setCurvEps(float eps){
      curv_eps_ = eps;
    }
    
    void setUniformRemain(bool flag){
      keep_uniform_ = flag;
    }

    void setFstGNIter(int iters){
      fst_iter_GN_ = iters;
    }

    void setGndThre(float thre){
      gnd_thre_ = thre;
    }

    void clear_all(){
      target_clear();
      source_clear();
    }

  protected:
    void computeCovariances(
      PointCloudTypePtr<PointT> cloud_in,
      PointCloudTypePtr<PointT> cloud_out,
      KDTreeTypePtr<PointT>     cloud_tree,
      covVecPtr                 pt_covs
    );

    void pln2pln_step(
      int index, PointCloudTypePtr<PointT> source_t,
      KDTreeTypePtr<PointT> target_cloud_tree,
      covVecPtr             target_covariances,
      covVecPtr             source_covariances,
      std::vector<Eigen::Matrix<float, 6, 6>> &A_vec,
      std::vector<Eigen::Matrix<float, 6, 1>> &b_vec
    );

    void localPlaneStep(
      int index, int start, float gnd_weight,
      PointCloudTypePtr<PointT> source_gnd_t,
      KDTreeTypePtr<PointT> target_gnd_tree,
      std::vector<Eigen::Matrix<float, 6, 6>> &A_vec,
      std::vector<Eigen::Matrix<float, 6, 1>> &b_vec
    );

    bool is_converged(const Eigen::Matrix4f &delta);

    bool corresRejector(float dist){
      return (dist <= d_threshold_);
    }

    bool pointValid(PointT p_a){
      if (p_a.x == initPt.x && p_a.y == initPt.y && p_a.z == initPt.z)
        return false;
      return true;
    }
    
    void target_clear();
    void source_clear();

    void voxelFiltering(PointCloudTypePtr<PointT> raw,
                        PointCloudTypePtr<PointT> cloud);

  private:
    PointCloudTypePtr<PointT> source_pln_;
    PointCloudTypePtr<PointT> target_pln_;

    PointCloudTypePtr<PointT> source_pln_voxel_;
    PointCloudTypePtr<PointT> target_pln_voxel_;

    KDTreeTypePtr<PointT> source_pln_tree_;
    KDTreeTypePtr<PointT> target_pln_tree_;

    covVecPtr source_covs_;
    covVecPtr target_covs_;

    PointCloudTypePtr<PointT> source_gnd_;
    PointCloudTypePtr<PointT> target_gnd_;
    std::vector<model_t> source_gnd_info_;
    std::vector<model_t> target_gnd_info_;

    KDTreeTypePtr<PointT> source_gnd_tree_;
    KDTreeTypePtr<PointT> target_gnd_tree_;

    Sophus::SE3f T0_;
    Sophus::SE3f T1_;

    // GICP hyperparameters.
    int nn_k_ = 15;
    int max_iter_ = 1;
    int max_iter_GN_ = 1;
    int fst_iter_GN_ = 1;
    float rot_eps_ = 2e-3;
    float trans_eps_ = 3e-4;
    float gicp_eps_ = 1e-5;
    float d_threshold_ = 1.0;
    float gnd_thre_ = 2.5;
    // To remove points with uniform covariance.
    PointT initPt;

    /* OMP_NUM is the processing number. */
    int OMP_NUM = 1;

    // parameters.
    bool keep_uniform_ = false;
    float voxel_size_x_ = 0.8;
    float voxel_size_y_ = 0.8;
    float voxel_size_z_ = 0.8;
    float normal_deg_ = 90;
    float curv_eps_ = 8;
    float gnd_weights_ = 1;
    float pln_weights_ = 1;
};

}
#endif