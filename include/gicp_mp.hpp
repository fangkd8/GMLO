#ifndef GICP_MP_HPP
#define GICP_MP_HPP
#undef NDEBUG

#include "nanoflann/nanoflann.hpp"
#include "utils/libs_included.hpp"
#include "utils/cloud_types.hpp"

namespace gicp_mapping{

template <typename PointT>
class gicp_mp {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    gicp_mp(int max_iter_in, int max_iter_GN_in);
    
    void setInputSource(PointCloudTypePtr<PointT> input_cloud);
    void setInputTarget(PointCloudTypePtr<PointT> input_cloud);
    
    bool align(Sophus::SE3f initial_guess);

    void setNextTarget();

    void setInputSource(
      PointCloudTypePtr<PointT> obj_cloud,
      covVec                    &obj_covs,
      PointCloudTypePtr<PointT> gnd_cloud
    ){
      pcl::copyPointCloud(*obj_cloud, *source_pln_);
      *source_covs_ = obj_covs;

      PointCloudTypePtr<PointT> gnd_voxel(new PointCloudType<PointT>);
      PointCloudTypePtr<PointT> source_gnd(new PointCloudType<PointT>);
      voxelFiltering(gnd_cloud, gnd_voxel);
      
      covVecPtr gnd_covs = new covVec;
      KDTreeTypePtr<PointT> gnd_tree;
      gnd_tree.reset(new KDTreeType<PointT>);
      gnd_tree->setInputCloud(gnd_cloud);
      computeCovariances(gnd_voxel, source_gnd, gnd_tree, gnd_covs);
      assert(source_gnd->size() == gnd_covs->size());

      *source_pln_ += *source_gnd;
      source_covs_->insert(source_covs_->end(), gnd_covs->begin(), gnd_covs->end());
      source_pln_tree_->setInputCloud(source_pln_);
    }
    
    void setInputTarget(
      PointCloudTypePtr<PointT> obj_cloud,
      covVec                    &obj_covs,
      PointCloudTypePtr<PointT> gnd_cloud
    ){
      pcl::copyPointCloud(*obj_cloud, *target_pln_);
      *target_covs_ = obj_covs;

      PointCloudTypePtr<PointT> gnd_voxel(new PointCloudType<PointT>);
      PointCloudTypePtr<PointT> target_gnd(new PointCloudType<PointT>);
      voxelFiltering(gnd_cloud, gnd_voxel);
      
      covVecPtr gnd_covs = new covVec;
      KDTreeTypePtr<PointT> gnd_tree;
      gnd_tree.reset(new KDTreeType<PointT>);
      gnd_tree->setInputCloud(gnd_cloud);
      computeCovariances(gnd_voxel, target_gnd, gnd_tree, gnd_covs);
      assert(target_gnd->size() == gnd_covs->size());

      *target_pln_ += *target_gnd;
      target_covs_->insert(target_covs_->end(), gnd_covs->begin(), gnd_covs->end());
      target_pln_tree_->setInputCloud(target_pln_);
    }

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

    KDTreeTypePtr<PointT> source_pln_tree_;
    KDTreeTypePtr<PointT> target_pln_tree_;

    covVecPtr source_covs_;
    covVecPtr target_covs_;


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