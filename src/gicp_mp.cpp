#include "gicp_mp.hpp"

namespace gicp_mapping{

template <typename PointT>
gicp_mp<PointT>::gicp_mp(int max_iter_in, int max_iter_GN_in):
                         source_pln_(new PointCloudType<PointT>),
                         target_pln_(new PointCloudType<PointT>){
  OMP_NUM = omp_get_max_threads();
  Eigen::initParallel();
  Eigen::setNbThreads(OMP_NUM);

  initPt.x = -50;
  initPt.y = -50;
  initPt.z = -50;

  // set max iters for debugging.
  max_iter_ = max_iter_in;
  max_iter_GN_ = max_iter_GN_in;
  fst_iter_GN_ = max_iter_GN_in;

  source_pln_tree_.reset(new KDTreeType<PointT>);
  target_pln_tree_.reset(new KDTreeType<PointT>);
  source_covs_ = new covVec;
  target_covs_ = new covVec;
}

template <typename PointT>
void gicp_mp<PointT>::setInputSource(PointCloudTypePtr<PointT> input_cloud){
  PointCloudTypePtr<PointT> source_voxel(new PointCloudType<PointT>);
  PointCloudTypePtr<PointT> source_in(new PointCloudType<PointT>);
  source_in = input_cloud;
  source_pln_tree_->setInputCloud(source_in);
  voxelFiltering(source_in, source_voxel);
  computeCovariances(source_voxel, source_pln_, source_pln_tree_, source_covs_);
}

template <typename PointT>
void gicp_mp<PointT>::setInputTarget(PointCloudTypePtr<PointT> input_cloud){
  PointCloudTypePtr<PointT> target_voxel(new PointCloudType<PointT>);
  PointCloudTypePtr<PointT> target_in(new PointCloudType<PointT>);
  target_in = input_cloud;

  target_pln_tree_->setInputCloud(target_in);
  voxelFiltering(target_in, target_voxel);
  computeCovariances(target_voxel, target_pln_, target_pln_tree_, target_covs_);
}

template <typename PointT>
bool gicp_mp<PointT>::align(Sophus::SE3f initial_guess){
  T0_ = initial_guess;

  bool converged = false;
  int iters = 0;
  int gn_iter = 0;
  PointCloudTypePtr<PointT> source_pln_transformed(new PointCloudType<PointT>);
      
  T1_ = T0_;
  while (iters < max_iter_ && !converged){
    std::cout << "Current GICP Iter: " << iters << std::endl;
    pcl::transformPointCloud(*source_pln_, *source_pln_transformed, T1_.matrix());
    gn_iter = 0;
    bool gn_converged = false;
    int iter_GN_num = max_iter_GN_;
    if (iters == 0)
      iter_GN_num = fst_iter_GN_;
    while (gn_iter < iter_GN_num && !gn_converged){
      pcl::transformPointCloud(*source_pln_, *source_pln_transformed, T1_.matrix());
      
      // Ax = b Least Square Problem for lie-se(3), R^6 vector.
      std::vector<Eigen::Matrix<float, 6, 6>> A_omp_list;
      std::vector<Eigen::Matrix<float, 6, 1>> b_omp_list;
      
      int pln_count = source_pln_->size();

      A_omp_list.resize(pln_count, Eigen::Matrix<float, 6, 6>::Zero());
      b_omp_list.resize(pln_count, Eigen::Matrix<float, 6, 1>::Zero());

      #pragma omp parallel for num_threads(OMP_NUM)
      for (int i = 0; i < pln_count; i++){
        pln2pln_step(
          i, source_pln_transformed, 
          target_pln_tree_, 
          target_covs_, 
          source_covs_, 
          A_omp_list, b_omp_list
        );
      }
      Eigen::Matrix<float, 6, 6> A = Eigen::Matrix<float, 6, 6>::Zero();
      Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
      assert(A_omp_list.size() == b_omp_list.size());
      #pragma omp declare reduction( + : Eigen::Matrix<float, 6, 6> : omp_out += omp_in ) \
              initializer( omp_priv = Eigen::Matrix<float, 6, 6>::Zero() )
      #pragma omp declare reduction( + : Eigen::Matrix<float, 6, 1> : omp_out += omp_in ) \
              initializer( omp_priv = Eigen::Matrix<float, 6, 1>::Zero() )
      #pragma omp parallel for num_threads(OMP_NUM) reduction(+ : A) reduction(+ : b) schedule(guided, 8)
      for (int i = 0; i < A_omp_list.size(); i++){
        A += A_omp_list[i];
        b += b_omp_list[i];
      }

      Eigen::LDLT<Eigen::Matrix<float, 6, 6>> solver(A);
      Eigen::Matrix<float, 6, 1> delta_xi = solver.solve(b);

      T1_ = Sophus::SE3f::exp(delta_xi) * T1_;

      gn_converged = is_converged(Sophus::SE3f::exp(delta_xi).matrix());
      gn_iter += 1;
    }

    std::cout << "Gauss-Newton Iteration: " << gn_iter << std::endl;
  
    // 3. GICP converged?
    // if converged: exit loop.
    // else: T0 = T1, once again.
    if (is_converged((T0_.inverse() * T1_).matrix())){
      converged = true;
    }
    else{
      T0_ = T1_;
    }
    iters += 1;
  }
  std::cout << "GICP Iterations: " << iters << std::endl;
  converged = true;
  if (iters == max_iter_ && gn_iter == max_iter_GN_)
    converged = false;
  return converged;
}

template <typename PointT>
void gicp_mp<PointT>::pln2pln_step(
      int index, PointCloudTypePtr<PointT> source_t,
      KDTreeTypePtr<PointT> target_cloud_tree,
      covVecPtr     target_covariances,
      covVecPtr     source_covariances,
      std::vector<Eigen::Matrix<float, 6, 6>> &A_vec,
      std::vector<Eigen::Matrix<float, 6, 1>> &b_vec){

  std::vector<int> nn_id;
  std::vector<float> nn_dist;
  target_cloud_tree->nearestKSearch(source_t->at(index), 1, nn_id, nn_dist);
  int source_idx = index ;
  int target_idx = nn_id[0];
  if (!corresRejector(nn_dist[0])){
    return;
  }
  // Current nonlinear least square block.
  Eigen::Matrix<float, 6, 6> A_i = Eigen::Matrix<float, 6, 6>::Zero();
  Eigen::Matrix<float, 6, 1> b_i = Eigen::Matrix<float, 6, 1>::Zero();

  // Derivative of SE(3) on Manifold, following Sophus definition.
  Eigen::Matrix<float, 3, 6> J = Eigen::Matrix<float, 3, 6>::Zero();
  J.block<3, 3>(0, 3) = Sophus::SO3f::hat(
                          Eigen::Vector3f(
                            source_t->at(source_idx).x,
                            source_t->at(source_idx).y, 
                            source_t->at(source_idx).z)
                        );
  J.block<3, 3>(0, 0) = -Eigen::Matrix3f::Identity();
  Eigen::Vector3f tbi(source_t->at(source_idx).x, 
                      source_t->at(source_idx).y, 
                      source_t->at(source_idx).z);
  
  // Constructing GICP costs.
  Eigen::Matrix3f R = T1_.matrix().block<3, 3>(0, 0);
  Eigen::Matrix3f Sigma = target_covariances->at(target_idx) + R * source_covariances->at(source_idx) * R.transpose();
  Eigen::Matrix3f L(Sigma.llt().matrixL());
  Eigen::Matrix3f invL = L.inverse();

  // Jacobian of df/d delta_xi of GICP cost.
  Eigen::Matrix<float, 3, 6> J1 = invL * J; // 3 x 6.
  PointT mi_pt = target_cloud_tree->getInputCloud()->at(target_idx);
  Eigen::Vector3f mi(mi_pt.x, mi_pt.y, mi_pt.z);
  Eigen::Vector3f f1 = invL * (mi - tbi);
  A_vec[index] += J1.transpose() * J1;
  b_vec[index] += -J1.transpose() * f1;
}

template <typename PointT>
void gicp_mp<PointT>::computeCovariances(
      PointCloudTypePtr<PointT> cloud_in,
      PointCloudTypePtr<PointT> cloud_out,
      KDTreeTypePtr<PointT>     cloud_tree,
      covVecPtr                 pt_covs){

  if (pt_covs->size() != 0)
    pt_covs->clear();
  
  
  cloud_out->points.resize(cloud_in->size(), initPt);
  pt_covs->resize(cloud_in->size(), Eigen::Matrix3f::Zero());
  Eigen::Vector3f eig_p2p(1, 1, 0.001);

  #pragma omp parallel for num_threads(OMP_NUM)
  for (int i = 0; i < cloud_in->size(); i++){
    std::vector<int> nn_idx(nn_k_);
    std::vector<float> nn_dists(nn_k_);
    int nn_num = cloud_tree->nearestKSearch(cloud_in->at(i), nn_k_, nn_idx, nn_dists);
    Eigen::MatrixXf D = Eigen::MatrixXf::Zero(3, nn_num);
    Eigen::Matrix3f covD = Eigen::Matrix3f::Zero();
    for (int j = 0; j < nn_num; j++){
      D(0, j) = cloud_tree->getInputCloud()->at(nn_idx[j]).x;
      D(1, j) = cloud_tree->getInputCloud()->at(nn_idx[j]).y;
      D(2, j) = cloud_tree->getInputCloud()->at(nn_idx[j]).z;
    }
    Eigen::MatrixXf D_centered = D.colwise() - D.rowwise().mean();
    covD = (D_centered * D_centered.transpose()) / float(nn_num);

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(covD, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f C_a_i = svd.matrixU() * eig_p2p.asDiagonal() * svd.matrixV().transpose();
    float curve = std::abs(svd.singularValues()(0) / svd.singularValues()(2));
    Eigen::Vector3f normal_vector = svd.matrixU().col(2);
    
    PointT currPt;
    if (keep_uniform_){
      if (curve <= curv_eps_){
        C_a_i = covD;
      }
      currPt.x = cloud_in->points[i].x;
      currPt.y = cloud_in->points[i].y;
      currPt.z = cloud_in->points[i].z;
    }
    else{
      if (curve <= curv_eps_){
        currPt.x = initPt.x;
        currPt.y = initPt.y;
        currPt.z = initPt.z;
      }
      else{
        currPt.x = cloud_in->points[i].x;
        currPt.y = cloud_in->points[i].y;
        currPt.z = cloud_in->points[i].z;
      }
    }
    pt_covs->at(i) = C_a_i;
    cloud_out->points[i].x = currPt.x;
    cloud_out->points[i].y = currPt.y;
    cloud_out->points[i].z = currPt.z;
    nn_idx.clear();
    nn_dists.clear();
  }
  if (!keep_uniform_){
    int j = 0;
    for (int i = 0; i < cloud_in->size(); i++){
      if (!pointValid(cloud_out->at(i)))
        continue;
      cloud_out->at(j) = cloud_out->at(i);
      pt_covs->at(j) = pt_covs->at(i);
      j++;
    }
    if (j != cloud_in->size()){
      cloud_out->resize(j);
      pt_covs->resize(j);
    }
  }
  cloud_tree->setInputCloud(cloud_out);
}

template <typename PointT>
bool gicp_mp<PointT>::is_converged(const Eigen::Matrix4f &delta){
  /*
    To determine Gauss-Newton optimization convergence by update se(3).
    From V-GICP codes.
  */
  Eigen::Matrix3f R = delta.block<3, 3>(0, 0) - Eigen::Matrix3f::Identity();
  Eigen::Vector3f t = delta.block<3, 1>(0, 3);

  Eigen::Matrix3f r_delta = 1.0 / rot_eps_ * R.array().abs();
  Eigen::Vector3f t_delta = 1.0 / trans_eps_ * t.array().abs();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

template<typename PointT>
void gicp_mp<PointT>::voxelFiltering(PointCloudTypePtr<PointT> raw_cloud,
                                         PointCloudTypePtr<PointT> cloud){
  pcl::ApproximateVoxelGrid<PointT> vg;
  vg.setLeafSize (voxel_size_x_, voxel_size_y_, voxel_size_z_);
  vg.setInputCloud(raw_cloud);
  vg.filter(*cloud);
}

template <typename PointT>
void gicp_mp<PointT>::setNextTarget(){
  source_pln_->swap(*target_pln_);
  target_pln_tree_->setInputCloud(target_pln_);
  source_covs_->swap(*target_covs_);
  source_clear();
}

template <typename PointT>
void gicp_mp<PointT>::target_clear(){
  target_pln_->clear();
  target_covs_->clear();
  target_pln_tree_.reset(new KDTreeType<PointT>);
}

template <typename PointT>
void gicp_mp<PointT>::source_clear(){
  source_pln_->clear();
  source_covs_->clear();
  source_pln_tree_.reset(new KDTreeType<PointT>);
}

template class gicp_mp<pcl::PointXYZ>;
template class gicp_mp<pcl::PointXYZRGB>;
template class gicp_mp<pcl::PointXYZI>;
}