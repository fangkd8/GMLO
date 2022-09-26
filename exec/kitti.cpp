#include "joint_matching.hpp"
#include "gicp_mapper.hpp"
#include "utils/libs_included.hpp"
#include "utils/utils.hpp"

#include <pcl/filters/passthrough.h>

using namespace std;
using namespace boost::filesystem;
using namespace std::chrono;

typedef vector<path> dir_vec;
dir_vec getFiles(string seq);
void addPointCloud(
  pcl::visualization::PCLVisualizer &viewer,
  pcl::PointCloud<pcl::PointXYZI>::Ptr input,
  int colornum, int cloudnum
);

int main(){
  string seq = "00";
  dir_vec v = getFiles(seq);

  pcl::PointCloud<pcl::PointXYZI>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZI>);

  float cloud_voxel = 0.4;
  float icp_voxel = 0.75;
  float gnd_dis = 1.5;

  gicp_mapping::joint_matching<pcl::PointXYZI> gicp_omp(3, 5);
  gicp_omp.setFstGNIter(10);
  gicp_omp.setVoxelSize(icp_voxel);
  gicp_omp.setGndThre(gnd_dis);
  gicp_omp.setCurvEps(2.5);
  gicp_mapping::gicp_mapper<pcl::PointXYZI> mapper;
  mapper.setMapSize(5);

  target_cloud = getKITTIData<pcl::PointXYZI>(true, false, v.begin()->string(), cloud_voxel);
  gicp_omp.setInputTarget(target_cloud);

  int count = 1;
  Sophus::SE3f init_guess;
  Sophus::SE3f cam_in_world;

  mapper.addKeyFrame(
    gicp_mapping::generateKeyFrameTgt<pcl::PointXYZI>(
      gicp_omp, cam_in_world
    )
  );

  std::ofstream results_file("../result/" + seq + "_pred.txt");
  std::ofstream time_file("../result/" + seq + "_time.txt");
  PoseToFile(cam_in_world.matrix().cast<float>(), results_file);
  for (dir_vec::const_iterator it (v.begin() + 1); it != v.end(); it ++){
    cout << "-------------- Reading " << count << "/" << v.size() << " point clouds.";
    cout << "--------------" << endl;
    source_cloud = getKITTIData<pcl::PointXYZI>(true, false, it->string(), cloud_voxel);

    auto start = high_resolution_clock::now();
    gicp_omp.setInputSource(source_cloud);
    gicp_omp.align(init_guess);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    cout << "Scan-to-Scan Registration Time: " << duration.count() << " (ms)." << endl;
    time_file << duration.count() << endl;
    if (std::abs(gicp_omp.getFinalTransformationSE3().matrix()(1, 3)) > 10){
      std::cout << "Invalid transformation at Frame " << count << "!" << std::endl;
      break;
    }
    Sophus::SE3f delta_s2s = gicp_omp.getFinalTransformationSE3();
    Eigen::Vector3f linear_velocity = delta_s2s.matrix().block<3, 1>(0, 3);
    init_guess = Sophus::SE3f(Eigen::Matrix3f::Identity(), linear_velocity);

    gicp_mapping::KeyFrame<pcl::PointXYZI> frame = 
      gicp_mapping::generateKeyFrameSrc<pcl::PointXYZI>(
        gicp_omp, cam_in_world
      );

    start = high_resolution_clock::now();
    Sophus::SE3f delta_s2m = mapper.mapAlign(delta_s2s, frame);
    end = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(end - start);
    cout << "Scan-to-Map Registration Time: " << duration.count() << " (ms)." << endl;
    if (false){
      if (count > 120){
        pcl::PointCloud<pcl::PointXYZI> map_cloud;
        mapper.getCurrMap(map_cloud);
        cout << "Map size: " << map_cloud.size() << endl; 
        pcl::visualization::PCLVisualizer viewer;
        addPointCloud(viewer, map_cloud.makeShared(), 0, 0);
        viewer.addCoordinateSystem(5.0);
        viewer.spin();
        break;
      }
    }
    if (delta_s2m.matrix().block<3, 1>(0, 3).norm() > 10){
      std::cout << "Invalid transformation at Frame " << count << "!" << std::endl;
      break;
    }
    cam_in_world *= delta_s2m;
    frame.Pose = cam_in_world;
    if (!delta_s2m.matrix().isApprox(delta_s2s.matrix())){
      mapper.setKeyPose(cam_in_world);
    }

    PoseToFile(cam_in_world.matrix(), results_file);
    mapper.addKeyFrame(frame);
    gicp_omp.setNextTarget();
    count += 1;
  }
}

dir_vec getFiles(string seq){
  string dir = "/media/fangkd/storage/dataset/sequences/";
  path p (dir + seq + "/velodyne/");
  directory_iterator end_itr;
  dir_vec v;
  copy(directory_iterator(p), directory_iterator(), back_inserter(v));
  sort(v.begin(), v.end());
  return v;
}

void addPointCloud(
  pcl::visualization::PCLVisualizer &viewer,
  pcl::PointCloud<pcl::PointXYZI>::Ptr input,
  int colornum, int cloudnum
)
{
  if (colornum == 0){
    viewer.addPointCloud<pcl::PointXYZI>(input, std::to_string(cloudnum));
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, std::to_string(cloudnum));
  }
  else if (colornum == 1){
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color(input, 255, 0, 0);
    viewer.addPointCloud<pcl::PointXYZI>(input, color, std::to_string(cloudnum));
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, std::to_string(cloudnum));
  }
  else if (colornum == 2){
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color(input, 0, 255, 0);
    viewer.addPointCloud<pcl::PointXYZI>(input, color, std::to_string(cloudnum));
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, std::to_string(cloudnum));
  }
  else if (colornum == 3){
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color(input, 0, 0, 255);
    viewer.addPointCloud<pcl::PointXYZI>(input, color, std::to_string(cloudnum));
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, std::to_string(cloudnum));
  }
  else if (colornum == 4){
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color(input, 255/2, 255/2, 255/2);
    viewer.addPointCloud<pcl::PointXYZI>(input, color, std::to_string(cloudnum));
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2.5, std::to_string(cloudnum));
  }
}

