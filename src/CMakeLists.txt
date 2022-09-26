add_library(nanoflann SHARED
  nanoflann.cpp
)

target_link_libraries(nanoflann
  ${THIRD_PARTY_LIBS}
)

add_library(GMLO SHARED
  polar_region_gpf.cpp
  joint_matching.cpp
  gicp_mapper.cpp
  gicp_mp.cpp
)

target_link_libraries(GMLO
  ${THIRD_PARTY_LIBS}
  nanoflann
)
