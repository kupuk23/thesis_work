cmake_minimum_required(VERSION 3.8)
project(go_icp_vendor LANGUAGES CXX)

find_package(ament_cmake REQUIRED)

# ---------- 1. Build the original sources as a STATIC library ----------
add_library(go_icp STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/go_icp_src/jly_3ddt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/go_icp_src/jly_goicp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/go_icp_src/jly_main.cpp        # optional – remove if you don’t need the CLI
  ${CMAKE_CURRENT_SOURCE_DIR}/src/go_icp_src/matrix.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/go_icp_src/ConfigMap.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/go_icp_src/StringTokenizer.cpp
)
target_include_directories(go_icp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/go_icp_src>
  $<INSTALL_INTERFACE:include>)
target_compile_features(go_icp PUBLIC cxx_std_14)   # Humble default

# ---------- 2. Install ----------
install(TARGETS go_icp
  EXPORT export_go_icp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
install(DIRECTORY src/go_icp_src/
  DESTINATION include/go_icp
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# ---------- 3. Export to other ROS 2 packages ----------
ament_export_include_directories(include)
ament_export_targets(export_go_icp HAS_LIBRARY_TARGET)  # ← critical!
# ament_export_libraries(go_icp)

ament_package()
