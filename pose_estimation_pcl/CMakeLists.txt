cmake_minimum_required(VERSION 3.8)
project(pose_estimation_pcl)

# Add optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(Threads REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

# Add link directories
link_directories(
  ${PCL_LIBRARY_DIRS}
)

# Add definitions
add_definitions(
  ${PCL_DEFINITIONS}
)


# Create the executable - include both source files directly
add_executable(pose_estimation_pcl 
  src/pose_estimation_pcl.cpp
  src/pcl_utils.cpp
)

# Add include directories
target_include_directories(pose_estimation_pcl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)


# Link libraries
ament_target_dependencies(pose_estimation_pcl
  rclcpp
  sensor_msgs
  geometry_msgs
  pcl_conversions
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
)
target_link_libraries(pose_estimation_pcl ${PCL_LIBRARIES})



# Install executables
install(TARGETS pose_estimation_pcl
  DESTINATION lib/${PROJECT_NAME}
)

# Install include directory
install(DIRECTORY include/
  DESTINATION include
)

# Install launch files if you have any
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
  OPTIONAL
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
