# Shader variants are compiled at build time into C headers; the image size
# and penalties are specialization constants, the disparity range is not.
# The shaders are shared with the nx_stereo Elixir library.
function(vkstereo_shader_headers out_var generated_dir)
  set(shader_dir ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
  file(MAKE_DIRECTORY ${generated_dir})
  set(headers)
  foreach(disparities 64 128)
    math(EXPR tile "1024 / ${disparities}")
    set(common -V --target-env vulkan1.1 -S comp
        -DDISPARITIES=${disparities} -DBLOCK=8 -DFEATURE_TILE=${tile})
    foreach(variant
        "census_pair;census_pair;"
        "cost_first;cost_first;-DFUSE_VERTICAL=0"
        "cost_reverse;cost_scan;-DFUSE_VERTICAL=0"
        "cost_vertical;cost_scan;-DFUSE_VERTICAL=1"
        "consistency;consistency_projected;"
        "median;filter;")
      list(GET variant 0 name)
      list(GET variant 1 source)
      list(GET variant 2 flag)
      set(header ${generated_dir}/${name}_d${disparities}.h)
      add_custom_command(
        OUTPUT ${header}
        COMMAND ${GLSLANG_VALIDATOR} ${common} ${flag} --vn ${name}_d${disparities}
                -o ${header} ${shader_dir}/${source}.comp
        DEPENDS ${shader_dir}/${source}.comp ${shader_dir}/cost_dp.glsl
                ${shader_dir}/cost_select.glsl
        COMMENT "Compiling ${name} for ${disparities} disparities")
      list(APPEND headers ${header})
    endforeach()
  endforeach()
  set(${out_var} ${headers} PARENT_SCOPE)
endfunction()

if(COMMAND ocv_add_module)
  # Built by OpenCV as an extra module (OPENCV_EXTRA_MODULES_PATH), which also
  # generates the Python bindings (cv2.vkstereo)
  find_package(Vulkan QUIET)
  find_program(GLSLANG_VALIDATOR glslangValidator)
  if(NOT Vulkan_FOUND OR NOT GLSLANG_VALIDATOR)
    ocv_module_disable(vkstereo)
  endif()

  set(the_description "Semi-global stereo matching on Vulkan compute")
  ocv_add_module(vkstereo opencv_core opencv_imgproc opencv_calib3d WRAP python)
  vkstereo_shader_headers(VKSTEREO_SHADER_HEADERS ${CMAKE_CURRENT_BINARY_DIR}/generated)
  ocv_glob_module_sources(SOURCES ${VKSTEREO_SHADER_HEADERS})
  ocv_module_include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)
  ocv_create_module(Vulkan::Vulkan)
  # OpenCV builds its modules as C++11
  set_target_properties(${the_module} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  return()
endif()

set(VKSTEREO_GENERATED ${CMAKE_CURRENT_BINARY_DIR}/generated)
vkstereo_shader_headers(VKSTEREO_SHADER_HEADERS ${VKSTEREO_GENERATED})

add_library(opencv_vkstereo src/vkstereo.cpp ${VKSTEREO_SHADER_HEADERS})
target_include_directories(opencv_vkstereo
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE ${VKSTEREO_GENERATED})
target_link_libraries(opencv_vkstereo PUBLIC ${OpenCV_LIBS} PRIVATE Vulkan::Vulkan)
target_compile_options(opencv_vkstereo PRIVATE -Wall -Wextra)

install(TARGETS opencv_vkstereo LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

if(VKSTEREO_BUILD_SAMPLES)
  add_executable(vkstereo_bench samples/vkstereo_bench.cpp)
  target_link_libraries(vkstereo_bench opencv_vkstereo)
  if(VKSTEREO_WITH_IMGCODECS)
    target_compile_definitions(vkstereo_bench PRIVATE VKSTEREO_HAVE_IMGCODECS=1)
  endif()
endif()

if(VKSTEREO_BUILD_TESTS)
  find_package(GTest REQUIRED)
  add_executable(test_vkstereo test/test_vkstereo.cpp)
  target_link_libraries(test_vkstereo opencv_vkstereo GTest::gtest GTest::gtest_main)
  enable_testing()
  add_test(NAME vkstereo COMMAND test_vkstereo)
endif()
