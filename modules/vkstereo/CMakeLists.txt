# Shader variants are compiled at build time into C headers; the image size
# and penalties are specialization constants, the disparity range is not.
set(VKSTEREO_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(VKSTEREO_GENERATED ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${VKSTEREO_GENERATED})

set(VKSTEREO_SHADER_HEADERS)
foreach(disparities 64 128)
  math(EXPR tile "1024 / ${disparities}")
  math(EXPR lanes "${disparities} / 8")
  set(common -V --target-env vulkan1.1 -S comp
      -DDISPARITIES=${disparities} -DBLOCK=8 -DCLASSIC_CENSUS=1 -DFEATURE_TILE=${tile}
      -DSELECT_LANES=${lanes} -DROW_SCATTER=1 -DMEDIAN_ONLY=1 -DTILE_ROWS=1)
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
    set(header ${VKSTEREO_GENERATED}/${name}_d${disparities}.h)
    add_custom_command(
      OUTPUT ${header}
      COMMAND ${GLSLANG_VALIDATOR} ${common} ${flag} --vn ${name}_d${disparities}
              -o ${header} ${VKSTEREO_SHADER_DIR}/${source}.comp
      DEPENDS ${VKSTEREO_SHADER_DIR}/${source}.comp ${VKSTEREO_SHADER_DIR}/cost_dp.glsl
              ${VKSTEREO_SHADER_DIR}/cost_select.glsl
      COMMENT "Compiling ${name} for ${disparities} disparities")
    list(APPEND VKSTEREO_SHADER_HEADERS ${header})
  endforeach()
endforeach()

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
