file(GLOB SRC_FILES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_SOURCE_DIR}/include/*.hpp
)

add_library(SplitObject STATIC ${SRC_FILES})

target_include_directories(SplitObject PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(SplitObject PUBLIC ${CMAKE_SOURCE_DIR}/ThirdParty/eigen-3.3.9)

SET(OpenCV_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/opencv-4.5.3/build)

find_package(OpenCV 4.5.3 REQUIRED)
target_link_libraries(SplitObject PUBLIC ${OpenCV_LIBS})
target_link_libraries(SplitObject PUBLIC DetectEllipse)

target_compile_options(SplitObject PUBLIC -W3)

target_compile_features(SplitObject PUBLIC cxx_std_17)

target_compile_definitions(SplitObject PUBLIC _DEBUG _VIS)
