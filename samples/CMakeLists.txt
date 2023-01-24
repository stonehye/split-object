file(GLOB SRC_FILES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
)

add_executable(samples ${SRC_FILES})
target_link_libraries(samples PUBLIC SplitObject)
