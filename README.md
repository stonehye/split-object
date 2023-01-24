# SplitObject

## Dependency

- OpenCV-4.5.3
- Eigen-3.3.9


## Generate and build projects

### Prerequisite

- Cmake 3.11 or higher version.
- Install dependency in ./ThirdParty.
    - [eigen-3.3.9](https://gitlab.com/libeigen/eigen/-/releases/3.3.9)
    - [opencv-4.5.3](https://github.com/opencv/opencv/releases/4.5.3)
- Set library path as own installed path in ./src/cmakeLists.txt.


### Windows, Visual Studio

1. Download Visual Studio 2019. (Project generating and building is tested under Visual Studio 2019).

2. Generate VS Solution with below commands.

   ```
   mkdir build
   cd build
   cmake ..  -G "Visual Studio 16 2019" -A x64
   ```

   

3. Open '.sln' file in ./build directory. After change build configuration, build and run 'samples' project which is example console program for processing SplitObject module with sample data.

