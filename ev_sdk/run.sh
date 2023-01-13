# build yolodet
cd /usr/local/ev_sdk/src/yolo_src/
rm -rf build/
mkdir build
cd build
cmake ..
make install


# build plateocr
cd /usr/local/ev_sdk/src/paddle_src/
rm -rf build/
mkdir build
cd build
cmake ..
make install


# build ji
cd /usr/local/ev_sdk/
rm -rf build/
mkdir build
cd build
cmake ..
make install


# build test-api
cd /usr/local/ev_sdk/test/
rm -rf build/ 3rd/ CMakeFiles/
rm CMakeCache.txt cmake_install.cmake Makefile
mkdir build
cmake ..
make instal

