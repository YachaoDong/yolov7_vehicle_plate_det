# bash /usr/local/ev_sdk/re_build.sh

find / -name .ipynb_checkpoints* | xargs rm -rf

# 删除原来文件
cd /usr/local/ev_sdk/lib/
rm   libji.so   #libplateocr.so libyolodet.so 

# cd /usr/local/ev_sdk/bin/
# rm test-ji-api


# # build yolodet
# cd /usr/local/ev_sdk/src/yolo_src/
# rm -rf build/
# mkdir build
# cd build
# cmake ..
# make install -j16


# # build plateocr
# cd /usr/local/ev_sdk/src/paddle_src/
# rm -rf build/
# mkdir build
# cd build
# cmake ..
# make install -j16


# build ji
cd /usr/local/ev_sdk/
rm -rf build/
mkdir build
cd build
cmake ..
make install -j16
echo compile ji.so ok 

# build test-api
cd /usr/local/ev_sdk/test/
rm -rf build/ 3rd/ CMakeFiles/
rm CMakeCache.txt cmake_install.cmake Makefile
mkdir build
cd build
cmake ..
make install -j16

