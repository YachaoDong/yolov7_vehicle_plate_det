# bash /usr/local/ev_sdk/test.sh
echo "1. 测试单张图片";
/usr/local/ev_sdk/bin/test-ji-api -f 1 -i /project/train/src_repo/test_imgs/60.jpg -o /project/outputs/result60.jpg  # 941错误

# echo "1.1 测试单张图片";
# /usr/local/ev_sdk/bin/test-ji-api -f 1 -i /project/train/src_repo/test_imgs/941.jpg -o /project/outputs/result941.jpg  # 941错误

# echo "2. 测试多张图片";
# /usr/local/ev_sdk/bin/test-ji-api -f 1 -i /project/train/src_repo/test_imgs/941.jpg,/project/train/src_repo/test_imgs/60.jpg -o /project/outputs/result.jpg

echo "3. 测试单视频";
/usr/local/ev_sdk/bin/test-ji-api -f 1 -i /project/inputs/vechile1.mp4 -o /project/outputs/vehicle1_result.mp4

echo "3.1 测试单视频";
/usr/local/ev_sdk/bin/test-ji-api -f 1 -i /project/inputs/vechile2.mp4 -o /project/outputs/vehicle2_result.mp4

# echo "4. 测试图片文件夹";
# /usr/local/ev_sdk/bin/test-ji-api -f 1 -i /project/train/src_repo/test_imgs/    # /home/data/1111/


# echo "5. 测试在循环创建/调用/释放的过程中是否存在内存/显存的泄露";
# echo "5.2 循环调用100次";
# /project/ev_sdk/test/build/test-ji-api -f 3 -i /usr/local/ev_sdk/1.jpg -o /project/outputs/result5_2.jpg -r 100
# echo "5.1 无限循环调用";
# /project/ev_sdk/test/build/test-ji-api -f 3 -i /usr/local/ev_sdk/1.jpg -o /project/outputs/result5_1.jpg -r -1



# echo "6. 获取并打印算法的版本信息";
# /usr/local/ev_sdk/bin/test-ji-api -f 5


# /usr/local/ev_sdk/bin/test-ji-api -f 1 -i /usr/local/ev_sdk/1.jpg -o /project/outputs/result1.jpg


# gdb调试
# directory /usr/src/glibc/glibc-2.23/malloc/
# gdb --args /usr/local/ev_sdk/bin/test-ji-api -f 1 -i /usr/local/ev_sdk/1.jpg -o /project/outputs/result1.jpg