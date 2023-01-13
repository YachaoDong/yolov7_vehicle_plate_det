#echo "1. 测试单张图片";
#./test-ji-api -f 1 -i ../data/persons.jpg -o result.jpg

#echo "2. 测试多张图片";
#./test-ji-api -f 1 -i ../data/persons.jpg,../data/persons.jpg -o result.jpg

echo "3. 测试单视频";


#echo "4. 测试图片文件夹";
#/project/ev_sdk/test/build/test-ji-api -f 1 -i ../data/


echo "5. 测试在循环创建/调用/释放的过程中是否存在内存/显存的泄露";
echo "5.1 无限循环调用";

echo "5.2 循环调用100次";



echo "6. 获取并打印算法的版本信息";
