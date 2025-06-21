#ifndef OPENCV_UTILS_HPP
#define OPENCV_UTILS_HPP
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <experimental/filesystem>
#include <set>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#define PI 3.1415926

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;


void process(Mat& dst_image, const Mat& src_image);
void process_image(const std::string& save_path, const std::string& input_path);
void process_video(const std::string& save_path, const std::string& input_path);

bool is_image(const std::string& path);
bool is_video(const std::string& path);


#endif // OPENCV_UTILS_HPP