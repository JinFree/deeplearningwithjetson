#ifndef OPENCV_FUNCTIONS_HPP
#define OPENCV_FUNCTIONS_HPP

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <set>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define PI 3.1415926

using namespace std;

void copy_image(cv::Mat& dst_image, const cv::Mat& src_image);
void cut_rectangle_roi(cv::Mat& dst_image, const cv::Mat& src_image, const cv::Rect& roi);
void cut_rectangle_roi(cv::Mat& dst_image, const cv::Mat& src_image, const cv::Point& pt1, const cv::Point& pt2);
void paste_rectangle_roi(cv::Mat& dst_image, const cv::Mat& roi_image, const cv::Rect& roi);
void paste_rectangle_roi(cv::Mat& dst_image, const cv::Mat& roi_image, const cv::Point& pt);
cv::Mat make_black_image(int width, int height, int type = CV_8UC3);
cv::Mat make_black_image(const cv::Size& size, int type = CV_8UC3);
cv::Mat make_black_image(const cv::Mat& src_image);
cv::Mat make_white_image(int width, int height, int type = CV_8UC3);
cv::Mat make_white_image(const cv::Size& size, int type = CV_8UC3);
cv::Mat make_white_image(const cv::Mat& src_image);

#endif // OPENCV_FUNCTIONS_HPP