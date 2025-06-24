#include <opencv_functions.hpp>

void copy_image(cv::Mat& dst_image, const cv::Mat& src_image)
{
    if (src_image.empty()) {
        cerr << "Error: Source image is empty!" << endl;
        return;
    }
    dst_image = src_image.clone();
}

void cut_rectangle_roi(cv::Mat& dst_image, const cv::Mat& src_image, const cv::Rect& roi)
{
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > src_image.cols || roi.y + roi.height > src_image.rows) {
        cerr << "Error: ROI is out of bounds!" << endl;
        return;
    }
    dst_image = src_image(roi).clone();
}

void cut_rectangle_roi(cv::Mat& dst_image, const cv::Mat& src_image, const cv::Point& pt1, const cv::Point& pt2)
{
    cv::Rect roi(pt1, pt2);
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > src_image.cols || roi.y + roi.height > src_image.rows) {
        cerr << "Error: ROI is out of bounds!" << endl;
        return;
    }
    cut_rectangle_roi(dst_image, src_image, roi);
}

void paste_rectangle_roi(cv::Mat& dst_image, const cv::Mat& roi_image, const cv::Rect& roi)
{
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > dst_image.cols || roi.y + roi.height > dst_image.rows) {
        cerr << "Error: ROI is out of bounds!" << endl;
        return;
    }
    roi_image.copyTo(dst_image(roi));
}

void paste_rectangle_roi(cv::Mat& dst_image, const cv::Mat& roi_image, const cv::Point& pt)
{
    cv::Rect roi(pt, cv::Point(pt.x + roi_image.cols, pt.y + roi_image.rows));
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > dst_image.cols || roi.y + roi.height > dst_image.rows) {
        cerr << "Error: ROI is out of bounds!" << endl;
        return;
    }
    paste_rectangle_roi(dst_image, roi_image, roi);
}

cv::Mat make_black_image(int width, int height, int type)
{
    return cv::Mat::zeros(height, width, type);
}

cv::Mat make_black_image(const cv::Size& size, int type)
{
    return cv::Mat::zeros(size, type);
}

cv::Mat make_black_image(const cv::Mat& src_image)
{
    return cv::Mat::zeros(src_image.size(), src_image.type());
}

cv::Mat make_white_image(int width, int height, int type)
{
    return cv::Mat::ones(height, width, type) * 255;
}

cv::Mat make_white_image(const cv::Size& size, int type)
{
    return cv::Mat::ones(size, type) * 255;
}

cv::Mat make_white_image(const cv::Mat& src_image)
{
    return cv::Mat::ones(src_image.size(), src_image.type()) * 255;
}