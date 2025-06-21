#include <main.hpp>

using namespace std;
int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <image_path or video_path> <save_path>" << endl;
        return -1;
    }
    string input_path = argv[1];
    string save_path = (argc > 2) ? argv[2] : "";
    cout << "Input Path: " << input_path << endl;
    cout << "Save Path: " << save_path << endl;
    if (is_image(input_path)) {
        cout << "Processing image: " << input_path << endl;
        process_image(save_path, input_path);
    } 
    else if (is_video(input_path)) {
        cout << "Processing Video: " << input_path << endl;
        process_video(save_path, input_path);
    } 
    else {
        cout << "Error: Unsupported file type!" << endl;
    }
    return 0;
}