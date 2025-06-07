// main.cpp
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>

using namespace nvinfer1;

// 로그 콜백
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

// 유틸: engine 파일 로드
std::unique_ptr<ICudaEngine> loadEngine(const std::string& engineFile, Logger& logger) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open engine file");
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> buf(size);
    file.read(buf.data(), size);
    IRuntime* runtime = createInferRuntime(logger);
    auto engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(buf.data(), size, nullptr));
    runtime->destroy();
    return engine;
}

// 전처리: BGR->RGB, resize, normalize, CHW, batch=1
std::vector<float> preprocess(const cv::Mat& img, int inputW, int inputH) {
    cv::Mat rgb, resized;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(inputW, inputH));
    resized.convertTo(resized, CV_32F, 1.0/255);
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    std::vector<float> flat;
    for (auto& c : channels) {
        flat.insert(flat.end(), (float*)c.datastart, (float*)c.dataend);
    }
    return flat;
}

// IoU 계산
float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float interArea = (a & b).area();
    return interArea / (a.area() + b.area() - interArea);
}

// NMS
std::vector<int> NMS(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores,
                     float scoreThresh, float nmsThresh) {
    std::vector<int> idxs;
    for (int i = 0; i < scores.size(); i++)
        if (scores[i] > scoreThresh) idxs.push_back(i);
    std::sort(idxs.begin(), idxs.end(),
              [&](int l, int r){ return scores[l] > scores[r]; });
    std::vector<int> keep;
    while (!idxs.empty()) {
        int cur = idxs.front();
        keep.push_back(cur);
        std::vector<int> rest;
        for (int i = 1; i < idxs.size(); i++) {
            if (iou(boxes[cur], boxes[idxs[i]]) < nmsThresh)
                rest.push_back(idxs[i]);
        }
        idxs.swap(rest);
    }
    return keep;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " engine_path image_path input_size conf_thresh nms_thresh\n";
        return -1;
    }
    std::string enginePath = argv[1];
    std::string imagePath  = argv[2];
    int inputSize          = std::stoi(argv[3]);
    float confThresh       = std::stof(argv[4]);
    float nmsThresh        = std::stof(argv[5]);

    // 1) 이미지 로드
    cv::Mat orig = cv::imread(imagePath);
    if (orig.empty()) {
        std::cerr << "Failed to read image\n";
        return -1;
    }
    int origW = orig.cols, origH = orig.rows;

    // 2) 엔진 & 컨텍스트 생성
    Logger logger;
    auto engine = loadEngine(enginePath, logger);
    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());

    // 3) 버퍼 할당
    int nbBindings = engine->getNbBindings();
    std::vector<void*> devicePtrs(nbBindings);
    std::vector<void*> hostPtrs(nbBindings);
    std::vector<size_t> sizes(nbBindings);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < nbBindings; i++) {
        Dims dims = engine->getBindingDimensions(i);
        DataType dt = engine->getBindingDataType(i);
        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; d++) vol *= dims.d[d];
        size_t typeSize = sizeof(float);  // assume float32
        sizes[i] = vol * typeSize;
        cudaHostAlloc(&hostPtrs[i], sizes[i], cudaHostAllocDefault);
        cudaMalloc(&devicePtrs[i], sizes[i]);
    }

    // 4) 전처리 후 호스트→디바이스 복사
    auto inputData = preprocess(orig, inputSize, inputSize);
    std::memcpy(hostPtrs[0], inputData.data(), sizes[0]);
    cudaMemcpyAsync(devicePtrs[0], hostPtrs[0], sizes[0], cudaMemcpyHostToDevice, stream);

    // 5) 추론
    context->enqueueV2(devicePtrs.data(), stream, nullptr);

    // 6) 디바이스→호스트 복사
    for (int i = 1; i < nbBindings; i++) {
        cudaMemcpyAsync(hostPtrs[i], devicePtrs[i], sizes[i], cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    // 7) 결과 파싱 (예: 바인딩1 하나만 가정)
    Dims outDim = context->getBindingDimensions(1);  // (1,num_preds,5+cls)
    int numPred = outDim.d[1];
    int dimPred = outDim.d[2];
    float* outBuf = reinterpret_cast<float*>(hostPtrs[1]);

    std::vector<cv::Rect2f> boxes;
    std::vector<float>        scores;
    std::vector<int>          classIds;
    for (int i = 0; i < numPred; i++) {
        float* det = outBuf + i * dimPred;
        float conf = det[4];
        if (conf < confThresh) continue;
        // xywh->xyxy
        float cx=det[0], cy=det[1], w=det[2], h=det[3];
        float x1=cx-w/2, y1=cy-h/2;
        float x2=cx+w/2, y2=cy+h/2;
        boxes.emplace_back(x1, y1, w, h);
        // 클래스 & 스코어
        int cls = std::max_element(det+5, det+5+(dimPred-5)) - (det+5);
        float score = det[5+cls];
        scores.push_back(score);
        classIds.push_back(cls);
    }

    // 8) NMS
    auto keep = NMS(boxes, scores, confThresh, nmsThresh);

    // 9) 시각화
    static const std::vector<std::string> classNames = {
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush'
    };
    for (int idx : keep) {
        auto& r = boxes[idx];
        cv::Rect roi(
            int(r.x / inputSize * origW),
            int(r.y / inputSize * origH),
            int(r.width / inputSize * origW),
            int(r.height / inputSize * origH)
        );
        cv::rectangle(orig, roi, cv::Scalar(255,0,0), 2);
        std::string label = classNames[classIds[idx]] + " " + cv::format("%.2f", scores[idx]);
        cv::putText(orig, label, cv::Point(roi.x, roi.y-5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,0,0));
    }

    cv::imwrite("result.jpg", orig);
    std::cout << "Done. Result saved to result.jpg\n";

    // 10) 정리
    for (int i = 0; i < nbBindings; i++) {
        cudaFree(devicePtrs[i]);
        cudaFreeHost(hostPtrs[i]);
    }
    cudaStreamDestroy(stream);
    return 0;
}
