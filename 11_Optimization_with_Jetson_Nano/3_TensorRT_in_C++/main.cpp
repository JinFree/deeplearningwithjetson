#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvInferRuntime.h>    // for createInferRuntime
#include <cuda_runtime_api.h>  // for CUDA functions (cudaMalloc, cudaMemcpy)

// =============================
// 1) 환경 설정 및 클래스 이름 리스트
// =============================
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // INFO보다 심각한 로그만 출력
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

// COCO 클래스 이름 벡터 (80개 클래스)
static const std::vector<std::string> CLASS_NAMES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
};

// =============================
// 2) TensorRT 추론 클래스 정의
// =============================
// TensorRT 래퍼 클래스 (입력 크기 고정)
class TRTUltralyticsYOLO {
public:
    TRTUltralyticsYOLO(const std::string& enginePath) {
        // 1) 엔진 읽기 & 생성
        std::ifstream file(enginePath, std::ios::binary);
        if (!file) throw std::runtime_error("Engine 파일 열기 실패");
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> buf(size);
        file.read(buf.data(), size);
        auto runtime = nvinfer1::createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(buf.data(), size, nullptr);
        context = engine->createExecutionContext();
        cudaStreamCreate(&stream);

        // 2) 입력/출력 바인딩 인덱스 및 메모리 할당
        int nBindings = engine->getNbBindings();
        buffers.resize(nBindings);
        for (int i = 0; i < nBindings; ++i) {
            auto dims = engine->getBindingDimensions(i);
            size_t vol = 1;
            for (int d = 0; d < dims.nbDims; ++d) vol *= dims.d[d];
            bool isInput = engine->bindingIsInput(i);
            if (isInput) {
                inputIndex = i;
                inputH = dims.d[2];
                inputW = dims.d[3];
            } else {
                outputIndices.push_back(i);
            }
            size_t bytes = vol * sizeof(float);
            cudaMalloc(&buffers[i], bytes);
        }
    }

    ~TRTUltralyticsYOLO() {
        for (auto& b : buffers) cudaFree(b);
        context->destroy();
        engine->destroy();
        cudaStreamDestroy(stream);
    }

    // 추론: resizedRGB는 inputW×inputH의 RGB Mat
    void infer(const cv::Mat& resizedRGB, std::vector<float>& outputHost) {
        int volIn = 3 * inputH * inputW;
        // HWC→CHW, 정규화
        std::vector<float> hostData(volIn);
        for (int y = 0; y < inputH; ++y) {
            for (int x = 0; x < inputW; ++x) {
                auto px = resizedRGB.at<cv::Vec3b>(y, x);
                int idx = y * inputW + x;
                hostData[0 * inputH * inputW + idx] = px[0] / 255.f;
                hostData[1 * inputH * inputW + idx] = px[1] / 255.f;
                hostData[2 * inputH * inputW + idx] = px[2] / 255.f;
            }
        }
        cudaMemcpyAsync(buffers[inputIndex], hostData.data(), volIn * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        // 추론
        context->enqueueV2(buffers.data(), stream, nullptr);

        // 출력 크기 계산 (첫 번째 출력 바인딩 기준)
        int outIdx = outputIndices[0];
        auto outDims = context->getBindingDimensions(outIdx);
        size_t volOut = 1;
        for (int d = 0; d < outDims.nbDims; ++d) volOut *= outDims.d[d];

        outputHost.resize(volOut);
        cudaMemcpyAsync(outputHost.data(), buffers[outIdx], volOut * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    int getInputH() const { return inputH; }
    int getInputW() const { return inputW; }

private:
    nvinfer1::ICudaEngine* engine{};
    nvinfer1::IExecutionContext* context{};
    cudaStream_t stream{};
    int inputIndex{}, inputH{}, inputW{};
    std::vector<int> outputIndices;
    std::vector<void*> buffers;
};

// =============================
// 3) 보조 함수: NMS 및 시각화
// =============================

// IoU 계산 함수 (두 박스의 IoU를 구함, 박스는 [x1,y1,x2,y2] 좌표계)
float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float interArea;
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    interArea = w * h;
    float areaA = a.width * a.height;
    float areaB = b.width * b.height;
    float unionArea = areaA + areaB - interArea;
    if (unionArea <= 0.0f) return 0.0f;
    return interArea / unionArea;
}

// NMS 알고리즘 구현 (scores는 신뢰도, boxes는 cv::Rect2f 형 벡터)
std::vector<int> NMS(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores,
                     float scoreThreshold, float nmsThreshold) {
    std::vector<int> indices;
    // 처음에 scoreThreshold보다 큰 박스 인덱스만 고려
    for (int i = 0; i < (int)scores.size(); ++i) {
        if (scores[i] > scoreThreshold) {
            indices.push_back(i);
        }
    }
    // 점수 내림차순 정렬
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });
    std::vector<int> result;  // 최종 NMS를 통과한 인덱스
    std::vector<int> temp = indices;
    // NMS 루프
    while (!temp.empty()) {
        int current = temp.front();
        result.push_back(current);
        // current와 비교하여 IoU가 높은 인덱스 제거
        std::vector<int> remaining;
        for (size_t j = 1; j < temp.size(); ++j) {
            int idx = temp[j];
            float iou = IoU(boxes[current], boxes[idx]);
            if (iou <= nmsThreshold) {
                remaining.push_back(idx);
            }
        }
        temp = remaining;
    }
    return result;
}

// 결과 이미지에 박스와 라벨 그리기
void visualizeDetections(cv::Mat& image, const std::vector<int>& indices,
                         const std::vector<cv::Rect2f>& boxes,
                         const std::vector<int>& classIds,
                         const std::vector<float>& scores) {
    for (int idx : indices) {
        cv::Rect box_int = cv::Rect(cv::Point(std::round(boxes[idx].x), std::round(boxes[idx].y)),
                                    cv::Point(std::round(boxes[idx].x + boxes[idx].width),
                                              std::round(boxes[idx].y + boxes[idx].height)));
        // 경계 상자 그리기 (파란색)
        cv::rectangle(image, box_int, cv::Scalar(255, 0, 0), 2);
        // 라벨 텍스트 구성 ("class score")
        std::string label = CLASS_NAMES[classIds[idx]] + " " + cv::format("%.2f", scores[idx]);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &baseline);
        // 텍스트 배경 사각형 (가독성 높이기 위해 옵션으로 사용할 수 있음)
        cv::rectangle(image, cv::Point(box_int.x, box_int.y - textSize.height - 5),
                      cv::Point(box_int.x + textSize.width, box_int.y), cv::Scalar(255, 255, 255), cv::FILLED);
        // 텍스트 그리기 (박스 상단에)
        cv::putText(image, label, cv::Point(box_int.x, box_int.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1);
    }
}

// =============================
// 4) 메인 함수: 추론 & 시각화
// =============================
int main(int argc, char** argv) {
    // 명령행 인자 처리
    if (argc < 3) {
        std::cout << "사용법: " << argv[0] << " --engine <엔진파일> --image <이미지파일> [--conf <신뢰도임계값>] [--nms <NMS임계값>] [--display]\n";
        return 1;
    }
    std::string enginePath;
    std::string imagePath;
    float confThreshold = 0.25f;
    float nmsThreshold = 0.45f;
    bool displayResult = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--engine" || arg == "-e") && i + 1 < argc) {
            enginePath = argv[++i];
        } else if ((arg == "--image" || arg == "-i") && i + 1 < argc) {
            imagePath = argv[++i];
        } else if (arg == "--conf" && i + 1 < argc) {
            confThreshold = std::stof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            nmsThreshold = std::stof(argv[++i]);
        } else if (arg == "--display") {
            displayResult = true;
        }
    }
    if (enginePath.empty() || imagePath.empty()) {
        std::cerr << "엔진 경로와 이미지 경로는 필수입니다.\n";
        return 1;
    }

    // 이미지 로드
    cv::Mat img = cv::imread(imagePath);
    int origW = img.cols, origH = img.rows;

    TRTUltralyticsYOLO yolo(enginePath);
    int IW = yolo.getInputW(), IH = yolo.getInputH();

    cv::Mat resized, resizedRGB;
    cv::resize(img, resized, {IW, IH});
    cv::cvtColor(resized, resizedRGB, cv::COLOR_BGR2RGB);

    std::vector<float> outputData;
    yolo.infer(resizedRGB, outputData);

    // outputData → boxes, classIds, scores 파싱
    const int numClasses = static_cast<int>(CLASS_NAMES.size()); // 80
    const int C = 4 + numClasses;                               // 84
    // outputData 전체 요소 개수에서 C로 나누어 N을 구함
    int total = static_cast<int>(outputData.size());
    if (total % C != 0) {
        std::cerr << "출력 크기가 예상과 다릅니다: total=" << total << ", C=" << C << std::endl;
        return 1;
    }
    int N = total / C;

    std::vector<cv::Rect2f> boxes;
    std::vector<int> clsIds;
    std::vector<float> scores;
    for (int i = 0; i < N; ++i) {
        float cx = outputData[0 * N + i];
        float cy = outputData[1 * N + i];
        float w  = outputData[2 * N + i];
        float h  = outputData[3 * N + i];
        float maxScore = 0.0f;
        int maxClassId = -1;
        for (int c = 0; c < numClasses; ++c) {
            float score = outputData[(4 + c) * N + i];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = c;
            }
        }
        if (maxClassId >= 0 && maxScore > confThreshold) {
            float x1 = (cx - w/2) * origW/IW;
            float y1 = (cy - h/2) * origH/IH;
            float ww = w * origW/IW, hh = h * origH/IH;
            boxes.emplace_back(x1, y1, ww, hh);
            clsIds.push_back(maxClassId);
            scores.push_back(maxScore);
        }
    }

    auto keep = NMS(boxes, scores, confThreshold, nmsThreshold);
    visualizeDetections(img, keep, boxes, clsIds, scores);
    cv::imwrite("result.jpg", img);
    if (displayResult) {
        cv::imshow("Result", img);
        cv::waitKey(0);
    }
    return 0;
}
