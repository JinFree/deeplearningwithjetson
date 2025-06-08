C++에서 TensorRT를 이용한 추론을 해보겠습니다.

Include 헤더는 아래와 같습니다.
```
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

// OpenCV 관련 처리를 위해 필요합니다.
#include <opencv2/opencv.hpp>

// TensorRT 엔진 추론과 cuda memory 관리를 위해 필요합니다.
#include <NvInfer.h>
#include <NvInferRuntime.h> 
#include <cuda_runtime_api.h> 
```

TRT 추론을 위한 클래스는 아래와 같습니다.
```
class TRTUltralyticsYOLO {
public:
    // 클래스 인스턴스 생성자입니다. 엔진 파일 경로를 받아서 엔진을 읽고 메모리를 할당합니다.
    TRTUltralyticsYOLO(const std::string& enginePath) {}

    // 클래스 인스턴스 소멸자입니다. 메모리 할당해제, 엔진 할당해제를 수행합니다.
    ~TRTUltralyticsYOLO() {}

    // 추론 함수입니다. OpenCV matrix를 받아서 추론한 결과를 outputHost로 반환합니다.
    void infer(const cv::Mat& resizedRGB, std::vector<float>& outputHost) {}

    // 엔진의 입력 텐서 크기를 확인하기 위한 함수입니다.
    int getInputH() const { return inputH; }
    int getInputW() const { return inputW; }

    // 엔진의 출력력
    const std::vector<int>& getOutputIndices() const { return outputIndices; }
    nvinfer1::ICudaEngine* getEngine() const { return engine; }
private:
    // 추론을 위해 클래스 내부적으로 들고 있는 변수입니다.
    nvinfer1::ICudaEngine* engine{};
    nvinfer1::IExecutionContext* context{};
    cudaStream_t stream{};
    int inputIndex{}, inputH{}, inputW{};
    std::vector<int> outputIndices;
    std::vector<void*> buffers;
};
```

클래스 인스턴스 생성자는 아래와 같습니다.
```
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
```

클래스 인스턴스 소멸자는 아래와 같습니다
```
for (auto& b : buffers) cudaFree(b);
context->destroy();
engine->destroy();
cudaStreamDestroy(stream);
```

추론 함수는 아래와 같습니다.
```
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
```
