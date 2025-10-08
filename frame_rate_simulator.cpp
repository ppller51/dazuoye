#include <chrono>
#include <thread>
#include <iostream>

class FrameRateSimulator {
public:
    FrameRateSimulator(double target_fps) : target_fps_(target_fps) {
        frame_interval_ = std::chrono::milliseconds(static_cast<int>(1000 / target_fps_));
        last_frame_time_ = std::chrono::high_resolution_clock::now();  // 更高精度的时钟
        frames_processed_ = 0;
    }

    void simulateFrame() {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_diff = current_time - last_frame_time_;

        // 检查是否到达下一个帧采集的时间
        if (time_diff.count() >= frame_interval_.count() / 1000.0) {
            // 模拟图像捕获，减少延时
            last_frame_time_ = current_time;
            frames_processed_++;  // 增加已处理的帧数
        }
    }

    void printFrameRate() {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_diff = current_time - start_time_;
        double actual_fps = frames_processed_ / time_diff.count();

        std::cout << "Target FPS: " << target_fps_ << ", Actual FPS: " << actual_fps << std::endl;
    }

private:
    double target_fps_;
    int frames_processed_;
    std::chrono::high_resolution_clock::time_point last_frame_time_;
    std::chrono::high_resolution_clock::time_point start_time_ = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds frame_interval_;
};

int main() {
    double target_fps = 30.0;  // 设定目标帧率
    FrameRateSimulator simulator(target_fps);

    for (int i = 0; i < 1000; ++i) {  // 模拟采集 1000 帧
        simulator.simulateFrame();
        if (i % 100 == 0) {  // 每处理 100 帧，打印一次帧率
            simulator.printFrameRate();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));  // 控制循环速率，避免过度消耗 CPU
    }
    return 0;
}
