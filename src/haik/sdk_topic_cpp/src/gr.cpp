#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <mutex>
#include <sstream>

// 提前声明辅助函数
void draw_armor_base_info(cv::Mat& img, const class Armor& armor);
void write_debug_log(const std::string& log_content, rclcpp::Logger logger);

// 日志工具函数
void write_debug_log(const std::string& log_content, rclcpp::Logger logger) {
    time_t now = time(nullptr);
    tm* local_time = localtime(&now);
    char time_buf[32];
    strftime(time_buf, sizeof(time_buf), "[%Y-%m-%d %H:%M:%S]", local_time);

    std::string full_log = std::string(time_buf) + " " + log_content;
    std::cout << full_log << std::endl;  // 强制终端输出
    RCLCPP_INFO(logger, "%s", full_log.c_str());

    std::ofstream log_file("/home/pl/5/armor_detection_debug.log", std::ios::app);
    if (log_file.is_open()) {
        log_file << full_log << std::endl;
        log_file.close();
    } else {
        std::cerr << "【日志错误】无法打开日志文件，仅终端打印日志" << std::endl;
        RCLCPP_WARN(logger, "Failed to open log file, only print to terminal");
    }
}

// 装甲板配对候选结构体
struct ArmorCandidate {
    int idx1;          
    int idx2;          
    float area_diff;   
    float hsv_diff;    
    float similarity;  

    bool operator<(const ArmorCandidate& other) const {
        if (std::abs(hsv_diff - other.hsv_diff) < 3.0f) {
            if (std::abs(area_diff - other.area_diff) < 1e-6) {
                return similarity > other.similarity;
            }
            return area_diff < other.area_diff;
        }
        return hsv_diff < other.hsv_diff;
    }
};

// 灯条类
class Lightbar {
public:
    cv::RotatedRect rect;
    cv::Point2f delta;  
    int id;             
    bool is_valid;      
    bool has_qualified_pixels; 
    std::vector<std::string> reasons;  // 存储无效原因
    float length;       
    float area;         
    cv::Scalar hsv_mean; 
    int qualified_pixel_count; 

    Lightbar(cv::RotatedRect r, int identifier) : rect(r), id(identifier), is_valid(false), 
                                                has_qualified_pixels(false), hsv_mean(0,0,0),
                                                qualified_pixel_count(0) {
        length = std::max(rect.size.width, rect.size.height);
        area = rect.size.width * rect.size.height;
        
        float angle_rad = rect.angle * CV_PI / 180.0f;
        if (rect.size.width > rect.size.height) {
            delta.x = cos(angle_rad) * length / 2;
            delta.y = sin(angle_rad) * length / 2;
        } else {
            delta.x = cos(angle_rad + CV_PI/2) * length / 2;
            delta.y = sin(angle_rad + CV_PI/2) * length / 2;
        }
    }

    // 检查是否有符合条件的像素
    bool check_for_qualified_pixels(const cv::Mat& src_bgr, rclcpp::Logger logger) {
        cv::Rect roi = rect.boundingRect() & cv::Rect(0, 0, src_bgr.cols, src_bgr.rows);
        if (roi.width <= 0 || roi.height <= 0) {
            reasons.push_back("像素检查失败：ROI超出图像边界");
            write_debug_log("【灯条" + std::to_string(id) + "】" + reasons.back(), logger);
            return false;
        }

        cv::Mat hsv_roi;
        cv::cvtColor(src_bgr(roi), hsv_roi, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv_roi, hsv_channels);
        cv::Mat s_channel = hsv_channels[1];
        cv::Mat v_channel = hsv_channels[2];

        cv::Mat s_mask, v_mask, qualified_mask;
        cv::inRange(s_channel, 0, 50, s_mask);    
        cv::inRange(v_channel, 250, 255, v_mask); 
        cv::bitwise_and(s_mask, v_mask, qualified_mask);

        qualified_pixel_count = cv::countNonZero(qualified_mask);
        has_qualified_pixels = (qualified_pixel_count > 0);
        
        if (!has_qualified_pixels) {
            reasons.push_back("未找到符合条件的点（需同时满足S=0-50且V=250-255，当前符合数：0）");
            write_debug_log("【灯条" + std::to_string(id) + "】" + reasons.back(), logger);
            return false;
        }

        hsv_mean = cv::mean(hsv_roi);
        write_debug_log("【灯条" + std::to_string(id) + "】像素检查通过：符合条件的点数量=" + std::to_string(qualified_pixel_count) + 
                       "，HSV均值=(" + std::to_string(hsv_mean[0]) + "," + std::to_string(hsv_mean[1]) + "," + std::to_string(hsv_mean[2]) + ")", logger);
        return true;
    }

    // 计算HSV差异
    float calculate_hsv_diff(const Lightbar& other, rclcpp::Logger logger) const {
        if (!is_valid || !other.is_valid) {
            std::string log = "【灯条" + std::to_string(id) + "-" + std::to_string(other.id) + "】HSV差异计算失败：存在无效灯条（灯条" + std::to_string(is_valid ? id : other.id) + "无效）";
            write_debug_log(log, logger);
            return 100.0f;
        }
        
        float h_diff = std::abs(hsv_mean[0] - other.hsv_mean[0]);
        h_diff = std::min(h_diff, 180.0f - h_diff);
        float s_diff = std::abs(hsv_mean[1] - other.hsv_mean[1]);
        float v_diff = std::abs(hsv_mean[2] - other.hsv_mean[2]);
        
        return (h_diff * 0.4f) + (s_diff * 0.3f / 255.0f * 45.0f) + (v_diff * 0.3f / 255.0f * 45.0f);
    }

    // 检查灯条几何特征（重点强化无效原因输出）
    bool check_geometry(cv::Mat &img, const cv::Mat& src_bgr, rclcpp::Logger logger) {
        reasons.clear();  // 清空之前的原因
        float min_dim = std::min(rect.size.width, rect.size.height);
        float max_dim = std::max(rect.size.width, rect.size.height);
        float aspect_ratio = 0.0f;

        // 几何特征检查，记录所有无效原因
        if (area < 300) {
            reasons.push_back("面积<300像素（当前=" + std::to_string((int)area) + "）");
        }
        if (min_dim <= 1.5) {
            reasons.push_back("最小尺寸<1.5像素（当前=" + std::to_string(min_dim) + "）");
        }
        if (max_dim <= 5) {
            reasons.push_back("最大尺寸<5像素（当前=" + std::to_string(max_dim) + "）");
        }
        if (min_dim > 0) {
            aspect_ratio = max_dim / min_dim;
            if (aspect_ratio < 2.0f) {
                reasons.push_back("宽高比<2.0（当前=" + std::to_string(aspect_ratio) + "）");
            }
            if (aspect_ratio > 20.0f) {
                reasons.push_back("宽高比>20.0（当前=" + std::to_string(aspect_ratio) + "）");
            }
        } else {
            reasons.push_back("尺寸为0（无效轮廓）");
            aspect_ratio = std::numeric_limits<float>::infinity();
        }

        // 验证结果
        if (reasons.empty()) {
            // 几何检查通过后检查像素条件
            if (!check_for_qualified_pixels(src_bgr, logger)) {
                is_valid = false;
            } else {
                is_valid = true;
            }
        } else {
            is_valid = false;
        }

        // 绘制宽高比信息
        std::string ar_text = "AR: " + std::to_string((int)(aspect_ratio * 10) / 10.0f);
        cv::putText(img, ar_text, rect.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                   is_valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        if (is_valid) {
            std::string qualified_text = "S0-50&V250-255: " + std::to_string(qualified_pixel_count);
            cv::putText(img, qualified_text, rect.center + cv::Point2f(-45, 50), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 0), 2);
        }

        // 输出验证结果（强制终端显示）
        std::string log_prefix = "【灯条" + std::to_string(id) + "】";
        if (is_valid) {
            write_debug_log(log_prefix + "有效（面积=" + std::to_string((int)area) + ", 宽高比=" + std::to_string(aspect_ratio) + 
                           ", 符合条件的点数量=" + std::to_string(qualified_pixel_count) + "）", logger);
        } else {
            // 拼接所有无效原因并输出
            std::string reason_str = "";
            for (size_t i = 0; i < reasons.size(); i++) {
                reason_str += (i > 0 ? " | " : "") + reasons[i];
            }
            // 额外添加终端专用输出（红色文字突出显示）
            std::cerr << "\033[1;31m" << log_prefix << "无效：" << reason_str << "\033[0m" << std::endl;
            write_debug_log(log_prefix + "无效：" + reason_str + "（面积=" + std::to_string((int)area) + ", 宽高比=" + std::to_string(aspect_ratio) + "）", logger);
        }

        return is_valid;
    }

    // 计算两个灯条的相似度
    float calculate_similarity(const Lightbar& other, rclcpp::Logger logger) const {
        if (!is_valid || !other.is_valid) {
            std::string log = "【灯条" + std::to_string(id) + "-" + std::to_string(other.id) + "】相似度计算失败：存在无效灯条";
            write_debug_log(log, logger);
            return 0.0f;
        }

        float angle_diff = std::abs(rect.angle - other.rect.angle);
        float angle_sim = 1.0f - std::min(angle_diff, 180.0f - angle_diff) / 90.0f;
        angle_sim = std::max(0.0f, angle_sim);
        
        float area_ratio = std::min(area / other.area, other.area / area);
        float length_ratio = std::min(length / other.length, other.length / length);
        float hsv_diff = calculate_hsv_diff(other, logger);
        float hsv_sim = std::max(0.0f, 1.0f - hsv_diff / 45.0f);

        float total_sim = 0.4f * hsv_sim + 0.3f * angle_sim + 0.15f * length_ratio + 0.15f * area_ratio;
        std::string log = "【灯条" + std::to_string(id) + "-" + std::to_string(other.id) + "】相似度计算完成（HSV相似度=" + std::to_string(hsv_sim) + ", 角度相似度=" + std::to_string(angle_sim) + ", 综合相似度=" + std::to_string(total_sim) + "）";
        write_debug_log(log, logger);
        
        return total_sim;
    }

    // 辅助函数：拼接字符串
    std::string join(const std::vector<std::string>& v, const std::string& delim) const {
        std::string res;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) res += delim;
            res += v[i];
        }
        return res;
    }

    // 获取无效原因文本
    std::string get_reasons_text() const { return join(reasons, "\n"); }
};

// 装甲板类
class Armor {
public:
    Lightbar left, right;
    cv::Mat pattern;              
    cv::Mat normalized_pattern;   
    int id;                       
    bool is_valid;                
    float similarity;             
    float area_diff;              
    float hsv_diff;                
    std::vector<cv::Point2f> from_points;  
    std::vector<cv::Point2f> to_points;    
    int margin = 50;              
    float height_expansion_ratio = 0.7f;  
    std::vector<std::string> reasons;  

    Armor(Lightbar l, Lightbar r, int identifier) : left(l), right(r), id(identifier), is_valid(false) {}

    // 检查装甲板几何特征
    bool check_geometry(float similarity_threshold = 0.7f, float hsv_diff_threshold = 15.0f, rclcpp::Logger logger = rclcpp::get_logger("armor")) {
        reasons.clear();
        similarity = round(left.calculate_similarity(right, logger) * 100) / 100;
        area_diff = round(std::abs(left.area - right.area) * 100) / 100;
        hsv_diff = round(left.calculate_hsv_diff(right, logger) * 10) / 10;

        float distance = cv::norm(left.rect.center - right.rect.center);
        float angle_diff = std::abs(left.rect.angle - right.rect.angle);
        
        if (!left.is_valid) reasons.push_back("左侧灯条（ID=" + std::to_string(left.id) + "）无效");
        if (!right.is_valid) reasons.push_back("右侧灯条（ID=" + std::to_string(right.id) + "）无效");
        if (angle_diff >= 15.0f) reasons.push_back("灯条角度差>15°（当前=" + std::to_string(angle_diff) + "°）");
        if (hsv_diff > hsv_diff_threshold) reasons.push_back("HSV差异>" + std::to_string(hsv_diff_threshold) + "（当前=" + std::to_string(hsv_diff) + "）");
        if (similarity < similarity_threshold) reasons.push_back("综合相似度<" + std::to_string(similarity_threshold) + "（当前=" + std::to_string(similarity) + "）");
        
        is_valid = (reasons.empty()) && left.is_valid && right.is_valid;
        
        std::string log_prefix = "【装甲板" + std::to_string(id) + "（灯条" + std::to_string(left.id) + "-" + std::to_string(right.id) + "）】";
        if (is_valid) {
            write_debug_log(log_prefix + "有效（间距=" + std::to_string(distance) + "像素, HSV差异=" + std::to_string(hsv_diff) + ", 相似度=" + std::to_string(similarity) + "）", logger);
        } else {
            std::string reason_str = "";
            for (size_t i=0; i<reasons.size(); i++) reason_str += (i>0 ? " | " : "") + reasons[i];
            write_debug_log(log_prefix + "无效：" + reason_str, logger);
        }
        
        return is_valid;
    }

    // 获取有效性原因文本
    std::string get_validity_reason() const {
        if (is_valid) {
            return "有效（相似度=" + std::to_string(similarity) + " | 面积差异=" + std::to_string(area_diff) + " | HSV差异=" + std::to_string(hsv_diff) + "）";
        }
        
        std::string reason_str = "";
        for (size_t i=0; i<reasons.size(); i++) reason_str += (i>0 ? " | " : "") + reasons[i];
        return reason_str;
    }

    // 提取装甲板图案
    bool extract_pattern(const cv::Mat& src, cv::Mat& debug_img, rclcpp::Logger logger) {
        if (!is_valid) {
            std::string log = "【装甲板" + std::to_string(id) + "】图案提取失败：装甲板无效";
            write_debug_log(log, logger);
            return false;
        }
        
        cv::Point2f top_left = left.rect.center - left.delta * 0.9f;
        cv::Point2f bottom_left = left.rect.center + left.delta * 0.9f;
        cv::Point2f top_right = right.rect.center - right.delta * 0.9f;
        cv::Point2f bottom_right = right.rect.center + right.delta * 0.9f;
        
        float original_height = cv::norm(bottom_left - top_left);
        float expand_pixels = original_height * height_expansion_ratio;
        cv::Point2f vertical_dir = (bottom_left - top_left) / original_height;
        
        from_points = {
            top_left - vertical_dir * expand_pixels,    
            top_right - vertical_dir * expand_pixels,   
            bottom_right + vertical_dir * expand_pixels,
            bottom_left + vertical_dir * expand_pixels  
        };

        int h = 100 + 2 * static_cast<int>(100 * height_expansion_ratio);  
        int w = 100 + 2 * margin;  
        to_points = {
            cv::Point2f(0, 0), cv::Point2f(w, 0), 
            cv::Point2f(w, h), cv::Point2f(0, h)
        };

        cv::Mat perspective_mat = cv::getPerspectiveTransform(from_points, to_points);
        cv::warpPerspective(src, pattern, perspective_mat, cv::Size(w, h));

        if (margin >= 0 && (2 * margin) < w) {
            cv::Mat cropped = pattern(cv::Rect(margin, 0, w - 2 * margin, h)).clone();
            
            cv::GaussianBlur(cropped, cropped, cv::Size(5, 5), 0);  
            cv::cvtColor(cropped, normalized_pattern, cv::COLOR_BGR2GRAY);  
            cv::threshold(normalized_pattern, normalized_pattern, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  
            
            draw_debug_log(debug_img);
            write_debug_log("【装甲板" + std::to_string(id) + "】图案提取成功（预处理后尺寸=" + std::to_string(normalized_pattern.cols) + "x" + std::to_string(normalized_pattern.rows) + "）", logger);
            return true;
        } else {
            std::string log = "【装甲板" + std::to_string(id) + "】图案提取失败：边距无效（边距=" + std::to_string(margin) + ", 总宽度=" + std::to_string(w) + "）";
            write_debug_log(log, logger);
            return false;
        }
    }

private:
    // 绘制调试信息
    void draw_debug_log(cv::Mat& debug_img) {
        for (const auto& pt : from_points) {
            cv::circle(debug_img, pt, 5, cv::Scalar(0, 0, 255), -1);
        }
        for (int i = 0; i < 4; ++i) {
            cv::line(debug_img, from_points[i], from_points[(i+1)%4], cv::Scalar(0, 255, 0), 2);
        }
    }

    // 辅助函数：拼接字符串
    std::string join(const std::vector<std::string>& v, const std::string& delim) const {
        std::string res;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) res += delim;
            res += v[i];
        }
        return res;
    }
};

// ONNX推理函数
std::pair<std::string, float> onnx_infer(cv::dnn::Net& net, const cv::Mat& input_img, const std::vector<std::string>& class_labels, int armor_id, rclcpp::Logger logger) {
    if (input_img.size() != cv::Size(100, 240)) {
        std::string log = "【装甲板" + std::to_string(armor_id) + "】推理失败：输入尺寸异常（预期100x240，实际" + std::to_string(input_img.cols) + "x" + std::to_string(input_img.rows) + "）";
        write_debug_log(log, logger);
        return {"invalid_input_size", 0.0f};
    }

    cv::Mat input_3ch;
    cv::cvtColor(input_img, input_3ch, cv::COLOR_GRAY2BGR);  
    
    cv::Mat blob;
    try {
        cv::dnn::blobFromImage(
            input_3ch,          
            blob,               
            1.0 / 127.5,        
            cv::Size(100, 240), 
            cv::Scalar(127.5, 127.5, 127.5),  
            false,              
            false,              
            CV_32F              
        );
    } catch (const cv::Exception& e) {
        std::string log = "【装甲板" + std::to_string(armor_id) + "】推理失败：Blob生成异常（原因：" + std::string(e.what()) + "）";
        write_debug_log(log, logger);
        return {"blob_error", 0.0f};
    }

    net.setInput(blob, "input");  
    cv::Mat output;
    try {
        output = net.forward("output");  
    } catch (const cv::Exception& e) {
        std::string log = "【装甲板" + std::to_string(armor_id) + "】推理失败：模型前向传播异常（原因：" + std::string(e.what()) + "）";
        write_debug_log(log, logger);
        return {"infer_error", 0.0f};
    }

    cv::Mat softmax_output;
    if (output.rows == 1 && output.cols == 9) {  
        double min_val, max_val;
        cv::minMaxLoc(output, &min_val, &max_val);
        cv::exp(output - max_val, softmax_output);
        softmax_output /= cv::sum(softmax_output)[0];  

        double max_conf;
        cv::Point max_loc;
        cv::minMaxLoc(softmax_output, nullptr, &max_conf, nullptr, &max_loc);
        
        std::string log = "【装甲板" + std::to_string(armor_id) + "】推理成功（类别=" + class_labels[max_loc.x] + ", 置信度=" + std::to_string(max_conf*100) + "%）";
        write_debug_log(log, logger);
        
        return {class_labels[max_loc.x], static_cast<float>(max_conf)};
    } else {
        std::string log = "【装甲板" + std::to_string(armor_id) + "】推理失败：输出尺寸异常（预期1x9，实际" + std::to_string(output.rows) + "x" + std::to_string(output.cols) + "）";
        write_debug_log(log, logger);
        return {"invalid_output_size", 0.0f};
    }
}

// 装甲板检测节点类（集成PNP解算）
class ArmorDetectionNode : public rclcpp::Node {
public:
    ArmorDetectionNode() : Node("armor_detection_node") {
        // 1. 声明 ROS 2 参数
        declare_parameter<std::string>("image_topic", "image_raw");
        declare_parameter<std::string>("onnx_model_path", "/home/pl/5/shuzi_juanji/cnn_model_100x240_binary.onnx");
        declare_parameter<std::string>("camera_params_path", "/home/pl/ros2_ws1/src/haik/sdk_topic_cpp/src/can.yaml");
        declare_parameter<bool>("enable_debug_display", true);
        declare_parameter<float>("similarity_threshold", 0.7f);
        declare_parameter<float>("hsv_diff_threshold", 15.0f);

        // 2. 获取参数值
        get_parameter("image_topic", image_topic_);
        get_parameter("onnx_model_path", onnx_model_path_);
        get_parameter("camera_params_path", camera_params_path_);
        get_parameter("enable_debug_display", enable_debug_display_);
        get_parameter("similarity_threshold", similarity_threshold_);
        get_parameter("hsv_diff_threshold", hsv_diff_threshold_);

        // 3. 加载相机参数
        load_camera_params();

        // 4. 初始化 ONNX 模型
        init_onnx_model();

        // 5. 创建图像订阅者
        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            image_topic_,
            rclcpp::QoS(rclcpp::KeepLast(10)).best_effort(),
            std::bind(&ArmorDetectionNode::image_callback, this, std::placeholders::_1)
        );

        // 6. 初始化调试显示窗口
        if (enable_debug_display_) {
            cv::namedWindow("Armor Detection Result", cv::WINDOW_NORMAL);
            cv::resizeWindow("Armor Detection Result", 800, 600);
        }

        // 7. 打印启动日志
        write_debug_log("==================================== 装甲板检测节点启动 ====================================", get_logger());
        write_debug_log("订阅图像话题：" + image_topic_, get_logger());
        write_debug_log("ONNX模型路径：" + onnx_model_path_, get_logger());
        write_debug_log("相机参数路径：" + camera_params_path_, get_logger());
        write_debug_log("调试显示启用：" + std::string(enable_debug_display_ ? "是" : "否"), get_logger());
        write_debug_log("===========================================================================================", get_logger());
    }

    ~ArmorDetectionNode() {
        if (enable_debug_display_) {
            cv::destroyAllWindows();
        }
        write_debug_log("装甲板检测节点退出，资源已释放", get_logger());
    }

private:
    // 相机参数
    cv::Mat camera_matrix_;       // 相机内参矩阵 (3x3)
    cv::Mat distortion_coeffs_;   // 畸变系数 (1x5)

    // 矩阵转字符串工具函数
    std::string mat_to_string(const cv::Mat& mat) {
        std::stringstream ss;
        ss << mat;
        return ss.str();
    }

    // 加载相机参数（从YAML文件）
    void load_camera_params() {
        cv::FileStorage fs(camera_params_path_, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            RCLCPP_FATAL(get_logger(), "Failed to open camera params file: %s", camera_params_path_.c_str());
            rclcpp::shutdown();
            return;
        }

        // 读取内参和畸变系数
        fs["camera_matrix"] >> camera_matrix_;
        fs["distortion_coefficients"] >> distortion_coeffs_;
        fs.release();

        // 验证参数维度
        if (camera_matrix_.rows != 3 || camera_matrix_.cols != 3) {
            RCLCPP_FATAL(get_logger(), "Invalid camera matrix dimension! Expected 3x3, got %dx%d", 
                         camera_matrix_.rows, camera_matrix_.cols);
            rclcpp::shutdown();
            return;
        }
        if (distortion_coeffs_.rows != 1 || distortion_coeffs_.cols != 5) {
            RCLCPP_FATAL(get_logger(), "Invalid distortion coefficients dimension! Expected 1x5, got %dx%d", 
                         distortion_coeffs_.rows, distortion_coeffs_.cols);
            rclcpp::shutdown();
            return;
        }

        write_debug_log("相机内参加载成功：", get_logger());
        write_debug_log("camera_matrix:\n" + mat_to_string(camera_matrix_), get_logger());
        write_debug_log("distortion_coefficients:\n" + mat_to_string(distortion_coeffs_), get_logger());
    }

    // 初始化 ONNX 模型
    void init_onnx_model() {
        // 加载 ONNX 模型
        net_ = cv::dnn::readNetFromONNX(onnx_model_path_);
        if (net_.empty()) {
            RCLCPP_FATAL(get_logger(), "Failed to load ONNX model! Path: %s", onnx_model_path_.c_str());
            rclcpp::shutdown();
            return;
        }

        // 设置推理后端
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        // 初始化类别标签
        class_labels_ = {"1", "2", "3", "4", "5", "6outpost", "7guard", "8base", "9neg"};
        write_debug_log("ONNX模型加载成功，类别数：" + std::to_string(class_labels_.size()), get_logger());
    }

    // 图像回调函数
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(img_mutex_);

        try {
            // 1. 转换 ROS 图像消息为 OpenCV Mat
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            current_frame_ = cv_ptr->image.clone();
            frame_idx_++;

            // 2. 执行单帧装甲板检测
            process_single_frame();

            // 3. 显示检测结果
            if (enable_debug_display_ && !detection_result_img_.empty()) {
                cv::imshow("Armor Detection Result", detection_result_img_);
                cv::waitKey(1);
            }

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            return;
        } catch (cv::Exception& e) {
            RCLCPP_ERROR(get_logger(), "OpenCV exception: %s", e.what());
            return;
        }
    }

    // 单帧检测处理
    void process_single_frame() {
        if (current_frame_.empty()) return;

        // 初始化调试图像
        detection_result_img_ = current_frame_.clone();
        cv::Mat lightbar_result = current_frame_.clone();

        // 1. 灯条检测
        cv::Mat gray_img, binary_img;
        cv::cvtColor(current_frame_, gray_img, cv::COLOR_BGR2GRAY);
        cv::threshold(gray_img, binary_img, 200, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        write_debug_log("【帧" + std::to_string(frame_idx_) + "】轮廓检测完成：共找到" + std::to_string(contours.size()) + "个轮廓", get_logger());

        std::vector<Lightbar> lightbars;
        for (size_t i = 0; i < contours.size(); ++i) {
            double contour_area = cv::contourArea(contours[i]);
            if (contour_area < 300) {
                // 额外输出小面积轮廓被过滤的信息
                std::cout << "【帧" << frame_idx_ << "】轮廓" << i << "：面积<300像素（" << contour_area << "），过滤" << std::endl;
                continue;
            }

            cv::RotatedRect rect = cv::minAreaRect(contours[i]);
            Lightbar lb(rect, i);
            lb.check_geometry(lightbar_result, current_frame_, get_logger());  // 这里会输出无效原因
            if (lb.is_valid) {
                std::string h_text = "H: " + std::to_string((int)lb.hsv_mean[0]);
                cv::putText(lightbar_result, h_text, lb.rect.center + cv::Point2f(-20, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
            }
            lightbars.push_back(lb);

            // 绘制灯条边框、ID、面积
            cv::Point2f vertices[4];
            lb.rect.points(vertices);
            for (int j = 0; j < 4; ++j) {
                cv::line(lightbar_result, vertices[j], vertices[(j+1)%4], 
                       lb.is_valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
            }
            std::string id_text = "L" + std::to_string(lb.id);
            std::string area_text = "Area: " + std::to_string((int)lb.area);
            cv::putText(lightbar_result, id_text, lb.rect.center + cv::Point2f(-15, -20), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
            cv::putText(lightbar_result, area_text, lb.rect.center + cv::Point2f(-20, 10), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
        }

        // 统计有效灯条数量
        int valid_lightbar_count = 0;
        for (const auto& lb : lightbars) if (lb.is_valid) valid_lightbar_count++;
        write_debug_log("【帧" + std::to_string(frame_idx_) + "】灯条检测完成：共" + std::to_string(lightbars.size()) + "个灯条，有效灯条" + std::to_string(valid_lightbar_count) + "个", get_logger());

        // 2. 装甲板配对
        std::vector<Armor> armors;
        int armor_id = 0;
        std::vector<ArmorCandidate> candidates;

        if (valid_lightbar_count < 2) {
            write_debug_log("【帧" + std::to_string(frame_idx_) + "】装甲板配对跳过：有效灯条<2", get_logger());
        } else {
            // 生成配对候选
            for (size_t i = 0; i < lightbars.size(); ++i) {
                if (!lightbars[i].is_valid) continue;
                for (size_t j = i + 1; j < lightbars.size(); ++j) {
                    if (!lightbars[j].is_valid) continue;

                    float area_diff = std::abs(lightbars[i].area - lightbars[j].area);
                    float hsv_diff = lightbars[i].calculate_hsv_diff(lightbars[j], get_logger());
                    float sim = lightbars[i].calculate_similarity(lightbars[j], get_logger());
                    
                    if (hsv_diff <= hsv_diff_threshold_) {
                        candidates.push_back({(int)i, (int)j, area_diff, hsv_diff, sim});
                    }
                }
            }

            // 按优先级排序候选
            std::sort(candidates.begin(), candidates.end());
            write_debug_log("【帧" + std::to_string(frame_idx_) + "】装甲板候选：共" + std::to_string(candidates.size()) + "个", get_logger());

            // 筛选不重复的装甲板
            std::vector<bool> used(lightbars.size(), false);
            for (const auto& cand : candidates) {
                int i = cand.idx1;
                int j = cand.idx2;
                if (used[i] || used[j]) continue;

                Armor armor(lightbars[i], lightbars[j], armor_id++);
                if (armor.check_geometry(similarity_threshold_, hsv_diff_threshold_, get_logger())) {
                    armors.push_back(armor);
                    used[i] = true;
                    used[j] = true;
                }
            }
        }

        write_debug_log("【帧" + std::to_string(frame_idx_) + "】装甲板检测完成：有效装甲板" + std::to_string(armors.size()) + "个", get_logger());

        // 3. 装甲板图案提取、推理与PNP解算
        for (auto& armor : armors) {
            if (!armor.is_valid) continue;

            cv::Mat pattern_debug = current_frame_.clone();
            if (armor.extract_pattern(current_frame_, pattern_debug, get_logger())) {
                // 执行推理
                auto [pred_label, pred_conf] = onnx_infer(
                    net_, armor.normalized_pattern, class_labels_, armor.id, get_logger()
                );

                // 绘制推理结果
                draw_infer_result(detection_result_img_, armor, pred_label, pred_conf);

                // PNP解算核心逻辑
                std::vector<cv::Point3f> object_points = {
                    cv::Point3f(-67.5f, -28.0f, 0.0f),   // 左上
                    cv::Point3f(67.5f, -28.0f, 0.0f),    // 右上
                    cv::Point3f(67.5f, 28.0f, 0.0f),     // 右下
                    cv::Point3f(-67.5f, 28.0f, 0.0f)     // 左下
                };

                std::vector<cv::Point2f> image_points = armor.from_points;

                // 对图像点去畸变
                std::vector<cv::Point2f> undistorted_points;
                cv::undistortPoints(
                    image_points, 
                    undistorted_points, 
                    camera_matrix_, 
                    distortion_coeffs_, 
                    cv::Mat(), 
                    camera_matrix_
                );

                // 执行PNP解算
                cv::Mat rvec, tvec;
                cv::solvePnP(
                    object_points, 
                    undistorted_points, 
                    camera_matrix_, 
                    distortion_coeffs_, 
                    rvec, 
                    tvec, 
                    false, 
                    cv::SOLVEPNP_ITERATIVE
                );

                // 旋转向量转旋转矩阵
                cv::Mat R;
                cv::Rodrigues(rvec, R);

                // 打印完整位姿信息到日志
                write_debug_log("【装甲板" + std::to_string(armor.id) + "】PNP解算结果：", get_logger());
                write_debug_log("平移向量 t (mm):\n" + mat_to_string(tvec), get_logger());
                write_debug_log("旋转矩阵 R:\n" + mat_to_string(R), get_logger());

                // 绘制完整位姿信息（x,y,z和yaw,pitch,roll）
                draw_pose_info(detection_result_img_, armor, tvec, R);
            }

            // 绘制装甲板基础信息
            draw_armor_base_info(detection_result_img_, armor);
        }

        // 调试显示灯条检测结果
        if (enable_debug_display_ && !lightbar_result.empty()) {
            cv::imshow("Lightbar Detection", lightbar_result);
        }
    }

    // 绘制推理结果到图像
    void draw_infer_result(cv::Mat& img, const Armor& armor, const std::string& label, float conf) {
        cv::Point2f armor_center = (armor.left.rect.center + armor.right.rect.center) / 2.0f;
        std::string result_text = label + " (" + std::to_string((int)(conf * 100)) + "%)";
        std::string armor_id_text = "Armor " + std::to_string(armor.id);

        // 绘制文本背景
        cv::Size text_size = cv::getTextSize(result_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::Rect bg_rect(
            armor_center.x - text_size.width/2 - 5,
            armor_center.y - 30,
            text_size.width + 10,
            text_size.height + 5
        );
        cv::rectangle(img, bg_rect, cv::Scalar(0, 0, 0), -1);

        // 绘制文本
        cv::Scalar text_color = (label == "9neg") ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::putText(img, result_text, 
                   cv::Point(armor_center.x - text_size.width/2, armor_center.y - 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
        cv::putText(img, armor_id_text, 
                   cv::Point(armor_center.x - 20, armor_center.y - 50), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
    }

    // 绘制位姿信息到图像
    void draw_pose_info(cv::Mat& img, const Armor& armor, const cv::Mat& tvec, const cv::Mat& R) {
        // 1. 计算位姿数据（单位转换为米）
        float x = tvec.at<double>(0) / 1000.0f;  // X坐标
        float y = tvec.at<double>(1) / 1000.0f;  // Y坐标
        float z = tvec.at<double>(2) / 1000.0f;  // Z坐标
        
        // 2. 计算欧拉角（yaw, pitch, roll），单位转换为度
        float yaw = atan2(R.at<double>(0,1), R.at<double>(0,0)) * 180 / CV_PI;
        float pitch = asin(-R.at<double>(0,2)) * 180 / CV_PI;
        float roll = atan2(R.at<double>(1,2), R.at<double>(2,2)) * 180 / CV_PI;

        // 3. 格式化位姿文本
        std::string id_text = "Armor " + std::to_string(armor.id) + ":";
        std::string pos_text = cv::format("  X: %.2fm, Y: %.2fm, Z: %.2fm", x, y, z);
        std::string ang_text = cv::format("  Yaw: %d, Pitch: %d, Roll: %d", 
                                         (int)round(yaw), (int)round(pitch), (int)round(roll));
        std::vector<std::string> pose_lines = {id_text + pos_text, ang_text};

        // 4. 按ID从上到下排列
        int base_y = 40 + armor.id * 60;
        int line_height = 30;

        // 5. 绘制每行文本
        for (size_t i = 0; i < pose_lines.size(); ++i) {
            int y_pos = base_y + i * line_height;
            cv::Point text_pos(20, y_pos);

            // 绘制文本背景
            cv::Size text_size = cv::getTextSize(pose_lines[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, nullptr);
            cv::Rect bg_rect(
                text_pos.x - 5,
                text_pos.y - text_size.height - 5,
                text_size.width + 10,
                text_size.height + 10
            );
            cv::rectangle(img, bg_rect, cv::Scalar(0, 0, 0), -1);

            // 绘制文本
            cv::putText(
                img, pose_lines[i],
                text_pos,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(255, 0, 255),
                2
            );
        }
    }

    // ROS 2 成员变量
    std::string image_topic_;          
    std::string onnx_model_path_;      
    std::string camera_params_path_;   
    bool enable_debug_display_;        
    float similarity_threshold_;       
    float hsv_diff_threshold_;        

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

    // 检测相关变量
    cv::dnn::Net net_;                 
    std::vector<std::string> class_labels_;
    cv::Mat current_frame_;            
    cv::Mat detection_result_img_;     
    int frame_idx_ = 0;                
    std::mutex img_mutex_;             
};

// 辅助函数：绘制装甲板基础信息
void draw_armor_base_info(cv::Mat& img, const Armor& armor) {
    cv::Point2f vertices[4];
    armor.left.rect.points(vertices);
    for (int i = 0; i < 4; ++i) {
        cv::line(img, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 255, 0), 2);
    }
    armor.right.rect.points(vertices);
    for (int i = 0; i < 4; ++i) {
        cv::line(img, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 255, 0), 2);
    }

    cv::line(img, armor.left.rect.center, armor.right.rect.center, cv::Scalar(255, 0, 0), 2);
}

// 主函数
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArmorDetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
