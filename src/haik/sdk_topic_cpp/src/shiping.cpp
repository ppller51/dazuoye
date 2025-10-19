#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <iomanip>

// 提前声明类与函数
class Lightbar;
class Armor;
void draw_armor_base_info(cv::Mat& img, const Armor& armor);
std::string point_to_string(const cv::Point& pt);
std::vector<cv::Rect> get_first_code_valid_rects(const cv::Mat& frame, bool& is_single_rect_matched);
cv::Rect find_best_match_rect(const cv::Rect& target_rect, const std::vector<cv::Rect>& candidate_rects, const cv::Mat& frame);
float calculate_rect_similarity(const cv::Rect& r1, const cv::Rect& r2, const cv::Mat& frame);
Armor* rematch_by_history_lbs(const std::vector<Lightbar>& lightbars, int& armor_id, float sim_threshold, float hsv_diff_threshold);
std::pair<std::string, float> onnx_infer(cv::dnn::dnn4_v20211004::Net& net, const cv::Mat& input_img, const std::vector<std::string>& class_labels, int armor_id);

// 全局变量：历史灯条存储
cv::Point prev_armor_center = cv::Point(-1, -1); 
cv::Point prev_prev_armor_center = cv::Point(-1, -1);
float prev_armor_area = 0.0f;                     
float prev_prev_armor_area = 0.0f;                 
int prev_left_lb_id = -1;                         
int prev_right_lb_id = -1;                        
int prev_prev_left_lb_id = -1;                    
int prev_prev_right_lb_id = -1;                   
const float max_center_dist = 100.0f;             
const float max_area_ratio = 2.0f;                
const float MAX_LIGHTBAR_DISTANCE = 400.0f;       
const float min_area_first_code = 680.0f;         
const float MIN_CANDIDATE_AREA = 300.0f;          

// 控制目标矩形限制的标志位
bool relax_target_rect_check = false;  
int consecutive_failure_count = 0;     
const int MAX_FAILURES_BEFORE_RELAX = 0;  

// 坐标转字符串
std::string point_to_string(const cv::Point& pt) {
    return "(" + std::to_string(pt.x) + "," + std::to_string(pt.y) + ")";
}

// 计算两个矩形的相似度
float calculate_rect_similarity(const cv::Rect& r1, const cv::Rect& r2, const cv::Mat& frame) {
    float area1 = r1.width * r1.height;
    float area2 = r2.width * r2.height;
    float area_sim = std::min(area1, area2) / std::max(area1, area2);

    float ar1 = (float)r1.width / r1.height;
    float ar2 = (float)r2.width / r2.height;
    float ar_sim = 1.0f - std::abs(ar1 - ar2) / std::max(ar1, ar2);
    ar_sim = std::max(0.0f, ar_sim);

    cv::Rect safe_r1 = r1 & cv::Rect(0, 0, frame.cols, frame.rows);
    cv::Rect safe_r2 = r2 & cv::Rect(0, 0, frame.cols, frame.rows);
    cv::Mat hsv1, hsv2;
    cv::cvtColor(frame(safe_r1), hsv1, cv::COLOR_BGR2HSV);
    cv::cvtColor(frame(safe_r2), hsv2, cv::COLOR_BGR2HSV);
    cv::Scalar mean1 = cv::mean(hsv1);
    cv::Scalar mean2 = cv::mean(hsv2);

    float h_diff = std::abs(mean1[0] - mean2[0]);
    h_diff = std::min(h_diff, 180.0f - h_diff) / 90.0f;
    float s_diff = std::abs(mean1[1] - mean2[1]) / 255.0f;
    float v_diff = std::abs(mean1[2] - mean2[2]) / 255.0f;
    float hsv_sim = 1.0f - (h_diff * 0.4f + s_diff * 0.3f + v_diff * 0.3f);
    hsv_sim = std::max(0.0f, hsv_sim);

    return area_sim * 0.3f + ar_sim * 0.3f + hsv_sim * 0.4f;
}

// 为单个目标矩形寻找相似度最高的候选矩形
cv::Rect find_best_match_rect(const cv::Rect& target_rect, const std::vector<cv::Rect>& candidate_rects, const cv::Mat& frame) {
    if (candidate_rects.empty()) {
        return cv::Rect();
    }

    float max_sim = -1.0f;
    cv::Rect best_rect;
    for (const auto& cand : candidate_rects) {
        float cand_area = cand.width * cand.height;
        if (cand_area < MIN_CANDIDATE_AREA) {
            continue;
        }

        float sim = calculate_rect_similarity(target_rect, cand, frame);
        if (sim > max_sim) {
            max_sim = sim;
            best_rect = cand;
        }
    }

    if (max_sim < 0.3f) {
        return cv::Rect();
    }

    return best_rect;
}

// 第一版代码核心逻辑
std::vector<cv::Rect> get_first_code_valid_rects(const cv::Mat& frame, bool& is_single_rect_matched) {
    std::vector<cv::Rect> valid_rectangles;
    cv::Mat hsv_img, gray, binary, final_mask;
    std::vector<cv::Mat> hsv_ch;
    cv::Mat mask_S, mask_V, kernel;
    is_single_rect_matched = false;

    cv::cvtColor(frame, hsv_img, cv::COLOR_BGR2HSV);
    cv::split(hsv_img, hsv_ch);
    if (hsv_ch.size() < 3) return valid_rectangles;

    cv::Mat S = hsv_ch[1], V = hsv_ch[2];
    cv::inRange(S, 0, 50, mask_S);
    cv::inRange(V, 250, 255, mask_V);
    final_mask = mask_S & mask_V;

    cv::GaussianBlur(final_mask, gray, cv::Size(5, 5), 0);
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv::THRESH_BINARY_INV, 15, 3);

    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (auto& cnt : contours) {
        cv::Rect r = cv::boundingRect(cnt);
        double area = cv::contourArea(cnt);
        float ar = (float)r.width / r.height;
        if (area >= min_area_first_code && area <= 6000 && ar > 0.2 && ar < 1.5) {
            valid_rectangles.push_back(r);
        }
    }

    if (valid_rectangles.size() == 1) {
        std::vector<cv::Rect> all_candidate_rects;
        std::vector<std::vector<cv::Point>> all_contours;
        cv::Mat gray_all, binary_all;
        cv::cvtColor(frame, gray_all, cv::COLOR_BGR2GRAY);
        cv::threshold(gray_all, binary_all, 120, 255, cv::THRESH_BINARY);
        cv::findContours(binary_all, all_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (auto& cnt : all_contours) {
            cv::Rect r = cv::boundingRect(cnt);
            float area = r.width * r.height;
            if (area < MIN_CANDIDATE_AREA) continue;
            float overlap = (r & valid_rectangles[0]).area() / (float)(r | valid_rectangles[0]).area();
            if (overlap < 0.5f) {
                all_candidate_rects.push_back(r);
            }
        }

        cv::Rect best_match = find_best_match_rect(valid_rectangles[0], all_candidate_rects, frame);
        if (best_match.width > 0 && best_match.height > 0) {
            valid_rectangles.push_back(best_match);
            is_single_rect_matched = true;
        }
    }

    return valid_rectangles;
}

// 装甲板配对候选结构体
struct ArmorCandidate {
    int idx1;          
    int idx2;          
    float area_diff;   
    float hsv_diff;    
    float similarity;  
    float area_ratio;  
    float length_ratio;

    bool operator<(const ArmorCandidate& other) const {
        return similarity > other.similarity;
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
    bool is_in_target_rect;
    std::vector<std::string> reasons;  
    float length;       
    float area;         
    cv::Scalar hsv_mean; 
    int qualified_pixel_count; 

    Lightbar() : rect(cv::RotatedRect()), id(-1), is_valid(false), 
                has_qualified_pixels(false), is_in_target_rect(false),
                qualified_pixel_count(0), hsv_mean(0,0,0), length(0), area(0) {}

    Lightbar(cv::RotatedRect r, int identifier) : rect(r), id(identifier), is_valid(false), 
                                                has_qualified_pixels(false), is_in_target_rect(false),
                                                qualified_pixel_count(0), hsv_mean(0,0,0) {
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

    bool check_in_target_rects(const std::vector<cv::Rect>& target_rects) {
        cv::Point lb_center = rect.center;
        for (const auto& rect : target_rects) {
            if (lb_center.inside(rect)) {
                is_in_target_rect = true;
                return true;
            }
        }
        is_in_target_rect = false;
        reasons.push_back("灯条不在目标矩形内");
        return false;
    }

    bool check_for_qualified_pixels(const cv::Mat& src_bgr) {
        cv::Rect roi = rect.boundingRect() & cv::Rect(0, 0, src_bgr.cols, src_bgr.rows);
        if (roi.width <= 0 || roi.height <= 0) {
            reasons.push_back("ROI超出图像边界");
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
            reasons.push_back("无S0-50&V250-255像素");
            return false;
        }

        hsv_mean = cv::mean(hsv_roi);
        return true;
    }

    float calculate_hsv_diff(const Lightbar& other) const {
        float h_diff = std::abs(hsv_mean[0] - other.hsv_mean[0]);
        h_diff = std::min(h_diff, 180.0f - h_diff);
        float s_diff = std::abs(hsv_mean[1] - other.hsv_mean[1]);
        float v_diff = std::abs(hsv_mean[2] - other.hsv_mean[2]);
        
        return (h_diff * 0.4f) + (s_diff * 0.3f / 255.0f * 45.0f) + (v_diff * 0.3f / 255.0f * 45.0f);
    }

    bool check_geometry(cv::Mat &img, const cv::Mat& src_bgr, const std::vector<cv::Rect>& target_rects, bool check_target_rect = true) {
        reasons.clear();
        float min_dim = std::min(rect.size.width, rect.size.height);
        float max_dim = std::max(rect.size.width, rect.size.height);
        float aspect_ratio = 0.0f;

        if (check_target_rect && !check_in_target_rects(target_rects)) {
            is_valid = false;
        } else {
            if (min_dim <= 1.5) reasons.push_back("最小尺寸<1.5像素");
            if (max_dim <= 5) reasons.push_back("最大尺寸<5像素");
            if (min_dim > 0) {
                aspect_ratio = max_dim / min_dim;
                if (aspect_ratio < 2.0f) reasons.push_back("宽高比<2.0");
                if (aspect_ratio > 20.0f) reasons.push_back("宽高比>20.0");
            } else {
                reasons.push_back("尺寸为0");
                aspect_ratio = std::numeric_limits<float>::infinity();
            }

            if (reasons.empty()) {
                is_valid = check_for_qualified_pixels(src_bgr);
            } else {
                is_valid = false;
            }
        }

        std::string ar_text = "AR: " + std::to_string((int)(aspect_ratio * 10) / 10.0f);
        cv::Scalar color = (is_in_target_rect ? (is_valid ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255)) : cv::Scalar(128,128,128));
        cv::putText(img, ar_text, rect.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

        return is_valid;
    }

    struct SimilarityData {
        float hsv_sim;      
        float area_ratio;   
        float length_ratio; 
        float angle_sim;    
        float ar_ratio;     
        float total;        
    };

    SimilarityData calculate_similarity_details(const Lightbar& other) const {
        SimilarityData data;
        
        data.area_ratio = std::min(area / other.area, other.area / area);
        data.length_ratio = std::min(length / other.length, other.length / length);
        
        float hsv_diff = calculate_hsv_diff(other);
        data.hsv_sim = std::max(0.0f, 1.0f - hsv_diff / 45.0f);
        
        float angle_diff = std::abs(rect.angle - other.rect.angle);
        angle_diff = std::min(angle_diff, 180.0f - angle_diff);
        data.angle_sim = std::max(0.0f, 1.0f - angle_diff / 90.0f);
        
        float ar1 = std::max(rect.size.width, rect.size.height) / std::min(rect.size.width, rect.size.height);
        float ar2 = std::max(other.rect.size.width, other.rect.size.height) / std::min(other.rect.size.width, other.rect.size.height);
        float ar_diff = std::abs(ar1 - ar2);
        data.ar_ratio = std::max(0.0f, 1.0f - ar_diff / 10.0f);

        const float W_HSV = 0.05f;      
        const float W_LENGTH = 0.5f;    
        const float W_AREA = 0.1f;     
        const float W_ANGLE = 0.25f;    
        const float W_AR = 0.1f;       
        data.total = W_HSV * data.hsv_sim + 
                     W_LENGTH * data.length_ratio + 
                     W_AREA * data.area_ratio + 
                     W_ANGLE * data.angle_sim + 
                     W_AR * data.ar_ratio;
        return data;
    }

    float calculate_similarity(const Lightbar& other) const {
        return calculate_similarity_details(other).total;
    }

    std::string join(const std::vector<std::string>& v, const std::string& delim) const {
        std::string res;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) res += delim;
            res += v[i];
        }
        return res;
    }

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
    float total_area;             
    float lightbar_distance;      
    std::vector<cv::Point2f> from_points;  
    std::vector<cv::Point2f> to_points;    
    int margin = 50;              
    float height_expansion_ratio = 0.7f;  
    std::vector<std::string> reasons;  

    Armor(Lightbar l, Lightbar r, int identifier) : left(l), right(r), id(identifier), is_valid(false) {
        similarity = round(left.calculate_similarity(right) * 100) / 100;
        area_diff = round(std::abs(l.area - r.area) * 100) / 100;
        hsv_diff = round(left.calculate_hsv_diff(r) * 10) / 10;
        total_area = l.area + r.area;
        lightbar_distance = cv::norm(left.rect.center - right.rect.center);
    }

    bool check_geometry(float similarity_threshold = 0.5f, float hsv_diff_threshold = 15.0f) {
        reasons.clear();
        
        if (lightbar_distance > MAX_LIGHTBAR_DISTANCE) {
            reasons.push_back("灯条间距过大");
        }
        if (!left.is_valid) reasons.push_back("左灯条无效");
        if (!right.is_valid) reasons.push_back("右灯条无效");
        if (lightbar_distance <= 50) reasons.push_back("灯条间距过小");
        if (hsv_diff > hsv_diff_threshold) reasons.push_back("HSV差异过大");
        if (similarity < similarity_threshold) reasons.push_back("相似度不足");
        
        is_valid = (lightbar_distance <= MAX_LIGHTBAR_DISTANCE);
        return is_valid;
    }

    std::string get_validity_reason() const {
        if (is_valid) {
            std::string warn_str = reasons.empty() ? "" : "（警告：" + join(reasons, " | ") + "）";
            return "有效" + warn_str;
        }
        return join(reasons, " | ");
    }

    bool extract_pattern(const cv::Mat& src, cv::Mat& debug_img) {
        if (!is_valid) {
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
            
            for (const auto& pt : from_points) cv::circle(debug_img, pt, 5, cv::Scalar(0,0,255), -1);
            for (int i = 0; i < 4; ++i) cv::line(debug_img, from_points[i], from_points[(i+1)%4], cv::Scalar(0,255,0), 2);
            return true;
        }
        return false;
    }

    std::string join(const std::vector<std::string>& v, const std::string& delim) const {
        std::string res;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) res += delim;
            res += v[i];
        }
        return res;
    }

    std::string get_reasons_text() const {
        return join(reasons, " | ");
    }
};

// 历史灯条回溯配对
Armor* rematch_by_history_lbs(const std::vector<Lightbar>& lightbars, int& armor_id, float sim_threshold, float hsv_diff_threshold) {
    struct HistoryLevel {
        int left_id;
        int right_id;
    };
    std::vector<HistoryLevel> history_levels;

    if (prev_left_lb_id != -1 && prev_right_lb_id != -1) {
        history_levels.push_back({prev_left_lb_id, prev_right_lb_id});
    }
    if (prev_prev_left_lb_id != -1 && prev_prev_right_lb_id != -1) {
        history_levels.push_back({prev_prev_left_lb_id, prev_prev_right_lb_id});
    }

    if (history_levels.empty()) {
        return nullptr;
    }

    for (const auto& level : history_levels) {
        Lightbar* current_left_lb = nullptr;
        Lightbar* current_right_lb = nullptr;

        for (const auto& lb : lightbars) {
            if (lb.id == level.left_id) { 
                current_left_lb = const_cast<Lightbar*>(&lb);
            }
            if (lb.id == level.right_id) { 
                current_right_lb = const_cast<Lightbar*>(&lb);
            }
        }

        if (current_left_lb != nullptr && current_right_lb != nullptr) {
            Armor* rematched_armor = new Armor(*current_left_lb, *current_right_lb, armor_id++);
            rematched_armor->check_geometry(sim_threshold, hsv_diff_threshold);
            return rematched_armor;
        }
    }

    return nullptr;
}

// ONNX推理函数
std::pair<std::string, float> onnx_infer(cv::dnn::dnn4_v20211004::Net& net, const cv::Mat& input_img, const std::vector<std::string>& class_labels, int armor_id) {
    if (input_img.size() != cv::Size(100, 240)) {
        return {"invalid_input_size", 0.0f};
    }

    cv::Mat input_3ch;
    cv::cvtColor(input_img, input_3ch, cv::COLOR_GRAY2BGR);  
    cv::Mat blob;

    try {
        cv::dnn::blobFromImage(input_3ch, blob, 1.0/127.5, cv::Size(100,240), cv::Scalar(127.5,127.5,127.5), false, false, CV_32F);
    } catch (const cv::Exception& e) {
        return {"blob_error", 0.0f};
    }

    net.setInput(blob, "input");  
    cv::Mat output;
    try {
        output = net.forward("output");  
    } catch (const cv::Exception& e) {
        return {"infer_error", 0.0f};
    }

    if (output.rows == 1 && output.cols == 9) {  
        double max_conf;
        cv::Point max_loc;
        cv::minMaxLoc(output, nullptr, &max_conf, nullptr, &max_loc);
        float normalized_conf = static_cast<float>(max_conf);
        if (normalized_conf > 1.0f || normalized_conf < 0.0f) {
            normalized_conf = 1.0f / (1.0f + exp(-normalized_conf));
        }
        return {class_labels[max_loc.x], normalized_conf};
    }
    return {"invalid_output_size", 0.0f};
}

// 单帧处理函数
void process_single_frame(cv::Mat& frame, cv::dnn::dnn4_v20211004::Net& net, const std::vector<std::string>& class_labels, int frame_idx) {
    cv::Mat debug_img = frame.clone();
    cv::Mat lightbar_result = frame.clone();
    cv::Mat armor_result = frame.clone();
    bool is_single_rect_matched = false;
    bool has_valid_armor = false;

    std::vector<cv::Rect> target_rects = get_first_code_valid_rects(frame, is_single_rect_matched);

    for (size_t i=0; i<target_rects.size(); i++) {
        cv::Scalar rect_color = (target_rects.size() == 2 && is_single_rect_matched) ? 
                              ((i == 0) ? cv::Scalar(255,0,0) : cv::Scalar(0,255,255)) : cv::Scalar(255,0,0);
        cv::rectangle(lightbar_result, target_rects[i], rect_color, 2);
    }

    cv::Mat gray_img, binary_img;
    cv::cvtColor(frame, gray_img, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_img, binary_img, 200, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<Lightbar> lightbars;
    bool check_target_rect = !relax_target_rect_check;
    
    for (size_t i = 0; i < contours.size(); ++i) {
        double contour_area = cv::contourArea(contours[i]);
        if (contour_area < 10) continue;

        cv::RotatedRect rect = cv::minAreaRect(contours[i]);
        Lightbar lb(rect, i);
        lb.check_geometry(debug_img, frame, target_rects, check_target_rect);
        lightbars.push_back(lb);

        cv::Point2f vertices[4];
        lb.rect.points(vertices);
        cv::Scalar lb_color = (lb.is_in_target_rect ? (lb.is_valid ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255)) : cv::Scalar(128,128,128));
        for (int j = 0; j < 4; ++j) cv::line(lightbar_result, vertices[j], vertices[(j+1)%4], lb_color, 2);
        cv::putText(lightbar_result, "ID:" + std::to_string(lb.id), 
                    lb.rect.center + cv::Point2f(10, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, lb_color, 2);
    }

    int valid_lightbar_count = 0;
    for (const auto& lb : lightbars) {
        if (lb.is_valid) valid_lightbar_count++;
    }

    std::vector<Armor> armors;
    int armor_id = 0;
    const float sim_threshold = 0.5f;
    const float hsv_diff_threshold = 15.0f;
    std::vector<ArmorCandidate> candidates;
    bool is_center_jumped = false;
    bool is_area_jumped = false;
    std::string pair_history_type = "";

    if (valid_lightbar_count >= 2) {
        for (size_t i = 0; i < lightbars.size(); ++i) {
            if (!lightbars[i].is_valid) continue;
            for (size_t j = i + 1; j < lightbars.size(); ++j) {
                if (!lightbars[j].is_valid) continue;

                float temp_distance = cv::norm(lightbars[i].rect.center - lightbars[j].rect.center);
                auto sim_data = lightbars[i].calculate_similarity_details(lightbars[j]);
                float area_diff = std::abs(lightbars[i].area - lightbars[j].area);
                float hsv_diff = lightbars[i].calculate_hsv_diff(lightbars[j]);
                
                if (temp_distance > MAX_LIGHTBAR_DISTANCE) continue;
                
                candidates.push_back({
                    (int)i, (int)j, area_diff, hsv_diff, 
                    sim_data.total, sim_data.area_ratio, sim_data.length_ratio
                });
            }
        }

        std::sort(candidates.begin(), candidates.end());

        std::vector<bool> used(lightbars.size(), false);
        for (const auto& cand : candidates) {
            int i = cand.idx1;
            int j = cand.idx2;
            if (used[i] || used[j]) continue;

            Armor armor(lightbars[i], lightbars[j], armor_id++);
            if (armor.check_geometry(sim_threshold, hsv_diff_threshold)) {
                armors.push_back(armor);
                used[i] = true;
                used[j] = true;
                has_valid_armor = true;
            }
        }

        if (!armors.empty() && prev_armor_center != cv::Point(-1, -1) && prev_armor_area > 0) {
            Armor& current_armor = armors[0];
            cv::Point current_center = (current_armor.left.rect.center + current_armor.right.rect.center) / 2;
            float current_area = current_armor.total_area;
            float center_dist = cv::norm(current_center - prev_armor_center);
            float area_ratio = current_area / prev_armor_area;

            if (center_dist > max_center_dist) is_center_jumped = true;
            if (area_ratio > max_area_ratio || area_ratio < 1.0f / max_area_ratio) is_area_jumped = true;

            if (is_center_jumped || is_area_jumped) {
                armors.clear();
                has_valid_armor = false;
                Armor* rematched_armor = rematch_by_history_lbs(lightbars, armor_id, sim_threshold, hsv_diff_threshold);
                if (rematched_armor != nullptr) {
                    armors.push_back(*rematched_armor);
                    has_valid_armor = rematched_armor->is_valid;
                    delete rematched_armor;
                }
            }
        }

        if (armors.empty() && (prev_armor_center != cv::Point(-1, -1) || prev_prev_armor_center != cv::Point(-1, -1))) {
            Armor* rematched_armor = rematch_by_history_lbs(lightbars, armor_id, sim_threshold, hsv_diff_threshold);
            if (rematched_armor != nullptr) {
                armors.push_back(*rematched_armor);
                has_valid_armor = rematched_armor->is_valid;
                delete rematched_armor;
            }
        }
    } else {
        if (prev_armor_center != cv::Point(-1, -1) || prev_prev_armor_center != cv::Point(-1, -1)) {
            Armor* rematched_armor = rematch_by_history_lbs(lightbars, armor_id, sim_threshold, hsv_diff_threshold);
            if (rematched_armor != nullptr) {
                armors.push_back(*rematched_armor);
                has_valid_armor = rematched_armor->is_valid;
                delete rematched_armor;
            }
        }
    }

    if (!armors.empty() && has_valid_armor) {
        Armor& valid_armor = armors[0];
        prev_prev_armor_center = prev_armor_center;
        prev_prev_armor_area = prev_armor_area;
        prev_prev_left_lb_id = prev_left_lb_id;
        prev_prev_right_lb_id = prev_right_lb_id;
        prev_armor_center = (valid_armor.left.rect.center + valid_armor.right.rect.center) / 2;
        prev_armor_area = valid_armor.total_area;
        prev_left_lb_id = valid_armor.left.id;
        prev_right_lb_id = valid_armor.right.id;
        
        consecutive_failure_count = 0;
        relax_target_rect_check = false;
    } else {
        consecutive_failure_count++;
        if (consecutive_failure_count >= MAX_FAILURES_BEFORE_RELAX) {
            relax_target_rect_check = true;
        }
    }

    std::string pair_type = "常规配对";
    if (!pair_history_type.empty()) {
        pair_type = "复用历史灯条";
    }

    for (auto& armor : armors) {
        if (!armor.is_valid) continue;

        cv::Mat pattern_debug = frame.clone();
        if (armor.extract_pattern(frame, pattern_debug)) {
            auto [pred_label, pred_conf] = onnx_infer(net, armor.normalized_pattern, class_labels, armor.id);
            float normalized_conf = std::max(0.0f, std::min(1.0f, pred_conf));
            if (pred_label.find("invalid") == std::string::npos && pred_label.find("error") == std::string::npos) {
                std::cout << "Frame " << frame_idx << " | Armor " << armor.id << " | " << pair_type << std::endl;
                std::cout << "Label: " << pred_label << " | Confidence: " << std::fixed << std::setprecision(2) << (normalized_conf * 100) << "%" << std::endl;
                std::cout << "使用灯条：左ID=" << armor.left.id << " 右ID=" << armor.right.id << std::endl;
                std::cout << "============================" << std::endl;

                std::string info = "Label: " + pred_label + " (" + std::to_string((int)(normalized_conf * 100)) + "%)";
                cv::putText(armor_result, info, (armor.left.rect.center + armor.right.rect.center)/2 + cv::Point2f(-50, -50), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,255), 2);
            }
        }

        draw_armor_base_info(armor_result, armor);
    }

    std::string frame_text = "Frame: " + std::to_string(frame_idx);
    cv::putText(armor_result, frame_text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
    cv::imshow("最终检测结果", armor_result);
}

// 辅助函数：绘制装甲板基础信息
void draw_armor_base_info(cv::Mat& img, const Armor& armor) {
    cv::Point2f vertices[4];
    armor.left.rect.points(vertices);
    for (int i = 0; i < 4; ++i) cv::line(img, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0), 2);
    armor.right.rect.points(vertices);
    for (int i = 0; i < 4; ++i) cv::line(img, vertices[i], vertices[(i+1)%4], cv::Scalar(0,255,0), 2);
    cv::line(img, armor.left.rect.center, armor.right.rect.center, cv::Scalar(255,0,0), 2);
}

// 主函数
int main() {
    std::string onnx_model_path = "/home/pl/5/shuzi_juanji/cnn_model_100x240_binary.onnx";
    cv::dnn::dnn4_v20211004::Net net = cv::dnn::readNetFromONNX(onnx_model_path);
    if (net.empty()) {
        std::cerr << "无法加载ONNX模型！" << std::endl;
        return -1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::vector<std::string> class_labels = {"1", "2", "3", "4", "5", "6outpost", "7guard", "8base", "9neg"};

    std::string video_path = "/home/pl/5/resources/blue.mp4";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件！" << std::endl;
        return -1;
    }
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    int delay_ms = static_cast<int>(1000.0 / video_fps);

    std::cout << "===== 控制说明 =====" << std::endl;
    std::cout << "P键：暂停/继续 | N键：单帧步进（暂停时） | Q键：退出" << std::endl;

    cv::Mat frame;
    int frame_idx = 0;
    bool is_paused = false;
    bool is_single_step = false;
    bool is_first_frame = true;

    relax_target_rect_check = false;
    consecutive_failure_count = 0;

    while (true) {
        if (is_first_frame || !is_paused || is_single_step) {
            if (!cap.read(frame)) {
                std::cout << "视频处理完成！" << std::endl;
                break;
            }
            
            process_single_frame(frame, net, class_labels, frame_idx);
            frame_idx++;
            is_first_frame = false;
            is_single_step = false;
        }

        char key = cv::waitKey(delay_ms) & 0xFF;
        if (key == 'p' || key == 'P') {
            is_paused = !is_paused;
        } else if (key == 'n' || key == 'N') {
            if (is_paused) {
                is_single_step = true;
            }
        } else if (key == 'q' || key == 'Q') {
            std::cout << "用户手动退出！" << std::endl;
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
