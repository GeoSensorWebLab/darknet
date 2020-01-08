#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <future>
#include <atomic>
#include <mutex>         // std::mutex, std::unique_lock
#include <cmath>
#include "httplib.h"
#include <time.h>
#include <ctime>
#include <cuchar>
// It makes sense only for video-Camera (not for video-File)
// To use - uncomment the following line. Optical-flow is supported only by OpenCV 3.x - 4.x
//#define TRACK_OPTFLOW
//#define GPU

// To use 3D-stereo camera ZED - uncomment the following line. ZED_SDK should be installed.
//#define ZED_STEREO


#include "yolo_v2_class.hpp"    // imported functions from DLL
#include "base64.cpp"
#ifdef OPENCV
#ifdef ZED_STEREO
#include <sl_zed/Camera.hpp>
#pragma comment(lib, "sl_core64.lib")
#pragma comment(lib, "sl_input64.lib")
#pragma comment(lib, "sl_zed64.lib")

float getMedian(std::vector<float> &v) {
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}
//check

std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba)
{
    bool valid_measure;
    int i, j;
    const unsigned int R_max_global = 10;

    std::vector<bbox_t> bbox3d_vect;

    for (auto &cur_box : bbox_vect) {

        const unsigned int obj_size = std::min(cur_box.w, cur_box.h);
        const unsigned int R_max = std::min(R_max_global, obj_size / 2);
        int center_i = cur_box.x + cur_box.w * 0.5f, center_j = cur_box.y + cur_box.h * 0.5f;

        std::vector<float> x_vect, y_vect, z_vect;
        for (int R = 0; R < R_max; R++) {
            for (int y = -R; y <= R; y++) {
                for (int x = -R; x <= R; x++) {
                    i = center_i + x;
                    j = center_j + y;
                    sl::float4 out(NAN, NAN, NAN, NAN);
                    if (i >= 0 && i < xyzrgba.cols && j >= 0 && j < xyzrgba.rows) {
                        cv::Vec4f &elem = xyzrgba.at<cv::Vec4f>(j, i);  // x,y,z,w
                        out.x = elem[0];
                        out.y = elem[1];
                        out.z = elem[2];
                        out.w = elem[3];
                    }
                    valid_measure = std::isfinite(out.z);
                    if (valid_measure)
                    {
                        x_vect.push_back(out.x);
                        y_vect.push_back(out.y);
                        z_vect.push_back(out.z);
                    }
                }
            }
        }

        if (x_vect.size() * y_vect.size() * z_vect.size() > 0)
        {
            cur_box.x_3d = getMedian(x_vect);
            cur_box.y_3d = getMedian(y_vect);
            cur_box.z_3d = getMedian(z_vect);
        }
        else {
            cur_box.x_3d = NAN;
            cur_box.y_3d = NAN;
            cur_box.z_3d = NAN;
        }

        bbox3d_vect.emplace_back(cur_box);
    }

    return bbox3d_vect;
}

cv::Mat slMat2cvMat(sl::Mat &input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
    case sl::MAT_TYPE_32F_C1:
        cv_type = CV_32FC1;
        break;
    case sl::MAT_TYPE_32F_C2:
        cv_type = CV_32FC2;
        break;
    case sl::MAT_TYPE_32F_C3:
        cv_type = CV_32FC3;
        break;
    case sl::MAT_TYPE_32F_C4:
        cv_type = CV_32FC4;
        break;
    case sl::MAT_TYPE_8U_C1:
        cv_type = CV_8UC1;
        break;
    case sl::MAT_TYPE_8U_C2:
        cv_type = CV_8UC2;
        break;
    case sl::MAT_TYPE_8U_C3:
        cv_type = CV_8UC3;
        break;
    case sl::MAT_TYPE_8U_C4:
        cv_type = CV_8UC4;
        break;
    default:
        break;
    }
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM_CPU));
}

cv::Mat zed_capture_rgb(sl::Camera &zed) {
    sl::Mat left;
    zed.retrieveImage(left);
    cv::Mat left_rgb;
    cv::cvtColor(slMat2cvMat(left), left_rgb, CV_RGBA2RGB);
    return left_rgb;
}

cv::Mat zed_capture_3d(sl::Camera &zed) {
    sl::Mat cur_cloud;
    zed.retrieveMeasure(cur_cloud, sl::MEASURE_XYZ);
    return slMat2cvMat(cur_cloud).clone();
}

static sl::Camera zed; // ZED-camera

#else   // ZED_STEREO
std::vector<bbox_t> get_3d_coordinates(std::vector<bbox_t> bbox_vect, cv::Mat xyzrgba) {
    return bbox_vect;
}
#endif  // ZED_STEREO


#include <opencv2/opencv.hpp>            // C++
#include <opencv2/core/version.hpp>
#ifndef CV_VERSION_EPOCH     // OpenCV 3.x and 4.x
#include <opencv2/videoio/videoio.hpp>
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#ifdef TRACK_OPTFLOW
#pragma comment(lib, "opencv_cudaoptflow" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_cudaimgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif    // TRACK_OPTFLOW
#endif    // USE_CMAKE_LIBS
#else     // OpenCV 2.x
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_video" OPENCV_VERSION ".lib")
#endif    // USE_CMAKE_LIBS
#endif    // CV_VERSION_EPOCH
std::string app_mode;
bool ccw(cv::Point2f A, cv::Point2f B, cv::Point2f C)
{
    bool val = true;
    if ((C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x))
    {
        val = true;
    }
    else
    {
        val = false;
    }
    return val;
}
bool intersect(cv::Point2f A, cv::Point2f B, cv::Point2f C, cv::Point2f D)
{
    bool val = true;
    if (ccw(A, C, D) != ccw(B, C, D) && ccw(A, B, C) != ccw(A, B, D))
    {
        val = true;
    }
    else
    {
        val = false;
    }
    return val;
}

//add logo
cv::Mat addUCalgaryLogo_init(cv::Mat temp, std::string filename) //check
{
    using namespace std;
    using namespace cv;

    std::cout << "init logo" << std::endl;

    int h = 0;
    int w = 0;

    if (filename == "web_camera") {
        cv::VideoCapture cap(1);

        w = (int)(cap.get(3));
        h = (int)(cap.get(4));

        cap.release();
    }
    else {
        cv::VideoCapture cap(filename);
        cap.open(filename);

        if (!cap.isOpened()) {
            std::cout << "file cannot be opened!" << std::endl;
        }

        w = (int)(cap.get(3));
        h = (int)(cap.get(4));

        cap.release();
    }

    std::cout << "frame size = w: " + std::to_string(w) + ", H: " + std::to_string(h) << std::endl;

    cv::Mat logo = cv::imread("./data/logo.png", cv::IMREAD_UNCHANGED);
    std::cout << "logo size = w: " + std::to_string(logo.cols) + ", H: " + std::to_string(logo.rows) << std::endl;
    cv::cvtColor(logo, logo, CV_BGRA2GRAY);

    //resize logo image based on the shrink value and the frame size
    int shrink = 10;
    float height = float(h / shrink);
    float width = float(logo.cols * h / (shrink * logo.rows));
    std::cout << "logo size = w: " + std::to_string(int(width)) + ", H: " + std::to_string(int(height)) << std::endl;
    cv::Size size((width), (height));
    cv::resize(logo, logo, size, 0, 0);

    cv::threshold(logo, logo, 127, 255, THRESH_BINARY);
    std::vector <Mat> channels;
    channels.push_back(logo);
    channels.push_back(logo);
    channels.push_back(logo);
    channels.push_back(logo);
    cv::merge(channels, logo);
    cv::bitwise_not(logo, logo);
    std::cout << logo.channels() << std::endl;

    cv::Mat test(h, w, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    cv::bitwise_or(logo, test(cv::Rect(w - logo.cols - 5, h - logo.rows - 5, logo.cols, logo.rows)), test(cv::Rect(w - logo.cols - 5, h - logo.rows - 5, logo.cols, logo.rows)));

    return test;

}

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names,
    int current_det_fps = -1, int current_cap_fps = -1)
{
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

    for (auto &i : result_vec) {
        cv::Scalar color = obj_id_to_color(i.obj_id);
        if (app_mode == "t")
        {
            if (i.obj_id == 2)
            {
                cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
            }
        }
        else
        {
            cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
        }
        //check
               // if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::string test = "x = " + std::to_string(i.x) + ", y = " + std::to_string(i.y) + ", w = " + std::to_string(i.w) + ", h = " + std::to_string(i.h);// + std::setprecision(3) + ", prob = " + std::to_string(i.prob);
        if (i.track_id > 0) test += " - " + std::to_string(i.track_id);

        //std::cout << test << std::endl;

        if (obj_names.size() > i.obj_id) {//check
            std::string obj_name = "";//obj_names[i.obj_id] + " - " + std::to_string(i.prob);
            if (i.track_id > 0) obj_name += std::to_string(i.track_id);
            cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
            int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
            max_width = std::max(max_width, (int)i.w + 2);
            //max_width = std::max(max_width, 283);
            std::string coords_3d;
            if (!std::isnan(i.z_3d)) {
                std::stringstream ss;
                // ss << std::fixed << std::setprecision(2) << "x:" << i.x_3d << "m y:" << i.y_3d << "m z:" << i.z_3d << "m ";
                coords_3d = ss.str();
                cv::Size const text_size_3d = getTextSize(ss.str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, 1, 0);
                int const max_width_3d = (text_size_3d.width > i.w + 2) ? text_size_3d.width : (i.w + 2);
                if (max_width_3d > max_width) max_width = max_width_3d;
            }
            if (app_mode == "t")
            {
                if (i.obj_id == 2)
                {
                    cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 10, 0)),
                        cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
                        color, CV_FILLED, 8, 0);
                    putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 8), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0), 1);
                    //check
                    if (!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
                }
            }
            else
            {
                obj_name = obj_names[i.obj_id];
                cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 10, 0)),
                cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)),
                color, CV_FILLED, 8, 0);
                putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 8), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0), 1);
                //check
                if (!coords_3d.empty()) putText(mat_img, coords_3d, cv::Point2f(i.x, i.y - 1), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
                
            }
        }
    }
    /* check
    if (current_det_fps >= 0 && current_cap_fps >= 0) {
        std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
        putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
    }
    */
}
#endif    // OPENCV


void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
    if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::string test = "x = " + std::to_string(i.x) + ", y = " + std::to_string(i.y) + ", w = " + std::to_string(i.w) + ", h = " + std::to_string(i.h);// + std::setprecision(3) + ", prob = " + std::to_string(i.prob);
        if (i.track_id > 0) test += " - " + std::to_string(i.track_id);
        //std::cout << test << std::endl;
    }
}

void send2_compusult(std::string js, std::string temp,std::string resval)
{
    bool SensorUp = true;
    httplib::Headers hdr;
    std::string host;
    time_t rawtime;
    struct tm*timeinfo;
    const char* queryPart;
    if (SensorUp) {
        std::cout << "send to SensorUP" << std::endl;
        host = "ucalgary-sandbox-02.sensorup.com"; // 52.11.65.90 SensorUp IP
        // 52.11.65.90:80/v1.0/Datastreams%281630%29/Observations
        hdr.emplace(httplib::make_basic_authentication_header("main", "bdbe16d7-f55e-520b-8c97-5354ec272aed")); //sandbox
        //temp = "/v1.0/Datastreams(1630)/Observations";
        queryPart = temp.c_str();
        time(&rawtime);
        timeinfo = gmtime(&rawtime);
        std::string strtime = asctime(timeinfo);
        char buf[sizeof "2019-12-09T19:49:39Z"];
        strftime(buf, sizeof buf, "%FT%TZ", timeinfo);
        std::string bufString(buf);
        std::string buffer = "\"" + bufString + "\"";

        // js = "{\"phenomenonTime\": " + buffer + ",\"resultTime\" : " + buffer + ",\"result\" :\"" + resval + "\", \"FeatureOfInterest\": {\"@iot.id\": \"5321\"}" + "}";
        js = "{\"result\" :\"" + resval + "\", \"FeatureOfInterest\": {\"@iot.id\": \"5321\"}" + "}";
    }
    else {
        std::cout << "send to Compusult" << std::endl;
        host = "ogc-hub.compusult.com"; // /SensorHub/SensorThings"; // "https://ogc-hub.compusult.com"; // 24.137.216.70 // Compusult IP
        // 24.137.216.70:80/SensorHub/SensorThings/v1.0/Datastreams%28199%29/Observations

        temp = "/SensorThings/v1.0/Datastreams(199)/Observations";
        queryPart = temp.c_str();

        hdr.emplace(httplib::make_basic_authentication_header("ucalgary", "scira01")); //Compusult
    }

    //hdr.emplace("Content-Type", "application/json");
    const char *hostUrl = host.c_str();
    httplib::Client cli(hostUrl);

    std::string tt(queryPart);
    std::cout << "query: " + temp << std::endl;

    // cli.set_follow_location(true); // it will give the permission for redirecting to another address
    std::string contentStr = "application/json";
    const char* content_type = contentStr.c_str();
    auto res = cli.Post(queryPart, hdr, js, content_type);

    bool iiii = cli.is_valid();
    if (res && (res->status == 200 || res->status == 201)) {
       // std::cout << res->body << std::endl;
    }
    std::cout << res->body << std::endl;
}

void show_console_result2(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int fcount) {
    if (fcount >= 0) std::cout << " Frame: " << fcount << std::endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::string test = "x = " + std::to_string(i.x) + ", y = " + std::to_string(i.y) + ", w = " + std::to_string(i.w) + ", h = " + std::to_string(i.h);// + std::setprecision(3) + ", prob = " + std::to_string(i.prob);
        if (i.track_id > 0) test += " - " + std::to_string(i.track_id);
        std::cout << test << std::endl;
    }
}
std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for (std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}

template<typename T>
class send_one_replaceable_object_t {
    const bool sync;
    std::atomic<T *> a_ptr;
public:

    void send(T const& _obj) {
        T *new_ptr = new T;
        *new_ptr = _obj;
        if (sync) {
            while (a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
    }

    T receive() {
        std::unique_ptr<T> ptr;
        do {
            while (!a_ptr.load()) std::this_thread::sleep_for(std::chrono::milliseconds(3));
            ptr.reset(a_ptr.exchange(NULL));
        } while (!ptr);
        T obj = *ptr;
        return obj;
    }

    bool is_object_present() {
        return (a_ptr.load() != NULL);
    }

    send_one_replaceable_object_t(bool _sync) : sync(_sync), a_ptr(NULL)
    {}
};

cv::Mat temp;
int find_maximum(int array_vals[3])
{
    int indx;
    int* i1;
    i1 = std::max_element(array_vals , array_vals+3);
    std::cout << *i1 << "\n";
    if (*i1 == array_vals[0])
        indx = 0;
    else if (*i1 == array_vals[1])
        indx = 1;
    else if (*i1 == array_vals[2])
        indx = 2;
  
    return indx;
}

int main(int argc, char *argv[])
{
    std::string  names_file = "data/coco.names";
    std::string  cfg_file = "cfg/yolov3.cfg";
    std::string  weights_file = "yolov3.weights";
    std::string filename;
    std::vector<double> car_arr;
    std::vector<double> pers_arr;
    std::vector<double> speed_arr;
    int flood_frame_counter = 0;
    //time_t startsec;
    //const long double startsec = time(0) * 1000;
    std::chrono::steady_clock::time_point startsec, nowsec;

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;
    auto t0c = Time::now();
    auto t0s = Time::now();

    //startsec = time(NULL);
    if (argc > 4) {    //voc.names yolo-voc.cfg yolo-voc.weights test.mp4
        names_file = argv[1];
        cfg_file = argv[2];
        weights_file = argv[3];
        filename = argv[4];
    }
    else if (argc > 2 && argc<4)
    {
        
        if (std::string( argv[2]) == "traffic")
        {
            filename = argv[1];
            app_mode = "t";
        }
        else if (std::string(argv[2]) == "flood")

        {
            app_mode = "f";
            filename = argv[1];
            
            // for Jetson
            cfg_file = "/home/mahnoush/darknetV2_yolov3_/darknet-alexyabEdited/flood/yolo-obj.cfg"; 
            weights_file = "/home/mahnoush/darknetV2_yolov3_/darknet-alexyabEdited/flood/yolo-obj_final.weights";
            names_file = "/home/mahnoush/darknetV2_yolov3_/darknet-alexyabEdited/flood/obj.names";

            //for windows
            // cfg_file = "C:\\Users\\geose\\Documents\\darknet-master_uslib\\Release\\YOLO\\trainedModel\\yolo-obj.cfg"; 
            // weights_file = "C:\\Users\\geose\\Documents\\darknet-master_uslib\\Release\\YOLO\\trainedModel\\yolo-obj_final.weights";
            // names_file = "C:\\Users\\geose\\Documents\\darknet-master_uslib\\Release\\YOLO\\trainedModel\\obj.names";
        }
        else
        {
            std::cout << "You have to specify your app mode traffic|flood" << std::endl;
        }
    }
    else if (argc > 1) filename = argv[1];

    temp = addUCalgaryLogo_init(temp, filename);

    float const thresh = (argc > 5) ? std::stof(argv[5]) : 0.9;

    Detector detector(cfg_file, weights_file);

    auto obj_names = objects_names_from_file(names_file);
    std::string out_videofile = "result_uselib.avi";
    bool const save_output_videofile = true;   // true - for history
    bool const send_network = false;        // true - for remote detection
    bool const use_kalman_filter = false;   // true - for stationary camera

    bool detection_sync = true;             // true - for video-file
#ifdef TRACK_OPTFLOW    // for slow GPU
    detection_sync = false;
    Tracker_optflow tracker_flow;
    //detector.wait_stream = true;
#endif  // TRACK_OPTFLOW

    int frame_count = 0;
    std::vector<bbox_t> previous_box;
    std::vector<int> intersect_frame;
    std::vector<int> intersect_track;
    std::string speedtxt;
    int car_count = 0;
    int person_count = 0;
    int no_water_count = 0;
    int low_water_count = 0;
    int high_water_count = 0;
    int splash_water_count = 0;

    while (true)
    {
        std::cout << "input image or video filename: ";
        if (filename.size() == 0) std::cin >> filename;
        if (filename.size() == 0) break;

        try {
#ifdef OPENCV
            preview_boxes_t large_preview(100, 150, false), small_preview(50, 50, true);
            bool show_small_boxes = false;

            std::string const file_ext = filename.substr(filename.find_last_of(".") + 1);
            std::string const protocol = filename.substr(0, 7);
            if (file_ext == "avi" || file_ext == "mp4" || file_ext == "mjpg" || file_ext == "mov" ||     // video file
                protocol == "rtmp://" || protocol == "rtsp://" || protocol == "http://" || protocol == "https:/" ||    // video network stream
                filename == "zed_camera" || file_ext == "svo" || filename == "web_camera")   // ZED stereo camera

            {//check
                if (protocol == "rtsp://" || protocol == "http://" || protocol == "https:/" || filename == "zed_camera" /*|| filename == "web_camera"*/)
                    detection_sync = false;
                detection_sync = true;
                cv::Mat cur_frame;
                std::atomic<int> fps_cap_counter(0), fps_det_counter(0);
                std::atomic<int> current_fps_cap(0), current_fps_det(0);
                std::atomic<bool> exit_flag(false);
                std::chrono::steady_clock::time_point steady_start, steady_end;
                int video_fps = 25;
                bool use_zed_camera = false;

                track_kalman_t track_kalman;

#ifdef ZED_STEREO
                sl::InitParameters init_params;
                init_params.depth_minimum_distance = 0.5;
                init_params.depth_mode = sl::DEPTH_MODE_ULTRA;
                init_params.camera_resolution = sl::RESOLUTION_HD720;// sl::RESOLUTION_HD1080, sl::RESOLUTION_HD720
                init_params.coordinate_units = sl::UNIT_METER;
                //init_params.sdk_cuda_ctx = (CUcontext)detector.get_cuda_context();
                init_params.sdk_gpu_id = detector.cur_gpu_id;
                init_params.camera_buffer_count_linux = 2;
                if (file_ext == "svo") init_params.svo_input_filename.set(filename.c_str());
                if (filename == "zed_camera" || file_ext == "svo") {
                    std::cout << "ZED 3D Camera " << zed.open(init_params) << std::endl;
                    if (!zed.isOpened()) {
                        std::cout << " Error: ZED Camera should be connected to USB 3.0. And ZED_SDK should be installed. \n";
                        getchar();
                        return 0;
                    }
                    cur_frame = zed_capture_rgb(zed);
                    use_zed_camera = true;
                }
#endif  // ZED_STEREO

                cv::VideoCapture cap;
                if (filename == "web_camera") {
                    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
                    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

                    cap.open(1); //if error change this value to 0
                    cap >> cur_frame;
                }
                else if (!use_zed_camera) {
                    cap.open(filename);
                    cap >> cur_frame;
                }
#ifdef CV_VERSION_EPOCH // OpenCV 2.x
                video_fps = cap.get(CV_CAP_PROP_FPS);
#else
                video_fps = cap.get(cv::CAP_PROP_FPS);
#endif
                cv::Size const frame_size = cur_frame.size();
                //cv::Size const frame_size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
                std::cout << "\n Video size: " << frame_size << std::endl;

                cv::VideoWriter output_video;
                if (save_output_videofile)
#ifdef CV_VERSION_EPOCH // OpenCV 2.x
                    output_video.open(out_videofile, CV_FOURCC('D', 'I', 'V', 'X'), std::max(35, video_fps), frame_size, true);
#else
                    output_video.open(out_videofile, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), std::max(35, video_fps), frame_size, true);
#endif

                struct detection_data_t {
                    cv::Mat cap_frame;
                    std::shared_ptr<image_t> det_image;
                    std::vector<bbox_t> result_vec;
                    cv::Mat draw_frame;
                    bool new_detection;
                    uint64_t frame_id;
                    bool exit_flag;
                    cv::Mat zed_cloud;
                    std::queue<cv::Mat> track_optflow_queue;
                    detection_data_t() : exit_flag(false), new_detection(false) {}
                };

                const bool sync = detection_sync; // sync data exchange
                send_one_replaceable_object_t<detection_data_t> cap2prepare(sync), cap2draw(sync),
                    prepare2detect(sync), detect2draw(sync), draw2show(sync), draw2write(sync), draw2net(sync);

                std::thread t_cap, t_prepare, t_detect, t_post, t_draw, t_write, t_network;

                // capture new video-frame
                if (t_cap.joinable()) t_cap.join();
                t_cap = std::thread([&]()
                {
                    uint64_t frame_id = 0;
                    detection_data_t detection_data;
                    do {
                        detection_data = detection_data_t();
#ifdef ZED_STEREO
                        if (use_zed_camera) {
                            while (zed.grab() != sl::SUCCESS) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                            detection_data.cap_frame = zed_capture_rgb(zed);
                            detection_data.zed_cloud = zed_capture_3d(zed);
                        }
                        else
#endif   // ZED_STEREO
                        {
                            cap >> detection_data.cap_frame;
                        }
                        fps_cap_counter++;
                        detection_data.frame_id = frame_id++;
                        if (detection_data.cap_frame.empty() || exit_flag) {
                            std::cout << " exit_flag: detection_data.cap_frame.size = " << detection_data.cap_frame.size() << std::endl;
                            detection_data.exit_flag = true;
                            detection_data.cap_frame = cv::Mat(frame_size, CV_8UC3);
                        }

                        if (!detection_sync) {
                            cap2draw.send(detection_data);       // skip detection
                        }
                        cap2prepare.send(detection_data);
                    } while (!detection_data.exit_flag);
                    std::cout << " t_cap exit \n";
                });


                // pre-processing video frame (resize, convertion)
                t_prepare = std::thread([&]()
                {
                    std::shared_ptr<image_t> det_image;
                    detection_data_t detection_data;
                    do {
                        detection_data = cap2prepare.receive();

                        det_image = detector.mat_to_image_resize(detection_data.cap_frame);
                        detection_data.det_image = det_image;
                        prepare2detect.send(detection_data);    // detection

                    } while (!detection_data.exit_flag);
                    std::cout << " t_prepare exit \n";
                });


                // detection by Yolo
                if (t_detect.joinable()) t_detect.join();
                t_detect = std::thread([&]()
                {
                    std::shared_ptr<image_t> det_image;
                    detection_data_t detection_data;
                    do {
                        detection_data = prepare2detect.receive();
                        det_image = detection_data.det_image;
                        std::vector<bbox_t> result_vec;

                        if (det_image)
                            result_vec = detector.detect_resized(*det_image, frame_size.width, frame_size.height, thresh, true);  // true
                        fps_det_counter++;
                        //std::this_thread::sleep_for(std::chrono::milliseconds(150));

                        detection_data.new_detection = true;
                        detection_data.result_vec = result_vec;
                        detect2draw.send(detection_data);
                        //show_console_result(result_vec, obj_names, frame_count);//detection_data.frame_id); //check
                    } while (!detection_data.exit_flag);
                    std::cout << " t_detect exit \n";
                });

                // draw rectangles (and track objects)
                t_draw = std::thread([&]()
                {
                    std::queue<cv::Mat> track_optflow_queue;
                    detection_data_t detection_data;
                    do {

                        // for Video-file
                        if (detection_sync) {
                            detection_data = detect2draw.receive();
                        }
                        // for Video-camera
                        else
                        {
                            // get new Detection result if present
                            if (detect2draw.is_object_present()) {
                                cv::Mat old_cap_frame = detection_data.cap_frame;   // use old captured frame
                                detection_data = detect2draw.receive();
                                if (!old_cap_frame.empty()) detection_data.cap_frame = old_cap_frame;
                            }
                            // get new Captured frame
                            else {
                                std::vector<bbox_t> old_result_vec = detection_data.result_vec; // use old detections
                                detection_data = cap2draw.receive();
                                detection_data.result_vec = old_result_vec;
                            }
                        }

                        cv::Mat cap_frame = detection_data.cap_frame;
                        cv::Mat draw_frame = detection_data.cap_frame.clone();
                        std::vector<bbox_t> result_vec = detection_data.result_vec;

#ifdef TRACK_OPTFLOW
                        if (detection_data.new_detection) {
                            tracker_flow.update_tracking_flow(detection_data.cap_frame, detection_data.result_vec);
                            while (track_optflow_queue.size() > 0) {
                                draw_frame = track_optflow_queue.back();
                                result_vec = tracker_flow.tracking_flow(track_optflow_queue.front(), false);
                                track_optflow_queue.pop();
                            }
                        }
                        else {
                            track_optflow_queue.push(cap_frame);
                            result_vec = tracker_flow.tracking_flow(cap_frame, false);
                        }
                        detection_data.new_detection = true;    // to correct kalman filter
#endif //TRACK_OPTFLOW

                        // track ID by using kalman filter
                        if (use_kalman_filter) {
                            if (detection_data.new_detection) {
                                result_vec = track_kalman.correct(result_vec);
                            }
                            else {
                                result_vec = track_kalman.predict();
                            }
                        }
                        // track ID by using custom function
                        else {
                            int frame_story = std::max(5, current_fps_cap.load());
                            result_vec = detector.tracking_id(result_vec, true, frame_story, 40);
                        }

                        if (use_zed_camera && !detection_data.zed_cloud.empty()) {
                            result_vec = get_3d_coordinates(result_vec, detection_data.zed_cloud);
                        }

                        //small_preview.set(draw_frame, result_vec);
                        //large_preview.set(draw_frame, result_vec);
                        draw_boxes(draw_frame, result_vec, obj_names, current_fps_det, current_fps_cap);
                        //show_console_result(result_vec, obj_names, detection_data.frame_id);
                        //large_preview.draw(draw_frame);
                        //small_preview.draw(draw_frame, true);

                        detection_data.result_vec = result_vec;
                        detection_data.draw_frame = draw_frame;
                        draw2show.send(detection_data);
                        if (send_network) draw2net.send(detection_data);
                        if (output_video.isOpened()) draw2write.send(detection_data);
                    } while (!detection_data.exit_flag);
                    std::cout << " t_draw exit \n";
                });


                // write frame to videofile
                t_write = std::thread([&]()
                {
                    if (output_video.isOpened()) {
                        detection_data_t detection_data;
                        cv::Mat output_frame;
                        do {
                            detection_data = draw2write.receive();
                            
                            //add logo
                            cv::cvtColor(detection_data.draw_frame, detection_data.draw_frame, CV_BGR2BGRA);
                            cv::addWeighted(temp, 1, detection_data.draw_frame, 0.9, 0.0, detection_data.draw_frame); //addUCalgaryLogo
                            cv::cvtColor(detection_data.draw_frame, detection_data.draw_frame, CV_BGRA2BGR);
                            
                            if (detection_data.draw_frame.channels() == 4) cv::cvtColor(detection_data.draw_frame, output_frame, CV_RGBA2RGB);
                            else output_frame = detection_data.draw_frame;
                            output_video << output_frame;
                        } while (!detection_data.exit_flag);
                        output_video.release();
                    }
                    std::cout << " t_write exit \n";
                });

                // send detection to the network
                t_network = std::thread([&]()
                {
                    if (send_network) {
                        detection_data_t detection_data;
                        do {
                            detection_data = draw2net.receive();

                            detector.send_json_http(detection_data.result_vec, obj_names, detection_data.frame_id, filename);

                        } while (!detection_data.exit_flag);
                    }
                    std::cout << " t_network exit \n";
                });


                // show detection
                detection_data_t detection_data;
                do {
                    steady_end = std::chrono::steady_clock::now();
                    float time_sec = std::chrono::duration<double>(steady_end - steady_start).count();
                    if (time_sec >= 1) {
                        current_fps_det = fps_det_counter.load() / time_sec;
                        current_fps_cap = fps_cap_counter.load() / time_sec;
                        steady_start = steady_end;
                        fps_det_counter = 0;
                        fps_cap_counter = 0;
                    }

                    detection_data = draw2show.receive();
                    cv::Mat draw_frame = detection_data.draw_frame;

                    //add logo
                    cv::cvtColor(draw_frame, draw_frame, CV_BGR2BGRA);
                    cv::addWeighted(temp, 1, draw_frame, 0.9, 0.0, draw_frame); //addUCalgaryLogo
                    cv::cvtColor(draw_frame, draw_frame, CV_BGRA2BGR);
                    //if (extrapolate_flag) {
                    //    cv::putText(draw_frame, "extrapolate", cv::Point2f(10, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(50, 50, 0), 2);
                    //}
                    int w1 = 0;
                    int h1 = 0;
                    int w2 = 0;
                    int h2 = 0;
                    if (app_mode == "t")
                    {
                        for (auto &i : detection_data.result_vec) {
                            if (i.obj_id == 0) {
                                person_count++;
                            }

                            if (i.obj_id == 2) {
                                car_count++;
                                for (auto &j : previous_box) {
                                    if (j.obj_id == 2) {
                                        w1 = j.x;
                                        h1 = j.y;
                                        w2 = j.x + j.w;
                                        h2 = j.y + j.h;

                                        if (i.track_id == j.track_id) {
                                            std::cout << "**************************************************\n";
                                            std::cout << std::to_string(w2) + "," + std::to_string(h2) + "----------" + std::to_string(i.x + i.w) + "," + std::to_string(i.y + i.h) + "\n";
                                            cv::line(draw_frame, cv::Point2f(w2, h2), cv::Point2f(i.x + i.w, i.y + i.h), cv::Scalar(255, 0, 0), 2, 8);
                                            bool res_down = intersect(cv::Point2f(w2, h2), cv::Point2f(i.x + i.w, i.y + i.h), cv::Point2f(0, frame_size.height / 2), cv::Point2f(draw_frame.size().width, frame_size.height / 2));
                                            if (res_down == true) {
                                                intersect_frame.push_back(int(detection_data.frame_id));
                                                intersect_track.push_back(j.track_id);
                                                std::cout << "uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu\n";

                                            }

                                            if (intersect_track.size() > 0) {
                                                bool res_up = intersect(cv::Point2f(w1, h1), cv::Point2f(i.x, i.y), cv::Point2f(0, frame_size.height / 2), cv::Point2f(draw_frame.size().width, frame_size.height / 2));
                                                if (res_up == true) {
                                                    std::cout << "ddddddddddddddddddddddddddddddddddddddddd\n";
                                                    std::vector<int> ::iterator itr = std::find(intersect_track.begin(), intersect_track.end(), i.track_id);
                                                    if (itr != intersect_track.end())
                                                    {
                                                        int indx = std::distance(intersect_track.begin(), itr);
                                                    std::vector<int>::iterator it = intersect_frame.begin();
                                                    std::advance(it, indx);
                                                    //std::cout<<"---------------------------this frame----------------------------\n";
                                                    //std::cout<<std::to_string(detection_data.frame_id)+"\n";
                                                    //std::cout<<"---------------------------past frame----------------------------\n";
                                                    //std::cout<<std::to_string(*it)+"\n"; 
                                                    //std::cout<<"---------------------------delta t----------------------------\n";
                                                    int this_frame = int(*it);
                                                    double deltat = (double(int(detection_data.frame_id)) - double(this_frame)) / 90;
                                                    //std::cout<<std::to_string(deltat); 
                                                    double speedo = 8 / deltat;
                                                    intersect_frame.clear();
                                                    intersect_track.clear();
                                                    if (speedo < 200)
                                                    {
                                                        speedtxt = std::to_string(int(speedo));
                                                        speed_arr.push_back(speedo);
                                                    }
                                                    if (speedo < 40)
                                                        std::cout << "mmmmmmmmmmm" << detection_data.frame_id << "  " << this_frame << "  " << double(detection_data.frame_id - this_frame) << std::endl;
                                                    else
                                                        std::cout << detection_data.frame_id << "  " << this_frame << "  " << double(detection_data.frame_id - this_frame) << std::endl;
                                                    //time_t rawtime;
                                                    //struct tm*timeinfo;
                                                    //time(&rawtime);
                                                    //timeinfo = gmtime(&rawtime);
                                                    //std::string strtime = asctime(timeinfo);
                                                    //char buf[sizeof "2019-12-09T19:49:39Z"];
                                                    //strftime(buf, sizeof buf, "%FT%TZ", timeinfo);

                                                    //std::cout << buf;
                                                    //std::string bufString(buf);
                                                    //std::string buffer = "\"" + bufString + "\"";
                                                    // std::string jsn = "{\"phenomenonTime\": " + buffer + ",\"resultTime\" : " + buffer + ",\"result\" :\""+ std::to_string(int(speedo))+ "\", \"FeatureOfInterest\": {\"@iot.id\": \"28\"}" + "}";
                                                    //send2_compusult(jsn,speed_query);                                   
                                                    std::cout << "speed:" + speedtxt + "\n";
                                                    // std::string jsn = "{\"result\" :\"" + speedtxt + "\", \"FeatureOfInterest\": {\"@iot.id\": \"28\"}" + "}";
                                                    std::string speed_querystr = "/v1.0/Datastreams(1630)/Observations";
                                                    const char* speed_query = speed_querystr.c_str();
                                                    // send2_compusult(jsn, speed_querystr, speedtxt);
                                                }
                                                }
                                            }
                                            // v contains x 
                                        }
                                    }
                                }
                            }
                        }
                        std::cout << "car:" + std::to_string(car_count) + "\n";
                        std::cout << "person:" + std::to_string(person_count) + "\n";

                        auto t1 = Time::now();
                        fsec fs_c = t1 - t0c; // number of cars
                        fsec fs_s = t1 - t0s; // speed
                        ms d_c = std::chrono::duration_cast<ms>(fs_c);
                        ms d_s = std::chrono::duration_cast<ms>(fs_s);

                        car_arr.push_back((double)car_count);
                        pers_arr.push_back((double)person_count);

                        //if ((int)floor(time_diff) % 6000==0)
                        if (d_c.count() > 5000)
                        {
                            t0c = t1;
                            double car_sum = 0;
                            for (int c = 0; c < (double)car_arr.size(); c++)
                            {
                                car_sum += car_arr[c];
                            }
                            if (car_arr.size() > 0)
                            {
                                int car_average = (int)floor(car_sum / (double)car_arr.size());
                                //std::string jsn_car = "{\"result\" :\"" + std::to_string(car_average) + "\", \"FeatureOfInterest\": {\"@iot.id\": \"5321\"}" + "}";
                                std::string car_querystr = "/v1.0/Datastreams(5318)/Observations";
                                send2_compusult("", car_querystr, std::to_string(car_average));
                            }
                            double pers_sum = 0;
                            for (int p = 0; p < (double)pers_arr.size(); p++)
                            {
                                pers_sum += pers_arr[p];
                            }
                            if (pers_arr.size() > 0)
                            {
                                int pers_average = (int)floor(pers_sum / (double)pers_arr.size());
                                std::string jsn_pers = "{\"result\" :\"" + std::to_string(pers_average) + "\", \"FeatureOfInterest\": {\"@iot.id\": \"5321\"}" + "}";
                                std::string pers_querystr = "/v1.0/Datastreams(5322)/Observations";
                                send2_compusult(jsn_pers, pers_querystr, std::to_string(pers_average));
                            }
                            car_arr.clear();
                            pers_arr.clear();
                        }
                        if (d_s.count() > 10000)
                        {
                            t0s = t1;
                            double speed_sum = 0;
                            for (int s = 0; s < (double)speed_arr.size(); s++)
                            {
                                speed_sum += speed_arr[s];
                            }
                            if (speed_arr.size() > 0)
                            {
                                int speed_average = (int)floor(speed_sum / (double)speed_arr.size());
                                std::cout << speed_sum << ", " << speed_arr.size() << ", " << floor(speed_sum / speed_arr.size()) << std::endl;
                                std::string jsn = "{\"result\" :\"" + std::to_string(speed_average) + "\", \"FeatureOfInterest\": {\"@iot.id\": \"5321\"}" + "}";
                                std::string speed_querystr = "/v1.0/Datastreams(5323)/Observations";
                                send2_compusult(jsn, speed_querystr, std::to_string(speed_average));
                                speed_arr.clear();
                            }
                        }
                        person_count = 0;
                        car_count = 0;
                        // cv::imshow("window name", mat_img);
                        previous_box = detection_data.result_vec;
                        //cv::rectangle(draw_frame, cv::Rect(20, 20, 100, 20), cv::Scalar(255, 0, 0), 2);
                        cv::putText(draw_frame, speedtxt, cv::Point(25, 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 3, cv::Scalar(255, 255, 255), 1);
                        cv::line(draw_frame, cv::Point(0, frame_size.height / 2), cv::Point(frame_size.width, frame_size.height / 2), cv::Scalar(255, 0, 0), 2, 8);

                    }
                    else if (app_mode == "f")
                    {
                        for (auto &i : detection_data.result_vec) {
                            if (i.obj_id == 0) {
                                low_water_count++;
                            }
                            else if (i.obj_id == 1) {
                                high_water_count++;
                            }
                            else if (i.obj_id == 2) {
                                no_water_count++;
                            }
                            else if (i.obj_id == 3) {
                                splash_water_count++;
                                low_water_count ++;
                                high_water_count++;
                            }
                        }
                        if ((double)detection_data.result_vec.size() > 0)
                        {
                            flood_frame_counter++;
                        }
                        if (flood_frame_counter % 150==0 && flood_frame_counter>0)
                        {
                            int vals[3] = { high_water_count,low_water_count, no_water_count   };
                           int indx= find_maximum(vals);
                           std::string flood_val;
                           if (indx == 0)
                               flood_val = "high";
                           else if (indx == 1)
                               flood_val = "low";
                           else if (indx == 2)
                               flood_val = "no flood";
                           std::cout << "jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj" << std::endl;
                           std::string jsn_flood = "{\"result\" :\"" + flood_val + "\", \"FeatureOfInterest\": {\"@iot.id\": \"9978\"}" + "}";
                           std::string flood_querystr = "/v1.0/Datastreams(9883)/Observations";
                           send2_compusult(jsn_flood, flood_querystr, flood_val);
                           high_water_count = 0;
                           low_water_count = 0;
                           splash_water_count = 0;
                           no_water_count = 0;
                        }
                    }
                    
                    /*
                    std::vector<uchar> bufr;
                    imencode(".jpg", draw_frame, bufr);
                    uchar *enc_msg = new uchar[bufr.size()];
                    for(int i=0; i< bufr.size(); i++) enc_msg[i] = bufr[i];
                    std::string encoded = base64_encode(enc_msg, bufr.size());
                    */


                    //std::cout<<encoded;
                    // show_console_result2(detection_data.result_vec, obj_names,detection_data.frame_id);                    
                    cv::resize(draw_frame, draw_frame, cv::Size(1280, 960));
                    cv::imshow("window name", draw_frame);
                     //if (detection_data.frame_id % 1000 == 0) {
                      //   cv::imwrite(std::to_string(detection_data.frame_id % 1000)+".jpg",draw_frame); //check				
                     //}

                    int key = cv::waitKey(3);    // 3 or 16ms
                    if (key == 'f') show_small_boxes = !show_small_boxes;
                    if (key == 'p') while (true) if (cv::waitKey(100) == 'p') break;
                    //if (key == 'e') extrapolate_flag = !extrapolate_flag;
                    if (key == 27) { exit_flag = true; }

                    //std::cout << " current_fps_det = " << current_fps_det << ", current_fps_cap = " << current_fps_cap << std::endl;
                } while (!detection_data.exit_flag);

                std::cout << " show detection exit \n";

                cv::destroyWindow("window name");
                // wait for all threads
                if (t_cap.joinable()) t_cap.join();
                if (t_prepare.joinable()) t_prepare.join();
                if (t_detect.joinable()) t_detect.join();
                if (t_post.joinable()) t_post.join();
                if (t_draw.joinable()) t_draw.join();
                if (t_write.joinable()) t_write.join();
                if (t_network.joinable()) t_network.join();

                break;

            }
            else if (file_ext == "txt") {    // list of image files
                std::ifstream file(filename);
                if (!file.is_open()) std::cout << "File not found! \n";
                else
                    for (std::string line; file >> line;) {
                        std::cout << line << std::endl;
                        cv::Mat mat_img = cv::imread(line);
                        std::vector<bbox_t> result_vec = detector.detect(mat_img);
                        show_console_result(result_vec, obj_names);
                        //draw_boxes(mat_img, result_vec, obj_names);
                        //cv::imwrite("res_" + line, mat_img);
                    }

            }
            else {    // image file
                // to achive high performance for multiple images do these 2 lines in another thread
                cv::Mat mat_img = cv::imread(filename);
                auto det_image = detector.mat_to_image_resize(mat_img);

                auto start = std::chrono::steady_clock::now();
                std::vector<bbox_t> result_vec = detector.detect_resized(*det_image, mat_img.size().width, mat_img.size().height);
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double> spent = end - start;
                std::cout << " Time: " << spent.count() << " sec \n";

                //result_vec = detector.tracking_id(result_vec);    // comment it - if track_id is not required

                frame_count++;
                cv::waitKey(0);
            }
#else   // OPENCV
            //std::vector<bbox_t> result_vec = detector.detect(filename);

            auto img = detector.load_image(filename);
            std::vector<bbox_t> result_vec = detector.detect(img);
            detector.free_image(img);
            show_console_result(result_vec, obj_names);
#endif  // OPENCV
        }
        catch (std::exception &e) { std::cerr << "exception: " << e.what() << "\n"; getchar(); }
        catch (...) { std::cerr << "unknown exception \n"; getchar(); }
        filename.clear();
    }

    return 0;
}
