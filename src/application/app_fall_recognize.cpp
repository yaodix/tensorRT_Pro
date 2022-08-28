
#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>

#include "app_yolo/yolo.hpp"
#include "app_alphapose_old/alpha_pose_old.hpp"
#include "app_fall_gcn/fall_gcn.hpp"
#include "tools/zmq_remote_show.hpp"
#include "tools/deepsort.hpp"

using namespace cv;
using namespace std;

bool requires(const char* name);


int app_fall_recognize(){
    cv::setNumThreads(0);

    INFO("===================== test alphapose FP32 ==================================");
    
    auto pose_model_file     = "/home/yao/workspace/tensorRT_Pro/weights/sppe.trt";  // singal person pose estimation
    auto detector_model_file = "/home/yao/workspace/tensorRT_Pro/weights/yolov5m.trt";
    auto gcn_model_file      = "/home/yao/workspace/tensorRT_Pro/weights/fall_bp.trt";  // 跌倒，图神经网络分类模型
    
    auto pose_model     = AlphaPoseOld::create_infer(pose_model_file, 0);
    auto detector_model = Yolo::create_infer(detector_model_file, Yolo::Type::V5, 0, 0.4f, 0.4f);
    auto gcn_model      = FallGCN::create_infer(gcn_model_file, 0);

    Mat image;
    Mat det_image;
    VideoCapture cap("/home/yao/workspace/tensorRT_Pro/workspace/exp/basketball.mp4");
    INFO("Video fps=%d, Width=%d, Height=%d", 
        (int)cap.get(cv::CAP_PROP_FPS), 
        (int)cap.get(cv::CAP_PROP_FRAME_WIDTH), 
        (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)
    );

    //auto remote_show = create_zmq_remote_show();
    INFO("Use tools/show.py to remote show");

    auto config  = DeepSORT::TrackerConfig();
    config.set_initiate_state({
        0.1,  0.1,  0.1,  0.1,
        0.2,  0.2,  1,    0.2
    });

    config.set_per_frame_motion({
        0.1,  0.1,  0.1,  0.1,
        0.2,  0.2,  1,    0.2
    });

    auto tracker = DeepSORT::create_tracker(config);
    // VideoWriter writer("fall_video.result.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 
    //     30,
    //     Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT))
    // );
    // if(!writer.isOpened()){
    //     INFOE("Writer failed.");
    //     return 0;
    // }
    int frame_id = 0;
    while(cap.read(image)){
        std::cout << "frame " << frame_id++ << std::endl;
        det_image = image.clone();
        auto objects = detector_model->commit(image).get();

        vector<DeepSORT::Box> boxes;
        for(int i = 0; i < objects.size(); ++i){
            auto& obj = objects[i];
            if(obj.class_label != 0) continue;

            rectangle(det_image, DeepSORT::convert_box_to_rect(obj), Scalar(0, 255, 0), 2);  // 检测框输出
            imwrite("det.jpg", det_image);

            boxes.emplace_back(std::move(DeepSORT::convert_to_box(obj)));
        }
        tracker->update(boxes);

        auto final_objects = tracker->get_objects();
        for(int i = 0; i < final_objects.size(); ++i){
            auto& person = final_objects[i];
            if(person->time_since_update() == 0 && person->state() == DeepSORT::State::Confirmed){
                Rect box = DeepSORT::convert_box_to_rect(person->last_position());
                auto keys   = pose_model->commit(make_tuple(image, box)).get();
                auto statev = gcn_model->commit(make_tuple(keys, box)).get();

                FallGCN::FallState state = get<0>(statev);
                float confidence         = get<1>(statev);
                const char* label_name   = FallGCN::state_name(state);
                rectangle(image, DeepSORT::convert_box_to_rect(person->predict_box()), Scalar(0, 255, 0), 1);
                rectangle(image, box, Scalar(0, 255, 255), 1);

                auto line = person->trace_line();
                for(int j = 0; j < (int)line.size() - 1; ++j){
                    auto& p = line[j];
                    auto& np = line[j + 1];
                    cv::line(image, p, np, Scalar(255, 128, 60), 2, 16);
                }

                // putText(image, iLogger::format("%d. [%s] %.2f %%", person->id(), label_name, confidence * 100), box.tl(), 0, 1, Scalar(0, 255, 0), 2, 16);
                putText(image, iLogger::format("%d", person->id()), box.tl(), 0, 1, Scalar(0, 255, 0), 2, 16);
                //INFO("Predict is [%s], %.2f %%", label_name, confidence * 100);
           }
        }
        //remote_show->post(image);
        //writer.write(image);
        // cv::imwrite("image_debug.jpg", image);
        cv::imshow("hello", image);
        cv::waitKey(5);
    }
    INFO("Done");
    return 0;
}