#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime/core/providers/dml/dml_provider_factory.h>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime/core/providers/tensor/onnxruntime_tensor.h>
#include <std_msgs/String.h>
#include <vector>

class GestureDetector {
public:
    GestureDetector(ros::NodeHandle& nh) {
        // Initialize ROS node and subscribe to camera topic
        image_sub = nh.subscribe("/uav2/rgbd/infra1/image_raw", 1, &GestureDetector::imageCallback, this);
        gesture_pub = nh.advertise<std_msgs::String>("/uav2/gesture_detector/detection", 1);

        // Initialize ONNX Runtime session for inference
        try {
            // Create an ONNX Runtime session
            onnx::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(onnxruntime::GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            session = std::make_shared<onnxruntime::InferenceSession>(session_options);
            session->Load("../models/model_float16.onnx");  // Path to the ONNX model file

            ROS_INFO("[GestureDetector - Detect]: ONNX model loaded successfully!");
        } catch (const std::exception& e) {
            ROS_ERROR("Failed to load ONNX model: %s", e.what());
            ros::shutdown();
        }

        ROS_INFO("[GestureDetector - Detect]: initialized!");
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
        // Convert ROS Image message to OpenCV image
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Prepare the image for the ONNX model (resize, normalization)
        cv::Mat input_image;
        cv::resize(cv_ptr->image, input_image, cv::Size(416, 416));  // Resize to model input size
        input_image.convertTo(input_image, CV_32F, 1.0 / 255.0);  // Normalize to [0, 1]

        // Prepare the input tensor for ONNX model
        std::vector<int64_t> input_dims = {1, 3, 640, 640};  // Input shape [1, 3, height, width]
        std::vector<float> input_data(input_image.begin<float>(), input_image.end<float>());
        onnxruntime::Tensor input_tensor(onnxruntime::DataTypeImpl::GetTensorType<float>(), input_dims, input_data.data());

        // Run the model
        std::vector<onnxruntime::Tensor> outputs;
        session->Run(onnxruntime::RunOptions(), {"input"}, {&input_tensor}, {"output"}, outputs);

        // Post-process the detection results
        bool detection = false;
        std::string gesture;
        // Example: Assuming output tensor contains detection results
        for (const auto& output : outputs) {
            auto* data = output.Data<float>();
            size_t num_results = output.Shape().Size();
            for (size_t i = 0; i < num_results; ++i) {
                // Extract the results and map them to gestures (class IDs, confidence)
                float confidence = data[i * 6 + 4];  // Assuming class confidence is at index 4
                if (confidence > 0.7) {
                    int class_id = static_cast<int>(data[i * 6 + 5]);  // Example: class ID at index 5
                    detection = true;
                    switch (class_id) {
                        case 0: gesture = "land"; break;
                        case 1: gesture = "move_ahead"; break;
                        case 2: gesture = "move_back"; break;
                        case 3: gesture = "yaw_to_left"; break;
                        case 4: gesture = "yaw_to_right"; break;
                        case 5: gesture = "move_to_left"; break;
                        case 6: gesture = "move_to_right"; break;
                        case 7: gesture = "takeoff"; break;
                        case 8: gesture = "move_up"; break;
                        case 9: gesture = "move_down"; break;
                    }
                }
            }
        }

        // Publish the detected gesture if found
        if (detection) {
            std_msgs::String msg;
            msg.data = gesture;
            gesture_pub.publish(msg);
        }
    }

private:
    ros::Subscriber image_sub;
    ros::Publisher gesture_pub;
    std::shared_ptr<onnxruntime::InferenceSession> session;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "gesture_detector");
    ros::NodeHandle nh;

    GestureDetector detector(nh);
    ros::spin();

    return 0;
}
