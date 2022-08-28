// 重要参考： https://blog.csdn.net/m0_55432927/article/details/121952068s 
// 
#ifndef DEEPSORT_HPP
#define DEEPSORT_HPP

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>

namespace DeepSORT
{

    struct Box
    {
        float left, top, right, bottom;
        cv::Mat feature;

        Box() = default;
        Box(float left, float top, float right, float bottom) : left(left), top(top), right(right), bottom(bottom) {}
        const float width() const { return right - left; }
        const float height() const { return bottom - top; }
        const cv::Point2f center() const { return cv::Point2f((left + right) / 2, (top + bottom) / 2); }
    };

    template <typename _T>
    inline Box convert_to_box(const _T &b)
    {
        return Box(b.left, b.top, b.right, b.bottom);
    }

    template <typename _T>
    inline cv::Rect convert_box_to_rect(const _T &b)
    {
        return cv::Rect(b.left, b.top, b.right - b.left, b.bottom - b.top);
    }

    enum class State : int
    {
        Tentative = 1, // 不确定的
        Confirmed = 2,
        Deleted = 3
    };

    struct TrackerConfig
    {

        int max_age = 150; // 在侦测状态(Confirmed)设置成Deleted前，最大的连续miss数
        int nhit = 3;      // 确认轨迹前的连续检测次数。
        float distance_threshold = 100;  // 马氏距离阈值
        int nbuckets = 0;  // 最大保存特征帧数，超过该帧数进行滚动保存
        bool has_feature = false;

        // kalman
        // /** 初始状态 **/
        float initiate_state[8]; // 初始状态分布的平均向量?

        // /** 每一侦的运动量协方差，下一侦 = 当前帧 + 运动量 **/
        float per_frame_motion[8];

        // /** 测量噪声，把输入映射到测量空间中后的噪声 **/
        float noise[4];

        void set_initiate_state(const std::vector<float> &values);
        void set_per_frame_motion(const std::vector<float> &values);
        void set_noise(const std::vector<float> &values);

        TrackerConfig();
    };

    typedef std::vector<Box> BBoxes;

    class TrackObject
    {
    public:
        virtual int id() const = 0;
        virtual State state() const = 0;
        virtual Box predict_box() const = 0;
        virtual Box last_position() const = 0;
        virtual bool is_confirmed() const = 0;
        virtual int time_since_update() const = 0;
        virtual std::vector<cv::Point> trace_line() const = 0;
        virtual int trace_size() const = 0;
        virtual Box &location(int time_since_update = 0) = 0;
        virtual const cv::Mat &feature_bucket() const = 0;
    };

    class Tracker
    {
    public:
        virtual std::vector<TrackObject *> get_objects() = 0;
        virtual void update(const BBoxes &boxes) = 0;
    };

    std::shared_ptr<Tracker> create_tracker(
        const TrackerConfig &config = TrackerConfig());

}

#endif // DEEPSORT_HPP