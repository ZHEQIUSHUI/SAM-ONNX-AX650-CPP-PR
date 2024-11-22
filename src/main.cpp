#include <opencv2/opencv.hpp>
#include <axcl.h>
#include "cmdline.hpp"
#include "Runner/SAM.hpp"

int main(int argc, char *argv[])
{
    std::string image_path = "./test.jpg";
    std::string encoder_model_path = "./onnx_models/mobile_sam_encoder.onnx";
    std::string decoder_model_path = "./onnx_models/mobile_sam_decoder.onnx";

    cmdline::parser cmd;
    cmd.add<std::string>("encoder", 'e', "encoder model(onnx model or axmodel)", true, encoder_model_path);
    cmd.add<std::string>("decoder", 'd', "decoder model(onnx)", true, decoder_model_path);
    cmd.add<std::string>("image", 'i', "image file(jpg png etc....)", true, image_path);

    cmd.parse_check(argc, argv);

    image_path = cmd.get<std::string>("image");
    encoder_model_path = cmd.get<std::string>("encoder");
    decoder_model_path = cmd.get<std::string>("decoder");

    // init axcl
    {
        if (auto ret = axclInit(0); 0 != ret)
        {
            fprintf(stderr, "Init AXCL failed{0x%8x}.\n", ret);
            return -1;
        }
        axclrtDeviceList lst;
        if (const auto ret = axclrtGetDeviceList(&lst); 0 != ret || 0 == lst.num)
        {
            fprintf(stderr, "Get AXCL device failed{0x%8x}, find total %d device.\n", ret, lst.num);
            return -1;
        }
        if (const auto ret = axclrtSetDevice(lst.devices[0]); 0 != ret)
        {
            fprintf(stderr, "Set AXCL device failed{0x%8x}.\n", ret);
            return -1;
        }
        int ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
        if (0 != ret)
        {
            fprintf(stderr, "axclrtEngineInit %d\n", ret);
            return ret;
        }
    }

    cv::Mat src = cv::imread(image_path);
    SAM sam;
    sam.Load(encoder_model_path, decoder_model_path);
    sam.Encode(src);

    cv::Point pointinfo(910, 641);
    cv::Rect boxinfo(pointinfo.x - 160, pointinfo.y - 430, 380, 940);

    auto outputs = sam.Decode(pointinfo);

    for (size_t i = 0; i < outputs.size(); i++)
    {
        printf("%2.2f \n", outputs[i].iou_pred);
        cv::cvtColor(outputs[i].mask, outputs[i].mask, cv::COLOR_GRAY2BGR);
        cv::circle(outputs[i].mask, pointinfo, 2, cv::Scalar(0, 0, 255), 4);
        cv::imwrite("result_" + std::to_string(i) + ".jpg", outputs[i].mask);
    }

    outputs = sam.Decode(boxinfo);

    for (size_t i = 0; i < outputs.size(); i++)
    {
        printf("%2.2f \n", outputs[i].iou_pred);
        cv::cvtColor(outputs[i].mask, outputs[i].mask, cv::COLOR_GRAY2BGR);
        cv::rectangle(outputs[i].mask, boxinfo, cv::Scalar(0, 0, 255), 4);
        cv::imwrite("rect_result_" + std::to_string(i) + ".jpg", outputs[i].mask);
    }
    
    axclFinalize();
    return EXIT_SUCCESS;
}
