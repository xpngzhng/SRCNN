#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

struct ConvLayerParams
{
    void clear()
    {
        weights.clear();
        biases.clear();
    }

    void fill(const cv::Ptr<cv::dnn::Layer>& layer)
    {
        CV_Assert(layer->blobs.size() == 2);

        CV_Assert(layer->blobs[0].dims == 4);
        numFilters = layer->blobs[0].size[0];
        numChannels = layer->blobs[0].size[1];
        height = layer->blobs[0].size[2];
        width = layer->blobs[0].size[3];
        weights.resize(numFilters);
        for (int i = 0; i < numFilters; i++)
        {
            weights[i].resize(numChannels);
            for (int j = 0; j < numChannels; j++)
                weights[i][j] = cv::Mat(height, width, CV_32FC1, layer->blobs[0].ptr<float>(i, j));
        }

        CV_Assert(layer->blobs[1].dims == 2);
        CV_Assert(layer->blobs[1].size[0] == numFilters);
        biases.resize(numFilters);
        for (int i = 0; i < numFilters; i++)
        {
            biases[i] = layer->blobs[1].at<float>(i);
        }
    }

    void conv(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst) const
    {
        CV_Assert(src.size() == numChannels);
        int rows = src[0].rows, cols = src[0].cols;
        dst.resize(numFilters);
        cv::Mat temp(rows, cols, CV_32FC1);
        for (int i = 0; i < numFilters; i++)
        {
            dst[i].create(rows, cols, CV_32FC1);
            dst[i].setTo(0);
            for (int j = 0; j < numChannels; j++)
            {
                cv::filter2D(src[j], temp, CV_32F, weights[i][j], cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
                cv::add(temp, dst[i], dst[i]);
            }
            cv::add(dst[i], biases[i], dst[i]);
        }
    }

    std::vector<std::vector<cv::Mat> > weights;
    std::vector<float> biases;
    int numFilters, numChannels, height, width;
};

void relu(std::vector<cv::Mat>& arr)
{
    int num = (int)arr.size();
    for (int i = 0; i < num; i++)
        cv::max(arr[i], 0, arr[i]);
}

int main()
{
    std::string modelTxt = "E:\\Projects\\SRCNN\\Train\\model\\91-images-9-1-5\\SRCNN_mat.prototxt";
    std::string modelBin = "E:\\Projects\\SRCNN\\Train\\model\\91-images-9-1-5\\x3\\snapshot_iter_3000000.caffemodel";

    cv::dnn::Net net = cv::dnn::readNetFromCaffe(modelTxt, modelBin);
    int numFilters, numChannels, height, width;

    cv::Ptr<cv::dnn::Layer> conv1Layer = net.getLayer(cv::dnn::DictValue("conv1"));
    ConvLayerParams conv1Params;
    conv1Params.fill(conv1Layer);

    cv::Ptr<cv::dnn::Layer> conv2Layer = net.getLayer(cv::dnn::DictValue("conv2"));
    ConvLayerParams conv2Params;
    conv2Params.fill(conv2Layer);

    cv::Ptr<cv::dnn::Layer> conv3Layer = net.getLayer(cv::dnn::DictValue("conv3"));
    ConvLayerParams conv3Params;
    conv3Params.fill(conv3Layer);

    cv::Mat orig = cv::imread("space_shuttle.jpg"), temp;
    cv::Mat bigBicubic;
    cv::resize(orig, temp, cv::Size(), 1.0 / 3, 1.0 / 3, cv::INTER_CUBIC);
    cv::resize(temp, bigBicubic, orig.size(), 0, 0, cv::INTER_CUBIC);

    cv::Mat bigBicubic32F;
    bigBicubic.convertTo(bigBicubic32F, CV_32F);

    std::vector<cv::Mat> src32F;
    cv::split(bigBicubic32F, src32F);

    std::vector<cv::Mat> dst32F(src32F.size());
    std::vector<cv::Mat> dst1, dst2, dst3;

    for (int i = 0, size = (int)src32F.size(); i < size; i++)
    {
        conv1Params.conv(std::vector<cv::Mat>{ src32F[i] }, dst1);
        relu(dst1);
        conv2Params.conv(dst1, dst2);
        relu(dst2);
        conv3Params.conv(dst2, dst3);
        dst32F[i] = dst3[0].clone();
    }

    cv::Mat bigSR32F, bigSR;
    cv::merge(dst32F, bigSR32F);
    bigSR32F.convertTo(bigSR, CV_8U);

    cv::imshow("bicubic", bigBicubic);
    cv::imshow("sr", bigSR);
    cv::waitKey(0);
    
    return 0;
}