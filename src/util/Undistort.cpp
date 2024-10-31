/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#include <fstream>
#include <iostream>
#include <sstream>

#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/ImageRW.h"
#include "util/Undistort.h"
#include "util/globalFuncs.h"
#include "util/settings.h"
#include <Eigen/Core>
#include <iterator>

namespace dso {

/**
 * @brief PhotometricUndistorter的构造函数
 * @details
 *  1. 读取G函数和V函数
 *  2. 使用区间初始化的方式，将G函数进行归一化，到[0, 255]区间
 *  3. 使用最大值归一化，将V函数进行归一化，到[0,1]区间
 * @param file          输入的G函数的文件路径
 * @param noiseImage    输入的噪声图像路径，一般为空字符串
 * @param vignetteImage 输入的晕影图像路径，即V函数
 * @param w_            输入的原始图像的宽度
 * @param h_            输入的原始图像的高度
 */
PhotometricUndistorter::PhotometricUndistorter(std::string file, std::string noiseImage, std::string vignetteImage,
                                               int w_, int h_) {
    valid = false;
    vignetteMap = 0;
    vignetteMapInv = 0;
    w = w_;
    h = h_;
    output = new ImageAndExposure(w, h);
    if (file == "" || vignetteImage == "") {
        printf("NO PHOTOMETRIC Calibration!\n");
    }

    // read G.
    std::ifstream f(file.c_str());
    printf("Reading Photometric Calibration from file %s\n", file.c_str());
    if (!f.good()) {
        printf("PhotometricUndistorter: Could not open file!\n");
        return;
    }

    // 得到响应函数的逆变换
    {
        std::string line;
        std::getline(f, line);
        std::istringstream l1i(line);
        // begin迭代器, end迭代器来初始化
        std::vector<float> Gvec = std::vector<float>(std::istream_iterator<float>(l1i), std::istream_iterator<float>());

        GDepth = Gvec.size();

        if (GDepth < 256) {
            printf("PhotometricUndistorter: invalid format! got %d entries in first line, expected at least 256!\n",
                   (int)Gvec.size());
            return;
        }

        for (int i = 0; i < GDepth; i++)
            G[i] = Gvec[i];

        /// 判断G函数是否是单调递增
        for (int i = 0; i < GDepth - 1; i++) {
            if (G[i + 1] <= G[i]) {
                printf("PhotometricUndistorter: G invalid! it has to be strictly increasing, but it isnt!\n");
                return;
            }
        }

        // 对响应值进行标准化，这里的标准化的区间为[0, 255]
        float min = G[0];
        float max = G[GDepth - 1];
        for (int i = 0; i < GDepth; i++)
            G[i] = 255.0 * (G[i] - min) / (max - min);
    }

    //! 如果没有标定值, 这里的G应该为空，因为GDepth在初始化时，应该为0
    if (setting_photometricCalibration == 0) {
        for (int i = 0; i < GDepth; i++)
            G[i] = 255.0f * i / (float)(GDepth - 1);
    }

    /// 猜测这里先读16位再读8位的目的，可能是因为影晕图像的像素值表示可能是16位的，也可能是8位的
    printf("Reading Vignette Image from %s\n", vignetteImage.c_str());
    MinimalImage<unsigned short> *vm16 = IOWrap::readImageBW_16U(vignetteImage.c_str());
    MinimalImageB *vm8 = IOWrap::readImageBW_8U(vignetteImage.c_str());
    vignetteMap = new float[w * h];
    vignetteMapInv = new float[w * h];

    /// 进行归一化操作，即除以整个图像中的最大值，变换到0-1之间
    if (vm16 != 0) {
        if (vm16->w != w || vm16->h != h) {
            printf("PhotometricUndistorter: Invalid vignette image size! got %d x %d, expected %d x %d\n", vm16->w,
                   vm16->h, w, h);
            if (vm16 != 0)
                delete vm16;
            if (vm8 != 0)
                delete vm8;
            return;
        }
        // 使用最大值来归一化
        float maxV = 0;
        for (int i = 0; i < w * h; i++)
            if (vm16->at(i) > maxV)
                maxV = vm16->at(i);

        for (int i = 0; i < w * h; i++)
            vignetteMap[i] = vm16->at(i) / maxV;
    } else if (vm8 != 0) {
        if (vm8->w != w || vm8->h != h) {
            printf("PhotometricUndistorter: Invalid vignette image size! got %d x %d, expected %d x %d\n", vm8->w,
                   vm8->h, w, h);
            if (vm16 != 0)
                delete vm16;
            if (vm8 != 0)
                delete vm8;
            return;
        }

        float maxV = 0;
        for (int i = 0; i < w * h; i++)
            if (vm8->at(i) > maxV)
                maxV = vm8->at(i);

        for (int i = 0; i < w * h; i++)
            vignetteMap[i] = vm8->at(i) / maxV;
    } else {
        printf("PhotometricUndistorter: Invalid vignette image\n");
        if (vm16 != 0)
            delete vm16;
        if (vm8 != 0)
            delete vm8;
        return;
    }

    if (vm16 != 0)
        delete vm16;
    if (vm8 != 0)
        delete vm8;

    // 求逆
    for (int i = 0; i < w * h; i++)
        vignetteMapInv[i] = 1.0f / vignetteMap[i];

    printf("Successfully read photometric calibration!\n");
    valid = true;
}
PhotometricUndistorter::~PhotometricUndistorter() {
    if (vignetteMap != 0)
        delete[] vignetteMap;
    if (vignetteMapInv != 0)
        delete[] vignetteMapInv;
    delete output;
}

//@ 给图像加上响应函数
void PhotometricUndistorter::unMapFloatImage(float *image) {
    int wh = w * h;
    for (int i = 0; i < wh; i++) {
        float BinvC;
        float color = image[i];

        if (color < 1e-3) // 小置零
            BinvC = 0.0f;
        else if (color > GDepth - 1.01f) // 大最大值
            BinvC = GDepth - 1.1;
        else // 中间对响应函数插值
        {
            int c = color;
            float a = color - c;
            BinvC = G[c] * (1 - a) + G[c + 1] * a;
        }

        float val = BinvC;
        if (val < 0)
            val = 0;
        image[i] = val;
    }
}

/**
 * @brief 去除光度畸变的部分，或者说是光度矫正的部分
 *
 * @tparam T
 * @param image_in 		输入的原始带有畸变参数的图像
 * @param exposure_time 输入的指定的曝光时间
 * @param factor 		光度模型不存在，则输出变为图像灰度和factor的乘积
 */
template <typename T> void PhotometricUndistorter::processFrame(T *image_in, float exposure_time, float factor) {
    int wh = w * h;
    float *data = output->image;
    assert(output->w == w && output->h == h);
    assert(data != 0);

    // 当没有光度模型的时候，会将image_in和factor乘到一起，放入output中
    if (!valid || exposure_time <= 0 || setting_photometricCalibration == 0) // disable full photometric calibration.
    {
        for (int i = 0; i < wh; i++) {
            data[i] = factor * image_in[i];
        }
        output->exposure_time = exposure_time;
        output->timestamp = 0;
    }
    // 当有广度模型时，还需要对是否仅去除函数G还是G和V同时去除
    else {
        /// 去掉响应函数G
        for (int i = 0; i < wh; i++) {
            data[i] = G[image_in[i]]; // 去掉响应函数
        }

        /// 去掉晕影V
        if (setting_photometricCalibration == 2) // 去掉衰减系数
        {
            for (int i = 0; i < wh; i++)
                data[i] *= vignetteMapInv[i];
        }

        /// 最后只剩下能量单位B * t
        output->exposure_time = exposure_time; // 设置曝光时间
        output->timestamp = 0;
    }

    /// 针对不设置Exposure的部分，需要将曝光参数同步设置为1
    if (!setting_useExposure)
        output->exposure_time = 1;
}
// 模板特殊化, 指定两个类型
template void PhotometricUndistorter::processFrame<unsigned char>(unsigned char *image_in, float exposure_time,
                                                                  float factor);
template void PhotometricUndistorter::processFrame<unsigned short>(unsigned short *image_in, float exposure_time,
                                                                   float factor);

//******************************** 矫正基类, 包括几何和光度 ************************************

Undistort::~Undistort() {
    if (remapX != 0)
        delete[] remapX;
    if (remapY != 0)
        delete[] remapY;
}

/**
 * @brief Undistort类型的工厂函数，用于构建Undistort类型，主要有RadTan、Pinhole、Fov、KannalaBrandt和EquiDistant 5种
 * @details 根据configFilename 文件第一行的畸变参数来判断畸变模型
 *  1. 仅8个float参数，判断为RadTan畸变模型
 *  2. 仅5个float参数，判断为Pinhole或Fov畸变模型
 *      1) 当最后一个参数为0时，判断为Pinhole畸变模型
 *      2) 当最后一个参数不为0时，判断为Fov畸变模型
 *  3. 以KannalaBrandt开头，并存在8个float时，判断为KannalaBrandt畸变模型
 *  4. 以RadTan开头，并存在8个float时，判断为RadTan畸变模型
 *  5. 以EquiDistant开头，并存在8个float时，判断为EquiDistant畸变模型
 *  6. 以FOV开头，并存在5个float时，判断为FOV畸变模型
 *  7. 以Pinhole开头，并存在5个float时，判断为Pinhole畸变模型
 * @param configFilename    输入的相机内参文件路径
 * @param gammaFilename     输入的光度映射函数文件路径
 * @param vignetteFilename  输入的晕影文件路径（和数据图片存在一个一比一的图像，用来标识相机晕影）
 * @return Undistort*       输出畸变指针
 */
Undistort *Undistort::getUndistorterForFile(std::string configFilename, std::string gammaFilename,
                                            std::string vignetteFilename) {
    printf("Reading Calibration from file %s", configFilename.c_str());

    std::ifstream f(configFilename.c_str());
    if (!f.good()) {
        f.close();
        printf(" ... not found. Cannot operate without calibration, shutting down.\n");
        f.close();
        return 0;
    }

    printf(" ... found!\n");
    std::string l1;
    std::getline(f, l1);
    f.close();

    float ic[10];

    Undistort *u; // 矫正基类, 作为返回值, 其他的类型继承自它

    //* 下面三种具体模型, 是针对没有指明模型名字的, 只给了参数
    // for backwards-compatibility: Use RadTan model for 8 parameters.
    if (std::sscanf(l1.c_str(), "%f %f %f %f %f %f %f %f", &ic[0], &ic[1], &ic[2], &ic[3], &ic[4], &ic[5], &ic[6],
                    &ic[7]) == 8) {
        printf("found RadTan (OpenCV) camera model, building rectifier.\n");
        u = new UndistortRadTan(configFilename.c_str(), true);
        if (!u->isValid()) {
            delete u;
            return 0;
        }
    }

    // for backwards-compatibility: Use Pinhole / FoV model for 5 parameter.
    else if (std::sscanf(l1.c_str(), "%f %f %f %f %f", &ic[0], &ic[1], &ic[2], &ic[3], &ic[4]) == 5) {
        if (ic[4] == 0) // 没有FOV的畸变参数, 只有pinhole
        {
            printf("found PINHOLE camera model, building rectifier.\n");
            u = new UndistortPinhole(configFilename.c_str(), true);
            if (!u->isValid()) {
                delete u;
                return 0;
            }
        } else // pinhole + FOV , atan
        {
            printf("found ATAN camera model, building rectifier.\n");
            u = new UndistortFOV(configFilename.c_str(), true);
            if (!u->isValid()) {
                delete u;
                return 0;
            }
        }
    }

    //* 以下是指明了相机模型的几种选择
    // clean model selection implementation.
    else if (std::sscanf(l1.c_str(), "KannalaBrandt %f %f %f %f %f %f %f %f", &ic[0], &ic[1], &ic[2], &ic[3], &ic[4],
                         &ic[5], &ic[6], &ic[7]) == 8) {
        u = new UndistortKB(configFilename.c_str(), false);
        if (!u->isValid()) {
            delete u;
            return 0;
        }
    }

    else if (std::sscanf(l1.c_str(), "RadTan %f %f %f %f %f %f %f %f", &ic[0], &ic[1], &ic[2], &ic[3], &ic[4], &ic[5],
                         &ic[6], &ic[7]) == 8) {
        u = new UndistortRadTan(configFilename.c_str(), false);
        if (!u->isValid()) {
            delete u;
            return 0;
        }
    }

    else if (std::sscanf(l1.c_str(), "EquiDistant %f %f %f %f %f %f %f %f", &ic[0], &ic[1], &ic[2], &ic[3], &ic[4],
                         &ic[5], &ic[6], &ic[7]) == 8) {
        u = new UndistortEquidistant(configFilename.c_str(), false);
        if (!u->isValid()) {
            delete u;
            return 0;
        }
    }

    else if (std::sscanf(l1.c_str(), "FOV %f %f %f %f %f", &ic[0], &ic[1], &ic[2], &ic[3], &ic[4]) == 5) {
        u = new UndistortFOV(configFilename.c_str(), false);
        if (!u->isValid()) {
            delete u;
            return 0;
        }
    }

    else if (std::sscanf(l1.c_str(), "Pinhole %f %f %f %f %f", &ic[0], &ic[1], &ic[2], &ic[3], &ic[4]) == 5) {
        u = new UndistortPinhole(configFilename.c_str(), false);
        if (!u->isValid()) {
            delete u;
            return 0;
        }
    }

    else {
        printf("could not read calib file! exit.");
        exit(1);
    }
    // 读入相机的光度标定参数，将G函数和V函数都保存在成员变量photometricUndist中
    u->loadPhotometricCalibration(gammaFilename, "", vignetteFilename);

    return u;
}

//* 得到光度矫正类
void Undistort::loadPhotometricCalibration(std::string file, std::string noiseImage, std::string vignetteImage) {
    photometricUndist =
        new PhotometricUndistorter(file, noiseImage, vignetteImage, getOriginalSize()[0], getOriginalSize()[1]);
}

/**
 * @brief 得到去光度畸变的能量图像B * t，并添加光度噪声和几何噪声
 * @details
 *  1. 首先，使用photometricUndist类，进行光度矫正
 *  2. 其次，使用remapX和remapY的方式，进行图像的二次差值，获得仅pinhole图像模型
 * @tparam T
 * @param image_raw 原始图像
 * @param exposure 	曝光时间，若没有曝光时间，则默认为1
 * @param timestamp 时间戳，若没有时间戳，则默认为0
 * @param factor 	当光度模型不存在时，使用原图乘factor的方式，代替
 * @return ImageAndExposure* 输出的能量图像+曝光时间，值得注意的是，能量图像是仅pinhole模型
 */
template <typename T>
ImageAndExposure *Undistort::undistort(const MinimalImage<T> *image_raw, float exposure, double timestamp,
                                       float factor) const {
    /// 判断这里的图像大小是否和畸变原始图像一致
    if (image_raw->w != wOrg || image_raw->h != hOrg) {
        printf("Undistort::undistort: wrong image size (%d %d instead of %d %d) \n", image_raw->w, image_raw->h, w, h);
        exit(1);
    }

    /// 使用photometricUndist对象，进行畸变图像的光度矫正处理，得到能量图像float *
    photometricUndist->processFrame<T>(image_raw->data, exposure, factor); // 去除光度参数影响

    /// 构建新的ImageAndExposure对象，并将曝光参数赋值给result
    ImageAndExposure *result = new ImageAndExposure(w, h, timestamp);
    photometricUndist->output->copyMetaTo(*result);

    /// 在Undistort类进行构建时，如果设置了fx、fy参数，则passthrough为false
    if (!passthrough) {

        float *out_data = result->image;                   ///< 仅pinhole模型的图像
        float *in_data = photometricUndist->output->image; ///< 光度矫正后的能量图像

        //[ ***step 2*** ] 如果定义了噪声值, 设置随机几何噪声大小, 并且添加到输出图像
        float *noiseMapX = 0;
        float *noiseMapY = 0;
        if (benchmark_varNoise > 0) {
            /// todo 这里设置的噪声个数为什么是benchmark_noiseGridsize + 8，这存在什么原理呢？
            int numnoise = (benchmark_noiseGridsize + 8) * (benchmark_noiseGridsize + 8);
            noiseMapX = new float[numnoise];
            noiseMapY = new float[numnoise];
            memset(noiseMapX, 0, sizeof(float) * numnoise);
            memset(noiseMapY, 0, sizeof(float) * numnoise);

            /// 生成的噪声是在[-benchmark_varNoise,benchmark_varNoise]的均匀分布
            for (int i = 0; i < numnoise; i++) {
                noiseMapX[i] = 2 * benchmark_varNoise * (rand() / (float)RAND_MAX - 0.5f);
                noiseMapY[i] = 2 * benchmark_varNoise * (rand() / (float)RAND_MAX - 0.5f);
            }
        }

        for (int idx = w * h - 1; idx >= 0; idx--) {
            // get interp. values
            float xx = remapX[idx];
            float yy = remapY[idx];

            if (benchmark_varNoise > 0) {
                //? 具体怎么算的?
                float deltax = getInterpolatedElement11BiCub(
                    noiseMapX, 4 + (xx / (float)wOrg) * benchmark_noiseGridsize,
                    4 + (yy / (float)hOrg) * benchmark_noiseGridsize, benchmark_noiseGridsize + 8);
                float deltay = getInterpolatedElement11BiCub(
                    noiseMapY, 4 + (xx / (float)wOrg) * benchmark_noiseGridsize,
                    4 + (yy / (float)hOrg) * benchmark_noiseGridsize, benchmark_noiseGridsize + 8);
                float x = idx % w + deltax;
                float y = idx / w + deltay;
                if (x < 0.01)
                    x = 0.01;
                if (y < 0.01)
                    y = 0.01;
                if (x > w - 1.01)
                    x = w - 1.01;
                if (y > h - 1.01)
                    y = h - 1.01;

                xx = getInterpolatedElement(remapX, x, y, w);
                yy = getInterpolatedElement(remapY, x, y, w);
            }

            // 插值得到带有几何噪声的输出图像
            if (xx < 0)
                out_data[idx] = 0;
            else {
                // get integer and rational parts
                int xxi = xx;
                int yyi = yy;
                xx -= xxi;
                yy -= yyi;
                float xxyy = xx * yy;

                // get array base pointer
                const float *src = in_data + xxi + yyi * wOrg;

                // interpolate (bilinear)
                out_data[idx] = xxyy * src[1 + wOrg] + (yy - xxyy) * src[wOrg] + (xx - xxyy) * src[1] +
                                (1 - xx - yy + xxyy) * src[0];
            }
        }

        if (benchmark_varNoise > 0) {
            delete[] noiseMapX;
            delete[] noiseMapY;
        }

    } else {
        /// 如果passthrough为true，代表着不需要进行后续的pinhole转换
        /// 将光度矫正后的图像赋值给输出图像result，这样输出图像就是能量图像和曝光时间的组合了
        memcpy(result->image, photometricUndist->output->image, sizeof(float) * w * h);
    }

    /// benchmark_varBlurNoise为0，不添加噪声，我猜测应该是测试鲁棒性相关的。
    applyBlurNoise(result->image);

    return result;
}
template ImageAndExposure *Undistort::undistort<unsigned char>(const MinimalImage<unsigned char> *image_raw,
                                                               float exposure, double timestamp, float factor) const;
template ImageAndExposure *Undistort::undistort<unsigned short>(const MinimalImage<unsigned short> *image_raw,
                                                                float exposure, double timestamp, float factor) const;

//* 添加图像高斯噪声
void Undistort::applyBlurNoise(float *img) const {
    if (benchmark_varBlurNoise == 0)
        return; // 不添加噪声

    int numnoise = (benchmark_noiseGridsize + 8) * (benchmark_noiseGridsize + 8);
    float *noiseMapX = new float[numnoise];
    float *noiseMapY = new float[numnoise];
    float *blutTmp = new float[w * h];

    if (benchmark_varBlurNoise > 0) {
        for (int i = 0; i < numnoise; i++) {
            noiseMapX[i] = benchmark_varBlurNoise * (rand() / (float)RAND_MAX);
            noiseMapY[i] = benchmark_varBlurNoise * (rand() / (float)RAND_MAX);
        }
    }

    // 高斯分布
    float gaussMap[1000];
    for (int i = 0; i < 1000; i++)
        gaussMap[i] = expf((float)(-i * i / (100.0 * 100.0)));

    // 对 X-Y 添加高斯噪声
    // x-blur.
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float xBlur = getInterpolatedElement11BiCub(noiseMapX, 4 + (x / (float)w) * benchmark_noiseGridsize,
                                                        4 + (y / (float)h) * benchmark_noiseGridsize,
                                                        benchmark_noiseGridsize + 8);

            if (xBlur < 0.01)
                xBlur = 0.01;

            int kernelSize = 1 + (int)(1.0f + xBlur * 1.5);
            float sumW = 0;
            float sumCW = 0;
            for (int dx = 0; dx <= kernelSize; dx++) {
                int gmid = 100.0f * dx / xBlur + 0.5f;
                if (gmid > 900)
                    gmid = 900;
                float gw = gaussMap[gmid];

                if (x + dx > 0 && x + dx < w) {
                    sumW += gw;
                    sumCW += gw * img[x + dx + y * this->w];
                }

                if (x - dx > 0 && x - dx < w && dx != 0) {
                    sumW += gw;
                    sumCW += gw * img[x - dx + y * this->w];
                }
            }

            blutTmp[x + y * this->w] = sumCW / sumW;
        }

    // y-blur.
    for (int x = 0; x < w; x++)
        for (int y = 0; y < h; y++) {
            float yBlur = getInterpolatedElement11BiCub(noiseMapY, 4 + (x / (float)w) * benchmark_noiseGridsize,
                                                        4 + (y / (float)h) * benchmark_noiseGridsize,
                                                        benchmark_noiseGridsize + 8);

            if (yBlur < 0.01)
                yBlur = 0.01;

            int kernelSize = 1 + (int)(0.9f + yBlur * 2.5);
            float sumW = 0;
            float sumCW = 0;
            for (int dy = 0; dy <= kernelSize; dy++) {
                int gmid = 100.0f * dy / yBlur + 0.5f;
                if (gmid > 900)
                    gmid = 900;
                float gw = gaussMap[gmid];

                if (y + dy > 0 && y + dy < h) {
                    sumW += gw;
                    sumCW += gw * blutTmp[x + (y + dy) * this->w];
                }

                if (y - dy > 0 && y - dy < h && dy != 0) {
                    sumW += gw;
                    sumCW += gw * blutTmp[x + (y - dy) * this->w];
                }
            }
            img[x + y * this->w] = sumCW / sumW;
        }

    delete[] noiseMapX;
    delete[] noiseMapY;
}

/**
 * @brief
 * 当指定切割为pinhole模型时候，会使用采样的方式，找到归一化坐标系下，不会产生黑边的极限位置Xmin、Ymin和Xmax、Ymax
 *
 * @details 使用这种采样的方式，的确是一个无可奈何的办法，因为带有畸变内容的部分，解出来一个极限距离的值比较困难
 *  1. 首先，使用归一化坐标系下[-5, +5]区间内的均匀分布的100000个点，判断一个较粗粒度的Xmin、Xmax和Ymin、Ymax
 *      1) 以y为0，x为[-5, +5]区间内的均匀分布的归一化点，投影到畸变图像系里面，判断一个Xmin和Xmax的极限距离
 *      2) 以x为0，y为[-5, +5]区间内的均匀分布的归一化点，投影到畸变图像系里面，判断一个Ymin和Ymax的极限距离
 *  2. 第二步，以某个极限轴距离的部分进行等比例划分，投影到畸变图像系里面，判断是否在边界内
 *      1) 对Xmin，设置x部分为Xmin，y部分进行行数目的等比例划分，判断是否有点在外，若在外，将oobLeft置为true
 *      2) 对Xmax，设置x部分为Xmax，y部分进行行数目的等比例划分，判断是否有点在外，若在外，将oobRight置为true
 *      3) 对Ymin，设置y部分为Ymin，x部分进行列数目的等比例划分，判断是否有点在外，若在外，将oobButton置为true
 *      4) 对Ymax，设置y部分为Ymax，x部分进行列数目的等比例划分，判断是否有点在外，若在外，将oobTop置为true
 *  3. 第三步，对oobXXX的部分对应的轴，向中心内部缩小
 *      1)
 * 当X轴方向和y轴方向同时出现超出部分为true的时候，仅缩小宽的那部分，因为缩小x轴会对y轴的部分越界内容给删除掉，防止多余删除
 *      2) 缩小的时候，也是仅仅缩小0.995倍
 *  4. 重复2和3过程，直到所有的oobXXX变量都为false
 *  5. 根据缩减后的Xmin和Ymin、Xmax和Ymax和裁剪的尺寸，进行fx、fy、cx和cy的计算
 *
 * @note 这里面有比较坑的部分
 *  1. remapX和remapY含义的内容，从remapX和remapY的构造部分来看，remapX和remapY的尺寸应该和切割后的图像一致的图像类，
 *     但是，可以发现DSO为了尽量的节省内存开支，将第二步构造等比例点相关的部分的点放入了remapX和remapY里面，造成了变量代表不一致性的问题
 *  2.
 * distortCoordinates的含义问题，可以发现，makeOptimalK_crop函数中大量的调用了distortCoordinates函数，来将某些坐标变换到
 *     畸变图像系下，这一切能够成立的前提，应该是在distortCoordinates开头设置了K为单位坐标系，也就是说，这个时候的参数K的含义仅仅代表了
 *     归一化坐标系到和归一化坐标系相同图片坐标系的变换。
 */
void Undistort::makeOptimalK_crop() {
    printf("finding CROP optimal new model!\n");
    K.setIdentity();

    // 1. 首先，使用归一化坐标系下[-5, +5]区间内的均匀分布的100000个点，判断一个较粗粒度的Xmin、Xmax和Ymin、Ymax
    float *tgX = new float[100000];
    float *tgY = new float[100000];
    float minX = 0;
    float maxX = 0;
    float minY = 0;
    float maxY = 0;

    for (int x = 0; x < 100000; x++) {
        tgX[x] = (x - 50000.0f) / 10000.0f;
        tgY[x] = 0;
    }                                               // -5 ~ 5 ?
    distortCoordinates(tgX, tgY, tgX, tgY, 100000); // 矫正

    for (int x = 0; x < 100000; x++) {
        if (tgX[x] > 0 && tgX[x] < wOrg - 1) {
            if (minX == 0)
                minX = (x - 50000.0f) / 10000.0f;
            maxX = (x - 50000.0f) / 10000.0f;
        }
    }

    for (int y = 0; y < 100000; y++) {
        tgY[y] = (y - 50000.0f) / 10000.0f;
        tgX[y] = 0;
    }
    distortCoordinates(tgX, tgY, tgX, tgY, 100000);
    for (int y = 0; y < 100000; y++) {
        if (tgY[y] > 0 && tgY[y] < hOrg - 1) {
            if (minY == 0)
                minY = (y - 50000.0f) / 10000.0f;
            maxY = (y - 50000.0f) / 10000.0f;
        }
    }
    delete[] tgX;
    delete[] tgY;

    minX *= 1.01;
    maxX *= 1.01;
    minY *= 1.01;
    maxY *= 1.01;

    printf("initial range: x: %.4f - %.4f; y: %.4f - %.4f!\n", minX, maxX, minY, maxY);

    /// 4. 重复2和3过程，直到所有的oobXXX变量都为false
    bool oobLeft = true, oobRight = true, oobTop = true, oobBottom = true;
    int iteration = 0;
    while (oobLeft || oobRight || oobTop || oobBottom) {
        oobLeft = oobRight = oobTop = oobBottom = false;

        /// 2. 第二步，以某个极限轴距离的部分进行等比例划分，投影到畸变图像系里面，判断是否在边界内
        for (int y = 0; y < h; y++) {
            remapX[y * 2] = minX;
            remapX[y * 2 + 1] = maxX;
            remapY[y * 2] = remapY[y * 2 + 1] = minY + (maxY - minY) * (float)y / ((float)h - 1.0f);
        }
        distortCoordinates(remapX, remapY, remapX, remapY, 2 * h);

        for (int y = 0; y < h; y++) {
            if (!(remapX[2 * y] > 0 && remapX[2 * y] < wOrg - 1))
                oobLeft = true;
            if (!(remapX[2 * y + 1] > 0 && remapX[2 * y + 1] < wOrg - 1))
                oobRight = true;
        }

        for (int x = 0; x < w; x++) {
            remapY[x * 2] = minY;
            remapY[x * 2 + 1] = maxY;
            remapX[x * 2] = remapX[x * 2 + 1] = minX + (maxX - minX) * (float)x / ((float)w - 1.0f);
        }
        distortCoordinates(remapX, remapY, remapX, remapY, 2 * w);

        for (int x = 0; x < w; x++) {
            if (!(remapY[2 * x] > 0 && remapY[2 * x] < hOrg - 1))
                oobTop = true;
            if (!(remapY[2 * x + 1] > 0 && remapY[2 * x + 1] < hOrg - 1))
                oobBottom = true;
        }

        /// 如果上下, 左右都超出去, 也只缩减最大的一侧，因为一侧的缩减会影响另一个轴
        if ((oobLeft || oobRight) && (oobTop || oobBottom)) {
            if ((maxX - minX) > (maxY - minY))
                oobBottom = oobTop = false;
            else
                oobLeft = oobRight = false;
        }

        // 缩减
        if (oobLeft)
            minX *= 0.995;
        if (oobRight)
            maxX *= 0.995;
        if (oobTop)
            minY *= 0.995;
        if (oobBottom)
            maxY *= 0.995;

        iteration++;

        printf("iteration %05d: range: x: %.4f - %.4f; y: %.4f - %.4f!\n", iteration, minX, maxX, minY, maxY);
        if (iteration > 500) // 迭代次数太多
        {
            printf("FAILED TO COMPUTE GOOD CAMERA MATRIX - SOMETHING IS SERIOUSLY WRONG. ABORTING \n");
            exit(1);
        }
    }

    /// 5. 根据缩减后的Xmin和Ymin、Xmax和Ymax和裁剪的尺寸，进行fx、fy、cx和cy的计算
    K(0, 0) = ((float)w - 1.0f) / (maxX - minX);
    K(1, 1) = ((float)h - 1.0f) / (maxY - minY);
    K(0, 2) = -minX * K(0, 0);
    K(1, 2) = -minY * K(1, 1);
}

void Undistort::makeOptimalK_full() {
    // todo
    assert(false);
}

/**
 * @brief 从畸变参数文件、参数数目和前缀名，进行相机的参数读取，并将带有畸变的相机转变为无畸变的小孔成像模型
 * @details
 * 计算无畸变小孔成像模型、并对新的小孔成像模型上的位置和畸变图像上的位置做了映射、除此之外，进行了valid和passthrough内容的设置
 *  1. 无畸变小孔成像模型参数计算，使用归一化坐标系均匀采样+投影到畸变图像系的方式，确定Xmin、Xmax、Ymin和Ymax
 *  2. 使用Xmin、Xmax、Ymin和Ymax的数值，进行fx,fy,cx,cy的计算
 *  3. 将无畸变坐标系，向畸变系进行投影，保留无畸变位置到有畸变位置之间的映射，保存在remapX和remapY中
 * @param configFileName    相机内参配置文件名
 * @param nPars             参数的数量
 * @param prefix            前缀名，即用来确定畸变模型是什么
 */
void Undistort::readFromFile(const char *configFileName, int nPars, std::string prefix) {
    photometricUndist = 0;
    valid = false;
    passthrough = false;
    remapX = 0;
    remapY = 0;

    float outputCalibration[5];

    parsOrg = VecX(nPars); // 相机原始参数

    // read parameters
    std::ifstream infile(configFileName);
    assert(infile.good());

    std::string l1, l2, l3, l4;

    std::getline(infile, l1);
    std::getline(infile, l2);
    std::getline(infile, l3);
    std::getline(infile, l4);

    //* 第一行, 相机模型参数; 第二行, 相机像素大小
    // l1 & l2
    if (nPars == 5) // fov model
    {
        char buf[1000];
        // 复制 prefix 最大1000个, 以%s格式, 到 buf (char) 数组, %%表示输出%
        // 因此buf中是 "fov%lf %lf %lf %lf %lf"
        snprintf(buf, 1000, "%s%%lf %%lf %%lf %%lf %%lf", prefix.c_str());

        // 使用buf做格式控制, 将l1输出到这5个参数
        if (std::sscanf(l1.c_str(), buf, &parsOrg[0], &parsOrg[1], &parsOrg[2], &parsOrg[3], &parsOrg[4]) == 5 &&
            std::sscanf(l2.c_str(), "%d %d", &wOrg, &hOrg) == 2) // 得到像素大小
        {
            printf("Input resolution: %d %d\n", wOrg, hOrg);
            printf("In: %f %f %f %f %f\n", parsOrg[0], parsOrg[1], parsOrg[2], parsOrg[3], parsOrg[4]);
        } else {
            printf("Failed to read camera calibration (invalid format?)\nCalibration file: %s\n", configFileName);
            infile.close();
            return;
        }
    } else if (nPars == 8) // KB, equi & radtan model
    {
        char buf[1000];
        snprintf(buf, 1000, "%s%%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf", prefix.c_str());

        if (std::sscanf(l1.c_str(), buf, &parsOrg[0], &parsOrg[1], &parsOrg[2], &parsOrg[3], &parsOrg[4], &parsOrg[5],
                        &parsOrg[6], &parsOrg[7]) == 8 &&
            std::sscanf(l2.c_str(), "%d %d", &wOrg, &hOrg) == 2) {
            printf("Input resolution: %d %d\n", wOrg, hOrg);
            printf("In: %s%f %f %f %f %f %f %f %f\n", prefix.c_str(), parsOrg[0], parsOrg[1], parsOrg[2], parsOrg[3],
                   parsOrg[4], parsOrg[5], parsOrg[6], parsOrg[7]);
        } else {
            printf("Failed to read camera calibration (invalid format?)\nCalibration file: %s\n", configFileName);
            infile.close();
            return;
        }
    } else {
        printf("called with invalid number of parameters.... forgot to implement me?\n");
        infile.close();
        return;
    }

    // cx, cy 小于1, 则说明是个相对值, 乘上图像大小
    if (parsOrg[2] < 1 && parsOrg[3] < 1) {
        printf("\n\nFound fx=%f, fy=%f, cx=%f, cy=%f.\n I'm assuming this is the \"relative\" calibration file format,"
               "and will rescale this by image width / height to fx=%f, fy=%f, cx=%f, cy=%f.\n\n",
               parsOrg[0], parsOrg[1], parsOrg[2], parsOrg[3], parsOrg[0] * wOrg, parsOrg[1] * hOrg,
               parsOrg[2] * wOrg - 0.5, parsOrg[3] * hOrg - 0.5);

        //?? 0.5 还是不是很理解, 为了使用积分来近似像素强度
        // rescale and substract 0.5 offset.
        // the 0.5 is because I'm assuming the calibration is given such that the pixel at (0,0)
        // contains the integral over intensity over [0,0]-[1,1], whereas I assume the pixel (0,0)
        // to contain a sample of the intensity ot [0,0], which is best approximated by the integral over
        // [-0.5,-0.5]-[0.5,0.5]. Thus, the shift by -0.5.
        parsOrg[0] = parsOrg[0] * wOrg;
        parsOrg[1] = parsOrg[1] * hOrg;
        parsOrg[2] = parsOrg[2] * wOrg - 0.5;
        parsOrg[3] = parsOrg[3] * hOrg - 0.5;
    }

    //* 第三行, 相机图像类别, 是否裁切
    // l3
    if (l3 == "crop") {
        outputCalibration[0] = -1;
        printf("Out: Rectify Crop\n");
    } else if (l3 == "full") {
        outputCalibration[0] = -2;
        printf("Out: Rectify Full\n");
    } else if (l3 == "none") {
        outputCalibration[0] = -3;
        printf("Out: No Rectification\n");
    }
    //? 这啥参数呢...
    else if (std::sscanf(l3.c_str(), "%f %f %f %f %f", &outputCalibration[0], &outputCalibration[1],
                         &outputCalibration[2], &outputCalibration[3], &outputCalibration[4]) == 5) {
        printf("Out: %f %f %f %f %f\n", outputCalibration[0], outputCalibration[1], outputCalibration[2],
               outputCalibration[3], outputCalibration[4]);

    } else {
        printf("Out: Failed to Read Output pars... not rectifying.\n");
        infile.close();
        return;
    }

    //* 第四行, 图像的大小, 会根据设置进行裁切...
    // l4
    if (std::sscanf(l4.c_str(), "%d %d", &w, &h) == 2) {
        // 如果有设置的大小
        if (benchmarkSetting_width != 0) {
            w = benchmarkSetting_width;
            if (outputCalibration[0] == -3)
                outputCalibration[0] = -1; // crop instead of none, since probably resolution changed.
        }
        if (benchmarkSetting_height != 0) {
            h = benchmarkSetting_height;
            if (outputCalibration[0] == -3)
                outputCalibration[0] = -1; // crop instead of none, since probably resolution changed.
        }

        printf("Output resolution: %d %d\n", w, h);
    } else {
        printf("Out: Failed to Read Output resolution... not rectifying.\n");
        valid = false;
    }

    /// 这里的w和h分别为裁剪之后的宽和高
    remapX = new float[w * h];
    remapY = new float[w * h];

    //* 得到合适的相机参数
    if (outputCalibration[0] == -1) {
        /// 针对裁剪的参数，进行相机仅小孔成像K矩阵的计算
        makeOptimalK_crop();
    } else if (outputCalibration[0] == -2)
        makeOptimalK_full();
    else if (outputCalibration[0] == -3) {
        if (w != wOrg || h != hOrg) {
            printf("ERROR: rectification mode none requires input and output dimenstions to match!\n\n");
            exit(1);
        }
        K.setIdentity();
        K(0, 0) = parsOrg[0];
        K(1, 1) = parsOrg[1];
        K(0, 2) = parsOrg[2];
        K(1, 2) = parsOrg[3];
        passthrough = true; // 读取参数成功, 通过
    } else {

        // 标定输出错误
        if (outputCalibration[2] > 1 || outputCalibration[3] > 1) {
            printf("\n\n\nWARNING: given output calibration (%f %f %f %f) seems wrong. It needs to be relative to "
                   "image width / height!\n\n\n",
                   outputCalibration[0], outputCalibration[1], outputCalibration[2], outputCalibration[3]);
        }

        // 相对于长宽的比例值（TUMmono这样）
        K.setIdentity();
        K(0, 0) = outputCalibration[0] * w;
        K(1, 1) = outputCalibration[1] * h;
        K(0, 2) = outputCalibration[2] * w - 0.5;
        K(1, 2) = outputCalibration[3] * h - 0.5;
    }

    /// benchmarkSetting_fxfyfac如果对这个进行了设置，则对K里面的fx和fy进行修改
    if (benchmarkSetting_fxfyfac != 0) {
        K(0, 0) = fmax(benchmarkSetting_fxfyfac, (float)K(0, 0));
        K(1, 1) = fmax(benchmarkSetting_fxfyfac, (float)K(1, 1));

        //! 如果benchmarkSetting_fxfyfac设置了之后，passthrough为false，是去畸变Undistort
        passthrough = false;
    }

    // remapX为列上的元素，为列索引，remapY为行上的元素为行
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            remapX[x + y * w] = x;
            remapY[x + y * w] = y;
        }

    /// 这个时候，K已经设置完成了，代表的是去畸变图像内参矩阵，将分割后的图像投影到畸变坐标下，target到畸变映射
    distortCoordinates(remapX, remapY, remapX, remapY, h * w);

    /// 这里是将映射过去的坐标，如果在畸变图像范围之内，则进行保存，不在的话，进行-1,-1处理
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            // make rounding resistant.
            float ix = remapX[x + y * w];
            float iy = remapY[x + y * w];

            if (ix == 0)
                ix = 0.001;
            if (iy == 0)
                iy = 0.001;
            if (ix == wOrg - 1)
                ix = wOrg - 1.001;
            if (iy == hOrg - 1)
                ix = hOrg - 1.001;

            if (ix > 0 && iy > 0 && ix < wOrg - 1 && iy < wOrg - 1) {
                remapX[x + y * w] = ix;
                remapY[x + y * w] = iy;
            } else {
                remapX[x + y * w] = -1;
                remapY[x + y * w] = -1;
            }
        }

    valid = true;

    printf("\nRectified Kamera Matrix:\n");
    std::cout << K << "\n\n";
}

//********************* 以下都是加畸变的算法 *******************************
UndistortFOV::UndistortFOV(const char *configFileName, bool noprefix) {
    printf("Creating FOV undistorter\n");

    if (noprefix)
        readFromFile(configFileName, 5);
    else
        readFromFile(configFileName, 5, "FOV ");
}
UndistortFOV::~UndistortFOV() {}
//* FOV加畸变
void UndistortFOV::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const {
    float dist = parsOrg[4];
    float d2t = 2.0f * tan(dist / 2.0f);

    // current camera parameters
    float fx = parsOrg[0];
    float fy = parsOrg[1];
    float cx = parsOrg[2];
    float cy = parsOrg[3];

    float ofx = K(0, 0);
    float ofy = K(1, 1);
    float ocx = K(0, 2);
    float ocy = K(1, 2);

    for (int i = 0; i < n; i++) {
        float x = in_x[i];
        float y = in_y[i];
        float ix = (x - ocx) / ofx;
        float iy = (y - ocy) / ofy;

        float r = sqrtf(ix * ix + iy * iy);
        float fac = (r == 0 || dist == 0) ? 1 : atanf(r * d2t) / (dist * r);

        ix = fx * fac * ix + cx;
        iy = fy * fac * iy + cy;

        out_x[i] = ix;
        out_y[i] = iy;
    }
}

UndistortRadTan::UndistortRadTan(const char *configFileName, bool noprefix) {
    printf("Creating RadTan undistorter\n");

    if (noprefix)
        readFromFile(configFileName, 8);
    else
        readFromFile(configFileName, 8, "RadTan ");
}
UndistortRadTan::~UndistortRadTan() {}

void UndistortRadTan::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const {
    // RADTAN
    float fx = parsOrg[0];
    float fy = parsOrg[1];
    float cx = parsOrg[2];
    float cy = parsOrg[3];
    float k1 = parsOrg[4];
    float k2 = parsOrg[5];
    float r1 = parsOrg[6];
    float r2 = parsOrg[7];

    float ofx = K(0, 0);
    float ofy = K(1, 1);
    float ocx = K(0, 2);
    float ocy = K(1, 2);

    for (int i = 0; i < n; i++) {
        float x = in_x[i];
        float y = in_y[i];

        // RADTAN
        float ix = (x - ocx) / ofx;
        float iy = (y - ocy) / ofy;
        float mx2_u = ix * ix;
        float my2_u = iy * iy;
        float mxy_u = ix * iy;
        float rho2_u = mx2_u + my2_u;
        float rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
        float x_dist = ix + ix * rad_dist_u + 2.0 * r1 * mxy_u + r2 * (rho2_u + 2.0 * mx2_u);
        float y_dist = iy + iy * rad_dist_u + 2.0 * r2 * mxy_u + r1 * (rho2_u + 2.0 * my2_u);
        float ox = fx * x_dist + cx;
        float oy = fy * y_dist + cy;

        out_x[i] = ox;
        out_y[i] = oy;
    }
}

UndistortEquidistant::UndistortEquidistant(const char *configFileName, bool noprefix) {
    printf("Creating Equidistant undistorter\n");

    if (noprefix)
        readFromFile(configFileName, 8);
    else
        readFromFile(configFileName, 8, "EquiDistant ");
}
UndistortEquidistant::~UndistortEquidistant() {}

void UndistortEquidistant::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const {
    // EQUI
    float fx = parsOrg[0];
    float fy = parsOrg[1];
    float cx = parsOrg[2];
    float cy = parsOrg[3];
    float k1 = parsOrg[4];
    float k2 = parsOrg[5];
    float k3 = parsOrg[6];
    float k4 = parsOrg[7];

    float ofx = K(0, 0);
    float ofy = K(1, 1);
    float ocx = K(0, 2);
    float ocy = K(1, 2);

    for (int i = 0; i < n; i++) {
        float x = in_x[i];
        float y = in_y[i];

        // EQUI
        float ix = (x - ocx) / ofx;
        float iy = (y - ocy) / ofy;
        float r = sqrt(ix * ix + iy * iy);
        float theta = atan(r);
        float theta2 = theta * theta;
        float theta4 = theta2 * theta2;
        float theta6 = theta4 * theta2;
        float theta8 = theta4 * theta4;
        float thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
        float scaling = (r > 1e-8) ? thetad / r : 1.0;
        float ox = fx * ix * scaling + cx;
        float oy = fy * iy * scaling + cy;

        out_x[i] = ox;
        out_y[i] = oy;
    }
}

UndistortKB::UndistortKB(const char *configFileName, bool noprefix) {
    printf("Creating KannalaBrandt undistorter\n");

    if (noprefix)
        readFromFile(configFileName, 8);
    else
        readFromFile(configFileName, 8, "KannalaBrandt ");
}
UndistortKB::~UndistortKB() {}

void UndistortKB::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const {
    const float fx = parsOrg[0];
    const float fy = parsOrg[1];
    const float cx = parsOrg[2];
    const float cy = parsOrg[3];
    const float k0 = parsOrg[4];
    const float k1 = parsOrg[5];
    const float k2 = parsOrg[6];
    const float k3 = parsOrg[7];

    const float ofx = K(0, 0);
    const float ofy = K(1, 1);
    const float ocx = K(0, 2);
    const float ocy = K(1, 2);

    for (int i = 0; i < n; i++) {
        float x = in_x[i];
        float y = in_y[i];

        // RADTAN
        float ix = (x - ocx) / ofx;
        float iy = (y - ocy) / ofy;

        const float Xsq_plus_Ysq = ix * ix + iy * iy;
        const float sqrt_Xsq_Ysq = sqrtf(Xsq_plus_Ysq);
        const float theta = atan2f(sqrt_Xsq_Ysq, 1);
        const float theta2 = theta * theta;
        const float theta3 = theta2 * theta;
        const float theta5 = theta3 * theta2;
        const float theta7 = theta5 * theta2;
        const float theta9 = theta7 * theta2;
        const float r = theta + k0 * theta3 + k1 * theta5 + k2 * theta7 + k3 * theta9;

        if (sqrt_Xsq_Ysq < 1e-6) {
            out_x[i] = fx * ix + cx;
            out_y[i] = fy * iy + cy;
        } else {
            out_x[i] = (r / sqrt_Xsq_Ysq) * fx * ix + cx;
            out_y[i] = (r / sqrt_Xsq_Ysq) * fy * iy + cy;
        }
    }
}

UndistortPinhole::UndistortPinhole(const char *configFileName, bool noprefix) {
    if (noprefix)
        readFromFile(configFileName, 5);
    else
        readFromFile(configFileName, 5, "Pinhole ");
}
UndistortPinhole::~UndistortPinhole() {}

void UndistortPinhole::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const {
    // current camera parameters
    float fx = parsOrg[0];
    float fy = parsOrg[1];
    float cx = parsOrg[2];
    float cy = parsOrg[3];

    float ofx = K(0, 0);
    float ofy = K(1, 1);
    float ocx = K(0, 2);
    float ocy = K(1, 2);

    for (int i = 0; i < n; i++) {
        float x = in_x[i];
        float y = in_y[i];
        float ix = (x - ocx) / ofx;
        float iy = (y - ocy) / ofy;
        ix = fx * ix + cx;
        iy = fy * iy + cy;
        out_x[i] = ix;
        out_y[i] = iy;
    }
}

} // namespace dso
