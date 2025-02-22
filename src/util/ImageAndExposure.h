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

#pragma once
#include <cstring>
#include <iostream>

namespace dso {

//* 辐照度值B和曝光时间t
class ImageAndExposure {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    float *image;        ///< irradiance. between 0 and 256
    int w, h;            ///< 图像宽高
    double timestamp;    ///< 图像时间戳
    float exposure_time; ///< 曝光时间

    inline ImageAndExposure(int w_, int h_, double timestamp_ = 0)
        : w(w_)
        , h(h_)
        , timestamp(timestamp_) {
        image = new float[w * h]; ///< 开辟图像空间
        exposure_time = 1;        ///< 曝光时间默认设置为1
    }
    inline ~ImageAndExposure() { delete[] image; }

    /**
     * @brief 将曝光时间赋值给other中的曝光时间
     *
     * @param other 输入的其他ImageAndExposure
     */
    inline void copyMetaTo(ImageAndExposure &other) { other.exposure_time = exposure_time; }

    /**
     * @brief 实现了ImageAndExposure的深拷贝
     * @details
     * 	1. 为img开辟新空间
     * 	2. 将曝光时间进行拷贝
     * 	3. 将图像数据进行拷贝（深）
     * @return ImageAndExposure*
     */
    inline ImageAndExposure *getDeepCopy() {
        ImageAndExposure *img = new ImageAndExposure(w, h, timestamp);
        img->exposure_time = exposure_time;
        memcpy(img->image, image, w * h * sizeof(float));
        return img;
    }
};

} // namespace dso
