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

#include "util/globalCalib.h"
#include "stdio.h"
#include <iostream>

//! 后面带G的是global变量
namespace dso {
/// 值得注意的是，金字塔的每一层都是仅pinhole模型对应的宽度和高度

int wG[PYR_LEVELS]; ///< 金字塔每层的宽度
int hG[PYR_LEVELS]; ///< 金字塔每层的高度

float fxG[PYR_LEVELS]; ///< 金字塔每层的fx
float fyG[PYR_LEVELS]; ///< 金字塔每层的fy
float cxG[PYR_LEVELS]; ///< 金字塔每层的cx
float cyG[PYR_LEVELS]; ///< 金字塔每层的cy

float fxiG[PYR_LEVELS]; ///< 金字塔每层的fx的逆
float fyiG[PYR_LEVELS]; ///< 金字塔每层的fy的逆
float cxiG[PYR_LEVELS]; ///< 金字塔每层的cx的逆
float cyiG[PYR_LEVELS]; ///< 金字塔每层的cy的逆

Eigen::Matrix3f KG[PYR_LEVELS];  ///< 金字塔每层的K，相机内参矩阵
Eigen::Matrix3f KiG[PYR_LEVELS]; ///< 金字塔每层的K矩阵的逆，相机内参矩阵的逆

float wM3G; ///< 第0层的宽度-3
float hM3G; ///< 第0层的高度-3

void setGlobalCalib(int w, int h, const Eigen::Matrix3f &K) {
    int wlvl = w;      ///< 金字塔最上层的宽度
    int hlvl = h;      ///< 金字塔最上层的高度
    pyrLevelsUsed = 1; ///< 一共有几层金字塔

    /// 这里要求存pinhole模型的宽度和高度应该尽量多的是2的整数次幂，最多除六次2，也就是最多7层金字塔
    while (wlvl % 2 == 0 && hlvl % 2 == 0 && wlvl * hlvl > 5000 && pyrLevelsUsed < PYR_LEVELS) {
        wlvl /= 2;
        hlvl /= 2;
        pyrLevelsUsed++;
    }

    printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n", pyrLevelsUsed - 1, wlvl, hlvl);

    /// 如果最高层的宽度和高度都大于100，则代表金字塔分层不充分，会警告w和h的2的整数次幂不够
    if (wlvl > 100 && hlvl > 100) {
        printf("\n\n===============WARNING!===================\n "
               "using not enough pyramid levels.\n"
               "Consider scaling to a resolution that is a multiple of a power of 2.\n");
    }

    /// 如果金字塔小于三层，也会警告，说明金字塔分层不充分
    if (pyrLevelsUsed < 3) {
        printf("\n\n===============WARNING!===================\n "
               "I need higher resolution.\n"
               "I will probably segfault.\n");
    }

    wM3G = w - 3;
    hM3G = h - 3;

    wG[0] = w;
    hG[0] = h;
    KG[0] = K;
    fxG[0] = K(0, 0);
    fyG[0] = K(1, 1);
    cxG[0] = K(0, 2);
    cyG[0] = K(1, 2);
    KiG[0] = KG[0].inverse();
    fxiG[0] = KiG[0](0, 0);
    fyiG[0] = KiG[0](1, 1);
    cxiG[0] = KiG[0](0, 2);
    cyiG[0] = KiG[0](1, 2);

    for (int level = 1; level < pyrLevelsUsed; ++level) {
        wG[level] = w >> level;
        hG[level] = h >> level;

        fxG[level] = fxG[level - 1] * 0.5;
        fyG[level] = fyG[level - 1] * 0.5;

        /// 使用这种先加再减的策略，保证像素的中心代表像素而不是像素的左上角
        cxG[level] = (cxG[0] + 0.5) / ((int)1 << level) - 0.5;
        cyG[level] = (cyG[0] + 0.5) / ((int)1 << level) - 0.5;

        KG[level] << fxG[level], 0.0, cxG[level], 0.0, fyG[level], cyG[level], 0.0, 0.0, 1.0; // synthetic
        KiG[level] = KG[level].inverse();

        fxiG[level] = KiG[level](0, 0);
        fyiG[level] = KiG[level](1, 1);
        cxiG[level] = KiG[level](0, 2);
        cyiG[level] = KiG[level](1, 2);
    }
}

} // namespace dso
