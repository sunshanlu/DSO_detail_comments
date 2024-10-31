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

#include "util/NumType.h"

namespace dso {

const float minUseGrad_pixsel = 10;

//@ 对于高层(0层以上)选择梯度最大的位置点
template <int pot> inline int gridMaxSelection(Eigen::Vector3f *grads, bool *map_out, int w, int h, float THFac) {

    memset(map_out, 0, sizeof(bool) * w * h);

    int numGood = 0;
    for (int y = 1; y < h - pot; y += pot) // 每隔一个pot遍历
    {
        for (int x = 1; x < w - pot; x += pot) {
            int bestXXID = -1; // gradx 最大
            int bestYYID = -1; // grady 最大
            int bestXYID = -1; // gradx-grady 最大
            int bestYXID = -1; // gradx+grady 最大

            float bestXX = 0, bestYY = 0, bestXY = 0, bestYX = 0;

            Eigen::Vector3f *grads0 = grads + x + y * w; // 当前网格的起点
            // 分别找到该网格内上面4个best
            for (int dx = 0; dx < pot; dx++)
                for (int dy = 0; dy < pot; dy++) {
                    int idx = dx + dy * w;
                    Eigen::Vector3f g = grads0[idx];                // 遍历网格内的每一个像素
                    float sqgd = g.tail<2>().squaredNorm();         // 梯度平方和
                    float TH = THFac * minUseGrad_pixsel * (0.75f); //阈值, 为什么都乘0.75 ? downweight

                    if (sqgd > TH * TH) {
                        float agx = fabs((float)g[1]);
                        if (agx > bestXX) {
                            bestXX = agx;
                            bestXXID = idx;
                        }

                        float agy = fabs((float)g[2]);
                        if (agy > bestYY) {
                            bestYY = agy;
                            bestYYID = idx;
                        }

                        float gxpy = fabs((float)(g[1] - g[2]));
                        if (gxpy > bestXY) {
                            bestXY = gxpy;
                            bestXYID = idx;
                        }

                        float gxmy = fabs((float)(g[1] + g[2]));
                        if (gxmy > bestYX) {
                            bestYX = gxmy;
                            bestYXID = idx;
                        }
                    }
                }

            bool *map0 = map_out + x + y * w; // 选出来的像素为TRUE

            // 选上这些最大的像素
            if (bestXXID >= 0) {
                if (!map0[bestXXID]) // 没有被选
                    numGood++;
                map0[bestXXID] = true;
            }
            if (bestYYID >= 0) {
                if (!map0[bestYYID])
                    numGood++;
                map0[bestYYID] = true;
            }
            if (bestXYID >= 0) {
                if (!map0[bestXYID])
                    numGood++;
                map0[bestXYID] = true;
            }
            if (bestYXID >= 0) {
                if (!map0[bestYXID])
                    numGood++;
                map0[bestYXID] = true;
            }
        }
    }

    return numGood;
}

/**
 * @brief 给定梯度，和pot大小，对点进行选择
 * @details
 *  1. 遍历区块和区块内的每个像素
 *  2. 若区块内的某像素满足:x梯度最大 OR y梯度最大 OR x梯度-y梯度最大 OR x梯度 + y梯度最大
 *      2.1 则将给像素点选中，即map_out中的相应位置置为true
 * @param grads     输入的梯度，dI / dt | dI / dx | dI / dy
 * @param map_out   输出的选择的点的位置，true / false
 * @param w         宽度
 * @param h         高度
 * @param pot       区域pot的大小，setting文件里面设为5
 * @param THFac     梯度阈值因子
 * @return int      输出的选择点的数目
 */
inline int gridMaxSelection(Eigen::Vector3f *grads, bool *map_out, int w, int h, int pot, float THFac) {

    memset(map_out, 0, sizeof(bool) * w * h);

    int numGood = 0;

    /// 遍历区块
    for (int y = 1; y < h - pot; y += pot)
        for (int x = 1; x < w - pot; x += pot) {
            /// 这里的x，y分别代表pot * pot区域内的左上角的像素位置

            int bestXXID = -1;
            int bestYYID = -1;
            int bestXYID = -1;
            int bestYXID = -1;

            float bestXX = 0, bestYY = 0, bestXY = 0, bestYX = 0;

            /// grads0，代表当前pot * pot区域左上角像素位置对应的梯度指针
            Eigen::Vector3f *grads0 = grads + x + y * w;

            /// 遍历pot * pot区块内的像素
            for (int dx = 0; dx < pot; dx++)
                for (int dy = 0; dy < pot; dy++) {
                    /// 这里的dx和dy代表以(x,y)为原点的相对像素位置
                    int idx = dx + dy * w;
                    Eigen::Vector3f g = grads0[idx];
                    float sqgd = g.tail<2>().squaredNorm();
                    float TH = THFac * minUseGrad_pixsel * (0.75f);

                    if (sqgd > TH * TH) {

                        /// 遍历像素的x方向上的能量梯度
                        float agx = fabs((float)g[1]);
                        if (agx > bestXX) {
                            bestXX = agx;
                            bestXXID = idx;
                        }

                        /// 遍历像素的y方向上的能量梯度
                        float agy = fabs((float)g[2]);
                        if (agy > bestYY) {
                            bestYY = agy;
                            bestYYID = idx;
                        }

                        /// 遍历像素的x - y梯度的绝对值
                        float gxpy = fabs((float)(g[1] - g[2]));
                        if (gxpy > bestXY) {
                            bestXY = gxpy;
                            bestXYID = idx;
                        }

                        /// 遍历像素的x + y梯度的绝对值
                        float gxmy = fabs((float)(g[1] + g[2]));
                        if (gxmy > bestYX) {
                            bestYX = gxmy;
                            bestYXID = idx;
                        }
                    }
                }

            /// map0代表一个pot内的左上角位置
            bool *map0 = map_out + x + y * w;

            if (bestXXID >= 0 && !map0[bestXXID]) {
                numGood++;
                map0[bestXXID] = true;
            }
            if (bestYYID >= 0 && !map0[bestYYID]) {
                numGood++;
                map0[bestYYID] = true;
            }
            if (bestXYID >= 0 && !map0[bestXYID]) {
                numGood++;
                map0[bestXYID] = true;
            }
            if (bestYXID >= 0 && !map0[bestYXID]) {
                numGood++;
                map0[bestYXID] = true;
            }
        }

    return numGood;
}

/**
 * @brief 除第0层金字塔外的，其他层能量函数，需要使用这个函数进行点的选取
 * @details
 * 	1. 使用是也是动态条件pot的方法进行，不过这里使用的pot是采用模型的方式完全算出来的
 * 	2. 使用gridMaxSelection进行点的选取，得到当前pot参数选取的点的数量
 *  3. 计算理想的pot，以num = K / (pot ^ 2)作为计算依据，认为点的选择严格和面积相关
 *  4. 得到的理想pot和当前pot，如果理想pot和当前pot都为1，说明pot为1时选择的数量都不够，这时候需要减小THFac了，控制阈值
 *  5. 如果(新旧pot之间的变化小于1 AND 阈值因子THFac无变化) OR (hav / want)在0.8到1.0/0.8之间 OR (递归次数归零)
 *      5.1 返回成功选择的数量
 *  6. 否则，将理想pot赋值给当前pot，递归次数减少，并进行递归
 * @param grads				图像的dI，即dI/dt | dI/dx | dI/dy
 * @param map				输出点的选择结果，true/false
 * @param w					输入的待选择图像的宽度
 * @param h					输入的待选择图像的高度
 * @param desiredDensity	numWant，即需要选择的点的数量
 * @param recsLeft			最大递归次数
 * @param THFac				阈值因子，函数gridMaxSelection使用 @see gridMaxSelection
 * @return int				输出的选择的点的数量
 */
inline int makePixelStatus(Eigen::Vector3f *grads, bool *map, int w, int h, float desiredDensity, int recsLeft = 5,
                           float THFac = 1) {
    if (sparsityFactor < 1)
        sparsityFactor = 1;

    int numGoodPoints;

    if (sparsityFactor == 1)
        numGoodPoints = gridMaxSelection<1>(grads, map, w, h, THFac);
    else if (sparsityFactor == 2)
        numGoodPoints = gridMaxSelection<2>(grads, map, w, h, THFac);
    else if (sparsityFactor == 3)
        numGoodPoints = gridMaxSelection<3>(grads, map, w, h, THFac);
    else if (sparsityFactor == 4)
        numGoodPoints = gridMaxSelection<4>(grads, map, w, h, THFac);
    else if (sparsityFactor == 5)
        numGoodPoints = gridMaxSelection<5>(grads, map, w, h, THFac);
    else if (sparsityFactor == 6)
        numGoodPoints = gridMaxSelection<6>(grads, map, w, h, THFac);
    else if (sparsityFactor == 7)
        numGoodPoints = gridMaxSelection<7>(grads, map, w, h, THFac);
    else if (sparsityFactor == 8)
        numGoodPoints = gridMaxSelection<8>(grads, map, w, h, THFac);
    else if (sparsityFactor == 9)
        numGoodPoints = gridMaxSelection<9>(grads, map, w, h, THFac);
    else if (sparsityFactor == 10)
        numGoodPoints = gridMaxSelection<10>(grads, map, w, h, THFac);
    else if (sparsityFactor == 11)
        numGoodPoints = gridMaxSelection<11>(grads, map, w, h, THFac);
    else
        numGoodPoints = gridMaxSelection(grads, map, w, h, sparsityFactor, THFac);

    float quotia = numGoodPoints / (float)(desiredDensity);

    int newSparsity = (sparsityFactor * sqrtf(quotia)) + 0.7f; // 更新网格大小

    if (newSparsity < 1)
        newSparsity = 1;

    float oldTHFac = THFac;
    if (newSparsity == 1 && sparsityFactor == 1)
        THFac = 0.5;

    if ((abs(newSparsity - sparsityFactor) < 1 && THFac == oldTHFac) || (quotia > 0.8 && 1.0f / quotia > 0.8) ||
        recsLeft == 0) {
        sparsityFactor = newSparsity;
        return numGoodPoints;
    } else {

        sparsityFactor = newSparsity;
        return makePixelStatus(grads, map, w, h, desiredDensity, recsLeft - 1, THFac);
    }
}

} // namespace dso
