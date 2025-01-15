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

#include "OptimizationBackend/MatrixAccumulators.h"
#include "util/IndexThreadReduce.h"
#include "util/NumType.h"
#include "vector"
#include <math.h>

namespace dso {

class EFPoint;
class EnergyFunctional;

class AccumulatedTopHessianSSE {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    inline AccumulatedTopHessianSSE() {
        for (int tid = 0; tid < NUM_THREADS; tid++) {
            nres[tid] = 0;
            acc[tid] = 0;
            nframes[tid] = 0;
        }
    };
    inline ~AccumulatedTopHessianSSE() {
        for (int tid = 0; tid < NUM_THREADS; tid++) {
            if (acc[tid] != 0)
                delete[] acc[tid];
        }
    };

    /**
     * @brief 构造AccumulatedTopHessianSSE累加器后，需要置零初始化
     * @details
     * 	1. 初始化累加器 nframes * nframes --> 某个host -> target 之间的累加
     *
     * @param nFrames	输入的滑动窗口的帧数
     * @param min		！没有用到
     * @param max		！没有用到
     * @param stats		！没有用到
     * @param tid		输入的线程id
     */
    inline void setZero(int nFrames, int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0) {
        if (nFrames != nframes[tid]) {
            if (acc[tid] != 0)
                delete[] acc[tid];
#if USE_XI_MODEL
            acc[tid] = new Accumulator14[nFrames * nFrames];
#else
            /// 构建累加器
            acc[tid] = new AccumulatorApprox[nFrames * nFrames]; ///< host->target
#endif
        }

        /// 对累加器进行初始化
        for (int i = 0; i < nFrames * nFrames; i++)
            acc[tid][i].initialize();

        nframes[tid] = nFrames;
        nres[tid] = 0;
    }

    void stitchDouble(MatXX &H, VecX &b, EnergyFunctional const *const EF, bool usePrior, bool useDelta, int tid = 0);

    template <int mode> void addPoint(EFPoint *p, EnergyFunctional const *const ef, int tid = 0);

    /**
     * @brief 构建系统的H矩阵和b矩阵 （考虑了滑窗中产生的先验（内参 + 帧），但是还没有加上HM 和 bM）
     * @details
     *  1. 使用 AccumulatedTopHessianSSE::stitchDoubleInternal 构建了H矩阵 和 b矩阵，但是这里的H矩阵带有方向性，并不是真正的H矩阵
     *  2. 根据 AccumulatedTopHessianSSE::stitchDoubleInternal 的注释中分析的结果，构建完整的H矩阵
     * 
     * @see AccumulatedTopHessianSSE::stitchDoubleInternal
     * @param red       多线程相关
     * @param H         输出的H矩阵
     * @param b         输出的b矩阵
     * @param EF        输入的EnergyFunctional
     * @param usePrior  输入的是否使用先验
     * @param MT        输入的是否使用多线程
     */
    void stitchDoubleMT(IndexThreadReduce<Vec10> *red, MatXX &H, VecX &b, EnergyFunctional const *const EF, bool usePrior, bool MT) {
        // sum up, splitting by bock in square.
        if (MT) {
            MatXX Hs[NUM_THREADS];
            VecX bs[NUM_THREADS];
            for (int i = 0; i < NUM_THREADS; i++) {
                assert(nframes[0] == nframes[i]);
                //* 所有的优化变量维度
                Hs[i] = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
                bs[i] = VecX::Zero(nframes[0] * 8 + CPARS);
            }

            red->reduce(boost::bind(&AccumulatedTopHessianSSE::stitchDoubleInternal, this, Hs, bs, EF, usePrior, _1, _2, _3, _4), 0, nframes[0] * nframes[0],
                        0);

            // sum up results
            H = Hs[0];
            b = bs[0];
            //* 所有线程求和
            for (int i = 1; i < NUM_THREADS; i++) {
                H.noalias() += Hs[i];
                b.noalias() += bs[i];
                nres[0] += nres[i];
            }
        }

        /// 非多线程状态
        else {
            H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS); ///< 构建滑窗内全局的H矩阵 --> 与某个残差的相对量之间有明显差别
            b = VecX::Zero(nframes[0] * 8 + CPARS);                          ///< 构建滑窗内全局的b向量 --> 与某个残差的相对量之间有明显差别

            /// stitchDoubleMT的核心函数 @see AccumulatedTopHessianSSE::stitchDoubleInternal
            stitchDoubleInternal(&H, &b, EF, usePrior, 0, nframes[0] * nframes[0], 0, -1);
        }

        for (int h = 0; h < nframes[0]; h++) {
            int hIdx = CPARS + h * 8;
            H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose(); ///< 补齐涉及内参的上三角部分

            /// 根据AccumulatedTopHessianSSE::stitchDoubleInternal中的分析，构建涉及帧上三角的部分，并将上三角部分赋值给下三角
            for (int t = h + 1; t < nframes[0]; t++) {
                int tIdx = CPARS + t * 8;
                H.block<8, 8>(hIdx, tIdx).noalias() += H.block<8, 8>(tIdx, hIdx).transpose();
                H.block<8, 8>(tIdx, hIdx).noalias() = H.block<8, 8>(hIdx, tIdx).transpose();
            }
        }
    }

    int nframes[NUM_THREADS]; //!< 每个线程的帧数

    EIGEN_ALIGN16 AccumulatorApprox *acc[NUM_THREADS]; //!< 计算hessian的累乘器

    int nres[NUM_THREADS]; //!< 残差计数

    template <int mode>
    void addPointsInternal(std::vector<EFPoint *> *points, EnergyFunctional const *const ef, int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0) {
        for (int i = min; i < max; i++)
            addPoint<mode>((*points)[i], ef, tid);
    }

private:
    void stitchDoubleInternal(MatXX *H, VecX *b, EnergyFunctional const *const EF, bool usePrior, int min, int max, Vec10 *stats, int tid);
};
} // namespace dso
