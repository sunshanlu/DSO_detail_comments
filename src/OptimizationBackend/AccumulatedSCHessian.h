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

class AccumulatedSCHessianSSE {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    inline AccumulatedSCHessianSSE() {
        for (int i = 0; i < NUM_THREADS; i++) // 多线程
        {
            accE[i] = 0;
            accEB[i] = 0;
            accD[i] = 0;
            nframes[i] = 0;
        }
    };
    inline ~AccumulatedSCHessianSSE() {
        for (int i = 0; i < NUM_THREADS; i++) {
            if (accE[i] != 0)
                delete[] accE[i];
            if (accEB[i] != 0)
                delete[] accEB[i];
            if (accD[i] != 0)
                delete[] accD[i];
        }
    };

    inline void setZero(int n, int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0) {
        if (n != nframes[tid]) // 如果帧数有变化
        {
            if (accE[tid] != 0)
                delete[] accE[tid];
            if (accEB[tid] != 0)
                delete[] accEB[tid];
            if (accD[tid] != 0)
                delete[] accD[tid];
            // 这三数组
            accE[tid] = new AccumulatorXX<8, CPARS>[n * n];
            accEB[tid] = new AccumulatorX<8>[n * n];
            accD[tid] = new AccumulatorXX<8, 8>[n * n * n];
        }
        accbc[tid].initialize();
        accHcc[tid].initialize();

        for (int i = 0; i < n * n; i++) {
            accE[tid][i].initialize();
            accEB[tid][i].initialize();

            for (int j = 0; j < n; j++)
                accD[tid][i * n + j].initialize();
        }
        nframes[tid] = n;
    }
    void stitchDouble(MatXX &H_sc, VecX &b_sc, EnergyFunctional const *const EF, int tid = 0);
    void addPoint(EFPoint *p, bool shiftPriorToZero, int tid = 0);

    /**
     * @brief 多线程构建Hsc和bsc
     *
     * @param red   多线程
     * @param H     输出的Hsc
     * @param b     输出的bsc
     * @param EF    输入的EnergyFunctional
     * @param MT    是否使用多线程
     */
    void stitchDoubleMT(IndexThreadReduce<Vec10> *red, MatXX &H, VecX &b, EnergyFunctional const *const EF, bool MT) {
        if (MT) {
            MatXX Hs[NUM_THREADS];
            VecX bs[NUM_THREADS];
            // 分配空间大小
            for (int i = 0; i < NUM_THREADS; i++) {
                assert(nframes[0] == nframes[i]);
                Hs[i] = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
                bs[i] = VecX::Zero(nframes[0] * 8 + CPARS);
            }

            red->reduce(boost::bind(&AccumulatedSCHessianSSE::stitchDoubleInternal, this, Hs, bs, EF, _1, _2, _3, _4), 0, nframes[0] * nframes[0], 0);

            // sum up results
            H = Hs[0];
            b = bs[0];

            for (int i = 1; i < NUM_THREADS; i++) {
                H.noalias() += Hs[i];
                b.noalias() += bs[i];
            }
        } else {
            H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
            b = VecX::Zero(nframes[0] * 8 + CPARS);
            stitchDoubleInternal(&H, &b, EF, 0, nframes[0] * nframes[0], 0, -1);
        }

        ///! 这部分在 stitchDoubleInternal 好像已经做过了，是否是无用功？
        for (int h = 0; h < nframes[0]; h++) {
            int hIdx = CPARS + h * 8;
            H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();
        }
    }

    AccumulatorXX<8, CPARS> *accE[NUM_THREADS]; //!< 位姿和内参关于逆深度的 Schur
    AccumulatorX<8> *accEB[NUM_THREADS];        //!< 位姿光度关于逆深度的 b*Schur
    AccumulatorXX<8, 8> *accD[NUM_THREADS];     //!< 两位姿光度关于逆深度的 Schur

    AccumulatorXX<CPARS, CPARS> accHcc[NUM_THREADS]; //!< 内参关于逆深度的 Schur
    AccumulatorX<CPARS> accbc[NUM_THREADS];          //!< 内参关于逆深度的 b*Schur
    int nframes[NUM_THREADS];

    void addPointsInternal(std::vector<EFPoint *> *points, bool shiftPriorToZero, int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0) {
        for (int i = min; i < max; i++)
            addPoint((*points)[i], shiftPriorToZero, tid);
    }

private:
    void stitchDoubleInternal(MatXX *H, VecX *b, EnergyFunctional const *const EF, int min, int max, Vec10 *stats, int tid);
};

} // namespace dso
