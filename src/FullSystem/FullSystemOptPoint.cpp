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

/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"

#include "IOWrapper/ImageDisplay.h"
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>

#include "FullSystem/ImmaturePoint.h"
#include "math.h"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace dso {

/**
 * @brief 优化未成熟点
 * @details
 *  1. 初始化临时残差项（除了host帧以外，其余滑窗内的帧都需要初始化，来构建残差）
 *  2. 遍历不同的临时残差项，进行线性化 --> E(idepth)、H(idepth)、b(idepth)
 *  3. 使用类LM的方法，来优化逆深度 idepth
 *  4. 对优化能量不超过阈值，并且 滑窗内的可观数目大于 minObs，则优化成功
 *
 * @param point 		输入的待优化的未成熟点
 * @param minObs 		输入的要求的滑窗内最少的观测数目（非host帧）
 * @param residuals 	输入的临时残差类型（）
 * @return PointHessian*
 */
PointHessian *FullSystem::optimizeImmaturePoint(ImmaturePoint *point, int minObs, ImmaturePointTemporaryResidual *residuals) {
    /// 1. 初始化和其它关键帧的res(点在其它关键帧上投影)
    int nres = 0;
    for (FrameHessian *fh : frameHessians) {
        if (fh != point->host) {
            residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
            residuals[nres].state_NewState = ResState::OUTLIER;
            residuals[nres].state_state = ResState::IN;
            residuals[nres].target = fh; ///< 指定临时残差项中的target，将 ph 投影到 target上
            nres++;
        }
    }
    assert(nres == ((int)frameHessians.size()) - 1);

    bool print = false;

    float lastEnergy = 0;
    float lastHdd = 0;
    float lastbd = 0;
    float currentIdepth = (point->idepth_max + point->idepth_min) * 0.5f; ///< 将idepth_max 和 idepth_min 的平均值作为逆深度初始值

    /// 2. 使用类LM(GN)的方法来优化逆深度
    for (int i = 0; i < nres; i++) {
        lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals + i, lastHdd, lastbd, currentIdepth);
        residuals[i].state_state = residuals[i].state_NewState;
        residuals[i].state_energy = residuals[i].state_NewEnergy;
    }

    if (!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act) {
        if (print)
            printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n", nres, lastHdd, lastEnergy);
        return 0;
    }

    if (print)
        printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n", nres, lastHdd, lastEnergy, currentIdepth);

    float lambda = 0.1;
    for (int iteration = 0; iteration < setting_GNItsOnPointActivation; iteration++) {
        float H = lastHdd;
        H *= 1 + lambda;
        float step = (1.0 / H) * lastbd;
        float newIdepth = currentIdepth - SCALE_IDEPTH * step;

        float newHdd = 0;
        float newbd = 0;
        float newEnergy = 0;
        for (int i = 0; i < nres; i++)
            newEnergy += point->linearizeResidual(&Hcalib, 1, residuals + i, newHdd, newbd, newIdepth);

        if (!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act) {
            if (print)
                printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n", nres, newHdd, lastEnergy);
            return 0;
        }

        if (print)
            printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n", (true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT", iteration, log10(lambda), "",
                   lastEnergy, newEnergy, newIdepth);

        if (newEnergy < lastEnergy) {
            currentIdepth = newIdepth;
            lastHdd = newHdd;
            lastbd = newbd;
            lastEnergy = newEnergy;
            for (int i = 0; i < nres; i++) {
                residuals[i].state_state = residuals[i].state_NewState;
                residuals[i].state_energy = residuals[i].state_NewEnergy;
            }

            lambda *= 0.5;
        } else {
            lambda *= 5;
        }

        if (fabsf(step) < 0.0001 * currentIdepth)
            break;
    }

    if (!std::isfinite(currentIdepth)) {
        printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
        // 丢弃无穷的点
        return (PointHessian *)((long)(-1)); // yeah I'm like 99% sure this is OK on 32bit systems.
    }

    /// 统计 ph 在滑动窗口中的被观测到的次数
    int numGoodRes = 0;
    for (int i = 0; i < nres; i++)
        if (residuals[i].state_state == ResState::IN)
            numGoodRes++;

    if (numGoodRes < minObs) {
        if (print)
            printf("OptPoint: OUTLIER!\n");
        /// 这是啥东西？？
        return (PointHessian *)((long)(-1)); // yeah I'm like 99% sure this is OK on 32bit systems.
    }

    /// 如果被认为优化成功，且被观测次数足够，则构建新点 ph
    PointHessian *p = new PointHessian(point, &Hcalib);
    if (!std::isfinite(p->energyTH)) {
        delete p;
        return (PointHessian *)((long)(-1));
    }

    /// 对 ph 的初始化，设置激活状态
    p->lastResiduals[0].first = 0;
    p->lastResiduals[0].second = ResState::OOB;
    p->lastResiduals[1].first = 0;
    p->lastResiduals[1].second = ResState::OOB;
    p->setIdepthZero(currentIdepth);
    p->setIdepth(currentIdepth);
    p->setPointStatus(PointHessian::ACTIVE);

    /// 构建PointFrameResidual，ph中还保存了近两次优化的状态 r
    for (int i = 0; i < nres; i++)
        /// 如果 滑窗中的 target 可观
        if (residuals[i].state_state == ResState::IN) {
            PointFrameResidual *r = new PointFrameResidual(p, p->host, residuals[i].target);
            r->state_NewEnergy = r->state_energy = 0;
            r->state_NewState = ResState::OUTLIER;
            r->setState(ResState::IN);
            p->residuals.push_back(r); ///< PointHessian::residuals里面维护的是所有优化的残差项

            /// PointHessian::lastResiduals[0] 和 PointHessian::lastResiduals[1] 维护的是 近两次 点优化的残差项
            if (r->target == frameHessians.back()) // 和最新帧的残差
            {
                p->lastResiduals[0].first = r;
                p->lastResiduals[0].second = ResState::IN;
            } else if (r->target == (frameHessians.size() < 2 ? 0 : frameHessians[frameHessians.size() - 2])) // 和最新帧上一帧的残差
            {
                p->lastResiduals[1].first = r;
                p->lastResiduals[1].second = ResState::IN;
            }
        }

    if (print)
        printf("point activated!\n");

    statistics_numActivatedPoints++;
    return p;
}

} // namespace dso
