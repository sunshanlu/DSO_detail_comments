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
#include "FullSystem/ResidualProjections.h"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "FullSystem/CoarseTracker.h"

namespace dso {

/**
 * @brief 标记需要被边缘化的帧
 * @details
 * 	1. 可以使用的点较少 in / (in + out) < 0.05
 * 	2. 与newFH之间，曝光参数差异较大 ---> 代表的环境变化较大
 * 	3. 在时间轴上，保证距离newFH较近的3帧不边缘化，并且在距离轴上，保证距离newFH较远的帧进行边缘化
 * @param newFH	输入的关键帧
 */
void FullSystem::flagFramesForMarginalization(FrameHessian *newFH) {
    //? 怎么会有这种情况呢?
    if (setting_minFrameAge > setting_maxFrames) {
        for (int i = setting_maxFrames; i < (int)frameHessians.size(); i++) {
            FrameHessian *fh = frameHessians[i - setting_maxFrames]; // setting_maxFrames个之前的都边缘化掉
            fh->flaggedForMarginalization = true;
        }
        return;
    }

    /// 这里也可以在2000的基础上，设置一个点的比例下限，来代替 in / (in + out)
    int flagged = 0;
    for (int i = 0; i < (int)frameHessians.size(); i++) {
        FrameHessian *fh = frameHessians[i];
        int in = fh->pointHessians.size() + fh->immaturePoints.size();                ///< 成熟和未成熟点数量
        int out = fh->pointHessiansMarginalized.size() + fh->pointHessiansOut.size(); ///< 边缘化和丢掉的点数量

        Vec2 refToFh = AffLight::fromToVecExposure(frameHessians.back()->ab_exposure, fh->ab_exposure, frameHessians.back()->aff_g2l(), fh->aff_g2l());

        /// (这一帧里的内点少 or 与newFH之间的曝光参数差的大) and (边缘化掉后还有5-7帧)，满足条件，则设置边缘化flag
        if ((in < setting_minPointsRemaining * (in + out) || fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow) &&
            ((int)frameHessians.size()) - flagged > setting_minFrames) {

            fh->flaggedForMarginalization = true;
            flagged++;
        }
    }

    /// 当窗口内的帧 - 被标记为边缘化的帧数 依然大于窗口内的FH的设定数量时，根据空间进行边缘化，要求距离 newFH 近的帧多，
    if ((int)frameHessians.size() - flagged >= setting_maxFrames) {
        double smallestScore = 1;
        FrameHessian *toMarginalize = 0;
        FrameHessian *latest = frameHessians.back();

        for (FrameHessian *fh : frameHessians) {
            /// 在时间上，离newFH较近的帧，不能被边缘化，或者说第一个关键帧的点的数目不满足关键帧的边缘化策略
            if (fh->frameID > latest->frameID - setting_minFrameAge || fh->frameID == 0)
                continue;

            double distScore = 0;
            for (FrameFramePrecalc &ffh : fh->targetPrecalc) {
                if (ffh.target->frameID > latest->frameID - setting_minFrameAge + 1 || ffh.target == ffh.host)
                    continue;
                distScore += 1 / (1e-5 + ffh.distanceLL); // 帧间距离
            }

            // 论文有提到, 启发式的良好的3D空间分布, 关键帧更接近，把距离newFH的一帧给边缘化掉
            distScore *= -sqrtf(fh->targetPrecalc.back().distanceLL);

            if (distScore < smallestScore) {
                smallestScore = distScore;
                toMarginalize = fh;
            }
        }

        toMarginalize->flaggedForMarginalization = true;
        flagged++;
    }
}

/**
 * @brief 滑动窗口边缘化掉某个帧
 * @details
 *  1. 在边缘化某个帧之前，需要保证该帧上的所有点都已经被丢掉或者被边缘化掉
 *  2. EnergyFunctional的帧边缘化 @see EnergyFunctional::marginalizeFrame （重点）
 *  3. 其他帧上，被当前边缘化的帧观测到的残差也需要统统删掉（留着也没用，构建不成约束了，因为残差所对应的顶点少了一个）
 *  4. 标记是在哪一个关键帧进来时，当前frame被淘汰掉的 --> marginalizedAt
 *  5. 统计被边缘化帧，在优化后距离线性化点的距离模长（仅位姿部分） --> movedByOpt
 *  6. 删除并重新构建滑窗内的idx
 *  7. 预计算一些值和状态更新量
 *  8. 计算线性化点处的 dlocal / dhost 和 dlocal / dtaret
 * 
 * @param frame 输出的待边缘化的帧
 */
void FullSystem::marginalizeFrame(FrameHessian *frame) {
    /// 在边缘化某个帧之前，需要保证该帧上的所有点都已经被丢掉或者被边缘化掉
    assert((int)frame->pointHessians.size() == 0);

    /// EnergyFunctional的帧边缘化 @see EnergyFunctional::marginalizeFrame
    ef->marginalizeFrame(frame->efFrame);

    /// 其他帧上，被当前边缘化的帧观测到的残差也需要统统删掉（留着也没用，构建不成约束了，因为残差所对应的顶点少了一个）
    for (FrameHessian *fh : frameHessians) {
        if (fh == frame)
            continue;

        for (PointHessian *ph : fh->pointHessians) {
            for (unsigned int i = 0; i < ph->residuals.size(); i++) {
                PointFrameResidual *r = ph->residuals[i];
                if (r->target == frame) {
                    if (ph->lastResiduals[0].first == r)
                        ph->lastResiduals[0].first = 0;
                    else if (ph->lastResiduals[1].first == r)
                        ph->lastResiduals[1].first = 0;

                    /// 统计前面帧上的点，被观测到的次数（强制删除）
                    if (r->host->frameID < frame->frameID)
                        statistics_numForceDroppedResFwd++;
                    /// 统计后面帧上的点，被观测到的次数（强制删除）
                    else
                        statistics_numForceDroppedResBwd++;

                    /// 删除被看到的残差
                    ef->dropResidual(r->efResidual);
                    deleteOut<PointFrameResidual>(ph->residuals, i);
                    break;
                }
            }
        }
    }

    {
        std::vector<FrameHessian *> v;
        v.push_back(frame);
        for (IOWrap::Output3DWrapper *ow : outputWrapper)
            ow->publishKeyframes(v, true, &Hcalib);
    }

    /// 标记是在哪一个关键帧进来时，当前frame被淘汰掉的
    frame->shell->marginalizedAt = frameHessians.back()->shell->id;

    /// 统计被边缘化帧，在优化后距离线性化点的距离模长（仅位姿部分）
    frame->shell->movedByOpt = frame->w2c_leftEps().norm();

    /// 删除并重新构建滑窗内的idx
    deleteOutOrder<FrameHessian>(frameHessians, frame);
    for (unsigned int i = 0; i < frameHessians.size(); i++)
        frameHessians[i]->idx = i;

    /// 预计算一些值和状态更新量
    setPrecalcValues();

    /// 计算线性化点处的 dlocal / dhost 和 dlocal / dtaret
    ef->setAdjointsF(&Hcalib);
}

} // namespace dso
