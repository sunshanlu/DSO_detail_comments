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

#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/ResidualProjections.h"
#include "util/FrameShell.h"

namespace dso {

ImmaturePoint::ImmaturePoint(int u_, int v_, FrameHessian *host_, float type, CalibHessian *HCalib)
    : u(u_)
    , v(v_)
    , host(host_)
    , my_type(type)
    , idepth_min(0)
    , idepth_max(NAN)
    , lastTraceStatus(IPS_UNINITIALIZED) {

    gradH.setZero();

    for (int idx = 0; idx < patternNum; idx++) {
        int dx = patternP[idx][0];
        int dy = patternP[idx][1];

        /// 计算得到以当前[uv]为中心的pattern的host Frame 差值内容 [像素值, dx, dy]
        Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u + dx, v + dy, wG[0]);

        /// 当差值获得的像素值为 finite 时，能量阈值设为NAN，后续会根据这个做判断，是否保留这个immaturepoint
        color[idx] = ptc[0];
        if (!std::isfinite(color[idx])) {
            energyTH = NAN;
            return;
        }

        /// pattern的梯度矩阵[sum(dx*2), sum(dxdy); sum(dydx), sum(dy^2)]
        gradH += ptc.tail<2>() * ptc.tail<2>().transpose();

        /// pattern上的点的权重信息 c^2 / ( c^2 + ||grad||^2 )，防止大梯度为外点，导致Energy的剧烈变化
        weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
    }

    /// 根据setting设置energyTH，这里为什么不直接使用一个参数来和energyTH打交道，而是设置两个参数，是否有些麻烦
    energyTH = patternNum * setting_outlierTH;
    energyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;

    idepth_GT = 0;   ///< idepth_GT被初始化为0
    quality = 10000; ///< 质量被初始化为1000
}

ImmaturePoint::~ImmaturePoint() {}

/**
 * @brief ImmaturePoint的深度估计，使用极线搜索 + GN优化的方式
 * @details
 *  1. 使用 idepth_min 和 idepth_max 选项来确定极线搜索的区域，和极线方向direction
 *  2. 由于idepth无法确定，因此在极线搜索中，pi的pattern不能精确投影到pj中，因此DSO只考虑了KRKi对pattern的影响（个人认为使用w * w的窗口依然可行）
 *  3. 根据极线方向和参考帧上，pi的pattern对应的像素梯度，建模由于不准确的极线导致的像素误差 error-pixel = 0.2 + 0.2 * (a + b) / a
 *      3.1 a 近似代表了cos<l,g>的大小，即极线方向和像素梯度方向的夹角
 *      3.2 b 近似代表了极线方向和像素等值线方向的夹角 --> 0.2 + 0.2 * (cos<l, g> ^2)
 *  4. 使用极线初步确定 最佳匹配的像素后，使用 GN来构建优化问题，求解最佳的step 即 pj = pj0 + step * [lx, ly]^T
 *      4.1 dr / dstep = (dr / dpj) * (dpj / dstep)
 *  5. 确定好最佳匹配的 pj 后， 可以根据pj，考虑 error-pixel 产生的影响，可以计算出4个 idepth，从中选择出最大idepth_max 和 idepth_min
 *  6. 更新idepth 范围，并返回跟踪状态
 *
 * @note 这里 唯一不理解的地方就是 极线搜索 像素误差的建模方法
 *
 * @param frame                 输入的前端确定位姿的帧
 * @param hostToFrame_KRKi      输入的中间变量KRKi
 * @param hostToFrame_Kt        输入的中间变量Kt
 * @param hostToFrame_affine    输入的中间变量affine，用于构建残差
 * @param HCalib                ---> 在函数中没有用到
 * @param debugPrint            调试输出
 * @return ImmaturePointStatus  当前ImmaturePoint是否跟踪成功
 */
ImmaturePointStatus ImmaturePoint::traceOn(FrameHessian *frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine,
                                           CalibHessian *HCalib, bool debugPrint) {

    /// 针对搜索区域超出图像的，尺度变化大，两次优化残差大于阈值的，置为IPS_OOB
    if (lastTraceStatus == ImmaturePointStatus::IPS_OOB)
        return lastTraceStatus; ///< IPS_OOB状态的点，不需要进行优化

    debugPrint = false;

    /// 通过setting_maxPixSearch超参数，控制极线搜索的长度（与图像的分辨率有关）
    float maxPixSearch = (wG[0] + hG[0]) * setting_maxPixSearch;

    if (debugPrint)
        printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f -> %f. t %f %f %f!\n", u, v, host->shell->id, frame->shell->id, idepth_min, idepth_max,
               hostToFrame_Kt[0], hostToFrame_Kt[1], hostToFrame_Kt[2]);

    /// idepth_min 对应的 像素坐标pj = (KRKi * [u, v, 1] + Kt * idepth_min) / Z
    Vec3f pr = hostToFrame_KRKi * Vec3f(u, v, 1);
    Vec3f ptpMin = pr + hostToFrame_Kt * idepth_min;
    float uMin = ptpMin[0] / ptpMin[2];
    float vMin = ptpMin[1] / ptpMin[2];

    /// 如果超出图像范围则设为 OOB，border为4像素
    if (!(uMin > 4 && vMin > 4 && uMin < wG[0] - 5 && vMin < hG[0] - 5)) {
        if (debugPrint)
            printf("OOB uMin %f %f - %f %f %f (id %f-%f)!\n", u, v, uMin, vMin, ptpMin[2], idepth_min, idepth_max);
        lastTraceUV = Vec2f(-1, -1);
        lastTracePixelInterval = 0;
        return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
    }

    float dist;
    float uMax;
    float vMax;
    Vec3f ptpMax;

    /// idepth_max 对应的 像素坐标pj = (KRKi * [u, v, 1] + Kt * idepth_max) / Z
    if (std::isfinite(idepth_max)) {
        ptpMax = pr + hostToFrame_Kt * idepth_max;
        uMax = ptpMax[0] / ptpMax[2];
        vMax = ptpMax[1] / ptpMax[2];

        /// 超出图像范围，设置为 OOB
        if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5)) {
            if (debugPrint)
                printf("OOB uMax  %f %f - %f %f!\n", u, v, uMax, vMax);
            lastTraceUV = Vec2f(-1, -1);
            lastTracePixelInterval = 0;
            return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
        }

        /// 检查极线搜索的区域大小，如果在1.5个像素之内，设置SKIPPED状态，代表无需优化了
        dist = (uMin - uMax) * (uMin - uMax) + (vMin - vMax) * (vMin - vMax);
        dist = sqrtf(dist);
        if (dist < setting_trace_slackInterval) {
            if (debugPrint)
                printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);

            lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5; // 直接设为中值
            lastTracePixelInterval = dist;
            return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
        }

    } else {
        /// idepth_max无穷大时，设置极线搜索大小为maxPixSearch
        dist = maxPixSearch;

        /// 使用 idepth = 0.01 以此来确定极线的搜索方向
        ptpMax = pr + hostToFrame_Kt * 0.01;
        uMax = ptpMax[0] / ptpMax[2];
        vMax = ptpMax[1] / ptpMax[2];
        float dx = uMax - uMin;
        float dy = vMax - vMin;
        float d = 1.0f / sqrtf(dx * dx + dy * dy);

        /// 根据搜索方向和最大搜索长度确定 uMax和vMax
        uMax = uMin + dist * dx * d;
        vMax = vMin + dist * dy * d;

        /// 对非法投影区域的像素点，设置为OOB状态
        if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5)) {
            if (debugPrint)
                printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax, ptpMax[2]);
            lastTraceUV = Vec2f(-1, -1);
            lastTracePixelInterval = 0;
            return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
        }
    }

    /// ptpMin[2]代表的是 dpi_min条件下，dpi / dpj 逆深度的比值 --> dj / di，如果深度比例之间不在0.75 - 1.5 之间，则认为是尺度变换过大
    /// 这种情况下，同样需要将点设置为 OOB 状态，没有优化的必要
    if (!(idepth_min < 0 || (ptpMin[2] > 0.75 && ptpMin[2] < 1.5))) {
        if (debugPrint)
            printf("OOB SCALE %f %f %f!\n", uMax, vMax, ptpMin[2]);
        lastTraceUV = Vec2f(-1, -1);
        lastTracePixelInterval = 0;
        return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
    }

    float dx = setting_trace_stepsize * (uMax - uMin);
    float dy = setting_trace_stepsize * (vMax - vMin);

    /// a = (lxdx + lydy) ^ 2，可以代表 host帧上的像素梯度 和 极线之间的夹角大小
    float a = (Vec2f(dx, dy).transpose() * gradH * Vec2f(dx, dy));
    /// b = (lydx - lxdy) ^ 2，可以代表 host帧上的像素等值线 和 极线方向的夹角大小
    float b = (Vec2f(dy, -dx).transpose() * gradH * Vec2f(dy, -dx));

    /// 误差像素为什么可以这样求解 0.2 + 0.2 / (cos(l, g) ^2)
    float errorInPixel = 0.2f + 0.2f * (a + b) / a;

    /// 如果像素误差(极线l和像素梯度g决定的) 的两倍 大于 极线搜索的长度，则认为没有足够的优化，设置为 BADCONDITION 状态
    if (errorInPixel * setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max)) {
        if (debugPrint)
            printf("NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);
        lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;
        lastTracePixelInterval = dist;
        return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
    }

    /// 下面的逆深度不确定性，要求这里的 errorInPixel <= 10, 规定了一个不确定性的尺度
    if (errorInPixel > 10)
        errorInPixel = 10;

    dx /= dist; ///< cos(l) * step，u方向上的搜索步长
    dy /= dist; ///< sin(l) * step，v方向上的搜索步长

    if (debugPrint)
        printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> %f (%.1f %.1f)! ErrorInPixel %.1f!\n", u, v, host->shell->id,
               frame->shell->id, idepth_min, uMin, vMin, idepth_max, uMax, vMax, errorInPixel);

    /// 当线搜的范围大于要求的范围时，缩小线搜范围到要求的maxPixSearch，以idepth_min为起点
    if (dist > maxPixSearch) {
        uMax = uMin + maxPixSearch * dx;
        vMax = vMin + maxPixSearch * dy;
        dist = maxPixSearch;
    }

    int numSteps = 1.9999f + dist / setting_trace_stepsize;
    Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2, 2>();

    float randShift = uMin * 1000 - floorf(uMin * 1000); ///< 获取uMin上千分位数上的小数部分
    float ptx = uMin - randShift * dx;                   ///< 这里为什么不直接使用uMin的u（极线的允许误差模拟）
    float pty = vMin - randShift * dy;                   ///< 这里为什么不直接使用vMin的v（极线的允许误差模拟）

    /// 这里不使用投影的方式是有原因的，因为这里不会考虑由于dpi造成的影响，如果将dpi考虑在内，则会导致不确定的dpi计算不出pattern中Pi投影到pj中的位置
    /// 所以这里，仅考虑pattern的一个小块，并且仅考虑旋转部分对小块造成的影响 --> 当然，我认为这里可以使用w * w的块来匹配
    Vec2f rotatetPattern[MAX_RES_PER_POINT];
    for (int idx = 0; idx < patternNum; idx++)
        rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);

    /// 这个判断是否有意义
    if (!std::isfinite(dx) || !std::isfinite(dy)) {
        lastTracePixelInterval = 0;
        lastTraceUV = Vec2f(-1, -1);
        return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
    }

    /// 沿着级线搜索误差最小的位置
    float errors[100];
    float bestU = 0, bestV = 0, bestEnergy = 1e10;
    int bestIdx = -1;

    /// 除了超参数规定的最大搜索长度以外，还有一个硬性条件，搜索steps不能超过100step
    if (numSteps >= 100)
        numSteps = 99;

    /// 沿着一个不太准确的极线，搜索一个不太准确的最优部分
    for (int i = 0; i < numSteps; i++) {
        float energy = 0;
        for (int idx = 0; idx < patternNum; idx++) {
            /// 求解某个pattern的 pi --> pj 的能量值
            float hitColor = getInterpolatedElement31(frame->dI, (float)(ptx + rotatetPattern[idx][0]), (float)(pty + rotatetPattern[idx][1]), wG[0]);

            if (!std::isfinite(hitColor)) {
                energy += 1e5;
                continue;
            }

            /// 构建残差
            float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

            /// 带有huber 核函数作用的能量
            energy += hw * residual * residual * (2 - hw);
        }

        if (debugPrint)
            printf("step %.1f %.1f (id %f): energy = %f!\n", ptx, pty, 0.0f, energy);

        errors[i] = energy; ///< 用来寻找最优附近的次优内容，计算质量
        if (energy < bestEnergy) {
            bestU = ptx;
            bestV = pty;
            bestEnergy = energy;
            bestIdx = i;
        }

        ptx += dx;
        pty += dy;
    }

    /// 在2半径内，寻找一个次优点对应的energy
    float secondBest = 1e10;
    for (int i = 0; i < numSteps; i++) {
        if ((i < bestIdx - setting_minTraceTestRadius || i > bestIdx + setting_minTraceTestRadius) && errors[i] < secondBest)
            secondBest = errors[i];
    }

    /// ImmaturePoint 质量更新比较考究
    /// 1. 如果numSteps比较大，认为这个ImmaturePoint不够稳定，这时不论质量如何，都需要进行更新
    /// 2. 如果numSteps比较小，认为这个ImmaturePoint比较稳定，这时如果质量低于阈值，就进行更新
    /// 以此来获得一个 质量高切稳定的 ImmaturePoint
    float newQuality = secondBest / bestEnergy;
    if (newQuality < quality || numSteps > 10)
        quality = newQuality;

    /// 使用GN 旨在一个小范围内，找到一个最优的值
    float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
    if (setting_trace_GNIterations > 0)
        bestEnergy = 1e5;

    int gnStepsGood = 0, gnStepsBad = 0;
    for (int it = 0; it < setting_trace_GNIterations; it++) {
        float H = 1, b = 0, energy = 0;
        for (int idx = 0; idx < patternNum; idx++) {
            Vec3f hitColor = getInterpolatedElement33(frame->dI, (float)(bestU + rotatetPattern[idx][0]), (float)(bestV + rotatetPattern[idx][1]), wG[0]);

            if (!std::isfinite((float)hitColor[0])) {
                energy += 1e5;
                continue;
            }
            float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
            float dResdDist = dx * hitColor[1] + dy * hitColor[2]; ///< J = drk / dstep = (dIi / dpi) * (dpi / dstep)
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

            H += hw * dResdDist * dResdDist;
            b += hw * residual * dResdDist;
            energy += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
        }

        /// 由于图像是一个非凸性很强的函数
        if (energy > bestEnergy) {
            gnStepsBad++;

            /// 如果由于非凸性的原因导致了 energy 增加，使用 0.5 倍的回溯步长进行恢复
            /// 猜测这里不完全恢复的原因，尝试找到一个合适的step，使得问题可解
            stepBack *= 0.5;
            bestU = uBak + stepBack * dx;
            bestV = vBak + stepBack * dy;
            if (debugPrint)
                printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n", it, energy, H, b, stepBack, uBak, vBak, bestU, bestV);
        } else {
            gnStepsGood++;

            /// 在计算成功step后，还需要考虑gnstepsize的一个步长，防止二次函数的近似不足的情况发生
            float step = -gnstepsize * b / H;

            /// 在采用 倍数缩放 step后，还要保证每次的step的绝对值不能超过0.5个大小
            if (step < -0.5)
                step = -0.5;
            else if (step > 0.5)
                step = 0.5;

            if (!std::isfinite(step))
                step = 0;

            /// 保持当前状态和当前步长，用于后续能量增加时的回溯
            uBak = bestU;
            vBak = bestV;
            stepBack = step;

            bestU += step * dx;
            bestV += step * dy;
            bestEnergy = energy;

            if (debugPrint)
                printf("GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n", it, energy, H, b, step, uBak, vBak, bestU, bestV);
        }

        if (fabsf(stepBack) < setting_trace_GNThreshold)
            break;
    }

    /// 对优化后的能量，还是大于设定的阈值，则认为 ImmaturePoint 是外点，如果两次判断为外点，则置为OOB
    if (!(bestEnergy < energyTH * setting_trace_extraSlackOnTH)) {
        if (debugPrint)
            printf("OUTLIER!\n");

        lastTracePixelInterval = 0;
        lastTraceUV = Vec2f(-1, -1);
        if (lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
            return lastTraceStatus = ImmaturePointStatus::IPS_OOB; ///< 两次被判断为外点，置为OOB
        else
            return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
    }

    /// 根据bestU和bestV可以分别计算出 idepth，DSO中将极线导致的像素误差考虑在内，得到四个idepth，将最大的设为idepth_max，最小的设为idepth_min
    /// u = (pr[0] + Kt[0]*idepth) / (pr[2] + Kt[2]*idepth) ==> idepth = (u*pr[2] - pr[0]) / (Kt[0] - u*Kt[2])
    /// v = (pr[1] + Kt[1]*idepth) / (pr[2] + Kt[2]*idepth) ==> idepth = (v*pr[2] - pr[1]) / (Kt[1] - v*Kt[2])
    if (dx * dx > dy * dy) {
        idepth_min = (pr[2] * (bestU - errorInPixel * dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU - errorInPixel * dx));
        idepth_max = (pr[2] * (bestU + errorInPixel * dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU + errorInPixel * dx));
    } else {
        idepth_min = (pr[2] * (bestV - errorInPixel * dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV - errorInPixel * dy));
        idepth_max = (pr[2] * (bestV + errorInPixel * dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV + errorInPixel * dy));
    }
    if (idepth_min > idepth_max)
        std::swap<float>(idepth_min, idepth_max);

    /// 判断idepth_min 和 idepth_max 是否合法，这里不判断 idepth_min 的原因是什么，为什么不直接置为OOB
    if (!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max < 0)) {
        lastTracePixelInterval = 0;
        lastTraceUV = Vec2f(-1, -1);
        return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
    }

    lastTracePixelInterval = 2 * errorInPixel;              ///< 设置这个为下一次像素搜索范围的原因在于，idepth的不确定性是通过这个errorInPixel给出的
    lastTraceUV = Vec2f(bestU, bestV);                      ///< 上一次得到的最优位置
    return lastTraceStatus = ImmaturePointStatus::IPS_GOOD; ///< 上一次，使用极线搜索时，搜索的状态
}

float ImmaturePoint::getdPixdd(CalibHessian *HCalib, ImmaturePointTemporaryResidual *tmpRes, float idepth) {
    FrameFramePrecalc *precalc = &(host->targetPrecalc[tmpRes->target->idx]);
    const Vec3f &PRE_tTll = precalc->PRE_tTll;
    float drescale, u = 0, v = 0, new_idepth;
    float Ku, Kv;
    Vec3f KliP;

    projectPoint(this->u, this->v, idepth, 0, 0, HCalib, precalc->PRE_RTll, PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth);

    float dxdd = (PRE_tTll[0] - PRE_tTll[2] * u) * HCalib->fxl();
    float dydd = (PRE_tTll[1] - PRE_tTll[2] * v) * HCalib->fyl();
    return drescale * sqrtf(dxdd * dxdd + dydd * dydd);
}

float ImmaturePoint::calcResidual(CalibHessian *HCalib, const float outlierTHSlack, ImmaturePointTemporaryResidual *tmpRes, float idepth) {
    FrameFramePrecalc *precalc = &(host->targetPrecalc[tmpRes->target->idx]);

    float energyLeft = 0;
    const Eigen::Vector3f *dIl = tmpRes->target->dI;
    const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
    const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
    Vec2f affLL = precalc->PRE_aff_mode;

    for (int idx = 0; idx < patternNum; idx++) {
        float Ku, Kv;
        if (!projectPoint(this->u + patternP[idx][0], this->v + patternP[idx][1], idepth, PRE_KRKiTll, PRE_KtTll, Ku, Kv)) {
            return 1e10;
        }

        Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
        if (!std::isfinite((float)hitColor[0])) {
            return 1e10;
        }
        // if(benchmarkSpecialOption==5) hitColor = (getInterpolatedElement13BiCub(tmpRes->target->I, Ku, Kv, wG[0]));

        float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

        float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
        energyLeft += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
    }

    if (energyLeft > energyTH * outlierTHSlack) {
        energyLeft = energyTH * outlierTHSlack;
    }
    return energyLeft;
}

/**
 * @brief 计算 idepth 线性化点处的 能量的雅可比矩阵J和能量的海塞矩阵H
 *
 * @param HCalib            相机内参
 * @param outlierTHSlack    外点阈值
 * @param tmpRes            临时残差项
 * @param Hdd               输出的能量的海塞矩阵 (idepth) --> 线性化
 * @param bd                输出的雅可比矩阵 (idepth) --> 线性化
 * @param idepth            输入的线性化点 idepth
 * @return double   输出的能量值 E(idepth)
 */
double ImmaturePoint::linearizeResidual(CalibHessian *HCalib, const float outlierTHSlack, ImmaturePointTemporaryResidual *tmpRes, float &Hdd, float &bd,
                                        float idepth) {

    if (tmpRes->state_state == ResState::OOB) {
        tmpRes->state_NewState = ResState::OOB;
        return tmpRes->state_energy;
    }

    /// 获取 未成熟点 ph 的host 和 target 的预计算信息
    FrameFramePrecalc *precalc = &(host->targetPrecalc[tmpRes->target->idx]);

    float energyLeft = 0;
    const Eigen::Vector3f *dIl = tmpRes->target->dI; ///< 第target的0层 [能量值， 梯度值]
    const Mat33f &PRE_RTll = precalc->PRE_RTll;      ///< 优化之后的旋转矩阵Rth
    const Vec3f &PRE_tTll = precalc->PRE_tTll;       ///< 优化之后的平移向量tth
    Vec2f affLL = precalc->PRE_aff_mode;             ///< ath, bth

    for (int idx = 0; idx < patternNum; idx++) {
        int dx = patternP[idx][0];
        int dy = patternP[idx][1];

        float drescale, u, v, new_idepth;
        float Ku, Kv;
        Vec3f KliP;

        if (!projectPoint(this->u, this->v, idepth, dx, dy, HCalib, PRE_RTll, PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth)) {
            tmpRes->state_NewState = ResState::OOB;
            return tmpRes->state_energy;
        }

        Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0])); ///< 根据双线性插值计算 I, dI / dx, dI / dy

        if (!std::isfinite((float)hitColor[0])) {
            tmpRes->state_NewState = ResState::OOB;
            return tmpRes->state_energy;
        }

        /// 计算残差 rk = Ij[pj] - aji * Ii[pi] - bji
        float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

        float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual); ///< huber 核函数
        energyLeft += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);      ///< 能量值

        float dxInterp = hitColor[1] * HCalib->fxl(); ///< dx * fx
        float dyInterp = hitColor[2] * HCalib->fyl(); ///< dy * fy

        /// 获取残差对逆深度的雅可比矩阵 drk / ddpi (J * A)
        float d_idepth = derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale);

        hw *= weights[idx] * weights[idx];

        Hdd += (hw * d_idepth) * d_idepth; /// 近似E对逆深度的hessian
        bd += (hw * residual) * d_idepth;  /// 近似E对逆深度的Jacobian
    }

    /// 内点和外点的判别，使用energyTH 和 outlierTHSlack 判断
    if (energyLeft > energyTH * outlierTHSlack) {
        energyLeft = energyTH * outlierTHSlack;
        tmpRes->state_NewState = ResState::OUTLIER;
    } else {
        tmpRes->state_NewState = ResState::IN;
    }

    tmpRes->state_NewEnergy = energyLeft;
    return energyLeft;
}

} // namespace dso
