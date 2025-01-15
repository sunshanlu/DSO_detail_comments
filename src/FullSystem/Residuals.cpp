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
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>

#include "FullSystem/ResidualProjections.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso {
int PointFrameResidual::instanceCounter = 0;

long runningResID = 0;

PointFrameResidual::PointFrameResidual() {
    assert(false);
    instanceCounter++;
}

PointFrameResidual::~PointFrameResidual() {
    assert(efResidual == 0);
    instanceCounter--;
    delete J;
}

PointFrameResidual::PointFrameResidual(PointHessian *point_, FrameHessian *host_, FrameHessian *target_)
    : point(point_)
    , host(host_)
    , target(target_) {
    efResidual = 0;
    instanceCounter++;
    resetOOB();
    J = new RawResidualJacobian(); // 各种雅克比
    assert(((long)J) % 16 == 0);   // 16位对齐

    isNew = true;
}

/**
 * @brief 残差的线性化部分，求解 残差对不同参数的雅可比矩阵 (FEJ)
 * @details
 *  1. 将残差对参数求导分开计算 drk / dpj * (dpj / d(delta_Tth(FEJ), delta_dpi(FEJ), delta_K(非FEJ)))
 *  2. 其中，针对后部分pj对参数的导数，在一个pattern内，仅使用了pi根据 Tth(FEJ)，delta_dpi(FEJ)和delta_K(非FEJ)状态投影过去的pj，没有对所有pattern进行求值
 *  3. 针对前部分rk对pj的导数，在一个pattern里面，对不同的pj进行了求导，不过这了在求导的过程中，可能考虑到图像的强非凸性的特点，使用非FEJ的参数进行的投影
 *  4. 后续，残差部分还对 sum(drk / dpj * drk / d(-aji)) ... 进行了保存，不清楚这里到底是用来干什么的？
 *  5. 维护了，使用非FEJ构造的残差rk（pattern中点分开存储），最后返回由于pattern中的rk构造的能量值
 *
 * @param HCalib    输入的相机参数
 * @return double   输出的由非FEJ参数投影构造的能量值
 */
double PointFrameResidual::linearize(CalibHessian *HCalib) {
    state_NewEnergyWithOutlier = -1;

    if (state_state == ResState::OOB) {
        state_NewState = ResState::OOB;
        return state_energy;
    }

    FrameFramePrecalc *precalc = &(host->targetPrecalc[target->idx]);
    float energyLeft = 0;

    const Eigen::Vector3f *dIl = target->dI;          ///< target 第0层上的能量值和图像梯度
    const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll; ///< 预计算的K * Rth * Kinv
    const Vec3f &PRE_KtTll = precalc->PRE_KtTll;      ///< 预计算的K * tth
    const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;   ///< 预计算的Rth0
    const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;    ///< 预计算的tth0
    const float *const color = point->color;          ///< 获取point的host帧上颜色
    const float *const weights = point->weights;      ///< 获取point在host帧上的权重（与host帧上的像素梯度有关）
    Vec2f affLL = precalc->PRE_aff_mode;              ///< ath 和 bth
    float b0 = precalc->PRE_b0_mode;                  ///< bh0

    /// x=0时候求几何的导数, 使用FEJ!! ,逆深度没有使用FEJ
    Vec6f d_xi_x, d_xi_y;
    Vec4f d_C_x, d_C_y;
    float d_d_x, d_d_y;
    {
        /// depth_tar / detph_hos, norm_target_u, norm_target_v, idepth_target
        float drescale, u, v, new_idepth;

        /// target_u, target_v
        float Ku, Kv;

        /// host_norm_point
        Vec3f KliP;

        /// 如果，投影不在图像里, 则返回OOB
        if (!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0, HCalib, PRE_RTll_0, PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth)) {
            state_NewState = ResState::OOB;
            return state_energy;
        }

        centerProjectedTo = Vec3f(Ku, Kv, new_idepth);

        /// point_target 对逆深度求导 像素点对host上逆深度求导 (由于乘了SCALE_IDEPTH倍，这里有雅可比缩放)
        d_d_x = drescale * (PRE_tTll_0[0] - PRE_tTll_0[2] * u) * SCALE_IDEPTH * HCalib->fxl(); ///< dpj[0] / ddpi
        d_d_y = drescale * (PRE_tTll_0[1] - PRE_tTll_0[2] * v) * SCALE_IDEPTH * HCalib->fyl(); ///< dpj[1] / ddpi

        /// duj / dfx = 1 / Pz'* Px * (R20*Px'/Pz' - R00) + uj
        /// duj / dfy = 1 / Pz' * Py * fx / fy * (R21*Px'/Pz' - R01)
        /// duj / dcx = 1 / Pz'* (R20 * Px' / Pz' - R00) + 1
        /// duj / dcy = 1 / Pz' * fx / fy * (R21 * Px' / Pz' - R01)
        d_C_x[2] = drescale * (PRE_RTll_0(2, 0) * u - PRE_RTll_0(0, 0));
        d_C_x[3] = HCalib->fxl() * drescale * (PRE_RTll_0(2, 1) * u - PRE_RTll_0(0, 1)) * HCalib->fyli();
        d_C_x[0] = KliP[0] * d_C_x[2];
        d_C_x[1] = KliP[1] * d_C_x[3];

        /// dvj / dfx = 1 / Pz' * Px * fy / fy * (R20 * Py' / Pz' - R10)
        /// dvj / dfy = 1 / Pz' *Py * (R21 * Py' / Pz' - R11) + vj
        /// dvj / dcx = 1 / Pz' * fy / fy * (R20 * Py' / Pz' - R10)
        /// dvj / dcy = 1 / Pz' * (R21 * Py' / Pz' - R11) + 1
        d_C_y[2] = HCalib->fyl() * drescale * (PRE_RTll_0(2, 0) * v - PRE_RTll_0(1, 0)) * HCalib->fxli();
        d_C_y[3] = drescale * (PRE_RTll_0(2, 1) * v - PRE_RTll_0(1, 1));
        d_C_y[0] = KliP[0] * d_C_y[2];
        d_C_y[1] = KliP[1] * d_C_y[3];

        /// 加上剩余部分，并进行尺度缩放
        d_C_x[0] = (d_C_x[0] + u) * SCALE_F;
        d_C_x[1] *= SCALE_F;
        d_C_x[2] = (d_C_x[2] + 1) * SCALE_C;
        d_C_x[3] *= SCALE_C;

        d_C_y[0] *= SCALE_F;
        d_C_y[1] = (d_C_y[1] + v) * SCALE_F;
        d_C_y[2] *= SCALE_C;
        d_C_y[3] = (d_C_y[3] + 1) * SCALE_C;

        /// duj / d delta_Tth
        d_xi_x[0] = new_idepth * HCalib->fxl();
        d_xi_x[1] = 0;
        d_xi_x[2] = -new_idepth * u * HCalib->fxl();
        d_xi_x[3] = -u * v * HCalib->fxl();
        d_xi_x[4] = (1 + u * u) * HCalib->fxl();
        d_xi_x[5] = -v * HCalib->fxl();

        /// dvj / d delta_Tth
        d_xi_y[0] = 0;
        d_xi_y[1] = new_idepth * HCalib->fyl();
        d_xi_y[2] = -new_idepth * v * HCalib->fyl();
        d_xi_y[3] = -(1 + v * v) * HCalib->fyl();
        d_xi_y[4] = u * v * HCalib->fyl();
        d_xi_y[5] = u * HCalib->fyl();
    }

    {
        J->Jpdxi[0] = d_xi_x; ///< duj / d delta_Tth
        J->Jpdxi[1] = d_xi_y; ///< dvj / d delta_Tth
        J->Jpdc[0] = d_C_x;   ///< duj / dK
        J->Jpdc[1] = d_C_y;   ///< dvj / dK
        J->Jpdd[0] = d_d_x;   ///< duj / ddpi
        J->Jpdd[1] = d_d_y;   ///< dvj / ddpi
    }

    float JIdxJIdx_00 = 0, JIdxJIdx_11 = 0, JIdxJIdx_10 = 0;
    float JabJIdx_00 = 0, JabJIdx_01 = 0, JabJIdx_10 = 0, JabJIdx_11 = 0;
    float JabJab_00 = 0, JabJab_01 = 0, JabJab_11 = 0;

    float wJI2_sum = 0;

    for (int idx = 0; idx < patternNum; idx++) {
        float Ku, Kv;

        /// 上面求解 dpj / d delta_Tth(FEJ) 和 dpj / dK (非FEJ) 和 dpj / ddpi (FEJ)
        /// 下面求解 drk / dpj (非FEJ) 和 drk / d(-aji) 和 drk / d(-bji)
        /// 并且对pattern的雅可比矩阵进行了加和处理 --> 但是由于pattern中点是不同的，并不清楚这具体代表的什么意义（意义不明）
        if (!projectPoint(point->u + patternP[idx][0], point->v + patternP[idx][1], point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv)) {
            state_NewState = ResState::OOB;
            return state_energy;
        }

        /// 像素坐标，维护了pattern中的像素坐标pj
        projectedTo[idx][0] = Ku;
        projectedTo[idx][1] = Kv;

        Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));          ///< Ij[pj], u方向梯度，v方向梯度
        float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]); ///< rk，残差

        /// drk / d(-aji_0)，在初次线性化点处的雅可比矩阵（FEJ）
        float drdA = (color[idx] - b0);

        if (!std::isfinite((float)hitColor[0])) {
            state_NewState = ResState::OOB;
            return state_energy;
        }

        /// 这里权重考虑了 pi 在Ii梯度下的影响，也考虑了 pj 在Ij梯度下的影响，但是根据雅可比矩阵来看，貌似pi部分的像素梯度并不会影响整个优化过程
        /// 个人猜测，这里先假设 已经优化到pi 和 pj部分对应上了，
        /// 那么pi的梯度和pj的梯度应该差不多，但是如果这部分对应上的pi的梯度较大，且是外点的情况下，说明优化方向已经错误了
        float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
        w = 0.5f * (w + weights[idx]); ///< pi的梯度和pj的梯度各考虑了0.5

        /// 根据 添加权重信息和huber和函数，构造能量值
        float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
        energyLeft += w * w * hw * residual * residual * (2 - hw);

        {
            if (hw < 1)
                hw = sqrtf(hw);
            hw = hw * w;

            hitColor[1] *= hw; ///< 在uj部分的梯度 * 权重
            hitColor[2] *= hw; ///< 在vj部分的梯度 * 权重

            J->resF[idx] = residual * hw; ///< 残差部分 * 权重（梯度权重 * huber 权重）

            J->JIdx[0][idx] = hitColor[1]; ///< drk / duj --> 针对图像梯度部分，不进行FEJ，线性化误差太大
            J->JIdx[1][idx] = hitColor[2]; ///< drk / dvj --> 针对图像梯度部分，不进行FEJ，线性化误差太大

            J->JabF[0][idx] = drdA * hw; ///< drk / d(-aji_0) --> 对于光度仿射参数aji，进行FEJ固定
            J->JabF[1][idx] = hw;        ///< drk / d(-bji_0) --> 对于光度仿射参数bji，进行FEJ固定

            JIdxJIdx_00 += hitColor[1] * hitColor[1]; ///< sum((drk / duj) ^ 2)
            JIdxJIdx_11 += hitColor[2] * hitColor[2]; ///< sum((drk / dvj) ^ 2)
            JIdxJIdx_10 += hitColor[1] * hitColor[2]; ///< sum((drk / duj) * (drk / dvj))

            JabJIdx_00 += drdA * hw * hitColor[1]; ///< sum((drk / d(-aji_0)) * (drk / duj))
            JabJIdx_01 += drdA * hw * hitColor[2]; ///< sum((drk / d(-aji_0)) * (drk / dvj))
            JabJIdx_10 += hw * hitColor[1];        ///< sum((drk / d(-bji_0)) * (drk / duj))
            JabJIdx_11 += hw * hitColor[2];        ///< sum((drk / d(-bji_0)) * (drk / dvj))

            JabJab_00 += drdA * drdA * hw * hw; ///< sum((drk / d(-aji_0)) * (drk / d(-aji_0)))
            JabJab_01 += drdA * hw * hw;        ///< sum((drk / d(-aji_0)) * (drk / d(-bji_0)))
            JabJab_11 += hw * hw;               ///< sum((drk / d(-bji_0)) * (drk / d(-bji_0)))

            wJI2_sum += hw * hw * (hitColor[1] * hitColor[1] + hitColor[2] * hitColor[2]); // 梯度平方

            if (setting_affineOptModeA < 0) ///< 小于0代表固定 a 不进行优化
                J->JabF[0][idx] = 0;
            if (setting_affineOptModeB < 0) ///< 小于0代表固定 b 不进行优化
                J->JabF[1][idx] = 0;
        }
    }

    J->JIdx2(0, 0) = JIdxJIdx_00;
    J->JIdx2(0, 1) = JIdxJIdx_10;
    J->JIdx2(1, 0) = JIdxJIdx_10;
    J->JIdx2(1, 1) = JIdxJIdx_11;
    J->JabJIdx(0, 0) = JabJIdx_00;
    J->JabJIdx(0, 1) = JabJIdx_01;
    J->JabJIdx(1, 0) = JabJIdx_10;
    J->JabJIdx(1, 1) = JabJIdx_11;
    J->Jab2(0, 0) = JabJab_00;
    J->Jab2(0, 1) = JabJab_01;
    J->Jab2(1, 0) = JabJab_01;
    J->Jab2(1, 1) = JabJab_11;

    state_NewEnergyWithOutlier = energyLeft; ///< 用于后续对帧的能量阈值进行设置

    ///? 大于 host 或者 target 中的frameEnergyTH 则被认为是外点 或者 wJI2_sum < 2（但是弄不清楚这里的意义是什么）
    if (energyLeft > std::max<float>(host->frameEnergyTH, target->frameEnergyTH) || wJI2_sum < 2) {
        energyLeft = std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
        state_NewState = ResState::OUTLIER;
    } else {
        state_NewState = ResState::IN;
    }

    state_NewEnergy = energyLeft;
    return energyLeft;
}

void PointFrameResidual::debugPlot() {
    if (state_state == ResState::OOB)
        return;
    Vec3b cT = Vec3b(0, 0, 0);

    if (freeDebugParam5 == 0) {
        float rT = 20 * sqrt(state_energy / 9);
        if (rT < 0)
            rT = 0;
        if (rT > 255)
            rT = 255;
        cT = Vec3b(0, 255 - rT, rT);
    } else {
        if (state_state == ResState::IN)
            cT = Vec3b(255, 0, 0);
        else if (state_state == ResState::OOB)
            cT = Vec3b(255, 255, 0);
        else if (state_state == ResState::OUTLIER)
            cT = Vec3b(0, 0, 255);
        else
            cT = Vec3b(255, 255, 255);
    }

    for (int i = 0; i < patternNum; i++) {
        if ((projectedTo[i][0] > 2 && projectedTo[i][1] > 2 && projectedTo[i][0] < wG[0] - 3 && projectedTo[i][1] < hG[0] - 3))
            target->debugImage->setPixel1((float)projectedTo[i][0], (float)projectedTo[i][1], cT);
    }
}

/**
 * @brief 拷贝PointFrameResidual在线性化中构建的雅可比矩阵，并进一步计算残差Hfd
 * @details
 *  1. 拷贝了在线性化过程中求解的雅可比矩阵和求解雅可比矩阵的中间量
 *  2. 进一步计算了drk / dstate (位姿部分 、 仿射参数部分)
 *  3. 仅对内点做雅可比矩阵拷贝和计算操作
 * 
 * @see EFResidual::takeDataF
 * 
 * @param copyJacobians
 */
void PointFrameResidual::applyRes(bool copyJacobians) {
    if (copyJacobians) {
        /// 判断上一次的残差状态，如果为OOB，则return
        if (state_state == ResState::OOB) {
            assert(!efResidual->isActiveAndIsGoodNEW);
            return;
        }

        /// 判断线性化后的点是否为内点
        if (state_NewState == ResState::IN) {
            efResidual->isActiveAndIsGoodNEW = true; ///< 残差的可参与优化标识
            efResidual->takeDataF();                 ///< 计算 Hfd
        } else {
            efResidual->isActiveAndIsGoodNEW = false; ///< 如果不是内点，则残差不能参与优化
        }
    }

    setState(state_NewState);
    state_energy = state_NewEnergy;
}
} // namespace dso
