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

#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/EnergyFunctional.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

/**
 * @brief 根据RawResidualJacobian 线性化后的中间量，计算 Hfd --> sum((drk / dTth)^T * (drk / ddpi))
 * @details
 *  1. 首先，将PointFrameResidual线性化的J拿到，里面包含一些计算需要的中间状态
 *  2. 在拿到的J基础上，构建 (drk / dTth)^T * (drk / ddpi) 和 (drk / dab)^T * (drk / ddpi)
 *
 * @note 针对 drk / dpj 时，由于图像非凸性，这时会使用pattern进行投影计算（非FEJ）
 * @note 针对 dpj / dstate 时，会使用 仅pi状态，使用state_zero的方式投影计算（FEJ）
 */
void EFResidual::takeDataF() {
    /// 将EFResidual 维护的雅可比 和 PointFrameResidual维护的雅可比进行交换
    std::swap<RawResidualJacobian *>(J, data->J);

    /// sum((drk / dpj)^T * (drk / dpj)) * (drk / ddpi)，前面部分为非FEJ，后部分为FEJ状态
    Vec2f JI_JI_Jd = J->JIdx2 * J->Jpdd;

    /// sum((drk / dTth)^T * (drk / ddpi))
    for (int i = 0; i < 6; i++)
        JpJdF[i] = J->Jpdxi[0][i] * JI_JI_Jd[0] + J->Jpdxi[1][i] * JI_JI_Jd[1];

    /// sum(drk / dab)^T * (drk / ddpi)
    JpJdF.segment<2>(6) = J->JabJIdx * J->Jpdd;
}

//@ 从 FrameHessian 中提取数据
void EFFrame::takeData() {
    prior = data->getPrior().head<8>();                                 // 得到先验状态, 主要是光度仿射变换
    delta = data->get_state_minus_stateZero().head<8>();                // 状态与FEJ零状态之间差
    delta_prior = (data->get_state() - data->getPriorZero()).head<8>(); // 状态与先验之间的差 //? 可先验是0啊?

    assert(data->frameID != -1);

    frameID = data->frameID; // 所有帧的ID序号
}

/**
 * @brief 构建先验 prior 和 当前逆深度和线性化点之间的差值delta
 * @details
 * 	1. 根据 预设的逆深度先验值 --> 构建先验 prior
 * 	2. 如何 优化器设置了 移除位姿先验的状态 --> 设置当前 prior 为 0
 * 	3. 构建 delta --> 当前估计的逆深度 - 线性化点处的状态
 */
void EFPoint::takeData() {
    priorF = data->hasDepthPrior ? setting_idepthFixPrior * SCALE_IDEPTH * SCALE_IDEPTH : 0;
    if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR)
        priorF = 0;

    // TODO 每次都更新线性化点，这不一直是零？？
    deltaF = data->idepth - data->idepth_zero;
}

/**
 * @brief 根据新状态残差和FEJ的雅可比矩阵，求解线性化点处的残差
 * @details
 *  1. 获取残差对应host和target之间的帧状态增量
 *  2. 根据FEJ部分的dpj / dstate 和 非FEJ部分的 drk / dpj 计算delta_xj, delta_yj。并读取delta_a和delta_b
 *  3. 根据FEJ，和新状态下的rk，计算线性化状态下的rk
 * @param ef 输入的EnergyFunctional --> 用于提供帧状态增量和内参状态增量
 */
void EFResidual::fixLinearizationF(EnergyFunctional *ef) {
    /// 获取残差对应host和target之间的帧状态增量
    Vec8f dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX];

    /// 根据FEJ部分的dpj / dstate 和 非FEJ部分的 drk / dpj 计算delta_xj, delta_yj。并读取delta_a和delta_b
    __m128 Jp_delta_x = _mm_set1_ps(J->Jpdxi[0].dot(dp.head<6>()) + J->Jpdc[0].dot(ef->cDeltaF) + J->Jpdd[0] * point->deltaF);
    __m128 Jp_delta_y = _mm_set1_ps(J->Jpdxi[1].dot(dp.head<6>()) + J->Jpdc[1].dot(ef->cDeltaF) + J->Jpdd[1] * point->deltaF);
    __m128 delta_a = _mm_set1_ps((float)(dp[6]));
    __m128 delta_b = _mm_set1_ps((float)(dp[7]));

    /// 根据FEJ，和新状态下的rk，计算线性化状态下的rk
    for (int i = 0; i < patternNum; i += 4) {
        __m128 rtz = _mm_load_ps(((float *)&J->resF) + i);
        rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(J->JIdx)) + i), Jp_delta_x));
        rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(J->JIdx + 1)) + i), Jp_delta_y));
        rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(J->JabF)) + i), delta_a));
        rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(J->JabF + 1)) + i), delta_b));
        _mm_store_ps(((float *)&res_toZeroF) + i, rtz);
    }

    isLinearized = true;
}

} // namespace dso
