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

#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "util/FrameShell.h"

namespace dso {

//@ 从ImmaturePoint构造函数, 不成熟点变地图点
PointHessian::PointHessian(const ImmaturePoint *const rawPoint, CalibHessian *Hcalib) {
    instanceCounter++;
    host = rawPoint->host; // 主帧
    hasDepthPrior = false;

    idepth_hessian = 0;
    maxRelBaseline = 0;
    numGoodResiduals = 0;

    u = rawPoint->u;
    v = rawPoint->v;
    assert(std::isfinite(rawPoint->idepth_max));

    my_type = rawPoint->my_type; // 似乎是显示用的

    setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min) * 0.5); ///< 将不成熟点的逆深度范围的均值作为逆深度
    setPointStatus(PointHessian::INACTIVE);                               ///< 设置为未激活状态

    int n = patternNum;
    memcpy(color, rawPoint->color, sizeof(float) * n);     ///< 将未成熟点的color拷贝给PointHessian's color
    memcpy(weights, rawPoint->weights, sizeof(float) * n); ///< 将未成熟点的weight拷贝给PointHessian's weights
    energyTH = rawPoint->energyTH;                         ///< 拷贝未成熟点的能量阈值

    efPoint = 0; // 指针=0
}

//@ 释放residual
void PointHessian::release() {
    for (unsigned int i = 0; i < residuals.size(); i++)
        delete residuals[i];
    residuals.clear();
}

//@ 设置固定线性化点位置的状态
// TODO 后面求nullspaces地方没看懂, 回头再看<2019.09.18> 数学原理是啥?
void FrameHessian::setStateZero(const Vec10 &state_zero) {
    //! 前六维位姿必须是0
    assert(state_zero.head<6>().squaredNorm() < 1e-20);

    this->state_zero = state_zero;

    //! 感觉这个nullspaces_pose就是 Adj_T
    //! Exp(Adj_T*zeta)=T*Exp(zeta)*T^{-1}
    // 全局转为局部的，左乘边右乘
    //! T_c_w * delta_T_g * T_c_w_inv = delta_T_l
    // TODO 这个是数值求导的方法么???
    for (int i = 0; i < 6; i++) {
        Vec6 eps;
        eps.setZero();
        eps[i] = 1e-3;
        SE3 EepsP = Sophus::SE3::exp(eps);
        SE3 EepsM = Sophus::SE3::exp(-eps);
        SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
        SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
        nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);
    }
    // nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
    // nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

    //? rethink
    // scale change
    SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
    w2c_leftEps_P_x0.translation() *= 1.00001;
    w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
    SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
    w2c_leftEps_M_x0.translation() /= 1.00001;
    w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
    nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);

    //? 仿射部分的零空间是怎么求出来的？
    nullspaces_affine.setZero();
    nullspaces_affine.topLeftCorner<2, 1>() = Vec2(1, 0);
    assert(ab_exposure > 0);
    nullspaces_affine.topRightCorner<2, 1>() = Vec2(0, expf(aff_g2l_0().a) * ab_exposure);
};

void FrameHessian::release() {
    // DELETE POINT
    // DELETE RESIDUAL
    for (unsigned int i = 0; i < pointHessians.size(); i++)
        delete pointHessians[i];
    for (unsigned int i = 0; i < pointHessiansMarginalized.size(); i++)
        delete pointHessiansMarginalized[i];
    for (unsigned int i = 0; i < pointHessiansOut.size(); i++)
        delete pointHessiansOut[i];
    for (unsigned int i = 0; i < immaturePoints.size(); i++)
        delete immaturePoints[i];

    pointHessians.clear();
    pointHessiansMarginalized.clear();
    pointHessiansOut.clear();
    immaturePoints.clear();
}

/**
 * @brief 计算金字塔的各层图像和梯度
 * @details
 *  1. 构建各层的能量图像金字塔 dIp[lvl][0]
 *  2. 构建各层的能量图梯度dIp[lvl][1] 和 dIp[lvl][2]
 *  3. 构建各层梯度平方和absSquaredGrad[lvl] --> 去除V影响的原图梯度平方和
 * @param color  光度矫正后的能量图像
 * @param HCalib 当梯度平方的计算需要灰度图像时，这里的Hcalib用于提供G-1函数的导数
 * @note 在计算梯度时，有个问题，就是x的梯度在0和w-1的位置计算都是不对的，计算时取得并不是同一行
 */
void FrameHessian::makeImages(float *color, CalibHessian *HCalib) {
    /// 对所有层金字塔 需要保存的图像内容进行 空间分配
    for (int i = 0; i < pyrLevelsUsed; i++) {
        dIp[i] = new Eigen::Vector3f[wG[i] * hG[i]];
        absSquaredGrad[i] = new float[wG[i] * hG[i]];
    }

    /// 将dIp第0层的指针拷贝给dI，dI代表的是金字塔第0层的梯度函数
    dI = dIp[0];

    int w = wG[0], h = hG[0];

    for (int i = 0; i < w * h; i++)
        dI[i][0] = color[i];

    /// 图像金字塔的构造方式，是对上一层的能量图像进行均值滤波，得到下一层的图像，四合一！
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
        int wl = wG[lvl], hl = hG[lvl];      ///< 该层图像大小
        Eigen::Vector3f *dI_l = dIp[lvl];    ///< 该层的dI
        float *dabs_l = absSquaredGrad[lvl]; ///< 该层需要保存的梯度平方和（去除V的原始图像的梯度平方和）

        /// 这部分是通过第一层开始计算的
        if (lvl > 0) {
            int lvlm1 = lvl - 1;                 ///< 金字塔上一层索引
            int wlm1 = wG[lvlm1];                ///< 金字塔上一层图像宽度
            Eigen::Vector3f *dI_lm = dIp[lvlm1]; ///< 上一层的图像梯度dI

            /// 遍历该层图像，对dI的0个位置进行均值滤波（能量值）
            for (int y = 0; y < hl; y++)
                for (int x = 0; x < wl; x++) {
                    dI_l[x + y * wl][0] = 0.25f * (dI_lm[2 * x + 2 * y * wlm1][0] + dI_lm[2 * x + 1 + 2 * y * wlm1][0] + dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                                                   dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
                }
        }

        /// 梯度的计算是第0层开始计算的，梯度+梯度的平方
        //! 从第二行开始，到倒数第二行结束，因为idx的初始化值为wl，但是这里有个问题，就是x的梯度在0和w-1的位置计算都是不对的！
        for (int idx = wl; idx < wl * (hl - 1); idx++) {
            float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);   ///< x方向梯度，能量差值的均值
            float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]); ///< y方向梯度，能量差值的均值

            if (!std::isfinite(dx))
                dx = 0;
            if (!std::isfinite(dy))
                dy = 0;

            /// 计算能量图像上的梯度和梯度的平方
            dI_l[idx][1] = dx;               ///< dI的第一个位置为，x方向梯度
            dI_l[idx][2] = dy;               ///< dI的第二个位置为，y方向梯度
            dabs_l[idx] = dx * dx + dy * dy; ///< 梯度平方和，顾名思义

            /// 去除V(x)后的，原始图像的梯度平方和
            if (setting_gammaWeightsPixelSelect == 1 && HCalib != 0) {
                float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
                dabs_l[idx] *= gw * gw; // convert to gradient of original color space (before removing response).
            }
        }
    }
}

/**
 * @brief 设置host帧和target帧之间的预计算量
 * @details
 *  1. host和target优化之前的位姿变换（线性化点处的） Tth0, Rth0, tth0
 *  2. host和target优化之后的位姿变换 Tth, Rth, tth
 *  3. 优化后，两帧之间的距离 distanceLL
 *  4. host帧上的像素变换到target帧上的中间量 KRKinv RK_inv Kt
 *
 * @param host      输入的host帧
 * @param target    输入的target帧
 * @param HCalib    输入的相机内参
 */
void FrameFramePrecalc::set(FrameHessian *host, FrameHessian *target, CalibHessian *HCalib) {
    this->host = host;
    this->target = target;

    /// 获取滑动窗口优化之前的 Tth
    SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse(); ///< Tth0
    PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();                                   ///< Rth0
    PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();                                      ///< tth0

    /// 获取滑动窗口优化之后的 Tth
    SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld; ///< Tth
    PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();         ///< Rth
    PRE_tTll = (leftToLeft.translation()).cast<float>();            ///< tth
    distanceLL = leftToLeft.translation().norm();                   ///< 两帧之间的距离

    Mat33f K = Mat33f::Zero();
    K(0, 0) = HCalib->fxl();
    K(1, 1) = HCalib->fyl();
    K(0, 2) = HCalib->cxl();
    K(1, 2) = HCalib->cyl();
    K(2, 2) = 1;
    PRE_KRKiTll = K * PRE_RTll * K.inverse(); ///< K * Rth* K_inv ---> host帧上的像素点变换target上的中间量
    PRE_RKiTll = PRE_RTll * K.inverse();      ///< Rth * K_inv ---> host帧上的像素点变换target上的中间量
    PRE_KtTll = K * PRE_tTll;                 ///< K * tth ---> host帧上的像素点变换target上的中间量

    // 光度仿射值 ath 和 bth (相对量)
    PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
    PRE_b0_mode = host->aff_g2l_0().b; ///< bh (绝对量)
}

} // namespace dso
