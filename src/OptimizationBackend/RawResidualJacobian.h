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
struct RawResidualJacobian {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VecNRf resF; ///< 每个patch的8个残差

    Vec6f Jpdxi[2]; ///< dpjf / dTth
    VecCf Jpdc[2];  ///< dpjf / dC
    Vec2f Jpdd;     ///< dpjf / ddpi
    VecNRf JIdx[2]; ///< drk / dpj (8个pattern分开存储)
    VecNRf JabF[2]; ///< drk / dab (8个pattern分开存储)

    Mat22f JIdx2;   ///< sum((drk / dpj)^T * (drk / dpj))，求解雅可比中间量
    Mat22f JabJIdx; ///< sum((drk / dab)^T * (drk / dpj))，求解雅可比中间量
    Mat22f Jab2;    ///< sum((drk / dab)^T * (drk / dab))，求解雅可比中间量
};
} // namespace dso
