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
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include "util/settings.h"

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <sstream>

#include "IOWrapper/ImageRW.h"
#include "util/Undistort.h"

#if HAS_ZIPLIB
#include "zip.h"
#endif

#include <boost/thread.hpp>

using namespace dso;

/**
 * @brief 将dir目录内的所有文件的绝对路径放入files内
 *
 * @param dir 	输入的文件夹的绝对路径
 * @param files 输出的dir文件夹中的所有文件绝对路径，使用字典序排序
 * @return int 	输出的files的size，文件夹中的文件内容
 */
inline int getdir(std::string dir, std::vector<std::string> &files) {
    DIR *dp;             ///< 指向目录流的指针
    struct dirent *dirp; ///< 指向目录的指针

    /// opendir尝试打开dir目录，打开返回目录流指针，失败返回null
    if ((dp = opendir(dir.c_str())) == NULL) {
        return -1;
    }

    /// readdir读取目录流中的所有内容，并将除.和..以外的所有内容都放在files的vector里面
    while ((dirp = readdir(dp)) != NULL) {
        std::string name = std::string(dirp->d_name);

        if (name != "." && name != "..")
            files.push_back(name);
    }
    closedir(dp);

    /// 使用字典序进行files排列
    std::sort(files.begin(), files.end());

    /// files里面的文件名加上前缀，以表示绝对路径，用于后续读取
    if (dir.at(dir.length() - 1) != '/')
        dir = dir + "/";
    for (unsigned int i = 0; i < files.size(); i++) {
        if (files[i].at(0) != '/')
            files[i] = dir + files[i];
    }

    return files.size();
}

struct PrepImageItem {
    int id;
    bool isQueud;
    ImageAndExposure *pt;

    inline PrepImageItem(int _id) {
        id = _id;
        isQueud = false;
        pt = 0;
    }

    inline void release() {
        if (pt != 0)
            delete pt;
        pt = 0;
    }
};

class ImageFolderReader {
public:
    ImageFolderReader(std::string path, std::string calibFile, std::string gammaFile, std::string vignetteFile) {
        this->path = path;
        this->calibfile = calibFile;

#if HAS_ZIPLIB
        ziparchive = 0;
        databuffer = 0;
#endif

        isZipped = (path.length() > 4 && path.substr(path.length() - 4) == ".zip");

        /// 对zip格式的数据集，使用ziplib库进行读取
        if (isZipped) {
#if HAS_ZIPLIB
            int ziperror = 0;
            ziparchive = zip_open(path.c_str(), ZIP_RDONLY, &ziperror);
            if (ziperror != 0) {
                printf("ERROR %d reading archive %s!\n", ziperror, path.c_str());
                exit(1);
            }

            files.clear();
            int numEntries = zip_get_num_entries(ziparchive, 0);
            for (int k = 0; k < numEntries; k++) {
                const char *name = zip_get_name(ziparchive, k, ZIP_FL_ENC_STRICT);
                std::string nstr = std::string(name);
                if (nstr == "." || nstr == "..")
                    continue;
                files.push_back(name);
            }

            printf("got %d entries and %d files!\n", numEntries, (int)files.size());
            std::sort(files.begin(), files.end());
#else
            /// 当没有HAS_ZIPLIB定义，即没有ziplib库，则报错并退出
            printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
            exit(1);
#endif
        } else {
            /// 如果是文件夹形式的数据库，可以直接使用c语言的DIR文件流进行读取
            getdir(path, files);
        }

        /// 创建undistort去畸变核心类型，输入calibFile、gammaFile和vignetteFile，分别为仅小孔模型和光度模型
        undistort = Undistort::getUndistorterForFile(calibFile, gammaFile, vignetteFile);

        /// 这里的尺寸相关的变量会在undistort里面做一个清晰的界定
        widthOrg = undistort->getOriginalSize()[0];
        heightOrg = undistort->getOriginalSize()[1];
        width = undistort->getSize()[0];
        height = undistort->getSize()[1];

        // load timestamps if possible.
        loadTimestamps();
        printf("ImageFolderReader: got %d files in %s!\n", (int)files.size(), path.c_str());
    }
    ~ImageFolderReader() {
#if HAS_ZIPLIB
        if (ziparchive != 0)
            zip_close(ziparchive);
        if (databuffer != 0)
            delete databuffer;
#endif

        delete undistort;
    };

    Eigen::VectorXf getOriginalCalib() { return undistort->getOriginalParameter().cast<float>(); }
    Eigen::Vector2i getOriginalDimensions() { return undistort->getOriginalSize(); }

    void getCalibMono(Eigen::Matrix3f &K, int &w, int &h) {
        K = undistort->getK().cast<float>();
        w = undistort->getSize()[0];
        h = undistort->getSize()[1];
    }

    void setGlobalCalibration() {
        int w_out, h_out;
        Eigen::Matrix3f K;
        getCalibMono(K, w_out, h_out);
        setGlobalCalib(w_out, h_out, K);
    }

    int getNumImages() { return files.size(); }

    double getTimestamp(int id) {
        if (timestamps.size() == 0)
            return id * 0.1f;
        if (id >= (int)timestamps.size())
            return 0;
        if (id < 0)
            return 0;
        return timestamps[id];
    }

    void prepImage(int id, bool as8U = false) {}

    MinimalImageB *getImageRaw(int id) { return getImageRaw_internal(id, 0); }

    ImageAndExposure *getImage(int id, bool forceLoadDirectly = false) { return getImage_internal(id, 0); }

    inline float *getPhotometricGamma() {
        if (undistort == 0 || undistort->photometricUndist == 0)
            return 0;
        return undistort->photometricUndist->getG();
    }

    // undistorter. [0] always exists, [1-2] only when MT is enabled.
    Undistort *undistort;

private:
    MinimalImageB *getImageRaw_internal(int id, int unused) {
        if (!isZipped) {
            // CHANGE FOR ZIP FILE
            return IOWrap::readImageBW_8U(files[id]);
        } else {
#if HAS_ZIPLIB
            if (databuffer == 0)
                databuffer = new char[widthOrg * heightOrg * 6 + 10000];
            zip_file_t *fle = zip_fopen(ziparchive, files[id].c_str(), 0);
            long readbytes = zip_fread(fle, databuffer, (long)widthOrg * heightOrg * 6 + 10000);

            if (readbytes > (long)widthOrg * heightOrg * 6) {
                printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes,
                       (long)widthOrg * heightOrg * 6 + 10000, files[id].c_str());
                delete[] databuffer;
                databuffer = new char[(long)widthOrg * heightOrg * 30];
                fle = zip_fopen(ziparchive, files[id].c_str(), 0);
                readbytes = zip_fread(fle, databuffer, (long)widthOrg * heightOrg * 30 + 10000);

                if (readbytes > (long)widthOrg * heightOrg * 30) {
                    printf("buffer still to small (read %ld/%ld). abort.\n", readbytes,
                           (long)widthOrg * heightOrg * 30 + 10000);
                    exit(1);
                }
            }

            return IOWrap::readStreamBW_8U(databuffer, readbytes);
#else
            printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
            exit(1);
#endif
        }
    }
    
    /**
     * @brief 真正读取图像的操作函数，读取并进行光度矫正
     * 
     * @param id        输入的图像id
     * @param unused    在这个函数里面并没有使用，但是在后续的getImageRaw_internal函数也没有使用，估计是设计有误
     * @return ImageAndExposure* 返回校正后的图像和曝光时间的参数
     */
    ImageAndExposure *getImage_internal(int id, int unused) {
        /// 返回的是uint8的图像信息
        MinimalImageB *minimg = getImageRaw_internal(id, 0);
        
        /// 这一部分的undistort仅仅是进行了光度矫正而已
        ImageAndExposure *ret2 = undistort->undistort<unsigned char>(
            minimg, (exposures.size() == 0 ? 1.0f : exposures[id]), (timestamps.size() == 0 ? 0.0 : timestamps[id]));
        delete minimg;
        return ret2;
    }

    /**
     * @brief 如果在数据的相同路径下，存在times.txt文件，则进行读取
     * @details 将时间读取分为两种情况，并且针对曝光时间进行判断和矫正
     *  1. 仅时间戳判断
     *  2. 时间戳+曝光时间判断
     *  3. 如果曝光时间存在的话，对曝光时间的合理性进行判断和矫正
     *      3.1 如果曝光时间为0，则要求它的前方或后方的数据不为零，然后进行加和求平均的方式进行处理
     *      3.2 如果曝光时间为0，且不满足上述要求，则曝光时间读取失败，设置exposuresGood为false
     */
    inline void loadTimestamps() {
        std::ifstream tr;
        std::string timesFile = path.substr(0, path.find_last_of('/')) + "/times.txt";
        tr.open(timesFile.c_str());
        while (!tr.eof() && tr.good()) {
            std::string line;
            char buf[1000];
            tr.getline(buf, 1000);

            int id;
            double stamp;
            float exposure = 0;

            if (3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure)) {
                timestamps.push_back(stamp);
                exposures.push_back(exposure);
            }

            else if (2 == sscanf(buf, "%d %lf", &id, &stamp)) {
                timestamps.push_back(stamp);
                exposures.push_back(exposure);
            }
        }
        tr.close();

        // check if exposures are correct, (possibly skip)
        bool exposuresGood = ((int)exposures.size() == (int)getNumImages());
        for (int i = 0; i < (int)exposures.size(); i++) {
            if (exposures[i] == 0) {
                // fix!
                float sum = 0, num = 0;
                if (i > 0 && exposures[i - 1] > 0) {
                    sum += exposures[i - 1];
                    num++;
                }
                if (i + 1 < (int)exposures.size() && exposures[i + 1] > 0) {
                    sum += exposures[i + 1];
                    num++;
                }

                if (num > 0)
                    exposures[i] = sum / num;
            }

            if (exposures[i] == 0)
                exposuresGood = false;
        }

        if ((int)getNumImages() != (int)timestamps.size()) {
            printf("set timestamps and exposures to zero!\n");
            exposures.clear();
            timestamps.clear();
        }

        if ((int)getNumImages() != (int)exposures.size() || !exposuresGood) {
            printf("set EXPOSURES to zero!\n");
            exposures.clear();
        }

        printf("got %d images and %d timestamps and %d exposures.!\n", (int)getNumImages(), (int)timestamps.size(),
               (int)exposures.size());
    }

    std::vector<ImageAndExposure *> preloadedImages;
    std::vector<std::string> files;
    std::vector<double> timestamps;
    std::vector<float> exposures;

    int width, height;
    int widthOrg, heightOrg;

    std::string path;
    std::string calibfile;

    bool isZipped;

#if HAS_ZIPLIB
    zip_t *ziparchive;
    char *databuffer;
#endif
};
