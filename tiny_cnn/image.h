/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY 
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <vector>
#include <fstream>
#include <cstdint>
#include <algorithm>

namespace tiny_cnn {


class image {
public:
    typedef unsigned char intensity_t;

    image() : width_(0), height_(0) {}

    image(size_t width, size_t height) : width_(width), height_(height), data_(width * height, 0) {}

    image(const image& rhs) : width_(rhs.width_), height_(rhs.height_), data_(rhs.data_) {}

    image(const image&& rhs) : width_(rhs.width_), height_(rhs.height_), data_(std::move(rhs.data_)) {}

    image& operator = (const image& rhs) {
        width_ = rhs.width_;
        height_ = rhs.height_;
        data_ = rhs.data_;
        return *this;
    }

    image& operator = (const image&& rhs) {
        width_ = rhs.width_;
        height_ = rhs.height_;
        data_ = std::move(rhs.data_);
        return *this;
    }

    void write(const std::string& path) const { // WARNING: This is OS dependent (writes of bytes with reinterpret_cast depend on endianness)
        std::ofstream ofs(path.c_str(), std::ios::binary | std::ios::out);

        if (!is_little_endian())
            throw nn_error("image::write for bit-endian is not supported");

        const uint32_t line_pitch = ((width_ + 3) / 4) * 4;
        const uint32_t header_size = 14 + 12 + 256 * 3;
        const uint32_t data_size = line_pitch * height_;
        
        // file header(14 byte)
        const uint16_t file_type = ('M' << 8) | 'B';
        const uint32_t file_size = header_size + data_size;
        const uint32_t reserved = 0;
        const uint32_t offset_bytes = header_size;

        ofs.write(reinterpret_cast<const char*>(&file_type), 2);
        ofs.write(reinterpret_cast<const char*>(&file_size), 4);
        ofs.write(reinterpret_cast<const char*>(&reserved), 4);
        ofs.write(reinterpret_cast<const char*>(&offset_bytes), 4);

        // info header(12byte)
        const uint32_t info_header_size = 12;
        const int16_t width = static_cast<int16_t>(width_);
        const int16_t height = static_cast<int16_t>(height_);
        const uint16_t planes = 1;
        const uint16_t bit_count = 8;

        ofs.write(reinterpret_cast<const char*>(&info_header_size), 4);
        ofs.write(reinterpret_cast<const char*>(&width), 2);
        ofs.write(reinterpret_cast<const char*>(&height), 2);
        ofs.write(reinterpret_cast<const char*>(&planes), 2);
        ofs.write(reinterpret_cast<const char*>(&bit_count), 2);

        // color palette (256*3byte)
        for (int i = 0; i < 256; i++) {
			const char v = static_cast<const char>(i);
            ofs.write(&v, 1);//R
            ofs.write(&v, 1);//G
            ofs.write(&v, 1);//B
        }

        // data
        for (size_t i = 0; i < height_; i++) {
            ofs.write(reinterpret_cast<const char*>(&data_[(height_ - 1 - i) * width_]), width_);
            if (line_pitch != width_) {
                uint32_t dummy = 0;
                ofs.write(reinterpret_cast<const char*>(&dummy), line_pitch - width_);
            }
        }
    }

    void resize(size_t width, size_t height) {
        data_.resize(width * height);
        width_ = width;
        height_ = height;
    }

    void fill(intensity_t value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    intensity_t& at(size_t x, size_t y) {
        assert(x < width_);
        assert(y < height_);
        return data_[y * width_ + x];
    }

    const intensity_t& at(size_t x, size_t y) const {
        assert(x < width_);
        assert(y < height_);
        return data_[y * width_ + x];
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    const std::vector<intensity_t>& data() const { return data_; }
private:
    size_t width_;
    size_t height_;
    std::vector<intensity_t> data_;
};

/**
 * visualize 1d-vector
 *
 * @example
 *
 * vec:[1,5,3]
 *
 * img:
 *   ----------
 *   -11-55-33-
 *   -11-55-33-
 *   ----------
 **/
inline image vec2image(const vec_t& vec, int block_size = 2, int max_cols = 20)
{
    if (vec.empty())
        throw nn_error("failed to visialize image: vector is empty");

    image img;
    const layer_size_t border_width = 1;
    const size_t cols = vec.size() >= (size_t)max_cols ? (size_t)max_cols : vec.size();
    const size_t rows = (vec.size() - 1) / cols + 1;
    const size_t pitch = block_size + border_width;
    const size_t width = pitch * cols + border_width;
    const size_t height = pitch * rows + border_width;
    const image::intensity_t bg_color = 255;
    size_t current_idx = 0;

    img.resize(width, height);
    img.fill(bg_color);

	std::pair<double, double> minmax;
	minmax.first = *std::min_element(vec.begin(), vec.end());
	minmax.second = *std::max_element(vec.begin(), vec.end());

    for (unsigned int r = 0; r < rows; r++) {
        int topy = pitch * r + border_width;

        for (unsigned int c = 0; c < cols; c++, current_idx++) {
            int leftx = pitch * c + border_width;
            const float_t src = vec[current_idx];
            image::intensity_t dst
                = static_cast<image::intensity_t>(rescale(src, minmax.first, minmax.second, 0, 255));

            for (int y = 0; y < block_size; y++)
              for (int x = 0; x < block_size; x++)
                img.at(x + leftx, y + topy) = dst;

            if (current_idx == vec.size()) return img;
        }
    }
    return img;
}

/**
 * visualize 1d-vector
 *
 * @example
 *
 * vec:[5,2,1,3,6,3,0,9,8,7,4,2] maps:[width=2,height=3,depth=2]
 *
 * img:
 *  -------
 *  -52-09-
 *  -13-87-
 *  -63-42-
 *  -------
 **/
inline image vec2image(const vec_t& vec, const index3d<layer_size_t>& maps) {
    if (vec.empty())
        throw nn_error("failed to visualize image: vector is empty");
    if (vec.size() != maps.size())
        throw nn_error("failed to visualize image: vector size invalid");

    const layer_size_t border_width = 1;
    const int pitch = maps.width_ + border_width;
    const int width = maps.depth_ * pitch + border_width;
    const int height = maps.height_ + 2 * border_width;
    const image::intensity_t bg_color = 255;
    image img;

    img.resize(width, height);
    img.fill(bg_color);

	std::pair<double, double> minmax;
	minmax.first = *std::min_element(vec.begin(), vec.end());
	minmax.second = *std::max_element(vec.begin(), vec.end());

    for (layer_size_t c = 0; c < maps.depth_; ++c) {
        const size_t top = border_width;
        const size_t left = c * pitch + border_width;

        for (layer_size_t y = 0; y < maps.height_; ++y) {
            for (layer_size_t x = 0; x < maps.width_; ++x) {
                const float_t val = vec[maps.get_index(x, y, c)];

                img.at(left + x, top + y)
                    = static_cast<image::intensity_t>(rescale(val, minmax.first, minmax.second, 0, 255));
            }
        }
    }
    return img;
}

} // namespace tiny_cnn
