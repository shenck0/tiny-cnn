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
#include "util.h"
#include "partial_connected_layer.h"
#include "image.h"

namespace tiny_cnn {


template<typename Activation>
class average_pooling_layer : public partial_connected_layer<Activation> {
public:
    typedef partial_connected_layer<Activation> Base;
    CNN_USE_LAYER_MEMBERS;

    average_pooling_layer(layer_size_t in_width, layer_size_t in_height, layer_size_t in_channels, layer_size_t pooling_size)
    : Base(in_width * in_height * in_channels, 
           in_width * in_height * in_channels / sqr(pooling_size), 
           in_channels, in_channels, 1.0 / sqr(pooling_size)),
      in_(in_width, in_height, in_channels), 
      out_(in_width/pooling_size, in_height/pooling_size, in_channels)
    {
        if ((in_width % pooling_size) || (in_height % pooling_size))
            pooling_size_mismatch(in_width, in_height, pooling_size);

        init_connection(pooling_size);
#ifdef CNN_USE_SHRINK_LAYER
		this->shrink_after_init();
#endif
    }

    image<> output_to_image(size_t worker_index = 0) const {
        return vec2image(output_[worker_index], out_);
    }

    index3d<layer_size_t> in_shape() const override { return in_; }
    index3d<layer_size_t> out_shape() const override { return out_; }
    std::string layer_type() const override { return "ave-pool"; }

private:
    void init_connection(layer_size_t pooling_size) {
        for (layer_size_t c = 0; c < in_.depth_; ++c)
            for (layer_size_t y = 0; y < in_.height_; y += pooling_size)
                for (layer_size_t x = 0; x < in_.width_; x += pooling_size)
                    connect_kernel(pooling_size, x, y, c);


        for (layer_size_t c = 0; c < in_.depth_; ++c)
            for (layer_size_t y = 0; y < out_.height_; ++y)
                for (layer_size_t x = 0; x < out_.width_; ++x)
                    this->connect_bias(c, out_.get_index(x, y, c));
    }

    void connect_kernel(layer_size_t pooling_size, layer_size_t x, layer_size_t y, layer_size_t inc) {
        for (layer_size_t dy = 0; dy < pooling_size; ++dy)
            for (layer_size_t dx = 0; dx < pooling_size; ++dx)
                this->connect_weight(
                    in_.get_index(x + dx, y + dy, inc), 
                    out_.get_index(x / pooling_size, y / pooling_size, inc),
                    inc);
    }

    index3d<layer_size_t> in_;
    index3d<layer_size_t> out_;
};

} // namespace tiny_cnn
