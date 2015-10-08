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
#include "layer.h"
#include "input_layer.h"

namespace tiny_cnn {

class layers {
public:
    layers() { add(new input_layer()); }

	~layers() {
		//imlpement destructor because we remove all "shared_ptr" (C++ 11 feature)
		// warning: layers in a network will be released when network was destroyed
		layer_base* l = this->head();
		while (l) { 
			layer_base* todel = l;
			l = l->next();
			delete todel;
		}
	}

    layers(const layers& rhs) { construct(rhs); }

    layers& operator = (const layers& rhs) {
        layers_.clear();
        construct(rhs);
        return *this;
    }

    void add(layer_base* new_tail) {
        if (tail())  tail()->connect(new_tail);
        layers_.push_back(new_tail);
    }

    bool empty() const { return layers_.size() == 0; }

    layer_base* head() const { return empty() ? 0 : layers_[0]; }

    layer_base* tail() const { return empty() ? 0 : layers_[layers_.size() - 1]; }

    template <typename T>
    const T& at(size_t index) const {
        const T* v = dynamic_cast<const T*>(layers_[index + 1]);
        if (v) return *v;
        throw nn_error("failed to cast");
    }

    const layer_base* operator [] (size_t index) const {
        return layers_[index + 1];
    }

    layer_base* operator [] (size_t index) {
        return layers_[index + 1];
    }

    void init_weight() {
		for (std::vector<layer_base*>::iterator pl = layers_.begin(); pl != layers_.end();pl++ )
            (*pl)->init_weight();
    }

    bool is_exploded() const {
		for (std::vector<layer_base*>::const_iterator pl = layers_.begin(); pl != layers_.end(); pl++)
            if ((*pl)->is_exploded()) return true;
        return false;
    }

    void divide_hessian(int denominator) {
		for (std::vector<layer_base*>::iterator pl = layers_.begin(); pl != layers_.end(); pl++)
            (*pl)->divide_hessian(denominator);
    }

    template <typename Optimizer>
    void update_weights(Optimizer *o, size_t worker_size, size_t batch_size) {
		for (std::vector<layer_base*>::iterator pl = layers_.begin(); pl != layers_.end(); pl++)
            (*pl)->update_weight(o, worker_size, batch_size);
    }
    
    void set_parallelize(bool parallelize) {
		for (std::vector<layer_base*>::iterator pl = layers_.begin(); pl != layers_.end(); pl++)
            (*pl)->set_parallelize(parallelize);
    }

    // get depth(number of layers) of networks
    size_t depth() const {
        return layers_.size() - 1; // except input-layer
    }

private:
    void construct(const layers& rhs) {
        add(new input_layer());
        for (size_t i = 1; i < rhs.layers_.size(); i++)
            add(rhs.layers_[i]);
    }

    std::vector<layer_base*> layers_;
};

} // namespace tiny_cnn
