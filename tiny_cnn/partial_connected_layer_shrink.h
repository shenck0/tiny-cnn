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
#ifdef CNN_USE_SHRINK_LAYER
#include "util.h"
#include "layer.h"

namespace tiny_cnn {

	template<typename VectorType>
	void RELEASE_VECTOR(VectorType& v) {
		VectorType e;
		v.swap(e);
	}

	/*
		The triplet which represents a arithmetic progression.
	*/
	struct ap_triplet
	{
		// len(a0,a0 + d,......) = length
		int a0;//a0
		int d;//a1 - a0
		int length;//length of progression
		ap_triplet(int a, int d, int l) {
			this->a0 = a;
			this->d = d;
			this->length = l;
		}
	};

	/*
		Iterator of a vector<ap_triplet>
		  create a ap_triplet_iterator from a std::vector<ap_triplet>*
		  use iter_next() to get next value until end() return true
	*/
	struct ap_triplet_iterator
	{
		const std::vector<ap_triplet>* triplet_list;
		const ap_triplet* cur_triplet;
		int cur_value;
		int ap_index;
		int len_remain;
		ap_triplet_iterator(const std::vector<ap_triplet>* s) {
			cur_value = ap_index = len_remain = 0;
			triplet_list = s;
		}
		inline bool end() {
			return (len_remain == 0 && ap_index == triplet_list->size());
		}
		inline int iter_next() {
			if (len_remain > 0) {
				cur_value += cur_triplet->d;
			}
			else {
				cur_triplet = &triplet_list->at(ap_index++);
				cur_value = cur_triplet->a0;
				len_remain = cur_triplet->length;
			}
			len_remain -= 1;
			return cur_value;
		}
	};

	/*
		Use std::vector<ap_triplet> to replace all of the first/second element std::vector<io/wi/wo_connections>.
		*hack*
	*/
	template<typename ConnectionType, typename GetVal>
	void _shrink_connections_(std::vector<ConnectionType>& src, std::vector<ap_triplet>& dst, GetVal fval) {
		dst.clear();
		int state = 0;
		int base, diff = 0, len = 0, lastVal;
		for (int i = 0; i < src.size(); i++) {
			int curVal = fval(src[i]);
			if (state == 0) {//State0 : init state, haven't read any number before now.
				base = curVal;
				state = len = 1;
			}
			else if (state == 1) {//State1 : have read two numbers until now, and we can get a0,d
				diff = curVal - lastVal;
				state = 2;
				len++;
			}
			else if (state == 2) {//State2 : the third number has arrived
				if (curVal - lastVal == diff) {//same diff
					state = 2;
					len++;
				}
				else {//different diff, create a new ap_triplet and start counting a new AP
					dst.push_back(ap_triplet(base, diff, len));
					base = curVal;
					len = state = 1;
				}
			}
			lastVal = curVal;
		}
		if (len > 0) {
			dst.push_back(ap_triplet(base, diff, len));
		}
		return;
	}

	template<typename Activation>
	class partial_connected_layer : public layer<Activation> {
	public:
		CNN_USE_LAYER_MEMBERS;

		typedef std::vector<std::pair<layer_size_t, layer_size_t> > io_connections;
		typedef std::vector<std::pair<layer_size_t, layer_size_t> > wi_connections;
		typedef std::vector<std::pair<layer_size_t, layer_size_t> > wo_connections;

		typedef std::vector<ap_triplet> io_connections_shrink_;
		typedef std::vector<ap_triplet> wi_connections_shrink_;
		typedef std::vector<ap_triplet> wo_connections_shrink_;

		typedef layer<Activation> Base;

		partial_connected_layer(layer_size_t in_dim, layer_size_t out_dim, size_t weight_dim, size_t bias_dim, float_t scale_factor = 1.0)
			: Base(in_dim, out_dim, weight_dim, bias_dim),
			weight2io_(weight_dim), out2wi_(out_dim), in2wo_(in_dim), bias2out_(bias_dim), out2bias_(out_dim),
			scale_factor_(scale_factor) {}

		size_t param_size() const override {
			size_t total_param = 0;
			for (auto w : weight2io_size_)
			if (w > 0) total_param++;
			for (auto b : bias2out_)
			if (b.size() > 0) total_param++;
			return total_param;
		}

		size_t connection_size() const override {
			size_t total_size = 0;
			for (auto io : weight2io_size_)
				total_size += io;
			for (auto b : bias2out_)
				total_size += b.size();
			return total_size;
		}

		size_t fan_in_size() const override {
			//return max_size(out2wi_);
			return *std::max_element(out2wi_size_.begin(), out2wi_size_.end());
		}

		size_t fan_out_size() const override {
			//return max_size(in2wo_);
			return *std::max_element(in2wo_size_.begin(), in2wo_size_.end());
		}

		void connect_weight(layer_size_t input_index, layer_size_t output_index, layer_size_t weight_index) {
			weight2io_[weight_index].emplace_back(input_index, output_index);
			out2wi_[output_index].emplace_back(weight_index, input_index);
			in2wo_[input_index].emplace_back(weight_index, output_index);
		}

		void connect_bias(layer_size_t bias_index, layer_size_t output_index) {
			out2bias_[output_index] = bias_index;
			bias2out_[bias_index].push_back(output_index);
		}

		const vec_t& forward_propagation(const vec_t& in, size_t index) override {
			vec_t& a = a_[index];

			for_i(parallelize_, out_size_, [&](int i) {
				//const wi_connections& connections = out2wi_[i];
				ap_triplet_iterator fi = ap_triplet_iterator(&out2wi_shrink_first[i]);//first iterator
				ap_triplet_iterator si = ap_triplet_iterator(&out2wi_shrink_second[i]);//second iterator

				a[i] = 0.0;

				//for (auto connection : connections)// 13.1%
				while (!fi.end())
					a[i] += W_[fi.iter_next()] * in[si.iter_next()]; // 3.2%
				//a[i] += W_[connection.first] * in[connection.second]; // 3.2%

				a[i] *= scale_factor_;
				a[i] += b_[out2bias_[i]];
			});

			for_i(parallelize_, out_size_, [&](int i) {
				output_[index][i] = h_.f(a, i);
			});

			return next_ ? next_->forward_propagation(output_[index], index) : output_[index]; // 15.6%
		}

		virtual const vec_t& back_propagation(const vec_t& current_delta, size_t index) {
			const vec_t& prev_out = prev_->output(index);
			const activation::function& prev_h = prev_->activation_function();
			vec_t& prev_delta = prev_delta_[index];

			for_(parallelize_, 0, in_size_, [&](const blocked_range& r) {
				for (int i = r.begin(); i != r.end(); i++) {
					//const wo_connections& connections = in2wo_[i];
					ap_triplet_iterator fi = ap_triplet_iterator(&in2wo_shrink_first[i]);
					ap_triplet_iterator si = ap_triplet_iterator(&in2wo_shrink_second[i]);
					float_t delta = 0.0;

					//for (auto connection : connections) 
					while (!fi.end())
						delta += W_[fi.iter_next()] * current_delta[si.iter_next()]; // 40.6%
					//delta += W_[connection.first] * current_delta[connection.second]; // 40.6%

					prev_delta[i] = delta * scale_factor_ * prev_h.df(prev_out[i]); // 2.1%
				}
			});

			for_(parallelize_, 0, weight2io_shrink_first.size(), [&](const blocked_range& r) {
				for (int i = r.begin(); i < r.end(); i++) {
					//const io_connections& connections = weight2io_[i];
					ap_triplet_iterator fi = ap_triplet_iterator(&weight2io_shrink_first[i]);
					ap_triplet_iterator si = ap_triplet_iterator(&weight2io_shrink_second[i]);
					float_t diff = 0.0;

					//for (auto connection : connections) // 11.9%
					while (!fi.end())
						diff += prev_out[fi.iter_next()] * current_delta[si.iter_next()];
					//diff += prev_out[connection.first] * current_delta[connection.second];

					dW_[index][i] += diff * scale_factor_;
				}
			});

			for (size_t i = 0; i < bias2out_.size(); i++) {
				const std::vector<layer_size_t>& outs = bias2out_[i];
				float_t diff = 0.0;

				for (auto o : outs)
					diff += current_delta[o];

				db_[index][i] += diff;
			}

			return prev_->back_propagation(prev_delta_[index], index);
		}

		const vec_t& back_propagation_2nd(const vec_t& current_delta2) {
			const vec_t& prev_out = prev_->output(0);
			const activation::function& prev_h = prev_->activation_function();

			for (size_t i = 0; i < weight2io_shrink_first.size(); i++) {
				//const io_connections& connections = weight2io_[i];
				ap_triplet_iterator fi = ap_triplet_iterator(&weight2io_shrink_first[i]);
				ap_triplet_iterator si = ap_triplet_iterator(&weight2io_shrink_second[i]);
				float_t diff = 0.0;

				//for (auto connection : connections)
				while (!fi.end())
					diff += sqr(prev_out[fi.iter_next()]) * current_delta2[si.iter_next()];
				//diff += sqr(prev_out[connection.first]) * current_delta2[connection.second];

				diff *= sqr(scale_factor_);
				Whessian_[i] += diff;
			}

			for (size_t i = 0; i < bias2out_.size(); i++) {
				const std::vector<layer_size_t>& outs = bias2out_[i];
				float_t diff = 0.0;

				for (auto o : outs)
					diff += current_delta2[o];

				bhessian_[i] += diff;
			}

			for (int i = 0; i < in_size_; i++) {
				//const wo_connections& connections = in2wo_[i];
				ap_triplet_iterator fi = ap_triplet_iterator(&in2wo_shrink_first[i]);
				ap_triplet_iterator si = ap_triplet_iterator(&in2wo_shrink_second[i]);
				prev_delta2_[i] = 0.0;

				//for (auto connection : connections)
				while (!fi.end())
					prev_delta2_[i] += sqr(W_[fi.iter_next()]) * current_delta2[si.iter_next()];
				//prev_delta2_[i] += sqr(W_[connection.first]) * current_delta2[connection.second];

				prev_delta2_[i] *= sqr(scale_factor_ * prev_h.df(prev_out[i]));
			}
			return prev_->back_propagation_2nd(prev_delta2_);
		}

		// remove unused weight to improve cache hits
		void remap() {
			std::map<int, int> swaps;
			size_t n = 0;

			for (size_t i = 0; i < weight2io_.size(); i++)
				swaps[i] = weight2io_[i].empty() ? -1 : n++;

			for (size_t i = 0; i < out_size_; i++) {
				wi_connections& wi = out2wi_[i];
				for (size_t j = 0; j < wi.size(); j++)
					wi[j].first = static_cast<layer_size_t>(swaps[wi[j].first]);
			}

			for (size_t i = 0; i < in_size_; i++) {
				wo_connections& wo = in2wo_[i];
				for (size_t j = 0; j < wo.size(); j++)
					wo[j].first = static_cast<layer_size_t>(swaps[wo[j].first]);
			}

			std::vector<io_connections> weight2io_new(n);
			for (size_t i = 0; i < weight2io_.size(); i++)
			if (swaps[i] >= 0) weight2io_new[swaps[i]] = weight2io_[i];

			weight2io_.swap(weight2io_new);
		}

		void shrink_during_init() {
			//weight2io_
			weight2io_shrink_first.resize(weight2io_.size());
			weight2io_shrink_second.resize(weight2io_.size());
			for (int i = 0; i < weight2io_.size(); i++) {
				_shrink_connections_(weight2io_[i], weight2io_shrink_first[i], [](auto& v) {  return v.first; });
				_shrink_connections_(weight2io_[i], weight2io_shrink_second[i], [](auto& v) {  return v.second; });
			}
			for (auto& v : weight2io_)
				RELEASE_VECTOR(v);

			//out2wi_
			out2wi_shrink_first.resize(out2wi_.size());
			out2wi_shrink_second.resize(out2wi_.size());
			for (int i = 0; i < out2wi_.size(); i++) {
				_shrink_connections_(out2wi_[i], out2wi_shrink_first[i], [](auto& v) {  return v.first; });
				_shrink_connections_(out2wi_[i], out2wi_shrink_second[i], [](auto& v) {  return v.second; });
			}
			for (auto& v : out2wi_)
				RELEASE_VECTOR(v);

			//in2wo_
			in2wo_shrink_first.resize(in2wo_.size());
			in2wo_shrink_second.resize(in2wo_.size());
			for (int i = 0; i < in2wo_.size(); i++) {
				_shrink_connections_(in2wo_[i], in2wo_shrink_first[i], [](auto& v) {  return v.first; });
				_shrink_connections_(in2wo_[i], in2wo_shrink_second[i], [](auto& v) {  return v.second; });
			}
			for (auto& v : in2wo_)
				RELEASE_VECTOR(v);
		}

		void shrink_after_init() {
			//weight2io_
			for (int i = 0; i < weight2io_.size(); i++) {
				weight2io_size_.push_back(weight2io_[i].size());
			}
			weight2io_shrink_first.resize(weight2io_.size());
			weight2io_shrink_second.resize(weight2io_.size());
			for (int i = 0; i < weight2io_.size(); i++) {
				_shrink_connections_(weight2io_[i], weight2io_shrink_first[i], [](auto& v) {  return v.first; });
				_shrink_connections_(weight2io_[i], weight2io_shrink_second[i], [](auto& v) {  return v.second; });
			}
			RELEASE_VECTOR(weight2io_);

			//out2wi_
			for (int i = 0; i < out2wi_.size(); i++) {
				out2wi_size_.push_back(out2wi_[i].size());
			}
			out2wi_shrink_first.resize(out2wi_.size());
			out2wi_shrink_second.resize(out2wi_.size());
			for (int i = 0; i < out2wi_.size(); i++) {
				_shrink_connections_(out2wi_[i], out2wi_shrink_first[i], [](auto& v) {  return v.first; });
				_shrink_connections_(out2wi_[i], out2wi_shrink_second[i], [](auto& v) {  return v.second; });
			}
			RELEASE_VECTOR(out2wi_);

			//in2wo_
			for (int i = 0; i < in2wo_.size(); i++) {
				in2wo_size_.push_back(in2wo_[i].size());
			}
			in2wo_shrink_first.resize(in2wo_.size());
			in2wo_shrink_second.resize(in2wo_.size());
			for (int i = 0; i < in2wo_.size(); i++) {
				_shrink_connections_(in2wo_[i], in2wo_shrink_first[i], [](auto& v) {  return v.first; });
				_shrink_connections_(in2wo_[i], in2wo_shrink_second[i], [](auto& v) {  return v.second; });
			}
			RELEASE_VECTOR(in2wo_);
		}

	protected:
		std::vector<io_connections> weight2io_; // weight_id -> [(in_id, out_id)]
		std::vector<wi_connections> out2wi_; // out_id -> [(weight_id, in_id)]
		std::vector<wo_connections> in2wo_; // in_id -> [(weight_id, out_id)]

		std::vector<size_t> weight2io_size_;
		std::vector<size_t> out2wi_size_;
		std::vector<size_t> in2wo_size_;

		std::vector<io_connections_shrink_> weight2io_shrink_first;
		std::vector<io_connections_shrink_> weight2io_shrink_second;
		std::vector<wi_connections_shrink_> out2wi_shrink_first;
		std::vector<wi_connections_shrink_> out2wi_shrink_second;
		std::vector<wo_connections_shrink_> in2wo_shrink_first;
		std::vector<wo_connections_shrink_> in2wo_shrink_second;

		std::vector<std::vector<layer_size_t> > bias2out_;
		std::vector<size_t> out2bias_;
		float_t scale_factor_;
	};

} // namespace tiny_cnn

#endif