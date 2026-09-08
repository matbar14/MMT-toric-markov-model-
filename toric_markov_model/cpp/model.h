#pragma once

#include <torch/torch.h>
#include <torch/script.h>

#include <map>
#include <string>
#include <vector>

namespace toric {

using torch::Tensor;
using torch::indexing::Slice;

inline std::string archive_name(std::string name) {
    std::string result = "weight__";
    for (char character : name) {
        result += character == '.' ? "__" : std::string(1, character);
    }
    return result;
}

struct Output {
    Tensor pattern;
    Tensor gate;
    Tensor auxiliary;
};

class Model {
public:
    Model(torch::jit::Module& archive, torch::Device device, int stage)
        : device_(device), stage_(stage) {
        auto config = archive.attr("model_dimensions").toTensor().to(torch::kCPU);
        TORCH_CHECK(config.numel() == 9, "invalid model dimensions");
        features_ = config[0].item<int64_t>();
        dimensions_ = config[1].item<int64_t>();
        length_ = config[2].item<int64_t>();
        states_ = config[3].item<int64_t>();
        levels_ = config[4].item<int64_t>();
        layers_ = config[5].item<int64_t>();
        bits_ = config[6].item<int64_t>();
        attention_ = config[7].item<int64_t>() != 0;
        patterns_ = config[8].item<int64_t>();
        dropout_ = archive.attr("dropout").toTensor().item<double>();
        TORCH_CHECK(features_ > 0 && dimensions_ > 0 && length_ > 0 && states_ > 0 &&
                    levels_ > 0 && layers_ > 0 && dimensions_ % levels_ == 0 &&
                    bits_ > 0 && bits_ <= 16 && patterns_ >= 2 && dropout_ >= 0 && dropout_ < 1,
                    "invalid model configuration");
        for (const auto& named : archive.named_buffers()) {
            if (named.name.rfind("weight__", 0) != 0) continue;
            auto value = named.value.to(device_).clone().detach();
            bool complex = value.is_complex();
            auto leaf = complex ? torch::view_as_real(value).detach() : value;
            bool pattern_head = named.name.rfind("weight__pattern_head__", 0) == 0;
            bool auxiliary_head = named.name.rfind("weight__aux_head__", 0) == 0;
            bool trainable = stage_ == 0 || (stage_ == 1 && !pattern_head && !auxiliary_head) ||
                             (stage_ == 2 && pattern_head);
            leaf.requires_grad_(trainable);
            leaves_[named.name] = leaf;
            complex_[named.name] = complex;
            if (trainable) parameters_.push_back(leaf);
        }
        TORCH_CHECK(!parameters_.empty(), "archive contains no trainable parameters");
    }

    int64_t sequence_length() const { return length_; }
    int64_t feature_count() const { return features_; }
    int64_t pattern_count() const { return patterns_; }
    std::vector<Tensor>& parameters() { return parameters_; }

    Output forward(const Tensor& input, bool training) {
        TORCH_CHECK(input.dim() == 3 && input.size(1) == length_ && input.size(2) == features_,
                    "incorrect input shape");
        const auto batch = input.size(0);
        auto token = phase(torch::tanh(linear(input, "feature_emb.feature_to_angles")) * pi_);
        auto position = phase(torch::tanh(weight("pos_emb.raw_shifts")) * pi_).unsqueeze(0);
        auto token_levels = token.chunk(levels_, -1);
        auto position_levels = position.chunk(levels_, -1);
        std::vector<Tensor> hidden(levels_);
        for (auto& level : hidden) level = torch::ones({batch, dimensions_ / levels_}, token.options());
        auto probabilities = torch::zeros({batch, states_}, input.options());
        probabilities.index_put_({Slice(), 0}, 1);
        for (int64_t timestep = 0; timestep < length_; ++timestep) {
            auto current = token.select(1, timestep);
            auto real_features = torch::cat({torch::real(current), torch::imag(current)}, -1);
            auto normalized = torch::layer_norm(real_features, {2 * dimensions_},
                                weight("markov_chain.norm.weight"), weight("markov_chain.norm.bias"));
            auto logits = torch::matmul(normalized, weight("markov_chain.token_proj.weight").t()) +
                          torch::matmul(probabilities, weight("markov_chain.state_proj")) +
                          weight("markov_chain.state_bias");
            probabilities = torch::softmax(logits, -1);
            auto complex_probs = probabilities.to(token.scalar_type());
            for (int64_t layer = 0; layer < layers_; ++layer) {
                const auto prefix = "toric_layers." + std::to_string(layer) + ".";
                auto mix = weight(prefix + "gate").sigmoid();
                for (int64_t level = 0; level < levels_; ++level) {
                    const auto suffix = std::to_string(level);
                    auto rotation_forward = torch::einsum("bs,sd->bd", {complex_probs, weight(prefix + "state_rotation_fwd." + suffix)});
                    auto rotation_backward = torch::einsum("bs,sd->bd", {complex_probs, weight(prefix + "state_rotation_bwd." + suffix)});
                    auto input_phase = token_levels[level].select(1, timestep) * position_levels[level].select(1, timestep);
                    auto forward = normalize(hidden[level] * (input_phase * rotation_forward));
                    auto backward = normalize(hidden[level] * (input_phase * rotation_backward.conj()));
                    auto candidate = normalize((forward + backward) / 2.0);
                    if (attention_) {
                        auto context = weight(prefix + "context." + suffix);
                        auto score = (candidate * context.conj()).sum(-1).abs();
                        auto attention = (score * weight(prefix + "attn_scale")).sigmoid().unsqueeze(-1);
                        candidate = normalize(candidate + attention * context);
                    }
                    hidden[level] = normalize(mix * candidate + (1 - mix) * hidden[level]);
                }
            }
        }
        auto combined = torch::cat(hidden, -1);
        auto representation = linear(torch::cat({torch::real(combined), torch::imag(combined)}, -1), "complex_feature_fusion.0");
        representation = torch::gelu(torch::layer_norm(representation, {2 * dimensions_},
                            weight("complex_feature_fusion.1.weight"), weight("complex_feature_fusion.1.bias")));
        representation = representation + 0.1 * linear(input.select(1, length_ - 1), "feature_anchor");
        auto gate = head(representation, "non_hold_gate_head", training && stage_ != 2);
        auto conditional = head(representation, "pattern_head", training);
        auto auxiliary = head(representation, "aux_head", training && stage_ != 2);
        return {torch::cat({conditional, -gate}, -1), gate, auxiliary};
    }

    void append_weights(torch::jit::Module& archive, bool gradients = false) const {
        for (const auto& named : leaves_) {
            auto value = named.second.detach();
            if (complex_.at(named.first)) value = torch::view_as_complex(value);
            archive.register_buffer(named.first, value.to(torch::kCPU));
            if (gradients && named.second.grad().defined()) {
                auto gradient = named.second.grad().detach();
                if (complex_.at(named.first)) gradient = torch::view_as_complex(gradient);
                archive.register_buffer("gradient__" + named.first, gradient.to(torch::kCPU));
            }
        }
    }

private:
    Tensor weight(const std::string& name) const {
        auto key = archive_name(name);
        auto value = leaves_.at(key);
        return complex_.at(key) ? torch::view_as_complex(value) : value;
    }

    Tensor linear(const Tensor& input, const std::string& prefix) const {
        return torch::linear(input, weight(prefix + ".weight"), weight(prefix + ".bias"));
    }

    Tensor head(const Tensor& input, const std::string& prefix, bool training) const {
        auto hidden = torch::gelu(linear(input, prefix + ".0"));
        return linear(torch::dropout(hidden, dropout_, training), prefix + ".3");
    }

    Tensor phase(const Tensor& angles) const {
        const int64_t count = int64_t{1} << bits_;
        auto indices = torch::floor(((angles + pi_) / (2 * pi_)) * count).to(torch::kLong).clamp(0, count - 1);
        auto quantized = indices.to(torch::kFloat) * ((2 * pi_) / count) - pi_;
        auto straight_through = quantized.detach() + (angles - angles.detach());
        return torch::complex(straight_through.cos(), straight_through.sin());
    }

    static Tensor normalize(const Tensor& value) { return value / (value.abs() + 1e-8); }

    torch::Device device_;
    int stage_;
    int64_t features_, dimensions_, length_, states_, levels_, layers_, bits_, patterns_;
    bool attention_;
    double dropout_;
    static constexpr double pi_ = 3.14159265358979323846;
    std::map<std::string, Tensor> leaves_;
    std::map<std::string, bool> complex_;
    std::vector<Tensor> parameters_;
};

}
