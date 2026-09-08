#include "model.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace toric;

struct Options {
    std::string input, output, device = "cpu";
    int64_t epochs = 10, batch = 32, threads = 1, patience = 10, seed = 42, stage = 0;
    double learning_rate = 1e-4, weight_decay = 5e-5;
    bool check = false;
};

Options parse(int argc, char** argv) {
    Options options;
    for (int argument = 1; argument < argc; ++argument) {
        std::string key = argv[argument];
        if (key == "--check") { options.check = true; continue; }
        if (key == "--help") {
            std::cout << "toric_train --input bundle.pt --output-dir DIR [--epochs N --batch-size N "
                         "--threads N --device cpu|cuda --lr RATE --weight-decay RATE "
                         "--patience N --seed N --stage 0|1|2 --check]\n";
            std::exit(0);
        }
        if (argument + 1 == argc) throw std::invalid_argument("missing value for " + key);
        std::string value = argv[++argument];
        if (key == "--input") options.input = value;
        else if (key == "--output-dir") options.output = value;
        else if (key == "--device") options.device = value;
        else if (key == "--epochs") options.epochs = std::stoll(value);
        else if (key == "--batch-size") options.batch = std::stoll(value);
        else if (key == "--threads") options.threads = std::stoll(value);
        else if (key == "--patience") options.patience = std::stoll(value);
        else if (key == "--seed") options.seed = std::stoll(value);
        else if (key == "--stage") options.stage = std::stoll(value);
        else if (key == "--lr") options.learning_rate = std::stod(value);
        else if (key == "--weight-decay") options.weight_decay = std::stod(value);
        else throw std::invalid_argument("unknown argument " + key);
    }
    TORCH_CHECK(!options.input.empty() && !options.output.empty(), "input and output-dir are required");
    TORCH_CHECK(options.epochs > 0 && options.batch > 0 && options.threads > 0 && options.patience > 0,
                "epoch, batch, threads and patience must be positive");
    TORCH_CHECK(std::isfinite(options.learning_rate) && options.learning_rate > 0 &&
                std::isfinite(options.weight_decay) && options.weight_decay >= 0, "invalid optimizer settings");
    TORCH_CHECK(options.stage >= 0 && options.stage <= 2, "unknown training stage");
    TORCH_CHECK(options.device == "cpu" || options.device == "cuda", "device must be cpu or cuda");
    return options;
}

struct Dataset {
    Tensor windows, labels, auxiliary;
    Dataset(torch::jit::Module& archive, const std::string& prefix, const Model& model) {
        auto features = archive.attr(prefix + "_features").toTensor();
        labels = archive.attr(prefix + "_labels").toTensor();
        auxiliary = archive.attr(prefix + "_auxiliary").toTensor();
        TORCH_CHECK(features.dim() == 2 && features.size(1) == model.feature_count(), "invalid feature matrix");
        TORCH_CHECK(labels.dim() == 2 && labels.size(0) > 0 && labels.size(1) == model.pattern_count(), "invalid labels");
        TORCH_CHECK(auxiliary.dim() == 2 && auxiliary.size(0) == labels.size(0) && auxiliary.size(1) == 4, "invalid targets");
        TORCH_CHECK(features.scalar_type() == torch::kFloat && labels.scalar_type() == torch::kFloat &&
                    auxiliary.scalar_type() == torch::kFloat, "float32 data required");
        TORCH_CHECK(torch::isfinite(features).all().item<bool>() && torch::isfinite(labels).all().item<bool>() &&
                    torch::isfinite(auxiliary).all().item<bool>(), "nonfinite dataset");
        TORCH_CHECK(((labels == 0) | (labels == 1)).all().item<bool>(), "labels must be binary");
        TORCH_CHECK((labels.select(1, labels.size(1) - 1).to(torch::kBool) ==
                     ~labels.slice(1, 0, -1).any(1)).all().item<bool>(), "inconsistent hold labels");
        TORCH_CHECK(features.size(0) >= labels.size(0) + model.sequence_length() - 1, "incomplete windows");
        windows = features.unfold(0, model.sequence_length(), 1).permute({0, 2, 1}).narrow(0, 0, labels.size(0));
    }
    int64_t size() const { return labels.size(0); }
};

struct Loss {
    Tensor total, pattern, gate, auxiliary;
};

Loss compute_loss(const Output& output, const Tensor& labels, const Tensor& auxiliary,
                  const Tensor& positive_weight, const Tensor& gate_weight, int stage, double auxiliary_weight) {
    namespace functional = torch::nn::functional;
    auto targets = labels.slice(1, 0, -1);
    auto event = targets.any(1);
    auto pattern = output.pattern.slice(1, 0, -1).sum() * 0;
    if (event.any().item<bool>()) {
        pattern = functional::binary_cross_entropy_with_logits(
            output.pattern.index({event, Slice(0, -1)}), targets.index({event}),
            functional::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(positive_weight));
    }
    auto gate = functional::binary_cross_entropy_with_logits(output.gate, event.unsqueeze(1).to(torch::kFloat),
                  functional::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(gate_weight));
    auto regression = pattern * 0;
    if (stage == 0) regression = functional::smooth_l1_loss(output.auxiliary, auxiliary);
    auto total = stage == 1 ? gate : stage == 2 ? pattern : pattern + gate + auxiliary_weight * regression;
    return {total, pattern, gate, regression};
}

struct Metrics {
    int64_t samples = 0, events = 0, updates = 0;
    double pattern_loss = 0, gate_loss = 0, auxiliary_loss = 0, seconds = 0;
    Tensor counts = torch::zeros({3, 16}, torch::kDouble);
    Tensor gate_counts = torch::zeros({3}, torch::kDouble);

    static double f1(const Tensor& counts) {
        auto values = counts.to(torch::kCPU);
        double positive = values[0].item<double>();
        return 2 * positive / std::max(1.0, 2 * positive + values[1].item<double>() + values[2].item<double>());
    }
    double pattern_f1() const { return f1(counts.sum(1)); }
    double gate_f1() const { return f1(gate_counts); }
    double selection_loss(int stage, double auxiliary_weight) const {
        auto pattern = pattern_loss / std::max<int64_t>(events, 1);
        auto gate = gate_loss / samples;
        if (stage == 1) return gate;
        if (stage == 2) return pattern;
        return pattern + gate + auxiliary_weight * auxiliary_loss / samples;
    }
    void json(std::ostream& stream) const {
        stream << std::setprecision(12) << "{\"samples\":" << samples << ",\"optimizer_updates\":" << updates
               << ",\"pattern_loss\":" << pattern_loss / std::max<int64_t>(events, 1)
               << ",\"gate_loss\":" << gate_loss / samples << ",\"aux_loss\":" << auxiliary_loss / samples
               << ",\"pattern_f1\":" << pattern_f1() << ",\"gate_f1\":" << gate_f1()
               << ",\"seconds\":" << seconds << ",\"samples_per_second\":" << samples / seconds << "}";
    }
};

void atomic_save(torch::jit::Module& archive, const fs::path& path) {
    auto temporary = path.string() + ".tmp";
    try {
        archive.save(temporary);
        fs::rename(temporary, path);
    } catch (...) {
        fs::remove(temporary);
        throw;
    }
}

Metrics run_epoch(Model& model, Dataset& dataset, torch::optim::AdamW& optimizer, const Options& options,
                  const Tensor& positive_weight, const Tensor& gate_weight, const Tensor& settings,
                  bool training, torch::Device device) {
    auto start = std::chrono::steady_clock::now();
    Metrics metrics;
    metrics.counts = torch::zeros({3, model.pattern_count() - 1}, torch::kDouble);
    double auxiliary_weight = settings[0].item<double>();
    double conditional_threshold = settings[1].item<double>();
    double confidence_threshold = settings[2].item<double>();
    double gate_threshold = settings[3].item<double>();
    auto order = training && !options.check ? torch::randperm(dataset.size(), torch::kLong) :
                                             torch::arange(dataset.size(), torch::kLong);
    torch::AutoGradMode grad_mode(training);
    for (int64_t offset = 0; offset < dataset.size(); offset += options.batch) {
        auto indices = order.narrow(0, offset, std::min(options.batch, dataset.size() - offset));
        auto features = dataset.windows.index_select(0, indices).to(device);
        auto labels = dataset.labels.index_select(0, indices).to(device);
        auto auxiliary = dataset.auxiliary.index_select(0, indices).to(device);
        auto output = model.forward(features, training);
        auto loss = compute_loss(output, labels, auxiliary, positive_weight, gate_weight, options.stage, auxiliary_weight);
        TORCH_CHECK(torch::isfinite(loss.total).item<bool>(), "nonfinite loss; weights not saved");
        auto truth = labels.slice(1, 0, -1).to(torch::kBool);
        int64_t events = truth.any(1).sum().item<int64_t>();
        if (training) {
            optimizer.zero_grad();
            if (options.stage != 2 || events > 0) {
                loss.total.backward();
                if (options.check) {
                    torch::jit::Module diagnostic("Diagnostic");
                    model.append_weights(diagnostic, true);
                    diagnostic.register_buffer("pattern", output.pattern.detach().cpu());
                    diagnostic.register_buffer("gate", output.gate.detach().cpu());
                    diagnostic.register_buffer("auxiliary", output.auxiliary.detach().cpu());
                    diagnostic.register_buffer("loss", loss.total.detach().cpu());
                    atomic_save(diagnostic, fs::path(options.output) / "diagnostic.pt");
                }
                torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0, 2.0, true);
                optimizer.step();
                metrics.updates++;
            }
        }
        torch::NoGradGuard no_grad;
        metrics.samples += indices.size(0);
        metrics.events += events;
        metrics.pattern_loss += loss.pattern.item<double>() * events;
        metrics.gate_loss += loss.gate.item<double>() * indices.size(0);
        metrics.auxiliary_loss += loss.auxiliary.item<double>() * indices.size(0);
        auto gate = output.gate.sigmoid();
        auto conditional = output.pattern.slice(1, 0, -1).sigmoid();
        auto prediction = (conditional >= conditional_threshold) & (gate >= gate_threshold) &
                          (conditional * gate >= confidence_threshold);
        metrics.counts += torch::stack({(prediction & truth).sum(0), (prediction & ~truth).sum(0),
                                       (~prediction & truth).sum(0)}).cpu();
        auto gate_prediction = gate.squeeze(1) >= gate_threshold;
        auto event = truth.any(1);
        metrics.gate_counts += torch::stack({(gate_prediction & event).sum(), (gate_prediction & ~event).sum(),
                                            (~gate_prediction & event).sum()}).cpu();
        if (options.check) break;
    }
    metrics.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return metrics;
}

int main(int argc, char** argv) {
    try {
        auto options = parse(argc, argv);
        torch::set_num_threads(options.threads);
        torch::set_num_interop_threads(1);
        torch::manual_seed(options.seed);
        if (options.device == "cuda") TORCH_CHECK(torch::cuda::is_available(), "CUDA is unavailable");
        torch::Device device(options.device);
        fs::create_directories(options.output);
        auto archive = torch::jit::load(options.input, torch::kCPU);
        TORCH_CHECK(archive.attr("bundle_version").toTensor().item<int>() == 1, "unsupported bundle version");
        TORCH_CHECK(archive.attr("stage").toTensor().item<int>() == options.stage, "stage differs from prepared bundle");
        Model model(archive, device, options.stage);
        Dataset training(archive, "train", model), validation(archive, "validation", model);
        auto positive_weight = archive.attr("positive_weight").toTensor().to(device);
        auto gate_weight = archive.attr("gate_weight").toTensor().to(device);
        auto settings = archive.attr("loss_settings").toTensor();
        TORCH_CHECK(settings.numel() == 4 && torch::isfinite(settings).all().item<bool>(), "invalid loss settings");
        TORCH_CHECK(settings[0].item<double>() >= 0 && (settings.slice(0, 1) >= 0).all().item<bool>() &&
                    (settings.slice(0, 1) <= 1).all().item<bool>(), "invalid thresholds");
        torch::optim::AdamW optimizer(model.parameters(), torch::optim::AdamWOptions(options.learning_rate)
                                                        .weight_decay(options.weight_decay));
        double best = std::numeric_limits<double>::infinity(), scheduler_best = best;
        int64_t stale = 0, scheduler_stale = 0;
        std::ofstream log(fs::path(options.output) / "metrics.jsonl", std::ios::trunc);
        TORCH_CHECK(log.good(), "cannot open metrics log");
        for (int64_t epoch = 1; epoch <= options.epochs; ++epoch) {
            auto train = run_epoch(model, training, optimizer, options, positive_weight, gate_weight, settings, true, device);
            auto val = run_epoch(model, validation, optimizer, options, positive_weight, gate_weight, settings, false, device);
            auto write_metrics = [&](std::ostream& stream) {
                stream << "{\"epoch\":" << epoch << ",\"train\":";
                train.json(stream);
                stream << ",\"validation\":";
                val.json(stream);
                stream << "}\n";
                stream.flush();
            };
            write_metrics(log);
            write_metrics(std::cout);
            double metric = val.selection_loss(options.stage, settings[0].item<double>());
            if (metric < scheduler_best * (1 - 1e-4)) { scheduler_best = metric; scheduler_stale = 0; }
            else if (++scheduler_stale > 3) {
                for (auto& group : optimizer.param_groups()) {
                    auto& config = static_cast<torch::optim::AdamWOptions&>(group.options());
                    auto reduced = std::max(0.0, config.lr() * 0.1);
                    if (config.lr() - reduced > 1e-8) config.lr(reduced);
                }
                scheduler_stale = 0;
            }
            torch::jit::Module result("TrainedWeights");
            model.append_weights(result);
            result.register_buffer("epoch", torch::tensor(epoch, torch::kLong));
            result.register_buffer("validation_pattern_counts", val.counts);
            result.register_buffer("validation_gate_counts", val.gate_counts);
            atomic_save(result, fs::path(options.output) / "last_weights.pt");
            if (metric < best - 1e-4) {
                best = metric;
                stale = 0;
                atomic_save(result, fs::path(options.output) / "best_weights.pt");
            } else ++stale;
            if (options.check || stale >= options.patience) break;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "toric_train: " << error.what() << '\n';
        return 1;
    }
}
