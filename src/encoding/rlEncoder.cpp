#include "../../include/encoding/rlEncoder.hpp"
#include <cmath>
#include <algorithm>

RLEncoder::RLEncoder(EncodingType type, double max_rate, size_t population_size, size_t neurons_per_var)
    : type(type), max_rate(max_rate), population_size(population_size), neurons_per_var(neurons_per_var) {
}

std::vector<double> RLEncoder::encode(double value, double duration, double dt) const {
    (void)dt;
    if (!isValid(value)) {
        return {};
    }

    std::vector<double> spike_times;

    switch (type) {
        case EncodingType::RATE: {
            double firing_rate = value * max_rate;
            double inter_spike_interval = 1000.0 / firing_rate;
            for (double t = 0; t < duration; t += inter_spike_interval) {
                spike_times.push_back(t);
            }
            break;
        }
        case EncodingType::TEMPORAL: {
            double spike_time = (1.0 - value) * duration * 0.8;
            spike_times.push_back(spike_time);
            break;
        }
        case EncodingType::POPULATION:
        case EncodingType::SIGN_MAGNITUDE:
        case EncodingType::INTERVAL:
            // Estos tipos usan métodos propios (encodePopulation, encodeSignMagnitude, encodeInterval)
            break;
    }

    return spike_times;
}

// Population coding: múltiples neuronas con receptive fields gaussianos solapados
std::vector<std::vector<double>> RLEncoder::encodePopulation(
    double value,
    double duration,
    double dt) const {

    (void)dt;
    if (!isValid(value)) {
        return std::vector<std::vector<double>>(population_size);
    }

    std::vector<std::vector<double>> population_spikes(population_size);

    for (size_t i = 0; i < population_size; i++) {
        double center = static_cast<double>(i) / (population_size - 1);
        double sigma = 0.3;
        double distance = std::abs(value - center);
        double activation = std::exp(-(distance * distance) / (2 * sigma * sigma));
        double firing_rate = activation * max_rate;

        if (firing_rate > 1.0) {
            double inter_spike_interval = 1000.0 / firing_rate;
            for (double t = 0; t < duration; t += inter_spike_interval) {
                population_spikes[i].push_back(t);
            }
        }
    }

    return population_spikes;
}

// Sign-magnitude: 2 spike trains por feature
// [0] = neurona negativa, [1] = neurona positiva
std::vector<std::vector<double>> RLEncoder::encodeSignMagnitude(
    double value,
    double duration,
    double dt,
    double weight) const {

    (void)dt;
    std::vector<std::vector<double>> spikes(2);

    double magnitude = std::abs(value) * weight;
    double firing_rate = std::min(magnitude * max_rate, max_rate);

    if (firing_rate < 1.0) {
        return spikes;  // Ambas neuronas silenciosas
    }

    double inter_spike_interval = 1000.0 / firing_rate;
    int active = (value < 0) ? 0 : 1;  // 0 = negativo, 1 = positivo
    for (double t = 0; t < duration; t += inter_spike_interval) {
        spikes[active].push_back(t);
    }

    return spikes;
}

// Encodes una observación completa con SIGN_MAGNITUDE (2*N spike trains)
std::vector<std::vector<double>> RLEncoder::encodeObservationSignMagnitude(
    const std::vector<double>& observation,
    const std::vector<double>& weights,
    double duration,
    double dt) const {

    std::vector<std::vector<double>> all_spikes;
    all_spikes.reserve(observation.size() * 2);

    for (size_t i = 0; i < observation.size(); i++) {
        double w = (i < weights.size()) ? weights[i] : 1.0;
        auto pair_spikes = encodeSignMagnitude(observation[i], duration, dt, w);
        all_spikes.push_back(std::move(pair_spikes[0]));
        all_spikes.push_back(std::move(pair_spikes[1]));
    }

    return all_spikes;
}

// Interval coding: neurons_per_var spike trains por feature
// Solo dispara la neurona cuyo intervalo uniforme contiene el valor
std::vector<std::vector<double>> RLEncoder::encodeInterval(
    double value,
    double duration,
    double dt,
    double low,
    double high) const {

    (void)dt;
    std::vector<std::vector<double>> spikes(neurons_per_var);

    if (neurons_per_var == 0 || high <= low) {
        return spikes;
    }

    // Límites de intervalo uniformemente distribuidos en [low, high]
    double step = (high - low) / static_cast<double>(neurons_per_var);

    // Clamp value al rango para asegurar que siempre cae en algún intervalo
    double clamped = std::max(low, std::min(value, high));

    int idx = static_cast<int>((clamped - low) / step);
    // La neurona del último intervalo incluye el valor máximo
    if (idx >= static_cast<int>(neurons_per_var)) {
        idx = static_cast<int>(neurons_per_var) - 1;
    }

    // La neurona activa dispara a max_rate de forma determinista
    double inter_spike_interval = 1000.0 / max_rate;
    for (double t = 0; t < duration; t += inter_spike_interval) {
        spikes[idx].push_back(t);
    }

    return spikes;
}

// Codifica observación completa con INTERVAL (neurons_per_var * N spike trains)
std::vector<std::vector<double>> RLEncoder::encodeObservationInterval(
    const std::vector<double>& observation,
    const std::vector<std::pair<double,double>>& limits,
    double duration,
    double dt) const {

    std::vector<std::vector<double>> all_spikes;
    all_spikes.reserve(observation.size() * neurons_per_var);

    for (size_t i = 0; i < observation.size(); i++) {
        double low  = (i < limits.size()) ? limits[i].first  : 0.0;
        double high = (i < limits.size()) ? limits[i].second : 1.0;
        auto var_spikes = encodeInterval(observation[i], duration, dt, low, high);
        for (auto& s : var_spikes) {
            all_spikes.push_back(std::move(s));
        }
    }

    return all_spikes;
}

// Codificar una observación completa (tipos RATE/TEMPORAL/POPULATION)
std::vector<std::vector<double>> RLEncoder::encodeObservation(
    const std::vector<double>& observation,
    double duration,
    double dt) const {

    std::vector<std::vector<double>> all_spikes;

    if (type == EncodingType::POPULATION) {
        for (double value : observation) {
            auto pop_spikes = encodePopulation(value, duration, dt);
            all_spikes.insert(all_spikes.end(), pop_spikes.begin(), pop_spikes.end());
        }
    } else {
        for (double value : observation) {
            all_spikes.push_back(encode(value, duration, dt));
        }
    }

    return all_spikes;
}
