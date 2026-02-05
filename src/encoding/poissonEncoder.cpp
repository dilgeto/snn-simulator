// src/encoding/poisson_encoder.cpp
#include "../../include/encoding/poissonEncoder.hpp"
#include <random>

PoissonEncoder::PoissonEncoder(double max_rate, bool use_refractory, double refractory_period)
    : max_rate(max_rate), 
      use_refractory(use_refractory),
      refractory_period(refractory_period) {
}

std::vector<double> PoissonEncoder::encode(double value, double duration, double dt) const {
    if (!isValid(value)) {
        return {};
    }
    
    std::vector<double> spike_times;
    
    double firing_rate = value * max_rate;
    double spike_prob = (firing_rate / 1000.0) * dt;
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    double last_spike_time = -refractory_period - 1.0;
    
    for (double t = 0; t < duration; t += dt) {
        // Verificar período refractario
        if (use_refractory && (t - last_spike_time) < refractory_period) {
            continue;
        }
        
        if (dis(gen) < spike_prob) {
            spike_times.push_back(t);
            last_spike_time = t;
        }
    }
    
    return spike_times;
}

std::vector<double> PoissonEncoder::encodeTemporalPattern(
    const std::vector<double>& rate_pattern,
    double dt) const {
    
    std::vector<double> spike_times;
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    double last_spike_time = -refractory_period - 1.0;
    
    for (size_t i = 0; i < rate_pattern.size(); i++) {
        double t = i * dt;
        double firing_rate = rate_pattern[i] * max_rate;
        double spike_prob = (firing_rate / 1000.0) * dt;
        
        if (use_refractory && (t - last_spike_time) < refractory_period) {
            continue;
        }
        
        if (dis(gen) < spike_prob) {
            spike_times.push_back(t);
            last_spike_time = t;
        }
    }
    
    return spike_times;
}