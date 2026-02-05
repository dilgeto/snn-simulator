// src/encoding/rate_encoder.cpp
#include "../../include/encoding/rateEncoder.hpp"
#include <random>

RateEncoder::RateEncoder(double max_rate) 
    : max_rate(max_rate) {
}

std::vector<double> RateEncoder::encode(double value, double duration, double dt) const {
    if (!isValid(value)) {
        return {};
    }
    
    std::vector<double> spike_times;
    
    // Frecuencia de disparo proporcional al valor
    double firing_rate = value * max_rate;  // Hz
    
    // Probabilidad de spike en cada paso de tiempo
    double spike_prob = (firing_rate / 1000.0) * dt;  // convertir Hz a ms
    
    // Generador aleatorio
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    // Generar spikes estocásticamente (proceso de Poisson)
    for (double t = 0; t < duration; t += dt) {
        if (dis(gen) < spike_prob) {
            spike_times.push_back(t);
        }
    }
    
    return spike_times;
}