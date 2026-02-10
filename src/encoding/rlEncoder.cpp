#include "../../include/encoding/rlEncoder.hpp"
#include <cmath>

RLEncoder::RLEncoder(EncodingType type, double max_rate, size_t population_size)
    : type(type), max_rate(max_rate), population_size(population_size) {
}

std::vector<double> RLEncoder::encode(double value, double duration, double dt) const {
    if (!isValid(value)) {
        return {};
    }
    
    std::vector<double> spike_times;
    
    switch (type) {
        case EncodingType::RATE: {
            // Rate encoding DETERMINISTA (sin aleatoriedad)
            double firing_rate = value * max_rate;
            double inter_spike_interval = 1000.0 / firing_rate;  // ms
            
            for (double t = 0; t < duration; t += inter_spike_interval) {
                spike_times.push_back(t);
            }
            break;
        }
        
        case EncodingType::TEMPORAL: {
            // Temporal coding: un spike cuyo tiempo codifica el valor
            double spike_time = (1.0 - value) * duration * 0.8;  // Usar 80% de la ventana
            spike_times.push_back(spike_time);
            break;
        }
        
        case EncodingType::POPULATION:
            // Para population coding, usar encodePopulation
            break;
    }
    
    return spike_times;
}

// Population coding: múltiples neuronas con receptive fields solapados
std::vector<std::vector<double>> RLEncoder::encodePopulation(
    double value,
    double duration,
    double dt) const {
    
    if (!isValid(value)) {
        return std::vector<std::vector<double>>(population_size);
    }
    
    std::vector<std::vector<double>> population_spikes(population_size);
    
    // Cada neurona tiene un "receptive field" centrado en un valor diferente
    for (size_t i = 0; i < population_size; i++) {
        // Centro del receptive field de esta neurona
        double center = static_cast<double>(i) / (population_size - 1);
        
        // Ancho del receptive field (con solapamiento)
        double sigma = 0.3;  // Controla el solapamiento
        
        // Activación basada en gaussiana
        double distance = std::abs(value - center);
        double activation = std::exp(-(distance * distance) / (2 * sigma * sigma));
        
        // Convertir activación a rate
        double firing_rate = activation * max_rate;
        
        // Generar spikes deterministas
        if (firing_rate > 1.0) {  // Umbral mínimo
            double inter_spike_interval = 1000.0 / firing_rate;
            for (double t = 0; t < duration; t += inter_spike_interval) {
                population_spikes[i].push_back(t);
            }
        }
    }
    
    return population_spikes;
}

// Codificar una observación completa (vector de features)
std::vector<std::vector<double>> RLEncoder::encodeObservation(
    const std::vector<double>& observation,
    double duration,
    double dt) const {
    
    std::vector<std::vector<double>> all_spikes;
    
    if (type == EncodingType::POPULATION) {
        // Con population coding, cada feature genera múltiples spike trains
        for (double value : observation) {
            auto pop_spikes = encodePopulation(value, duration, dt);
            all_spikes.insert(all_spikes.end(), pop_spikes.begin(), pop_spikes.end());
        }
    } else {
        // Sin population coding, una neurona por feature
        for (double value : observation) {
            all_spikes.push_back(encode(value, duration, dt));
        }
    }
    
    return all_spikes;
}