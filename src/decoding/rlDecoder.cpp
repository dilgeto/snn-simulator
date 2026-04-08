// src/decoding/rl_decoder.cpp
#include "../../include/decoding/rlDecoder.hpp"
#include <algorithm>
#include <limits>

RLDecoder::RLDecoder(DecodingType type, double window_duration)
    : type(type), window_duration(window_duration) {
}

int RLDecoder::decodeDiscreteAction(
    const std::vector<std::vector<double>>& output_spikes) const {
    
    switch (type) {
        case DecodingType::SPIKE_COUNT: {
            // Acción = neurona con más spikes
            size_t max_spikes = 0;
            int best_action = 0;
            
            for (size_t i = 0; i < output_spikes.size(); i++) {
                if (output_spikes[i].size() > max_spikes) {
                    max_spikes = output_spikes[i].size();
                    best_action = i;
                }
            }
            return best_action;
        }
        
        case DecodingType::FIRST_SPIKE: {
            // Acción = neurona que disparó primero
            double earliest_spike = std::numeric_limits<double>::max();
            int best_action = 0;
            
            for (size_t i = 0; i < output_spikes.size(); i++) {
                if (!output_spikes[i].empty() && output_spikes[i][0] < earliest_spike) {
                    earliest_spike = output_spikes[i][0];
                    best_action = i;
                }
            }
            return best_action;
        }
        
        case DecodingType::WINNER_TAKES_ALL:
        default: {
            // Similar a spike count pero normalizado por tiempo
            double max_rate = 0.0;
            int best_action = 0;
            
            for (size_t i = 0; i < output_spikes.size(); i++) {
                double rate = output_spikes[i].size() / window_duration * 1000.0; // Hz
                if (rate > max_rate) {
                    max_rate = rate;
                    best_action = i;
                }
            }
            return best_action;
        }
    }
}

double RLDecoder::decodeContinuousAction(const std::vector<double>& spike_times) const {
    if (spike_times.empty()) {
        return 0.0;  // Sin spikes = acción neutral
    }
    
    switch (type) {
        case DecodingType::SPIKE_COUNT:
            // Normalizar conteo al rango [-1, 1]
            return std::min(1.0, spike_times.size() / 10.0) * 2.0 - 1.0;
        
        case DecodingType::FIRST_SPIKE: {
            // Latencia normalizada al rango [0, 1]
            double latency = spike_times[0] / window_duration;
            return 1.0 - latency;  // Spike temprano = valor alto
        }
        
        case DecodingType::RATE: {
            // Frecuencia de disparo normalizada
            double rate = spike_times.size() / window_duration * 1000.0; // Hz
            return std::min(1.0, rate / 100.0);  // Asumir max 100Hz
        }
        
        default:
            return 0.0;
    }
}

std::vector<double> RLDecoder::decodeContinuousActions(
    const std::vector<std::vector<double>>& output_spikes) const {
    
    std::vector<double> actions;
    actions.reserve(output_spikes.size());
    
    for (const auto& spikes : output_spikes) {
        actions.push_back(decodeContinuousAction(spikes));
    }
    
    return actions;
}