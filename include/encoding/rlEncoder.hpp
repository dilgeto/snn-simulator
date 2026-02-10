// include/encoding/rl_encoder.hpp
#ifndef RL_ENCODER_HPP
#define RL_ENCODER_HPP

#include "encoder.hpp"

// Encoder optimizado para Reinforcement Learning
// Determinista y rápido, diseñado para observaciones continuas
class RLEncoder : public Encoder {
public:
    enum class EncodingType {
        POPULATION,    // Multiple neuronas por feature (recomendado)
        RATE,          // Frecuencia constante (determinista)
        TEMPORAL       // Timing-based (más eficiente)
    };
    
private:
    EncodingType type;
    double max_rate;
    size_t population_size;  // Para population coding
    
public:
    RLEncoder(EncodingType type = EncodingType::POPULATION, 
              double max_rate = 100.0,
              size_t population_size = 5);
    
    // Encoding básico (para una sola feature)
    std::vector<double> encode(double value, double duration, double dt) const override;
    
    // NUEVO: Codificar observación completa del ambiente
    // Retorna spikes para cada neurona de entrada
    std::vector<std::vector<double>> encodeObservation(
        const std::vector<double>& observation,
        double duration,
        double dt) const;
    
    // NUEVO: Codificar con population coding (múltiples neuronas por feature)
    // Retorna un vector de spike trains (uno por neurona en la población)
    std::vector<std::vector<double>> encodePopulation(
        double value,
        double duration,
        double dt) const;
    
    void setType(EncodingType t) { type = t; }
    void setPopulationSize(size_t size) { population_size = size; }
};

#endif