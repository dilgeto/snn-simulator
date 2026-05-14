// include/encoding/rl_encoder.hpp
#ifndef RL_ENCODER_HPP
#define RL_ENCODER_HPP

#include "encoder.hpp"
#include <utility>

// Encoder optimizado para Reinforcement Learning
// Determinista y rápido, diseñado para observaciones continuas
class RLEncoder : public Encoder {
public:
    enum class EncodingType {
        POPULATION,     // Multiple neuronas por feature con Gaussian RFs
        RATE,           // Frecuencia constante (determinista)
        TEMPORAL,       // Timing-based (más eficiente)
        SIGN_MAGNITUDE, // 2 neuronas por feature: una para valores negativos, otra para positivos
        INTERVAL        // N neuronas por feature: solo dispara la del intervalo correspondiente
    };

private:
    EncodingType type;
    double max_rate;
    size_t population_size;   // Para POPULATION coding
    size_t neurons_per_var;   // Para INTERVAL coding (neuronas por variable)

public:
    RLEncoder(EncodingType type = EncodingType::POPULATION,
              double max_rate = 100.0,
              size_t population_size = 5,
              size_t neurons_per_var = 20);

    // Encoding básico (para una sola feature, tipos RATE/TEMPORAL)
    std::vector<double> encode(double value, double duration, double dt) const override;

    // Codificar observación completa del ambiente (RATE/TEMPORAL/POPULATION)
    std::vector<std::vector<double>> encodeObservation(
        const std::vector<double>& observation,
        double duration,
        double dt) const;

    // Codificar con population coding (múltiples neuronas por feature con Gaussian RFs)
    std::vector<std::vector<double>> encodePopulation(
        double value,
        double duration,
        double dt) const;

    // --- SIGN_MAGNITUDE ---
    // Codifica un valor escalar en 2 spike trains: [negativo, positivo]
    // Si value < 0: neurona 0 dispara a tasa ∝ |value|*weight; neurona 1 silenciosa
    // Si value >= 0: neurona 0 silenciosa; neurona 1 dispara a tasa ∝ value*weight
    std::vector<std::vector<double>> encodeSignMagnitude(
        double value,
        double duration,
        double dt,
        double weight = 1.0) const;

    // Codifica observación completa con SIGN_MAGNITUDE (2*N spike trains)
    std::vector<std::vector<double>> encodeObservationSignMagnitude(
        const std::vector<double>& observation,
        const std::vector<double>& weights,
        double duration,
        double dt) const;

    // --- INTERVAL ---
    // Codifica un valor escalar en neurons_per_var spike trains
    // Solo la neurona cuyo intervalo contiene value dispara (a max_rate)
    // Los intervalos se distribuyen uniformemente en [low, high]
    std::vector<std::vector<double>> encodeInterval(
        double value,
        double duration,
        double dt,
        double low,
        double high) const;

    // Codifica observación completa con INTERVAL (neurons_per_var * N spike trains)
    std::vector<std::vector<double>> encodeObservationInterval(
        const std::vector<double>& observation,
        const std::vector<std::pair<double,double>>& limits,
        double duration,
        double dt) const;

    void setType(EncodingType t) { type = t; }
    void setPopulationSize(size_t size) { population_size = size; }
    void setNeuronsPerVar(size_t n) { neurons_per_var = n; }
    size_t getNeuronsPerVar() const { return neurons_per_var; }
};

#endif