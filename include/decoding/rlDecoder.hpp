// include/decoding/rl_decoder.hpp
#ifndef RL_DECODER_HPP
#define RL_DECODER_HPP

#include <vector>

// Decodificador para convertir spikes de salida en acciones
class RLDecoder {
public:
    enum class DecodingType {
        SPIKE_COUNT,      // Contar spikes (discreto)
        FIRST_SPIKE,      // Tiempo del primer spike (continuo)
        RATE,             // Frecuencia promedio (continuo)
        WINNER_TAKES_ALL  // Argmax para acciones discretas
    };
    
private:
    DecodingType type;
    double window_duration;  // Ventana de observación en ms
    
public:
    RLDecoder(DecodingType type = DecodingType::WINNER_TAKES_ALL,
              double window_duration = 50.0);
    
    // Decodificar a acción discreta (e.g., arriba, abajo, izquierda, derecha)
    int decodeDiscreteAction(const std::vector<std::vector<double>>& output_spikes) const;
    
    // Decodificar a acción continua (e.g., torque, velocidad)
    double decodeContinuousAction(const std::vector<double>& spike_times) const;
    
    // Decodificar múltiples outputs continuos
    std::vector<double> decodeContinuousActions(
        const std::vector<std::vector<double>>& output_spikes) const;
    
    void setType(DecodingType t) { type = t; }
    void setWindowDuration(double duration) { window_duration = duration; }
};

#endif