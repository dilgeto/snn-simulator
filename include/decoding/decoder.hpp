#ifndef RL_DECODER_HPP
#define RL_DECODER_HPP

#include <vector>

// Decodificador para convertir spikes de salida en acciones
class RLDecoder {
public:
    enum class DecodingType {
        SPIKE_COUNT,      // Contar spikes — neurona con más spikes gana
        FIRST_SPIKE,      // Neurona que disparó primero gana / latencia → valor continuo
        RATE,             // Frecuencia promedio → valor continuo
        WINNER_TAKES_ALL, // Tasa normalizada por ventana (argmax)
        VOTING            // Ventana dividida en sub-ventanas; winner por sub-ventana acumula votos
    };

private:
    DecodingType type;
    double window_duration;
    int vote_windows;

public:
    RLDecoder(DecodingType type = DecodingType::WINNER_TAKES_ALL,
              double window_duration = 50.0,
              int vote_windows = 5);
    
    // Decodificar a acción discreta (e.g., arriba, abajo, izquierda, derecha)
    int decodeDiscreteAction(const std::vector<std::vector<double>>& output_spikes) const;
    
    // Decodificar a acción continua (e.g., torque, velocidad)
    double decodeContinuousAction(const std::vector<double>& spike_times) const;
    
    // Decodificar múltiples outputs continuos
    std::vector<double> decodeContinuousActions(
        const std::vector<std::vector<double>>& output_spikes) const;
    
    void setType(DecodingType t) { type = t; }
    void setWindowDuration(double duration) { window_duration = duration; }
    void setVoteWindows(int n) { vote_windows = n; }
};

#endif