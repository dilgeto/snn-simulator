#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include "network.hpp"
#include "../encoding/encoder.hpp"
#include "../decoding/decoder.hpp"
#include <vector>
#include <memory>
#include <functional>

// Configuración de simulación
struct SimulationConfig {
    double dt;                    // Paso de tiempo (ms)
    double encoding_duration;     // Duración de codificación (ms)
    double evaluation_duration;   // Duración total de evaluación (ms)
    bool verbose;                 // Mostrar información detallada
    
    SimulationConfig() 
        : dt(1.0), 
          encoding_duration(50.0),
          evaluation_duration(100.0),
          verbose(false) {}
};

// Resultado de una simulación
struct SimulationResult {
    std::vector<std::vector<double>> output_spike_times;  // Spikes de cada neurona de salida
    std::vector<std::vector<double>> all_spike_times;     // Todos los spikes (opcional)
    std::vector<double> output_voltages;                  // Voltajes finales de salida
    double total_spikes;                                  // Total de spikes generados
    double simulation_time_ms;                            // Tiempo real de simulación
    
    SimulationResult() : total_spikes(0), simulation_time_ms(0) {}
};

// Clase principal del simulador
class Simulator {
private:
    Network* network;
    std::unique_ptr<Encoder> encoder;
    std::unique_ptr<RLDecoder> decoder;
    SimulationConfig config;
    
    bool owns_network;  // Si el simulador debe destruir la red
    
public:
    // Constructores
    explicit Simulator(Network* net, const SimulationConfig& cfg = SimulationConfig());
    Simulator(double dt = 1.0);
    ~Simulator();
    
    // Configuración
    void setNetwork(Network* net, bool take_ownership = false);
    void setEncoder(std::unique_ptr<Encoder> enc);
    void setDecoder(std::unique_ptr<RLDecoder> dec);
    void setConfig(const SimulationConfig& cfg);
    
    // ===== SIMULACIÓN BÁSICA =====
    
    // Simular con entradas directas (corrientes)
    SimulationResult simulate(
        const std::vector<double>& input_currents,
        double shared_weight,
        double duration = 0.0);  // 0 = usar config.evaluation_duration
    
    // Simular con entradas codificadas (valores -> spikes)
    SimulationResult simulateEncoded(
        const std::vector<double>& input_values,
        double shared_weight,
        double duration = 0.0);
    
    // ===== PARA REINFORCEMENT LEARNING =====
    
    // Ejecutar un episodio completo de RL
    // step_callback: función llamada en cada paso con (state, action, reward)
    double runEpisode(
        const std::vector<double>& initial_state,
        double shared_weight,
        int max_steps,
        std::function<bool(const std::vector<double>&, int&, double&)> step_callback);
    
    // Evaluar fitness en múltiples episodios
    double evaluateFitness(
        const std::vector<std::vector<double>>& test_cases,
        double shared_weight,
        std::function<double(const SimulationResult&)> fitness_func);
    
    // ===== PARA WANN =====
    
    // Evaluar red con múltiples pesos (para búsqueda de peso óptimo)
    std::vector<double> evaluateMultipleWeights(
        const std::vector<double>& weights,
        const std::vector<double>& input_values,
        std::function<double(const SimulationResult&)> fitness_func);
    
    // Encontrar mejor peso para una tarea
    double findBestWeight(
        const std::vector<double>& candidate_weights,
        const std::vector<std::vector<double>>& test_cases,
        std::function<double(const SimulationResult&)> fitness_func);
    
    // ===== UTILIDADES =====
    
    // Reset del simulador
    void reset();
    
    // Obtener información
    const SimulationConfig& getConfig() const { return config; }
    Network* getNetwork() { return network; }
    const Network* getNetwork() const { return network; }
    
    // Estadísticas
    void printStats(const SimulationResult& result) const;
};

#endif