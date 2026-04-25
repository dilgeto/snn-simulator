// src/core/simulator.cpp
#include "../../include/core/simulator.hpp"
#include <iostream>
#include <chrono>
#include <cmath>

Simulator::Simulator(Network* net, const SimulationConfig& cfg)
    : network(net), config(cfg), owns_network(false) {
}

Simulator::Simulator(double dt)
    : network(nullptr), owns_network(false) {
    config.dt = dt;
}

Simulator::~Simulator() {
    if (owns_network && network) {
        delete network;
    }
}

void Simulator::setNetwork(Network* net, bool take_ownership) {
    if (owns_network && network) {
        delete network;
    }
    network = net;
    owns_network = take_ownership;
}

void Simulator::setEncoder(std::unique_ptr<Encoder> enc) {
    encoder = std::move(enc);
}

void Simulator::setDecoder(std::unique_ptr<RLDecoder> dec) {
    decoder = std::move(dec);
}

void Simulator::setConfig(const SimulationConfig& cfg) {
    config = cfg;
}

void Simulator::reset() {
    if (network) {
        network->reset();
    }
}

void Simulator::removeNeuron(int neuron_id) {
    if (network) network->removeNeuron(neuron_id);
}

int Simulator::insertNeuron(int syn_id, NeuronType type) {
    if (!network) return -1;
    return network->insertNeuron(syn_id, type);
}

int Simulator::insertSynapse(int pre_id, int post_id, bool excitatory) {
    if (!network) return -1;
    return network->addSynapse(pre_id, post_id, excitatory);
}

void Simulator::deleteSynapse(int syn_id) {
    if (network) network->removeSynapse(syn_id);
}

void Simulator::toggleSynapsePolarity(int syn_id) {
    if (!network) return;
    const Synapse& syn = network->getSynapse(syn_id);
    network->setSynapseExcitatory(syn_id, !syn.isExcitatory());
}

void Simulator::changeActivation(int neuron_id, NeuronType type) {
    if (network) network->setNeuronType(neuron_id, type);
}

// ===== SIMULACIÓN BÁSICA =====

SimulationResult Simulator::simulate(
    const std::vector<double>& input_currents,
    double shared_weight,
    double duration) {
    
    if (!network) {
        std::cerr << "Error: No network set in simulator" << std::endl;
        return SimulationResult();
    }
    
    if (duration <= 0) {
        duration = config.evaluation_duration;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    SimulationResult result;
    network->reset();
    
    size_t num_outputs = network->getOutputCount();
    result.output_spike_times.resize(num_outputs);
    
    double total_spikes = 0;
    
    // Ejecutar simulación
    for (double t = 0; t < duration; t += config.dt) {
        // Aplicar corrientes de entrada
        network->setInputCurrents(input_currents);
        
        // Ejecutar paso
        network->step(shared_weight);
        
        // Recolectar spikes de salida
        auto output_spikes = network->getOutputSpikes();
        for (size_t i = 0; i < output_spikes.size(); i++) {
            if (output_spikes[i]) {
                result.output_spike_times[i].push_back(t);
                total_spikes++;
            }
        }
    }
    
    // Obtener voltajes finales
    auto voltages = network->getVoltages();
    auto output_ids = network->getOutputNeuronIds();
    for (int id : output_ids) {
        result.output_voltages.push_back(voltages[id]);
    }
    
    result.total_spikes = total_spikes;
    
    auto end = std::chrono::high_resolution_clock::now();
    result.simulation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (config.verbose) {
        printStats(result);
    }
    
    return result;
}

SimulationResult Simulator::simulateEncoded(
    const std::vector<double>& input_values,
    double shared_weight,
    double duration) {
    
    if (!network) {
        std::cerr << "Error: No network set in simulator" << std::endl;
        return SimulationResult();
    }
    
    if (!encoder) {
        std::cerr << "Error: No encoder set in simulator" << std::endl;
        return SimulationResult();
    }
    
    if (duration <= 0) {
        duration = config.evaluation_duration;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    SimulationResult result;
    network->reset();
    
    // Codificar entradas
    auto encoded_inputs = encoder->encodeVector(input_values, config.encoding_duration, config.dt);
    
    // Verificar que el número de entradas coincida
    if (encoded_inputs.size() != network->getInputCount()) {
        std::cerr << "Error: Input size mismatch. Expected " 
                  << network->getInputCount() << ", got " 
                  << encoded_inputs.size() << std::endl;
        return SimulationResult();
    }
    
    size_t num_outputs = network->getOutputCount();
    result.output_spike_times.resize(num_outputs);
    
    double total_spikes = 0;
    
    // Ejecutar simulación
    for (double t = 0; t < duration; t += config.dt) {
        // Aplicar spikes codificados a las neuronas de entrada
        auto input_ids = network->getInputNeuronIds();
        for (size_t i = 0; i < encoded_inputs.size(); i++) {
            // Buscar spikes que ocurren en este tiempo
            for (double spike_time : encoded_inputs[i]) {
                if (std::abs(spike_time - t) < config.dt / 2.0) {
                    // Inyectar corriente fuerte para generar spike
                    network->setInputCurrent(input_ids[i], 100.0);
                }
            }
        }
        
        // Ejecutar paso
        network->step(shared_weight);
        
        // Recolectar spikes de salida
        auto output_spikes = network->getOutputSpikes();
        for (size_t i = 0; i < output_spikes.size(); i++) {
            if (output_spikes[i]) {
                result.output_spike_times[i].push_back(t);
                total_spikes++;
            }
        }
    }
    
    // Voltajes finales
    auto voltages = network->getVoltages();
    auto output_ids = network->getOutputNeuronIds();
    for (int id : output_ids) {
        result.output_voltages.push_back(voltages[id]);
    }
    
    result.total_spikes = total_spikes;
    
    auto end = std::chrono::high_resolution_clock::now();
    result.simulation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (config.verbose) {
        printStats(result);
    }
    
    return result;
}

// ===== REINFORCEMENT LEARNING =====

double Simulator::runEpisode(
    const std::vector<double>& initial_state,
    double shared_weight,
    int max_steps,
    std::function<bool(const std::vector<double>&, int&, double&)> step_callback) {
    
    if (!network || !encoder || !decoder) {
        std::cerr << "Error: Network, encoder, or decoder not set" << std::endl;
        return 0.0;
    }
    
    double total_reward = 0.0;
    std::vector<double> current_state = initial_state;
    
    for (int step = 0; step < max_steps; step++) {
        auto result = simulateEncoded(current_state, shared_weight, config.evaluation_duration);
        
        // Decodificar acción
        int action = decoder->decodeDiscreteAction(result.output_spike_times);
        
        // Ejecutar paso en el ambiente
        double reward = 0.0;
        bool done = step_callback(current_state, action, reward);
        
        total_reward += reward;
        
        if (done) {
            break;
        }
    }
    
    return total_reward;
}

double Simulator::evaluateFitness(
    const std::vector<std::vector<double>>& test_cases,
    double shared_weight,
    std::function<double(const SimulationResult&)> fitness_func) {
    
    if (!network) {
        return 0.0;
    }
    
    double total_fitness = 0.0;
    
    for (const auto& test_case : test_cases) {
        auto result = simulateEncoded(test_case, shared_weight, config.evaluation_duration);
        total_fitness += fitness_func(result);
    }
    
    return total_fitness / test_cases.size();
}

// ===== PARA WANN =====

std::vector<double> Simulator::evaluateMultipleWeights(
    const std::vector<double>& weights,
    const std::vector<double>& input_values,
    std::function<double(const SimulationResult&)> fitness_func) {
    
    std::vector<double> fitness_scores;
    fitness_scores.reserve(weights.size());
    
    for (double weight : weights) {
        auto result = simulateEncoded(input_values, weight, config.evaluation_duration);
        fitness_scores.push_back(fitness_func(result));
    }
    
    return fitness_scores;
}

double Simulator::findBestWeight(
    const std::vector<double>& candidate_weights,
    const std::vector<std::vector<double>>& test_cases,
    std::function<double(const SimulationResult&)> fitness_func) {
    
    double best_weight = candidate_weights[0];
    double best_fitness = -std::numeric_limits<double>::infinity();
    
    for (double weight : candidate_weights) {
        double avg_fitness = evaluateFitness(test_cases, weight, fitness_func);
        
        if (config.verbose) {
            std::cout << "Weight " << weight << ": fitness = " << avg_fitness << std::endl;
        }
        
        if (avg_fitness > best_fitness) {
            best_fitness = avg_fitness;
            best_weight = weight;
        }
    }
    
    if (config.verbose) {
        std::cout << "Best weight: " << best_weight 
                  << " (fitness: " << best_fitness << ")" << std::endl;
    }
    
    return best_weight;
}

// ===== UTILIDADES =====

void Simulator::printStats(const SimulationResult& result) const {
    std::cout << "\n=== Simulation Statistics ===" << std::endl;
    std::cout << "Total output spikes: " << result.total_spikes << std::endl;
    std::cout << "Simulation time: " << result.simulation_time_ms << " ms" << std::endl;
    
    std::cout << "Spikes per output neuron:" << std::endl;
    for (size_t i = 0; i < result.output_spike_times.size(); i++) {
        std::cout << "  Output " << i << ": " 
                  << result.output_spike_times[i].size() << " spikes";
        if (!result.output_spike_times[i].empty()) {
            std::cout << " (first at " << result.output_spike_times[i][0] << "ms)";
        }
        std::cout << std::endl;
    }
    
    if (!result.output_voltages.empty()) {
        std::cout << "Final output voltages:" << std::endl;
        for (size_t i = 0; i < result.output_voltages.size(); i++) {
            std::cout << "  Output " << i << ": " 
                      << result.output_voltages[i] << " mV" << std::endl;
        }
    }
    
    std::cout << "=========================" << std::endl;
}