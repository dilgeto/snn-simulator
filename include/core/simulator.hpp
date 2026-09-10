#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include "network.hpp"
#include "../encoding/encoder.hpp"
#include "../decoding/decoder.hpp"
#include <cstddef>
#include <vector>
#include <memory>
#include <functional>

// Proxy que representa un individuo dentro del Simulator.
// Permite operar sobre una red concreta sin manejar índices manualmente.
class Individual {
    Network* net;
public:
    explicit Individual(Network* n) : net(n) {}

    // Mutaciones de topología (reversibles)
    Network::NeuronInsertionResult insertNeuron(int syn_id, NeuronType type = NeuronType::REGULAR_SPIKING) { return net->insertNeuron(syn_id, type); }
    int  insertSynapse(int pre_id, int post_id, bool excitatory = true)  { return net->addSynapse(pre_id, post_id, excitatory); }
    void disableSynapse(int syn_id)                                       { net->disableSynapse(syn_id); }
    void enableSynapse(int syn_id)                                        { net->enableSynapse(syn_id); }
    void disableNeuron(int neuron_id)                                     { net->disableNeuron(neuron_id); }
    void enableNeuron(int neuron_id)                                      { net->enableNeuron(neuron_id); }
    void toggleSynapsePolarity(int syn_id)                                { net->setSynapseExcitatory(syn_id, !net->getSynapse(syn_id).isExcitatory()); }
    void changeActivation(int neuron_id, NeuronType type)                 { net->setNeuronType(neuron_id, type); }
    void setRecurrent(bool recurrent)                                     { net->setRecurrent(recurrent); }

    // Eliminación física (permanente, para limpieza)
    void deleteSynapse(int syn_id)   { net->removeSynapse(syn_id); }
    void deleteNeuron(int neuron_id) { net->removeNeuron(neuron_id); }

    // Acceso directo a la red
    Network* getNetwork() { return net; }
    const Network* getNetwork() const { return net; }
};

struct SimulationConfig {
    double dt;
    double encoding_duration;
    double evaluation_duration;
    bool verbose;

    SimulationConfig()
        : dt(1.0),
          encoding_duration(50.0),
          evaluation_duration(100.0),
          verbose(false) {}
};

struct SimulationResult {
    std::vector<std::vector<double>> output_spike_times;
    std::vector<std::vector<double>> all_spike_times;
    std::vector<double> output_voltages;
    double total_spikes;
    double simulation_time_ms;

    SimulationResult() : total_spikes(0), simulation_time_ms(0) {}
};

class Simulator {
private:
    std::vector<Network*> networks;
    std::vector<bool> owns_networks;
    std::unique_ptr<Encoder> encoder;
    std::unique_ptr<RLDecoder> decoder;
    SimulationConfig config;

    Network* getNet(int idx) const;
    SimulationResult runSimulation(Network* net, const std::vector<double>& input_currents, double shared_weight, double duration);
    SimulationResult runSimulationEncoded(Network* net, const std::vector<double>& input_values, double shared_weight, double duration);

public:
    explicit Simulator(const SimulationConfig& cfg = SimulationConfig());
    explicit Simulator(Network* net, const SimulationConfig& cfg = SimulationConfig());
    Simulator(double dt = 1.0);
    ~Simulator();

    // ===== POBLACIÓN =====

    int addNetwork(Network* net, bool take_ownership = false);
    void removeNetwork(int idx);
    void replaceNetwork(int idx, Network* net, bool take_ownership = false);
    int cloneNetwork(int src_idx);
    Simulator* clone() const;

    Network* getNetwork(int idx = 0);
    const Network* getNetwork(int idx = 0) const;
    size_t getNetworkCount() const { return networks.size(); }
    Individual getIndividual(int idx) { return Individual(getNet(idx)); }

    // ===== CONFIGURACIÓN =====

    void setEncoder(std::unique_ptr<Encoder> enc);
    void setDecoder(std::unique_ptr<RLDecoder> dec);
    void setConfig(const SimulationConfig& cfg);
    const SimulationConfig& getConfig() const { return config; }

    // ===== SIMULACIÓN =====

    SimulationResult simulate(int net_idx, const std::vector<double>& input_currents, double shared_weight, double duration = 0.0);
    SimulationResult simulateEncoded(int net_idx, const std::vector<double>& input_values, double shared_weight, double duration = 0.0);

    std::vector<SimulationResult> simulateAll(const std::vector<double>& input_currents, double shared_weight, double duration = 0.0);
    std::vector<SimulationResult> simulateAllEncoded(const std::vector<double>& input_values, double shared_weight, double duration = 0.0);

    // ===== REINFORCEMENT LEARNING =====

    double runEpisode(int net_idx, const std::vector<double>& initial_state, double shared_weight, int max_steps,
                      std::function<bool(const std::vector<double>&, int&, double&)> step_callback);

    double evaluateFitness(int net_idx, const std::vector<std::vector<double>>& test_cases, double shared_weight,
                           std::function<double(const SimulationResult&)> fitness_func);

    std::vector<double> evaluateMultipleWeights(int net_idx, const std::vector<double>& weights,
                                                const std::vector<double>& input_values,
                                                std::function<double(const SimulationResult&)> fitness_func);

    double findBestWeight(int net_idx, const std::vector<double>& candidate_weights,
                          const std::vector<std::vector<double>>& test_cases,
                          std::function<double(const SimulationResult&)> fitness_func);

    // ===== RESET =====

    void reset(int net_idx);
    void resetAll();

    // ===== UTILIDADES =====

    void printStats(const SimulationResult& result) const;
};

#endif
