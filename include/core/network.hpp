#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include "neuron.hpp"
#include "synapse.hpp"

class Network {
private:
    // Contenedores principales
    std::vector<IzhikevichNeuron> neurons;
    std::vector<Synapse> synapses;
    
    // Mapeo para acceso rápido a conexiones
    // post_synapses[i] = lista de índices de sinapsis que SALEN de la neurona i
    std::vector<std::vector<int>> post_synapses;
    
    // pre_synapses[i] = lista de índices de sinapsis que LLEGAN a la neurona i
    std::vector<std::vector<int>> pre_synapses;
    
    // Información de capas
    std::vector<int> input_neurons;
    std::vector<int> output_neurons;
    std::vector<int> hidden_neurons;
    
    // Estado de simulación
    double current_time;
    double dt;
    
public:
    // Constructor
    Network(double timestep = 1.0);
    
    // Construcción de la red
    int addNeuron(NeuronType type);
    int addInputNeuron();
    int addOutputNeuron();
    int addHiddenNeuron(NeuronType type = NeuronType::REGULAR_SPIKING);
    
    // Agregar sinapsis SIN peso
    int addSynapse(int pre_id, int post_id);
    
    // Simulación - el peso compartido se pasa como parámetro
    void step(double shared_weight);
    void reset();
    void run(double duration, double shared_weight);
    
    // Entrada/Salida
    void setInputCurrent(int neuron_id, double current);
    void setInputCurrents(const std::vector<double>& currents);
    
    std::vector<bool> getSpikes() const;
    std::vector<bool> getOutputSpikes() const;
    std::vector<double> getVoltages() const;
    std::vector<int> getOutputNeuronIds() const { return output_neurons; }
    std::vector<int> getInputNeuronIds() const { return input_neurons; }
    
    // Información de la red
    size_t getNeuronCount() const { return neurons.size(); }
    size_t getSynapseCount() const { return synapses.size(); }
    size_t getInputCount() const { return input_neurons.size(); }
    size_t getOutputCount() const { return output_neurons.size(); }
    double getCurrentTime() const { return current_time; }
    double getTimestep() const { return dt; }
    
    // Acceso directo (para análisis)
    const IzhikevichNeuron& getNeuron(int id) const { return neurons[id]; }
    const Synapse& getSynapse(int id) const { return synapses[id]; }
    
    // Debug
    void printInfo() const;
    void printTopology() const;
    
    // Métodos para RL

    // Aplicar spikes codificacados directamente a neuronas de entrada
    void applyInputSpikes(const std::vector<std::vector<double>>& encoded_spikes,
                         double current_time);
    
    // Ejecutar y recolectar spikes de salida durante una ventana
    std::vector<std::vector<double>> runAndCollectOutputSpikes(
        double duration,
        double shared_weight);
    
    // Reset rápido (sin limpiar topología)
    void fastReset();
};

#endif