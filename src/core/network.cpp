#include <algorithm>
#include <iostream>
#include "../../include/core/network.hpp"

Network::Network(double timestep)
    : current_time(0.0), dt(timestep) {
}

int Network::addNeuron(NeuronType type) {
    int id = neurons.size();
    neurons.emplace_back(type);
    
    // Expandir vectores de mapeo
    post_synapses.push_back(std::vector<int>());
    pre_synapses.push_back(std::vector<int>());
    
    return id;
}

// TODO: incluir la función de activación que se desee
int Network::addInputNeuron(NeuronType type) {
    int id = addNeuron(type);
    input_neurons.push_back(id);
    return id;
}

// TODO: incluir la función de activación que se desee
int Network::addOutputNeuron(NeuronType type) {
    int id = addNeuron(type);
    output_neurons.push_back(id);
    return id;
}

int Network::addHiddenNeuron(NeuronType type) {
    int id = addNeuron(type);
    hidden_neurons.push_back(id);
    return id;
}

// Agregar sinapsis - SOLO topología, sin peso
int Network::addSynapse(int pre_id, int post_id, bool excitatory) {
    // Validación
    if (pre_id < 0 || pre_id >= static_cast<int>(neurons.size()) ||
        post_id < 0 || post_id >= static_cast<int>(neurons.size())) {
        std::cerr << "Error: Invalid neuron IDs in synapse (" 
                  << pre_id << " -> " << post_id << ")" << std::endl;
        return -1;
    }
    
    // Evitar auto-conexiones (opcional)
    if (pre_id == post_id) {
        std::cerr << "Warning: Self-connection not allowed (" 
                  << pre_id << " -> " << post_id << ")" << std::endl;
        return -1;
    }
    
    int synapse_id = synapses.size();
    synapses.emplace_back(pre_id, post_id, excitatory);
    
    // Actualizar mapeos
    post_synapses[pre_id].push_back(synapse_id);
    pre_synapses[post_id].push_back(synapse_id);
    
    return synapse_id;
}

// Paso de simulación - CLAVE: recibe el peso compartido
void Network::step(double shared_weight) {
    // 1. Propagar spikes del paso anterior como conductancias
    for (size_t i = 0; i < neurons.size(); i++) {
        if (neurons[i].hasSpiked()) {
            for (int syn_id : post_synapses[i]) {
                const Synapse& syn = synapses[syn_id];
                int post_id = syn.getPostNeuronId();
                double w = syn.getCurrent(shared_weight);
                if (syn.isExcitatory())
                    neurons[post_id].addExcConductance(w);
                else
                    neurons[post_id].addInhConductance(w);
            }
        }
    }

    // 2. Integrar cada neurona
    for (size_t i = 0; i < neurons.size(); i++) {
        neurons[i].step(dt);
    }

    // 3. Avanzar el tiempo
    current_time += dt;
}

// Ejecutar simulación por una duración con peso compartido
void Network::run(double duration, double shared_weight) {
    double end_time = current_time + duration;
    
    while (current_time < end_time) {
        step(shared_weight);
    }
}

// Reset completo de la red
void Network::reset() {
    // Reset neuronas
    for (auto& neuron : neurons) {
        neuron.reset();
    }
    
    // Reset tiempo
    current_time = 0.0;
}

// Establecer corriente de entrada a una neurona específica
void Network::setInputCurrent(int neuron_id, double current) {
    if (neuron_id >= 0 && neuron_id < static_cast<int>(neurons.size())) {
        neurons[neuron_id].addExcConductance(current);
    }
}

// Establecer corrientes a todas las neuronas de entrada
void Network::setInputCurrents(const std::vector<double>& currents) {
    size_t n = std::min(currents.size(), input_neurons.size());
    for (size_t i = 0; i < n; i++) {
        neurons[input_neurons[i]].addExcConductance(currents[i]);
    }
}

// Obtener estado de spikes de todas las neuronas
std::vector<bool> Network::getSpikes() const {
    std::vector<bool> spikes(neurons.size());
    for (size_t i = 0; i < neurons.size(); i++) {
        spikes[i] = neurons[i].hasSpiked();
    }
    return spikes;
}

// Obtener spikes solo de neuronas de salida
std::vector<bool> Network::getOutputSpikes() const {
    std::vector<bool> spikes(output_neurons.size());
    for (size_t i = 0; i < output_neurons.size(); i++) {
        spikes[i] = neurons[output_neurons[i]].hasSpiked();
    }
    return spikes;
}

// Obtener voltajes de todas las neuronas
std::vector<double> Network::getVoltages() const {
    std::vector<double> voltages(neurons.size());
    for (size_t i = 0; i < neurons.size(); i++) {
        voltages[i] = neurons[i].getVoltage();
    }
    return voltages;
}

// Información de debug
void Network::printInfo() const {
    std::cout << "\n=== Network Information ===" << std::endl;
    std::cout << "Total neurons: " << neurons.size() << std::endl;
    std::cout << "  Input neurons: " << input_neurons.size() << std::endl;
    std::cout << "  Hidden neurons: " << hidden_neurons.size() << std::endl;
    std::cout << "  Output neurons: " << output_neurons.size() << std::endl;
    std::cout << "Total synapses: " << synapses.size() << std::endl;
    std::cout << "Current time: " << current_time << " ms" << std::endl;
    std::cout << "Timestep: " << dt << " ms" << std::endl;
    
    // Calcular uso de memoria
    size_t neuron_mem = neurons.size() * sizeof(IzhikevichNeuron);
    size_t synapse_mem = synapses.size() * sizeof(Synapse);
    size_t total_mem = neuron_mem + synapse_mem;
    
    std::cout << "Memory usage:" << std::endl;
    std::cout << "  Neurons: " << neuron_mem / 1024.0 << " KB" << std::endl;
    std::cout << "  Synapses: " << synapse_mem / 1024.0 << " KB" << std::endl;
    std::cout << "  Total: " << total_mem / 1024.0 << " KB" << std::endl;
    std::cout << "=========================" << std::endl;
}

// Mostrar topología de la red
void Network::printTopology() const {
    std::cout << "\n=== Network Topology ===" << std::endl;
    
    std::cout << "\nInput neurons: ";
    for (int id : input_neurons) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Hidden neurons: ";
    for (int id : hidden_neurons) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Output neurons: ";
    for (int id : output_neurons) {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nConnections:" << std::endl;
    for (size_t i = 0; i < synapses.size(); i++) {
        const auto& syn = synapses[i];
        std::cout << "  [" << i << "] " 
                  << syn.getPreNeuronId() << " -> " 
                  << syn.getPostNeuronId() << std::endl;
    }
    std::cout << "=========================" << std::endl;
}

void Network::applyInputSpikes(
    const std::vector<std::vector<double>>& encoded_spikes,
    double current_time) {
    
    size_t n = std::min(encoded_spikes.size(), input_neurons.size());
    
    for (size_t i = 0; i < n; i++) {
        int neuron_id = input_neurons[i];
        
        // Verificar si hay algún spike en el tiempo actual
        for (double spike_time : encoded_spikes[i]) {
            // Si el spike ocurre en este paso de tiempo (con tolerancia)
            if (std::abs(spike_time - current_time) < dt / 2.0) {
                // Inyectar corriente fuerte para forzar spike
                neurons[neuron_id].addExcConductance(100.0);
            }
        }
    }
}

std::vector<std::vector<double>> Network::runAndCollectOutputSpikes(
    double duration,
    double shared_weight) {
    
    std::vector<std::vector<double>> output_spike_times(output_neurons.size());
    
    double start_time = current_time;
    double end_time = start_time + duration;
    
    while (current_time < end_time) {
        step(shared_weight);
        
        // Recolectar spikes de salida
        for (size_t i = 0; i < output_neurons.size(); i++) {
            if (neurons[output_neurons[i]].hasSpiked()) {
                output_spike_times[i].push_back(current_time - start_time);
            }
        }
    }
    
    return output_spike_times;
}

void Network::fastReset() {
    for (auto& neuron : neurons) {
        neuron.reset();
    }
    current_time = 0.0;
}

// ===== ELIMINACIÓN DE TOPOLOGÍA =====

bool Network::isIsolated(int id) const {
    return post_synapses[id].empty() && pre_synapses[id].empty();
}

// Elimina una sinapsis y actualiza los mapeos de adyacencia (sin cascade).
// Usa swap-and-pop para mantener vectores contiguos.
void Network::removeSynapseOnly(int syn_id) {
    int pre_id  = synapses[syn_id].getPreNeuronId();
    int post_id = synapses[syn_id].getPostNeuronId();

    // Borrar de las listas de adyacencia
    auto& out = post_synapses[pre_id];
    out.erase(std::remove(out.begin(), out.end(), syn_id), out.end());
    auto& in = pre_synapses[post_id];
    in.erase(std::remove(in.begin(), in.end(), syn_id), in.end());

    int last = static_cast<int>(synapses.size()) - 1;
    if (syn_id != last) {
        // La última sinapsis se mueve a la posición syn_id; actualizar referencias
        int lp = synapses[last].getPreNeuronId();
        int lq = synapses[last].getPostNeuronId();
        std::replace(post_synapses[lp].begin(), post_synapses[lp].end(), last, syn_id);
        std::replace(pre_synapses[lq].begin(),  pre_synapses[lq].end(),  last, syn_id);
        synapses[syn_id] = synapses[last];
    }
    synapses.pop_back();
}

// Elimina una neurona (asume que ya no tiene sinapsis) y actualiza índices (sin cascade).
void Network::removeNeuronOnly(int neuron_id) {
    // Borrar de su lista de rol
    for (auto* list : {&input_neurons, &hidden_neurons, &output_neurons}) {
        list->erase(std::remove(list->begin(), list->end(), neuron_id), list->end());
    }

    int last = static_cast<int>(neurons.size()) - 1;
    if (neuron_id != last) {
        // La última neurona se mueve a neuron_id; actualizar sinapsis y listas de rol
        for (int sid : post_synapses[last])
            synapses[sid].setPreNeuronId(neuron_id);
        for (int sid : pre_synapses[last])
            synapses[sid].setPostNeuronId(neuron_id);

        for (auto* list : {&input_neurons, &hidden_neurons, &output_neurons})
            std::replace(list->begin(), list->end(), last, neuron_id);

        neurons[neuron_id]       = neurons[last];
        post_synapses[neuron_id] = post_synapses[last];
        pre_synapses[neuron_id]  = pre_synapses[last];
    }
    neurons.pop_back();
    post_synapses.pop_back();
    pre_synapses.pop_back();
}

void Network::cascadeRemoveIsolated() {
    bool changed = true;
    while (changed) {
        changed = false;
        for (int id : hidden_neurons) {
            if (isIsolated(id)) {
                removeNeuronOnly(id);
                changed = true;
                break;  // hidden_neurons cambió; reiniciar iteración
            }
        }
    }
}

void Network::removeSynapse(int syn_id) {
    if (syn_id < 0 || syn_id >= static_cast<int>(synapses.size())) {
        std::cerr << "removeSynapse: ID inválido " << syn_id << std::endl;
        return;
    }
    removeSynapseOnly(syn_id);
    cascadeRemoveIsolated();
}

void Network::removeNeuron(int neuron_id) {
    if (neuron_id < 0 || neuron_id >= static_cast<int>(neurons.size())) {
        std::cerr << "removeNeuron: ID inválido " << neuron_id << std::endl;
        return;
    }

    // Eliminar todas las sinapsis conectadas
    while (!post_synapses[neuron_id].empty())
        removeSynapseOnly(post_synapses[neuron_id][0]);
    while (!pre_synapses[neuron_id].empty())
        removeSynapseOnly(pre_synapses[neuron_id][0]);

    removeNeuronOnly(neuron_id);
    cascadeRemoveIsolated();
}
