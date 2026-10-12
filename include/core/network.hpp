#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <cstddef>
#include <vector>
#include "neuron.hpp"
#include "synapse.hpp"

class Network {
private:
    // Contenedores principales
    std::vector<IzhikevichNeuron> neurons;
    std::vector<Synapse> synapses;

    // Peso por sinapsis para el modo "per-sinapsis" (ver setSynapseWeight()).
    // Paralelo a `synapses` (mismo índice = mismo synapse id), 0.0 por
    // defecto. Solo se mantiene sincronizado con altas (addSynapse); no se
    // re-sincroniza en removeSynapse/insertNeuron/cascada de aislados — válido
    // únicamente en una red recién construida que no se vuelve a mutar después
    // de fijar los pesos (el caso de uso: evaluar una topología ya entrenada).
    std::vector<double> synapse_weights;

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

    // Topología
    bool allow_recurrent;

    // Helpers internos de eliminación
    void removeSynapseOnly(int syn_id);
    void removeNeuronOnly(int neuron_id);
    bool isIsolated(int neuron_id) const;
    void cascadeRemoveIsolated();

    // Devuelve true si 'target' es alcanzable desde 'start' siguiendo sinapsis salientes
    bool canReach(int start, int target) const;
    
public:
    // Constructor y copia
    Network(double timestep = 1.0, bool allow_recurrent = true);
    Network* clone() const { return new Network(*this); }
    
    // Construcción de la red
    int addNeuron(NeuronType type);
    int addInputNeuron(NeuronType type);
    int addOutputNeuron(NeuronType type);
    int addHiddenNeuron(NeuronType type);
    
    // Agregar sinapsis SIN peso
    int addSynapse(int pre_id, int post_id, bool excitatory = true);

    // Eliminación de topología (con cascade: neuronas ocultas que queden aisladas se eliminan)
    void removeSynapse(int syn_id);
    void removeNeuron(int neuron_id);

    // Resultado de insertNeuron: IDs necesarios para revertir la mutación
    struct NeuronInsertionResult {
        int neuron_id;    // nueva neurona oculta (-1 si falló)
        int syn_in;       // sinapsis pre → nueva neurona
        int syn_out;      // sinapsis nueva neurona → post
        int syn_disabled; // sinapsis original deshabilitada
    };

    // Mutación: inserta una neurona oculta deshabilitando la sinapsis existente.
    // Las dos sinapsis nuevas heredan el g_type original.
    NeuronInsertionResult insertNeuron(int syn_id, NeuronType type = NeuronType::REGULAR_SPIKING);

    // Habilitar / deshabilitar elementos (reversible, estilo WANN)
    void disableSynapse(int syn_id);
    void enableSynapse(int syn_id);
    void disableNeuron(int neuron_id);
    void enableNeuron(int neuron_id);
    
    // Simulación - el peso compartido se pasa como parámetro
    void step(double shared_weight);
    void reset();
    void run(double duration, double shared_weight);

    // Variantes "per-sinapsis": usan el peso fijado por synapse vía
    // setSynapseWeight() en vez de un escalar compartido. No modifican ni
    // comparten código con step(double)/run(double,double) — ver comentario
    // de `synapse_weights` arriba sobre su alcance.
    void setSynapseWeight(int syn_id, double magnitude);
    void step();
    void run(double duration);

    // Entrada/Salida
    void setInputCurrent(int neuron_id, double current);
    void setInputCurrents(const std::vector<double>& currents);
    
    std::vector<bool> getSpikes() const;
    std::vector<bool> getOutputSpikes() const;
    std::vector<double> getVoltages() const;
    std::vector<int> getOutputNeuronIds() const { return output_neurons; }
    std::vector<int> getInputNeuronIds() const { return input_neurons; }
    
    // Recurrencia
    void setRecurrent(bool recurrent) { allow_recurrent = recurrent; }
    bool isRecurrent() const { return allow_recurrent; }

    // Información de la red
    size_t getNeuronCount() const { return neurons.size(); }
    size_t getSynapseCount() const { return synapses.size(); }
    size_t getActiveNeuronCount() const;
    size_t getActiveSynapseCount() const;
    size_t getInputCount() const { return input_neurons.size(); }
    size_t getOutputCount() const { return output_neurons.size(); }
    double getCurrentTime() const { return current_time; }
    double getTimestep() const { return dt; }
    
    // Acceso directo (para análisis)
    const IzhikevichNeuron& getNeuron(int id) const { return neurons[id]; }
    void setNeuronType(int neuron_id, NeuronType type) { neurons[neuron_id].setNeuronType(type); }

    // Aplica las mismas constantes de tiempo de decaimiento de conductancia
    // (AMPA/GABA) a TODAS las neuronas de la red. Llamar después de construir
    // la topología completa (nuevas neuronas agregadas después no se ven
    // afectadas retroactivamente si se llama antes).
    void setTimeConstants(double tau_exc, double tau_inh);
    void setSynapseExcitatory(int syn_id, bool excitatory) { synapses[syn_id].setExcitatory(excitatory); }
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