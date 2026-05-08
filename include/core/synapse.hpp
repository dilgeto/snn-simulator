#ifndef SYNAPSE_HPP
#define SYNAPSE_HPP

class Synapse {
private:
    int pre_neuron_id;
    int post_neuron_id;
    bool g_type;   // true = excitatoria, false = inhibitoria
    bool enabled;  // false = deshabilitada (WANN: no participa en simulación)

public:
    Synapse(int pre_id, int post_id, bool excitatory)
        : pre_neuron_id(pre_id),
          post_neuron_id(post_id),
          g_type(excitatory),
          enabled(true) {}

    // El peso se pasa como parámetro en tiempo de evaluación
    double getCurrent(double weight) const {
        return weight;
    }

    // Getters
    int getPreNeuronId() const { return pre_neuron_id; }
    int getPostNeuronId() const { return post_neuron_id; }
    bool isExcitatory() const { return g_type; }

    void setPreNeuronId(int id) { pre_neuron_id = id; }
    void setPostNeuronId(int id) { post_neuron_id = id; }
    void setExcitatory(bool excitatory) { g_type = excitatory; }

    bool isEnabled() const { return enabled; }
    void enable()          { enabled = true; }
    void disable()         { enabled = false; }
};

#endif