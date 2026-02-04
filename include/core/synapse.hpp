#ifndef SYNAPSE_HPP
#define SYNAPSE_HPP

class Synapse {
private:
    int pre_neuron_id;
    int post_neuron_id;
    
public:
    // Constructor simplificado
    Synapse(int pre_id, int post_id)
        : pre_neuron_id(pre_id), 
        post_neuron_id(post_id) {}
    
    // El peso se pasa como parámetro en tiempo de evaluación
    double getCurrent(double weight) const {
        return weight;
    }
    
    // Getters
    int getPreNeuronId() const { return pre_neuron_id; }
    int getPostNeuronId() const { return post_neuron_id; }
};

#endif