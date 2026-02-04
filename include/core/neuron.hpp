enum class NeuronType {
    REGULAR_SPIKING,
    FAST_SPIKING,
    INTRINSICALLY_BURSTING,
    CHATTERING,
    LOW_THRESHOLD_SPIKING,
    RESONATOR
};

class IzhikevichNeuron {
private:
    // Parámetros de las ecuaciones diferenciales
    double a, b, c, d;
    double v;
    double u;
    double I_syn; // 
    bool spiked; // Indica si la neurona se dispara o no
    
    static constexpr double V_THRESHOLD = 30.0;
    
public:
    IzhikevichNeuron(NeuronType type);
    IzhikevichNeuron(double a, double b, double c, double d);
    
    void step(double dt);
    void addCurrent(double current);
    void reset();
    
    // Agregar 'const' a estos getters
    bool hasSpiked() const { return spiked; }          // ← AGREGAR const
    double getVoltage() const { return v; }            // ← AGREGAR const
    double getRecovery() const { return u; }           // ← AGREGAR const
    
    void setVoltage(double voltage) { v = voltage; }
    void setRecovery(double recovery) { u = recovery; }
};