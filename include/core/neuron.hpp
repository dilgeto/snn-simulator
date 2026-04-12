#ifndef NEURON_HPP
#define NEURON_HPP

#include <random>

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
    double a; // Velocidad de recuperación
    double b; // Sensibilidad a v
    double c; // Reset del voltaje
    double d; // Aumento de u tras la espiga
    double v; // Potencial de membrana (voltaje)
    double u; // Recuperación
    double i_offset; // Corriente de offset constante (mA)
    double noise;    // Amplitud de ruido uniforme [-0.5, 0.5] * noise
    bool spiked;     // Indica si la neurona se dispara o no
    double g_exc;   // Conductancia excitatoria (AMPA)
    double g_inh;   // Conductancia inhibitoria (GABA)
    double tau_exc; // Constante de tiempo AMPA (ms)
    double tau_inh; // Constante de tiempo GABA (ms)

    static constexpr double V_THRESHOLD = 30.0;

    // Generador de ruido (thread-local para evitar condiciones de carrera)
    static thread_local std::mt19937 rng;
    static thread_local std::uniform_real_distribution<double> uniform;

public:
    IzhikevichNeuron(NeuronType type, double i_offset = 0.0, double noise = 0.0,
                     double tau_exc = 5.0, double tau_inh = 10.0);
    IzhikevichNeuron(double a, double b, double c, double d,
                     double i_offset = 0.0, double noise = 0.0,
                     double tau_exc = 5.0, double tau_inh = 10.0);

    void step(double dt);
    void addExcConductance(double g);  // Conductancia excitatoria: g_exc += g
    void addInhConductance(double g);  // Conductancia inhibitoria: g_inh += g
    void reset();

    bool hasSpiked() const { return spiked; }
    double getVoltage() const { return v; }
    double getRecovery() const { return u; }
    double getGExc() const { return g_exc; }
    double getGInh() const { return g_inh; }

    void setVoltage(double voltage) { v = voltage; }
    void setRecovery(double recovery) { u = recovery; }
    void setIOffset(double offset) { i_offset = offset; }
    void setNoise(double n) { noise = n; }
    void setTauExc(double tau) { tau_exc = tau; }
    void setTauInh(double tau) { tau_inh = tau; }

    double getIOffset() const { return i_offset; }
    double getNoise() const { return noise; }
};

#endif