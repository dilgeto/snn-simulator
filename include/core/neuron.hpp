enum class NeuronType {
    REGULAR_SPIKING,
    FAST_SPIKING,
    INTRINSICALLY_BURSTING,
    CHATTERING,
    LOW_THRESHOLD_SPIKING,
    RESONATOR
};

class IzhikevichNeuron {
    // Atributos
    private:
        // Parámetros para las ecuaciones diferenciales
        double a, b, c, d;

        // Variables de estado
        double v; // Potencial de membrana
        double u; // Variable de recuperación

        // Para simulación
        double I_syn; // Corriente sináptica acumulada
        bool spiked; // Estado de spike

        // Umbral de spike
        static constexpr double V_THRESHOLD = 30.0; // mV
    
    // Métodos
    public:
        // Constructor con diferentes tipos neuronales
        IzhikevichNeuron(NeuronType type);
        IzhikevichNeuron(double a, double b, double c, double d);
        // Paso de integración (método de Euler)
        void step(double dt);
        // Recibir corriente sináptica
        void add_current(double current);
        // Reset después de spike
        void reset();
        bool has_spiked() { return spiked; }
        double get_voltage() { return v; }
};