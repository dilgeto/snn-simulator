#ifndef POISSON_ENCODER_HPP
#define POISSON_ENCODER_HPP

#include "encoder.hpp"
#include <random>

class PoissonEncoder : public Encoder {
private:
    double max_rate;
    bool use_refractory;
    double refractory_period;
    mutable std::mt19937 rng_;

public:
    PoissonEncoder(double max_rate = 100.0,
                   bool use_refractory = true,
                   double refractory_period = 2.0);

    std::vector<double> encode(double value, double duration, double dt) const override;
    std::vector<double> encodeTemporalPattern(const std::vector<double>& rate_pattern, double dt) const;

    void reseed(uint32_t seed) { rng_.seed(seed); }
    void setMaxRate(double rate) { max_rate = rate; }
    void setRefractoryPeriod(double period) { refractory_period = period; }
};

#endif
