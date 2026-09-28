// GENERADO por scripts/bipedal_reference/gen_reset_table.py: no editar a mano.
//
// Estado que deja BipedalWalker.reset() (incluido su paso de física con colisión continua) en
// función de la fuerza horizontal inicial F del casco, F en [-INITIAL_RANDOM, INITIAL_RANDOM].
// Por cuerpo (casco, muslo izq., pantorrilla izq., muslo der., pantorrilla der.):
//   origen x, origen y, ángulo, velocidad x, velocidad y (centro de masa), velocidad angular.
//   estado(F) = S0 + F * SLOPE_NEG   si F < 0
//   estado(F) = S0 + F * SLOPE_POS   si F >= 0
#pragma once

namespace bipedal_walker::physics {

    template <typename T>
    struct ResetTable {
        static constexpr int VALUES = 30;

        static constexpr T S0[VALUES] = {
        4.66364527, 5.68434095, 0.00274759624, 0, -0.199999988, 0,
        4.68962526, 4.85240507, 0.0448987633, 0, -0.199999988, 0,
        4.65028954, 3.92240524, -0.0949422494, -0.00487888977, -0.000145293583, 0.00842024107,
        4.71266937, 4.85296249, 0.0853052139, 0, -0.199999988, 0,
        4.67228651, 3.92029047, -0.0609888993, 4.50911981e-11, 9.62001536e-08, 1.69906045e-07
        };

        static constexpr T SLOPE_NEG[VALUES] = {
        7.42218701e-05, 7.76571429e-07, 5.47035948e-07, 0.00357239418, 2.74233766e-07, -8.21905268e-05,
        1.50940779e-05, -1.7838961e-06, -0.000104940422, 0.00125152348, 0.000113693797, -0.00406287262,
        -2.65604156e-05, -3.58909091e-07, 1.02111496e-05, -0.000397437796, -1.19682342e-05, 0.000685913279,
        3.11064935e-05, -2.87516883e-06, -7.68477694e-05, 0.00125160889, -0.000116435808, -0.00406272164,
        -2.04932468e-05, 1.68275325e-06, -2.30710977e-05, -1.7946293e-10, -1.45041962e-08, 4.47682529e-08
        };

        static constexpr T SLOPE_POS[VALUES] = {
        7.39599481e-05, 2.63428571e-07, -6.34192208e-08, 0.00349791984, 4.48296104e-07, -0.000134898661,
        3.19868052e-05, -1.55246753e-06, -7.43500597e-05, 0.00213236086, 6.43177506e-05, -0.00235061017,
        -2.39173506e-05, 1.68015584e-06, -3.1709426e-05, -6.04586673e-05, -2.00693636e-06, 0.000104294067,
        3.16902857e-05, -3.29854545e-06, -7.46618525e-05, 0.00213244739, -6.88232156e-05, -0.00235045705,
        -2.36217143e-05, 2.22207792e-06, -3.10473683e-05, -3.23262042e-09, -2.71324784e-08, -4.70666175e-08
        };
    };

}
