#ifndef ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP
#define ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP

#include <petsc.h>
#include <memory>
#include <string>
#include <vector>
#include "domain/range.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
// #include "twoPhaseEulerAdvection.hpp"
#include "nPhaseAllaireAdvection.hpp"

namespace ablate::finiteVolume::processes {

class NPhaseIntSharp : public Process {

   public:
    /**
     * Selects the discrete sharpening form applied per RK stage.
     *
     *   PARAMESWARAN_MANDAL: cell-centered scalar form (default; bit-identical
     *     to historical behavior),
     *       d alpha_k / d_tau = Gamma_k * [ -alpha_k(1-alpha_k)(1-2 alpha_k)
     *                                       + epsilon_k(1-2 alpha_k) |grad alpha_k| ]
     *
     *   CHIU_LIN: conservative-divergence form (mass-conservative by
     *     construction; modulo discrete divergence approximation),
     *       d alpha_k / d_tau = div( Gamma_k * [ epsilon_k * grad(alpha_k)
     *                                            - alpha_k(1-alpha_k) * grad(alpha_k)/|grad(alpha_k)| ] )
     */
    enum class Form { PARAMESWARAN_MANDAL, CHIU_LIN };

   private:
    //mesh for vertex information
    DM vertexDM{};
    // Cell-centered flux DM used only in the CHIU_LIN code path.
    // Holds dim*phases dofs per cell so a single Vec can store the
    // per-phase flux vector for every cell. Built lazily in the first
    // PreStage call (when phases is known) to keep the constructor
    // independent of the EOS.
    DM fluxDM{};
    std::shared_ptr<ablate::domain::SubDomain> subDomain;
    std::vector<PetscReal> Gammak;
    std::vector<PetscReal> epsilonk;
    std::vector<PetscInt> flipPhiTildek;
    Form form;

    PetscReal boundingBox[6];  // [xmin, xmax, ymin, ymax, zmin, zmax]
    PetscReal minRadius;
    PetscReal boundaryLayerThickness;
    PetscReal boundaryLayerMultiplier;
    std::map<PetscInt, PetscReal> cellBoundaryDistances;
    std::map<PetscInt, PetscReal> cellBoundaryWeights;

    void ComputeBoundaryInformation(DM dm);

    /** Build the CHIU_LIN per-cell flux DM (idempotent). */
    void EnsureFluxDM(DM dm, PetscInt phases);

    /**
     * Get boundary weight for a cell (1.0 for interior, 0.0 for boundary, smooth transition in between)
     * @param cell The cell index
     * @return Boundary weight between 0.0 and 1.0
     */
    PetscReal GetBoundaryWeight(PetscInt cell) const;

    /** Static helper: parse a form string from yaml ("parameswaran_mandal" / "chiu_lin"). */
    static Form ParseForm(const std::string &s);

   public:

    explicit NPhaseIntSharp(
        const std::vector<PetscReal>& Gammak, 
        const std::vector<PetscReal>& epsilonk, 
        const std::vector<PetscInt>& flipPhiTildek,
        PetscReal boundaryLayerMultiplier = 3.0,
        std::string form = "parameswaran_mandal");

    ~NPhaseIntSharp() override;

    PetscErrorCode PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime);

    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;


};
}  // namespace ablate::finiteVolume::processes
#endif
