#ifndef ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP
#define ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP

#include <petsc.h>
#include <memory>
#include <vector>
#include "domain/range.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
#include "twoPhaseEulerAdvection.hpp"

namespace ablate::finiteVolume::processes {

class IntSharp : public Process {

   private:
    //coeffs
    PetscReal Gamma;
    PetscReal epsilon;
    bool flipPhiTilde;
    // If true: IS::PreStage updates ONLY the volumeFraction (alpha) slot.
    // rho, rhoAlpha, rhoE, rhoU are left untouched.  Per the user, this
    // matches earlier IS+SF runs that were stable; the full mixture rebuild
    // (alphaOnly=false) introduces step-discontinuous EOS state changes that
    // amplify at cusps (right-cusp tip drives |v|->63 m/s -> NaN).
    // Default true.
    bool alphaOnly;
    // LAPLACE-YOUNG TEST MODE (analogous to ZalesakTest's hardcode of
    // non-essential fields in twoPhaseEulerAdvection).
    //
    // When laplaceYoungTest=true, IS::PreStage applies bulk-phase pinning
    // every stage:
    //   * Pure-gas cells (alpha >= 0.99): forced to ambient gas state
    //     (rho=rho_G, v=0, rhoE=rho_G*eG).  Eliminates gas-side acoustics.
    //   * Pure-liquid cells (alpha <= 0.01): velocity pinned to 0
    //     (rhoU=0).  Preserves rho/rhoE so the LY pressure jump can still
    //     develop in the bulk liquid via EOS, but kills bulk acoustics.
    //   * Interface band (0.01 < alpha < 0.99): UNTOUCHED.  Full IS+SF
    //     physics, exactly as in production code.
    //
    // The flag is OFF by default; pinning is gated entirely on this
    // single switch and never affects production code paths.  Per the
    // dissertation (Sec. 4.2.1, 4.3), the IS+SF case is not committed
    // to the full compressible flow model -- only the IS+SF interface
    // physics matters.  This mode is the LY analogue of ZalesakTest.
    bool laplaceYoungTest;
    // BJ (Barth-Jespersen) slope limiter on volumeFraction +
    // densityvolumeFraction is auto-enabled at Setup when Gamma>0.  This
    // flag allows force-DISABLING it independently for diagnostic isolation:
    // does Gamma>0 instability come from the Parameswaran-Mandal sharpening
    // term, or from the BJ limiter being on the vof field?  The two are
    // orthogonal but historically coupled via the Gamma>0 gate.
    // Default false (current production behavior: when Gamma>0, BJ ON).
    // Inverted naming so OPT(bool) absence in YAML preserves legacy behavior.
    bool disableBJSlopeLimiter;
    //mesh for vertex information
    DM vertexDM{};
    std::shared_ptr<ablate::domain::SubDomain> subDomain;
    
    // Boundary information
    PetscReal boundingBox[6];  // [xmin, xmax, ymin, ymax, zmin, zmax]
    PetscReal minRadius;
    PetscReal boundaryLayerThickness;
    PetscReal boundaryLayerMultiplier;
    std::map<PetscInt, PetscReal> cellBoundaryDistances;
    std::map<PetscInt, PetscReal> cellBoundaryWeights;

   public:
    /**
     *
     * @param Gamma
     * @param epsilon
     * @param flipPhiTilde
     * @param boundaryLayerMultiplier Optional multiplier for boundary layer thickness (default: 3.0)
     */
    explicit IntSharp(PetscReal Gamma, PetscReal epsilon, bool flipPhiTilde, PetscReal boundaryLayerMultiplier = 3.0, bool alphaOnly = true, bool laplaceYoungTest = false, bool disableBJSlopeLimiter = false);

    /**
     * Clean up the dm created
     */
    ~IntSharp() override;

    /**
     * Setup the process to define the vertex dm
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

    /**
     * Compute boundary distances and weights for all cells
     * @param dm The domain mesh
     */
    void ComputeBoundaryInformation(DM dm);

    /**
     * Get boundary weight for a cell (1.0 for interior, 0.0 for boundary, smooth transition in between)
     * @param cell The cell index
     * @return Boundary weight between 0.0 and 1.0
     */
    PetscReal GetBoundaryWeight(PetscInt cell) const;

    /**
     * static function private function to compute interface regularization term and add source to eulerset
     * @param solver
     * @param dm
     * @param time
     * @param locX
     * @param fVec
     * @param ctx
     * @return
     */
    static PetscErrorCode ComputeTerm(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx);

    //intsharp prestage stuff
    inline const static std::string VOLUME_FRACTION_FIELD = eos::TwoPhase::VF;
    inline const static std::string DENSITY_VF_FIELD = ablate::finiteVolume::CompressibleFlowFields::CONSERVED + VOLUME_FRACTION_FIELD;
    PetscErrorCode PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime);
    std::map<PetscInt, std::vector<PetscInt>> cellNeighbors;
    std::map<PetscReal, std::vector<PetscReal>> cellWeights;
    std::map<PetscInt, std::vector<PetscInt>> vertexNeighbors;
};
}  // namespace ablate::finiteVolume::processes
#endif
