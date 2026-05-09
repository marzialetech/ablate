#ifndef ABLATELIBRARY_ZERODERBOUNDARY_HPP
#define ABLATELIBRARY_ZERODERBOUNDARY_HPP

#include <mathFunctions/fieldFunction.hpp>
#include "boundaryCell.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "finiteVolume/processes/flowProcess.hpp"
#include "finiteVolume/processes/process.hpp"
#include "solver/solver.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"

namespace ablate::finiteVolume::boundaryConditions {

class ZeroDerBoundary : public BoundaryCell {
   private:
    // updateFunction is unused for ZeroDerBoundary (we override ComputeBoundary
    // directly to do the interior-cell copy), but the BoundaryCell interface
    // requires it.  Provide a fallback that calls the user formula if given.
    void updateFunction(PetscReal time, const PetscReal* x, PetscScalar* vals, PetscInt point) override;

    void ExtraSetup() override;

    const std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction;

    // boundaryToInterior[bcell] = interior cell adjacent to boundary cell `bcell`.
    // Built once in ExtraSetup; consulted every step in ComputeBoundary.
    std::vector<PetscInt> boundaryToInteriorMap;
    std::vector<PetscInt> boundaryCellList;  // parallel to map (same indices)

   public:
    ZeroDerBoundary(std::string boundaryName, std::vector<std::string> labelIds, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction);
    inline const static std::string VOLUME_FRACTION_FIELD = eos::TwoPhase::VF;
    inline const static std::string DENSITY_VF_FIELD = ablate::finiteVolume::CompressibleFlowFields::CONSERVED + VOLUME_FRACTION_FIELD;

    // Override ComputeBoundary to apply true zero-derivative (Neumann) BC:
    // copy each boundary ghost cell's solution from its adjacent interior
    // cell.  This makes the boundary "transparent" to outgoing pressure
    // waves -- without it, EssentialBoundary's fixed Dirichlet state acts
    // as a hard reflector (waves bounce back and accumulate inside the box,
    // crashing IS+SF in star2d at t=1.5e-4).
    void ComputeBoundary(PetscReal time, Vec locX, Vec locX_t, Vec cellGeomVec) override;
};
}  // namespace ablate::finiteVolume::boundaryConditions
#endif  // ABLATELIBRARY_ZERODERBOUNDARY_HPP

