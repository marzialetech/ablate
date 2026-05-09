#ifndef ABLATELIBRARY_FINITEVOLUME_SURFACEFORCE_HPP
#define ABLATELIBRARY_FINITEVOLUME_SURFACEFORCE_HPP

#include <petsc.h>
#include <memory>
#include <vector>
#include "domain/range.hpp"
#include "domain/RBF/rbf.hpp"
#include "domain/reverseRange.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
#include "twoPhaseEulerAdvection.hpp"

namespace ablate::finiteVolume::processes {

class SurfaceForce : public Process {


   private:
    //surface tension coefficient
    PetscReal sigma;
    PetscReal C;
    PetscReal N;
    bool flipPhiTilde;
    bool applyToSolution;  // Control whether surface forces are applied to solution field
    // Optional bottom-wall mask: surface tension is suppressed within
    // bottomWallMaskCells layers from the y-min boundary so the no-slip wall
    // cannot be lifted off by spurious capillary currents.  Default 0 = off.
    PetscInt bottomWallMaskCells;
    // Optional periodic-seam mask: surface tension is suppressed within
    // periodicSeamMaskCells layers from the x periodic boundary on either
    // side.  Default 0 = off.  Use this when the FV flux solver imperfectly
    // couples the periodic seam (small one-sided pressure-wave reflection
    // from the seam).  As long as the interface is far from the seam, this
    // has no physical impact.  Distinct from the SF Gaussian-stencil
    // periodic neighbor-coordinate fix (which is always on); that fix
    // ensures the SF stencil itself is symmetric across the seam, while
    // this mask zeros the body-force output near the seam to prevent any
    // residual asymmetry from feeding a checkerboard mode.
    PetscInt periodicSeamMaskCells;
    // LAPLACE-YOUNG TEST MODE.  Mirrors the laplaceYoungTest flag on
    // IntSharp.  When true, ComputeSource clips |kappa| to
    // laplaceYoungTestKappaC / dx_min before forming sigma*kappa*n.
    // Justification: at t=0 the dissertation's 7-pointed sin(7*theta)
    // star has cusp tips with grid-limited curvature kappa ~ 1/dx that
    // produce singular SF impulses; 8 interface cells go NaN within
    // ~150 stages with |v|max~2 m/s only in those cells while the bulk
    // is calm.  A grid-relative cap (C/dx, C ~ 5-10) leaves the natural
    // unperturbed curvature 1/R0 = O(50) m^-1 untouched (~90x headroom)
    // while taming only the cusp singularity.  OFF by default; never
    // affects production runs.
    bool laplaceYoungTest;
    PetscReal laplaceYoungTestKappaC;
    //mesh for vertex information
    DM vertexDM{};
    std::shared_ptr<ablate::domain::SubDomain> subDomain;
    // Cached y-min of the mesh bounding box, set in Setup().
    PetscReal yMinCached;
    PetscReal hCached;
    // Full domain bounding box cached at Setup() time.  Used by the periodic
    // neighbor coordinate fix in ComputeSource (per-step DMGetBoundingBox is
    // expensive).  xLenCached/yLenCached/zLenCached store xmax-xmin etc.
    PetscReal xMinCached, xMaxCached, xLenCached, xHalfCached;
    PetscReal yMaxCached, yLenCached, yHalfCached;
    PetscReal zMinCached, zMaxCached, zLenCached, zHalfCached;

   public:

    /**
     *
     * @param sigma
     * @param C
     * @param N
     * @param flipPhiTilde
     * @param applyToSolution Whether to apply surface forces to solution field (default: false)
     */
    explicit SurfaceForce(PetscReal sigma, PetscReal C, PetscReal N, bool flipPhiTilde, bool applyToSolution = false, PetscInt bottomWallMaskCells = 0, PetscInt periodicSeamMaskCells = 0, bool laplaceYoungTest = false, PetscReal laplaceYoungTestKappaC = 5.0);

    /**
     * Clean up the dm created
     */
    ~SurfaceForce() override;

    /**
     * Setup the process to define the vertex dm
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

    /**
     * static function private function to compute surface force and add source to eulerset
     * @param solver
     * @param dm
     * @param time
     * @param locX
     * @param fVec
     * @param ctx
     * @return
     */
    static PetscErrorCode ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx);

    std::map<PetscInt, std::vector<PetscInt>> cellNeighbors;
    std::map<PetscInt, std::vector<PetscInt>> cellNeighbors1;
    std::map<PetscInt, std::vector<PetscInt>> vertexNeighbors;
};
}  // namespace ablate::finiteVolume::processes
#endif
