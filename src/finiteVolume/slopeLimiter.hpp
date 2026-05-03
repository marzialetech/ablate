#pragma once

#include <petsc.h>
#include <domain/domain.hpp>
#include <domain/range.hpp>
#include <set>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace ablate::finiteVolume {

/**
 * Barth-Jespersen slope limiter for cell-centered FV gradients.
 *
 * For each cell c with computed gradient g_c[component][d], BJ ensures that
 * the reconstructed face value
 *     u_face = u_c + g_c . (faceCentroid - cellCentroid)
 * never exceeds the local neighbor range [min_neighbor(u), max_neighbor(u)].
 * It does this by scaling the entire gradient by a single factor
 *     alpha = min over faces f of   alpha_f
 *     alpha_f = 1                            if delta_f == 0
 *             = min(1, (u_max - u_c)/delta)  if delta_f > 0
 *             = min(1, (u_min - u_c)/delta)  if delta_f < 0
 * with delta_f = g_c . (faceCentroid_f - cellCentroid_c).
 *
 * The 2D/3D path consumes PetscFVCellGeom/PetscFVFaceGeom vectors that the
 * caller already has available (from the same source PetscFV uses for face
 * reconstruction). The 1D path keeps a simple uniform-mesh shortcut.
 */
class SlopeLimiter {
   private:
    //! cell-to-faces map indexed by (cell - cStart). Entry [i] is the list of
    //! face DAG indices bounding cell (cStart + i).
    std::vector<std::vector<PetscInt>> cellToFaces;

    //! cStart used for the cellToFaces index translation
    PetscInt cStart = 0;
    //! cEnd used for the cellToFaces index translation
    PetscInt cEnd = 0;

    //! flag to indicate if we are in 1D mode
    bool is1D = false;

    //! flag to indicate if we are set up
    bool isSetup = false;

    //! 1D uniform-mesh half-cell-size shortcut for face vector |r|
    PetscReal minCellRadius = 0.0;

    //! Set of field names (`domain::Field::name`) for which the limiter is
    //! active. Default is empty: any field passed to `ApplyLimiter` whose name
    //! is not in this set has its gradient zeroed (donor-cell face value).
    //! The single opt-in site is `NPhaseIntSharp::Setup` via
    //! `FiniteVolumeSolver::EnableSlopeLimiterFor`, which propagates here.
    std::set<std::string> activatedFields_;

   public:
    SlopeLimiter() = default;
    ~SlopeLimiter() = default;

    bool IsSetup() const { return isSetup; }

    /**
     * Mark a field (by domain::Field::name) as eligible for BJ limiting.
     * Idempotent. Fields not registered receive a zeroed gradient on
     * `ApplyLimiter` entry, which collapses the downstream MUSCL face
     * reconstruction to donor-cell.
     */
    void EnableForField(const std::string& fieldName) { activatedFields_.insert(fieldName); }

    /**
     * Has `EnableForField(fieldName)` been called? Used by `ApplyLimiter` to
     * decide whether to limit or to zero the gradient.
     */
    [[nodiscard]] bool IsActiveFor(const std::string& fieldName) const {
        return activatedFields_.find(fieldName) != activatedFields_.end();
    }

    /**
     * Build cell connectivity for the limiter.
     * @param dm  the cell DM
     * @param cellRange  unused beyond sanity; kept for API compatibility
     */
    void Setup(DM dm, const domain::Range& cellRange);

    /**
     * Apply the BJ limiter in-place to a global gradient vector.
     *
     * The gradient layout matches PETSc's convention: per-cell
     * `[c0_d0, c0_d1, c0_d{dim-1}, c1_d0, ...]` for `comp` components per
     * cell, accessed via `DMPlexPointGlobalRef(dmGrad, cell, gradArray, &g)`.
     *
     * @param dm  the cell DM (used for ghost label, geometry vectors)
     * @param dmGrad  the gradient DM (1 dof per (component * dim) per cell)
     * @param dim  spatial dimension
     * @param field  field being limited
     * @param cellRange  iteration range over real cells
     * @param cellGeomVec  PetscFVCellGeom vector (centroid + volume per cell)
     * @param faceGeomVec  PetscFVFaceGeom vector (centroid + normal per face)
     * @param xLocalArray  raw pointer to the local solution Vec
     * @param gradGlobArray  raw pointer to the global gradient Vec (modified in place)
     */
    void ApplyLimiter(DM dm, DM dmGrad, PetscInt dim, const domain::Field& field,
                      const domain::Range& cellRange, Vec cellGeomVec, Vec faceGeomVec,
                      const PetscScalar* xLocalArray, PetscScalar* gradGlobArray);
};

}  // namespace ablate::finiteVolume
