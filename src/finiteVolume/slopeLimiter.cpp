#include "slopeLimiter.hpp"
#include <petsc/private/dmpleximpl.h>
#include <utilities/petscUtilities.hpp>
#include <cmath>

namespace ablate::finiteVolume {

// -----------------------------------------------------------------------------
// SlopeLimiter::Setup
//
// Builds the cell-to-faces map. For each real cell we record every bounding
// face (via DMPlexGetCone), regardless of whether the face is interior or
// boundary. The boundary case is handled at the per-face loop in ApplyLimiter
// (we skip the BJ contribution from a face that has only one cell on its
// support side, which is the correct first-order treatment for a cell against
// a domain wall: the gradient simply isn't constrained by that face).
// -----------------------------------------------------------------------------
void SlopeLimiter::Setup(DM dm, const domain::Range& cellRange) {
    if (isSetup) return;

    PetscInt dim;
    DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;

    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;

    is1D = (dim == 1);

    // Half-mesh-size shortcut used by the 1D path.
    DMPlexGetMinRadius(dm, &minCellRadius) >> utilities::PetscUtilities::checkError;

    cellToFaces.assign(cEnd - cStart, std::vector<PetscInt>());
    for (PetscInt c = cStart; c < cEnd; ++c) {
        const PetscInt* faces;
        PetscInt numFaces;
        DMPlexGetCone(dm, c, &faces) >> utilities::PetscUtilities::checkError;
        DMPlexGetConeSize(dm, c, &numFaces) >> utilities::PetscUtilities::checkError;
        if (numFaces > 0) {
            cellToFaces[c - cStart].assign(faces, faces + numFaces);
        }
    }

    (void)cellRange;  // not used; iteration is driven by ApplyLimiter
    isSetup = true;
}

// -----------------------------------------------------------------------------
// SlopeLimiter::ApplyLimiter
//
// In-place BJ limiting of the gradient stored in `gradGlobArray` against
// `dmGrad`'s section. Per-component, alpha is the minimum of all per-face
// alpha_f. Boundary faces (only one cell on support) contribute no constraint
// (the BJ literature standard). Ghost cells are skipped via the dm's "ghost"
// label.
// -----------------------------------------------------------------------------
void SlopeLimiter::ApplyLimiter(DM dm, DM dmGrad, PetscInt dim, const domain::Field& field,
                                const domain::Range& cellRange, Vec cellGeomVec, Vec faceGeomVec,
                                const PetscScalar* xLocalArray, PetscScalar* gradGlobArray) {
    if (!isSetup) {
        throw std::runtime_error("SlopeLimiter must be set up before applying limiter");
    }

    DMLabel ghostLabel = nullptr;
    DMGetLabel(dm, "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

    // Single user-facing toggle (driven by NPhaseIntSharp presence): if this
    // field has not been opted into BJ limiting, zero the gradient and exit.
    // Downstream MUSCL face reconstruction collapses to phi_face = phi_cell,
    // i.e. donor-cell behavior. Applies to both 1D and 2D/3D paths -- there is
    // no neighbor scan to do, so we never touch face/cell geometry vectors.
    if (!IsActiveFor(field.name)) {
        const PetscInt nCompZero = field.numberComponents;
        for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
            const PetscInt cell = cellRange.points ? cellRange.points[c] : c;
            if (cell < cStart || cell >= cEnd) continue;
            if (ghostLabel) {
                PetscInt g = -1;
                DMLabelGetValue(ghostLabel, cell, &g) >> utilities::PetscUtilities::checkError;
                if (g >= 0) continue;
            }
            PetscScalar* gradCell = nullptr;
            DMPlexPointGlobalRef(dmGrad, cell, gradGlobArray, &gradCell) >> utilities::PetscUtilities::checkError;
            if (!gradCell) continue;
            for (PetscInt comp = 0; comp < nCompZero; ++comp) {
                for (PetscInt d = 0; d < dim; ++d) {
                    gradCell[comp * dim + d] = 0.0;
                }
            }
        }
        return;
    }

    const PetscScalar* cellGeomArray = nullptr;
    const PetscScalar* faceGeomArray = nullptr;
    DM dmCellGeom = nullptr;
    DM dmFaceGeom = nullptr;
    if (!is1D) {
        VecGetDM(cellGeomVec, &dmCellGeom) >> utilities::PetscUtilities::checkError;
        VecGetDM(faceGeomVec, &dmFaceGeom) >> utilities::PetscUtilities::checkError;
        VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
        VecGetArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;
    }

    const PetscInt nComp = field.numberComponents;
    const PetscInt fieldId = field.id;

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        // skip out-of-cell-stratum points (e.g. faces) that the cellRange may
        // accidentally include in odd usage patterns
        if (cell < cStart || cell >= cEnd) continue;

        // skip ghost cells (real cells get gradient writes; ghost rows of the
        // global vec are out of scope of DMPlexPointGlobalRef)
        if (ghostLabel) {
            PetscInt g = -1;
            DMLabelGetValue(ghostLabel, cell, &g) >> utilities::PetscUtilities::checkError;
            if (g >= 0) continue;
        }

        // grab the cell's gradient slot; if the cell is owned-but-ghosted in
        // the global vec, the global ref returns NULL -- skip it
        PetscScalar* gradCell = nullptr;
        DMPlexPointGlobalRef(dmGrad, cell, gradGlobArray, &gradCell) >> utilities::PetscUtilities::checkError;
        if (!gradCell) continue;

        // pull this cell's centered field values
        const PetscScalar* xCell = nullptr;
        DMPlexPointLocalFieldRead(dm, cell, fieldId, xLocalArray, &xCell) >> utilities::PetscUtilities::checkError;
        if (!xCell) continue;

        // cell centroid: only needed for 2D/3D; 1D shortcut uses minCellRadius
        const PetscFVCellGeom* cg = nullptr;
        if (!is1D) {
            DMPlexPointLocalRead(dmCellGeom, cell, cellGeomArray, &cg) >> utilities::PetscUtilities::checkError;
            if (!cg) continue;
        }

        // For each component, compute a single BJ scaling factor and apply.
        for (PetscInt comp = 0; comp < nComp; ++comp) {
            const PetscReal cellVal = PetscRealPart(xCell[comp]);

            // gather min/max from face neighbors
            PetscReal minVal = cellVal;
            PetscReal maxVal = cellVal;

            const auto& faces = cellToFaces[cell - cStart];
            for (PetscInt face : faces) {
                PetscInt supSize = 0;
                const PetscInt* support = nullptr;
                DMPlexGetSupportSize(dm, face, &supSize) >> utilities::PetscUtilities::checkError;
                DMPlexGetSupport(dm, face, &support) >> utilities::PetscUtilities::checkError;
                if (supSize != 2) continue;  // domain-boundary face; no neighbor
                const PetscInt nb = (support[0] == cell) ? support[1] : support[0];

                const PetscScalar* xNb = nullptr;
                DMPlexPointLocalFieldRead(dm, nb, fieldId, xLocalArray, &xNb) >> utilities::PetscUtilities::checkError;
                if (!xNb) continue;
                const PetscReal nbVal = PetscRealPart(xNb[comp]);
                minVal = PetscMin(minVal, nbVal);
                maxVal = PetscMax(maxVal, nbVal);
            }

            // BJ alpha across faces of this cell
            PetscReal alpha = 1.0;
            for (PetscInt face : faces) {
                PetscInt supSize = 0;
                const PetscInt* support = nullptr;
                DMPlexGetSupportSize(dm, face, &supSize) >> utilities::PetscUtilities::checkError;
                DMPlexGetSupport(dm, face, &support) >> utilities::PetscUtilities::checkError;
                if (supSize != 2) continue;

                // r = faceCentroid - cellCentroid
                PetscReal r[3] = {0.0, 0.0, 0.0};
                if (is1D) {
                    // signed half-mesh: small face index -> -h, large -> +h
                    const auto& fs = cellToFaces[cell - cStart];
                    if (fs.size() == 2) {
                        const PetscInt leftFace = PetscMin(fs[0], fs[1]);
                        r[0] = (face == leftFace) ? -minCellRadius : minCellRadius;
                    } else {
                        r[0] = minCellRadius;
                    }
                } else {
                    const PetscFVFaceGeom* fg = nullptr;
                    DMPlexPointLocalRead(dmFaceGeom, face, faceGeomArray, &fg) >> utilities::PetscUtilities::checkError;
                    if (!fg) continue;
                    for (PetscInt d = 0; d < dim; ++d) {
                        r[d] = fg->centroid[d] - cg->centroid[d];
                    }
                }

                PetscReal delta = 0.0;
                for (PetscInt d = 0; d < dim; ++d) {
                    delta += PetscRealPart(gradCell[comp * dim + d]) * r[d];
                }

                if (PetscAbsReal(delta) < 1e-14) continue;

                PetscReal alpha_f;
                if (delta > 0.0) {
                    alpha_f = (maxVal - cellVal) / delta;
                } else {
                    alpha_f = (minVal - cellVal) / delta;
                }
                if (alpha_f < 0.0) alpha_f = 0.0;
                if (alpha_f > 1.0) alpha_f = 1.0;
                alpha = PetscMin(alpha, alpha_f);
            }

            // apply scaling to this component's gradient
            for (PetscInt d = 0; d < dim; ++d) {
                gradCell[comp * dim + d] *= alpha;
            }
        }
    }

    if (!is1D) {
        VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
        VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> utilities::PetscUtilities::checkError;
    }
}

}  // namespace ablate::finiteVolume
