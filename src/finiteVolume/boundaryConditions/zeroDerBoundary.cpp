#include "zeroDerBoundary.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "solver/solver.hpp"
#include "utilities/petscUtilities.hpp"
#include "utilities/petscSupport.hpp"

#include <petsc.h>
#include <memory>
#include <vector>

#include "domain/range.hpp"

ablate::finiteVolume::boundaryConditions::ZeroDerBoundary::ZeroDerBoundary(std::string boundaryName, std::vector<std::string> labelIds, std::shared_ptr<ablate::mathFunctions::FieldFunction> boundaryFunction)
    : BoundaryCell(boundaryFunction->GetName(), boundaryName, labelIds), boundaryFunction(boundaryFunction) {
    if (!boundaryFunction) {
        throw std::invalid_argument("ZeroDerBoundary must be constructed with a valid boundary function");
    }
    PetscPrintf(PETSC_COMM_WORLD, "ZeroDerBoundary created: %s\n", boundaryName.c_str());
    PetscPrintf(PETSC_COMM_WORLD, "Associated labels: ");
    for (const auto& label : labelIds) {
        PetscPrintf(PETSC_COMM_WORLD, "%s ", label.c_str());
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n");
}

// Build the boundary->interior cell mapping.  For each boundary ghost cell,
// walk its faces (cone) and find the support cell on each face that is NOT
// a boundary cell -- that is the adjacent interior neighbor.  In a box mesh
// with single-layer ghost cells (BoxMeshBoundaryCells), each boundary cell
// has exactly one such neighbor along the inward normal; corner boundary
// cells are part of two side labels and may not have any pure-interior
// neighbor (we leave their map entry as -1 and fall back to the user formula
// for those).
void ablate::finiteVolume::boundaryConditions::ZeroDerBoundary::ExtraSetup() {
    DM dm = subDomain->GetDM();

    // Build a quick lookup: is point P labeled as a boundary cell on ANY side?
    // We use the standard side labels emitted by BoxMeshBoundaryCells.  We
    // ALSO use the "interiorCells" label as the positive interior criterion
    // -- this is critical because BoxMeshBoundaryCells does NOT label the
    // *corner* boundary cells with any of the side labels (a corner cell
    // belongs to no single side).  Without this, the corner cell would be
    // misidentified as interior, and edge-adjacent boundary cells would copy
    // FROM the corner instead of from the genuine interior cell along the
    // inward normal -- which crashes IS+SF immediately at step 2.
    static const std::vector<std::string> kBoundarySideLabels = {
        "boundaryCellsLeft", "boundaryCellsRight",
        "boundaryCellsTop",  "boundaryCellsBottom",
        "boundaryCellsFront", "boundaryCellsBack",
    };
    std::vector<DMLabel> bdyLabels;
    for (const auto& name : kBoundarySideLabels) {
        DMLabel lbl = nullptr;
        DMGetLabel(dm, name.c_str(), &lbl);
        if (lbl) bdyLabels.push_back(lbl);
    }
    DMLabel interiorLabel = nullptr;
    DMGetLabel(dm, "interiorCells", &interiorLabel);
    auto isInteriorCell = [&](PetscInt pt) -> bool {
        if (!interiorLabel) return false;
        PetscInt v = -1;
        DMLabelGetValue(interiorLabel, pt, &v);
        return v >= 0;
    };
    auto isBoundaryCell = [&](PetscInt pt) -> bool {
        for (DMLabel lbl : bdyLabels) {
            PetscInt v = -1;
            DMLabelGetValue(lbl, pt, &v);
            if (v >= 0) return true;
        }
        return false;
    };

    // Pull the IS of boundary points this BC applies to (set up by parent class).
    // The parent stores it in `pointsIS` (private) -- but ComputeBoundary
    // re-creates `points`/`nPoints` on-the-fly.  We rebuild the same view here
    // by iterating over our labelIds (same logic as parent's SetupBoundary).
    PetscInt pStart, pEnd;
    DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd) >> utilities::PetscUtilities::checkError;

    std::vector<PetscInt> uniqueBdyPoints;
    {
        std::vector<bool> seen(static_cast<size_t>(pEnd - pStart), false);
        // (we don't have direct access to BoundaryCell::pointsIS; reconstruct
        // by iterating over all known boundary side labels for our labelIds)
        for (DMLabel lbl : bdyLabels) {
            IS valuesIS;
            DMLabelGetNonEmptyStratumValuesIS(lbl, &valuesIS);
            if (!valuesIS) continue;
            PetscInt nValues;
            const PetscInt *values;
            ISGetSize(valuesIS, &nValues);
            ISGetIndices(valuesIS, &values);
            for (PetscInt v = 0; v < nValues; ++v) {
                IS subIS;
                DMLabelGetStratumIS(lbl, values[v], &subIS);
                if (!subIS) continue;
                PetscInt nPts;
                const PetscInt *pts;
                ISGetLocalSize(subIS, &nPts);
                ISGetIndices(subIS, &pts);
                for (PetscInt i = 0; i < nPts; ++i) {
                    PetscInt p = pts[i];
                    if (p >= pStart && p < pEnd && !seen[p - pStart]) {
                        seen[p - pStart] = true;
                        uniqueBdyPoints.push_back(p);
                    }
                }
                ISRestoreIndices(subIS, &pts);
                ISDestroy(&subIS);
            }
            ISRestoreIndices(valuesIS, &values);
            ISDestroy(&valuesIS);
        }
    }

    // For each boundary cell, find an interior neighbor by walking faces.
    boundaryCellList.clear();
    boundaryToInteriorMap.clear();
    boundaryCellList.reserve(uniqueBdyPoints.size());
    boundaryToInteriorMap.reserve(uniqueBdyPoints.size());
    PetscInt cornerCount = 0;

    for (PetscInt bdyCell : uniqueBdyPoints) {
        PetscInt interior = -1;

        // Get the faces of this cell.
        PetscInt nFaces = 0;
        const PetscInt *faces = nullptr;
        DMPlexGetConeSize(dm, bdyCell, &nFaces);
        DMPlexGetCone(dm, bdyCell, &faces);

        for (PetscInt f = 0; f < nFaces; ++f) {
            PetscInt nSupports = 0;
            const PetscInt *supports = nullptr;
            DMPlexGetSupportSize(dm, faces[f], &nSupports);
            DMPlexGetSupport(dm, faces[f], &supports);

            for (PetscInt s = 0; s < nSupports; ++s) {
                if (supports[s] == bdyCell) continue;
                // Prefer the positive interior criterion (interiorCells label).
                // If that label exists, ONLY accept cells that have it.
                // Otherwise, fall back to "not a boundary cell".  The positive
                // check correctly excludes corner boundary cells, which have
                // no side label and would be wrongly accepted by the negative
                // check alone.
                if (interiorLabel) {
                    if (!isInteriorCell(supports[s])) continue;
                } else {
                    if (isBoundaryCell(supports[s])) continue;
                }
                interior = supports[s];
                break;
            }
            if (interior >= 0) break;
        }

        boundaryCellList.push_back(bdyCell);
        boundaryToInteriorMap.push_back(interior);
        if (interior < 0) cornerCount++;
    }

    PetscPrintf(PETSC_COMM_WORLD,
                "[ZeroDerBoundary::ExtraSetup] Built boundary->interior map: %zu boundary cells, "
                "%d corner cells (no pure-interior neighbor; fallback to user formula).\n",
                boundaryCellList.size(), (int)cornerCount);
}

// updateFunction fallback used only for corner cells that have no purely-
// interior face-neighbor (the user-supplied formula is evaluated there).
void ablate::finiteVolume::boundaryConditions::ZeroDerBoundary::updateFunction(PetscReal time, const PetscReal *x, PetscScalar *vals, PetscInt point) {
    boundaryFunction->GetSolutionField().GetPetscFunction()(dim, time, x, fieldSize, vals, boundaryFunction->GetSolutionField().GetContext());
}

// Override the parent ComputeBoundary to do the zero-derivative copy.
void ablate::finiteVolume::boundaryConditions::ZeroDerBoundary::ComputeBoundary(PetscReal time, Vec locX, Vec locX_t, Vec cellGeomVec) {
    // Build the boundary->interior mapping lazily on first call.  ExtraSetup
    // is never invoked by the parent (it's a dead virtual in the current
    // codebase), so we have to do this here.
    if (boundaryCellList.empty()) {
        ExtraSetup();
    }

    PetscScalar *array = nullptr;
    DM dmData = nullptr;
    DM dmCell = nullptr;
    const PetscScalar *cellGeomArray = nullptr;
    ablate::domain::Field field = subDomain->GetField(GetFieldName());

    VecGetDM(locX, &dmData) >> utilities::PetscUtilities::checkError;
    VecGetDM(cellGeomVec, &dmCell) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
    VecGetArray(locX, &array) >> utilities::PetscUtilities::checkError;

    for (size_t i = 0; i < boundaryCellList.size(); ++i) {
        PetscInt bdyCell  = boundaryCellList[i];
        PetscInt interior = boundaryToInteriorMap[i];

        PetscScalar *bdyVals = nullptr;
        xDMPlexPointLocalRef(dmData, bdyCell, field.id, array, &bdyVals) >> utilities::PetscUtilities::checkError;
        if (!bdyVals) continue;

        if (interior >= 0) {
            PetscScalar *intVals = nullptr;
            xDMPlexPointLocalRef(dmData, interior, field.id, array, &intVals) >> utilities::PetscUtilities::checkError;
            if (intVals) {
                for (PetscInt c = 0; c < fieldSize; ++c) {
                    bdyVals[c] = intVals[c];
                }
                continue;
            }
        }

        PetscFVCellGeom *cg = nullptr;
        DMPlexPointLocalRead(dmCell, bdyCell, cellGeomArray, &cg) >> utilities::PetscUtilities::checkError;
        if (cg) {
            updateFunction(time, cg->centroid, bdyVals, bdyCell);
        }
    }

    VecRestoreArray(locX, &array) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::boundaryConditions::BoundaryCondition,
         ablate::finiteVolume::boundaryConditions::ZeroDerBoundary,
         "Zero-derivative (Neumann) BC: copies the adjacent interior cell's "
         "solution into each boundary ghost cell every step.  Acts as a "
         "(zeroth-order) non-reflecting outflow boundary, so outgoing pressure "
         "waves are absorbed instead of reflected.  Pair with BoxMeshBoundaryCells.",
         ARG(std::string, "boundaryName", "name to register this boundary under"),
         ARG(std::vector<std::string>, "labelIds", "boundary side labels (e.g. boundaryCellsLeft)"),
         ARG(ablate::mathFunctions::FieldFunction, "boundaryValue",
             "fallback formula evaluated at corner ghost cells where no purely-interior face neighbor exists; "
             "for non-corner cells the formula is ignored and the interior value is copied verbatim"));
