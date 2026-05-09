#include "domain/RBF/mq.hpp"
#include "intSharp.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"
#include <fstream>
#include <PetscTime.h>


void ablate::finiteVolume::processes::IntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
    IntSharp::subDomain = solver.GetSubDomainPtr();
}
ablate::finiteVolume::processes::IntSharp::IntSharp(PetscReal Gamma, PetscReal epsilon, bool flipPhiTilde, PetscReal boundaryLayerMultiplier, bool alphaOnly, bool laplaceYoungTest, bool disableBJSlopeLimiter) : Gamma(Gamma), epsilon(epsilon), flipPhiTilde(flipPhiTilde), alphaOnly(alphaOnly), laplaceYoungTest(laplaceYoungTest), disableBJSlopeLimiter(disableBJSlopeLimiter), boundaryLayerMultiplier(boundaryLayerMultiplier) {
    // Initialize boundary layer thickness as a multiple of minRadius (will be set in Setup)
    boundaryLayerThickness = 0.0;
    minRadius = 0.0;
    for (int i = 0; i < 6; ++i) {
        boundingBox[i] = 0.0;
    }
}
ablate::finiteVolume::processes::IntSharp::~IntSharp() { DMDestroy(&vertexDM) >> utilities::PetscUtilities::checkError; }

void intSharpPreStageWrapper(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime, ablate::finiteVolume::processes::IntSharp* intSharpProcess) {
    intSharpProcess->PreStage(flowTs, solver, stagetime);
  }

void ablate::finiteVolume::processes::IntSharp::ComputeBoundaryInformation(DM dm) {
    PetscInt dim;
    DMGetDimension(dm, &dim);
    
    // Get bounding box of the domain
    PetscReal xymin[3], xymax[3];
    DMGetBoundingBox(dm, xymin, xymax);
    
    // Store bounding box in the format [xmin, xmax, ymin, ymax, zmin, zmax]
    boundingBox[0] = xymin[0];  // xmin
    boundingBox[1] = xymax[0];  // xmax
    boundingBox[2] = xymin[1];  // ymin
    boundingBox[3] = xymax[1];  // ymax
    boundingBox[4] = xymin[2];  // zmin
    boundingBox[5] = xymax[2];  // zmax
    
    // Get minimum radius (characteristic mesh size)
    DMPlexGetMinRadius(dm, &minRadius);
    
    // Set boundary layer thickness as a multiple of minRadius (e.g., 3-5 cell layers)
    boundaryLayerThickness = boundaryLayerMultiplier * minRadius;
    
    PetscInt cStart, cEnd;
    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
    
    // Compute boundary distance and weight for each cell
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        PetscReal centroid[3];
        DMPlexComputeCellGeometryFVM(dm, cell, nullptr, centroid, nullptr);
        
        // Compute minimum distance to any boundary
        PetscReal minDistToBoundary = PETSC_INFINITY;
        
        // Check distance to each boundary face
        for (int d = 0; d < dim; ++d) {
            // Distance to lower boundary
            PetscReal distToLower = centroid[d] - boundingBox[2*d];
            if (distToLower < minDistToBoundary) {
                minDistToBoundary = distToLower;
            }
            
            // Distance to upper boundary
            PetscReal distToUpper = boundingBox[2*d + 1] - centroid[d];
            if (distToUpper < minDistToBoundary) {
                minDistToBoundary = distToUpper;
            }
        }
        
        cellBoundaryDistances[cell] = minDistToBoundary;
        
        // Compute boundary weight: 1.0 for interior, 0.0 for boundary (binary)
        PetscReal weight = (minDistToBoundary >= boundaryLayerThickness) ? 1.0 : 0.0;
        cellBoundaryWeights[cell] = weight;
    }
    
    PetscPrintf(PETSC_COMM_WORLD, "[IntSharp] Binary boundary damping: thickness = %g (%.1f * minRadius)\n", 
                boundaryLayerThickness, boundaryLayerThickness / minRadius);
    PetscPrintf(PETSC_COMM_WORLD, "[IntSharp] Bounding box: [%g, %g] x [%g, %g] x [%g, %g]\n",
                boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3], boundingBox[4], boundingBox[5]);
}

PetscReal ablate::finiteVolume::processes::IntSharp::GetBoundaryWeight(PetscInt cell) const {
    auto it = cellBoundaryWeights.find(cell);
    if (it != cellBoundaryWeights.end()) {
        return it->second;
    }
    return 1.0;  // Default to interior weight if cell not found
}

void ablate::finiteVolume::processes::IntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    auto dim = flow.GetSubDomain().GetDimensions();
    auto dm = flow.GetSubDomain().GetDM();
    PetscFE fe_coords;
    PetscInt k = 1;

    DMClone(dm, &vertexDM) >> utilities::PetscUtilities::checkError;
    PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, PETSC_TRUE, k, PETSC_DETERMINE, &fe_coords) >> utilities::PetscUtilities::checkError;
    DMSetField(vertexDM, 0, nullptr, (PetscObject)fe_coords) >> utilities::PetscUtilities::checkError;
    PetscFEDestroy(&fe_coords) >> utilities::PetscUtilities::checkError;
    DMCreateDS(vertexDM) >> utilities::PetscUtilities::checkError;

    // Compute boundary information
    ComputeBoundaryInformation(dm);

    ablate::domain::Range cellRange; 
    auto fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver*>(&flow);

    if (!fvSolver) {
      return;
    }

    // BJ slope limiter on EULER is the "limiter that interferes with conserved
    // variables" the user warned about: it caused SEGV in viscous-flux
    // integration at mu>=0.06.  Limiting the EULER (rho, rhoE, rhoU) reconstruct
    // before the viscous flux was inconsistent with how PETSc's DMPlexFV
    // expects the gradient to be computed, dereferencing past the section.
    // We retain limiting on alpha/rhoAlpha (where IS *needs* monotonicity for
    // its update to be physically meaningful) but skip EULER -- the FV solver
    // already has its own slope limiting for EULER via the flux calculator.
    if (Gamma > 0.0 && !disableBJSlopeLimiter) {
        fvSolver->EnableSlopeLimiterFor(VOLUME_FRACTION_FIELD);
        fvSolver->EnableSlopeLimiterFor(DENSITY_VF_FIELD);
        PetscPrintf(PETSC_COMM_WORLD,
                    "[IntSharp::Setup] BJ slope limiter (MUSCL) enabled for %s, %s "
                    "(EULER limiter skipped to avoid conflict with viscous flux)\n",
                    VOLUME_FRACTION_FIELD.c_str(), DENSITY_VF_FIELD.c_str());
    } else if (Gamma > 0.0 && disableBJSlopeLimiter) {
        PetscPrintf(PETSC_COMM_WORLD,
                    "[IntSharp::Setup] Gamma=%.3g but BJ slope limiter explicitly DISABLED via disableBJSlopeLimiter=true "
                    "(PM sharpening term still active; diagnostic isolation mode)\n", (double)Gamma);
    } else {
        PetscPrintf(PETSC_COMM_WORLD,
                    "[IntSharp::Setup] Gamma=0: limiter NOT enabled (diffuse-equivalent mode)\n");
    }

    PetscReal h;
    DMPlexGetMinRadius(dm, &h);

    PetscInt cStart, cEnd; 
    DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
    for (PetscInt i = cStart; i < cEnd; ++i) {

        PetscInt cell = cellRange.GetPoint(i);
        PetscInt nNeighbors, *neighbors;
        PetscReal layers=3;
        DMPlexGetNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);
        cellNeighbors[cell] = std::vector<PetscInt>(neighbors, neighbors + nNeighbors);
        //corresponding to each of the neighbors, get the weight of the neighbor which is calculated via (inverse distance) /(total weight) such that the sum of the weights is 1
        cellWeights[cell] = std::vector<PetscReal>(nNeighbors, 0);

        for (PetscInt j = 0; j < nNeighbors; ++j) {
            //define a centroid for the cell and store it via DMPlexComputeCellGeometryFVM
            PetscReal ccentroid[3];
            DMPlexComputeCellGeometryFVM(dm, cell, nullptr, ccentroid, nullptr);
            //define a neighbor centroid and store it in the same way
            PetscReal ncentroid[3];
            DMPlexComputeCellGeometryFVM(dm, neighbors[j], nullptr, ncentroid, nullptr);
            PetscReal d = std::sqrt(PetscSqr(ncentroid[0] - ccentroid[0]) + PetscSqr(ncentroid[1] - ccentroid[1]) + PetscSqr(ncentroid[2] - ccentroid[2]));
            
            if (d > 1e-10) {
                //h is the getminradius
                cellWeights[cell][j] = PetscExpReal( -d*d/ (2*PetscSqr(2 * h)) );
            }
            else {
                cellWeights[cell][j] = 1.0;
            }
        }

        PetscReal totalWeight = std::accumulate(cellWeights[cell].begin(), cellWeights[cell].end(), 0.0);
        for (PetscInt j = 0; j < nNeighbors; ++j) {
            cellWeights[cell][j] /= totalWeight;
        }

        //for the last neighbor, set the weight such that the sum of the weights is exactly 1
        // if (nNeighbors > 0) {
        //     cellWeights[cell][nNeighbors - 1] = 1.0 - std::accumulate(cellWeights[cell].begin(), cellWeights[cell].end() - 1, 0.0);
        // }
        DMPlexRestoreNeighbors(dm, cell, layers, 0, 0, PETSC_FALSE, PETSC_FALSE, &nNeighbors, &neighbors);

    }
    
    PetscInt vStart, vEnd;
    DMPlexGetDepthStratum(vertexDM, 0, &vStart, &vEnd);
    for (PetscInt vertex = vStart; vertex < vEnd; ++vertex) {
        PetscInt nvn, *vertexneighbors;
        DMPlexVertexGetCells(dm, vertex, &nvn, &vertexneighbors);
        vertexNeighbors[vertex] = std::vector<PetscInt>(vertexneighbors, vertexneighbors + nvn);
        DMPlexVertexRestoreCells(dm, vertex, &nvn, &vertexneighbors);
    }

    if (Gamma > 0.0) {
        auto intSharpPreStage = std::bind(intSharpPreStageWrapper, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, this);
        flow.RegisterPreStage(intSharpPreStage);
    } else {
        PetscPrintf(PETSC_COMM_WORLD,
                    "[IntSharp::Setup] Gamma=0: PreStage NOT registered (process is inert; equivalent to no IntSharp in deck)\n");
    }
}

PetscErrorCode ablate::finiteVolume::processes::IntSharp::PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime) {
    PetscFunctionBegin;

    const auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);
    ablate::domain::Range cellRange; 
    fvSolver.GetCellRangeWithoutGhost(cellRange);
    PetscInt dim; 
    PetscCall(DMGetDimension(fvSolver.GetSubDomain().GetDM(), &dim));
    DM dm = fvSolver.GetSubDomain().GetDM();
    Vec globFlowVec; 
    PetscCall(TSGetSolution(flowTs, &globFlowVec));
    PetscScalar *flowArray; 
    PetscCall(VecGetArray(globFlowVec, &flowArray));
    Vec locFVec; PetscCall(DMGetLocalVector(dm, &locFVec)); 
    PetscCall(VecZeroEntries(locFVec));

    const auto &eulerOffset = fvSolver.GetSubDomain().GetField(CompressibleFlowFields::EULER_FIELD).offset;
    const auto &vfOffset = fvSolver.GetSubDomain().GetField(VOLUME_FRACTION_FIELD).offset;
    const auto &rhoAlphaOffset = fvSolver.GetSubDomain().GetField(DENSITY_VF_FIELD).offset;
    PetscInt uOff[3]; uOff[0] = vfOffset; uOff[1] = rhoAlphaOffset; uOff[2] = eulerOffset;

    Vec locX = solver.GetSubDomain().GetSolutionVector(); 
    ablate::finiteVolume::processes::IntSharp *process = this;

    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
    subDomain->UpdateAuxLocalVector();

    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector(); //LOCAL aux vector, not global

    Vec vertexVec; DMGetLocalVector(process->vertexDM, &vertexVec);
    const PetscScalar *solArray; 
    VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError; //solution (cell centered) variables rho, rhoe, rhou
    PetscScalar *auxArray; 
    VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError; //aux (cell centered) variables phi, p, T, etc
    PetscScalar *vertexArray; 
    VecGetArray(vertexVec, &vertexArray); //vertex based info
    PetscScalar *fArray; 
    PetscCall(VecGetArray(locFVec, &fArray)); //rhs vector
    PetscInt vStart, vEnd; 
    DMPlexGetDepthStratum(process->vertexDM, 0, &vStart, &vEnd);
    PetscInt cStart, cEnd; 
    DMPlexGetHeightStratum(auxDM, 0, &cStart, &cEnd);

    //get the volumeFraction field
    // const auto &phiField = subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD);
    const auto &phiField = subDomain->GetField("volumeFraction");
    //get the euler field in the same manner
    // const auto &gasDensityField = subDomain->GetField("gasDensity");
    // const auto &densityVFField = subDomain->GetField("densityvolumeFraction");
    //make a debug field
    const auto &ofield = subDomain->GetField("debug1");
    const auto &gasDensityField = subDomain->GetField("gasDensity");
    const auto &liquidDensityField = subDomain->GetField("liquidDensity");
    const auto &gasEnergyField = subDomain->GetField("gasEnergy");
    const auto &liquidEnergyField = subDomain->GetField("liquidEnergy");

    // --- Determinism: Print checksum of solution vector at start ---
    PetscReal solNormStart = 0.0;
    VecNorm(locX, NORM_2, &solNormStart);
    PetscPrintf(PETSC_COMM_WORLD, "[IntSharp::PreStage] Start solution norm: %g\n", solNormStart);

    // Zero locFVec and vertexVec before use (already done)
    PetscCall(VecZeroEntries(locFVec));
    PetscCall(VecZeroEntries(vertexVec));

    // Synchronize ghost cells for input vectors (if needed)
    DMGlobalToLocalBegin(dm, globFlowVec, INSERT_VALUES, locX);
    DMGlobalToLocalEnd(dm, globFlowVec, INSERT_VALUES, locX);
    DMGlobalToLocalBegin(auxDM, auxVec, INSERT_VALUES, auxVec);
    DMGlobalToLocalEnd(auxDM, auxVec, INSERT_VALUES, auxVec);

    // Get the field offset and number of components for debug1 and debug2
    PetscInt offset = ofield.offset;
    PetscInt nComp = ofield.numberComponents;
    const auto &phitildeField = subDomain->GetField("debug2");
    PetscInt phitildeOffset = phitildeField.offset;
    PetscInt phitildeNComp = phitildeField.numberComponents;

    // Zero just the debug1 and debug2 fields for all cells
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        PetscScalar* auxCell = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, ofield.id, auxArray, &auxCell);
        if (auxCell) {
            for (PetscInt c = 0; c < nComp; ++c) {
                auxCell[offset + c] = 0.0;
            }
        }
        PetscScalar* phitildeCell = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, phitildeField.id, auxArray, &phitildeCell);
        if (phitildeCell) {
            for (PetscInt c = 0; c < phitildeNComp; ++c) {
                phitildeCell[phitildeOffset + c] = 0.0;
            }
        }
    }

    // --- Compute phitilde (smoothed volume fraction) for all cells and store in debug2 field ---
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        PetscReal sum = 0.0;
        const auto& neighbors = cellNeighbors[cell];
        const auto& weights = cellWeights[cell];
        for (std::size_t j = 0; j < neighbors.size(); ++j) {
            PetscInt neighbor = neighbors[j];
            PetscReal weight = weights[j];
            // Get phi value for neighbor
            const PetscScalar* neighborPhi = nullptr;
            xDMPlexPointLocalRead(dm, neighbor, phiField.id, solArray, &neighborPhi);
            sum += weight * (neighborPhi ? *neighborPhi : 0.0);
        }
        // Store phitilde in debug2 field
        PetscScalar* phitildeCell = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, phitildeField.id, auxArray, &phitildeCell);
        *phitildeCell = sum;
    }

    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
        const PetscScalar *phic; 
        xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
        PetscScalar *fsharp; 
        xDMPlexPointLocalRef(auxDM, cell, ofield.id, auxArray, &fsharp);
        PetscScalar *phitildeCell = nullptr;
        xDMPlexPointLocalRef(auxDM, cell, phitildeField.id, auxArray, &phitildeCell);
        PetscScalar *allFields = nullptr; 
        DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;
        // PetscReal phitilde = *phitildeCell;

        // Only process interfacial cells based on phitilde; set fsharp=0 and skip otherwise
        if (*phic <= 1e-3 || *phic >= 1.0 - 1e-3) {
            *fsharp = 0.0;
            continue;
        }

        //get the magnitude of the gradient of the phitilde field using DMPlexCellGradFromCell
        PetscScalar gradphic[3] = {0.0, 0.0, 0.0};
        // DMPlexCellGradFromCell(auxDM, cell, auxVec, phitildeField.id, 0, gradphic);
        //take grad from phic field
        DMPlexCellGradFromCell(dm, cell, locX, phiField.id, 0, gradphic);
        // DMPlexCellGradFromCell(dm, cell, vertexVec, -1, 0, gradphic);
        PetscReal normgradphi = 0.0;
        for (int k = 0; k < dim; ++k) {
            normgradphi += PetscSqr(gradphic[k]);
        }
        normgradphi = PetscSqrtReal(normgradphi);

        // Compute fsharp for interfacial cells using phitilde
        // *fsharp = process->Gamma * ( (-1 * phitilde) * (1 - phitilde) * (1 - 2 * phitilde) + process->epsilon * (1 - 2 * phitilde) * normgradphi );
        //get it from phic instead of phitilde
        *fsharp = process->Gamma * ( (-1 * *phic) * (1 - *phic) * (1 - 2 * *phic) + process->epsilon * (1 - 2 * *phic) * normgradphi );
        
        // Apply boundary weight to fsharp (dampen near boundaries)
        PetscReal boundaryWeight = process->GetBoundaryWeight(cell);
        *fsharp *= boundaryWeight;


        // *rhophiSource += rhog * *fsharp;

        const PetscReal rho_old  = allFields[ablate::finiteVolume::CompressibleFlowFields::RHO];
        const PetscReal rhoE_old = allFields[ablate::finiteVolume::CompressibleFlowFields::RHOE];
        PetscReal velocity[3] = {0.0, 0.0, 0.0};
        for (PetscInt d = 0; d < dim; d++) {
            velocity[d] = allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / rho_old;
        }
        PetscReal v2_old = 0.0;
        for (PetscInt d = 0; d < dim; ++d) v2_old += velocity[d] * velocity[d];
        const PetscReal e_old = (rhoE_old - 0.5 * rho_old * v2_old) / rho_old;
        (void)e_old;  // retained for traceability; new energy update uses per-phase eG/eL.

        PetscReal pseudoTime = 1e-3;
        PetscReal *densityG, *densityL, *eG, *eL;
        xDMPlexPointLocalRead(auxDM, cell, gasDensityField.id, auxArray, &densityG) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, cell, liquidDensityField.id, auxArray, &densityL) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, cell, gasEnergyField.id, auxArray, &eG) >> utilities::PetscUtilities::checkError;
        xDMPlexPointLocalRead(auxDM, cell, liquidEnergyField.id, auxArray, &eL) >> utilities::PetscUtilities::checkError;
        // (void)eG; (void)eL; -- now used in the energy update below

        // GUARD: aux fields (densityG, densityL) get populated by the EOS during the
        // first ComputeSource call; on the FIRST PreStage call they are still zero.
        // Skip the alpha update on uninitialised cells (drho_per_dalpha ~ 0); the
        // physics will catch up on the next stage once aux is populated.
        const PetscReal drho_per_dalpha = (*densityG) - (*densityL);
        const bool auxInitialised = (PetscAbsReal(drho_per_dalpha) > 1e-10);
        if (process->Gamma > 0.0 && *phic > 1e-3 && *phic < 1.0 - 1e-3 && auxInitialised) {
            //  1) Advance alpha by pseudoTime * fsharp, clamp to [0,1].
            //  2) [optional, alphaOnly=false] Rebuild rho, rhoAlpha, rhoE, rhoU
            //     directly from the mixture formulas using the current per-phase
            //     aux densities/energies (Marziale-style; matches
            //     intSharp-marziale.cpp:591-598).  CompressibleFlowFields::RHOE
            //     is TOTAL energy (internal + KE), so we add 0.5*rho*|v|^2.
            //
            // alphaOnly=true (default): skip the rho/rhoE rebuild.  Per the
            // user's history with stable IS+SF runs, ONLY alpha was updated.
            // The full mixture rebuild introduces step-discontinuous EOS state
            // changes that amplify at cusps -- empirically this drove
            // |v|->63 m/s and NaN at right cusp tip cell 2839 in
            // star2d_isf_g1em2 within 1.5e-4 s.
            PetscReal alpha_new = allFields[vfOffset] + pseudoTime * (*fsharp);
            if (alpha_new < 0.0)      alpha_new = 0.0;
            else if (alpha_new > 1.0) alpha_new = 1.0;
            allFields[vfOffset] = alpha_new;

            if (!process->alphaOnly) {
                const PetscReal rhoAlpha_new = (*densityG) * alpha_new;
                const PetscReal rho_new      = alpha_new * (*densityG) + (1.0 - alpha_new) * (*densityL);
                const PetscReal rhoEint_new  = rhoAlpha_new * (*eG) + (rho_new - rhoAlpha_new) * (*eL);
                allFields[rhoAlphaOffset] = rhoAlpha_new;
                allFields[ablate::finiteVolume::CompressibleFlowFields::RHO]  = rho_new;
                allFields[ablate::finiteVolume::CompressibleFlowFields::RHOE] = rhoEint_new + 0.5 * rho_new * v2_old;
                for (PetscInt d = 0; d < dim; ++d) {
                    allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = rho_new * velocity[d];
                }
            } else {
                // alphaOnly: also update rhoAlpha so it stays consistent with
                // the new alpha (using the *current* gas density).  Without
                // this, rhoAlpha drifts from rho*alpha and the FV-advected
                // densityvolumeFraction goes incoherent.
                allFields[rhoAlphaOffset] = (*densityG) * alpha_new;
            }
        }
    }

    // --- LAPLACE-YOUNG TEST MODE: bulk-phase pinning ---
    // (See header for full justification.  Analogous to ZalesakTest's
    // hardcode of momentum/energy in twoPhaseEulerAdvection.)
    // OFF in production -- gated entirely on `laplaceYoungTest` flag.
    if (process->laplaceYoungTest) {
        // Per-stage tally: how many cells got pinned.
        PetscInt nPinnedGas = 0, nPinnedLiq = 0, nSkippedAux = 0, nSkippedNoPhic = 0;
        const PetscReal gasThreshold    = 0.99;  // alpha >= this -> pure gas
        const PetscReal liquidThreshold = 0.01;  // alpha <= this -> pure liquid
        // Iterate over the same cell range as the IS update loop above
        // (height stratum 0 -- all owned cells, including boundary).
        // The interior-only cellRange from solver.GetRange iterates a
        // different (boundary-cell-only) IS in this BoxMeshBoundaryCells
        // setup; using the wider [cStart, cEnd) range matches the IS
        // update and the diagnostic scan.
        for (PetscInt cell = cStart; cell < cEnd; ++cell) {
            const PetscScalar *phic = nullptr;
            xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &phic);
            if (!phic || !std::isfinite(*phic)) {
                ++nSkippedNoPhic;
                continue;
            }
            // Only pin pure-phase cells; interface cells (already updated above) are untouched.
            const bool isPureGas    = (*phic >= gasThreshold);
            const bool isPureLiquid = (*phic <= liquidThreshold);
            if (!isPureGas && !isPureLiquid) continue;

            PetscScalar *allFields = nullptr;
            DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;
            if (!allFields) continue;

            const PetscReal *densityG = nullptr, *densityL = nullptr, *eG = nullptr, *eL = nullptr;
            xDMPlexPointLocalRead(auxDM, cell, gasDensityField.id,    auxArray, &densityG);
            xDMPlexPointLocalRead(auxDM, cell, liquidDensityField.id, auxArray, &densityL);
            xDMPlexPointLocalRead(auxDM, cell, gasEnergyField.id,     auxArray, &eG);
            xDMPlexPointLocalRead(auxDM, cell, liquidEnergyField.id,  auxArray, &eL);
            // Aux fields populate only after first ComputeSource (decoder).
            // Until then, we don't know rho_G, rho_L, eG, eL -- skip to avoid
            // overwriting solution with zeros.
            if (!densityG || !densityL || !eG || !eL ||
                *densityG < 0.1 || *densityL < 0.1) {
                ++nSkippedAux;
                continue;
            }

            // PINNING POLICY (gas-only):
            //   * Pure-gas (alpha >= 0.99): full ambient state pin.  Per
            //     dissertation Sec. 4.2.1, gas-side dynamics is not part
            //     of the LY validation -- it just needs to provide a
            //     constant background.  Pinning rho=rho_G, v=0,
            //     rhoE=rho_G*eG eliminates gas-side acoustic transients
            //     that otherwise propagate into the interface band and
            //     produce NaN.
            //   * Pure-liquid (alpha <= 0.01): UNTOUCHED.  Bulk liquid
            //     must pressurize via EOS as the SF body force at the
            //     interface propagates pressure information inward; this
            //     IS the LY pressure jump we are trying to measure.
            if (!isPureGas) continue;
            // Need fresh aux fields for ambient values (rho_G, eG).
            if (!densityG || !eG || !std::isfinite(*densityG) || !std::isfinite(*eG) || *densityG < 0.1) {
                ++nSkippedAux;
                continue;
            }
            allFields[ablate::finiteVolume::CompressibleFlowFields::RHO]  = *densityG;
            allFields[ablate::finiteVolume::CompressibleFlowFields::RHOE] = (*densityG) * (*eG);
            for (PetscInt d = 0; d < dim; ++d) {
                allFields[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = 0.0;
            }
            // rhoAlpha = rho_G * alpha (alpha untouched)
            allFields[rhoAlphaOffset] = (*densityG) * (*phic);
            ++nPinnedGas;
            (void)densityL; (void)eL;  // liquid aux not needed
        }
        PetscPrintf(PETSC_COMM_WORLD,
                    "[IS::laplaceYoungTest] pinned cells: gas=%d liquid=%d skipped(aux uninit)=%d skipped(no phic)=%d\n",
                    (int)nPinnedGas, (int)nPinnedLiq, (int)nSkippedAux, (int)nSkippedNoPhic);
    }

    // Synchronize ghost cells for output vectors (if needed)
    DMLocalToGlobalBegin(dm, locFVec, ADD_VALUES, globFlowVec);
    DMLocalToGlobalEnd(dm, locFVec, ADD_VALUES, globFlowVec);
    DMLocalToGlobalBegin(auxDM, auxVec, INSERT_VALUES, auxVec);
    DMLocalToGlobalEnd(auxDM, auxVec, INSERT_VALUES, auxVec);

    // --- Determinism: Print checksum of solution vector at end ---
    PetscReal solNormEnd = 0.0;
    VecNorm(locX, NORM_2, &solNormEnd);
    PetscPrintf(PETSC_COMM_WORLD, "[IntSharp::PreStage] End solution norm: %g\n", solNormEnd);

    // --- Verbose per-stage diagnostics: scan all cells for extrema and pathological values ---
    // (Heavy, but invaluable for diagnosing where IS+SF runaway/cusp events happen.
    //  Once a cell goes rho<0.1 or alpha out of bounds, NaN/SEGV is moments away.)
    {
        PetscReal rhoMin =  PETSC_INFINITY, rhoMax = -PETSC_INFINITY;
        PetscReal rhoEMin = PETSC_INFINITY, rhoEMax = -PETSC_INFINITY;
        PetscReal alphaMin = PETSC_INFINITY, alphaMax = -PETSC_INFINITY;
        PetscReal vMag2Max = 0.0;
        PetscReal pMin = PETSC_INFINITY, pMax = -PETSC_INFINITY;
        PetscInt  nNanCons = 0, nRare = 0, nAlphaBad = 0, nNanAux = 0;
        PetscInt  worstRhoCell = -1, worstVelCell = -1, worstAlphaCell = -1;
        PetscReal worstRhoVal = PETSC_INFINITY, worstVelVal = 0.0, worstAlphaDist = 0.0;

        const auto rhoOff   = ablate::finiteVolume::CompressibleFlowFields::RHO;
        const auto rhoEOff  = ablate::finiteVolume::CompressibleFlowFields::RHOE;
        const auto rhoUOff  = ablate::finiteVolume::CompressibleFlowFields::RHOU;
        const auto &pAuxField = subDomain->GetField("pressure");
        // Restrict the diagnostic scan to cells in the "interiorCells" label.
        // Corner ghost cells (BoxMeshBoundaryCells leaves them in NO label) do
        // NOT have the volumeFraction field allocated; reading volumeFraction
        // there returns aliased memory (looks like rho_air=1.16 in the alpha
        // slot), spamming false-positive WARNs on every step.
        DMLabel interiorLabel = nullptr;
        DMGetLabel(dm, "interiorCells", &interiorLabel);

        for (PetscInt cell = cStart; cell < cEnd; ++cell) {
            if (interiorLabel) {
                PetscInt isInterior = -1;
                DMLabelGetValue(interiorLabel, cell, &isInterior);
                if (isInterior < 0) continue;  // skip non-interior (boundary/corner) cells
            }
            const PetscScalar *q = nullptr;
            xDMPlexPointLocalRead(dm, cell, 0 /* euler field id, =0 in this YAML */, solArray, &q);
            if (!q) continue;
            PetscReal rho  = q[rhoOff];
            PetscReal rhoE = q[rhoEOff];
            PetscReal rhoU = q[rhoUOff], rhoV = q[rhoUOff + 1];
            if (!std::isfinite(rho) || !std::isfinite(rhoE) || !std::isfinite(rhoU) || !std::isfinite(rhoV)) {
                ++nNanCons;
                continue;
            }
            if (rho < rhoMin)  { rhoMin = rho;  if (rho < worstRhoVal) { worstRhoVal = rho; worstRhoCell = cell; } }
            if (rho > rhoMax)  rhoMax = rho;
            if (rhoE < rhoEMin) rhoEMin = rhoE;
            if (rhoE > rhoEMax) rhoEMax = rhoE;
            if (rho > 0) {
                PetscReal u = rhoU/rho, v = rhoV/rho;
                PetscReal vmag2 = u*u + v*v;
                if (vmag2 > vMag2Max) { vMag2Max = vmag2; worstVelCell = cell; worstVelVal = PetscSqrtReal(vmag2); }
                if (rho < 0.1) ++nRare;
            }

            const PetscScalar *qVF = nullptr;
            xDMPlexPointLocalRead(dm, cell, phiField.id, solArray, &qVF);
            if (qVF && std::isfinite(qVF[0])) {
                PetscReal a = qVF[0];
                if (a < alphaMin) alphaMin = a;
                if (a > alphaMax) alphaMax = a;
                PetscReal d = (a < 0.0) ? -a : (a > 1.0 ? a - 1.0 : 0.0);
                if (d > 0.0) { ++nAlphaBad; if (d > worstAlphaDist) { worstAlphaDist = d; worstAlphaCell = cell; } }
            }
            const PetscScalar *pp = nullptr;
            xDMPlexPointLocalRead(auxDM, cell, pAuxField.id, auxArray, &pp);
            if (pp && std::isfinite(pp[0])) {
                if (pp[0] < pMin) pMin = pp[0];
                if (pp[0] > pMax) pMax = pp[0];
            } else {
                ++nNanAux;
            }
        }

        PetscReal vmax = PetscSqrtReal(vMag2Max);
        // Get worst-cell centroid for actionable diagnostics.
        PetscReal worstRhoCent[3] = {0,0,0}, worstVelCent[3] = {0,0,0}, worstAlphaCent[3] = {0,0,0};
        if (worstRhoCell   >= 0) DMPlexComputeCellGeometryFVM(dm, worstRhoCell,   nullptr, worstRhoCent,   nullptr);
        if (worstVelCell   >= 0) DMPlexComputeCellGeometryFVM(dm, worstVelCell,   nullptr, worstVelCent,   nullptr);
        if (worstAlphaCell >= 0) DMPlexComputeCellGeometryFVM(dm, worstAlphaCell, nullptr, worstAlphaCent, nullptr);

        PetscPrintf(PETSC_COMM_WORLD,
                    "[IS::diag]  rho [%.3f, %.3f]  rhoE [%.3e, %.3e]  alpha [%.4f, %.4f]  |v|max=%.2f  p [%.2e, %.2e]\n",
                    (double)rhoMin, (double)rhoMax, (double)rhoEMin, (double)rhoEMax,
                    (double)alphaMin, (double)alphaMax, (double)vmax, (double)pMin, (double)pMax);

        if (nNanCons + nRare + nAlphaBad + nNanAux > 0) {
            PetscPrintf(PETSC_COMM_WORLD,
                        "[IS::diag]  WARN  NaN_conservatives=%d  rho<0.1=%d  alpha out [0,1]=%d  NaN_pressure=%d\n",
                        (int)nNanCons, (int)nRare, (int)nAlphaBad, (int)nNanAux);
            if (worstRhoCell >= 0) {
                PetscPrintf(PETSC_COMM_WORLD, "[IS::diag]    worst rho cell=%d @ (%.4e, %.4e) rho=%.4e\n",
                            (int)worstRhoCell, (double)worstRhoCent[0], (double)worstRhoCent[1], (double)worstRhoVal);
            }
            if (worstVelCell >= 0 && vmax > 50.0) {
                PetscPrintf(PETSC_COMM_WORLD, "[IS::diag]    worst |v| cell=%d @ (%.4e, %.4e) |v|=%.2f\n",
                            (int)worstVelCell, (double)worstVelCent[0], (double)worstVelCent[1], (double)worstVelVal);
            }
            if (worstAlphaCell >= 0) {
                PetscPrintf(PETSC_COMM_WORLD, "[IS::diag]    worst alpha cell=%d @ (%.4e, %.4e) dist=%.4e\n",
                            (int)worstAlphaCell, (double)worstAlphaCent[0], (double)worstAlphaCent[1], (double)worstAlphaDist);
            }
        }
    }

    // Count cells with boundary damping (binary: 0 = damped, 1 = not damped)
    PetscInt boundaryCells = 0;
    for (const auto& pair : cellBoundaryWeights) {
        if (pair.second == 0.0) {
            boundaryCells++;
        }
    }
    if (boundaryCells > 0) {
        PetscPrintf(PETSC_COMM_WORLD, "[IntSharp::PreStage] Applied binary boundary damping to %d cells (fsharp = 0)\n", boundaryCells);
    }

    PetscCall(VecRestoreArray(globFlowVec, &flowArray)); 
    VecRestoreArrayRead(locX, &solArray);
    VecRestoreArray(auxVec, &auxArray);
    VecRestoreArray(vertexVec, &vertexArray);
    VecRestoreArray(locFVec, &fArray);
    solver.RestoreRange(cellRange);

    DMRestoreLocalVector(process->vertexDM, &vertexVec);
    VecDestroy(&vertexVec); 

    PetscFunctionReturn(0);
}



REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::IntSharp, "calculates interface regularization term",
         ARG(PetscReal, "Gamma", "Gamma, velocity scale parameter (approx. umax)"),
         ARG(PetscReal, "epsilon", "epsilon, interface thickness scale parameter (approx. h)"),
         ARG(bool, "flipPhiTilde", "if true: phiTilde-->1-phiTilde (set it to true if primary phase is phi=0 or false if phi=1)"),
         OPT(PetscReal, "boundaryLayerMultiplier", "multiplier for boundary layer thickness (default: 3.0)"),
         OPT(bool, "alphaOnly", "if true (default): IS::PreStage updates ONLY the volumeFraction (alpha) slot + rhoAlpha for consistency. "
                                "if false: also rebuild rho, rhoE, rhoU from per-phase EOS aux fields (Marziale-style, more invasive, "
                                "less stable at cusps with strong SF)"),
         OPT(bool, "laplaceYoungTest", "if true: enable Laplace-Young test mode (analogous to ZalesakTest in twoPhaseEulerAdvection). "
                                       "Bulk-phase cells outside the interface band are pinned each stage: pure-gas (alpha>=0.99) -> "
                                       "ambient gas state (rho=rho_G, v=0, rhoE=rho_G*eG); pure-liquid (alpha<=0.01) -> v=0 only "
                                       "(rho/rhoE preserved so LY pressure jump can develop). Interface band (0.01<alpha<0.99) "
                                       "is untouched.  OFF by default; never affects production runs."),
         OPT(bool, "disableBJSlopeLimiter", "if true: even when Gamma>0, force-DISABLE the BJ slope limiter on volumeFraction and "
                                            "densityvolumeFraction (PM sharpening term still active).  Default false preserves "
                                            "legacy behavior (BJ ON whenever Gamma>0).  Used for diagnostic isolation: separates "
                                            "PM-term instability from BJ-on-vof-field instability when Gamma>0.")
);
