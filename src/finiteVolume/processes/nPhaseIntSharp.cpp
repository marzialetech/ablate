#include "nPhaseIntSharp.hpp"
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include "eos/kthStiffenedGas.hpp"
#include "eos/nPhase.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "nPhaseAllaireAdvection.hpp"
#include "utilities/constants.hpp"
#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"
#include <petsc/private/dmpleximpl.h>

namespace ablate::finiteVolume::processes {

    void ablate::finiteVolume::processes::NPhaseIntSharp::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
        NPhaseIntSharp::subDomain = solver.GetSubDomainPtr();
    }

    ablate::finiteVolume::processes::NPhaseIntSharp::Form
    ablate::finiteVolume::processes::NPhaseIntSharp::ParseForm(const std::string &raw) {
        // Accept any case + ignore separators (- _ space) so users can write
        // "chiu_lin", "Chiu-Lin", "ChiuLin", etc.
        std::string s;
        s.reserve(raw.size());
        for (char c : raw) {
            if (c == '_' || c == '-' || c == ' ') continue;
            s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        if (s.empty() || s == "parameswaranmandal" || s == "pm" || s == "default") {
            return Form::PARAMESWARAN_MANDAL;
        }
        if (s == "chiulin" || s == "cl" || s == "conservative" || s == "div") {
            return Form::CHIU_LIN;
        }
        throw std::invalid_argument("NPhaseIntSharp: unknown form '" + raw +
                                    "' (expected 'parameswaran_mandal' or 'chiu_lin')");
    }

    ablate::finiteVolume::processes::NPhaseIntSharp::NPhaseIntSharp(const std::vector<PetscReal>& Gammak, const std::vector<PetscReal>& epsilonk, const std::vector<PetscInt>& flipPhiTildek, PetscReal boundaryLayerMultiplier, std::string formIn) : Gammak(Gammak), epsilonk(epsilonk), flipPhiTildek(flipPhiTildek), form(ParseForm(formIn)), boundaryLayerMultiplier(boundaryLayerMultiplier) {
        // Initialize boundary layer thickness as a multiple of minRadius (will be set in Setup)
        boundaryLayerThickness = 0.0;
        minRadius = 0.0;
        for (int i = 0; i < 6; ++i) {
            boundingBox[i] = 0.0;
        }
        PetscPrintf(MPI_COMM_WORLD,
                    "[NPhaseIntSharp] form = %s\n",
                    form == Form::CHIU_LIN ? "chiu_lin (conservative divergence)" : "parameswaran_mandal (cell-centered scalar)");
    }

    ablate::finiteVolume::processes::NPhaseIntSharp::~NPhaseIntSharp() {
        DMDestroy(&vertexDM) >> utilities::PetscUtilities::checkError;
        if (fluxDM) {
            DMDestroy(&fluxDM) >> utilities::PetscUtilities::checkError;
        }
    }

    void ablate::finiteVolume::processes::NPhaseIntSharp::EnsureFluxDM(DM dm, PetscInt phases) {
        if (fluxDM) return;

        PetscInt dim;
        DMGetDimension(dm, &dim) >> utilities::PetscUtilities::checkError;

        // Clone the topology of the simulation DM and attach a fresh local section
        // with dim*phases dofs per cell. The clone shares topology + coordinate DM
        // with the parent, so the per-face geometry queries used by
        // DMPlexCellGradFromCell continue to work.
        DMClone(dm, &fluxDM) >> utilities::PetscUtilities::checkError;

        DM coordDM = nullptr;
        DMGetCoordinateDM(dm, &coordDM) >> utilities::PetscUtilities::checkError;
        DMSetCoordinateDM(fluxDM, coordDM) >> utilities::PetscUtilities::checkError;

        PetscInt cStart, cEnd;
        DMPlexGetHeightStratum(fluxDM, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;

        PetscSection section;
        PetscSectionCreate(PetscObjectComm((PetscObject)fluxDM), &section) >> utilities::PetscUtilities::checkError;
        PetscSectionSetChart(section, cStart, cEnd) >> utilities::PetscUtilities::checkError;
        const PetscInt dofPerCell = dim * phases;
        for (PetscInt c = cStart; c < cEnd; ++c) {
            PetscSectionSetDof(section, c, dofPerCell) >> utilities::PetscUtilities::checkError;
        }
        PetscSectionSetUp(section) >> utilities::PetscUtilities::checkError;
        DMSetLocalSection(fluxDM, section) >> utilities::PetscUtilities::checkError;
        PetscSectionDestroy(&section) >> utilities::PetscUtilities::checkError;
        DMSetUp(fluxDM) >> utilities::PetscUtilities::checkError;

        // Trigger geometry caches (also populates DMPlexGetMinRadius for fluxDM).
        Vec cellGeom = nullptr, faceGeom = nullptr;
        DMPlexComputeGeometryFVM(fluxDM, &cellGeom, &faceGeom) >> utilities::PetscUtilities::checkError;
        if (cellGeom) VecDestroy(&cellGeom) >> utilities::PetscUtilities::checkError;
        if (faceGeom) VecDestroy(&faceGeom) >> utilities::PetscUtilities::checkError;
    }

    void ablate::finiteVolume::processes::NPhaseIntSharp::ComputeBoundaryInformation(DM dm) {
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

        // Ghost cells (created by PETSc for BC stencils) appear in the height stratum but have
        // no real geometry; calling DMPlexComputeCellGeometryFVM on them errors. Skip them via
        // the "ghost" DMLabel which DMPlex sets when ghost cells are created.
        DMLabel ghostLabel = nullptr;
        DMGetLabel(dm, "ghost", &ghostLabel);

        // Compute boundary distance and weight for each cell
        for (PetscInt cell = cStart; cell < cEnd; ++cell) {
            if (ghostLabel) {
                PetscInt ghostVal = -1;
                DMLabelGetValue(ghostLabel, cell, &ghostVal);
                if (ghostVal >= 0) {
                    continue;
                }
            }
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
    }

    PetscReal ablate::finiteVolume::processes::NPhaseIntSharp::GetBoundaryWeight(PetscInt cell) const {
        auto it = cellBoundaryWeights.find(cell);
        if (it != cellBoundaryWeights.end()) {
            return it->second;
        }
        // Cells absent from the map were excluded by ComputeBoundaryInformation
        // (e.g. ghost cells with no real geometry). Treat them as boundary-like so
        // the PreStage loop skips them via the existing weight < 0.5 check, instead
        // of trying to compute geometry on cells that have no cell type assigned.
        return 0.0;
    }

    void nPhaseIntSharpPreStageWrapper(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime, ablate::finiteVolume::processes::NPhaseIntSharp* nPhaseIntSharpProcess) {
        nPhaseIntSharpProcess->PreStage(flowTs, solver, stagetime);
    }



    void NPhaseIntSharp::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Starting Setup\n");

        NPhaseIntSharp::subDomain = flow.GetSubDomainPtr();
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Got subDomain\n");

        auto dim = flow.GetSubDomain().GetDimensions();
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Got dimensions: %d\n", dim);
        
        auto dm = flow.GetSubDomain().GetDM();
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Got DM\n");
        
        PetscFE fe_coords;
        PetscInt k = 1;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to clone DM\n");
        DMClone(dm, &vertexDM) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] DM cloned successfully\n");
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to create FE\n");
        PetscFECreateLagrange(PETSC_COMM_SELF, dim, dim, PETSC_TRUE, k, PETSC_DETERMINE, &fe_coords) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] FE created\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to set field\n");
        DMSetField(vertexDM, 0, nullptr, (PetscObject)fe_coords) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Field set\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to destroy FE\n");
        PetscFEDestroy(&fe_coords) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] FE destroyed\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to create DS\n");
        DMCreateDS(vertexDM) >> utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] DS created\n");

        // Compute boundary information
        ComputeBoundaryInformation(dm);

                //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to get cell range\n");
        ablate::domain::Range cellRange; 
        auto fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver*>(&flow);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Dynamic cast done\n");

        if (!fvSolver) {
          //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] fvSolver cast failed, returning\n");
          return;
        }
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] fvSolver cast successful\n");

        // Single user-facing toggle: presence of NPhaseIntSharp in the YAML
        // processes block opts the volume-fraction / per-phase-mass fields
        // into BJ slope limiting (MUSCL face reconstruction). Without this
        // call the SlopeLimiter zeroes the gradient on those fields and the
        // downstream face reconstruction is donor-cell. CellInterpolant is
        // built lazily inside FiniteVolumeSolver::ComputeRHSFunction; the
        // FV solver caches these requests and replays them at construction.
        fvSolver->EnableSlopeLimiterFor(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
        fvSolver->EnableSlopeLimiterFor(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);
        PetscPrintf(PETSC_COMM_WORLD,
                    "[NPhaseIntSharp::Setup] BJ slope limiter (MUSCL) enabled for %s and %s\n",
                    ablate::finiteVolume::NPhaseFlowFields::ALPHAK.c_str(),
                    ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK.c_str());

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to get height stratum\n");
        PetscInt cStart, cEnd; DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Height stratum: %d to %d\n", cStart, cEnd);
        cellRange.start = cStart; cellRange.end = cEnd;

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to get depth stratum\n");
        PetscInt vStart, vEnd;
        DMPlexGetDepthStratum(vertexDM, 0, &vStart, &vEnd);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] Depth stratum: %d to %d\n", vStart, vEnd);

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] About to register PreStage\n");
        // flow.RegisterRHSFunction(ComputeTerm, this);
        auto nPhaseIntSharpPreStage = std::bind(nPhaseIntSharpPreStageWrapper, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, this);
        flow.RegisterPreStage(nPhaseIntSharpPreStage);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::Setup] PreStage registered successfully\n");
        
    }

    PetscErrorCode ablate::finiteVolume::processes::NPhaseIntSharp::PreStage(TS flowTs, ablate::solver::Solver &solver, PetscReal stagetime) {
        PetscFunctionBegin;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Starting PreStage\n");

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to do dynamic cast\n");
        const auto &fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver &>(solver);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Dynamic cast successful\n");
        
        ablate::domain::Range cellRange; 
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get cell range\n");
        fvSolver.GetCellRangeWithoutGhost(cellRange);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell range: %d to %d\n", cellRange.start, cellRange.end);
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get dimension\n");
        PetscInt dim; 
        PetscCall(DMGetDimension(fvSolver.GetSubDomain().GetDM(), &dim));
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got dimension: %d\n", dim);
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get DM\n");
        DM dm = fvSolver.GetSubDomain().GetDM();
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got DM\n");
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get solution vector\n");
        Vec globFlowVec; 
        PetscCall(TSGetSolution(flowTs, &globFlowVec));
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got solution vector\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get flow array\n");
        PetscScalar *flowArray; 
        PetscCall(VecGetArray(globFlowVec, &flowArray));
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got flow array\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get local vector\n");
        Vec locFVec; PetscCall(DMGetLocalVector(dm, &locFVec)); 
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got local vector\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to zero entries\n");
        PetscCall(VecZeroEntries(locFVec));
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Zeroed entries\n");

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get solution vector from solver\n");
        Vec locX = solver.GetSubDomain().GetSolutionVector(); 
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got solution vector from solver\n");
        
        ablate::finiteVolume::processes::NPhaseIntSharp *process = this;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got process pointer\n");

        std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got subDomain\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get aux DM\n");
        DM auxDM = subDomain->GetAuxDM();
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got aux DM\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get aux vector\n");
        Vec auxVec = subDomain->GetAuxVector();
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got aux vector\n");
        
        // Note: cell iteration in this routine uses cellRange (acquired above via
        // GetCellRangeWithoutGhost), not raw [cStart, cEnd) from DMPlexGetHeightStratum.
        // The latter would include FVM ghost cells which have no cell type assigned and
        // would crash any DMPlexComputeCellGeometryFVM call below.
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get vertex vector\n");
        Vec vertexVec; 
        DMGetLocalVector(process->vertexDM, &vertexVec);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got vertex vector\n");

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get arrays\n");
        const PetscScalar *solArray;
        PetscScalar *auxArray;
        PetscScalar *vertexArray;
        PetscScalar *fArray;

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get sol array\n");
        VecGetArrayRead(locX, &solArray) >> ablate::utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got sol array\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get aux array\n");
        VecGetArray(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got aux array\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get vertex array\n");
        VecGetArray(vertexVec, &vertexArray);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got vertex array\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get f array\n");
        VecGetArray(locFVec, &fArray);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got f array\n");

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get fields\n");
        // ablate::domain::Range cellRange;
        // solver.GetCellRangeWithoutGhost(cellRange);

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get alphak field\n");
        const auto &alphakField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAK);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got alphak field\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get alphakrhok field\n");
        const auto &alphakrhokField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::ALPHAKRHOK);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got alphakrhok field\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get allaire field\n");
        // const auto &allaireField = solver.GetSubDomain().GetField(ablate::finiteVolume::NPhaseFlowFields::ALLAIRE_FIELD);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got allaire field\n");
        
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get fsharpk field\n");
        const auto &fsharpkField = subDomain->GetField(ablate::finiteVolume::NPhaseFlowFields::FSHARPK);
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Got fsharpk field\n");

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get phases\n");
        // const auto alphakOffset = alphakField.offset;
        // const auto alphakrhokOffset = alphakrhokField.offset;
        // const auto allaireOffset = allaireField.offset;
        // const auto fsharpkOffset = fsharpkField.offset;

        std::size_t phases = alphakField.numberComponents;
        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Number of phases: %zu\n", phases);

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to get min radius\n");
        // PetscReal h;
        // DMPlexGetMinRadius(auxDM, &h);
        // //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Min radius: %f\n", h);

        // -----------------------------------------------------------------------------
        // CHIU_LIN: face-based finite-volume divergence of the antidiffusive flux.
        //
        // For each phase k the conservative form is:
        //     d(alpha_k)/d(tau) = div(F_k),
        //     F_k = Gamma_k * [ epsilon_k * grad(alpha_k)
        //                       - alpha_k(1 - alpha_k) * grad(alpha_k) / |grad(alpha_k)| ].
        //
        // The previous implementation computed F_k at cell centers using ablate's
        // least-squares cell-gradient operator and then took the divergence using the
        // SAME LSQ operator on F_k. That stencil is not TVD-friendly: at sharp
        // alpha interfaces it produced oscillatory cell-to-cell gradient values
        // (aliasing) which the divergence operator amplified into checkerboard fsharp
        // patterns inside the disk interior. Visually we observed phase-swap speckle
        // in the disk cores at any meaningful Gamma; the result was structurally wrong
        // regardless of Gamma magnitude.
        //
        // This new implementation evaluates F_k . n at each MESH FACE using
        //     dalpha_dn = (alpha_R - alpha_L) / |L->R|              (FD, monotone)
        //     alpha_f   = 0.5 * (alpha_L + alpha_R)
        //     sign_g    = sign(alpha_R - alpha_L)
        //     F . n     = Gamma * eps * dalpha_dn  -  Gamma * af*(1-af) * sign_g
        // and integrates around each cell:
        //     fsharp_c = (1/V_c) * Sum_{faces of c} (F . n_out) * area.
        // This is a Green-Gauss face-flux divergence: conservative by construction,
        // bounded (the antidiffusive coefficient is alpha*(1-alpha) <= 1/4), and free
        // of the LSQ aliasing because the normal gradient at each face is just a
        // 2-point FD between adjacent cell centers.
        //
        // boundaryLayerMultiplier still gates the result -- the cell-level
        // GetBoundaryWeight check inside the projection cell loop below zeroes any
        // fsharp values written into the boundary band, mirroring PM behavior.
        // -----------------------------------------------------------------------------
        if (form == Form::CHIU_LIN) {
            // Zero fsharpk for every cell that has local storage (real + ghost). The
            // projection loop only reads real cells via cellRange.
            for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
                const PetscInt cell = cellRange.GetPoint(c);
                PetscScalar *fsharpkCell = nullptr;
                xDMPlexPointLocalRef(auxDM, cell, fsharpkField.id, auxArray, &fsharpkCell);
                if (fsharpkCell) {
                    for (std::size_t k = 0; k < phases; ++k) fsharpkCell[k] = 0.0;
                }
            }

            PetscInt fStart = 0, fEnd = 0;
            DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;

            DMLabel ghostLabel = nullptr;
            DMGetLabel(dm, "ghost", &ghostLabel) >> utilities::PetscUtilities::checkError;

            for (PetscInt face = fStart; face < fEnd; ++face) {
                if (ghostLabel) {
                    PetscInt gv = -1;
                    DMLabelGetValue(ghostLabel, face, &gv) >> utilities::PetscUtilities::checkError;
                    if (gv > 0) continue;
                }

                PetscInt supSize = 0;
                const PetscInt *support = nullptr;
                DMPlexGetSupportSize(dm, face, &supSize) >> utilities::PetscUtilities::checkError;
                DMPlexGetSupport(dm, face, &support) >> utilities::PetscUtilities::checkError;
                if (supSize != 2) continue; // physical boundary face: zero-flux Neumann
                const PetscInt L = support[0];
                const PetscInt R = support[1];

                PetscReal centroidL[3] = {0.0, 0.0, 0.0};
                PetscReal centroidR[3] = {0.0, 0.0, 0.0};
                PetscReal volL = 0.0;
                PetscReal volR = 0.0;
                DMPlexComputeCellGeometryFVM(dm, L, &volL, centroidL, nullptr) >> utilities::PetscUtilities::checkError;
                DMPlexComputeCellGeometryFVM(dm, R, &volR, centroidR, nullptr) >> utilities::PetscUtilities::checkError;
                if (volL <= 0.0 || volR <= 0.0) continue;

                PetscReal areaFace = 0.0;
                PetscReal centroidF[3] = {0.0, 0.0, 0.0};
                PetscReal normalF[3]   = {0.0, 0.0, 0.0};
                DMPlexComputeCellGeometryFVM(dm, face, &areaFace, centroidF, normalF) >> utilities::PetscUtilities::checkError;

                // L->R distance (used as the FD denominator for the normal gradient).
                PetscReal dn = 0.0;
                for (PetscInt d = 0; d < dim; ++d) {
                    PetscReal dxd = centroidR[d] - centroidL[d];
                    dn += dxd * dxd;
                }
                dn = PetscSqrtReal(dn);
                if (dn <= ablate::utilities::Constants::tiny) continue;

                const PetscScalar *alphakL = nullptr;
                const PetscScalar *alphakR = nullptr;
                xDMPlexPointLocalRead(dm, L, alphakField.id, solArray, &alphakL);
                xDMPlexPointLocalRead(dm, R, alphakField.id, solArray, &alphakR);
                if (!alphakL || !alphakR) continue;

                PetscScalar *fsharpL = nullptr;
                PetscScalar *fsharpR = nullptr;
                xDMPlexPointLocalRef(auxDM, L, fsharpkField.id, auxArray, &fsharpL);
                xDMPlexPointLocalRef(auxDM, R, fsharpkField.id, auxArray, &fsharpR);

                // Boundary-band check at L and R (skip writing into a cell that the
                // boundary mask wants to hold fixed).
                const PetscReal wL = process->GetBoundaryWeight(L);
                const PetscReal wR = process->GetBoundaryWeight(R);

                for (std::size_t k = 0; k < phases; ++k) {
                    const PetscReal aL = alphakL[k];
                    const PetscReal aR = alphakR[k];

                    // No interface across this face: both sides are deep in the same
                    // extreme. Skip any flux; this also zeros the antidiffusive driver
                    // before it can grow on machine-precision noise inside a phase core.
                    if ((aL <= 1e-3 && aR <= 1e-3) || (aL >= 1.0 - 1e-3 && aR >= 1.0 - 1e-3)) {
                        continue;
                    }

                    const PetscReal af  = 0.5 * (aL + aR);
                    const PetscReal dad = (aR - aL) / dn;             // normal gradient (FD)
                    const PetscReal sg  = (aR > aL) ? 1.0 : (aR < aL ? -1.0 : 0.0);

                    PetscReal aftilde = af;
                    if (process->flipPhiTildek[k] == 1) aftilde = 1.0 - af;

                    const PetscReal Gk = process->Gammak[k];
                    const PetscReal Ek = process->epsilonk[k];

                    // F . n_LR = Gamma*eps * (dalpha/dn)  -  Gamma * af*(1-af) * sign(dalpha/dn)
                    const PetscReal F_dot_n = Gk * Ek * dad - Gk * aftilde * (1.0 - aftilde) * sg;
                    const PetscReal flux_LR = F_dot_n * areaFace;     // signed mass flux across face

                    // div(F)_c = (1/V_c) * Sum (F . n_out) * area. n_out at L is L->R, so
                    // contribution at L is +flux_LR/V_L. n_out at R is the opposite, so
                    // contribution at R is -flux_LR/V_R.
                    if (fsharpL && wL >= 0.5) fsharpL[k] += flux_LR / volL;
                    if (fsharpR && wR >= 0.5) fsharpR[k] -= flux_LR / volR;
                }
            }
        }

        // Iterate real cells via cellRange.GetPoint(c). cellRange came from
        // GetCellRangeWithoutGhost above, which excludes FVM ghost cells. This is the
        // ablate convention used by nPhaseAllaireAdvection / chemistry / navierStokesTransport
        // / etc.; it makes the loop correct in 2D, 3D, and parallel without any
        // raw-stratum + ghost-label band-aid.
        for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
            const PetscInt cell = cellRange.GetPoint(c);

            // Boundary-layer guard: skip cells inside the analytical-boundary buffer.
            // ComputeBoundaryInformation already excluded any ghost cells from the weight
            // map, and GetBoundaryWeight returns 0.0 for unmapped cells, so a stray ghost
            // making it here would also be skipped via this same path.
            PetscReal boundaryWeight = process->GetBoundaryWeight(cell);
            if (boundaryWeight < 0.5) continue;

            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: About to get old values\n", cell);
            //keep old values
            std::vector<PetscReal> alphakold(phases);
            std::vector<PetscReal> alphakrhokold(phases);
            PetscReal rhoold = 0.0;
            std::vector<PetscReal> uiold(dim);
            std::vector<PetscReal> rhokold(phases);
            
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: About to get allFields\n", cell);
            PetscScalar *allFields = nullptr;
            DMPlexPointLocalRef(dm, cell, flowArray, &allFields) >> utilities::PetscUtilities::checkError;
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Got allFields\n", cell);
            
            //coordinates of this cell:
            PetscReal centroid[3];
            DMPlexComputeCellGeometryFVM(dm, cell, nullptr, centroid, nullptr);
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Centroid: %g, %g, %g\n", cell, centroid[0], centroid[1], centroid[2]);
            
            // Get field pointers ONCE outside the loop
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: About to get field pointers\n", cell);
            const PetscScalar *alphakFieldPtr;
            const PetscScalar *alphakrhokFieldPtr;
            xDMPlexPointLocalRead(dm, cell, alphakField.id, solArray, &alphakFieldPtr);
            xDMPlexPointLocalRead(dm, cell, alphakrhokField.id, solArray, &alphakrhokFieldPtr);
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Got field pointers\n", cell);
            
            // Now access the components correctly
            for (std::size_t k = 0; k < phases; ++k) {
                alphakold[k] = alphakFieldPtr[k];
                alphakrhokold[k] = alphakrhokFieldPtr[k];
                rhoold += alphakrhokFieldPtr[k];
                
                // Avoid division by zero
                if (alphakFieldPtr[k] > PETSC_SMALL) {
                    rhokold[k] = alphakrhokFieldPtr[k] / alphakFieldPtr[k];
                } else {
                    rhokold[k] = 0.0;
                    //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Warning - alphak[%zu] = %g (very small)\n", cell, k, alphakFieldPtr[k]);
                }
            }
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: About to get uiold\n", cell);
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Current uiold: %g, %g, %g\n", cell, uiold[0], uiold[1], uiold[2]);
            if (rhoold > PETSC_SMALL) {
                // PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Current rhoold: %g\n", cell, rhoold);
            }
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Current alphakrhok for all k: ");
            for (std::size_t k = 0; k < phases; ++k) {
                //PetscPrintf(MPI_COMM_WORLD, "%g ", alphakrhokFieldPtr[k]);
            }
            //PetscPrintf(MPI_COMM_WORLD, "\n");
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Current allFields: %g, %g, %g\n", cell, allFields[ablate::finiteVolume::NPhaseFlowFields::RHOU], allFields[ablate::finiteVolume::NPhaseFlowFields::RHOU + 1], allFields[ablate::finiteVolume::NPhaseFlowFields::RHOU + 2]);
            
            // Check for zero density to avoid division by zero
            if (rhoold < PETSC_SMALL) {
                //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: Zero density detected, skipping cell\n", cell);
                continue;  // Skip this cell entirely
            }
            
            for (PetscInt d = 0; d < dim; ++d) {
                uiold[d] = allFields[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] / rhoold;
            }

            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: About to compute fsharpk\n", cell);

            //compute fsharpk for all k
            for (std::size_t k = 0; k < phases; ++k) {
                // Use the field pointers we already got, or get them if needed
                PetscScalar *fsharpk;
                xDMPlexPointLocalRef(auxDM, cell, fsharpkField.id, auxArray, &fsharpk);
                
                if (alphakold[k] <= 1e-3 || alphakold[k] >= 1.0 - 1e-3) {
                    fsharpk[k] = 0.0;
                    continue;
                }

                PetscReal Gammak = process->Gammak[k];
                PetscReal epsilonk = process->epsilonk[k];

                PetscReal alphaktilde = alphakold[k];  // Use the stored old value
                if (process->flipPhiTildek[k] == 1) {
                    alphaktilde = 1 - alphakold[k];
                }

                if (form == Form::PARAMESWARAN_MANDAL) {
                    PetscScalar gradalphak[3];
                    DMPlexCellGradFromCell(auxDM, cell, auxVec, alphakField.id, k, gradalphak);

                    PetscReal normgradalphak = 0.0;
                    for (PetscInt d = 0; d < dim; ++d) {
                        normgradalphak += PetscSqr(gradalphak[d]);
                    }
                    normgradalphak = PetscSqrtReal(normgradalphak);

                    fsharpk[k] = Gammak * ( (-1 * alphaktilde) * (1 - alphaktilde) * (1 - 2 * alphaktilde) + epsilonk * (1 - 2 * alphaktilde) * normgradalphak );
                } else {
                    // CHIU_LIN: fsharpk[k] was already computed in the face-based pass
                    // before this cell loop. Leave it as-is; the gate above (extreme
                    // alphak -> fsharp = 0) still applies via the early-continue earlier
                    // in this k-loop.
                }
                // Boundary weight already checked at start of cell loop - no need to multiply here
                // PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: fsharpk[%zu] = %g\n", cell, k, fsharpk[k]);
            }
            //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] Cell %d: fsharpk computed\n", cell);

            // Sharpening pseudo-time update applied as a thermodynamically-consistent
            // PROJECTION step (mirrors intSharp-marziale.cpp:591-599 for the n-phase case).
            //
            // The previous version applied fsharp to alpha, clipped per-phase to [0,1], and
            // rebuilt alphakrhok = alpha * rhokold + RHOU = rho * uiold -- but it left out
            // (a) the partition-of-unity renormalization that the per-phase clip breaks,
            // (b) the RHOE rebalance that keeps specific internal energy fixed when rho
            //     drifts, and
            // (c) a fallback rhok for cells where intsharp pushes alpha into a phase that
            //     was previously absent (rhokold = 0 there, so mass would silently vanish).
            // For n-identical-EOS test cases, leaving any of these out drives a slow but
            // unbounded pressure leak through the EOS coupling alpha -> alphakrhok ->
            // internal energy -> pressure (observed empirically: chiu_lin Γ=8e-3 NaN'd
            // at step 350, Γ=5e-3 at extrapolated ~step 1500). Fixing all three makes the
            // sharpening update an internal-energy-preserving phase-indicator regularization.

            // Cache specific internal energy and squared velocity from the pre-update
            // state; these are the invariants we want to preserve across the projection.
            PetscReal v2_old = 0.0;
            for (PetscInt d = 0; d < dim; ++d) v2_old += uiold[d] * uiold[d];
            PetscReal rhoe_old = allFields[ablate::finiteVolume::NPhaseFlowFields::RHOE];
            PetscReal e_old = (rhoe_old - 0.5 * rhoold * v2_old) / rhoold;

            // Default per-phase density for cells where alphak[k] was 0 at entry: use the
            // mixture density rhoold. For n-identical-EOS this is exactly correct (every
            // rhok is identically rhoold). For non-identical phases this is the best
            // local proxy we have; the EOS will correct on the next NPhaseAllaireAdvection
            // PreStage anyway.
            PetscReal rhok_fallback = rhoold;

            // Apply sharpening source per-phase, then clip to [0,1].
            for (std::size_t k = 0; k < phases; ++k) {
                PetscScalar *fsharpk;
                xDMPlexPointLocalRef(auxDM, cell, fsharpkField.id, auxArray, &fsharpk);
                allFields[alphakField.offset + k] += fsharpk[k];
                if (allFields[alphakField.offset + k] < 0.0)      allFields[alphakField.offset + k] = 0.0;
                else if (allFields[alphakField.offset + k] > 1.0) allFields[alphakField.offset + k] = 1.0;
            }

            // Renormalize so sum_k alphak = 1 (the per-phase clip above generally breaks
            // partition-of-unity by O(eps); without this rebalance, the broken sum drifts
            // total density which drives the pressure leak).
            PetscReal alphasum = 0.0;
            for (std::size_t k = 0; k < phases; ++k) alphasum += allFields[alphakField.offset + k];
            if (alphasum > PETSC_SMALL) {
                PetscReal invsum = 1.0 / alphasum;
                for (std::size_t k = 0; k < phases; ++k) allFields[alphakField.offset + k] *= invsum;
            }

            // Rebuild alphakrhok using cached per-phase density (with fallback for newly
            // appearing phases) and accumulate the new mixture density.
            PetscReal rho = 0.0;
            for (std::size_t k = 0; k < phases; ++k) {
                PetscReal rhok_k = (rhokold[k] > PETSC_SMALL) ? rhokold[k] : rhok_fallback;
                allFields[alphakrhokField.offset + k] = allFields[alphakField.offset + k] * rhok_k;
                rho += allFields[alphakrhokField.offset + k];
            }

            // Preserve velocity (uiold) and specific internal energy (e_old) by
            // rebuilding RHOU and RHOE from the new mixture density. For n-identical-EOS
            // this is exact (rho == rhoold, so RHOU and RHOE return to their pre-fsharp
            // values modulo roundoff). For general n-phase it preserves specific
            // thermodynamic state through the projection.
            for (PetscInt d = 0; d < dim; ++d) {
                allFields[ablate::finiteVolume::NPhaseFlowFields::RHOU + d] = rho * uiold[d];
            }
            allFields[ablate::finiteVolume::NPhaseFlowFields::RHOE] = rho * (e_old + 0.5 * v2_old);
        }

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to restore arrays\n");
        VecRestoreArrayRead(locX, &solArray);
        VecRestoreArray(auxVec, &auxArray);
        VecRestoreArray(vertexVec, &vertexArray);
        VecRestoreArray(locFVec, &fArray);
        solver.RestoreRange(cellRange);

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] About to restore vertex vector\n");
        DMRestoreLocalVector(process->vertexDM, &vertexVec);
        VecDestroy(&vertexVec); 

        //PetscPrintf(MPI_COMM_WORLD, "[NPhaseIntSharp::PreStage] PreStage completed successfully\n");
        PetscFunctionReturn(0);
    }

}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, 
    ablate::finiteVolume::processes::NPhaseIntSharp, 
    "N-phase interface regularization term",
    ARG(std::vector<PetscReal>, "Gammak", "Gamma, velocity scale parameter (approx. umax)"),
    ARG(std::vector<PetscReal>, "epsilonk", "epsilon, interface thickness scale parameter (approx. h)"),
    ARG(std::vector<PetscInt>, "flipPhiTildek", "if 1: phiTilde-->1-phiTilde, if 0: keep phiTilde (set to 1 if primary phase is phi=0 or 0 if phi=1)"),
    ARG(PetscReal, "boundaryLayerMultiplier", "multiplier for boundary layer thickness (default: 3.0)"),
    OPT(std::string, "form", "discrete sharpening form: 'parameswaran_mandal' (default; cell-centered scalar) or 'chiu_lin' (conservative divergence)"));
