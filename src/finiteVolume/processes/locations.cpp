#include "locations.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"

ablate::finiteVolume::processes::locations::locations() {}


void ablate::finiteVolume::processes::locations::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) { flow.RegisterRHSFunction(ComputeSource, this); }

// Called every time the mesh changes
void ablate::finiteVolume::processes::locations::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
  locations::subDomain = solver.GetSubDomainPtr();

}



PetscErrorCode ablate::finiteVolume::processes::locations::ComputeSource(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void *ctx) {
    PetscFunctionBegin;


    auto process = (ablate::finiteVolume::processes::locations *)ctx;
    std::shared_ptr<ablate::domain::SubDomain> subDomain = process->subDomain;

    // const ablate::domain::Field *cellLocs = &(subDomain->GetField("cellLocations"));
    // const ablate::domain::Field *vertLocs = &(subDomain->GetField("vertexLocations"));
    // const ablate::domain::Field *rankLocs = &(subDomain->GetField("rank"));

    ablate::domain::Range cellRange, vertRange;
    solver.GetCellRangeWithoutGhost(cellRange);
    solver.GetRange(0, vertRange);

    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();
    PetscScalar *auxArray = nullptr;

    PetscMPIInt  rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank) >> ablate::utilities::PetscUtilities::checkError;

    if (rank == 0) {
        PetscPrintf(PetscObjectComm((PetscObject)dm),
                    "locations::ComputeSource CALLED at time = %g\n", (double)time);
    }

    VecGetArray(auxVec, &auxArray);




//    DM dmCell;
//    const PetscScalar* cellGeomArray;
//    Vec cellGeomVec = solver.cellGeomVec;
//    VecGetDM(cellGeomVec, &dmCell) >> utilities::PetscUtilities::checkError;
//    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
//    PetscInt dim;
//    DMGetDimension(dmCell, &dim);

const ablate::domain::Field& xposField = subDomain->GetField("xpos");
const ablate::domain::Field& yposField = subDomain->GetField("ypos");

for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
  const PetscInt cell = c;

  // compute centroid
  PetscReal centroid[3];
  DMPlexComputeCellGeometryFVM(dm, cell, NULL, centroid, NULL) >> ablate::utilities::PetscUtilities::checkError;

  // write xpos
  PetscScalar* px;
  xDMPlexPointLocalRef(auxDM, cell, xposField.id, auxArray, &px) >> ablate::utilities::PetscUtilities::checkError;
  px[0] = centroid[0];

  // write ypos
  PetscScalar* py;
  xDMPlexPointLocalRef(auxDM, cell, yposField.id, auxArray, &py) >> ablate::utilities::PetscUtilities::checkError;
  py[0] = centroid[1];

}

// for (PetscInt cell = cellRange.start; cell < cellRange.end; ++cell) {
//   PetscScalar* x;
//   xDMPlexPointLocalRef(auxDM, cell, cellLocs->id, auxArray, &x) >> ablate::utilities::PetscUtilities::checkError;

//   PetscReal centroid[3];
//   DMPlexComputeCellGeometryFVM(dm, cell, NULL, centroid, NULL) >> ablate::utilities::PetscUtilities::checkError;

//   x[0] = centroid[0];
//   x[1] = centroid[1];
//   // dimension is 2 so we ignore centroid[2]
  
//   // rank recording (you had this already)
//   PetscScalar* r;
//   xDMPlexPointLocalRef(auxDM, cell, rankLocs->id, auxArray, &r) >> ablate::utilities::PetscUtilities::checkError;
//   *r = rank;

//   if (rank == 0 && cell < cellRange.start+5) {
//     PetscPrintf(PETSC_COMM_WORLD, 
//         "centroid cell %d = (%g, %g)\n",
//         (int)cell, centroid[0], centroid[1]);
// }
// }

//    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;


    // for (PetscInt v = vertRange.start; v < vertRange.end; ++v){
    //   const PetscInt vert = vertRange.GetPoint(v);

    //   PetscScalar *x;
    //   xDMPlexPointLocalRef(auxDM, vert, vertLocs->id, auxArray, &x) >> ablate::utilities::PetscUtilities::checkError;

    //   DMPlexComputeCellGeometryFVM(dm, vert, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
    // }


    VecRestoreArray(auxVec, &auxArray);

    solver.RestoreRange(cellRange);
    solver.RestoreRange(vertRange);
    PetscFunctionReturn(0);
}

ablate::finiteVolume::processes::locations::~locations() { }


REGISTER_WITHOUT_ARGUMENTS(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::locations, "saves vertex and cell locations");
