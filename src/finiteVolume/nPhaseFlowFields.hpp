#ifndef ABLATELIBRARY_NPHASEFLOWFIELDS_HPP
#define ABLATELIBRARY_NPHASEFLOWFIELDS_HPP

#include <domain/region.hpp>
#include <memory>
#include <string>
#include <vector>
#include "domain/fieldDescriptor.hpp"
#include "eos/eos.hpp"
#include "parameters/mapParameters.hpp"

namespace ablate::finiteVolume {

class NPhaseFlowFields : public domain::FieldDescriptor {
   public:
   //unchanging conserved components
    typedef enum {RHOE, RHOU, RHOV, RHOW} AllaireComponents;

    //! the primary field containing the AllaireComponents
    inline const static std::string ALLAIRE_FIELD = "allaire";

    //! The conserved prefix used for fields that have a conserved and non conserved form
    // inline const static std::string CONSERVED = "density";

    inline const static std::string ALPHAKRHOK = "alphakrhok";
    inline const static std::string ALPHAK = "alphak";

    // alpha_k rho_k for each phase; this might actually belong in nPhaseEulerAdvection ?
    // inline static std::string ALPHAKRHOK(PetscInt phase){
    //     return "alpharho" + std::to_string(phase);
    // } 
    // inline static std::string ALPHAK(PetscInt phase){
    //     return "alpha" + std::to_string(phase);
    // }

    // inline static std::string RHOK(PetscInt phase){
    //     return "rho" + std::to_string(phase);
    // }

    // inline static std::string TK(PetscInt phase){
    //     return "t" + std::to_string(phase);
    // }

    // inline static std::string EPSK(PetscInt phase){
    //     return "eps" + std::to_string(phase);
    // }

    //! some common aux fields
    inline const static std::string UI = "ui";
    inline const static std::string TK = "tk";
    inline const static std::string PRESSURE = "p";
    inline const static std::string RHO = "rho";
    inline const static std::string RHOK = "rhok";
    inline const static std::string EPSILON = "epsilon";
    inline const static std::string EPSILONK = "epsilonk";
    inline const static std::string SOSK = "sosk";
    inline const static std::string AIJ = "aij";
    // inline const static std::string USTAR = "ustar";

    //! per-phase interface-sharpening RHS contribution (NPhaseIntSharp); aux field with one component per phase
    inline const static std::string FSHARPK = "fsharpk";

    //! Volume-fraction floor below which a phase is treated as computationally absent
    //! by the EOS decode and downstream flux/mixing rules. Trace amounts (alpha ~ 1e-10)
    //! produced by numerical advection diffusion would otherwise be amplified by
    //! rhok = (alpha*rhok)/alpha and ek = (p + gamma*pi)/((gamma-1)*rhok + eps), driving
    //! per-phase epsilonk and sosk to spurious large values that eventually NaN through
    //! the Riemann flux. Empirically PETSC_SMALL (1e-20) is too low to suppress this,
    //! while 1e-6 is well above any meaningful interface smearing on the meshes we care
    //! about and orders of magnitude below physically resolved volume fractions.
    inline const static PetscReal ALPHAK_FLOOR = 1e-6;

   protected:
    const std::shared_ptr<eos::EOS> eos;
    const std::shared_ptr<domain::Region> region;
    const std::shared_ptr<parameters::Parameters> conservedFieldOptions;
    const std::shared_ptr<parameters::Parameters> auxFieldOptions = ablate::parameters::MapParameters::Create({
        {"petscfv_type", "leastsquares"}, 
        {"petsclimiter_type", "none"},
        {"petscfv_compute_gradients", "false"}
    });
    const PetscInt dim;

   public:
    /**
     * Create a helper class that produces the required compressible flow fields based upon the eos and specifed region
     * @param eos the eos used to determine the species
     * @param region the region for all of the fields
     * @param conservedFieldParameters override the default field parameters for the conserved field
     */
    explicit NPhaseFlowFields(std::shared_ptr<eos::EOS> eos, 
        std::shared_ptr<domain::Region> region = {}, 
        std::shared_ptr<parameters::Parameters> conservedFieldParameters = {}, 
        PetscInt dimensions = 2);

    /**
     * override and return the compressible flow fields
     * @return
     */
    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;
};

// std::istream& operator>>(std::istream& is, NPhaseFlowFields::ValidRange& v);

}  // namespace ablate::finiteVolume

#endif  // ABLATELIBRARY_NPHASEFLOWFIELDS_HPP
