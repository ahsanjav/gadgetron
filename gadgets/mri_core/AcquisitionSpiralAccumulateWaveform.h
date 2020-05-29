#pragma once

#include "Node.h"
#include "hoNDArray.h"
#include "mri_core_acquisition_bucket.h"
#include <complex>
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/xml.h>
#include "gadgetron_mricore_export.h"
#include "mri_core_data.h"
#include <fstream>
#include <vector>
#include "armadillo"
#include "mri_core_utility.h"
#include <mri_core_girf_correction.h>

namespace Gadgetron
{

    class AcquisitionSpiralAccumulateWaveform
        : public Core::ChannelGadget<Core::variant<Core::Acquisition, Core::Waveform>>
    {
    public:
        using Core::ChannelGadget<Core::variant<Core::Acquisition, Core::Waveform>>::ChannelGadget;
        AcquisitionSpiralAccumulateWaveform(const Core::Context &context, const Core::GadgetProperties &props);

        enum class SortingDimension {
            kspace_encode_step_1,
            kspace_encode_step_2,
            average,
            slice,
            contrast,
            phase,
            repetition,
            set,
            segment,
            user_0,
            user_1,
            user_2,
            user_3,
            user_4,
            user_5,
            user_6,
            user_7,
            n_acquisitions,
            none
        };
        
        hoNDArray<std::complex<float>> girf_kernel;
        float girf_sampletime;
        int girf_numpoint;
        float newscaling=0.5; 
        float newscaling1=0.5;

    protected:
        ISMRMRD::IsmrmrdHeader header;
        NODE_PROPERTY(sorting_dimension, SortingDimension, "Dimension to Sort on", SortingDimension::none);

        int curr_avg = 0;
        float kspace_scaling = 0;
        hoNDArray<float> prepare_trajectory_from_waveforms(const Core::Waveform &grad_waveform_x, const Core::Waveform &grad_waveform_y,
                                                           const ISMRMRD::AcquisitionHeader &head);
        
        hoNDArray<floatd2> sincInterpolation(const hoNDArray<floatd2> input, int zpadFactor);
        hoNDArray<floatd2> zeroHoldInterpolation(const hoNDArray<floatd2> input, int zpadFactor);

        void send_data(Core::OutputChannel &out, std::map<unsigned short, AcquisitionBucket> &buckets,
                       std::vector<Core::Waveform> &waveforms);
        
        void process(Core::InputChannel<Core::variant<Core::Acquisition, Core::Waveform>> &in,
                     Core::OutputChannel &out) override;

        void readGIRFKernel();
        void printGradtoFile(std::string fname_grad, hoNDArray<floatd2> grad_traj);
        void printTrajtoFile(std::string fname_grad, hoNDArray<float> grad_traj);

        hoNDArray<float> calculate_weights_Hoge(const hoNDArray<floatd2> &gradients, const hoNDArray<float> &trajectories);
    };

    void from_string(const std::string& str, AcquisitionSpiralAccumulateWaveform::SortingDimension& val);

} // namespace Gadgetron

// #ifndef SpiralToGenericWaveformGadget_H
// #define SpiralToGenericWaveformGadget_H
// #pragma once

// #include "gadgetron_spiral_export.h"
// #include "Gadget.h"
// #include "GadgetMRIHeaders.h"
// #include "hoNDArray.h"
// #include "TrajectoryParameters.h"

// #include <ismrmrd/ismrmrd.h>
// #include <complex>
// #include <boost/shared_ptr.hpp>
// #include <ismrmrd/xml.h>
// #include <boost/optional.hpp>

// namespace Gadgetron {

//     class EXPORTGADGETS_SPIRAL SpiralToGenericWaveformGadget :
//             public Gadget2<ISMRMRD::AcquisitionHeader, ISMRMRD::WaveformHeader> {

//     public:
//         GADGET_DECLARE(SpiralToGenericWaveformGadget);

//         SpiralToGenericWaveformGadget();

//         virtual ~SpiralToGenericWaveformGadget();

//     protected:

//         virtual int process_config(ACE_Message_Block *mb);

//         virtual int process(GadgetContainerMessage<ISMRMRD::AcquisitionHeader> *m1,
//                             GadgetContainerMessage<ISMRMRD::WaveformHeader> *m2);

//     private:

//         bool prepared_;

//         int samples_to_skip_start_;
//         int samples_to_skip_end_;
//         hoNDArray<float> trajectory_and_weights;
//         Spiral::TrajectoryParameters trajectory_parameters;
//         void prepare_trajectory(const ISMRMRD::AcquisitionHeader &acq_header);

//     };
// }
// #endif //SpiralToGenericWaveformGadget_H