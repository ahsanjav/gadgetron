#pragma once

#include "Node.h"
#include "hoNDArray.h"

#include "mri_core_acquisition_bucket.h"
#include <complex>
#include <ismrmrd/ismrmrd.h>
#include <map>

namespace Gadgetron {

    class AcquisitionSpiralAccumulateWaveform
        : public Core::ChannelGadget<Core::variant<Core::Acquisition, Core::Waveform>> {
    public:
        using Core::ChannelGadget<Core::variant<Core::Acquisition, Core::Waveform>>::ChannelGadget;
        void process(Core::InputChannel<Core::variant<Core::Acquisition, Core::Waveform>>& in,
            Core::OutputChannel& out) override;
    private:
        hoNDArray<float> trajectory_and_weights;
        
        void prepare_trajectory_from_waveforms(const Core::Waveform &grad_waveform);
        
        void send_data(Core::OutputChannel& out, std::map<unsigned short, AcquisitionBucket>& buckets,
                       std::vector<Core::Waveform>& waveforms);
    };

}

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