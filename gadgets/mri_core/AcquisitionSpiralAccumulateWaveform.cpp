#include "AcquisitionSpiralAccumulateWaveform.h"
#include "log.h"
#include "mri_core_data.h"
#include <boost/algorithm/string.hpp>

namespace Gadgetron {
    namespace {
            bool is_noise(Core::Acquisition& acq) {
                return std::get<ISMRMRD::AcquisitionHeader>(acq).isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT);
            }
    }
    void AcquisitionSpiralAccumulateWaveform::send_data(Core::OutputChannel& out, std::map<unsigned short, AcquisitionBucket>& buckets,
                                                       std::vector<Core::Waveform>& waveforms) {
        //trigger_events++;
        //GDEBUG("Trigger (%d) occurred, sending out %d buckets\n", trigger_events, buckets.size());
        buckets.begin()->second.waveform_ = std::move(waveforms);
        // Pass all buckets down the chain
        for (auto& bucket : buckets)
            out.push(std::move(bucket.second));

        buckets.clear();
    }
    void AcquisitionSpiralAccumulateWaveform ::process(
        Core::InputChannel<Core::variant<Core::Acquisition, Core::Waveform>>& in, Core::OutputChannel& out) {

        auto waveforms = std::vector<Core::Waveform>{};
        auto grad_waveforms = std::vector<Core::Waveform>{};
        auto buckets   = std::map<unsigned short, AcquisitionBucket>{};
       // auto trigger   = get_trigger(*this);
        int counterData;
        for (auto message : in) {
            if (Core::holds_alternative<Core::Waveform>(message)) {
                auto& temp_waveform = Core::get<Core::Waveform>(message);
                auto whead = std::get<ISMRMRD::WaveformHeader>(Core::get<Core::Waveform>(message));
                
                if(whead.waveform_id<10)
                  waveforms.emplace_back(std::move(Core::get<Core::Waveform>(message)));
                else
                  grad_waveforms.emplace_back(std::move(Core::get<Core::Waveform>(message)));
                
                continue;
            }
          if(~Core::holds_alternative<Core::Waveform>(message))
          {
            auto& acq = Core::get<Core::Acquisition>(message);
           if (is_noise(acq))
               continue;
            auto& head = std::get<ISMRMRD::AcquisitionHeader>(acq);
            auto data = std::get<hoNDArray<std::complex<float>>>(acq);

            // Prepare Trajectory for each acq and push the bucked through 
             head.trajectory_dimensions=3; 
            hoNDArray<float> *trajectory_and_weights = new hoNDArray<float>(head.trajectory_dimensions,head.number_of_samples);
           

            //auto traj = std::get<hoNDArray<uint32_t>>(acq);
            //GadgetContainerMessage<hoNDArray<float> > *cont = new GadgetContainerMessage<hoNDArray<float> >();
            Core::Acquisition acq2 = Core::Acquisition(std::move(head),std::move(data),std::move(trajectory_and_weights));
               
            
      //      acq = t(std::move(head),std::move(data),std::move(traj));
          //  *(&acq)->_M_head
                  
            counterData++;
         //   head.idx.kspace_encode_step_1;
          }

    //        if (trigger_before(trigger, head))
     //           send_data(out, buckets, waveforms);
            // It is enough to put the first one, since they are linked
     //       unsigned short sorting_index = get_index(head, sorting_dimension);

    //        AcquisitionBucket& bucket = buckets[sorting_index];
      //      add_acquisition(bucket, std::move(acq));

        //    if (trigger_after(trigger, head))
          //      send_data(out, buckets, waveforms);
        //}
        
    }
      send_data(out,buckets,waveforms);
  }
  void AcquisitionSpiralAccumulateWaveform::prepare_trajectory_from_waveforms(const Core::Waveform &grad_waveform){
    
    //auto& wave_head = std::get<ISMRMRD::WaveformHeader>(grad_waveform);
    //auto& wave_data = std::get<hoNDArray<uint32_t>>(grad_waveform);    
    auto& [wave_head,wave_data] = grad_waveform;
    hoNDArray<u_int32_t> wav = hoNDArray<float>(wave_data);
    
    //wav = wave_data.get_data_ptr();
  }
    GADGETRON_GADGET_EXPORT(AcquisitionSpiralAccumulateWaveform);

}
