#include "AcquisitionSpiralAccumulateWaveform.h"

#include "log.h"
#include "mri_core_data.h"
#include "hoNDArray_utils.h"
#include "hoNDArray_math.h"
#include "hoNDFFT.h"
#include <math.h>
#include <stdio.h>
#include "hoNDArray_math.h"
#include "ismrmrd/xml.h"
#include <boost/algorithm/string.hpp>

constexpr double GAMMA = 4258.0;        /* Hz/G */
constexpr double PI = boost::math::constants::pi<double>();
namespace Gadgetron
{
      using SortingDimension = AcquisitionSpiralAccumulateWaveform::SortingDimension;

  AcquisitionSpiralAccumulateWaveform::AcquisitionSpiralAccumulateWaveform(const Core::Context& context, const Core::GadgetProperties& props)
        : ChannelGadget(context, props), header{ context.header } {}

namespace
{
bool is_noise(Core::Acquisition &acq)
{
  return std::get<ISMRMRD::AcquisitionHeader>(acq).isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT);
}
void add_stats(AcquisitionBucketStats &stats, const ISMRMRD::AcquisitionHeader &header)
{
  stats.average.insert(header.idx.average);
  stats.kspace_encode_step_1.insert(header.idx.kspace_encode_step_1);
  stats.kspace_encode_step_2.insert(header.idx.kspace_encode_step_2);
  stats.slice.insert(header.idx.slice);
  stats.contrast.insert(header.idx.contrast);
  stats.phase.insert(header.idx.phase);
  stats.repetition.insert(header.idx.repetition);
  stats.set.insert(header.idx.set);
  stats.segment.insert(header.idx.segment);
}
unsigned short get_index(const ISMRMRD::AcquisitionHeader& header, SortingDimension index) {
            switch (index) {
            case SortingDimension::kspace_encode_step_1: return header.idx.kspace_encode_step_1;
            case SortingDimension::kspace_encode_step_2: return header.idx.kspace_encode_step_2;
            case SortingDimension::average: return header.idx.average;
            case SortingDimension::slice: return header.idx.slice;
            case SortingDimension::contrast: return header.idx.contrast;
            case SortingDimension::phase: return header.idx.phase;
            case SortingDimension::repetition: return header.idx.repetition;
            case SortingDimension::set: return header.idx.set;
            case SortingDimension::segment: return header.idx.segment;
            case SortingDimension::user_0: return header.idx.user[0];
            case SortingDimension::user_1: return header.idx.user[1];
            case SortingDimension::user_2: return header.idx.user[2];
            case SortingDimension::user_3: return header.idx.user[3];
            case SortingDimension::user_4: return header.idx.user[4];
            case SortingDimension::user_5: return header.idx.user[5];
            case SortingDimension::user_6: return header.idx.user[6];
            case SortingDimension::user_7: return header.idx.user[7];
            case SortingDimension::n_acquisitions: return 0;
            case SortingDimension::none: return 0;
            }
            throw std::runtime_error("Illegal enum");
        }

void add_acquisition(AcquisitionBucket &bucket, Core::Acquisition acq)
{
  auto &head = std::get<ISMRMRD::AcquisitionHeader>(acq);
  auto espace = head.encoding_space_ref;

  if (ISMRMRD::FlagBit(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION).isSet(head.flags) || ISMRMRD::FlagBit(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING).isSet(head.flags))
  {
    bucket.ref_.push_back(acq);
    if (bucket.refstats_.size() < (espace + 1))
    {
      bucket.refstats_.resize(espace + 1);
    }
    add_stats(bucket.refstats_[espace], head);
  }
  if (!(ISMRMRD::FlagBit(ISMRMRD::ISMRMRD_ACQ_IS_PARALLEL_CALIBRATION).isSet(head.flags) || ISMRMRD::FlagBit(ISMRMRD::ISMRMRD_ACQ_IS_PHASECORR_DATA).isSet(head.flags)))
  {
    if (bucket.datastats_.size() < (espace + 1))
    {
      bucket.datastats_.resize(espace + 1);
    }
    add_stats(bucket.datastats_[espace], head);
    bucket.data_.emplace_back(std::move(acq));
  }
}
} // namespace
void AcquisitionSpiralAccumulateWaveform::send_data(Core::OutputChannel &out, std::map<unsigned short, AcquisitionBucket> &buckets,
                                                    std::vector<Core::Waveform> &waveforms)
{
  //trigger_events++;
  //GDEBUG("Trigger (%d) occurred, sending out %d buckets\n", trigger_events, buckets.size());
  //buckets.begin()->second.waveform_ = std::move(waveforms);
  // Pass all buckets down the chain
  for (auto &bucket : buckets)
    out.push(std::move(bucket.second));

  buckets.clear();
}
void AcquisitionSpiralAccumulateWaveform ::process(
    Core::InputChannel<Core::variant<Core::Acquisition, Core::Waveform>> &in, Core::OutputChannel &out)
{
  using namespace Gadgetron::Indexing;
  auto waveforms = std::vector<Core::Waveform>{};
  auto grad_waveforms = std::vector<Core::Waveform>{};
  auto buckets = std::map<unsigned short, AcquisitionBucket>{};
  //auto header_vector = std::vector<ISMRMRD::AcquisitionHeader>;
  // auto trigger   = get_trigger(*this);
  int counterData;
  int waveForm_samples;
  int upsampleFactor;
  
   auto matrixsize = header.encoding.front().encodedSpace.matrixSize;
	 auto fov = header.encoding.front().encodedSpace.fieldOfView_mm;

   kspace_scaling= 1e-3*fov.x/matrixsize.x;

  for (auto message : in)
  {

    if (Core::holds_alternative<Core::Waveform>(message))
    {
      auto &temp_waveform = Core::get<Core::Waveform>(message);
      auto &wave_head = std::get<ISMRMRD::WaveformHeader>(Core::get<Core::Waveform>(message));

      if (wave_head.waveform_id < 10)
        waveforms.emplace_back(std::move(Core::get<Core::Waveform>(message)));
      else
      {
        waveForm_samples = wave_head.number_of_samples - 16;
        grad_waveforms.emplace_back(std::move(Core::get<Core::Waveform>(message)));
      }
      continue;
    }
    
//    if(Core::holds_alternative<ISMRMRD::ImageHeader>(message))

    if (Core::holds_alternative<Core::Acquisition>(message))
    {
      if (is_noise(Core::get<Core::Acquisition>(message)))
        continue;
      //auto& head = std::get<ISMRMRD::AcquisitionHeader>(acq);
      //auto data = std::get<hoNDArray<std::complex<float>>>(acq);
      
      auto &[head, data, traj] = Core::get<Core::Acquisition>(message);

      // Prepare Trajectory for each acq and push the bucked through
      head.trajectory_dimensions = 3;
      upsampleFactor = head.number_of_samples / waveForm_samples;
      
      auto trajectory_and_weights= prepare_trajectory_from_waveforms(grad_waveforms[counterData],grad_waveforms[counterData+1], head);
      head.trajectory_dimensions = 3;
      //grad_waveforms.erase(grad_waveforms.begin(), grad_waveforms.begin() + 1);
      int extraSamples = head.number_of_samples - waveForm_samples * upsampleFactor;

      std::vector<size_t> tmp_dims;
      tmp_dims.push_back(head.number_of_samples);
      tmp_dims.push_back(head.active_channels);
      data.reshape(tmp_dims);
      head.number_of_samples = head.number_of_samples - extraSamples;

      if (extraSamples != 0)
      {
        hoNDArray<std::complex<float>> data_short(data.get_size(0) - extraSamples, head.active_channels);
        for (int ii = extraSamples; ii < data.get_size(0); ii++)
        {
          data_short(ii - extraSamples, slice) = data(ii, slice);
        }
        data = data_short;
      }
      
      unsigned short sorting_index = get_index(head,sorting_dimension);

      Core::Acquisition acq = Core::Acquisition(std::move(head), std::move(data), std::move(trajectory_and_weights));

      counterData+=2;
      
      AcquisitionBucket &bucket = buckets[sorting_index];
      add_acquisition(bucket, std::move(acq));
    }
     
  }
   send_data(out, buckets, waveforms);
}
hoNDArray<float> AcquisitionSpiralAccumulateWaveform::prepare_trajectory_from_waveforms(const Core::Waveform &grad_waveform_x, const Core::Waveform &grad_waveform_y, const ISMRMRD::AcquisitionHeader &head)
{
  using namespace Gadgetron::Indexing;

  auto &[wave_head_x, wave_data_x] = grad_waveform_x;
  auto &[wave_head_y, wave_data_y] = grad_waveform_y;
  
  int upsampleFactor = head.number_of_samples / (wave_head_x.number_of_samples - 16);
  
  hoNDArray<float> gradient_x(wave_data_x.size() - 16);
  hoNDArray<float> gradient_y(wave_data_y.size() - 16);

  size_t grad_end_index = wave_data_x.size() - 16;
  size_t ghead_st_index = wave_data_x.size() - 16;
  size_t size_gradOVS = gradient_x.size() * upsampleFactor;

auto axisx = wave_data_x[11+ghead_st_index];
auto axisy = wave_data_y[11+ghead_st_index];
auto w1x=wave_data_x[ghead_st_index+12];
auto w2x=wave_data_x[ghead_st_index+13];
auto w1y=wave_data_y[ghead_st_index+12];
auto w2y=wave_data_y[ghead_st_index+13];

if(head.idx.kspace_encode_step_1 != wave_data_x[ghead_st_index+12] ||
   head.idx.kspace_encode_step_2 != wave_data_x[ghead_st_index+13] ||
   head.idx.kspace_encode_step_1 != wave_data_y[ghead_st_index+12] ||
   head.idx.kspace_encode_step_2 != wave_data_y[ghead_st_index+13] )  
    GERROR("Trajectory and data interleaves dont match: Becareful the images won't come out right becuase the trajectories are not correct \n");

if(axisx==axisy)
  GERROR("Waveforms are messeed up \n");

  auto trajectory_and_weights = hoNDArray<float>(head.trajectory_dimensions, size_gradOVS);
  auto gradients_interpolated = hoNDArray<float>(head.trajectory_dimensions - 1, size_gradOVS);

  auto wave_data_float_x = hoNDArray<float>(wave_data_x);
  auto wave_data_float_y = hoNDArray<float>(wave_data_y);
  for (int ii = 0; ii < grad_end_index; ii++)
  {
    gradient_x[ii] = (wave_data_float_x[ii] / std::numeric_limits<uint32_t>::max()) * 80 - 40;
    gradient_y[ii] = (wave_data_float_y[ii] / std::numeric_limits<uint32_t>::max()) * 80 - 40;

  //   GDEBUG("Gradient [%d]: value: %0.2f \t %0.2f\n", ii, gradient_x[ii],gradient_y[ii]);
  }

  gradients_interpolated(0,slice) = sincInterpolation(gradient_x, upsampleFactor);
  gradients_interpolated(1,slice) = sincInterpolation(gradient_y, upsampleFactor);

  trajectory_and_weights(0,0) = real(gradients_interpolated(0,0));
  trajectory_and_weights(1,0) = real(gradients_interpolated(1,0));
  for (int ii = 1; ii < size_gradOVS; ii++)
  {
    trajectory_and_weights(0,ii) = (real(gradients_interpolated(0,ii))*GAMMA*10*2/1000000*kspace_scaling + trajectory_and_weights(0,ii - 1)); // mT/m * Hz/G * 10G * 2e-6
    trajectory_and_weights(1,ii) = (real(gradients_interpolated(1,ii))*GAMMA*10*2/1000000*kspace_scaling + trajectory_and_weights(1,ii - 1)); // mT/m * Hz/G * 10G * 2e-6
  }

   float maxTx;
   float minTx;
   auto temp=permute(trajectory_and_weights,{1,0});
   maxValue(hoNDArray<float>(temp(slice,0)), maxTx);
   minValue(hoNDArray<float>(temp(slice,0)), minTx);

  hoNDArray<float> trajectories_temp(2,trajectory_and_weights.get_size(1));
  temp=permute(trajectory_and_weights,{1,0});
  trajectories_temp(0,slice)=hoNDArray<float>(temp(slice,0));
  trajectories_temp(1,slice)=hoNDArray<float>(temp(slice,1));

  trajectory_and_weights(2,slice) = calculate_weights_Hoge(gradients_interpolated, trajectories_temp);
  
  return trajectory_and_weights;
}

hoNDArray<float> AcquisitionSpiralAccumulateWaveform::sincInterpolation(const hoNDArray<float> input, int zpadFactor)
{
  hoNDArray<std::complex<float>> output(input.size() * zpadFactor);
  std::fill(output.begin(), output.end(), 0);
  auto cinput = hoNDArray<std::complex<float>>(input);
  hoNDFFT<float>::instance()->fft1c(cinput);
  for (int ii = 0; ii < output.size(); ii++)
  {
    if (ii > output.size() / 2 - cinput.size() / 2 - 1 && ii < output.size() / 2 + (cinput.size() / 2))
    {
      output(ii) = cinput(ii - (output.size() / 2 - cinput.size() / 2));
    //   GDEBUG("output [%d]: value: %0.2f + i %0.2f\n", ii, real(output[ii]), imag(output[ii]));
    }
  }

  hoNDFFT<float>::instance()->ifft1c(output);
  output *= sqrt(zpadFactor);
  return real(output);
}

hoNDArray<float> AcquisitionSpiralAccumulateWaveform::calculate_weights_Hoge(const hoNDArray<float> &gradients, const hoNDArray<float> &trajectories) {

    using namespace Gadgetron::Indexing;
        hoNDArray<float> weights(gradients.get_size(1),1);
        for (int ii=0;ii<gradients.get_size(1);ii++)
        {

        auto abs_g = sqrt(gradients(0, ii) * gradients(0, ii) + gradients(1, ii) * gradients(1, ii));
        auto abs_t = sqrt(trajectories(0, ii) * trajectories(0, ii) + trajectories(1, ii) * trajectories(1, ii));
        auto ang_g = atan2(gradients(1,ii),gradients(0,ii));
        auto ang_t = atan2(trajectories(1,ii),trajectories(0,ii));
          weights(ii)=abs(cos(ang_g-ang_t))*abs_g*abs_t;
        }           
         

        return weights;
    }
    
GADGETRON_GADGET_EXPORT(AcquisitionSpiralAccumulateWaveform);

namespace {
        const std::map<std::string, SortingDimension> sortdimension_from_name = {

            { "kspace_encode_step_1", SortingDimension::kspace_encode_step_1 },
            { "kspace_encode_step_2", SortingDimension::kspace_encode_step_2 },
            { "average", SortingDimension::average }, { "slice", SortingDimension::slice },
            { "contrast", SortingDimension::contrast }, { "phase", SortingDimension::phase },
            { "repetition", SortingDimension::repetition }, { "set", SortingDimension::set },
            { "segment", SortingDimension::segment }, { "user_0", SortingDimension::user_0 },
            { "user_1", SortingDimension::user_1 }, { "user_2", SortingDimension::user_2 },
            { "user_3", SortingDimension::user_3 }, { "user_4", SortingDimension::user_4 },
            { "user_5", SortingDimension::user_5 }, { "user_6", SortingDimension::user_6 },
            { "user_7", SortingDimension::user_7 }, { "n_acquisitions", SortingDimension::n_acquisitions },
            { "none", SortingDimension::none }, { "", SortingDimension::none }
        };
    }
    void from_string(const std::string& str, SortingDimension& sort) {
        auto lower = str;
        boost::to_lower(lower);
        sort = sortdimension_from_name.at(lower);
    }
} // namespace Gadgetron
