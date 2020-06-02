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
#include <iostream>
#include <string>
#include <fstream>
#include <boost/filesystem/fstream.hpp>
#include "armadillo"
#include "mri_core_utility.h"
#include "GadgetronTimer.h"


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


  readGIRFKernel(); // Read GIRF Kernel from file

  GadgetronTimer timer("Preparing Acquisition");

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
      hoNDArray<float> trajectory_and_weights;

      if(trajectory_map.find(head.idx.kspace_encode_step_1)==trajectory_map.end())
      {
        trajectory_and_weights = prepare_trajectory_from_waveforms(grad_waveforms[counterData],grad_waveforms[counterData+1], head);
        trajectory_map.insert(std::pair<size_t,hoNDArray<float>>(head.idx.kspace_encode_step_1,trajectory_and_weights));
      }
      else
      {
        trajectory_and_weights = trajectory_map.find(head.idx.kspace_encode_step_1)->second;
      }
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
      
       if(sorting_dimension==SortingDimension::average && head.idx.average>curAvg){
        curAvg=head.idx.average;
        send_data(out, buckets, waveforms);
      }

      unsigned short sorting_index = get_index(head,sorting_dimension);

      Core::Acquisition acq = Core::Acquisition(std::move(head), std::move(data), std::move(trajectory_and_weights));

      counterData+=2;
      
      AcquisitionBucket &bucket = buckets[sorting_index];
      add_acquisition(bucket, std::move(acq));

     
    }
 
  }
  GadgetronTimer timer1("Ending Acquisition");

  send_data(out, buckets, waveforms);
}
hoNDArray<float> AcquisitionSpiralAccumulateWaveform::prepare_trajectory_from_waveforms(const Core::Waveform &grad_waveform_x, const Core::Waveform &grad_waveform_y, const ISMRMRD::AcquisitionHeader &head)
{
  using namespace Gadgetron::Indexing;

  arma::fmat33 rotation_matrix;
  rotation_matrix(0, 0) = head.read_dir[0];
  rotation_matrix(1, 0) = head.read_dir[1];
  rotation_matrix(2, 0) = head.read_dir[2];
  rotation_matrix(0, 1) = head.phase_dir[0];
  rotation_matrix(1, 1) = head.phase_dir[1];
  rotation_matrix(2, 1) = head.phase_dir[2];
  rotation_matrix(0, 2) = head.slice_dir[0];
  rotation_matrix(1, 2) = head.slice_dir[1];
  rotation_matrix(2, 2) = head.slice_dir[2];

  auto TE_ = header.sequenceParameters.get().TE.get().at(0);
  auto &[wave_head_x, wave_data_x] = grad_waveform_x;
  auto &[wave_head_y, wave_data_y] = grad_waveform_y;
  
  int upsampleFactor = head.number_of_samples / (wave_head_x.number_of_samples - 16);
  
  hoNDArray<float> gradient_x(wave_data_x.size() - 16);
  hoNDArray<float> gradient_y(wave_data_y.size() - 16);
  hoNDArray<floatd2> gradients(wave_data_y.size() - 16);

  size_t grad_end_index = wave_data_x.size() - 16;
  size_t ghead_st_index = wave_data_x.size() - 16;
  size_t size_gradOVS = gradient_x.size() * upsampleFactor;

// Checking if the trajectories index and data indexes match 
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
    GERROR("Trajectory and data interleaves dont match: Be careful the images won't come out right becuase the trajectories are not correct \n");

if(axisx==axisy)
  GERROR("Waveforms are messeed up \n");

  auto trajectory_and_weights = hoNDArray<float>(head.trajectory_dimensions, size_gradOVS);

  auto wave_data_float_x = hoNDArray<float>(wave_data_x);
  auto wave_data_float_y = hoNDArray<float>(wave_data_y);

  for (int ii = 0; ii < grad_end_index; ii++)
  {
    gradients(ii)[0] = (wave_data_float_x[ii] / std::numeric_limits<uint32_t>::max()) * 80 - 40;
    gradients(ii)[1] = (wave_data_float_y[ii] / std::numeric_limits<uint32_t>::max()) * 80 - 40;

  }
  
   auto gradients_interpolated = zeroHoldInterpolation(gradients, upsampleFactor);
    
   auto corrected_interpolated_gradients  = GIRF::girf_correct(gradients_interpolated, girf_kernel, rotation_matrix, 2e-6, 10e-6, 0.85e-6);

  trajectory_and_weights(0,0) = (corrected_interpolated_gradients(0)[0])*GAMMA*10*2/1000000*kspace_scaling;
  trajectory_and_weights(1,0) = (corrected_interpolated_gradients(0)[1])*GAMMA*10*2/1000000*kspace_scaling;
  for (int ii = 1; ii < size_gradOVS; ii++)
  {
    trajectory_and_weights(0,ii) = ((corrected_interpolated_gradients(ii)[0])*GAMMA*10*2/1000000*kspace_scaling + trajectory_and_weights(0,ii - 1)); // mT/m * Hz/G * 10G * 2e-6
    trajectory_and_weights(1,ii) = ((corrected_interpolated_gradients(ii)[1])*GAMMA*10*2/1000000*kspace_scaling + trajectory_and_weights(1,ii - 1)); // mT/m * Hz/G * 10G * 2e-6
  }
  
  hoNDArray<float> trajectories_temp(2,trajectory_and_weights.get_size(1));
  auto temp2=permute(trajectory_and_weights,{1,0});
  trajectories_temp(0,slice)=hoNDArray<float>(temp2(slice,0));
  trajectories_temp(1,slice)=hoNDArray<float>(temp2(slice,1));

  trajectory_and_weights(2,slice) = calculate_weights_Hoge(corrected_interpolated_gradients, trajectories_temp);
  
  return trajectory_and_weights;
}

hoNDArray<floatd2> AcquisitionSpiralAccumulateWaveform::sincInterpolation(const hoNDArray<floatd2> input, int zpadFactor)
{
  hoNDArray<floatd2> output(input.size() * zpadFactor);
  //std::fill(output.begin(), output.end(), 0);
  hoNDArray<std::complex<float>> cinput  = hoNDArray<std::complex<float>>(input.size());
  hoNDArray<std::complex<float>> coutput = hoNDArray<std::complex<float>>(input.size()* zpadFactor);

  for (int jj=0;jj<2;jj++){

    std::fill(coutput.begin(), coutput.end(), 0);
    
    for (int zz=0;zz<cinput.size();zz++){
      cinput(zz)=(input(zz)[jj]);
  //     GDEBUG("cinput [%d]: value: %d \n", zz, real(cinput[zz]));

    }
    //auto temp = hoNDArray<std::complex<float>>(input[jj]);
    //cinput=hoNDArray<std::complex<float>>(cinput);
    hoNDFFT<float>::instance()->fft1c(cinput);

    for (int ii = 0; ii < coutput.size(); ii++)
    {
      if (ii > coutput.size() / 2 - cinput.size() / 2 - 1 && ii < coutput.size() / 2 + (cinput.size() / 2))
      {
        coutput(ii) = cinput(ii - (output.size() / 2 - cinput.size() / 2));
        //   GDEBUG("output [%d]: value: %0.2f + i %0.2f\n", ii, real(output[ii]), imag(output[ii]));
      }
    }

    hoNDFFT<float>::instance()->ifft1c(coutput);
    coutput *= sqrt(zpadFactor);
    for (int zz=0;zz<coutput.size();zz++)
    {
      output(zz)[jj]=real(coutput(zz));
  //               GDEBUG("output [%d]: value: %d \n", zz, output(zz)[jj]);

    }

  }
 // output *= sqrt(zpadFactor);
  return output;
}
hoNDArray<floatd2> AcquisitionSpiralAccumulateWaveform::zeroHoldInterpolation(const hoNDArray<floatd2> input, int zpadFactor)
{
   hoNDArray<floatd2> output(input.size()*zpadFactor);

     for (int ii=0; ii<input.size()*zpadFactor; ii++)
     {
       output(ii)=input(int(ii/zpadFactor));
     } 
     return output;
}

hoNDArray<float> AcquisitionSpiralAccumulateWaveform::calculate_weights_Hoge(const hoNDArray<floatd2> &gradients, const hoNDArray<float> &trajectories) {

    using namespace Gadgetron::Indexing;
        hoNDArray<float> weights(trajectories.get_size(1),1);
        for (int ii=0;ii<trajectories.get_size(1);ii++)
        {

        auto abs_g = sqrt(gradients(ii)[0] * gradients(ii)[0] + gradients(ii)[1] * gradients(ii)[1]);
        auto abs_t = sqrt(trajectories(0, ii) * trajectories(0, ii) + trajectories(1, ii) * trajectories(1, ii));
        auto ang_g = atan2(gradients(ii)[1],gradients(ii)[0]);
        auto ang_t = atan2(trajectories(1,ii),trajectories(0,ii));
          weights(ii)=abs(cos(ang_g-ang_t))*abs_g*abs_t;
    //       GDEBUG("weights [%d]: value: %0.6f \n", ii, weights(ii));
        }           
         

        return weights;
    }
  void AcquisitionSpiralAccumulateWaveform::readGIRFKernel(){
    using namespace std;
    using namespace boost::filesystem;
    using namespace Gadgetron::Indexing;
    path fnamex = "/opt/data/GIRF/GIRFx.txt";
    path fnamey = "/opt/data/GIRF/GIRFy.txt";
    path fnamez = "/opt/data/GIRF/GIRFz.txt";

    boost::filesystem::fstream filex, filey, filez;
    filex.open(fnamex);
    filey.open(fnamey);
    filez.open(fnamez);

    vector<std::complex<float>> girfx;
    vector<std::complex<float>> girfy;
    vector<std::complex<float>> girfz;
    string temp_line;

    // Header first 4 lines
    for (int ii = 0; ii < 4; ii++)
    {
      getline(filex, temp_line);
      if (ii == 0)
        girf_numpoint = stoi(temp_line);
      if (ii == 2)
        girf_sampletime = stod(temp_line);

      getline(filey, temp_line); // got info from x no need for y and z but still need to skip the lines
      getline(filez, temp_line);
    }

    girf_kernel = hoNDArray<std::complex<float>>(girf_numpoint,3);
    string temp_liner;
    string temp_linei;
    int index=0;
    while (getline(filex, temp_liner))
    {
      getline(filex, temp_linei);
      girf_kernel(index,1)=complex<float>(stod(temp_liner),stod(temp_linei));
      
      getline(filey, temp_liner);
      getline(filey, temp_linei);
      girf_kernel(index,0)=complex<float>(stod(temp_liner),stod(temp_linei));
      
      getline(filez, temp_liner);
      getline(filez, temp_linei);
      girf_kernel(index,2)=complex<float>(stod(temp_liner),stod(temp_linei));
      index++;
      //pushback(stod(temp_line));
    }

   // std::copy(girfx.begin(),girfx.end(),);

 //   for (auto inp : girfx)
 //    GDEBUG("GIRFX: %0.6f + j%0.6f \n", real(inp),imag(inp));
  }

void AcquisitionSpiralAccumulateWaveform::printGradtoFile(std::string fname_grad, hoNDArray<floatd2> grad_traj)
{
  std::ofstream of(fname_grad);
  for (auto ele : grad_traj)
    of << ele[0] << "\t" << ele[1] << "\n";
  of.close();  
}

void AcquisitionSpiralAccumulateWaveform::printTrajtoFile(std::string fname_grad, hoNDArray<float> grad_traj)
{
  std::ofstream of(fname_grad);
  for (int i=0;i<grad_traj.get_size(1);i++)
    of << grad_traj(0,i) << "\t" << grad_traj(1,i) << "\n";
  of.close();  
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
