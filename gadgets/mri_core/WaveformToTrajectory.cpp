#include "WaveformToTrajectory.h"

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

constexpr double GAMMA = 4258.0; /* Hz/G */
constexpr double PI = boost::math::constants::pi<double>();
namespace Gadgetron
{

  WaveformToTrajectory::WaveformToTrajectory(const Core::Context &context, const Core::GadgetProperties &props)
      : ChannelGadget(context, props), header{context.header} {}

  namespace
  {
    bool is_noise(Core::Acquisition &acq)
    {
      return std::get<ISMRMRD::AcquisitionHeader>(acq).isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NOISE_MEASUREMENT);
    }
  } // namespace
  void WaveformToTrajectory ::process(
      Core::InputChannel<Core::variant<Core::Acquisition, Core::Waveform>> &in, Core::OutputChannel &out)
  {
    using namespace Gadgetron::Indexing;

    int waveForm_samples;
    int upsampleFactor;

    auto matrixsize = header.encoding.front().encodedSpace.matrixSize;
    auto fov = header.encoding.front().encodedSpace.fieldOfView_mm;

    kspace_scaling = 1e-3 * fov.x / matrixsize.x;
    if(perform_GIRF)
      readGIRFKernel(); // Read GIRF Kernel from file

    GadgetronTimer timer("WaveformToTrajectory");

    for (auto message : in)
    {

      if (Core::holds_alternative<Core::Waveform>(message))
      {
        auto &temp_waveform = Core::get<Core::Waveform>(message);
        auto &wave_head = std::get<ISMRMRD::WaveformHeader>(Core::get<Core::Waveform>(message));

        if (wave_head.waveform_id >= 10)
        {
          waveForm_samples = wave_head.number_of_samples - 16;
          gradient_wave_store.insert(std::pair<size_t, Core::Waveform>(wave_head.scan_counter, std::move(Core::get<Core::Waveform>(message))));
        }
        continue;
      }

      if (Core::holds_alternative<Core::Acquisition>(message))
      {
        if (is_noise(Core::get<Core::Acquisition>(message)))
          continue;

        auto &[head, data, traj] = Core::get<Core::Acquisition>(message);

        // Prepare Trajectory for each acq and push the bucked through
        head.trajectory_dimensions = 3;
        upsampleFactor = head.number_of_samples / waveForm_samples;
        hoNDArray<float> trajectory_and_weights;

        if (trajectory_map.find(head.idx.kspace_encode_step_1) == trajectory_map.end())
        {

          prepare_trajectory_from_waveforms(gradient_wave_store.find(head.scan_counter)->second, head);
        }

        trajectory_and_weights = trajectory_map.find(head.idx.kspace_encode_step_1)->second;

        head.trajectory_dimensions = 3;

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

        auto acq = Core::Acquisition(std::move(head), std::move(data), std::move(trajectory_and_weights));

        out.push(acq);
      }
    }
    GadgetronTimer timer1("WaveformToTrajectory");
  }
  void WaveformToTrajectory::prepare_trajectory_from_waveforms(Core::Waveform &grad_waveform, const ISMRMRD::AcquisitionHeader &head)
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
    auto &[wave_head, wave_data] = grad_waveform;

    hoNDArray<float> wave_data_float(wave_data.size() / 3, 3);
    auto wave_data_floatx = reinterpret_cast<const float *>(wave_data.get_data_ptr());

    int numberofGradSamples = wave_data_floatx[0];
    //auto wave_data_float = hoNDArray<const float>(wave_data_floatx);
    int sizeofCustomHeader = (wave_data.size() - 3 * numberofGradSamples) / 3;

    std::copy(wave_data_floatx, wave_data_floatx + wave_data.size(), wave_data_float.begin());

    int upsampleFactor = head.number_of_samples / numberofGradSamples;

    hoNDArray<floatd2> gradients(numberofGradSamples);

    size_t size_gradOVS = numberofGradSamples * upsampleFactor;

    auto trajectory_and_weights = hoNDArray<float>(head.trajectory_dimensions, size_gradOVS);

    for (int ii = 0; ii < numberofGradSamples; ii++)
    {
      gradients(ii)[0] = wave_data_float(sizeofCustomHeader + ii, 0); // / std::numeric_limits<uint32_t>::max()) * 80 - 40;
      gradients(ii)[1] = wave_data_float(sizeofCustomHeader + ii, 1); // / std::numeric_limits<uint32_t>::max()) * 80 - 40;
    }

    auto gradients_interpolated = zeroHoldInterpolation(gradients, upsampleFactor);

    if(perform_GIRF)
      gradients_interpolated = GIRF::girf_correct(gradients_interpolated, girf_kernel, rotation_matrix, 2e-6, 10e-6, 0.85e-6);

    trajectory_and_weights(0, 0) = (gradients_interpolated(0)[0]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling;
    trajectory_and_weights(1, 0) = (gradients_interpolated(0)[1]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling;
    for (int ii = 1; ii < size_gradOVS; ii++)
    {
      trajectory_and_weights(0, ii) = ((gradients_interpolated(ii)[0]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling + trajectory_and_weights(0, ii - 1)); // mT/m * Hz/G * 10G * 2e-6
      trajectory_and_weights(1, ii) = ((gradients_interpolated(ii)[1]) * GAMMA * 10 * 2 / 1000000 * kspace_scaling + trajectory_and_weights(1, ii - 1)); // mT/m * Hz/G * 10G * 2e-6
    }

    hoNDArray<float> trajectories_temp(2, trajectory_and_weights.get_size(1));
    auto temp2 = permute(trajectory_and_weights, {1, 0});
    trajectories_temp(0, slice) = hoNDArray<float>(temp2(slice, 0));
    trajectories_temp(1, slice) = hoNDArray<float>(temp2(slice, 1));

    trajectory_and_weights(2, slice) = calculate_weights_Hoge(gradients_interpolated, trajectories_temp);

    trajectory_map.insert(std::pair<size_t, hoNDArray<float>>(head.idx.kspace_encode_step_1, trajectory_and_weights));
  }

  hoNDArray<floatd2> WaveformToTrajectory::sincInterpolation(const hoNDArray<floatd2> input, int zpadFactor)
  {
    hoNDArray<floatd2> output(input.size() * zpadFactor);
    //std::fill(output.begin(), output.end(), 0);
    hoNDArray<std::complex<float>> cinput = hoNDArray<std::complex<float>>(input.size());
    hoNDArray<std::complex<float>> coutput = hoNDArray<std::complex<float>>(input.size() * zpadFactor);

    for (int jj = 0; jj < 2; jj++)
    {

      std::fill(coutput.begin(), coutput.end(), 0);

      for (int zz = 0; zz < cinput.size(); zz++)
      {
        cinput(zz) = (input(zz)[jj]);
      }

      hoNDFFT<float>::instance()->fft1c(cinput);

      for (int ii = 0; ii < coutput.size(); ii++)
      {
        if (ii > coutput.size() / 2 - cinput.size() / 2 - 1 && ii < coutput.size() / 2 + (cinput.size() / 2))
        {
          coutput(ii) = cinput(ii - (output.size() / 2 - cinput.size() / 2));
        }
      }

      hoNDFFT<float>::instance()->ifft1c(coutput);
      coutput *= sqrt(zpadFactor);
      for (int zz = 0; zz < coutput.size(); zz++)
      {
        output(zz)[jj] = real(coutput(zz));
      }
    }
    // output *= sqrt(zpadFactor);
    return output;
  }
  hoNDArray<floatd2> WaveformToTrajectory::zeroHoldInterpolation(const hoNDArray<floatd2> input, int zpadFactor)
  {
    hoNDArray<floatd2> output(input.size() * zpadFactor);

    for (int ii = 0; ii < input.size() * zpadFactor; ii++)
    {
      output(ii) = input(int(ii / zpadFactor));
    }
    return output;
  }

  hoNDArray<float> WaveformToTrajectory::calculate_weights_Hoge(const hoNDArray<floatd2> &gradients, const hoNDArray<float> &trajectories)
  {

    using namespace Gadgetron::Indexing;
    hoNDArray<float> weights(trajectories.get_size(1), 1);
    for (int ii = 0; ii < trajectories.get_size(1); ii++)
    {

      auto abs_g = sqrt(gradients(ii)[0] * gradients(ii)[0] + gradients(ii)[1] * gradients(ii)[1]);
      auto abs_t = sqrt(trajectories(0, ii) * trajectories(0, ii) + trajectories(1, ii) * trajectories(1, ii));
      auto ang_g = atan2(gradients(ii)[1], gradients(ii)[0]);
      auto ang_t = atan2(trajectories(1, ii), trajectories(0, ii));
      weights(ii) = abs(cos(ang_g - ang_t)) * abs_g * abs_t;
    }

    return weights;
  }
  void WaveformToTrajectory::readGIRFKernel()
  {
    using namespace std;
    using namespace boost::filesystem;
    using namespace Gadgetron::Indexing;
    path fnamex = GIRF_folder + "GIRFx.txt";
    path fnamey = GIRF_folder + "GIRFy.txt";
    path fnamez = GIRF_folder + "GIRFz.txt";

    boost::filesystem::fstream filex, filey, filez;
    filex.open(fnamex);
    filey.open(fnamey);
    filez.open(fnamez);

    if (!filex.is_open() || !filey.is_open() || !filez.is_open())
      throw std::runtime_error("GIRF Kernel is not loaded and Waveform to Trajectories need this");
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

    girf_kernel = hoNDArray<std::complex<float>>(girf_numpoint, 3);
    string temp_liner;
    string temp_linei;
    int index = 0;
    while (getline(filex, temp_liner))
    {
      getline(filex, temp_linei);
      girf_kernel(index, 1) = complex<float>(stod(temp_liner), stod(temp_linei));

      getline(filey, temp_liner);
      getline(filey, temp_linei);
      girf_kernel(index, 0) = complex<float>(stod(temp_liner), stod(temp_linei));

      getline(filez, temp_liner);
      getline(filez, temp_linei);
      girf_kernel(index, 2) = complex<float>(stod(temp_liner), stod(temp_linei));
      index++;
    }
  }

  void WaveformToTrajectory::printGradtoFile(std::string fname_grad, hoNDArray<floatd2> grad_traj)
  {
    std::ofstream of(fname_grad);
    for (auto ele : grad_traj)
      of << ele[0] << "\t" << ele[1] << "\n";
    of.close();
  }

  void WaveformToTrajectory::printTrajtoFile(std::string fname_grad, hoNDArray<float> grad_traj)
  {
    std::ofstream of(fname_grad);
    for (int i = 0; i < grad_traj.get_size(1); i++)
      of << grad_traj(0, i) << "\t" << grad_traj(1, i) << "\n";
    of.close();
  }

  GADGETRON_GADGET_EXPORT(WaveformToTrajectory);

} // namespace Gadgetron
