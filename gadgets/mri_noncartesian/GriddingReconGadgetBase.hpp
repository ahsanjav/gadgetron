#include "GriddingReconGadgetBase.h"
#include "mri_core_grappa.h"

#include "vector_td_utilities.h"
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include "cgSolver.h"
#include "hoNDArray_math.h"
#include "hoNDArray_utils.h"
#include <numeric>
#include <random>
#include "NonCartesianTools.h"
#include "NFFTOperator.h"
#include "hoNDFFT.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <boost/range/combine.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
namespace Gadgetron
{

template <template <class> class ARRAY>
GriddingReconGadgetBase<ARRAY>::GriddingReconGadgetBase() : Gadget1() {}
template <template <class> class ARRAY>
GriddingReconGadgetBase<ARRAY>::~GriddingReconGadgetBase() {}
template <template <class> class ARRAY>
int GriddingReconGadgetBase<ARRAY>::process_config(ACE_Message_Block *mb)
{
    // -------------------------------------------------

    ISMRMRD::IsmrmrdHeader h;
    deserialize(mb->rd_ptr(), h);

    auto matrixsize = h.encoding.front().encodedSpace.matrixSize;
    auto fov = h.encoding.front().encodedSpace.fieldOfView_mm;

    trajectory_scaling_ = 1e-2*fov.x/matrixsize.x;
    kernel_width_ = kernel_width.value();
    oversampling_factor_ = gridding_oversampling_factor.value();

    image_dims_.push_back(matrixsize.x);
    image_dims_.push_back(matrixsize.y);

    //Figure out what the oversampled matrix size should be taking the warp size into consideration.
    unsigned int warp_size = 32;
    image_dims_os_ = uint64d2(((static_cast<size_t>(std::ceil(image_dims_[0] * oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size,
                              ((static_cast<size_t>(std::ceil(image_dims_[1] * oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size);

    // In case the warp_size constraint kicked in
    oversampling_factor_ = float(image_dims_os_[0]) / float(image_dims_[0]);
    this->initialize_encoding_space_limits(h);

    return GADGET_OK;
}

template <template <class> class ARRAY>
int GriddingReconGadgetBase<ARRAY>::process(Gadgetron::GadgetContainerMessage<IsmrmrdReconData> *m1)
{
    using namespace Gadgetron::Indexing;
    std::unique_ptr<GadgetronTimer> timer;
    if (perform_timing)
    {
        timer = std::make_unique<GadgetronTimer>("Gridding Recon");
    }
    process_called_times++;

    IsmrmrdReconData *recon_bit_ = m1->getObjectPtr();

    // for every encoding space
    for (size_t e = 0; e < recon_bit_->rbit_.size(); e++)
    {

        IsmrmrdDataBuffered *buffer = &(recon_bit_->rbit_[e].data_);

        size_t RO = buffer->data_.get_size(0);
        size_t E1 = buffer->data_.get_size(1);
        size_t E2 = buffer->data_.get_size(2);
        size_t CHA = buffer->data_.get_size(3);
        size_t N = buffer->data_.get_size(4);
        size_t S = buffer->data_.get_size(5);
        size_t SLC = buffer->data_.get_size(6);

        if (E2 > 1)
        {
            //GERROR("3D data is not supported in GriddingReconGadgetBase\n");
            //m1->release();
            //return GADGET_FAIL;
        }

        if (buffer->trajectory_ == Core::none)
        {
            GERROR("Trajectories not found. Bailing out.\n");
            m1->release();
            return GADGET_FAIL;
        }
    
//      float maxTx;
//   float minTx;
//   auto temp=permute(trajectory_and_weights,{1,0});
//   maxValue(hoNDArray<float>(temp(slice,0)), maxTx);
//   minValue(hoNDArray<float>(temp(slice,0)), minTx);

//   float maxTy;
//   float minTy;
//   maxValue(hoNDArray<float>(temp(slice,1)), maxTy);
//   minValue(hoNDArray<float>(temp(slice,1)), minTy);


        std::vector<size_t> reshapeVec = {RO, E1, E2, CHA, N, S, SLC};
        boost::shared_ptr<ARRAY<float>> dcw;
		boost::shared_ptr<ARRAY<floatd2>> traj;

		//auto & trajectory = *buffer->trajectory_;
		auto &trajectory2 = *buffer->trajectory_;
		for (int iSL = 0; iSL < E2; iSL++)
		{
			hoNDArray<float> trajectory = hoNDArray<float>(trajectory2(slice, slice, slice, iSL, 0));

			for (int ii = 0; ii < trajectory.get_size(1); ii++)
			{
				//  ofs << trajectory(0,ii,0)<< "\t" << trajectory(1,ii,0)<< "\t" << trajectory(2,ii,0)<< std::endl;
				for (int jj = 0; jj < trajectory.get_size(2); jj++)
				{
					if (abs(trajectory(0, ii, jj)) > 0.5 || abs(trajectory(1, ii, jj)) > 0.5 || abs(trajectory(2, ii, jj)) == 14.5349)
						GDEBUG("trajectoryXX [%d]: value: %0.2f \t %0.2f \t weights:%0.2f\n", ii, trajectory(0, ii, jj), trajectory(1, ii, jj), trajectory(2, ii, jj));
				}
			}
			GDEBUG("Header Dim: %d \n", buffer->headers_[0].trajectory_dimensions);
			if (buffer->headers_[0].trajectory_dimensions == 3 || trajectory.get_size(0) == 3)
			{ // the second condition is due to a bug
				auto traj_dcw = separate_traj_and_dcw(&trajectory);
				dcw = boost::make_shared<ARRAY<float>>(std::get<1>(traj_dcw).get());
				traj = boost::make_shared<ARRAY<floatd2>>(std::get<0>(traj_dcw).get());
			}
			else if (buffer->headers_[0].trajectory_dimensions == 2)
			{
				auto old_traj_dims = *trajectory.get_dimensions();
				std::vector<size_t> traj_dims(old_traj_dims.begin() + 1, old_traj_dims.end()); //Remove first element
				hoNDArray<floatd2> tmp_traj(traj_dims, (floatd2 *)trajectory.get_data_ptr());
				traj = boost::make_shared<ARRAY<floatd2>>(tmp_traj);
			}
			else
			{
				throw std::runtime_error("Unsupported number of trajectory dimensions");
			}

			std::vector<size_t> tmp_dims;
			tmp_dims.push_back((size_t)2);
			auto &bdata = *(hoNDArray<std::complex<float>> *)(&buffer->data_);
			
			hoNDFFT<float>::instance()->ifft(&bdata, 2); // how to do a fftshift
			// auto bdata2r = real(*bdata2); //(slice,slice,1,slice,slice,slice,slice)
			// auto bdata2i = imag(*bdata2);

			//hoNDArray<float> bdata2rs(RO,E1,1,CHA,N,S,SLC);
			//hoNDArray<float> bdata2is(RO,E1,1,CHA,N,S,SLC);
			//bdata2r.reshape(reshapeVec);
			//bdata2i.reshape(reshapeVec);
			// bdata2r = permute(bdata2r,new_order2);
			// bdata2i = permute(bdata2i,new_order2);

			// auto bdata2rs = bdata2r(slice, slice, slice, slice, slice, slice, 11);
			// auto bdata2is = bdata2i(slice, slice, slice, slice, slice, slice, 11);
			// auto bdata2rsf = hoNDArray<float>(bdata2rs);
			// auto bdata2isf = hoNDArray<float>(bdata2is);
			// //bdata2rsf.get_data_ptr() = bdata2rs.data();
			//bdata2rsf.dimensions = bdata2rs.dimensions;
			std::vector<size_t> new_order2 = {0, 1, 3, 5, 6, 2, 4};
			auto bdata_permuted = permute(bdata, new_order2);
			auto bdata2 = hoNDArray<complex_float_t>(bdata_permuted(slice, slice, slice, slice, slice, iSL, 0));
			//hoNDArray<float> bdata2isf =
			//hoNDArray<std::complex<float>> bdata2s;
			//real_imag_to_complex(real(hoNDArray<float>(bdata2rsf)), real(hoNDArray<float>(bdata2isf)), bdata2s);

			//auto permuted = permute(*(hoNDArray<float_complext>*)&buffer->data_,new_order);
            std::vector<size_t> new_order = {0, 1, 3, 4, 2};
        auto permuted = permute(hoNDArray<float_complext>(bdata2), new_order);
        ARRAY<float_complext> data(permuted);

        if (dcw)
        {
            float scale_factor = float(prod(image_dims_os_)) / asum(dcw.get());
            *dcw *= scale_factor;
        }

        //Gridding
        auto images = reconstruct(&data, traj.get(), dcw.get(), CHA);

        //Calculate coil sensitivity map
        auto csm = estimate_b1_map<float, 2>(*images);

        //Coil combine
        *images *= *conj(&csm);
        auto combined = sum(images.get(), images->get_number_of_dimensions() - 1);

        auto host_img = as_hoNDArray(combined);

        IsmrmrdImageArray imarray;

        auto elements = imarray.data_.get_number_of_elements();
        imarray.data_ = std::move(*boost::reinterpret_pointer_cast<decltype(imarray.data_)>(host_img));
        //          memcpy(imarray.data_.get_data_ptr(), host_img->get_data_ptr(), host_img->get_number_of_bytes());
        recon_bit_->rbit_[e].data_.headers_[0].idx.slice=iSL;

        NonCartesian::append_image_header(imarray, recon_bit_->rbit_[e], e);
        this->prepare_image_array(imarray, e, ((int)e + 1), GADGETRON_IMAGE_REGULAR);
        //imarray.headers_[0].slice=iSL;
        GadgetContainerMessage<Gadgetron::IsmrmrdImageArray> *m4 = new GadgetContainerMessage<Gadgetron::IsmrmrdImageArray>();
        *(m4->getObjectPtr()) = imarray;

        GadgetContainerMessage<ISMRMRD::ImageHeader> *m3 = new GadgetContainerMessage<ISMRMRD::ImageHeader>;
        *(m3->getObjectPtr()) = m4->getObjectPtr()->headers_[0];
        // m3->cont(m4); Shouldnt this add to the que m4 as well
        
        
        if (this->next()->putq(m3) < 0)
        {
            GDEBUG("Failed to put job on queue. Stupid Griding Recon\n");
            m3->release();
            return GADGET_FAIL;
        }

        if (this->next()->putq(m4) < 0)
        {
            GDEBUG("Failed to put job on queue. Stupid Griding Recon\n");
            m4->release();
            return GADGET_FAIL;
        }

        //Is this where we measure SNR?
        if (replicas.value() > 0 && snr_frame.value() == process_called_times)
        {

            pseudo_replica(buffer->data_, *traj, *dcw, csm, recon_bit_->rbit_[e], e, CHA);
        }
		}
}
    m1->release();

    return GADGET_OK;
}

template <template <class> class ARRAY>
boost::shared_ptr<ARRAY<float_complext>> GriddingReconGadgetBase<ARRAY>::reconstruct(
    ARRAY<float_complext> *data,
    ARRAY<floatd2> *traj,
    ARRAY<float> *dcw,
    size_t ncoils)
{
    GadgetronTimer timer("Reconstruct");
    //We have density compensation and iteration is set to false
    if (!iterate.value() && dcw)
    {

        auto plan = NFFT<ARRAY, float, 2>::make_plan(from_std_vector<size_t, 2>(image_dims_), image_dims_os_, kernel_width_);
        std::vector<size_t> recon_dims = image_dims_;
        recon_dims.push_back(ncoils);
        auto result = new ARRAY<float_complext>(recon_dims);

        std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
        ARRAY<floatd2> flat_traj(flat_dims, traj->get_data_ptr());

        plan->preprocess(flat_traj, NFFT_prep_mode::NC2C);
        plan->compute(*data, *result, dcw, NFFT_comp_mode::BACKWARDS_NC2C);

        return boost::shared_ptr<ARRAY<float_complext>>(result);
    }
    else
    { //No density compensation, we have to do iterative reconstruction.
        std::vector<size_t> recon_dims = image_dims_;
        recon_dims.push_back(ncoils);

        auto E = boost::make_shared<NFFTOperator<ARRAY, float, 2>>();

        E->setup(from_std_vector<size_t, 2>(image_dims_), image_dims_os_, kernel_width_);
        if (dcw)
        {
            E->set_dcw(boost::make_shared<ARRAY<float>>(*dcw));
        }
        std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
        ARRAY<floatd2> flat_traj(flat_dims, traj->get_data_ptr());

        E->set_domain_dimensions(&recon_dims);
        cgSolver<ARRAY<float_complext>> solver;
        solver.set_max_iterations(iteration_max.value());
        solver.set_encoding_operator(E);
        solver.set_tc_tolerance(iteration_tol.value());
        solver.set_output_mode(decltype(solver)::OUTPUT_SILENT);
        E->set_codomain_dimensions(data->get_dimensions().get());
        E->preprocess(flat_traj);
        auto res = solver.solve(data);
        return res;
    }
}

template <template <class> class ARRAY>
std::tuple<boost::shared_ptr<hoNDArray<floatd2>>, boost::shared_ptr<hoNDArray<float>>> GriddingReconGadgetBase<ARRAY>::separate_traj_and_dcw(
    hoNDArray<float> *traj_dcw)
{
    std::vector<size_t> dims = *traj_dcw->get_dimensions();
    std::vector<size_t> reduced_dims(dims.begin() + 1, dims.end()); //Copy vector, but leave out first dim
    auto dcw = boost::make_shared<hoNDArray<float>>(reduced_dims);

    auto traj = boost::make_shared<hoNDArray<floatd2>>(reduced_dims);

    auto dcw_ptr = dcw->get_data_ptr();
    auto traj_ptr = traj->get_data_ptr();
    auto ptr = traj_dcw->get_data_ptr();
    std::ofstream ofs("/tmp/traj_grad_flat.log");
    for (size_t i = 0; i < traj_dcw->get_number_of_elements() / 3; i++)
    {
        traj_ptr[i][0] = ptr[i * 3];
        traj_ptr[i][1] = ptr[i * 3 + 1];
        dcw_ptr[i] = ptr[i * 3 + 2];
        if (abs(ptr[i * 3]) > 0.5 || abs(ptr[i * 3 + 1]) > 0.5)
            GDEBUG("trajectory [%d]: value: %0.2f \t %0.2f \t weights:%0.2f\n", i, ptr[i * 3], ptr[i * 3 + 1], ptr[i * 3 + 2]);

        ofs << ptr[i * 3] << "\t" << ptr[i * 3 + 1] << "\t" << ptr[i * 3 + 2] << std::endl;
    }

    return std::make_tuple(traj, dcw);
}

template <template <class> class ARRAY>
void GriddingReconGadgetBase<ARRAY>::pseudo_replica(const hoNDArray<std::complex<float>> &data,
                                                    ARRAY<floatd2> &traj, ARRAY<float> &dcw, const ARRAY<float_complext> &csm,
                                                    const IsmrmrdReconBit &recon_bit, size_t encoding, size_t ncoils)
{
    hoNDArray<std::complex<float>> rep_array(image_dims_[0], image_dims_[1], replicas.value());

    std::mt19937 engine;
    std::normal_distribution<float> distribution;

    for (size_t r = 0; r < replicas.value(); ++r)
    {

        if (r % 10 == 0)
        {
            GDEBUG("Running pseudo replics %d of %d\n", r, replicas.value());
        }
        hoNDArray<std::complex<float>> dtmp = data;
        std::vector<size_t> new_order = {0, 1, 2, 4, 5, 6, 3};
        auto permuted_rep = permute(*(hoNDArray<float_complext> *)&dtmp, new_order);
        auto dataptr = permuted_rep.get_data_ptr();

        for (size_t k = 0; k < permuted_rep.get_number_of_elements(); k++)
        {
            dataptr[k] += std::complex<float>(distribution(engine), distribution(engine));
        }

        ARRAY<float_complext> data_rep(permuted_rep);

        auto images = reconstruct(&data_rep, &traj, &dcw, ncoils);

        //Coil combine
        *images *= *conj(&csm);
        auto combined = sum(images.get(), images->get_number_of_dimensions() - 1);

        auto host_img = as_hoNDArray(combined);

        size_t offset = image_dims_[0] * image_dims_[1] * r;

        memcpy(rep_array.get_data_ptr() + offset, host_img->get_data_ptr(), host_img->get_number_of_bytes());
    }

    hoNDArray<float> mag(rep_array.get_dimensions());
    hoNDArray<float> mean(image_dims_[0], image_dims_[1]);
    hoNDArray<float> std(image_dims_[0], image_dims_[1]);

    Gadgetron::abs(rep_array, mag);
    Gadgetron::sum_over_dimension(mag, mean, 2);
    Gadgetron::scal(1.0f / replicas.value(), mean);

    mag -= mean;
    mag *= mag;

    Gadgetron::sum_over_dimension(mag, std, 2);
    Gadgetron::scal(1.0f / (replicas.value() - 1), std);
    Gadgetron::sqrt_inplace(&std);

    //SNR image
    mean /= std;
    IsmrmrdImageArray imarray;
    imarray.data_ = *real_to_complex<std::complex<float>>(&mean);

    NonCartesian::append_image_header(imarray, recon_bit, encoding);
    this->prepare_image_array(imarray, encoding, image_series + 100 * ((int)encoding + 3), GADGETRON_IMAGE_SNR_MAP);
    this->next()->putq(new GadgetContainerMessage<IsmrmrdImageArray>(imarray));
}

} // namespace Gadgetron

