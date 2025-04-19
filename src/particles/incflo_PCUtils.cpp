#include <AMReX_ParticleInterpolators.H>
#include <incflo_PC.H>

#ifdef INCFLO_USE_PARTICLES

#ifdef USE_INCFLO_PYBIND11
#include <pybind11/embed.h>
namespace py = pybind11;
#endif

using namespace amrex;

void incflo_PC::massDensity ( MultiFab&  a_mf,
                              const int& a_lev,
                              const int& a_comp ) const
{
    BL_PROFILE("incflo_PC::massDensity()");

    AMREX_ASSERT(OK());
    AMREX_ASSERT(numParticlesOutOfRange(*this, 0) == 0);

    const auto& geom = Geom(a_lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();

    const Real inv_cell_volume = dxi[0]*dxi[1]*dxi[2];
    a_mf.setVal(0.0);

    ParticleToMesh( *this, a_mf, a_lev,
        [=] AMREX_GPU_DEVICE (  const incflo_PC::ParticleTileType::ConstParticleTileDataType& ptd,
                                int i, Array4<Real> const& rho)
        {
            auto p = ptd.m_aos[i];
            ParticleInterpolator::Linear interp(p, plo, dxi);
            interp.ParticleToMesh ( p, rho, 0, a_comp, 1,
                [=] AMREX_GPU_DEVICE ( const incflo_PC::ParticleType&, int)
                {
                    auto mass = ptd.m_rdata[incflo_ParticlesRealIdxSoA::mass][i];
                    return mass*inv_cell_volume;
                });
        });

    return;
}

/*! Copy pre-reaction fluid properties TO particles */
void incflo_PC::oldFluidComponentsToParticles (const MultiFab&  a_mf,
                                               const int&       a_lev
                                              )
{
    BL_PROFILE("incflo_PC::oldFluidComponentsToParticles()");

    AMREX_ASSERT(OK());
    AMREX_ASSERT(numParticlesOutOfRange(*this, 0) == 0);

    const auto& geom = Geom(a_lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const int n_fluid_spcs = a_mf.nComp();

    MeshToParticle( *this, a_mf, a_lev,
        [=] AMREX_GPU_DEVICE ( const incflo_PC::ParticleTileType::ParticleTileDataType& ptd,
                                int i, Array4<const Real> const& rho)
        {
            auto p = ptd.m_aos[i];
            ParticleInterpolator::Nearest interp(p, plo, dxi);
            interp.MeshToParticle( p, rho, 0,
                0, n_fluid_spcs,
                [=] AMREX_GPU_DEVICE (Array4<const Real>const& arr,
                                      int ii, int jj, int kk, int comp)
                {
                   return arr(ii,jj,kk,comp);
                },
                [=] AMREX_GPU_DEVICE (incflo_PC::ParticleType&, int comp, Real val)
                {
		   ptd.m_runtime_rdata[comp][i] = val;
                });
        });
    return;
}

/*! Obtain post-reaction fluid properties FROM particles */
void incflo_PC::newFluidComponentsFromParticles ( MultiFab&  a_mf,
                                                  const int& a_lev
                                                ) const
{
    BL_PROFILE("incflo_PC::newFluidComponentsFromParticles()");

    AMREX_ASSERT(OK());
    AMREX_ASSERT(numParticlesOutOfRange(*this, 0) == 0);

    const auto& geom = Geom(a_lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const int n_fluid_spcs = a_mf.nComp();
    // Only cells that have particles need to be modified
    MultiFab copy_mf(a_mf.boxArray(), a_mf.DistributionMap(),
                     a_mf.nComp(),a_mf.nGrow());

    MultiFab nptcl_mf(a_mf.boxArray(), a_mf.DistributionMap(),
                      1,a_mf.nGrow());
    // Get number of particles in each cell
    ParticleToMesh( *this, nptcl_mf, a_lev,
        [=] AMREX_GPU_DEVICE (  const incflo_PC::ParticleTileType::ConstParticleTileDataType& ptd,
                                int i, Array4<Real> const& rho)
        {
            auto p = ptd.m_aos[i];
            ParticleInterpolator::Nearest interp(p, plo, dxi);
            interp.ParticleToMesh( p, rho, 0, 0, 1,
                [=] AMREX_GPU_DEVICE ( const incflo_PC::ParticleType&, int)
                {
                    return Real(1.0);
                });
        });

    // Copy fluid properties from Particle to copy_mf
    // Here cells without particles will have a value of zero
    ParticleToMesh( *this, copy_mf, a_lev,
        [=] AMREX_GPU_DEVICE (  const incflo_PC::ParticleTileType::ConstParticleTileDataType& ptd,
                                int i, Array4<Real> const& rho)
        {
            auto p = ptd.m_aos[i];
            ParticleInterpolator::Nearest interp(p, plo, dxi);
            interp.ParticleToMesh( p, rho, 0,
                0, n_fluid_spcs,
                [=] AMREX_GPU_DEVICE ( const incflo_PC::ParticleType&, int comp)
                {
                    auto r_value = ptd.m_runtime_rdata[comp][i];
                    return r_value;
                });
        });
    // For cells with particles copy values from copy_mf to a_mf
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(copy_mf,TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
       Box const& bx = mfi.tilebox();
       Array4<Real const> const& copy_mf_arr  = copy_mf.const_array(mfi);
       Array4<Real const> const& nptcl_mf_arr = nptcl_mf.const_array(mfi);
       Array4<Real      > const& a_mf_arr     = a_mf.array(mfi);
       const int ncomp                        = copy_mf.nComp();

       ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
       {
          if (nptcl_mf_arr(i,j,k) > Real(0.)) {
             a_mf_arr(i,j,k,n) = copy_mf_arr(i,j,k,n)/nptcl_mf_arr(i,j,k);
          }
       });
    }

    a_mf.FillBoundary(geom.periodicity());
    return;
}

#ifdef USE_INCFLO_PYBIND11
void incflo_PC::pythonChemicalReactions (const int& a_lev,
                                         Real a_dt,
                                         py::module& a_data_transfer_mod
                                        )
{
    for (ParIterType pti(*this, a_lev); pti.isValid(); ++pti)
    {
        auto& ptile       = ParticlesAt(a_lev, pti);
        const int n       = ptile.numParticles();
        const int n_comps = ptile.NumRuntimeRealComps();
        auto ptd          = ptile.getParticleTileData();

        py::object info_sender = a_data_transfer_mod.attr("reaction_function");
        Gpu::DeviceVector<Real> py_vector(n*n_comps, -1.0);
        Real* py_vector_data = py_vector.data();
        // Copy TO vector
        for (int j=0; j<n_comps; j++) {
           ParallelFor(n, [=] AMREX_GPU_DEVICE (int i)
           {
                 py_vector_data[j*n+i] = ptd.m_runtime_rdata[j][i];
           });
        }
        intptr_t ptr_addrs = reinterpret_cast<intptr_t>(py_vector.data());
        py::object not_useful = info_sender(ptr_addrs,sizeof(Real),
                                            py_vector.size(), a_dt);
        // Copy FROM modifed vector
        for (int j=0; j<n_comps; j++) {
           ParallelFor(n, [=] AMREX_GPU_DEVICE (int i)
           {
                 ptd.m_runtime_rdata[j][i] = py_vector_data[j*n+i];
           });
        }
    }
}
#endif

#endif
