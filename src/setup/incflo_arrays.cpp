#include <incflo.H>

using namespace amrex;

incflo::LevelData::LevelData (amrex::BoxArray const& ba,
                              amrex::DistributionMapping const& dm,
                              amrex::FabFactory<FArrayBox> const& fact,
                              incflo* my_incflo)
    : velocity    (ba, dm, AMREX_SPACEDIM, my_incflo->nghost_state(), MFInfo(), fact),
      velocity_o  (ba, dm, AMREX_SPACEDIM, my_incflo->nghost_state(), MFInfo(), fact),

      density     (ba, dm, 1             , my_incflo->nghost_state(), MFInfo(), fact),
      density_o   (ba, dm, 1             , my_incflo->nghost_state(), MFInfo(), fact),
      density_nph (ba, dm, 1             , my_incflo->nghost_state(), MFInfo(), fact),

      tracer    (ba, dm, my_incflo->m_ntrac, my_incflo->nghost_state(), MFInfo(), fact),
      tracer_o  (ba, dm, my_incflo->m_ntrac, my_incflo->nghost_state(), MFInfo(), fact),

      mac_phi   (ba, dm, 1             , 1       , MFInfo(), fact),
      gp        (ba, dm, AMREX_SPACEDIM, 0 , MFInfo(), fact),

      conv_velocity_o (ba, dm, AMREX_SPACEDIM    , 0, MFInfo(), fact),
      conv_density_o  (ba, dm, 1                 , 0, MFInfo(), fact),
      conv_tracer_o   (ba, dm, my_incflo->m_ntrac, 0, MFInfo(), fact)
{
    if (my_incflo->m_use_cc_proj) {
        p_cc.define(ba                                  , dm, 1, 3, MFInfo(), fact);
    } else {
        p_nd.define(convert(ba,IntVect::TheNodeVector()), dm, 1, 0, MFInfo(), fact);
    }
    if (my_incflo->m_use_temperature) {
        temperature.define   (ba, dm, 1, my_incflo->nghost_state(), MFInfo(), fact);
        temperature_o.define (ba, dm, 1, my_incflo->nghost_state(), MFInfo(), fact);

        conv_temperature_o.define(ba, dm, 1, 0, MFInfo(), fact);
    }
#ifdef AMREX_USE_EB
    if (my_incflo->hasEBFlow()) {
        velocity_eb.define(ba, dm, AMREX_SPACEDIM, my_incflo->nghost_state(), MFInfo(), fact);
        density_eb.define (ba, dm, 1             , my_incflo->nghost_state(), MFInfo(), fact);
    }
    // Allow for Dirichlet BC on EB even if there's no flow through the EB
    if (my_incflo->m_advect_tracer && !my_incflo->m_eb_flow.tracer.empty()) {
        tracer_eb.define  (ba, dm, my_incflo->m_ntrac, my_incflo->nghost_state(), MFInfo(), fact);
    }
    if (my_incflo->m_use_temperature && !my_incflo->m_eb_flow.temperature.empty()) {
        temperature_eb.define(ba, dm, 1, my_incflo->nghost_state(), MFInfo(), fact);
    }
#endif
    if (my_incflo->m_advection_type != "MOL") {
        divtau_o.define(ba, dm, AMREX_SPACEDIM, 0, MFInfo(), fact);
        if (my_incflo->m_advect_tracer) {
            laps_o.define(ba, dm, my_incflo->m_ntrac, 0, MFInfo(), fact);
        }
        if (my_incflo->m_use_temperature) {
            laps_tem_o.define(ba, dm, 1, 0, MFInfo(), fact);
        }
    } else {
        conv_velocity.define(ba, dm, AMREX_SPACEDIM   , 0, MFInfo(), fact);
        conv_density.define (ba, dm, 1                , 0, MFInfo(), fact);
        conv_tracer.define (ba, dm, my_incflo->m_ntrac, 0, MFInfo(), fact);

        if (my_incflo->m_use_temperature) {
            conv_temperature.define(ba, dm, 1, 0, MFInfo(), fact);
        }

        bool need_velocity_divtau = my_incflo->need_velocity_divtau();
        bool need_scalar_laplacian = my_incflo->need_scalar_laplacian();
        if (need_velocity_divtau || my_incflo->use_tensor_correction)
        {
            divtau.define  (ba, dm, AMREX_SPACEDIM, 0, MFInfo(), fact);
            divtau_o.define(ba, dm, AMREX_SPACEDIM, 0, MFInfo(), fact);
        }
        if (need_scalar_laplacian)
        {
            if ( my_incflo->m_advect_tracer) {
                laps.define  (ba, dm, my_incflo->m_ntrac, 0, MFInfo(), fact);
                laps_o.define(ba, dm, my_incflo->m_ntrac, 0, MFInfo(), fact);
            }
            if (my_incflo->m_use_temperature) {
                laps_tem.define  (ba, dm, 1, 0, MFInfo(), fact);
                laps_tem_o.define(ba, dm, 1, 0, MFInfo(), fact);
            }
        }
    }

    if (my_incflo->m_gran_rheo_modified_time_stepping) {
        time_stepping_alpha = std::make_unique<MultiFab>(ba, dm, 1, 1, MFInfo(), fact);
        time_stepping_alpha->setVal(Real(0.));
    }
}

// Resize all arrays when instance of incflo class is constructed.
// This is only done at the very start of the simulation.
void incflo::ResizeArrays ()
{
    // Time holders for fillpatch stuff
    m_t_new.resize(max_level + 1);
    m_t_old.resize(max_level + 1);

    m_leveldata.resize(max_level+1);

    m_factory.resize(max_level+1);
#ifdef USE_AMREX_MPMD
    m_mpmd_copiers.resize(max_level+1);
#endif
}
