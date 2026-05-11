#include <incflo.H>

using namespace amrex;

void incflo::ReadRheologyParameters()
{
     amrex::ParmParse pp("incflo");

     std::string fluid_model_s = "newtonian";
     pp.query("fluid_model", fluid_model_s);
     pp.query("min_eta", m_eta_min);
     pp.query("max_eta", m_eta_max);
#ifdef AMREX_USE_EB
     pp.query("eb_smooth_cutcell_viscosity", m_eb_smooth_cutcell_viscosity);
     pp.query("eb_smooth_cutcell_viscosity_blend", m_eb_smooth_cutcell_viscosity_blend);
     m_eb_smooth_cutcell_viscosity_blend =
         amrex::Clamp(m_eb_smooth_cutcell_viscosity_blend, amrex::Real(0.0), amrex::Real(1.0));
     pp.query("eb_ho_vfrac_threshold", m_eb_ho_vfrac_threshold);
     m_eb_ho_vfrac_threshold =
         amrex::Clamp(m_eb_ho_vfrac_threshold, amrex::Real(0.0), amrex::Real(1.0));
#endif

     if(fluid_model_s == "newtonian")
     {
         m_fluid_model = FluidModel::Newtonian;
         amrex::Print() << "Newtonian fluid with"
                        << " mu = " << m_mu << std::endl;
     }
     else if(fluid_model_s == "powerlaw")
     {
         m_fluid_model = FluidModel::powerlaw;
         pp.query("n", m_n_0);
         AMREX_ALWAYS_ASSERT(m_n_0 > 0.0);
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_0 != 1.0,
                 "No point in using power-law rheology with n = 1");

         amrex::Print() << "Power-law fluid with"
                        << " mu = " << m_mu
                        << ", n = " << m_n_0 <<  std::endl;
     }
     else if(fluid_model_s == "bingham")
     {
         m_fluid_model = FluidModel::Bingham;
         pp.query("tau_0", m_tau_0);
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tau_0 > 0.0,
                 "No point in using Bingham rheology with tau_0 = 0");

         pp.query("papa_reg", m_papa_reg);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_papa_reg > 0.0,
                    "Papanastasiou regularisation parameter must be positive");

         amrex::Print() << "Bingham fluid with"
                        << " mu = " << m_mu
                        << ", tau_0 = " << m_tau_0
                        << ", papa_reg = " << m_papa_reg << std::endl;
     }
     else if(fluid_model_s == "hb")
     {
         m_fluid_model = FluidModel::HerschelBulkley;
         pp.query("n", m_n_0);
         AMREX_ALWAYS_ASSERT(m_n_0 > 0.0);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_0 != 1.0,
                 "No point in using Herschel-Bulkley rheology with n = 1");

         pp.query("tau_0", m_tau_0);
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tau_0 > 0.0,
                 "No point in using Herschel-Bulkley rheology with tau_0 = 0");

         pp.query("papa_reg", m_papa_reg);
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_papa_reg > 0.0,
                 "Papanastasiou regularisation parameter must be positive");

         amrex::Print() << "Herschel-Bulkley fluid with"
                        << " mu = " << m_mu
                        << ", n = " << m_n_0
                        << ", tau_0 = " << m_tau_0
                        << ", papa_reg = " << m_papa_reg << std::endl;
     }
     else if(fluid_model_s == "smd")
     {
         m_fluid_model = FluidModel::deSouzaMendesDutra;
         pp.query("n", m_n_0);
         AMREX_ALWAYS_ASSERT(m_n_0 > 0.0);

         pp.query("tau_0", m_tau_0);
         AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tau_0 > 0.0,
                 "No point in using de Souza Mendes-Dutra rheology with tau_0 = 0");

         pp.query("eta_0", m_eta_0);
         AMREX_ALWAYS_ASSERT(m_eta_0 > 0.0);

         amrex::Print() << "de Souza Mendes-Dutra fluid with"
                        << " mu = " << m_mu
                        << ", n = " << m_n_0
                        << ", tau_0 = " << m_tau_0
                        << ", eta_0 = " << m_eta_0 << std::endl;
     }
#ifdef USE_AMREX_MPMD
     else if(fluid_model_s == "mpmd")
     {
         m_fluid_model = FluidModel::DataDrivenMPMD;
         amrex::Print() << "Data-driven model through AMReX-MPMD."<<std::endl;
     }
#endif
     else
     {
         amrex::Abort("Unknown fluid_model! Choose either newtonian, powerlaw, bingham, hb, smd");
     }

     if (fluid_model_s != "newtonian") {
         amrex::Print() << "Clamps for the fluid eta are = [ " << m_eta_min
             <<" , " << m_eta_max << " ]" << std::endl;
     }
#ifdef AMREX_USE_EB
     if (m_eb_smooth_cutcell_viscosity) {
         amrex::Print() << "EB cut-cell viscosity smoothing enabled with blend = "
                        << m_eb_smooth_cutcell_viscosity_blend << std::endl;
     }
     if (m_eb_ho_vfrac_threshold > amrex::Real(0.0)) {
         amrex::Print() << "EB high-order rheology vfrac threshold = "
                        << m_eb_ho_vfrac_threshold << std::endl;
     }
#endif

     if (m_two_fluid) {
        amrex::ParmParse pp_scnd("incflo.second_fluid");
        pp_scnd.query("ro_0", m_ro_0_second);
        AMREX_ALWAYS_ASSERT(m_ro_0_second >= 0.0);
        // Initially setting ro_grain the same as ro_0
        m_ro_grain_second = m_ro_0_second;
        pp_scnd.query("ro_grain",m_ro_grain_second);
        pp_scnd.query("mu", m_mu_second);
        std::string fluid_model_s_snd = "newtonian";
        pp_scnd.query("fluid_model", fluid_model_s_snd);
        pp_scnd.get("min_conc", m_min_conc_second);

        if (fluid_model_s_snd != "newtonian") {
            pp_scnd.get("min_eta", m_eta_min_second);
            pp_scnd.get("max_eta", m_eta_max_second);
            pp_scnd.query("diameter", m_diam_second);
        }

        amrex::Print() << "Second fluid properties : " << std::endl;
        if(fluid_model_s_snd == "newtonian")
        {
            m_fluid_model_second = FluidModel::Newtonian;
            amrex::Print() << "Newtonian fluid with"
                           << " mu = " << m_mu_second << std::endl;
        }
        else if(fluid_model_s_snd == "powerlaw")
        {
            m_fluid_model_second = FluidModel::powerlaw;
            pp_scnd.query("n", m_n_0_second);
            AMREX_ALWAYS_ASSERT(m_n_0_second > 0.0);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_0_second != 1.0,
                    "No point in using power-law rheology with n = 1");

            amrex::Print() << "Power-law fluid with"
                           << " mu = " << m_mu_second
                           << ", n = " << m_n_0_second <<  std::endl;
        }
        else if(fluid_model_s_snd == "bingham")
        {
            m_fluid_model_second = FluidModel::Bingham;
            pp_scnd.query("tau_0", m_tau_0_second);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tau_0_second > 0.0,
                    "No point in using Bingham rheology with tau_0 = 0");

            pp_scnd.query("papa_reg", m_papa_reg_second);
               AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_papa_reg_second > 0.0,
                       "Papanastasiou regularisation parameter must be positive");

            amrex::Print() << "Bingham fluid with"
                           << " mu = " << m_mu_second
                           << ", tau_0 = " << m_tau_0_second
                           << ", papa_reg = " << m_papa_reg_second << std::endl;
        }
        else if(fluid_model_s_snd == "hb")
        {
            m_fluid_model_second = FluidModel::HerschelBulkley;
            pp_scnd.query("n", m_n_0_second);
            AMREX_ALWAYS_ASSERT(m_n_0_second > 0.0);
               AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_0_second != 1.0,
                    "No point in using Herschel-Bulkley rheology with n = 1");

            pp_scnd.query("tau_0", m_tau_0_second);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tau_0_second > 0.0,
                    "No point in using Herschel-Bulkley rheology with tau_0 = 0");

            pp_scnd.query("papa_reg", m_papa_reg_second);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_papa_reg_second > 0.0,
                    "Papanastasiou regularisation parameter must be positive");

            amrex::Print() << "Herschel-Bulkley fluid with"
                           << " mu = " << m_mu_second
                           << ", n = " << m_n_0_second
                           << ", tau_0 = " << m_tau_0_second
                           << ", papa_reg = " << m_papa_reg_second << std::endl;
        }
        else if(fluid_model_s_snd == "smd")
        {
            m_fluid_model_second = FluidModel::deSouzaMendesDutra;
            pp_scnd.query("n", m_n_0_second);
            AMREX_ALWAYS_ASSERT(m_n_0_second > 0.0);

            pp_scnd.query("tau_0", m_tau_0_second);
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tau_0_second > 0.0,
                    "No point in using de Souza Mendes-Dutra rheology with tau_0 = 0");

            pp_scnd.query("eta_0", m_eta_0_second);
            AMREX_ALWAYS_ASSERT(m_eta_0_second > 0.0);

            amrex::Print() << "de Souza Mendes-Dutra fluid with"
                           << " mu = " << m_mu_second
                           << ", n = " << m_n_0_second
                           << ", tau_0 = " << m_tau_0_second
                           << ", eta_0 = " << m_eta_0_second << std::endl;
        }
#ifdef USE_AMREX_MPMD
        else if(fluid_model_s_snd == "mpmd")
        {
            m_fluid_model_second = FluidModel::DataDrivenMPMD;
            pp_scnd.query("mu_p_eps_second",m_mu_p_eps_second);
            pp_scnd.query("mu_sr_eps_second",m_mu_sr_eps_second);
            amrex::Print() << "Data-driven model through AMReX-MPMD."<<std::endl;
        }
#endif
        else if(fluid_model_s_snd == "rauter")
        {
            m_fluid_model_second = FluidModel::Rauter;
            pp_scnd.get("mu_1", m_mu_1_second);
            pp_scnd.get("mu_2", m_mu_2_second);
            pp_scnd.get("I_0", m_I_0_second);
            amrex::Print() << "Using mu(I) defined Rauter 2021 (Eq. 2.30)"<<std::endl;
        }
        else if(fluid_model_s_snd == "granularpowerlaw")
        {
            m_fluid_model_second = FluidModel::GranularPowerlaw;
            // Form of powerlaw: const + A (Inertial_Num ^ (alpha))
            // Ordering in table (const, A, alpha)
            pp_scnd.gettable("coeff_table", m_mu_powerlaw);
            // This is to get the lowe neutral inertial number
            pp_scnd.query("low_neutral_I", m_I_1_N_powerlaw);
            pp_scnd.query("mu_p_eps_second",m_mu_p_eps_second);
            pp_scnd.query("mu_sr_eps_second",m_mu_sr_eps_second);
            pp_scnd.query("min_eta_ho", m_eta_ho_min_second);
            pp_scnd.query("max_eta_ho", m_eta_ho_max_second);
            amrex::Print() << "Using mu(I) based on Granular Powerlaw"<<std::endl;
        }
        else if(fluid_model_s_snd == "granularpowerlaw_temperature")
        {
            m_fluid_model_second = FluidModel::GranularPowerlawTemperature;
            // Ordering in table:
            // row 0 = mu_1 coefficients
            // row 1 = mu_2 coefficients
            pp_scnd.gettable("coeff_table", m_mu_powerlaw_temperature);
            pp_scnd.query("mu_p_eps_second",m_mu_p_eps_second);
            pp_scnd.query("mu_sr_eps_second",m_mu_sr_eps_second);
            pp_scnd.query("min_eta_ho", m_eta_ho_min_second);
            pp_scnd.query("max_eta_ho", m_eta_ho_max_second);
            amrex::Print() << "Using temperature-aware granular powerlaw plumbing"
                           << std::endl;
        }
        else
        {
            amrex::Abort("Unknown fluid_model! Choose either newtonian, powerlaw, bingham, hb, smd");
        }
        if (fluid_model_s_snd != "newtonian") {
            amrex::Print() << "Clamps for the fluid eta are = [ " << m_eta_min_second
                <<" , " << m_eta_max_second << " ]" << std::endl;
        }
        if (fluid_model_s_snd == "granularpowerlaw" && m_mu_powerlaw.size() > 1) {
            amrex::Print() << "Clamps for the high-order fluid eta are = [ "
                           << m_eta_ho_min_second << " , "
                           << m_eta_ho_max_second << " ]" << std::endl;
        }
        if (fluid_model_s_snd == "granularpowerlaw_temperature"
            && m_mu_powerlaw_temperature.size() > 1) {
            amrex::Print() << "Clamps for the high-order fluid eta are = [ "
                           << m_eta_ho_min_second << " , "
                           << m_eta_ho_max_second << " ]" << std::endl;
        }
        // Additional checks
        if (fluid_model_s_snd == "granularpowerlaw_temperature") {
            if (!m_use_temperature) {
                amrex::Abort("granularpowerlaw_temperature requires use_temperature = true");
            }
            if (!m_use_granular_temperature) {
                amrex::Abort("granularpowerlaw_temperature requires use_granular_temperature = true");
            }
        }
        if (m_use_granular_temperature) {
            if (!(fluid_model_s_snd == "granularpowerlaw"
                  || fluid_model_s_snd == "granularpowerlaw_temperature")) {
                amrex::Abort("Granular Temperature needs granularpowerlaw or granularpowerlaw_temperature");
            }
        }
     }
}
