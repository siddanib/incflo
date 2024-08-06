#include <incflo.H>

using namespace amrex;

void incflo::ReadRheologyParameters()
{
     amrex::ParmParse pp("incflo");

     std::string fluid_model_s = "newtonian";
     pp.query("fluid_model", fluid_model_s);
     pp.query("min_eta", m_eta_min);
     pp.query("max_eta", m_eta_max);

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
            pp_scnd.get("diameter", m_diam_second);
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
        else
        {
            amrex::Abort("Unknown fluid_model! Choose either newtonian, powerlaw, bingham, hb, smd");
        }
        if (fluid_model_s_snd != "newtonian") {
            amrex::Print() << "Clamps for the fluid eta are = [ " << m_eta_min_second
                <<" , " << m_eta_max_second << " ]" << std::endl;
        }
     }
}
