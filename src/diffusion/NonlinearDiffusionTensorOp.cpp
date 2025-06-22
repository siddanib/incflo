#include <incflo.H>
#include <NonlinearDiffusionTensorOp.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>

#ifdef AMREX_USE_EB
#include <AMReX_EB_Redistribution.H>
#include <memory>
#endif

using namespace amrex;

NonlinearDiffusionTensorOp::NonlinearDiffusionTensorOp (incflo* a_incflo)
    : m_incflo(a_incflo)
{
    readParameters();
    int finest_level = m_incflo->finestLevel();

    // GMRES
    m_gmres = std::make_unique<GM>();
    m_gmres->define(*this);
    m_gmres->setMaxIters(m_gmres_max_iter);
    m_gmres->setVerbose(m_gmres_verbose);

    // These are related to some field variables that are required
    m_rhs_n.resize(finest_level+1);
    m_density.resize(finest_level+1);
    m_eta.resize(finest_level+1);
    m_newton_iter_vel.resize(finest_level+1);
    m_newton_iter_func.resize(finest_level+1);
    // Setting only lev=0 here as it will not change
#ifdef AMREX_USE_EB
    if (!m_incflo->EBFactory(0).isAllRegular())
    {
        m_rhs_n[0] = std::make_unique<MultiFab>(m_incflo->boxArray(0),
                                                m_incflo->DistributionMap(0),
                                                AMREX_SPACEDIM, 0, MFInfo(),
                                                m_incflo->EBFactory(0));

        m_density[0] = std::make_unique<MultiFab>(m_incflo->boxArray(0),
                                                m_incflo->DistributionMap(0),
                                                1, m_nghost_density, MFInfo(),
                                                m_incflo->EBFactory(0));
        if (m_incflo->m_nodal_vel_eta) {
           m_nghost_eta = 0;
           m_eta[0] = std::make_unique<MultiFab>(
                                     amrex::convert(
                                        m_incflo->boxArray(0),
                                        IndexType::TheNodeType().ixType()),
                                   m_incflo->DistributionMap(0),
                                   1, m_nghost_eta, MFInfo(),
                                   m_incflo->EBFactory(0));
        }
        else {
           m_eta[0] = std::make_unique<MultiFab>(m_incflo->boxArray(0),
                                              m_incflo->DistributionMap(0),
                                              1, m_nghost_eta, MFInfo(),
                                              m_incflo->EBFactory(0));
        }

        m_newton_iter_vel[0] = std::make_unique<MultiFab>(m_incflo->boxArray(0),
                                                m_incflo->DistributionMap(0),
                                                AMREX_SPACEDIM, m_nghost_vel,
                                                MFInfo(), m_incflo->EBFactory(0));

        m_newton_iter_func[0] = std::make_unique<MultiFab>(m_incflo->boxArray(0),
                                                m_incflo->DistributionMap(0),
                                                AMREX_SPACEDIM, 0,
                                                MFInfo(), m_incflo->EBFactory(0));
    }
    else
#endif
    {
        m_rhs_n[0] = std::make_unique<MultiFab>(m_incflo->boxArray(0),
                                                m_incflo->DistributionMap(0),
                                                AMREX_SPACEDIM, 0, MFInfo());

        m_density[0] = std::make_unique<MultiFab>(m_incflo->boxArray(0),
                                                m_incflo->DistributionMap(0),
                                                1, m_nghost_density);

        if (m_incflo->m_nodal_vel_eta) {
           m_nghost_eta = 0;
           m_eta[0] = std::make_unique<MultiFab>(
                                     amrex::convert(
                                        m_incflo->boxArray(0),
                                        IndexType::TheNodeType().ixType()),
                                   m_incflo->DistributionMap(0),
                                   1, m_nghost_eta);
        }
        else {
           m_eta[0] = std::make_unique<MultiFab>(m_incflo->boxArray(0),
                                              m_incflo->DistributionMap(0),
                                              1, m_nghost_eta);
        }

        m_newton_iter_vel[0] = std::make_unique<MultiFab>(m_incflo->boxArray(0),
                                                m_incflo->DistributionMap(0),
                                                AMREX_SPACEDIM, m_nghost_vel);

        m_newton_iter_func[0] = std::make_unique<MultiFab>(m_incflo->boxArray(0),
                                                m_incflo->DistributionMap(0),
                                                AMREX_SPACEDIM, 0);
    }
    // The below code is related to linear part of divtau
    LPInfo info_apply;
    info_apply.setMaxCoarseningLevel(0);
#ifdef AMREX_USE_EB
    if (!m_incflo->EBFactory(0).isAllRegular())
    {
        Vector<EBFArrayBoxFactory const*> ebfact;
        for (int lev = 0; lev <= finest_level; ++lev) {
            ebfact.push_back(&(m_incflo->EBFactory(lev)));
        }

        if (m_incflo->need_divtau() || m_incflo->useTensorCorrection())
        {
            m_eb_apply_op = std::make_unique<MLEBTensorOp>(m_incflo->Geom(0,finest_level),
                                                 m_incflo->boxArray(0,finest_level),
                                                 m_incflo->DistributionMap(0,finest_level),
                                                 info_apply, ebfact);
            m_eb_apply_op->setMaxOrder(m_mg_maxorder);
            m_eb_apply_op->setDomainBC(m_incflo->get_diffuse_tensor_bc(Orientation::low),
                                       m_incflo->get_diffuse_tensor_bc(Orientation::high));
        }
    }
    else
#endif
    {
        if (m_incflo->need_divtau() || m_incflo->useTensorCorrection())
        {
            m_reg_apply_op = std::make_unique<MLTensorOp>(m_incflo->Geom(0,finest_level),
                                                m_incflo->boxArray(0,finest_level),
                                                m_incflo->DistributionMap(0,finest_level),
                                                info_apply);
            m_reg_apply_op->setMaxOrder(m_mg_maxorder);
            m_reg_apply_op->setDomainBC(m_incflo->get_diffuse_tensor_bc(Orientation::low),
                                        m_incflo->get_diffuse_tensor_bc(Orientation::high));
        }
    }
}

void NonlinearDiffusionTensorOp::readParameters ()
{
    ParmParse pp("nonlinear_tensor_diffusion");

    pp.query("verbose", m_verbose);
    // This is for linear part of divtau
    pp.query("mg_maxorder", m_mg_maxorder);
}

void NonlinearDiffusionTensorOp::diffuse_velocity (
                       Vector<MultiFab*> const& velocity,
                       Vector<MultiFab*> const& density,
                       Vector<MultiFab const*> const& eta,
                       Real dt)
{
    // This function sets the internal member variables
    // It also initializes iteration 0 velocity to the
    // provided velocity
    update_member_multifabs(GetVecOfConstPtrs(density),
                            GetVecOfConstPtrs(velocity),
                            GetVecOfConstPtrs(eta), dt);

    int nlevels = velocity.size();
    // Create a Vector<MultiFab> for RHS of Newton Method
    // This is different from RHS of Viscous solve equation
    Vector<MultiFab> rhs_newton(nlevels);
    // Create a Vector<MultiFab> for vel_incrment from Newton Method
    Vector<MultiFab> vel_incrmt_newton(nlevels);
    for (int ilev=0; ilev < nlevels; ++ilev) {
        rhs_newton[ilev].define(velocity[ilev]->boxArray(),
                                velocity[ilev]->DistributionMap(),
                                AMREX_SPACEDIM,0,MFInfo(),
                                velocity[ilev]->Factory());

        vel_incrmt_newton[ilev].define(velocity[ilev]->boxArray(),
                                velocity[ilev]->DistributionMap(),
                                AMREX_SPACEDIM,0,MFInfo(),
                                velocity[ilev]->Factory());

        vel_incrmt_newton[ilev].setVal(Real(0.));
    }
    // Look into WarpX NewtonSolver for stopping criterion
    bool newton_success = false;
    for (int inewt=0; inewt < m_newton_max_iter; ++inewt) {
        for (int ilev=0; ilev < nlevels; ++ilev) {
            MultiFab::Copy(rhs_newton[ilev], *m_newton_iter_func[ilev],
                           0,0,AMREX_SPACEDIM,0);
            // Need to negate
            rhs_newton[ilev].mult(Real(-1.0),0);
        }
        m_gmres->solve(vel_incrmt_newton,rhs_newton,
                       m_gmres_rtol,m_gmres_atol);

        update_newton_iteration_multifabs(
                      GetVecOfConstPtrs(vel_incrmt_newton));
    }

    // Copy final newton iteration velocity
    for (int ilev=0; ilev < nlevels; ++ilev) {
        MultiFab::Copy(*velocity[ilev],*m_newton_iter_vel[ilev],
                       0, 0, AMREX_SPACEDIM, m_nghost_vel);
    }
}

void NonlinearDiffusionTensorOp::compute_divtau (
                         Vector<MultiFab*> const& divtau,
                         Vector<MultiFab const*> const& velocity,
                         Vector<MultiFab const*> const& density,
                         Vector<MultiFab const*> const& eta)
{
    compute_linear_part_of_divtau(divtau, velocity, density, eta);
    // NEED TO INCLUDE HIGH-ORDER divtau TERMS HERE BEFORE THE LOOP

    // This is to be consistent with incflo code
    bool advect_momentum = m_incflo->AdvectMomentum();
    int finest_level = velocity.size()-1;
    if (!advect_momentum) {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (int lev = 0; lev <= finest_level; ++lev) {
            for (MFIter mfi(*divtau[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Box const& bx = mfi.tilebox();
                Array4<Real> const& divtau_arr = divtau[lev]->array(mfi);
                Array4<Real const> const& rho_arr = density[lev]->const_array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real rhoinv = Real(1.0)/rho_arr(i,j,k);
                    AMREX_D_TERM(divtau_arr(i,j,k,0) *= rhoinv;,
                                 divtau_arr(i,j,k,1) *= rhoinv;,
                                 divtau_arr(i,j,k,2) *= rhoinv;);
                });
            } // mfi
        } // lev
    } // not m_advect_momentum
}


void NonlinearDiffusionTensorOp::compute_viscous_solve_equation (
                       Vector<MultiFab*> const& nonlin_func,
                       Vector<MultiFab const*> const& velocity)
{
    int nlevels = nonlin_func.size();
    int numcomp = nonlin_func[0]->nComp();
    compute_linear_part_of_divtau(nonlin_func,velocity,
                                  GetVecOfConstPtrs(m_density),
                                  GetVecOfConstPtrs(m_eta));
    // NEED TO INCLUDE HIGH-ORDER divtau TERMS HERE BEFORE THE LOOP
    for (int ilev=0; ilev < nlevels; ++ilev) {
        // First multiply divtau with (-dt)
        nonlin_func[ilev]->mult(Real(-1.0)*m_dt,0);
        MultiFab::Subtract(*nonlin_func[ilev],*m_rhs_n[ilev],
                           0,0,numcomp,0);
        for (int idim=0; idim < numcomp; ++idim) {
            MultiFab::AddProduct(*nonlin_func[ilev], *m_density[ilev],
                                 0, *velocity[ilev], idim, idim, 1, 0);
        }
    }
}

// This function computes the linear part of divtau using MLTensorOp
void NonlinearDiffusionTensorOp::compute_linear_part_of_divtau (Vector<MultiFab*> const& a_divtau,
                                        Vector<MultiFab const*> const& a_velocity,
                                        Vector<MultiFab const*> const& a_density,
                                        Vector<MultiFab const*> const& a_eta)
{
    BL_PROFILE("NonlinearDiffusionTensorOp::compute_linear_part_of_divtau");

    int finest_level = m_incflo->finestLevel();

    Vector<MultiFab> velocity(finest_level+1);
    for (int lev = 0; lev <= finest_level; ++lev) {
        velocity[lev].define(a_velocity[lev]->boxArray(),
                             a_velocity[lev]->DistributionMap(),
                             AMREX_SPACEDIM, 1, MFInfo(),
                             a_velocity[lev]->Factory());
        MultiFab::Copy(velocity[lev], *a_velocity[lev], 0, 0, AMREX_SPACEDIM, 1);
    }

#ifdef AMREX_USE_EB
    if (m_eb_apply_op)
    {
        Vector<MultiFab> divtau_tmp(finest_level+1);
        int tmp_comp = (m_incflo->m_redistribution_type == "StateRedist") ? 3 : 2;
        for (int lev = 0; lev <= finest_level; ++lev) {
            divtau_tmp[lev].define(a_divtau[lev]->boxArray(),
                                   a_divtau[lev]->DistributionMap(),
                                   AMREX_SPACEDIM, tmp_comp, MFInfo(),
                                   a_divtau[lev]->Factory());
            divtau_tmp[lev].setVal(0.0);
        }

        // We want to return div (mu grad)) phi
        m_eb_apply_op->setScalars(0.0, -1.0);

        // For when we use the stencil for centroid values
        // m_eb_apply_op->setPhiOnCentroid();

        for (int lev = 0; lev <= finest_level; ++lev) {
            m_eb_apply_op->setACoeffs(lev, *a_density[lev]);

            Array<MultiFab,AMREX_SPACEDIM> b = m_incflo->average_velocity_eta_to_faces(lev, *a_eta[lev]);

            m_eb_apply_op->setShearViscosity(lev, GetArrOfConstPtrs(b), MLMG::Location::FaceCentroid);

            if (m_incflo->hasEBFlow()) {
               m_eb_apply_op->setEBShearViscosityWithInflow(lev, *a_eta[lev], *(m_incflo->get_velocity_eb()[lev]));
            } else {
               m_eb_apply_op->setEBShearViscosity(lev, *a_eta[lev]);
            }
            m_eb_apply_op->setLevelBC(lev, &velocity[lev]);
        }

        MLMG mlmg(*m_eb_apply_op);
        mlmg.apply(GetVecOfPtrs(divtau_tmp), GetVecOfPtrs(velocity));

        for(int lev = 0; lev <= finest_level; lev++)
        {
        amrex::single_level_redistribute( divtau_tmp[lev], *a_divtau[lev], 0, AMREX_SPACEDIM, m_incflo->Geom(lev));
            // auto const& bc = m_incflo->get_velocity_bcrec_device_ptr();
            // m_incflo->redistribute_term(*a_divtau[lev], divtau_tmp[lev], *a_velocity[lev],
        //                 bc, lev);
        }
    }
    else
#endif
    {
        // We want to return div (mu grad)) phi
        m_reg_apply_op->setScalars(0.0, -1.0);
        for (int lev = 0; lev <= finest_level; ++lev) {
            m_reg_apply_op->setACoeffs(lev, *a_density[lev]);
            Array<MultiFab,AMREX_SPACEDIM> b;
            if (a_eta[lev]->boxArray().ixType().nodeCentered()) {
                b = incflo::average_nodal_velocity_eta_to_faces(lev, *a_eta[lev]);
            }
            else {
                b = m_incflo->average_velocity_eta_to_faces(lev, *a_eta[lev]);
            }
            m_reg_apply_op->setShearViscosity(lev, GetArrOfConstPtrs(b));
            m_reg_apply_op->setLevelBC(lev, &velocity[lev]);
        }

        MLMG mlmg(*m_reg_apply_op);
        mlmg.apply(a_divtau, GetVecOfPtrs(velocity));
    }
}

void NonlinearDiffusionTensorOp::update_member_multifabs (
         Vector<MultiFab const*> const& a_density,
         Vector<MultiFab const*> const& a_vel,
         Vector<MultiFab const*> const& a_eta,
         Real a_dt)
{
    m_dt = a_dt;
    int nlevels = a_density.size();
    // To take into account if finer levels change
    if (nlevels > 1) {
        m_rhs_n.resize(nlevels);
        m_density.resize(nlevels);
        m_newton_iter_vel.resize(nlevels);
        m_newton_iter_func.resize(nlevels);
        for (int ilev=1; ilev < nlevels; ++ilev) {
            m_rhs_n[ilev] = std::make_unique<MultiFab>(a_vel[ilev]->boxArray(),
                                            a_vel[ilev]->DistributionMap(),
                                            AMREX_SPACEDIM, 0, MFInfo(),
                                            a_vel[ilev]->Factory());

            m_density[ilev] = std::make_unique<MultiFab>(
                                            a_density[ilev]->boxArray(),
                                            a_density[ilev]->DistributionMap(),
                                            1, m_nghost_density, MFInfo(),
                                            a_density[ilev]->Factory());

            m_eta[ilev] = std::make_unique<MultiFab>(
                                            a_eta[ilev]->boxArray(),
                                            a_eta[ilev]->DistributionMap(),
                                            1, m_nghost_eta, MFInfo(),
                                            a_eta[ilev]->Factory());

            m_newton_iter_vel[ilev] = std::make_unique<MultiFab>(
                                            a_vel[ilev]->boxArray(),
                                            a_vel[ilev]->DistributionMap(),
                                            AMREX_SPACEDIM, m_nghost_vel,
                                            MFInfo(),a_vel[ilev]->Factory());

            m_newton_iter_func[ilev] = std::make_unique<MultiFab>(
                                            a_vel[ilev]->boxArray(),
                                            a_vel[ilev]->DistributionMap(),
                                            AMREX_SPACEDIM, 0, MFInfo(),
                                            a_vel[ilev]->Factory());
        }
    }
    // Update the member variables
    for (int ilev = 0; ilev < nlevels; ++ilev) {
        MultiFab::Copy(*m_density[ilev],*a_density[ilev],
                       0,0,1,m_nghost_density);

        MultiFab::Copy(*m_eta[ilev],*a_eta[ilev],
                       0,0,1,m_nghost_eta);

        MultiFab::Copy(*m_newton_iter_vel[ilev],*a_vel[ilev],
                       0,0,AMREX_SPACEDIM,m_nghost_vel);

        MultiFab::Copy(*m_rhs_n[ilev],*a_vel[ilev],
                       0,0,AMREX_SPACEDIM,0);

        for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
            MultiFab::Multiply(*m_rhs_n[ilev], *a_density[ilev],
                               0,idim,1,0);
        }
    }
    // Update m_newton_iter_func
    compute_viscous_solve_equation(GetVecOfPtrs(m_newton_iter_func),
                                   GetVecOfConstPtrs(m_newton_iter_vel));
}

void NonlinearDiffusionTensorOp::update_newton_iteration_multifabs (
                Vector<MultiFab const*> const& a_vel_increment)
{
    int nlevels = a_vel_increment.size();
    for (int ilev=0; ilev < nlevels; ++ilev) {
        // Increment without ghost cells
        MultiFab::Add(*m_newton_iter_vel[ilev], *a_vel_increment[ilev],
                      0,0,AMREX_SPACEDIM,0);
        // Use FillBoundary to update ghost cells
        m_newton_iter_vel[ilev]->FillBoundary(
                          m_incflo->Geom(ilev).periodicity());
    }
    // Update m_newton_iter_func
    compute_viscous_solve_equation(GetVecOfPtrs(m_newton_iter_func),
                                   GetVecOfConstPtrs(m_newton_iter_vel));
}

// Putting everything needed by GMRES below
// All these need to be public member functions
// Jv corresponds to matrix vector product of Jacobian and increment
// NOTE: norm2 calculations might have change when it becomes multi-level
void NonlinearDiffusionTensorOp::apply (VMF& Jv, VMF& v)
{
    int numcomp = v[0].nComp();
    int nlevels = v.size();
    Real eps_newton, old_vel_norm2, vel_incrmt_norm2;
    old_vel_norm2 = Real(0.);
    vel_incrmt_norm2 = Real(0.);
    Real a_tmp;
    for (int ilev=0; ilev < nlevels; ++ilev) {
        a_tmp = m_newton_iter_vel[ilev]->norm2(0,numcomp);
        old_vel_norm2 += a_tmp*a_tmp;
        a_tmp = v[ilev].norm2(0,numcomp);
        vel_incrmt_norm2 += a_tmp*a_tmp;
    }
    old_vel_norm2 = std::sqrt(old_vel_norm2);
    vel_incrmt_norm2 = std::sqrt(vel_incrmt_norm2);

    eps_newton = m_newton_epsilon;
    eps_newton *= (Real(1.0) + old_vel_norm2);
    eps_newton = std::sqrt(eps_newton);
    eps_newton /= (vel_incrmt_norm2 + Real(1.0e-18));

    Vector<MultiFab> vel_jacobian(nlevels);
    for (int ilev=0; ilev < nlevels; ++ilev) {
        vel_jacobian[ilev].define(v[ilev].boxArray(),
                                  v[ilev].DistributionMap(),
                                  numcomp,m_nghost_vel, MFInfo(),
                                  v[ilev].Factory());
        // Old iteration velocity has correct physical boundary
        // information so copy from its ghost cells
        MultiFab::Copy(vel_jacobian[ilev],*m_newton_iter_vel[ilev],
                       0,0,numcomp,m_nghost_vel);
        // Do NOT copy to ghost cells using v
        MultiFab::Saxpy(vel_jacobian[ilev],
                        eps_newton, v[ilev], 0, 0, numcomp,
                        0);
        // FillBoundary call for interior/periodic ghost cells
        vel_jacobian[ilev].FillBoundary(m_incflo->Geom(ilev).periodicity());
    }

    compute_viscous_solve_equation(GetVecOfPtrs(Jv),
                                   GetVecOfConstPtrs(vel_jacobian));

    for (int ilev=0; ilev < nlevels; ++ilev) {
        MultiFab::Subtract(Jv[ilev],*m_newton_iter_func[ilev],
                           0,0,numcomp,0);
        Jv[ilev].mult(Real(1.0)/(eps_newton+Real(1.0e-18)),0);
    }
}

void NonlinearDiffusionTensorOp::assign (VMF& lhs,
                                         VMF const& rhs)
{
    int numcomp = rhs[0].nComp();
    int nlevels = rhs.size();
    for (int ilev=0; ilev < nlevels; ++ilev) {
        MultiFab::Copy(lhs[ilev],rhs[ilev],0,0,numcomp,0);
    }
}

Real NonlinearDiffusionTensorOp::dotProduct (VMF const& v1,
                                             VMF const& v2)
{
    Real dot_all_lev = Real(0.);
    int numcomp = v1[0].nComp();
    int nlevels = v1.size();
    for (int ilev=0; ilev < nlevels; ++ilev)
    {
        dot_all_lev += MultiFab::Dot(v1[ilev],0,v2[ilev],0,numcomp,0);
    }
    return dot_all_lev;
}

void NonlinearDiffusionTensorOp::increment (VMF& lhs,
                                            VMF const& rhs, Real a)
{
    int numcomp = rhs[0].nComp();
    int nlevels = rhs.size();
    for (int ilev=0; ilev < nlevels; ++ilev)
    {
        MultiFab::Saxpy(lhs[ilev],a,rhs[ilev],0,0,numcomp,0);
    }
}

void NonlinearDiffusionTensorOp::linComb (VMF& lhs,
                                          Real a, VMF const& rhs_a,
                                          Real b, VMF const& rhs_b)
{
    int numcomp = rhs_a[0].nComp();
    int nlevels = rhs_a.size();
    for (int ilev=0; ilev < nlevels; ++ilev)
    {
        MultiFab::LinComb(lhs[ilev],a,rhs_a[ilev],0,
                          b,rhs_b[ilev],0,0,numcomp,0);
    }
}

Vector<MultiFab> NonlinearDiffusionTensorOp::makeVecRHS ()
{
    int nlevels = m_incflo->finestLevel()+1;
    Vector<MultiFab> rhs;
    rhs.resize(nlevels);
    /*
#ifdef AMREX_USE_EB
    if (!m_incflo->EBFactory(0).isAllRegular())
    {
        for (int ilev = 0; ilev < nlevels; ++ilev) {
            rhs[ilev].define(m_incflo->boxArray(ilev),
                             m_incflo->DistributionMap(ilev),
                             AMREX_SPACEDIM,0,MFInfo(),
                             m_incflo->EBFactory(ilev));
        }
    } else
#endif
    {
        for (int ilev = 0; ilev < nlevels; ++ilev) {
            rhs[ilev].define(m_incflo->boxArray(ilev),
                             m_incflo->DistributionMap(ilev),
                             AMREX_SPACEDIM,0);
        }

    }
    */
    for (int ilev = 0; ilev < nlevels; ++ilev) {
        rhs[ilev].define(m_newton_iter_vel[ilev]->boxArray(),
                         m_newton_iter_vel[ilev]->DistributionMap(),
                         AMREX_SPACEDIM,0, MFInfo(),
                         m_newton_iter_vel[ilev]->Factory());
    }
    return rhs;
}

Vector<MultiFab> NonlinearDiffusionTensorOp::makeVecLHS ()
{
    int nlevels = m_incflo->finestLevel()+1;
    Vector<MultiFab> lhs;
    lhs.resize(nlevels);
    /*
#ifdef AMREX_USE_EB
    if (!m_incflo->EBFactory(0).isAllRegular())
    {
        for (int ilev = 0; ilev < nlevels; ++ilev) {
            lhs[ilev].define(m_incflo->boxArray(ilev),
                             m_incflo->DistributionMap(ilev),
                             AMREX_SPACEDIM,m_nghost_vel,MFInfo(),
                             m_incflo->EBFactory(ilev));
        }
    } else
#endif
    {
        for (int ilev = 0; ilev < nlevels; ++ilev) {
            lhs[ilev].define(m_incflo->boxArray(ilev),
                             m_incflo->DistributionMap(ilev),
                             AMREX_SPACEDIM,m_nghost_vel);
        }
    }
    */
    for (int ilev = 0; ilev < nlevels; ++ilev) {
        lhs[ilev].define(m_newton_iter_vel[ilev]->boxArray(),
                         m_newton_iter_vel[ilev]->DistributionMap(),
                         AMREX_SPACEDIM,m_nghost_vel, MFInfo(),
                         m_newton_iter_vel[ilev]->Factory());
    }
    return lhs;
}

Real NonlinearDiffusionTensorOp::norm2 (VMF const& v)
{
    Real norm2_all_lev = Real(0.);
    Real norm2_lev;
    int nlevels = v.size();
    int numcomp = v[0].nComp();
    for (int ilev=0; ilev < nlevels; ++ilev)
    {
        norm2_lev = v[ilev].norm2(0,numcomp);
        norm2_all_lev += norm2_lev*norm2_lev;
    }
    return std::sqrt(norm2_all_lev);
}

void NonlinearDiffusionTensorOp::precond (VMF& lhs, VMF const& rhs)
{
    // Currently not leveraging any custom preconditioner
    int nlevels = rhs.size();
    int numcomp = rhs[0].nComp();
    for (int ilev=0; ilev < nlevels; ++ilev)
    {
        MultiFab::Copy(lhs[ilev],rhs[ilev],0,0,numcomp,0);
    }
}


void NonlinearDiffusionTensorOp::scale (VMF& v, Real fac)
{
    int nlevels = v.size();
    for (int ilev=0; ilev < nlevels; ++ilev)
    {
        v[ilev].mult(fac);
    }
}

void NonlinearDiffusionTensorOp::setToZero (VMF& v)
{
    int nlevels = v.size();
    for (int ilev=0; ilev < nlevels; ++ilev)
    {
        v[ilev].setVal(Real(0.));
    }
}
