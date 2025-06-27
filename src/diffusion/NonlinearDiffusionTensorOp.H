#ifndef INCFLO_NON_LIN_DIFF_TENSOR_OP_H_
#define INCFLO_NON_LIN_DIFF_TENSOR_OP_H_

#include <AMReX_GMRES.H>
#include <AMReX_MLMG.H>
#ifdef AMREX_USE_EB
#include <AMReX_MLEBTensorOp.H>
#endif
#include <AMReX_MLTensorOp.H>

class incflo;

class NonlinearDiffusionTensorOp
{
public:
    using RT     = amrex::Real;
    using VMF    = amrex::Vector<amrex::MultiFab>;
    using GM     = amrex::GMRES<VMF,NonlinearDiffusionTensorOp>;

    NonlinearDiffusionTensorOp (incflo* a_incflo);

    // diffuse_velocity will involve a Newton iteration loop
    void diffuse_velocity (amrex::Vector<amrex::MultiFab*> const& velocity,
                           amrex::Vector<amrex::MultiFab*> const& density,
                           amrex::Vector<amrex::MultiFab const*> const& eta,
                           amrex::Real dt);

    void compute_divtau (amrex::Vector<amrex::MultiFab*> const& divtau,
                         amrex::Vector<amrex::MultiFab const*> const& velocity,
                         amrex::Vector<amrex::MultiFab const*> const& density,
                         amrex::Vector<amrex::MultiFab const*> const& eta);

    // All functions required by GMRES
    // Note that most of these functions might have to be modified
    // for multi-level solve
    void apply (VMF& Ax, VMF& x);
    void assign (VMF& lhs, VMF const& rhs);
    amrex::Real dotProduct (VMF const& v1, VMF const& v2);
    void increment (VMF& lhs, VMF const& rhs, amrex::Real a);
    void linComb (VMF& lhs, amrex::Real a, VMF const& rhs_a,
                  amrex::Real b, VMF const& rhs_b);
    VMF makeVecRHS ();
    VMF makeVecLHS ();
    amrex::Real norm2 (VMF const& v);
    void precond (VMF& lhs, VMF const& rhs);
    void scale (VMF& v, amrex::Real fac);
    void setToZero (VMF& v);

private:

    void readParameters ();
    // Needs to be called inside of diffuse_velocity only once
    void update_member_multifabs (
                       amrex::Vector<amrex::MultiFab const*> const& a_density,
                       amrex::Vector<amrex::MultiFab const*> const& a_vel,
                       amrex::Vector<amrex::MultiFab const*> const& a_eta,
                       amrex::Real a_dt);
    // Needs to be called inside of Newton iteration loop
    void update_newton_iteration_multifabs (
                       amrex::Vector<amrex::MultiFab const*> const& a_vel_increment);
    // This will be rho*u - dt div (tau) - rhs = 0 that is solved implicitly
    void compute_viscous_solve_equation (
                       amrex::Vector<amrex::MultiFab      *> const& nonlin_func,
                       amrex::Vector<amrex::MultiFab const*> const& velocity);
    // This is the linear part of divtau
    void compute_linear_part_of_divtau (
                       amrex::Vector<amrex::MultiFab      *> const& divtau,
                       amrex::Vector<amrex::MultiFab const*> const& velocity,
                       amrex::Vector<amrex::MultiFab const*> const& density,
                       amrex::Vector<amrex::MultiFab const*> const& eta);

    void add_non_linear_part_of_divtau (
                       amrex::Vector<amrex::MultiFab      *> const& a_divtau,
                       amrex::Vector<amrex::MultiFab const*> const& a_velocity,
                       amrex::Vector<amrex::MultiFab const*> const& a_density,
                       amrex::Vector<amrex::MultiFab const*> const& a_eta);

    // This function evaluates the norm2 of the non-linear function
    amrex::Real get_norm_of_residual ();

    incflo* m_incflo;

    std::unique_ptr<GM> m_gmres;
    // This is the RHS term that depends on actual time previous state
    // member variables because they need to be called in "apply" function
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_rhs_n,
                             m_density, m_eta,
                             m_newton_iter_vel,
                             m_newton_iter_func;
    // This will be set at the beginning of diffuse_velocity
    amrex::Real m_dt;
    int m_nghost_density = 1;
    int m_nghost_eta     = 1;
    int m_nghost_vel     = 2;
    // DiffusionOp verbosity
    int m_verbose = 1;
    // Options to control GMRES behavior
    int m_gmres_verbose = 2;
    int m_gmres_max_iter = 1000;
    int m_newton_max_iter = 100;
    // These are for updating the velocity
    amrex::Real m_newton_update_alpha = amrex::Real(1.0e-3);
    int m_newton_update_max_iter = 100;
#ifdef AMREX_USE_FLOAT
    // This is for newton iterations
    amrex::Real m_newton_rtol    = amrex::Real(1.0e-4);
    amrex::Real m_newton_atol    = amrex::Real(1.0e-7);
    amrex::Real m_newton_epsilon = amrex::Real(1.0e-7);
#else
    // This is for newton iterations
    amrex::Real m_newton_rtol    = amrex::Real(1.0e-11);
    amrex::Real m_newton_atol    = amrex::Real(1.0e-14);
    amrex::Real m_newton_epsilon = amrex::Real(1.0e-16);
#endif

    // These member variables are for the linear part of divtau
    // Essentially MLMG related
#ifdef AMREX_USE_EB
    std::unique_ptr<amrex::MLEBTensorOp> m_eb_apply_op;
#endif
    std::unique_ptr<amrex::MLTensorOp> m_reg_apply_op;
    int m_mg_verbose = 1;
    int m_mg_maxorder = 2;
#ifdef AMREX_USE_FLOAT
    amrex::Real m_gmres_rtol = amrex::Real(1.0e-4);
    amrex::Real m_gmres_atol = amrex::Real(1.0e-7);
#else
    amrex::Real m_gmres_rtol = amrex::Real(1.0e-11);
    amrex::Real m_gmres_atol = amrex::Real(1.0e-14);
#endif
};

#endif
