#ifndef INCFLO_NON_LIN_DIFF_TENSOR_OP_H_
#define INCFLO_NON_LIN_DIFF_TENSOR_OP_H_

#include <AMReX_GMRES.H>
#include <AMReX_MLMG.H>
#ifdef AMREX_USE_EB
#include <AMReX_MLEBTensorOp.H>
#if (AMREX_SPACEDIM == 2)
#include <AMReX_EB_LeastSquares_2D_K.H>
#else
#include <AMReX_EB_LeastSquares_3D_K.H>
#endif
#endif
#include <AMReX_MLTensorOp.H>

class incflo;

class NonlinearDiffusionTensorOp
{
public:
    using RT      = amrex::Real;
    using VMF     = amrex::Vector<      amrex::MultiFab>;
    using VMFPtr  = amrex::Vector<      amrex::MultiFab*>;
    using VCMFPtr = amrex::Vector<const amrex::MultiFab*>;
    using GM      = amrex::GMRES<VMF,NonlinearDiffusionTensorOp>;

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

    void assign (VMFPtr const& lhs, VCMFPtr const& rhs);

    amrex::Real dotProduct (VMF const& v1, VMF const& v2);

    amrex::Real dotProduct (VCMFPtr const& v1, VCMFPtr const& v2);

    void increment (VMF& lhs, VMF const& rhs, amrex::Real a);

    void increment (VMFPtr const& lhs, VCMFPtr const& rhs, amrex::Real a);

    void linComb (VMF& lhs, amrex::Real a, VMF const& rhs_a,
                  amrex::Real b, VMF const& rhs_b);

    void linComb (VMFPtr const& lhs, amrex::Real a, VCMFPtr const& rhs_a,
                  amrex::Real b, VCMFPtr const& rhs_b);

    VMF makeVecRHS ();

    VMF makeVecLHS ();

    amrex::Real norm2 (VMF const& v);

    amrex::Real norm2 (VCMFPtr const& v);

    void precond (VMF& lhs, VMF const& rhs);

    void scale (VMF& v, amrex::Real fac);

    void scale (VMFPtr const& v, amrex::Real fac);

    void setToZero (VMF& v);

    void AddProduct (VMFPtr const& dst, VCMFPtr const& src1, int comp1,
                     VCMFPtr const& src2, int comp2, int dstcomp, int numcomp);

    // This function computes gradients on regular faces as well as EB faces
    void compVelGrad (int amrlev,
            amrex::MultiFab & sol, // temporary velocity
            amrex::MLMG::Location loc, // FaceCenter or FaceCentroid
            amrex::Array<amrex::MultiFab*, AMREX_SPACEDIM> const& gradVel // regular faces
#ifdef AMREX_USE_EB
            , amrex::MultiFab* gradVel_EB
#endif
            );


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
                       amrex::Vector<amrex::MultiFab const*> const& a_conc_second,
                       amrex::Vector<amrex::MultiFab const*> const& a_p_static,
                       amrex::Vector<amrex::MultiFab const*> const& a_old_velocity);

    // This function evaluates the norm2 of the non-linear function
    amrex::Real get_norm_of_residual ();

    // This function is similar to DiffusionTensorOp::diffuse_velocity
    // Used as preconditioner for GMRES
    void diffuse_velocity_mlmg (amrex::Vector<amrex::MultiFab*> const& velocity,
                                amrex::Vector<amrex::MultiFab*> const& density,
                                amrex::Vector<amrex::MultiFab const*> const& eta,
                                amrex::Vector<amrex::MultiFab const*> const& rhs,
                                amrex::Real dt);

    // This function essentially performs the Newton iteration loop
    void diffuse_velocity_alpha_factor (amrex::Vector<amrex::MultiFab*> const& velocity,
                                        amrex::Vector<amrex::MultiFab*> const& density,
                                        amrex::Vector<amrex::MultiFab const*> const& eta,
                                        amrex::Real dt);

    incflo* m_incflo;

    std::unique_ptr<GM> m_gmres;
    // This is the RHS term that depends on actual time previous state
    // member variables because they need to be called in "apply" function
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_rhs_n,
                             m_density, m_eta,
                             m_conc_second,
                             m_p_static,
                             m_newton_iter_vel,
                             // Frozen copy of the velocity passed into diffuse_velocity;
                             // not updated during Newton iterations.
                             m_old_iter_vel,
                             m_newton_iter_func;
    // This will be set at the beginning of diffuse_velocity
    amrex::Real m_dt;
    int m_nghost_density = 1;
    int m_nghost_eta     = 1;
    int m_nghost_vel     = 1;
    bool m_use_eta_from_prev_time = true;
    bool m_use_ho_coeff_from_prev_time = false;
    // This variable is used to gradually increase
    // the effect of nonlinear div(tau)
    amrex::Real m_alpha_factor = amrex::Real(1.);
    // DiffusionOp verbosity
    int m_verbose = 1;
    // Options to control GMRES behavior
    int m_gmres_verbose = 2;
    int m_gmres_max_iter = 1000;
    int m_newton_max_iter = 100;
    // These are for updating the velocity
    amrex::Real m_newton_update_alpha = amrex::Real(1.0e-3);
    int m_newton_update_max_iter = 100;
    bool m_gmres_use_precond = true;
#ifdef AMREX_USE_FLOAT
    // This is for newton iterations
    amrex::Real m_newton_rtol    = amrex::Real(1.0e-4);
    amrex::Real m_newton_atol    = amrex::Real(1.0e-7);
    amrex::Real m_newton_epsilon = amrex::Real(1.0e-7);
    amrex::Real m_gmres_rtol = amrex::Real(1.0e-4);
    amrex::Real m_gmres_atol = amrex::Real(1.0e-7);
#else
    // This is for newton iterations
    amrex::Real m_newton_rtol    = amrex::Real(1.0e-11);
    amrex::Real m_newton_atol    = amrex::Real(1.0e-14);
    amrex::Real m_newton_epsilon = amrex::Real(1.0e-18);
    amrex::Real m_gmres_rtol = amrex::Real(1.0e-11);
    amrex::Real m_gmres_atol = amrex::Real(1.0e-14);
#endif

    // These member variables are for the linear part of divtau
    // Essentially MLMG related
#ifdef AMREX_USE_EB
    std::unique_ptr<amrex::MLEBTensorOp> m_eb_solve_op;
    std::unique_ptr<amrex::MLEBTensorOp> m_eb_apply_op;
#endif
    std::unique_ptr<amrex::MLTensorOp> m_reg_solve_op;
    std::unique_ptr<amrex::MLTensorOp> m_reg_apply_op;

    int m_mg_verbose = 1;
    int m_mg_bottom_verbose = 0;
    int m_mg_max_iter = 1;
    int m_mg_bottom_maxiter = 100;
    int m_mg_max_fmg_iter = 0;
    int m_mg_max_coarsening_level = 100;
    int m_mg_maxorder = 2;
    int m_num_pre_smooth = 2;
    int m_num_post_smooth = 2;
#ifdef AMREX_USE_FLOAT
    amrex::Real m_mg_rtol = amrex::Real(1.0e-4);
    amrex::Real m_mg_atol = amrex::Real(1.0e-7);
#else
    amrex::Real m_mg_rtol = amrex::Real(1.0e-11);
    amrex::Real m_mg_atol = amrex::Real(1.0e-14);
#endif
    bool m_mg_use_gauss_seidel = true;
    std::string m_bottom_solver = "bicgstab";
};

#endif
