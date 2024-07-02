function pp_entangle(pKe::Vector{Float64}, IP1s::Vector{Float64}, Ipdouble::Real,
        E_X::Array{Float64,2}, t_X::Array{Float64,1}, config::Dict{String, Any}; 
        cohsum::Bool=true, avgEp::Bool=false)
"""
    Calculate the complex amplitudes from multiple continua
    Then Project the abs(sum(amplitudes)) .^2 along Pz
    """
    @assert size(E_X,2)==1 "N_T0=1 should be enough, other T0 can be acquired by shifting \theta"
    N_pz::Int = checkPzslice(config);
    Npi::Int = isa(config["Np"], Int) ? config["Np"]^2 : length(config["Np"][1]);
    
    intensity::Array{Float64} = fill(0,1);
    bp_outs::Array{ComplexF64} = fill(0,1);
    if avgEp
        intensity = fill(0, N_pz, Npi);
    else
        intensity = fill(0, N_pz, Npi,length(pKe));
    end
    if cohsum
        bp_outs = fill(0, N_pz, Npi);
    end
    
    dos::Vector{Float64} = sqrt.(pKe);
    dos ./= sum(dos);
    @showprogress 1 "Scan Ep..." for jpKe in 1:length(pKe)
    # for jpKe in 1:length(pKe)
        bp_outs .= 0;
        d = dos[jpKe];
        for iIp in 1:length(IP1s)
            Ip = IP1s[iIp]
            config["photoelectronKE"] = pKe[jpKe];
            config["Ip"] = Ip;
            config["AMelectronKE"] = Ip - Ipdouble;
            bp = complex_Amp_Pspace(E_X, t_X, config);
            bp = view(bp, :,1,:);
            # println(size(bp_outs), size(view(bp, :,1,:)))
            if cohsum
                bp_outs .+= bp;
            elseif avgEp
                @. intensity += abs(bp) ^2 * d;
            else
                @inbounds @. intensity[:,:,jpKe] += abs(bp) ^2;
            end
        end
        if cohsum
           if avgEp # Accumulate without tabulating over Ep
               @. intensity += abs(bp_outs) ^2 * d;
           else
               @inbounds @. intensity[:,:,jpKe] = abs(bp_outs) ^2;
           end
        end
    end
    # Pz=0 has half the normal contribution to the pz projection
    if (N_pz > 1) & (config["Pz_slice"][1] == 0)
        selectdim(intensity,1,1) .*= 0.5; 
    end
    dpz::Float64 = (N_pz > 1) ? config["dpz"] : 1;
    intensity = sum(intensity, dims=1) .* dpz;
    dropdims(intensity, dims=1)
end

using FLoops
# Multithreading with FLoops. It does not work with P_xm, P_ym being iterators
function complex_Amp_Pspace(E_X::Array{Float64,2}, t_X::Array{Float64,1}, 
    config::Dict{String, Any})::Array{ComplexF64,3}

    # Apply Configurations and convert units
    pKe::Float64 = config["photoelectronKE"] / E_AU;
    aKe::Float64 = config["AMelectronKE"] / E_AU;
    I_p::Float64 = config["Ip"] / E_AU; # Change eV to a.u.
    Gamma::Float64 = config["Gamma"] * T_AU;
    K_max::Float64 = config["Kmax"];
    K_min::Float64 = get(config, "Kmin", 0.0);
    dipoleM::Function = config["dipole_matrix"];
    Zaccum::Bool = get(config, "accumPz", false);
    
    t_X = t_X / T_AU; 
    t_X .-= t_X[1];
    # From now on, everything is in a.u., and t0 is shifted to 0 
    # because both E_X and A fields have been already evaluated
    
    NT0::Int = size(E_X,2); # Number of electric fields to simulate
    N_t::Int = size(t_X,1); # Number of time points
    difftX = diff(t_X);
    if Stat.std(difftX)  < 1E-3 * abs(Stat.mean(difftX)) # Check if step size is uniform
        dt = (t_X[end]-t_X[1]) / ( N_t - 1 ); 
        # println("Uniform t grid")
    else # Variable step size
        dt = t_X;
        # println("Non-uniform t grid")
    end
    difftX = nothing;

    # # Truncated E_X(t) needs a corresponding taxis
    # N_tE::Int = size(E_X,1);
    # t_XE = @view t_X[1:N_tE];
    
    # Setup momentum grid
    Pz_slice = get(config, "Pz_slice", nothing);
    P_xm, P_ym, P_z = get_Pgrid(config["Np"], Pz_slice, config["dpz"]);
    N_pz = length(P_z); # number of z-points
    N_pi = length(P_xm);
    if !(haskey(config, "Pz_slice"))
        config["Pz_slice"] = P_z;
    end

    b_p::Array{ComplexF64} = fill(0.0, N_pz, NT0, N_pi);
    
    #Phase from photoionization
    DPhase_PI::Array{ComplexF64,1} = exp.( (1im * (pKe + I_p) + Gamma/2 ).* t_X );
    DPhase_PI *= dipoleM(sqrt(2*pKe),0;Ip_au=I_p, Py=0,Pz=0); # size=(N_t,)
    PI_ints::Array{ComplexF64,2} = E_X .* DPhase_PI;
    cum_Integrate!(PI_ints, dt, dim=1); 
    # Inner integral finished
    
    # b_pvec = ThreadsX.map(P_xm, P_ym) do Px,Py
    #     b_pi::Array{ComplexF64,2} = fill(0.0, N_pz, NT0);
    P_xm = vec(collect(P_xm)); P_ym = vec(collect(P_ym));
    let dt=dt
        @floop ThreadedEx() for (ind_p, Px,Py) in zip(1:N_pi, P_xm, P_ym)
            K_xy = (Px^2 + Py^2)/2;
            for ind_calc in 1:NT0
                PI_int = @inbounds @view PI_ints[:,ind_calc];
                for (ind_z, Pz) in enumerate(P_z)
                    K = K_xy + (1/2) * Pz^2;
                    if (K_min > K)
                        continue
                    end
                    zphase_rate = 1im * K - (Gamma/2); 
                    PI_int = @inbounds @view PI_ints[:, ind_calc];
                    # PI_int_gen = genWithTail(PI_int, N_tE, N_t);
                    intgrand = (exp(1im * vlkv) * exp(zphase_rate * t) * PId 
                                for (t,vlkv,PId) in zip(t_X,I_xy,PI_int));
                    # b_pi[ind_z,ind_calc] = sum(intgrand)*dt;
                    @inbounds b_p[ind_z,ind_calc, ind_p] = sum(intgrand)*dt;
                end
            end
            # b_pi
        end
    end
    
    # for ind_p in 1:N_pi
    #     mat = only(@inbounds @view b_pvec[ind_p]); #@inbounds 
    #     # println(typeof(mat))
    #     for ind_calc in 1:NT0
    #         b_p[:, ind_calc,ind_p] .= @inbounds @view mat[:,ind_calc];
    #     end
    # end
    b_p
    # b_pvec
end

function genWithTail(vec::AbstractArray{T}, Nv::Int, Nt::Int) where {T<:Number}
    # @assert Nv <= length(vec);
    ( (t<=Nv) ? (@inbounds vec[t]) : (@inbounds vec[Nv]) for t in 1:Nt)
end