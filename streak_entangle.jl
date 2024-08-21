import Interpolations
import HDF5

function streak_entangle(pKe::Vector{Float64}, IP1s::Vector{Float64}, Ipdouble::Real,
        E_X::Array{Float64,2}, A_L::Array{Float64,2}, t_X::Array{Float64,1}, config::Dict{String, Any};
        cohsum::Bool=true, avgEp::Bool=false, grho=(x->1))
"""
    Calculate the complex amplitudes from multiple continua
    Project the abs(sum(amplitudes)) .^2 along Pz
    """
    # @assert size(E_X,2)==1 "N_T0=1 should be enough, other T0 can be acquired by shifting \theta"
    N_pz::Int = checkPzslice(config);
    NT0::Int = size(E_X,2); # Number of electric fields to simulate
    Npi::Int = isa(config["Np"], Int) ? config["Np"]^2 : length(config["Np"][1]);
    amphasors::Array{ComplexF64} = checkAMphasors(config);
    if NT0==1 && length(amphasors) > 1
        # use the Pulse dimension to store AM-phase scan
        NT0 = length(amphasors);
    else
        amphasors = amphasors[1:1];
    end

    intensity_size::Vector{Int32} = [NT0, Npi,];
    bp_size::Vector{Int32} = [1,];
    accum2Dint = accum2Dint_kpavg;
    if ~avgEp
        push!(intensity_size, length(pKe));
        accum2Dint = accum2Dint_nokpavg;
    end
    if cohsum
        bp_size = [N_pz, NT0, Npi,];
        amphasors = reshape(amphasors, 1, :, 1);
    end

    intensity::Array{Float64} = fill(0, intensity_size...);
    intensity_inc::Array{Float64} = fill(0, NT0, Npi);
    bp_outs::Array{ComplexF64} = fill(0, bp_size...);
    density::Array{Float64} = fill(0, N_pz, NT0, Npi);

    dos::Vector{Float64} = sqrt.(pKe);
    dos ./= sum(dos);

    gmsk::Array{Float64} = make_recagmap(config, grho);
    @showprogress 1 "Scan Ep..." for jpKe in 1:length(pKe)
    # for jpKe in 1:length(pKe)
        bp_outs .= 0;
        d = dos[jpKe];
        for (iIp, Ip) in enumerate(IP1s)
            config["photoelectronKE"] = pKe[jpKe];
            config["Ip"] = Ip;
            config["AMelectronKE"] = Ip - Ipdouble;
            bp = complex_Amp_Pspace(E_X, A_L, t_X, config);

            # @assert size(bp) == [N_pz, NT0, Npi,]
            if cohsum  # accumulate the complex amp over continuua
                @. bp_outs += bp * (amphasors ^ (iIp-1));
            else
                @. density = (abs(bp) ^ 2) * gmsk;
                intensity_inc .= pz_integrate(density, config["Pz_slice"]);
                accum2Dint(intensity_inc, intensity, d, jpKe);
            end
        end
        if cohsum
            @. density = (abs(bp_outs) ^ 2) * gmsk;
            intensity_inc .= pz_integrate(density, config["Pz_slice"]);
            accum2Dint(intensity_inc, intensity, d, jpKe);
        end
    end
    if NT0==1
        intensity = dropdims(intensity; dims=1);
    end
    intensity
end

function streak_entangle_gridIP(pKe::Vector{Float64}, IP1a::Vector{Float64},
            IP1b::Vector{Float64}, E_X::Vector{Float64}, A_L::Array{Float64,2},
            t_X::Vector{Float64}, config::Dict{String, Any},
        h5gh::HDF5.Group, tag::String;
        cohsum::Bool=true, grho=(x->1))
    """
    for each photoelectron eKE
        Calculate the complex amplitudes from two continua, with multiple IP1
        Cache them
        for each pair of (IP1a, IP1b)
            Project the abs(sum(amplitudes)) .^2 along Pz
        accumulate, w/ DoS accounted
    """
    N_pz::Int = checkPzslice(config);
    Npi::Int = isa(config["Np"], Int) ? config["Np"]^2 : length(config["Np"][1]);
    NIPa::Int = length(IP1a);
    NIPb::Int = length(IP1b);
    amphasors::Array{ComplexF64} = checkAMphasors(config);
    NT0::Int = length(amphasors);

    intensity_size::Tuple = (NT0, Npi, NIPb, NIPa);
    density_size::Vector{Int32} = [N_pz, NT0, Npi];
    # bp_size::Tuple = (N_pz, Npi, NIPa+NIPb);
    # IP_all::Vector{Float64} = vcat(IP1a..., IP1b...);
    amphasors = reshape(amphasors, 1, :, 1);

    intensity = HDF5.create_dataset(h5gh, tag, Float64,
                                    intensity_size; chunk=(NT0, Npi,1,1))
    # println("$(tag) created in the H5 file")
    tmp_int::Array{Float64} = fill(0, 1);
    density::Array{Float64} = fill(0, density_size...);
    bpa_all::Array{ComplexF64} = fill(0, N_pz, Npi, NIPa...);
    bpb_all::Array{ComplexF64} = fill(0, N_pz, Npi, NIPb...);

    dos::Vector{Float64} = sqrt.(pKe);
    gmsk::Array{Float64} = make_recagmap(config, grho);
    # @showprogress 1 "Scan Ep..." for (jpKe, pKej) in enumerate(pKe)
    E_Xm::Array{Float64,2} = reshape(E_X, :, 1);
    for (jpKe, pKej) in enumerate(pKe)
        bpa_all .= 0;
        bpb_all .= 0;
        dosj = @view dos[jpKe];
        config["photoelectronKE"] = pKej;
        complex_Amp_Pspace(E_X, A_L, t_X, IP1a, config, bpa_all);
        complex_Amp_Pspace(E_X, A_L, t_X, IP1b, config, bpb_all);
        for ia in 1:NIPa
            for ib in 1:NIPb
                bpa = @inbounds view(bpa_all, :,:,ia);
                bpa = reshape(bpa, N_pz, 1, Npi);
                bpb = @inbounds view(bpb_all, :,:,ib);
                bpb = reshape(bpb, N_pz, 1, Npi);
                if cohsum
                    @. density = abs(bpa + amphasors * bpb)^2;
                else
                    @. density = abs(bpa)^2 + abs(bpb)^2;
                end
                @. density *= gmsk;
                # println(sum(density),", ", dosj)
                tmp_int = @inbounds intensity[:,:,ib,ia];
                tmp_int .+= pz_integrate(density, config["Pz_slice"]) .* dosj;
                @inbounds intensity[:,:,ib,ia] = tmp_int;
            end
        end
    end
    intensity
end


using FLoops
# Multithreading with FLoops. It does not work with P_xm, P_ym being iterators
function complex_Amp_Pspace(E_X::Array{Float64,2}, A::Array{Float64,2},
         t_X::Array{Float64,1}, config::Dict{String, Any})::Array{ComplexF64,3}

    # Apply Configurations and convert units
    pKe::Float64 = config["photoelectronKE"] / E_AU;
    aKe::Float64 = config["AMelectronKE"] / E_AU;
    I_p::Float64 = config["Ip"] / E_AU; # Change eV to a.u.
    Gamma::Float64 = config["Gamma"] * T_AU;
    K_max::Float64 = config["Kmax"];
    K_min::Float64 = get(config, "Kmin", 0.0);
    Kabsmax::Float64 = get(config, "Kabsmax", K_max*5);
    dipoleM::Function = config["dipole_matrix"];
    Zaccum::Bool = get(config, "accumPz", false);

    t_X = t_X / T_AU;
    t_X .-= t_X[1];
    # From now on, everything is in a.u., and t0 is shifted to 0
    # because both E_X and A fields have been already evaluated

    NT0::Int = size(E_X,2); # Number of electric fields to simulate
    N_t::Int = size(t_X,1); # Number of time points
    (dt, tw_iter) = get_dtaxis(t_X);

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
    # cum_Integrate!(PI_ints, dt, dim=1);
    # Note that dt isa Union{Real, Vector{Float64}}
    PI_ints .*= dt;
    cumsum!(PI_ints, PI_ints; dims=1)
    # Inner integral finished
    PI_ints .*= dt; # For the outer integral, once for all

    P_xm = vec(collect(P_xm)); P_ym = vec(collect(P_ym));
    # let dt=dt
    @floop ThreadedEx(basesize=3) for (ind_p, Px,Py) in zip(1:N_pi, P_xm, P_ym)
        @init begin
            I_xy = Vector{Float64}(undef,N_t)
            V_x = Vector{Float64}(undef,N_t);
            V_y = Vector{Float64}(undef,N_t);
            vstates = Vector{ComplexF64}(undef,N_t);
            intgrand = Vector{ComplexF64}(undef,N_t);
        end
        V_x .= Px .- view(A, :, 1);
        V_y .= Py .- view(A, :, 2);
        @. I_xy = (1/2) * ( V_x^2 + V_y^2 ) - aKe;
        # cum_Integrate!(I_xy, dt, dim=1); # Volkov phase
        I_xy .*= dt;
        cumsum!(I_xy, I_xy; dims=1)
        @. vstates = exp(-1im * I_xy);
        for ind_calc in 1:NT0
            PI_int = @inbounds @view PI_ints[:,ind_calc];
            for (ind_z, Pz) in enumerate(P_z)
                K_z = (1/2) * Pz^2;
                K = K_z + (1/2) * (Px^2 + Py^2)
                if (K_min > K) | (Kabsmax < K)
                    continue
                end
                zphase_rate = 1im * K_z - (Gamma/2);
                # PI_int = @inbounds @view PI_ints[:, ind_calc];
                # PI_int_gen = genWithTail(PI_int, N_tE, N_t);
                # intgrand = (exp(1im * vlkv) * exp(zphase_rate * t) * PId * tw
                #             for (t,vlkv,PId,tw) in zip(t_X,I_xy,PI_int,tw_iter));
                @. intgrand = exp(zphase_rate * t_X) * PI_int;
                # b_pi[ind_z,ind_calc] = sum(intgrand)*dt;
                @inbounds b_p[ind_z,ind_calc, ind_p] = dot(vstates, intgrand)
            end
        end
    end
    # end
    b_p
end

function complex_Amp_Pspace(E_X::Array{Float64,1}, A::Array{Float64,2},
         t_X::Array{Float64,1}, IP1_eV::Vector{Float64}, config::Dict{String, Any}, bp_dest::Union{AbstractArray{ComplexF64,3},Nothing})::Array{ComplexF64,3}
    """Vectorized over IP1"""
    # Apply Configurations and convert units
    pKe::Float64 = config["photoelectronKE"] / E_AU;
    I_p_vec::Vector{Float64} = IP1_eV ./ E_AU; # Change eV to a.u.
    aKe_vec::Vector{Float64} = (IP1_eV .- config["Idouble"]) ./ E_AU ;
    Gamma::Float64 = config["Gamma"] * T_AU;
    K_max::Float64 = config["Kmax"];
    K_min::Float64 = get(config, "Kmin", 0.0);
    Kabsmax::Float64 = get(config, "Kabsmax", K_max*10);
    dipoleM::Function = config["dipole_matrix"];
    Zaccum::Bool = get(config, "accumPz", false);

    t_X = t_X / T_AU;
    t_X .-= t_X[1];
    # From now on, everything is in a.u., and t0 is shifted to 0
    # because both E_X and A fields have been already evaluated

    NIP::Int = length(I_p_vec);
    N_t::Int = size(t_X,1); # Number of time points
    (dt, tw_iter) = get_dtaxis(t_X);

    # Setup momentum grid
    Pz_slice = get(config, "Pz_slice", nothing);
    P_xm, P_ym, P_z = get_Pgrid(config["Np"], Pz_slice, config["dpz"]);
    N_pz = length(P_z); # number of z-points
    N_pi = length(P_xm);
    if !(haskey(config, "Pz_slice"))
        config["Pz_slice"] = P_z;
    end

    # b_p::Array{ComplexF64} = fill(0.0, N_pz, NIP, N_pi);
    @assert size(bp_dest) == (N_pz, N_pi, NIP)

    #Phase from photoionization
    IpT::Array{Float64} = transpose(I_p_vec);
    aKemean::Float64 = Stat.mean(aKe_vec);
    daKT::Array{Float64} = transpose(aKe_vec .- aKemean);
    DPhase_PI::Array{ComplexF64,2} = @. exp( (1im*(pKe+IpT) + Gamma/2) * t_X );
    for (i, I_p) in enumerate(I_p_vec)
        @inbounds DPhase_PI[:,i] .*= dipoleM(sqrt(2*pKe),0; Ip_au=I_p, Py=0,Pz=0);
    end
    PI_ints::Array{ComplexF64,2} = E_X .* DPhase_PI;
    # cum_Integrate!(PI_ints, dt, dim=1);
    PI_ints .*= dt;
    cumsum!(PI_ints, PI_ints; dims=1)
    # Inner integral finished
    # @. PI_ints *= exp(( - Gamma/2) * t_X);
    @. PI_ints *= exp((-1im * daKT - Gamma/2) * t_X);
    PI_ints .*= dt; # For the outer integral, once for all

    # h5cach = @sprintf("%s/PIDEBUGtmp1.h5", get_outdir());
    # h5c = HDF5.h5open(h5cach, "w")
    # h5c["PI_ints"] = PI_ints;
    # h5c["I_p"] = I_p_vec
    # HDF5.close(h5c)

    P_xm = vec(collect(P_xm)); P_ym = vec(collect(P_ym));
    @floop ThreadedEx() for (ind_p, Px,Py) in zip(1:N_pi, P_xm, P_ym)
        @init begin
            # I_xy_IP = Array{Float64,2}(undef,N_t,NIP)
            I_xy = Vector{Float64}(undef,N_t);
            V_x = Vector{Float64}(undef,N_t);
            V_y = Vector{Float64}(undef,N_t);
            vstates = Vector{ComplexF64}(undef,N_t);
            intgrand = Vector{ComplexF64}(undef,N_t);
        end
        V_x .= Px .- view(A, :, 1);
        V_y .= Py .- view(A, :, 2);
        @. I_xy = (1/2) * ( V_x^2 + V_y^2 ) - aKemean;
        # cum_Integrate!(I_xy, dt, dim=1); # Volkov phase, xy part
        I_xy .*= dt;
        cumsum!(I_xy, I_xy; dims=1);
        @. vstates = exp(-1im * I_xy);
        ## Here it's -1im for the convenience of taking the dot product later on
        ##
        for ind_calc in 1:NIP
            PI_int = @inbounds @view PI_ints[:,ind_calc];
            for (ind_z, Pz) in enumerate(P_z)
                K_z = (1/2) * Pz^2;
                K = K_z + (1/2) * (Px^2 + Py^2)
                if (K_min > K) | (Kabsmax < K)
                    continue
                end
                # PI_int = @inbounds @view PI_ints[:, ind_calc];
                # intgrand = (exp(1im * vlkv) * exp(1im * K_z * t) * PId * tw
                #             for (t,vlkv,PId,tw) in zip(t_X,I_xy,PI_int, tw_iter));
                @. intgrand = exp(1im * K_z * t_X) * PI_int;
                # b_pi[ind_z,ind_calc] = sum(intgrand)*dt;
                @inbounds bp_dest[ind_z, ind_p, ind_calc] = dot(vstates, intgrand);
            end
        end
    end
    # end
    bp_dest
end

function genWithTail(vec::AbstractArray{T}, Nv::Int, Nt::Int) where {T<:Number}
    # @assert Nv <= length(vec);
    ( (t<=Nv) ? (@inbounds vec[t]) : (@inbounds vec[Nv]) for t in 1:Nt)
end


function pz_integrate(density::Array{Float64}, pz_axis::Vector{T}) where {T<:Real}
    if length(pz_axis)<2
        return dropdims(density; dims=1)
    end
    if (pz_axis[1] >= (pz_axis[2]-pz_axis[1])/2)
        selectdim(density,1,1) .*= 2; # compensate the 1/2 factor for the first point in trapz
    end
    return itrapz(density, pz_axis, dim=1)
end

function make_recagmap(config::Dict{String,Any}, grho::Union{Function,Interpolations.Extrapolation})::Array{Float64}
    if all(grho.(0:0.02:1) .== 1)
        return fill(1.0, 1)
    end
    P_xm, P_ym, P_z = get_Pgrid(config["Np"], config["Pz_slice"], config["dpz"])
    Prisq = @. P_xm ^ 2 + P_ym ^ 2;
    Prisq = transpose(vec(collect(Prisq)));
    Pi = @. sqrt((P_z ^ 2) + Prisq);
    gmsk::Array{Float64} = @. grho(sqrt(Prisq)/Pi);
    return reshape(gmsk, size(gmsk,1), 1, :)
end

function accum2Dint_kpavg(intensity_inc::Array{T}, intensity::Array{Float64},
                          d::Real, j::Int) where {T<:Real}
    @. intensity += intensity_inc * d;
end

function accum2Dint_nokpavg(intensity_inc::Array{T}, intensity::Array{Float64},
                          d::Real, j::Int) where {T<:Real}
    @inbounds @. intensity[:,:,j] += intensity_inc;
end
